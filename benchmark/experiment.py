"""Bounded classifier/VAE/sigma experiment with untouched confirmation data."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Dict, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from benchmark.drift import fit_lot_drift_guard
from benchmark.policy_metrics import (
    CostModel,
    SafetyGates,
    evaluate_by_failure_mode,
    evaluate_policy,
    gate_candidate,
    select_candidate,
)
from benchmark.synthetic_data import FEATURE_NAMES, assert_split_isolation
from models.anomaly_detection.vae_model import VariationalAutoEncoder
from models.classification.classifier_model import ChipTestClassifier, CustomDataset
from models.statistical.sigma_rules import SigmaRule
from preprocessing.preprocessing import FeatureScaler


@dataclass(frozen=True)
class TrainingConfig:
    """Predeclared model and policy search space."""

    seed: int = 20260807
    batch_size: int = 512
    classifier_hidden_size: int = 24
    classifier_dropout: float = 0.15
    classifier_learning_rate: float = 0.001
    classifier_epochs: int = 35
    classifier_patience: int = 6
    vae_latent_size: int = 4
    vae_num_layers: int = 3
    vae_beta: float = 0.02
    vae_learning_rate: float = 0.001
    vae_epochs: int = 35
    vae_patience: int = 6
    classifier_thresholds: tuple[float, ...] = (0.01, 0.025, 0.05, 0.1, 0.2)
    vae_quantiles: tuple[float, ...] = (0.85, 0.9, 0.925, 0.95, 0.975, 0.99)
    sigma_tail_rates: tuple[float, ...] = (
        0.0001,
        0.00025,
        0.001,
        0.002,
        0.005,
        0.01,
    )
    cost_model: CostModel = field(default_factory=CostModel)
    safety_gates: SafetyGates = field(
        default_factory=lambda: SafetyGates(
            maximum_observed_escapees=10,
            maximum_relative_escape_rate=0.01,
            maximum_escape_rate_upper_95=0.02,
            maximum_overtest_rate=0.25,
        )
    )
    torch_threads: int = 4


@dataclass
class BenchmarkRun:
    """In-memory trained system plus its serializable evidence report."""

    classifier: ChipTestClassifier
    vae: VariationalAutoEncoder
    scaler: FeatureScaler
    sigma_rule: SigmaRule
    report: dict
    runtime_config: dict
    predictions: dict[str, dict[str, np.ndarray]]


def _set_reproducible_seed(config: TrainingConfig) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.torch_threads)
    torch.use_deterministic_algorithms(True)


def _copy_state_dict(model: torch.nn.Module) -> dict:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _classifier_loss(
    model: ChipTestClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features.astype(np.float32)).to(device))
        targets = torch.from_numpy(labels.astype(np.int64)).to(device)
        return float(criterion(logits, targets).item())


def _train_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[ChipTestClassifier, dict]:
    model = ChipTestClassifier(
        input_size=len(FEATURE_NAMES),
        hidden_size=config.classifier_hidden_size,
        dropout=config.classifier_dropout,
    ).to(device)
    class_weights = torch.tensor(
        [
            1.0,
            float((train_labels == 0).sum() / (train_labels == 1).sum()),
        ],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.classifier_learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        CustomDataset(train_features, train_labels),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )

    best_loss = float("inf")
    best_state = None
    patience = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, config.classifier_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_features)

        train_loss = total_loss / len(train_features)
        validation_loss = _classifier_loss(
            model,
            validation_features,
            validation_labels,
            criterion,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = _copy_state_dict(model)
            patience = 0
        else:
            patience += 1
        if patience >= config.classifier_patience:
            break

    if best_state is None:
        raise RuntimeError("Classifier training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, {
        "epochs_completed": len(history),
        "best_validation_loss": best_loss,
        "duration_seconds": time.perf_counter() - started,
        "history": history,
    }


def _deterministic_vae_loss(
    model: VariationalAutoEncoder,
    features: torch.Tensor,
) -> torch.Tensor:
    mu, logvar = model.encode(features)
    reconstruction = model.decode(mu)
    return model.loss_function(reconstruction, features, mu, logvar)


def _train_vae(
    train_pass_features: np.ndarray,
    validation_pass_features: np.ndarray,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[VariationalAutoEncoder, dict]:
    model = VariationalAutoEncoder(
        input_size=len(FEATURE_NAMES),
        latent_size=config.vae_latent_size,
        num_layers=config.vae_num_layers,
        beta=config.vae_beta,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.vae_learning_rate)
    generator = torch.Generator().manual_seed(config.seed + 1)
    train_tensor = torch.from_numpy(train_pass_features.astype(np.float32))
    validation_tensor = torch.from_numpy(
        validation_pass_features.astype(np.float32)
    ).to(device)
    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )

    best_loss = float("inf")
    best_state = None
    patience = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, config.vae_epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch_features,) in loader:
            batch_features = batch_features.to(device)
            reconstruction, mu, logvar = model(batch_features)
            loss = model.loss_function(
                reconstruction,
                batch_features,
                mu,
                logvar,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_features)

        model.eval()
        with torch.no_grad():
            validation_loss = float(
                _deterministic_vae_loss(model, validation_tensor).item()
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(train_pass_features),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = _copy_state_dict(model)
            patience = 0
        else:
            patience += 1
        if patience >= config.vae_patience:
            break

    if best_state is None:
        raise RuntimeError("VAE training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, {
        "epochs_completed": len(history),
        "best_validation_loss": best_loss,
        "duration_seconds": time.perf_counter() - started,
        "history": history,
    }


def _classifier_probabilities(
    model: ChipTestClassifier,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    tensor = torch.from_numpy(features.astype(np.float32)).to(device)
    return model.predict_proba(tensor)[:, 1].detach().cpu().numpy()


def _vae_errors(
    model: VariationalAutoEncoder,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    tensor = torch.from_numpy(features.astype(np.float32)).to(device)
    return model.get_reconstruction_error(tensor).detach().cpu().numpy()


def _expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    ece = 0.0
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(
            float(labels[mask].mean()) - float(probabilities[mask].mean())
        )
    return ece if total else 0.0


def _score_quality(labels: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def _classifier_quality(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    return {
        **_score_quality(labels, probabilities),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "ece_10_bin": _expected_calibration_error(labels, probabilities),
    }


def _fit_sigma_variants(
    train_pass: pd.DataFrame,
    config: TrainingConfig,
) -> Dict[str, SigmaRule]:
    variants = {}
    for tail_rate in config.sigma_tail_rates:
        target_count = max(1, int(round(len(train_pass) * tail_rate)))
        name = f"sigma_tail_{tail_rate:.5f}"
        variants[name] = SigmaRule(
            target_fail_count=target_count,
            min_sigma=2.0,
            max_sigma=8.0,
            step=0.05,
        ).fit(train_pass)
    return variants


def _candidate_definitions(
    config: TrainingConfig,
    vae_thresholds: Mapping[str, float],
    sigma_variants: Mapping[str, SigmaRule],
) -> list[dict]:
    candidates = [
        {
            "name": "all_test_baseline",
            "policy_family": "baseline",
            "eligible_for_selection": False,
            "configuration": {"always_test": True},
        }
    ]
    candidates.extend(
        {
            "name": f"classifier_t{threshold:.2f}",
            "policy_family": "classifier_ablation",
            "eligible_for_selection": False,
            "configuration": {"classifier_threshold": threshold},
        }
        for threshold in config.classifier_thresholds
    )
    candidates.extend(
        {
            "name": name,
            "policy_family": "vae_ablation",
            "eligible_for_selection": False,
            "configuration": {"vae_threshold": threshold},
        }
        for name, threshold in vae_thresholds.items()
    )
    candidates.extend(
        {
            "name": name,
            "policy_family": "sigma_ablation",
            "eligible_for_selection": False,
            "configuration": {"sigma_variant": name},
        }
        for name in sigma_variants
    )
    for classifier_threshold, (vae_name, vae_threshold), sigma_name in product(
        config.classifier_thresholds,
        vae_thresholds.items(),
        sigma_variants,
    ):
        candidates.append(
            {
                "name": (
                    f"hybrid_c{classifier_threshold:.2f}_"
                    f"{vae_name}_{sigma_name}"
                ),
                "policy_family": "hybrid_or",
                "eligible_for_selection": True,
                "configuration": {
                    "classifier_threshold": classifier_threshold,
                    "vae_threshold": vae_threshold,
                    "vae_threshold_name": vae_name,
                    "sigma_variant": sigma_name,
                },
            }
        )
    return candidates


def _candidate_flags(
    candidate: Mapping,
    classifier_probabilities: np.ndarray,
    vae_errors: np.ndarray,
    sigma_flags: Mapping[str, np.ndarray],
) -> np.ndarray:
    configuration = candidate["configuration"]
    if configuration.get("always_test"):
        return np.ones(len(classifier_probabilities), dtype=np.int64)
    flags = np.zeros(len(classifier_probabilities), dtype=np.int64)
    if "classifier_threshold" in configuration:
        flags |= (
            classifier_probabilities >= configuration["classifier_threshold"]
        ).astype(np.int64)
    if "vae_threshold" in configuration:
        flags |= (vae_errors > configuration["vae_threshold"]).astype(np.int64)
    if "sigma_variant" in configuration:
        flags |= sigma_flags[configuration["sigma_variant"]].astype(np.int64)
    return flags


def _component_disagreement(
    classifier_flags: np.ndarray,
    vae_flags: np.ndarray,
    sigma_flags: np.ndarray,
) -> dict:
    patterns = {}
    for classifier, vae, sigma in zip(
        classifier_flags,
        vae_flags,
        sigma_flags,
    ):
        key = f"{int(classifier)}{int(vae)}{int(sigma)}"
        patterns[key] = patterns.get(key, 0) + 1
    return dict(sorted(patterns.items()))


def run_benchmark(
    splits: Mapping[str, pd.DataFrame],
    config: TrainingConfig = TrainingConfig(),
) -> BenchmarkRun:
    """Train/select on development data, then open confirmation exactly once."""
    required_splits = {
        "train",
        "validation",
        "known_shift_validation",
        "ood_validation",
        "known_shift_validation_2",
        "ood_validation_2",
        "confirmation",
        "ood_confirmation",
    }
    if set(splits) != required_splits:
        raise ValueError(f"Expected splits {sorted(required_splits)}")
    assert_split_isolation(splits)
    _set_reproducible_seed(config)
    device = torch.device("cpu")

    train = splits["train"]
    scaler = FeatureScaler(method="robust").fit(train[FEATURE_NAMES])
    scaled = {
        name: scaler.transform(frame[FEATURE_NAMES])
        for name, frame in splits.items()
    }
    arrays = {
        name: frame.to_numpy(dtype=np.float32)
        for name, frame in scaled.items()
    }
    labels = {
        name: frame["label"].to_numpy(dtype=np.int64)
        for name, frame in splits.items()
    }

    classifier, classifier_training = _train_classifier(
        arrays["train"],
        labels["train"],
        arrays["validation"],
        labels["validation"],
        config,
        device,
    )
    vae, vae_training = _train_vae(
        arrays["train"][labels["train"] == 0],
        arrays["validation"][labels["validation"] == 0],
        config,
        device,
    )
    train_pass_frame = scaled["train"].loc[labels["train"] == 0].reset_index(
        drop=True
    )
    sigma_variants = _fit_sigma_variants(train_pass_frame, config)

    known_development_split_names = (
        "validation",
        "known_shift_validation",
        "known_shift_validation_2",
    )
    ood_development_split_names = (
        "ood_validation",
        "ood_validation_2",
    )
    lot_drift_guard = fit_lot_drift_guard(
        train,
        scaled["train"],
        [(splits[name], scaled[name]) for name in known_development_split_names],
        [(splits[name], scaled[name]) for name in ood_development_split_names],
    )

    train_pass_vae_errors = _vae_errors(
        vae,
        arrays["train"][labels["train"] == 0],
        device,
    )
    vae_thresholds = {
        f"vae_q{int(round(quantile * 1000)):03d}": float(
            np.quantile(train_pass_vae_errors, quantile)
        )
        for quantile in config.vae_quantiles
    }

    development_split_names = known_development_split_names
    development_features = np.concatenate(
        [arrays[name] for name in development_split_names]
    )
    development_labels = np.concatenate(
        [labels[name] for name in development_split_names]
    )
    development_scaled = pd.concat(
        [scaled[name] for name in development_split_names],
        ignore_index=True,
    )
    validation_classifier_probabilities = _classifier_probabilities(
        classifier,
        development_features,
        device,
    )
    validation_vae_errors = _vae_errors(vae, development_features, device)
    validation_sigma_flags = {
        name: rule.predict(development_scaled).to_numpy(dtype=np.int64)
        for name, rule in sigma_variants.items()
    }
    validation_drift_flags = np.concatenate(
        [
            lot_drift_guard.apply(splits[name], scaled[name])[1]
            for name in development_split_names
        ]
    )

    candidates = _candidate_definitions(config, vae_thresholds, sigma_variants)
    catalog = []
    for candidate in candidates:
        flags = _candidate_flags(
            candidate,
            validation_classifier_probabilities,
            validation_vae_errors,
            validation_sigma_flags,
        )
        flags |= validation_drift_flags
        record = copy.deepcopy(candidate)
        record["metrics"] = evaluate_policy(
            development_labels,
            flags,
            config.cost_model,
        )
        record["gate_result"] = gate_candidate(
            record["metrics"],
            config.safety_gates,
        )
        catalog.append(record)

    selection_pool = [
        record for record in catalog if record["eligible_for_selection"]
    ]
    selected, _ = select_candidate(selection_pool, config.safety_gates)
    selected_configuration = selected["configuration"]
    selected_sigma = sigma_variants[selected_configuration["sigma_variant"]]

    def score_frozen_split(
        split_name: str,
    ) -> tuple[dict, dict, dict[str, np.ndarray]]:
        classifier_probabilities = _classifier_probabilities(
            classifier,
            arrays[split_name],
            device,
        )
        vae_errors = _vae_errors(vae, arrays[split_name], device)
        sigma_flags = selected_sigma.predict(scaled[split_name]).to_numpy(
            dtype=np.int64
        )
        classifier_flags = (
            classifier_probabilities
            >= selected_configuration["classifier_threshold"]
        ).astype(np.int64)
        vae_flags = (vae_errors > selected_configuration["vae_threshold"]).astype(
            np.int64
        )
        ensemble_flags = classifier_flags | vae_flags | sigma_flags
        drift_scores, drift_flags, drift_summary = lot_drift_guard.apply(
            splits[split_name],
            scaled[split_name],
        )
        ensemble_flags |= drift_flags
        metrics = evaluate_policy(
            labels[split_name],
            ensemble_flags,
            config.cost_model,
        )
        details = {
            "classifier_quality": _classifier_quality(
                labels[split_name],
                classifier_probabilities,
            ),
            "vae_ranking_quality": _score_quality(
                labels[split_name],
                vae_errors,
            ),
            "failure_modes": evaluate_by_failure_mode(
                splits[split_name]["failure_mode"],
                labels[split_name],
                ensemble_flags,
            ),
            "component_metrics": {
                "classifier": evaluate_policy(
                    labels[split_name], classifier_flags, config.cost_model
                ),
                "vae": evaluate_policy(
                    labels[split_name], vae_flags, config.cost_model
                ),
                "sigma": evaluate_policy(
                    labels[split_name], sigma_flags, config.cost_model
                ),
                "ensemble": metrics,
            },
            "component_flag_patterns": _component_disagreement(
                classifier_flags,
                vae_flags,
                sigma_flags,
            ),
            "all_components_missed_failures": int(
                (
                    (labels[split_name] == 1)
                    & (classifier_flags == 0)
                    & (vae_flags == 0)
                    & (sigma_flags == 0)
                ).sum()
            ),
            "lot_drift_guard": {
                "blocked_lots": int(
                    sum(item["blocked"] for item in drift_summary.values())
                ),
                "total_lots": len(drift_summary),
                "lots": drift_summary,
            },
        }
        outputs = {
            "classifier_probabilities": classifier_probabilities,
            "vae_errors": vae_errors,
            "classifier_flags": classifier_flags,
            "vae_flags": vae_flags,
            "sigma_flags": sigma_flags,
            "lot_drift_scores": drift_scores,
            "lot_drift_flags": drift_flags,
            "ensemble_flags": ensemble_flags,
        }
        return metrics, details, outputs

    confirmation_metrics, confirmation_details, confirmation_outputs = (
        score_frozen_split("confirmation")
    )
    ood_metrics, ood_details, ood_outputs = score_frozen_split("ood_confirmation")
    confirmation_gate = gate_candidate(
        confirmation_metrics,
        config.safety_gates,
    )
    ood_drift = ood_details["lot_drift_guard"]
    ood_gate = {
        "passed": (
            ood_metrics["escapees"] == 0
            and ood_drift["blocked_lots"] == ood_drift["total_lots"]
        ),
        "checks": {
            "zero_escapees": ood_metrics["escapees"] == 0,
            "all_shifted_lots_blocked": (
                ood_drift["blocked_lots"] == ood_drift["total_lots"]
            ),
        },
        "policy": "all chips RUN when the lot drift guard blocks a lot",
    }

    report = {
        "schema_version": 1,
        "experiment_id": "public_synthetic_hybrid_v1",
        "selection_protocol": {
            "eligible_policy_family": "hybrid_or",
            "objective": "maximize validation time reduction subject to every safety gate",
            "tie_breakers": [
                "lower validation overtest rate",
                "candidate name",
            ],
            "confirmation_used_for_selection": False,
            "ood_confirmation_used_for_selection": False,
            "development_splits": list(development_split_names),
            "ood_drift_calibration_splits": list(ood_development_split_names),
            "safety_gates": asdict(config.safety_gates),
            "cost_model": asdict(config.cost_model),
        },
        "predecessor_experiments": [
            {
                "artifact": "evidence/experiments/first_confirmation_failure.json",
                "outcome": "rejected after 2 known-mode escapes and 76.5% OOD recall",
            },
            {
                "artifact": "evidence/experiments/second_robustness_confirmation_failure.json",
                "outcome": "known-mode gate passed; rejected after 96.5% OOD recall",
            },
        ],
        "unmet_targets": {
            "test_time_reduction_percent": 15.0,
            "observed_escapees": 0,
        },
        "training_config": asdict(config),
        "training": {
            "classifier": classifier_training,
            "vae": vae_training,
            "sigma_variants": {
                name: {
                    "target_fail_count_per_tail": rule.target_fail_count,
                    "feature_rule_count": len(rule.dict_sigma),
                    "aggregate_threshold": rule.aggregate_threshold,
                    "correlation_threshold": rule.correlation_threshold,
                }
                for name, rule in sigma_variants.items()
            },
            "vae_thresholds": vae_thresholds,
            "lot_drift_guard": lot_drift_guard.to_config(),
        },
        "validation": {
            "classifier_quality": _classifier_quality(
                development_labels,
                validation_classifier_probabilities,
            ),
            "vae_ranking_quality": _score_quality(
                development_labels,
                validation_vae_errors,
            ),
            "split_names": list(development_split_names),
            "samples": int(len(development_labels)),
            "candidate_count": len(catalog),
            "eligible_candidate_count": len(selection_pool),
            "candidates": catalog,
            "selected_candidate": selected,
        },
        "confirmation": {
            "metrics": confirmation_metrics,
            "gate_result": confirmation_gate,
            **confirmation_details,
        },
        "ood_confirmation": {
            "metrics": ood_metrics,
            "gate_result": ood_gate,
            **ood_details,
        },
        "acceptance": {
            "passed": confirmation_gate["passed"] and ood_gate["passed"],
            "requires_both_confirmation_gates": True,
        },
    }

    runtime_config = {
        "model_id": "public_synthetic_hybrid_v1",
        "feature_names": FEATURE_NAMES,
        "classifier": {
            "input_size": len(FEATURE_NAMES),
            "hidden_size": config.classifier_hidden_size,
            "dropout": config.classifier_dropout,
            "threshold": selected_configuration["classifier_threshold"],
        },
        "vae": {
            "input_size": len(FEATURE_NAMES),
            "latent_size": config.vae_latent_size,
            "num_layers": config.vae_num_layers,
            "beta": config.vae_beta,
            "threshold": selected_configuration["vae_threshold"],
        },
        "scaler": {
            "centre": [float(value) for value in scaler.scaler.center_],
            "scale": [float(value) for value in scaler.scaler.scale_],
        },
        "sigma": {
            "feature_rules": {
                name: {
                    key: float(value) for key, value in parameters.items()
                }
                for name, parameters in selected_sigma.dict_sigma.items()
            },
            "aggregate_threshold": float(selected_sigma.aggregate_threshold),
            "correlation_location": [
                float(value) for value in selected_sigma.correlation_location
            ],
            "correlation_precision": [
                [float(value) for value in row]
                for row in selected_sigma.correlation_precision
            ],
            "correlation_threshold": float(selected_sigma.correlation_threshold),
        },
        "cost_model": asdict(config.cost_model),
        "lot_drift_guard": lot_drift_guard.to_config(),
    }
    return BenchmarkRun(
        classifier=classifier,
        vae=vae,
        scaler=scaler,
        sigma_rule=selected_sigma,
        report=report,
        runtime_config=runtime_config,
        predictions={
            "confirmation": confirmation_outputs,
            "ood_confirmation": ood_outputs,
        },
    )
