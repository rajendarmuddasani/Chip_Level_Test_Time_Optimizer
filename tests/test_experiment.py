"""Reduced end-to-end smoke test for the benchmark experiment."""

from benchmark.experiment import TrainingConfig, run_benchmark
from benchmark.policy_metrics import SafetyGates
from benchmark.synthetic_data import SplitSpec, generate_benchmark


def test_reduced_experiment_trains_all_components_and_freezes_selection():
    specs = (
        SplitSpec("train", 0, 4, 80, 0.10, 0.00),
        SplitSpec("validation", 10, 2, 80, 0.10, 0.10),
        SplitSpec("known_shift_validation", 20, 2, 80, 0.10, 0.20),
        SplitSpec("ood_validation", 30, 1, 80, 0.20, 0.25, ood=True),
        SplitSpec("known_shift_validation_2", 40, 2, 80, 0.10, 0.30),
        SplitSpec("ood_validation_2", 50, 1, 80, 0.20, 0.35, ood=True),
        SplitSpec("confirmation", 60, 2, 80, 0.10, 0.40),
        SplitSpec("ood_confirmation", 70, 1, 80, 0.20, 0.45, ood=True),
    )
    config = TrainingConfig(
        batch_size=64,
        classifier_epochs=2,
        classifier_patience=2,
        vae_epochs=2,
        vae_patience=2,
        classifier_thresholds=(0.10, 0.50),
        vae_quantiles=(0.95,),
        sigma_tail_rates=(0.005,),
        safety_gates=SafetyGates(
            maximum_observed_escapees=100,
            maximum_relative_escape_rate=1.0,
            maximum_escape_rate_upper_95=1.0,
            maximum_overtest_rate=1.0,
        ),
        torch_threads=1,
    )

    run = run_benchmark(generate_benchmark(specs, base_seed=99), config)

    report = run.report
    assert report["training"]["classifier"]["epochs_completed"] == 2
    assert report["training"]["vae"]["epochs_completed"] == 2
    assert report["validation"]["selected_candidate"]["policy_family"] == "hybrid_or"
    assert report["selection_protocol"]["confirmation_used_for_selection"] is False
    assert report["confirmation"]["metrics"]["total_chips"] == 160
    assert report["ood_confirmation"]["metrics"]["total_chips"] == 80
    assert len(run.runtime_config["sigma"]["feature_rules"]) == 32