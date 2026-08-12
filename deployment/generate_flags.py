"""Generate bounded SKIP/RUN decisions from the frozen public policy bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from deployment.runtime import (  # noqa: E402
    ArtifactIntegrityError,
    HybridPolicyRuntime,
    InputValidationError,
)


DEFAULT_MANIFEST = REPOSITORY_ROOT / "artifacts" / "public_v1" / "runtime_manifest.json"


class FlagGenerator:
    """Run the hash-bound policy and write auditable per-chip decisions."""

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_MANIFEST,
        runtime: HybridPolicyRuntime | None = None,
    ) -> None:
        self.runtime = runtime or HybridPolicyRuntime(manifest_path)

    @staticmethod
    def _normalise_identifiers(
        data: pd.DataFrame,
        chip_id_col: str,
        lot_id_col: str,
    ) -> pd.DataFrame:
        rename = {}
        if chip_id_col in data.columns and chip_id_col != "chip_id":
            rename[chip_id_col] = "chip_id"
        if lot_id_col in data.columns and lot_id_col != "lot_id":
            rename[lot_id_col] = "lot_id"
        for source, destination in rename.items():
            if destination in data.columns:
                raise InputValidationError(
                    f"Both {source!r} and {destination!r} are present"
                )
        return data.rename(columns=rename)

    def generate_flags(
        self,
        data: pd.DataFrame,
        chip_id_col: str = "chip_id",
        lot_id_col: str = "lot_id",
    ) -> pd.DataFrame:
        """Return detailed decisions; absent lot context fails closed to RUN."""
        normalised = self._normalise_identifiers(data, chip_id_col, lot_id_col)
        return self.runtime.predict_dataframe(normalised)

    @staticmethod
    def save_predictions(
        predictions: pd.DataFrame,
        output: Path,
        output_format: str,
    ) -> None:
        """Write detailed CSV/JSON evidence or a minimal binary sortfile."""
        output.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "csv":
            predictions.to_csv(output, index=False)
        elif output_format == "json":
            json_ready = predictions.astype(object).where(
                pd.notna(predictions),
                None,
            )
            output.write_text(
                json.dumps(
                    {"records": json_ready.to_dict(orient="records")},
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        elif output_format == "sortfile":
            lines = (
                f"{record.chip_id} {int(record.flag)}"
                for record in predictions[["chip_id", "flag"]].itertuples(index=False)
            )
            output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def summary(self, predictions: pd.DataFrame) -> dict:
        """Summarize local decisions without projecting production savings."""
        skipped = int((predictions["flag"] == 0).sum())
        total = len(predictions)
        cost_model = self.runtime.config["cost_model"]
        optional_fraction = cost_model["optional_stage_units"] / (
            cost_model["early_stage_units"] + cost_model["optional_stage_units"]
        )
        return {
            "total_chips": total,
            "skip_count": skipped,
            "run_count": total - skipped,
            "skip_rate": skipped / total,
            "simulated_time_reduction_percent": (
                skipped / total * optional_fraction * 100
            ),
            "blocked_chips": int(predictions["lot_drift_blocked"].sum()),
            "model_id": predictions["model_id"].iloc[0],
            "bundle_id": predictions["bundle_id"].iloc[0],
            "scope": "local policy output; not a production outcome",
        }


def _read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("records") if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise InputValidationError(
                "JSON input must be a list or contain a records list"
            )
        return pd.DataFrame.from_records(records)
    raise InputValidationError("Input must be a .csv or .json file")


def _infer_output_format(path: Path, requested: str | None) -> str:
    if requested:
        return requested
    suffix_formats = {".csv": "csv", ".json": "json", ".txt": "sortfile"}
    try:
        return suffix_formats[path.suffix.lower()]
    except KeyError as error:
        raise InputValidationError(
            "Use --format when output is not .csv, .json, or .txt"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fail-closed chip test SKIP/RUN decisions"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV or JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output path")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Frozen runtime manifest",
    )
    parser.add_argument("--format", choices=("csv", "json", "sortfile"))
    parser.add_argument("--chip-id-col", default="chip_id")
    parser.add_argument("--lot-id-col", default="lot_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        generator = FlagGenerator(args.manifest)
        frame = _read_input(args.input)
        predictions = generator.generate_flags(
            frame,
            chip_id_col=args.chip_id_col,
            lot_id_col=args.lot_id_col,
        )
        generator.save_predictions(
            predictions,
            args.output,
            _infer_output_format(args.output, args.format),
        )
    except (ArtifactIntegrityError, InputValidationError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(generator.summary(predictions), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
