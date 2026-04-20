"""Run consolidated vision evaluations for one or more detector/localizer pairs."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from benchmarks.vision.dataset import load_csv, load_json
    from benchmarks.vision.evaluate import run_evaluation
    from scripts.evaluate_board_detector import evaluate_board_detector
    from scripts.evaluate_board_localizer import evaluate_board_localizer
    from scripts.evaluate_detector import evaluate_detector
    from scripts.prepare_benchmark_dataset import prepare_benchmark_dataset
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from benchmarks.vision.dataset import load_csv, load_json
    from benchmarks.vision.evaluate import run_evaluation
    from scripts.evaluate_board_detector import evaluate_board_detector
    from scripts.evaluate_board_localizer import evaluate_board_localizer
    from scripts.evaluate_detector import evaluate_detector
    from scripts.prepare_benchmark_dataset import prepare_benchmark_dataset

from chesscoach.logging_utils import add_logging_args, configure_logging
from chesscoach.vision.board_localizer import DEFAULT_BOARD_LOCALIZER_IMAGE_SIZE
from chesscoach.vision.board_localizer import BoardCornerLocalizer
from chesscoach.vision.piece_detector import (
    DEFAULT_DETECTOR_IMAGE_SIZE,
    DEFAULT_SCORE_THRESHOLD,
    PieceDetector,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Named detector/localizer evaluation target."""

    name: str
    detector_checkpoint: Path
    board_localizer_checkpoint: Path | None


def _json_ready(value: Any) -> Any:
    """Recursively convert rich Python objects into JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _load_benchmark_samples(dataset_path: Path, split: str | None) -> list[Any]:
    """Load benchmark samples and optionally filter by split."""
    samples = load_json(dataset_path) if dataset_path.suffix.lower() == ".json" else load_csv(dataset_path)
    if split is None:
        return samples
    return [sample for sample in samples if sample.split == split]


def _resolve_benchmark_dataset(
    *,
    benchmark_dataset: Path | None,
    benchmark_input: Path | None,
    benchmark_output: Path | None,
) -> Path:
    """Return a benchmark dataset path, preparing it from raw input if needed."""
    if benchmark_dataset is not None:
        return benchmark_dataset
    if benchmark_input is None:
        raise ValueError(
            "Either --benchmark-dataset or --benchmark-input must be provided."
        )
    output_path = benchmark_output or Path("data/chess_boards/benchmark_generated.csv")
    return prepare_benchmark_dataset(benchmark_input, output_path)


def _parse_model_spec(raw_value: str) -> ModelSpec:
    """Parse a CLI model spec.

    Format:
        ``name=<label>,detector=<path>[,localizer=<path>]``
    """
    fields: dict[str, str] = {}
    for chunk in raw_value.split(","):
        if "=" not in chunk:
            raise ValueError(
                "Invalid --model format. Expected "
                "'name=<label>,detector=<path>[,localizer=<path>]'"
            )
        key, value = chunk.split("=", 1)
        fields[key.strip()] = value.strip()

    name = fields.get("name")
    detector = fields.get("detector")
    if not name or not detector:
        raise ValueError(
            "Each --model must include at least name=<label> and detector=<path>."
        )
    localizer = fields.get("localizer")
    return ModelSpec(
        name=name,
        detector_checkpoint=Path(detector),
        board_localizer_checkpoint=Path(localizer) if localizer else None,
    )


def _round_float(value: float | None) -> float | None:
    """Round a float for summary output."""
    if value is None:
        return None
    return round(value, 4)


def _summarize_board_detector(
    input_dir: Path,
    *,
    split: str,
    bad_geometry_threshold_px: float,
) -> dict[str, float]:
    """Evaluate the raw board detector and return aggregate metrics."""
    diagnostics = evaluate_board_detector(
        input_dir,
        split=split,
        bad_geometry_threshold_px=bad_geometry_threshold_px,
        overlay_output_dir=None,
        overlay_limit=0,
    )
    total = len(diagnostics)
    detected = [item for item in diagnostics if item.mean_corner_error_px is not None]
    mean_errors = [item.mean_corner_error_px for item in detected if item.mean_corner_error_px is not None]
    return {
        "boards": float(total),
        "board_not_found_rate": (
            sum(1 for item in diagnostics if item.status == "board_not_found") / total
            if total
            else 0.0
        ),
        "good_geometry_rate": (
            sum(1 for item in diagnostics if item.status == "good_geometry") / total
            if total
            else 0.0
        ),
        "mean_corner_error_px": sum(mean_errors) / len(mean_errors) if mean_errors else 0.0,
    }


def _evaluate_model_spec(
    model: ModelSpec,
    *,
    detector_manifest: Path,
    localizer_manifest: Path,
    benchmark_dataset: Path,
    split: str,
    score_threshold: float,
    detector_image_size: int,
    localizer_image_size: int,
) -> dict[str, Any]:
    """Run all relevant evaluations for a single model spec."""
    detector_metrics = evaluate_detector(
        detector_manifest,
        model.detector_checkpoint,
        split=split,
        score_threshold=score_threshold,
        image_size=detector_image_size,
    )

    localizer_metrics: dict[str, float] | None = None
    if model.board_localizer_checkpoint is not None:
        localizer_metrics = evaluate_board_localizer(
            localizer_manifest,
            model.board_localizer_checkpoint,
            split=split,
            image_size=localizer_image_size,
        )

    benchmark_samples = _load_benchmark_samples(benchmark_dataset, split)
    detector_for_benchmark = PieceDetector(
        model.detector_checkpoint,
        score_threshold=score_threshold,
        image_size=detector_image_size,
    )
    e2e_detector_only = run_evaluation(benchmark_samples, detector_for_benchmark)

    e2e_with_localizer: dict[str, Any] | None = None
    if model.board_localizer_checkpoint is not None:
        board_localizer = BoardCornerLocalizer(
            model.board_localizer_checkpoint,
            image_size=localizer_image_size,
        )
        e2e_with_localizer = run_evaluation(
            benchmark_samples,
            detector_for_benchmark,
            board_localizer,
        )

    return {
        "model": asdict(model),
        "detector": detector_metrics,
        "localizer": localizer_metrics,
        "e2e_detector_only": e2e_detector_only,
        "e2e_with_localizer": e2e_with_localizer,
    }


def _summary_row(results: dict[str, Any]) -> dict[str, str]:
    """Flatten key metrics into one printable summary row."""
    detector = results["detector"]
    localizer = results["localizer"] or {}
    e2e_detector_only = results["e2e_detector_only"]
    e2e_with_localizer = results["e2e_with_localizer"] or {}
    return {
        "model": results["model"]["name"],
        "det_board": f"{detector['board_accuracy']:.4f}",
        "det_sq": f"{detector['square_accuracy']:.4f}",
        "loc_mean_px": (
            f"{localizer['mean_corner_error_px']:.2f}"
            if "mean_corner_error_px" in localizer
            else "-"
        ),
        "loc_leq20": (
            f"{localizer['boards_leq_20px_mean_error']:.4f}"
            if "boards_leq_20px_mean_error" in localizer
            else "-"
        ),
        "e2e_det_board": f"{e2e_detector_only['board_accuracy']:.4f}",
        "e2e_det_sq": f"{e2e_detector_only['square_accuracy']:.4f}",
        "e2e_loc_board": (
            f"{e2e_with_localizer['board_accuracy']:.4f}"
            if e2e_with_localizer
            else "-"
        ),
        "e2e_loc_sq": (
            f"{e2e_with_localizer['square_accuracy']:.4f}"
            if e2e_with_localizer
            else "-"
        ),
    }


def _print_summary(
    model_results: list[dict[str, Any]],
    *,
    board_detector_summary: dict[str, float] | None,
) -> None:
    """Print a concise final summary table."""
    if board_detector_summary is not None:
        LOGGER.info(
            "Raw board detector: boards=%s good_geometry_rate=%.4f "
            "board_not_found_rate=%.4f mean_corner_error_px=%.2f",
            int(board_detector_summary["boards"]),
            board_detector_summary["good_geometry_rate"],
            board_detector_summary["board_not_found_rate"],
            board_detector_summary["mean_corner_error_px"],
        )

    headers = [
        "model",
        "det_board",
        "det_sq",
        "loc_mean_px",
        "loc_leq20",
        "e2e_det_board",
        "e2e_det_sq",
        "e2e_loc_board",
        "e2e_loc_sq",
    ]
    rows = [_summary_row(result) for result in model_results]
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }
    header_line = "  ".join(header.ljust(widths[header]) for header in headers)
    separator_line = "  ".join("-" * widths[header] for header in headers)
    LOGGER.info("Vision evaluation summary:")
    LOGGER.info(header_line)
    LOGGER.info(separator_line)
    for row in rows:
        LOGGER.info("  ".join(row[header].ljust(widths[header]) for header in headers))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or more detector/localizer model sets and print a "
            "consolidated summary."
        )
    )
    add_logging_args(parser)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec: name=<label>,detector=<path>[,localizer=<path>]",
    )
    parser.add_argument("--detector-manifest", type=Path, required=True)
    parser.add_argument("--localizer-manifest", type=Path, required=True)
    parser.add_argument("--benchmark-dataset", type=Path, default=None)
    parser.add_argument(
        "--benchmark-input",
        type=Path,
        default=None,
        help="Optional raw split root to convert into benchmark CSV format automatically.",
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=None,
        help="Optional output path for the generated benchmark CSV.",
    )
    parser.add_argument(
        "--board-detector-input",
        type=Path,
        default=None,
        help="Optional raw split root for running the classic board-detector evaluation once.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        dest="score_threshold",
    )
    parser.add_argument(
        "--detector-image-size",
        type=int,
        default=DEFAULT_DETECTOR_IMAGE_SIZE,
        dest="detector_image_size",
    )
    parser.add_argument(
        "--localizer-image-size",
        type=int,
        default=DEFAULT_BOARD_LOCALIZER_IMAGE_SIZE,
        dest="localizer_image_size",
    )
    parser.add_argument(
        "--bad-geometry-threshold-px",
        type=float,
        default=20.0,
        dest="bad_geometry_threshold_px",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the full consolidated results.",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    models = [_parse_model_spec(value) for value in args.model]
    benchmark_dataset = _resolve_benchmark_dataset(
        benchmark_dataset=args.benchmark_dataset,
        benchmark_input=args.benchmark_input,
        benchmark_output=args.benchmark_output,
    )
    board_detector_summary = (
        _summarize_board_detector(
            args.board_detector_input,
            split=args.split,
            bad_geometry_threshold_px=args.bad_geometry_threshold_px,
        )
        if args.board_detector_input is not None
        else None
    )

    model_results = [
        _evaluate_model_spec(
            model,
            detector_manifest=args.detector_manifest,
            localizer_manifest=args.localizer_manifest,
            benchmark_dataset=benchmark_dataset,
            split=args.split,
            score_threshold=args.score_threshold,
            detector_image_size=args.detector_image_size,
            localizer_image_size=args.localizer_image_size,
        )
        for model in models
    ]
    _print_summary(
        model_results,
        board_detector_summary=board_detector_summary,
    )

    if args.output is not None:
        payload = {
            "split": args.split,
            "benchmark_dataset": str(benchmark_dataset),
            "board_detector": board_detector_summary,
            "models": model_results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(_json_ready(payload), indent=2))
        LOGGER.info("Wrote consolidated results to %s", args.output)


if __name__ == "__main__":
    main()
