"""CLI benchmark runner: evaluate vision pipeline accuracy on a labeled dataset.

Usage
-----
::

    uv run python -m benchmarks.vision.evaluate \\
        --dataset path/to/dataset.csv \\
        [--detector-checkpoint  models/piece_detector.pt] \\
        [--output results/run.json]

Omit the detector checkpoint to use the stub detector (all-empty output —
useful for verifying the pipeline wiring and board detection accuracy
independently). The legacy square-classifier path remains available only as an
explicit fallback.

The script prints a human-readable report and optionally writes a JSON file
that can be diffed between runs to track model improvement.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path

from chesscoach import mlops
from chesscoach.logging_utils import add_logging_args, configure_logging
from chesscoach.vision.board_detector import BoardNotFoundError
from chesscoach.vision.board_localizer import BoardCornerLocalizer
from chesscoach.vision.piece_classifier import PieceClassifier
from chesscoach.vision.piece_detector import PieceDetector
from chesscoach.vision.predictor import predict_fen
from benchmarks.vision.dataset import BoardSample, load_csv, load_json
from benchmarks.vision.metrics import (
    board_accuracy,
    per_piece_accuracy,
    square_accuracy,
)
LOGGER = logging.getLogger(__name__)


def _load_dataset(path: Path) -> list[BoardSample]:
    LOGGER.debug(f"Loading benchmark dataset from {path}")
    if path.suffix.lower() == ".json":
        return load_json(path)
    return load_csv(path)


def run_evaluation(
    samples: list[BoardSample],
    classifier: PieceClassifier | PieceDetector,
    board_localizer: BoardCornerLocalizer | None = None,
) -> dict[str, object]:
    """Run the pipeline on *samples* and compute accuracy metrics.

    Args:
        samples: Labeled board samples.
        classifier: Detector or legacy classifier to use.

    Returns:
        Dict with keys ``n_boards``, ``board_accuracy``, ``square_accuracy``,
        ``per_piece_accuracy``, and ``n_errors``.
    """
    predictions: list[str] = []
    expected: list[str] = []
    errors = 0
    failure_reasons: Counter[str] = Counter()

    for sample in samples:
        try:
            pred = predict_fen(sample.image_path, classifier, board_localizer)
        except BoardNotFoundError as exc:
            LOGGER.warning(f"Skipping {sample.image_path.name}: {exc}")
            failure_reasons["board_not_found"] += 1
            errors += 1
            continue
        except ValueError as exc:
            LOGGER.warning(f"Skipping {sample.image_path.name}: {exc}")
            failure_reasons["invalid_input"] += 1
            errors += 1
            continue
        except Exception as exc:  # pragma: no cover - defensive benchmark accounting
            LOGGER.exception(f"Skipping {sample.image_path.name} due to unexpected error: {exc}")
            failure_reasons["unexpected_error"] += 1
            errors += 1
            continue
        predictions.append(pred)
        expected.append(sample.fen_placement)
        LOGGER.debug(
            f"Evaluated sample={sample.image_path} predicted={pred} "
            f"expected={sample.fen_placement}"
        )

    n = len(predictions)
    if n == 0:
        return {
            "n_boards": 0,
            "n_errors": errors,
            "board_accuracy": 0.0,
            "square_accuracy": 0.0,
            "per_piece_accuracy": {},
            "failure_breakdown": dict(failure_reasons),
        }

    ba = sum(board_accuracy(p, e) for p, e in zip(predictions, expected)) / n
    sa = sum(square_accuracy(p, e) for p, e in zip(predictions, expected)) / n
    ppa = per_piece_accuracy(predictions, expected)

    return {
        "n_boards": n,
        "n_errors": errors,
        "board_accuracy": round(ba, 4),
        "square_accuracy": round(sa, 4),
        "per_piece_accuracy": {k: round(v, 4) for k, v in sorted(ppa.items())},
        "failure_breakdown": dict(failure_reasons),
    }


def _print_report(results: dict[str, object], model_name: str) -> None:
    LOGGER.info(f"=== Vision Benchmark: {model_name} ===")
    LOGGER.info(f"Boards evaluated: {results['n_boards']}")
    LOGGER.info(f"Errors / skipped: {results['n_errors']}")
    failure_breakdown = results.get("failure_breakdown", {})
    if isinstance(failure_breakdown, dict) and failure_breakdown:
        LOGGER.info("Failure breakdown:")
        for reason, count in sorted(failure_breakdown.items()):
            LOGGER.info(f"  {reason}: {count}")
    LOGGER.info(f"Board accuracy: {results['board_accuracy'] * 100:.1f}%")  # type: ignore[operator]
    LOGGER.info(f"Square accuracy: {results['square_accuracy'] * 100:.3f}%")  # type: ignore[operator]
    LOGGER.info("Per-piece accuracy:")
    for piece, acc in results["per_piece_accuracy"].items():  # type: ignore[union-attr]
        LOGGER.info(f"  {piece:>2}: {acc * 100:.1f}%")


def main(argv: list[str] | None = None) -> None:
    """Entry point for the benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate vision pipeline accuracy on a labeled dataset."
    )
    add_logging_args(parser)
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to CSV or JSON dataset file (columns: image_path, fen).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Optional dataset split filter when the dataset includes a split column.",
    )
    parser.add_argument(
        "--board-localizer-checkpoint",
        type=Path,
        default=None,
        dest="board_localizer_checkpoint",
        help="Optional learned board localizer checkpoint (.pt).",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=None,
        dest="detector_checkpoint",
        help="Path to the detector checkpoint (.pt).",
    )
    parser.add_argument(
        "--occupancy-checkpoint",
        type=Path,
        default=None,
        dest="occupancy_checkpoint",
        help="Legacy fallback: occupancy checkpoint for square classifier (.pt).",
    )
    parser.add_argument(
        "--piece-checkpoint",
        type=Path,
        default=None,
        dest="piece_checkpoint",
        help="Legacy fallback: piece checkpoint for square classifier (.pt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON results.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        default=False,
        help="Log results to MLflow (chesscoach-benchmark experiment).",
    )
    parser.add_argument(
        "--training-run-id",
        type=str,
        default=None,
        dest="training_run_id",
        help="MLflow run ID of the training run that produced the checkpoints.",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    samples = _load_dataset(args.dataset)
    if args.split is not None:
        samples = [sample for sample in samples if sample.split == args.split]
        LOGGER.info(
            f"Loaded {len(samples)} benchmark samples from {args.dataset} "
            f"after split filter={args.split}"
        )
    else:
        LOGGER.info(f"Loaded {len(samples)} benchmark samples from {args.dataset}")

    if args.detector_checkpoint is not None:
        classifier = PieceDetector(args.detector_checkpoint)
        model_name = args.detector_checkpoint.name
    elif args.occupancy_checkpoint and args.piece_checkpoint:
        classifier = PieceClassifier(
            occupancy_checkpoint=args.occupancy_checkpoint,
            piece_checkpoint=args.piece_checkpoint,
        )
        model_name = f"{args.occupancy_checkpoint.name}+{args.piece_checkpoint.name}"
    else:
        classifier = PieceDetector()
        model_name = "stub"

    board_localizer = (
        BoardCornerLocalizer(args.board_localizer_checkpoint)
        if args.board_localizer_checkpoint is not None
        else None
    )

    results = run_evaluation(samples, classifier, board_localizer)
    results["model"] = model_name

    _print_report(results, model_name)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        LOGGER.info(f"Results written to {args.output}")

    if args.mlflow:
        bench_params: dict[str, object] = {
            "dataset_path": str(args.dataset),
            "split": args.split,
            "n_samples": len(samples),
            "detector_checkpoint": str(args.detector_checkpoint),
            "board_localizer_checkpoint": str(args.board_localizer_checkpoint),
            "occupancy_checkpoint": str(args.occupancy_checkpoint),
            "piece_checkpoint": str(args.piece_checkpoint),
            "model_tag": model_name,
        }
        with mlops.training_run(
            mlops.EXPERIMENTS["benchmark"],
            f"benchmark-{model_name}",
            bench_params,
        ) as run:
            import mlflow as _mlflow

            if args.training_run_id:
                _mlflow.set_tag("source_training_run", args.training_run_id)

            scalar_metrics: dict[str, float] = {
                "board_accuracy": float(results["board_accuracy"]),  # type: ignore[arg-type]
                "square_accuracy": float(results["square_accuracy"]),  # type: ignore[arg-type]
            }
            for piece, acc in results["per_piece_accuracy"].items():  # type: ignore[union-attr]
                scalar_metrics[f"{piece}_acc"] = float(acc)
            mlops.log_epoch_metrics(scalar_metrics, step=1)
            _ = run


if __name__ == "__main__":
    main()
