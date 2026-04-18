"""Benchmark end-to-end detector-backed image to FEN latency."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import mean
import time

from benchmarks.vision.dataset import BoardSample, load_csv, load_json
from chesscoach.logging_utils import add_logging_args, configure_logging
from chesscoach.vision.board_detector import BoardNotFoundError
from chesscoach.vision.piece_detector import (
    DEFAULT_DETECTOR_IMAGE_SIZE,
    PieceDetector,
)
from chesscoach.vision.predictor import predict_fen

LOGGER = logging.getLogger(__name__)


def _load_dataset(path: Path) -> list[BoardSample]:
    if path.suffix.lower() == ".json":
        return load_json(path)
    return load_csv(path)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(round((len(sorted_values) - 1) * percentile))
    return sorted_values[index]


def benchmark_latency(
    samples: list[BoardSample],
    detector: PieceDetector,
    *,
    warmup: int,
    max_samples: int | None,
) -> dict[str, float]:
    """Benchmark end-to-end prediction latency on labeled samples."""
    timed_samples = samples[:max_samples] if max_samples is not None else samples
    timings_ms: list[float] = []
    errors = 0

    for sample in timed_samples[:warmup]:
        try:
            predict_fen(sample.image_path, detector)
        except (BoardNotFoundError, ValueError):
            continue

    for sample in timed_samples:
        start_time = time.perf_counter()
        try:
            predict_fen(sample.image_path, detector)
        except (BoardNotFoundError, ValueError) as exc:
            LOGGER.warning(f"Skipping {sample.image_path.name}: {exc}")
            errors += 1
            continue
        timings_ms.append((time.perf_counter() - start_time) * 1000)

    return {
        "n_samples": float(len(timed_samples)),
        "n_measured": float(len(timings_ms)),
        "n_errors": float(errors),
        "mean_ms": mean(timings_ms) if timings_ms else 0.0,
        "p50_ms": _percentile(timings_ms, 0.50),
        "p90_ms": _percentile(timings_ms, 0.90),
        "p95_ms": _percentile(timings_ms, 0.95),
        "max_ms": max(timings_ms) if timings_ms else 0.0,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark end-to-end image to FEN latency."
    )
    add_logging_args(parser)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.05,
        dest="score_threshold",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_DETECTOR_IMAGE_SIZE,
        dest="image_size",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    samples = _load_dataset(args.dataset)
    detector = PieceDetector(
        args.detector_checkpoint,
        score_threshold=args.score_threshold,
        image_size=args.image_size,
    )
    metrics = benchmark_latency(
        samples,
        detector,
        warmup=args.warmup,
        max_samples=args.max_samples,
    )
    LOGGER.info(f"Vision latency metrics={metrics}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2))
        LOGGER.info(f"Latency metrics written to {args.output}")


if __name__ == "__main__":
    main()
