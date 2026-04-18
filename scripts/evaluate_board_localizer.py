"""Evaluate a learned board-corner localizer on a prepared manifest."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from chesscoach.logging_utils import add_logging_args, configure_logging
from chesscoach.vision.board_localizer import (
    BoardCornerLocalizer,
    DEFAULT_BOARD_LOCALIZER_IMAGE_SIZE,
)

LOGGER = logging.getLogger(__name__)


def _mean_corner_error(expected: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.linalg.norm(expected - predicted, axis=1).mean())


def _max_corner_error(expected: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.linalg.norm(expected - predicted, axis=1).max())


def evaluate_board_localizer(
    manifest_path: Path,
    checkpoint: Path,
    *,
    split: str,
    image_size: int,
) -> dict[str, float]:
    """Evaluate a board localizer against manifest corner annotations."""
    localizer = BoardCornerLocalizer(checkpoint, image_size=image_size)
    mean_errors: list[float] = []
    max_errors: list[float] = []
    count = 0

    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["split"] != split:
            continue
        image = cv2.imread(record["image_path"])
        if image is None:
            raise FileNotFoundError(f"Missing board-localizer image: {record['image_path']}")
        expected = np.array(record["board_corners"], dtype=np.float32)
        predicted = localizer.detect_corners(image)
        mean_errors.append(_mean_corner_error(expected, predicted))
        max_errors.append(_max_corner_error(expected, predicted))
        count += 1

    metrics = {
        "boards": float(count),
        "mean_corner_error_px": float(np.mean(mean_errors)) if mean_errors else 0.0,
        "median_corner_error_px": float(np.median(mean_errors)) if mean_errors else 0.0,
        "max_corner_error_px": float(np.max(max_errors)) if max_errors else 0.0,
        "boards_leq_20px_mean_error": (
            sum(1 for error in mean_errors if error <= 20.0) / count if count else 0.0
        ),
    }
    LOGGER.info(
        f"Board localizer evaluation split={split} boards={count} "
        f"mean_corner_error_px={metrics['mean_corner_error_px']:.2f} "
        f"median_corner_error_px={metrics['median_corner_error_px']:.2f} "
        f"max_corner_error_px={metrics['max_corner_error_px']:.2f} "
        f"boards_leq_20px_mean_error={metrics['boards_leq_20px_mean_error']:.4f}"
    )
    return metrics


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate the board localizer.")
    add_logging_args(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_BOARD_LOCALIZER_IMAGE_SIZE,
        dest="image_size",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    evaluate_board_localizer(
        args.manifest,
        args.checkpoint,
        split=args.split,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
