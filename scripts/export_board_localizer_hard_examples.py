"""Export board-localizer hard-example weights from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import path fallback
    sys.path.append(str(_REPO_ROOT))

from chesscoach.logging_utils import add_logging_args, configure_logging  # noqa: E402
from chesscoach.vision.board_localizer import (  # noqa: E402
    BoardCornerLocalizer,
    DEFAULT_BOARD_LOCALIZER_IMAGE_SIZE,
)

LOGGER = logging.getLogger(__name__)


def _iter_records(
    manifest_path: Path,
    *,
    split: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["split"] == split:
            records.append(record)
    return records


def _mean_corner_error(expected: np.ndarray, predicted: np.ndarray) -> float:
    """Return the mean corner error in pixels for one board."""
    return float(np.linalg.norm(expected - predicted, axis=1).mean())


def export_board_localizer_hard_examples(
    manifest_path: Path,
    checkpoint: Path,
    output_path: Path,
    *,
    split: str,
    image_size: int,
    min_weight: float,
    max_weight: float,
    error_scale_px: float,
) -> None:
    """Export sample weights keyed by image_path for hard-example sampling."""
    localizer = BoardCornerLocalizer(checkpoint, image_size=image_size)
    samples: dict[str, float] = {}
    errors: list[float] = []

    for record in _iter_records(manifest_path, split=split):
        image_path = Path(record["image_path"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Missing board-localizer image: {image_path}")
        expected = np.array(record["board_corners"], dtype=np.float32)
        predicted = localizer.detect_corners(image)
        mean_error = _mean_corner_error(expected, predicted)
        normalized = mean_error / max(error_scale_px, 1e-6)
        weight = min(max(min_weight + normalized, min_weight), max_weight)
        samples[str(record["image_path"])] = round(weight, 6)
        errors.append(mean_error)

    payload = {
        "split": split,
        "checkpoint": str(checkpoint),
        "default_weight": min_weight,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "error_scale_px": error_scale_px,
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    LOGGER.info(
        f"Exported board-localizer hard-example weights split={split} "
        f"samples={len(samples)} mean_corner_error_px={float(np.mean(errors)):.2f} "
        f"output={output_path}"
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export board-localizer hard-example weights."
    )
    add_logging_args(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_BOARD_LOCALIZER_IMAGE_SIZE,
        dest="image_size",
    )
    parser.add_argument("--min-weight", type=float, default=1.0, dest="min_weight")
    parser.add_argument("--max-weight", type=float, default=4.0, dest="max_weight")
    parser.add_argument(
        "--error-scale-px",
        type=float,
        default=20.0,
        dest="error_scale_px",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    export_board_localizer_hard_examples(
        args.manifest,
        args.checkpoint,
        args.output,
        split=args.split,
        image_size=args.image_size,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        error_scale_px=args.error_scale_px,
    )


if __name__ == "__main__":
    main()
