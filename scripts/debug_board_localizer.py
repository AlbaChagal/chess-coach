"""Write worst-case board-localizer overlays and error summaries."""

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


def _mean_corner_error(expected: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.linalg.norm(expected - predicted, axis=1).mean())


def _max_corner_error(expected: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.linalg.norm(expected - predicted, axis=1).max())


def _load_json_payload(image_path: Path) -> dict[str, Any]:
    payload = json.loads(image_path.with_suffix(".json").read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload for {image_path}")
    return payload


def _draw_polygon(
    image: cv2.typing.MatLike,
    corners: np.ndarray,
    *,
    color: tuple[int, int, int],
) -> None:
    points = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)


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


def debug_board_localizer(
    manifest_path: Path,
    checkpoint: Path,
    output_dir: Path,
    *,
    split: str,
    image_size: int,
    limit: int,
) -> None:
    """Write overlays for the highest-error board-localizer cases."""
    localizer = BoardCornerLocalizer(checkpoint, image_size=image_size)
    evaluated: list[dict[str, object]] = []

    for record in _iter_records(manifest_path, split=split):
        image_path = Path(record["image_path"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Missing board-localizer image: {image_path}")
        expected = np.array(record["board_corners"], dtype=np.float32)
        predicted = localizer.detect_corners(image)
        evaluated.append(
            {
                "image_path": image_path,
                "expected": expected,
                "predicted": predicted,
                "mean_error": _mean_corner_error(expected, predicted),
                "max_error": _max_corner_error(expected, predicted),
            }
        )

    ranked = sorted(
        evaluated,
        key=lambda item: float(item["mean_error"]),
        reverse=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in ranked[:limit]:
        image_path = item["image_path"]
        assert isinstance(image_path, Path)
        expected = item["expected"]
        predicted = item["predicted"]
        mean_error = float(item["mean_error"])
        max_error = float(item["max_error"])
        assert isinstance(expected, np.ndarray)
        assert isinstance(predicted, np.ndarray)

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Missing board-localizer image: {image_path}")
        overlay = image.copy()
        _draw_polygon(overlay, expected, color=(255, 255, 0))
        _draw_polygon(overlay, predicted, color=(255, 0, 255))
        cv2.putText(
            overlay,
            f"mean_error_px={mean_error:.2f} max_error_px={max_error:.2f}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), overlay)
        (output_dir / f"{image_path.stem}.txt").write_text(
            "\n".join(
                [
                    f"image={image_path}",
                    f"mean_error_px={mean_error:.4f}",
                    f"max_error_px={max_error:.4f}",
                    f"expected_corners={expected.tolist()}",
                    f"predicted_corners={predicted.tolist()}",
                ]
            )
            + "\n"
        )

    if evaluated:
        mean_errors = [float(item["mean_error"]) for item in evaluated]
        LOGGER.info(
            f"Board localizer debug split={split} boards={len(evaluated)} "
            f"mean_corner_error_px={float(np.mean(mean_errors)):.2f} "
            f"median_corner_error_px={float(np.median(mean_errors)):.2f} "
            f"worst_mean_corner_error_px={float(max(mean_errors)):.2f}"
        )
    LOGGER.info(
        f"Board localizer debug overlays written count={min(len(ranked), limit)} "
        f"output={output_dir}"
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Write worst-case board-localizer overlays."
    )
    add_logging_args(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_BOARD_LOCALIZER_IMAGE_SIZE,
        dest="image_size",
    )
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    debug_board_localizer(
        args.manifest,
        args.checkpoint,
        args.output,
        split=args.split,
        image_size=args.image_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
