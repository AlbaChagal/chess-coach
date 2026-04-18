"""Evaluate board-corner detection against raw labeled board metadata."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from chesscoach.logging_utils import add_logging_args, configure_logging
from chesscoach.vision.board_detector import BoardNotFoundError, detect_board_corners

try:
    from scripts.prepare_detection_dataset import select_metadata_corners
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from scripts.prepare_detection_dataset import select_metadata_corners

LOGGER = logging.getLogger(__name__)
_IMAGE_PATTERNS = ("*.jpg", "*.png")
_GOOD_ERROR_THRESHOLDS = (5.0, 10.0, 20.0, 40.0)


@dataclass(frozen=True)
class BoardCornerDiagnostic:
    """Per-image board detector diagnostic result."""

    image_path: Path
    expected_corners: np.ndarray
    predicted_corners: np.ndarray | None
    mean_corner_error_px: float | None
    max_corner_error_px: float | None
    status: str


def mean_corner_error(expected: np.ndarray, predicted: np.ndarray) -> float:
    """Return the mean per-corner Euclidean error in pixels."""
    distances = np.linalg.norm(expected - predicted, axis=1)
    return float(distances.mean())


def max_corner_error(expected: np.ndarray, predicted: np.ndarray) -> float:
    """Return the maximum per-corner Euclidean error in pixels."""
    distances = np.linalg.norm(expected - predicted, axis=1)
    return float(distances.max())


def bucket_geometry_status(
    mean_error_px: float | None,
    *,
    bad_geometry_threshold_px: float,
) -> str:
    """Return a coarse status label for a board-corner prediction."""
    if mean_error_px is None:
        return "board_not_found"
    if mean_error_px > bad_geometry_threshold_px:
        return "bad_geometry"
    return "good_geometry"


def _load_json_payload(image_path: Path) -> dict[str, Any]:
    payload = json.loads(image_path.with_suffix(".json").read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload for {image_path}")
    return payload


def _iter_split_images(split_dir: Path) -> list[Path]:
    image_paths = [
        image_path
        for pattern in _IMAGE_PATTERNS
        for image_path in split_dir.glob(pattern)
    ]
    return sorted(image_paths)


def _evaluate_image(
    image_path: Path,
    *,
    bad_geometry_threshold_px: float,
) -> BoardCornerDiagnostic:
    payload = _load_json_payload(image_path)
    expected_corners = select_metadata_corners(payload)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Missing raw image: {image_path}")

    try:
        predicted_corners = detect_board_corners(image)
    except BoardNotFoundError:
        return BoardCornerDiagnostic(
            image_path=image_path,
            expected_corners=expected_corners,
            predicted_corners=None,
            mean_corner_error_px=None,
            max_corner_error_px=None,
            status="board_not_found",
        )

    mean_error_px = mean_corner_error(expected_corners, predicted_corners)
    max_error_px = max_corner_error(expected_corners, predicted_corners)
    status = bucket_geometry_status(
        mean_error_px,
        bad_geometry_threshold_px=bad_geometry_threshold_px,
    )
    return BoardCornerDiagnostic(
        image_path=image_path,
        expected_corners=expected_corners,
        predicted_corners=predicted_corners,
        mean_corner_error_px=mean_error_px,
        max_corner_error_px=max_error_px,
        status=status,
    )


def _draw_polygon(
    image: cv2.typing.MatLike,
    corners: np.ndarray,
    *,
    color: tuple[int, int, int],
) -> None:
    points = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
    for index, point in enumerate(corners.astype(np.int32)):
        cv2.circle(image, tuple(point), 5, color, -1)
        cv2.putText(
            image,
            str(index),
            (int(point[0]) + 6, int(point[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def _write_overlay(
    diagnostic: BoardCornerDiagnostic,
    *,
    output_dir: Path,
) -> None:
    image = cv2.imread(str(diagnostic.image_path))
    if image is None:
        raise FileNotFoundError(f"Missing raw image: {diagnostic.image_path}")

    overlay = image.copy()
    _draw_polygon(overlay, diagnostic.expected_corners, color=(255, 255, 0))
    if diagnostic.predicted_corners is not None:
        _draw_polygon(overlay, diagnostic.predicted_corners, color=(255, 0, 255))

    lines = [f"status={diagnostic.status}"]
    if diagnostic.mean_corner_error_px is not None:
        lines.append(f"mean_error_px={diagnostic.mean_corner_error_px:.2f}")
    if diagnostic.max_corner_error_px is not None:
        lines.append(f"max_error_px={diagnostic.max_corner_error_px:.2f}")

    for index, line in enumerate(lines, start=1):
        cv2.putText(
            overlay,
            line,
            (12, index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / diagnostic.image_path.name
    cv2.imwrite(str(output_path), overlay)


def _log_summary(
    diagnostics: list[BoardCornerDiagnostic],
    *,
    split: str,
    bad_geometry_threshold_px: float,
) -> None:
    total = len(diagnostics)
    board_not_found = sum(1 for item in diagnostics if item.status == "board_not_found")
    bad_geometry = sum(1 for item in diagnostics if item.status == "bad_geometry")
    good_geometry = sum(1 for item in diagnostics if item.status == "good_geometry")
    detected = total - board_not_found
    mean_errors = [
        item.mean_corner_error_px
        for item in diagnostics
        if item.mean_corner_error_px is not None
    ]
    max_errors = [
        item.max_corner_error_px
        for item in diagnostics
        if item.max_corner_error_px is not None
    ]

    LOGGER.info(f"Board detector split={split} images={total}")
    LOGGER.info(
        f"Detection summary: board_not_found={board_not_found} "
        f"bad_geometry={bad_geometry} good_geometry={good_geometry} "
        f"bad_geometry_threshold_px={bad_geometry_threshold_px:.1f}"
    )
    if detected == 0:
        LOGGER.info("No boards were detected successfully.")
        return

    assert mean_errors
    assert max_errors
    LOGGER.info(
        f"Corner errors (detected only): mean={statistics.fmean(mean_errors):.2f}px "
        f"median={statistics.median(mean_errors):.2f}px "
        f"p90={_percentile(mean_errors, 0.9):.2f}px "
        f"max_mean={max(mean_errors):.2f}px "
        f"max_corner={max(max_errors):.2f}px"
    )
    for threshold in _GOOD_ERROR_THRESHOLDS:
        under_threshold = sum(1 for value in mean_errors if value <= threshold)
        LOGGER.info(
            f"Detected boards with mean corner error <= {threshold:.0f}px: "
            f"{under_threshold}/{detected} ({under_threshold / detected:.1%})"
        )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def evaluate_board_detector(
    input_dir: Path,
    *,
    split: str,
    bad_geometry_threshold_px: float,
    overlay_output_dir: Path | None,
    overlay_limit: int,
) -> list[BoardCornerDiagnostic]:
    """Evaluate board-corner detection on a raw labeled split."""
    split_dir = input_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    diagnostics = [
        _evaluate_image(
            image_path,
            bad_geometry_threshold_px=bad_geometry_threshold_px,
        )
        for image_path in _iter_split_images(split_dir)
        if image_path.with_suffix(".json").exists()
    ]
    _log_summary(
        diagnostics,
        split=split,
        bad_geometry_threshold_px=bad_geometry_threshold_px,
    )

    if overlay_output_dir is not None and overlay_limit > 0:
        prioritized = sorted(
            diagnostics,
            key=lambda item: (
                item.status != "good_geometry",
                item.mean_corner_error_px or float("inf"),
            ),
        )
        for diagnostic in prioritized[-overlay_limit:]:
            _write_overlay(diagnostic, output_dir=overlay_output_dir)
        LOGGER.info(
            f"Board detector overlays written count={min(len(prioritized), overlay_limit)} "
            f"output={overlay_output_dir}"
        )

    return diagnostics


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate board-corner detection on raw labeled boards."
    )
    add_logging_args(parser)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--bad-geometry-threshold-px",
        type=float,
        default=20.0,
        dest="bad_geometry_threshold_px",
    )
    parser.add_argument(
        "--overlay-output",
        type=Path,
        default=None,
        dest="overlay_output",
    )
    parser.add_argument(
        "--overlay-limit",
        type=int,
        default=50,
        dest="overlay_limit",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    evaluate_board_detector(
        args.input,
        split=args.split,
        bad_geometry_threshold_px=args.bad_geometry_threshold_px,
        overlay_output_dir=args.overlay_output,
        overlay_limit=args.overlay_limit,
    )


if __name__ == "__main__":
    main()
