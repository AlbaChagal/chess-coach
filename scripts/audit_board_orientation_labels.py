"""Audit raw board annotations for orientation and square-label consistency."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import path fallback
    sys.path.append(str(_REPO_ROOT))

from chesscoach.logging_utils import add_logging_args, configure_logging  # noqa: E402
from chesscoach.vision.piece_assignment import _foot_strip_points  # noqa: E402
from scripts.prepare_detection_dataset import (  # noqa: E402
    select_metadata_corners,
)

LOGGER = logging.getLogger(__name__)
_CANONICAL_BOARD_SIZE = 1024.0
_CANONICAL_CORNERS = np.array(
    [
        [0.0, 0.0],
        [_CANONICAL_BOARD_SIZE - 1.0, 0.0],
        [_CANONICAL_BOARD_SIZE - 1.0, _CANONICAL_BOARD_SIZE - 1.0],
        [0.0, _CANONICAL_BOARD_SIZE - 1.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class OrientationAuditResult:
    """Per-board orientation/square-label audit summary."""

    image_path: Path
    total_pieces: int
    mismatches: int
    mismatch_examples: list[str]


def _iter_image_paths(input_dir: Path, split: str) -> list[Path]:
    """Return image paths for one split under the raw input tree."""
    split_dir = input_dir / split
    return sorted(
        path
        for path in split_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def _load_json_payload(image_path: Path) -> dict[str, Any]:
    json_path = image_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing JSON sidecar for {image_path}")
    payload = json.loads(json_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload for {image_path}")
    return payload


def _square_from_point(point: np.ndarray) -> str | None:
    """Map a canonical board-space point to algebraic square notation."""
    x = float(point[0])
    y = float(point[1])
    if not (0.0 <= x < _CANONICAL_BOARD_SIZE and 0.0 <= y < _CANONICAL_BOARD_SIZE):
        return None
    step = _CANONICAL_BOARD_SIZE / 8.0
    file_idx = min(max(int(x / step), 0), 7)
    rank_idx = min(max(int(y / step), 0), 7)
    file_char = chr(ord("a") + file_idx)
    rank_char = str(8 - rank_idx)
    return f"{file_char}{rank_char}"


def _project_piece_feet(
    payload: dict[str, Any],
    ordered_corners: np.ndarray,
) -> list[tuple[str, str | None]]:
    """Project annotated piece foot regions into canonical board space."""
    raw_pieces = payload.get("pieces")
    if not isinstance(raw_pieces, list):
        raise ValueError("JSON payload is missing piece annotations")
    homography = cv2.getPerspectiveTransform(
        ordered_corners.astype(np.float32),
        _CANONICAL_CORNERS,
    )
    labeled_squares: list[tuple[str, str | None]] = []
    for piece in raw_pieces:
        if not isinstance(piece, dict):
            continue
        square = piece.get("square")
        box = piece.get("box")
        if not isinstance(square, str) or not isinstance(box, list) or len(box) != 4:
            continue
        x, y, width, height = [float(value) for value in box]
        detection_box = (x, y, x + width, y + height)
        foot_points = np.array(
            [[list(point) for point in _foot_strip_points(detection_box)]],
            dtype=np.float32,
        )
        projected_points = cv2.perspectiveTransform(foot_points, homography)[0]
        in_bounds = [
            point
            for point in projected_points
            if 0.0 <= point[0] < _CANONICAL_BOARD_SIZE
            and 0.0 <= point[1] < _CANONICAL_BOARD_SIZE
        ]
        if not in_bounds:
            labeled_squares.append((square, None))
            continue
        anchor = np.mean(np.array(in_bounds, dtype=np.float32), axis=0)
        labeled_squares.append((square, _square_from_point(anchor)))
    return labeled_squares


def audit_board_orientation_labels(
    input_dir: Path,
    *,
    split: str,
) -> list[OrientationAuditResult]:
    """Audit raw board metadata for orientation/square-label consistency."""
    results: list[OrientationAuditResult] = []
    skipped_missing_json = 0
    for image_path in _iter_image_paths(input_dir, split):
        try:
            payload = _load_json_payload(image_path)
        except FileNotFoundError:
            skipped_missing_json += 1
            continue
        ordered_corners = select_metadata_corners(payload)
        labeled_squares = _project_piece_feet(payload, ordered_corners)
        mismatches: list[str] = []
        for expected_square, projected_square in labeled_squares:
            if projected_square == expected_square:
                continue
            mismatches.append(f"{expected_square}->{projected_square or 'outside'}")
        results.append(
            OrientationAuditResult(
                image_path=image_path,
                total_pieces=len(labeled_squares),
                mismatches=len(mismatches),
                mismatch_examples=mismatches[:5],
            )
        )
    if skipped_missing_json:
        LOGGER.info(
            f"Skipped {skipped_missing_json} images without JSON sidecars "
            f"for split={split}"
        )
    return results


def _write_report(output_path: Path, results: list[OrientationAuditResult]) -> None:
    """Write a machine-readable audit report."""
    payload = {
        "boards": [
            {
                "image_path": str(result.image_path),
                "total_pieces": result.total_pieces,
                "mismatches": result.mismatches,
                "mismatch_examples": result.mismatch_examples,
            }
            for result in results
        ]
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Audit raw board labels for orientation consistency."
    )
    add_logging_args(parser)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    results = audit_board_orientation_labels(args.input, split=args.split)
    total_boards = len(results)
    total_piece_labels = sum(result.total_pieces for result in results)
    total_mismatches = sum(result.mismatches for result in results)
    suspicious = [result for result in results if result.mismatches > 0]

    LOGGER.info(
        f"Orientation audit split={args.split} boards={total_boards} "
        f"piece_labels={total_piece_labels} mismatches={total_mismatches}"
    )
    LOGGER.info(
        f"Boards with at least one square-label mismatch: "
        f"{len(suspicious)}/{total_boards}"
    )
    for result in suspicious[:20]:
        LOGGER.info(
            f"  {result.image_path.name}: mismatches={result.mismatches}/"
            f"{result.total_pieces} examples={result.mismatch_examples}"
        )

    if args.output is not None:
        _write_report(args.output, results)
        LOGGER.info(f"Orientation audit report written to {args.output}")


if __name__ == "__main__":
    main()
