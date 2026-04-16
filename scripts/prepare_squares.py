"""Convert raw chesscog board images + FEN labels into task-specific square crops.

For each board image the script:
1. Runs :func:`~chesscoach.vision.board_detector.detect_board` to produce a
   warped 512×512 top-down view.
2. Splits it into 64 squares.
3. Labels each square from the FEN string.
4. Saves occupancy and piece crops under ``output/occupancy/{label}/`` and
   ``output/piece/{label}/``.
5. Records the sample in a CSV manifest (``output/squares.csv``).

Boards that fail detection are skipped and logged to stderr.

Usage::

    uv run python scripts/prepare_squares.py \\
        --input  data/chesscog/raw \\
        --output data/chesscog/squares

Expected input layout (chesscog convention)::

    data/chesscog/raw/
        train/
            <id>.jpg   (board image)
            <id>.fen   (FEN string, first line used)
        val/
            ...
        test/
            ...

Output layout::

    data/chesscog/squares/
        occupancy/
            wP/ ... empty/
        piece/
            wP/ ... empty/
        squares.csv     # columns: occupancy_image_path, piece_image_path, label, split
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from chesscoach.logging_utils import add_logging_args, configure_logging
from chesscoach.vision.board_detector import (
    BOARD_SIZE,
    BoardNotFoundError,
    detect_board,
    split_into_squares,
    warp_board_from_corners,
)
from chesscoach.vision.fen_builder import _LABEL_TO_FEN  # reuse the label mapping
from chesscoach.vision.types import PIECE_LABELS, PieceLabel

_SQUARE_SIZE = 100
_OCCUPANCY_CONTEXT_SCALE = 1.0
_PIECE_CROP_WIDTH_SCALE = 1.5
_PIECE_CROP_HEIGHT_SCALE = 2.4
_PIECE_CENTER_Y_OFFSET_SCALE = -0.45
LOGGER = logging.getLogger(__name__)

# Invert the label→FEN char map to get FEN char→label
_FEN_TO_LABEL: dict[str, PieceLabel] = {
    v: k for k, v in _LABEL_TO_FEN.items() if v != ""
}
_FEN_TO_LABEL["."] = "empty"
_CANONICAL_CORNERS = np.array(
    [
        [0, 0],
        [BOARD_SIZE - 1, 0],
        [BOARD_SIZE - 1, BOARD_SIZE - 1],
        [0, BOARD_SIZE - 1],
    ],
    dtype=np.float32,
)


def _fen_to_grid(fen_placement: str) -> list[list[PieceLabel]]:
    """Expand a FEN piece-placement string into an 8×8 label grid."""
    grid: list[list[PieceLabel]] = []
    for rank_str in fen_placement.split("/"):
        row: list[PieceLabel] = []
        for ch in rank_str:
            if ch.isdigit():
                row.extend(["empty"] * int(ch))  # type: ignore[arg-type]
            else:
                label = _FEN_TO_LABEL.get(ch)
                if label is None:
                    raise ValueError(f"Unknown FEN character: {ch!r}")
                row.append(label)
        if len(row) != 8:
            raise ValueError(f"Rank does not have 8 squares: {rank_str!r}")
        grid.append(row)
    if len(grid) != 8:
        raise ValueError(f"FEN does not have 8 ranks: {fen_placement!r}")
    return grid


def _read_fen_placement(image_path: Path) -> str | None:
    """Read the piece-placement FEN for a board image sidecar."""
    fen_path = image_path.with_suffix(".fen")
    if fen_path.exists():
        return fen_path.read_text().strip().split()[0]

    json_path = image_path.with_suffix(".json")
    if not json_path.exists():
        return None

    payload = json.loads(json_path.read_text())
    fen = payload.get("fen")
    if not isinstance(fen, str) or not fen.strip():
        raise ValueError(f"Missing 'fen' string in {json_path}")
    return fen.strip().split()[0]


def _load_json_payload(image_path: Path) -> dict[str, Any] | None:
    """Load the JSON sidecar payload when available."""
    json_path = image_path.with_suffix(".json")
    if not json_path.exists():
        return None
    payload = json.loads(json_path.read_text())
    return payload if isinstance(payload, dict) else None


def _square_center(square: str) -> np.ndarray:
    """Return the canonical board-space center point of an algebraic square."""
    file_idx = ord(square[0]) - ord("a")
    rank_idx = 8 - int(square[1])
    step = BOARD_SIZE / 8
    return np.array(
        [[(file_idx + 0.5) * step, (rank_idx + 0.5) * step]],
        dtype=np.float32,
    )


def _cyclic_corner_orders(corners: np.ndarray) -> list[np.ndarray]:
    """Return all cyclic orderings of the four board corners."""
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    clockwise = corners[np.argsort(angles)]
    orders: list[np.ndarray] = []
    for base in (clockwise, clockwise[::-1]):
        for shift in range(4):
            orders.append(np.roll(base, -shift, axis=0).astype(np.float32))
    return orders


def _corner_order_score(
    corners: np.ndarray,
    pieces: list[dict[str, Any]],
) -> float:
    """Score a candidate corner ordering against annotated piece boxes."""
    matrix = cv2.getPerspectiveTransform(_CANONICAL_CORNERS, corners.astype(np.float32))
    total_distance = 0.0
    n_scored = 0
    for piece in pieces:
        square = piece.get("square")
        box = piece.get("box")
        if not isinstance(square, str) or not isinstance(box, list) or len(box) != 4:
            continue
        x, y, w, h = box
        box_center = np.array([[x + w / 2, y + h / 2]], dtype=np.float32)
        projected = cv2.perspectiveTransform(_square_center(square)[None, :, :], matrix)[0]
        total_distance += float(np.linalg.norm(projected[0] - box_center[0]))
        n_scored += 1

    return total_distance / n_scored if n_scored else float("inf")


def _select_metadata_corners(payload: dict[str, Any]) -> np.ndarray | None:
    """Infer the board corner order from JSON metadata."""
    raw_corners = payload.get("corners")
    raw_pieces = payload.get("pieces")
    if not isinstance(raw_corners, list) or len(raw_corners) != 4:
        return None
    if not isinstance(raw_pieces, list) or not raw_pieces:
        return None

    corners = np.array(raw_corners, dtype=np.float32)
    pieces = [piece for piece in raw_pieces if isinstance(piece, dict)]
    if not pieces:
        return None

    best_order: np.ndarray | None = None
    best_score = float("inf")
    for order in _cyclic_corner_orders(corners):
        score = _corner_order_score(order, pieces)
        if score < best_score:
            best_score = score
            best_order = order

    return best_order


def _load_warped_board(image_path: Path, bgr: np.ndarray) -> np.ndarray:
    """Load a board warp using JSON metadata when available, else detection."""
    payload = _load_json_payload(image_path)
    if payload is not None:
        metadata_corners = _select_metadata_corners(payload)
        if metadata_corners is not None:
            LOGGER.debug(
                f"Using JSON board corners for {image_path.name}: "
                f"{metadata_corners.tolist()}"
            )
            return warp_board_from_corners(bgr, metadata_corners)
        LOGGER.debug(
            f"JSON metadata for {image_path.name} lacked usable board corners; "
            "falling back to detector"
        )

    return detect_board(bgr)


def _log_split_summary(split: str, counters: Counter[str]) -> None:
    """Log a compact per-split processing summary."""
    total_images = counters["total_images"]
    success_boards = counters["success_boards"]
    saved_squares = counters["saved_squares"]
    LOGGER.info(
        f"Split={split} summary total_images={total_images} "
        f"labeled_images={counters['labeled_images']} success_boards={success_boards} "
        f"saved_squares={saved_squares}"
    )
    LOGGER.info(
        f"Split={split} failures missing_labels={counters['missing_labels']} "
        f"unreadable={counters['unreadable_images']} "
        f"board_not_found={counters['board_not_found']} bad_fen={counters['bad_fen']} "
        f"bad_label={counters['bad_label_files']}"
    )
    if total_images:
        LOGGER.info(
            f"Split={split} retention labeled={100 * counters['labeled_images'] / total_images:.1f}% "
            f"detected={100 * success_boards / total_images:.1f}% "
            f"saved={100 * success_boards / total_images:.1f}%"
        )


def prepare_squares(input_dir: Path, output_dir: Path) -> None:
    """Convert raw board images to labeled square images.

    Args:
        input_dir: Root of the raw chesscog dataset (contains train/val/test).
        output_dir: Root of the output square image dataset.
    """
    LOGGER.info(f"Preparing square dataset input={input_dir} output={output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for variant in ("occupancy", "piece"):
        variant_dir = output_dir / variant
        variant_dir.mkdir(exist_ok=True)
        for label in PIECE_LABELS:
            (variant_dir / label).mkdir(exist_ok=True)

    csv_path = output_dir / "squares.csv"
    splits = ["train", "val", "test"]

    n_saved = 0
    n_skipped = 0
    split_counters: dict[str, Counter[str]] = {split: Counter() for split in splits}

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "occupancy_image_path",
                "piece_image_path",
                "label",
                "split",
            ],
        )
        writer.writeheader()

        for split in splits:
            split_dir = input_dir / split
            if not split_dir.exists():
                LOGGER.warning(f"Split directory not found: {split_dir}")
                continue

            image_paths = sorted(split_dir.glob("*.jpg")) + sorted(split_dir.glob("*.png"))
            LOGGER.info(f"Processing split={split} images={len(image_paths)}")
            split_counters[split]["total_images"] = len(image_paths)
            for img_path in tqdm(image_paths, desc=split):
                try:
                    fen_placement = _read_fen_placement(img_path)
                except ValueError as exc:
                    LOGGER.warning(
                        f"Skipping {img_path.name} due to bad label file: {exc}"
                    )
                    n_skipped += 1
                    split_counters[split]["bad_label_files"] += 1
                    continue

                if fen_placement is None:
                    LOGGER.warning(
                        f"Skipping {img_path.name} due to missing .fen/.json label file"
                    )
                    n_skipped += 1
                    split_counters[split]["missing_labels"] += 1
                    continue
                split_counters[split]["labeled_images"] += 1

                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    LOGGER.warning(f"Skipping unreadable image: {img_path}")
                    n_skipped += 1
                    split_counters[split]["unreadable_images"] += 1
                    continue

                try:
                    warped = _load_warped_board(img_path, bgr)
                except BoardNotFoundError as exc:
                    LOGGER.warning(
                        f"Skipping {img_path.name} because board was not found: {exc}"
                    )
                    n_skipped += 1
                    split_counters[split]["board_not_found"] += 1
                    continue

                try:
                    grid = _fen_to_grid(fen_placement)
                except ValueError as exc:
                    LOGGER.warning(
                        f"Skipping {img_path.name} due to invalid FEN: {exc}"
                    )
                    n_skipped += 1
                    split_counters[split]["bad_fen"] += 1
                    continue

                occupancy_squares = split_into_squares(
                    warped,
                    context_scale=_OCCUPANCY_CONTEXT_SCALE,
                )
                piece_squares = split_into_squares(
                    warped,
                    crop_width_scale=_PIECE_CROP_WIDTH_SCALE,
                    crop_height_scale=_PIECE_CROP_HEIGHT_SCALE,
                    center_y_offset_scale=_PIECE_CENTER_Y_OFFSET_SCALE,
                )
                for row_idx, label_row in enumerate(grid):
                    for col_idx, label in enumerate(label_row):
                        sq_filename = f"{img_path.stem}_r{row_idx}c{col_idx}.jpg"
                        occupancy_resized = cv2.resize(
                            occupancy_squares[row_idx][col_idx],
                            (_SQUARE_SIZE, _SQUARE_SIZE),
                        )
                        piece_resized = cv2.resize(
                            piece_squares[row_idx][col_idx],
                            (_SQUARE_SIZE, _SQUARE_SIZE),
                        )
                        occupancy_path = (
                            output_dir / "occupancy" / label / sq_filename
                        )
                        piece_path = output_dir / "piece" / label / sq_filename
                        cv2.imwrite(str(occupancy_path), occupancy_resized)
                        cv2.imwrite(str(piece_path), piece_resized)
                        writer.writerow(
                            {
                                "occupancy_image_path": str(
                                    occupancy_path.relative_to(output_dir)
                                ),
                                "piece_image_path": str(
                                    piece_path.relative_to(output_dir)
                                ),
                                "label": label,
                                "split": split,
                            }
                        )
                        n_saved += 1
                        split_counters[split]["saved_squares"] += 1
                split_counters[split]["success_boards"] += 1
                LOGGER.debug(
                    f"Saved 64 labeled squares for {img_path.name} split={split}"
                )

    for split in splits:
        _log_split_summary(split, split_counters[split])
    LOGGER.info(f"Square preparation complete saved={n_saved} skipped={n_skipped}")
    LOGGER.info(f"Manifest written to {csv_path}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert chesscog board images into labeled 100×100 square images."
    )
    add_logging_args(parser)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Root of the raw chesscog dataset (contains train/val/test sub-dirs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/chesscog/squares"),
        help="Output directory for square images and manifest CSV.",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    prepare_squares(args.input, args.output)


if __name__ == "__main__":
    main()
