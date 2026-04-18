"""Prepare a raw-image piece detection dataset from chess board annotations."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from chesscoach.logging_utils import add_logging_args, configure_logging
from chesscoach.vision.piece_detector import detector_label_to_index
from chesscoach.vision.types import PieceLabel

LOGGER = logging.getLogger(__name__)

_PIECE_CHAR_TO_LABEL: dict[str, PieceLabel] = {
    "P": "wP",
    "N": "wN",
    "B": "wB",
    "R": "wR",
    "Q": "wQ",
    "K": "wK",
    "p": "bP",
    "n": "bN",
    "b": "bB",
    "r": "bR",
    "q": "bQ",
    "k": "bK",
}


def _load_json_payload(image_path: Path) -> dict[str, Any]:
    payload = json.loads(image_path.with_suffix(".json").read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload for {image_path}")
    return payload


def _square_center(square: str) -> np.ndarray:
    file_idx = ord(square[0]) - ord("a")
    rank_idx = 8 - int(square[1])
    step = 1024 / 8
    return np.array(
        [[(file_idx + 0.5) * step, (rank_idx + 0.5) * step]],
        dtype=np.float32,
    )


def _cyclic_corner_orders(corners: np.ndarray) -> list[np.ndarray]:
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
    board_side = 1024.0
    destination = np.array(
        [[0, 0], [board_side - 1, 0], [board_side - 1, board_side - 1], [0, board_side - 1]],
        dtype=np.float32,
    )
    matrix = np.linalg.inv(
        np.vstack(
            [
                np.column_stack([corners, np.ones(4)]),
                np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            ]
        )[:3, :3]
    )
    _ = destination
    total_distance = 0.0
    n_scored = 0
    for piece in pieces:
        square = piece.get("square")
        box = piece.get("box")
        if not isinstance(square, str) or not isinstance(box, list) or len(box) != 4:
            continue
        x, y, w, h = box
        box_center = np.array([x + w / 2, y + h / 2], dtype=np.float32)
        tl, tr, br, bl = corners
        file_idx = ord(square[0]) - ord("a")
        rank_idx = 8 - int(square[1])
        alpha = file_idx / 8
        beta = rank_idx / 8
        top = tl + alpha * (tr - tl)
        bottom = bl + alpha * (br - bl)
        left = tl + beta * (bl - tl)
        right = tr + beta * (br - tr)
        square_center = (top + bottom + left + right) / 4
        total_distance += float(np.linalg.norm(square_center - box_center))
        n_scored += 1
        _ = matrix
    return total_distance / n_scored if n_scored else float("inf")


def select_metadata_corners(payload: dict[str, Any]) -> np.ndarray:
    """Infer the correct board corner order from JSON metadata."""
    raw_corners = payload.get("corners")
    raw_pieces = payload.get("pieces")
    if not isinstance(raw_corners, list) or len(raw_corners) != 4:
        raise ValueError("JSON payload is missing four board corners")
    if not isinstance(raw_pieces, list) or not raw_pieces:
        raise ValueError("JSON payload is missing piece annotations")

    corners = np.array(raw_corners, dtype=np.float32)
    pieces = [piece for piece in raw_pieces if isinstance(piece, dict)]
    best_order: np.ndarray | None = None
    best_score = float("inf")
    for order in _cyclic_corner_orders(corners):
        score = _corner_order_score(order, pieces)
        if score < best_score:
            best_score = score
            best_order = order

    if best_order is None:
        raise ValueError("Could not infer a valid board corner order")
    return best_order


def _build_annotations(payload: dict[str, Any]) -> list[dict[str, object]]:
    raw_pieces = payload.get("pieces")
    if not isinstance(raw_pieces, list):
        raise ValueError("JSON payload is missing piece annotations")

    annotations: list[dict[str, object]] = []
    for piece in raw_pieces:
        if not isinstance(piece, dict):
            continue
        piece_char = piece.get("piece")
        square = piece.get("square")
        box = piece.get("box")
        if not isinstance(piece_char, str) or piece_char not in _PIECE_CHAR_TO_LABEL:
            continue
        if not isinstance(square, str) or not isinstance(box, list) or len(box) != 4:
            continue
        x, y, w, h = [float(value) for value in box]
        label = _PIECE_CHAR_TO_LABEL[piece_char]
        annotations.append(
            {
                "label": label,
                "label_index": detector_label_to_index(label),
                "square": square,
                "box": [x, y, x + w, y + h],
            }
        )
    return annotations


def prepare_detection_dataset(input_dir: Path, output_dir: Path) -> Path:
    """Prepare raw-image detector annotations from raw data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    splits = ("train", "val", "test")

    with manifest_path.open("w") as manifest_file:
        for split in splits:
            split_dir = input_dir / split
            if not split_dir.exists():
                LOGGER.warning(f"Split directory not found: {split_dir}")
                continue
            split_output_dir = images_dir / split
            split_output_dir.mkdir(parents=True, exist_ok=True)
            image_paths = sorted(split_dir.glob("*.jpg")) + sorted(split_dir.glob("*.png"))
            LOGGER.info(f"Preparing detection split={split} images={len(image_paths)}")
            for image_path in image_paths:
                json_path = image_path.with_suffix(".json")
                if not json_path.exists():
                    continue
                payload = _load_json_payload(image_path)
                annotations = _build_annotations(payload)
                if not annotations:
                    continue
                ordered_corners = select_metadata_corners(payload)
                copied_path = split_output_dir / image_path.name
                shutil.copy2(image_path, copied_path)
                record = {
                    "image_path": str(copied_path.relative_to(output_dir)),
                    "split": split,
                    "width": payload.get("width"),
                    "height": payload.get("height"),
                    "board_corners": ordered_corners.tolist(),
                    "annotations": annotations,
                }
                manifest_file.write(json.dumps(record) + "\n")

    LOGGER.info(f"Detection dataset manifest written to {manifest_path}")
    return manifest_path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare a raw-image detection dataset from annotations."
    )
    add_logging_args(parser)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/chess_boards/detection"),
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    prepare_detection_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
