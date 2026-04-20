"""Convert ChessReD aggregate annotations into per-image raw sidecars.

The existing vision data pipeline expects one JSON sidecar per board image with
fields such as ``fen``, ``corners``, and ``pieces``. ChessReD stores
annotations in one aggregate ``annotations.json`` file with separate image,
piece, corner, and split tables.

This script converts the aggregate format into one sidecar per annotated image
while preserving as much source metadata as possible. The emitted JSON keeps
the current training scripts working and also records explicit orientation
information about where the white side and ``e1`` appear in the source image.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from chesscoach.logging_utils import add_logging_args, configure_logging

LOGGER = logging.getLogger(__name__)

_BOARD_SIZE = 1024.0
_CANONICAL_CORNERS = np.array(
    [
        [0.0, 0.0],
        [_BOARD_SIZE - 1.0, 0.0],
        [_BOARD_SIZE - 1.0, _BOARD_SIZE - 1.0],
        [0.0, _BOARD_SIZE - 1.0],
    ],
    dtype=np.float32,
)
_SOURCE_WHITE_SIDE_TO_ROTATION = {
    "bottom": 0,
    "right": 1,
    "top": 2,
    "left": 3,
}
_CATEGORY_NAME_TO_PIECE = {
    "white-pawn": "P",
    "white-rook": "R",
    "white-knight": "N",
    "white-bishop": "B",
    "white-queen": "Q",
    "white-king": "K",
    "black-pawn": "p",
    "black-rook": "r",
    "black-knight": "n",
    "black-bishop": "b",
    "black-queen": "q",
    "black-king": "k",
}


@dataclass(frozen=True)
class ConvertedAnnotation:
    """Converted sidecar payload plus destination metadata."""

    image_path: Path
    split: str
    payload: dict[str, Any]


def _square_center(square: str) -> np.ndarray:
    """Return the canonical 1024x1024 center point of an algebraic square."""
    file_idx = ord(square[0]) - ord("a")
    rank_idx = 8 - int(square[1])
    step = _BOARD_SIZE / 8.0
    return np.array(
        [[(file_idx + 0.5) * step, (rank_idx + 0.5) * step]],
        dtype=np.float32,
    )


def _square_quad(square: str) -> np.ndarray:
    """Return the canonical 1024x1024 quadrilateral of an algebraic square."""
    file_idx = ord(square[0]) - ord("a")
    rank_idx = 8 - int(square[1])
    step = _BOARD_SIZE / 8.0
    left = file_idx * step
    top = rank_idx * step
    return np.array(
        [
            [left, top],
            [left + step, top],
            [left + step, top + step],
            [left, top + step],
        ],
        dtype=np.float32,
    )


def _build_fen(piece_entries: list[dict[str, Any]]) -> str:
    """Build a FEN placement string from per-piece square annotations."""
    grid = [["" for _ in range(8)] for _ in range(8)]
    for piece in piece_entries:
        piece_char = piece["piece"]
        square = piece["square"]
        file_idx = ord(square[0]) - ord("a")
        rank_idx = 8 - int(square[1])
        grid[rank_idx][file_idx] = piece_char

    ranks: list[str] = []
    for row in grid:
        empty_run = 0
        rank_chars: list[str] = []
        for value in row:
            if value:
                if empty_run:
                    rank_chars.append(str(empty_run))
                    empty_run = 0
                rank_chars.append(value)
            else:
                empty_run += 1
        if empty_run:
            rank_chars.append(str(empty_run))
        ranks.append("".join(rank_chars))
    return "/".join(ranks)


def _build_piece_entries(
    piece_annotations: list[dict[str, Any]],
    category_to_piece: dict[int, str],
) -> list[dict[str, Any]]:
    """Map ChessReD piece annotations into the legacy per-piece sidecar shape."""
    pieces: list[dict[str, Any]] = []
    for annotation in piece_annotations:
        category_id = annotation["category_id"]
        piece_char = category_to_piece.get(category_id)
        if piece_char is None:
            continue

        square = annotation["chessboard_position"]
        raw_box = annotation.get("bbox")
        box = None
        if isinstance(raw_box, list) and len(raw_box) == 4:
            box = [float(value) for value in raw_box]
        pieces.append(
            {
                "piece": piece_char,
                "square": square,
                "box": box,
                "source_annotation_id": annotation["id"],
                "source_category_id": category_id,
            }
        )

    pieces.sort(key=lambda piece: (8 - int(piece["square"][1]), piece["square"][0]))
    return pieces


def _named_source_corners(corner_annotation: dict[str, Any]) -> dict[str, list[float]]:
    """Return source image-relative corner labels in a stable order."""
    corners = corner_annotation["corners"]
    return {
        "top_left": [float(value) for value in corners["top_left"]],
        "top_right": [float(value) for value in corners["top_right"]],
        "bottom_right": [float(value) for value in corners["bottom_right"]],
        "bottom_left": [float(value) for value in corners["bottom_left"]],
    }


def _legacy_corner_list(named_corners: dict[str, list[float]]) -> list[list[float]]:
    """Return a corner list shaped like the existing raw dataset payloads."""
    return [
        named_corners["top_left"],
        named_corners["top_right"],
        named_corners["bottom_right"],
        named_corners["bottom_left"],
    ]


def _project_points(
    ordered_corners: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Project canonical board-space points into the source image."""
    transform = cv2.getPerspectiveTransform(_CANONICAL_CORNERS, ordered_corners)
    return cv2.perspectiveTransform(points[None, :, :], transform)[0]


def _distance_to_edge(
    point: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
) -> float:
    """Return the shortest Euclidean distance from a point to a line segment."""
    edge = edge_end - edge_start
    edge_length_squared = float(np.dot(edge, edge))
    if edge_length_squared == 0.0:
        return float(np.linalg.norm(point - edge_start))
    ratio = float(np.dot(point - edge_start, edge) / edge_length_squared)
    clamped_ratio = max(0.0, min(1.0, ratio))
    projection = edge_start + clamped_ratio * edge
    return float(np.linalg.norm(point - projection))


def _infer_source_white_side(
    source_corners: dict[str, list[float]],
    e1_center: np.ndarray,
) -> str:
    """Infer which edge of the source image corresponds to the white side."""
    edges = {
        "top": (
            np.array(source_corners["top_left"], dtype=np.float32),
            np.array(source_corners["top_right"], dtype=np.float32),
        ),
        "right": (
            np.array(source_corners["top_right"], dtype=np.float32),
            np.array(source_corners["bottom_right"], dtype=np.float32),
        ),
        "bottom": (
            np.array(source_corners["bottom_left"], dtype=np.float32),
            np.array(source_corners["bottom_right"], dtype=np.float32),
        ),
        "left": (
            np.array(source_corners["top_left"], dtype=np.float32),
            np.array(source_corners["bottom_left"], dtype=np.float32),
        ),
    }
    return min(
        edges,
        key=lambda side: _distance_to_edge(e1_center, edges[side][0], edges[side][1]),
    )


def _split_lookup(data: dict[str, Any]) -> dict[int, list[str]]:
    """Build an image-id to split-name mapping from ChessReD split tables."""
    lookup: dict[int, list[str]] = defaultdict(list)
    splits = data["splits"]
    for split_name in ("train", "val", "test"):
        for image_id in splits.get(split_name, {}).get("image_ids", []):
            lookup[int(image_id)].append(split_name)
    chessred2k = splits.get("chessred2k", {})
    for split_name in ("train", "val", "test"):
        for image_id in chessred2k.get(split_name, {}).get("image_ids", []):
            lookup[int(image_id)].append(f"chessred2k_{split_name}")
    return lookup


def _resolve_image_path(raw_root: Path, image_record: dict[str, Any]) -> Path | None:
    """Resolve an image record to an on-disk image path."""
    path_value = image_record.get("path")
    file_name = image_record["file_name"]
    game_id = image_record["game_id"]
    candidates: list[Path] = []

    if isinstance(path_value, str):
        candidates.append(raw_root / path_value)
        candidates.append(raw_root / path_value.replace("images/", "images-2/", 1))

    candidates.extend(
        [
            raw_root / "images" / str(game_id) / file_name,
            raw_root / "images-2" / str(game_id) / file_name,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _orientation_payload(
    source_corners: dict[str, list[float]] | None,
) -> dict[str, Any] | None:
    """Build orientation metadata when board corners are available."""
    if source_corners is None:
        return None

    ordered_corners = np.array(_legacy_corner_list(source_corners), dtype=np.float32)
    e1_center = _project_points(ordered_corners, _square_center("e1"))[0]
    e1_quad = _project_points(ordered_corners, _square_quad("e1"))
    white_side = _infer_source_white_side(source_corners, e1_center)
    return {
        "white_side": white_side,
        "rotation_to_white_bottom": _SOURCE_WHITE_SIDE_TO_ROTATION[white_side],
        "e1": {
            "square": "e1",
            "center": e1_center.tolist(),
            "corners": e1_quad.tolist(),
        },
        "canonical_board_corners": {
            "a8": ordered_corners[0].tolist(),
            "h8": ordered_corners[1].tolist(),
            "h1": ordered_corners[2].tolist(),
            "a1": ordered_corners[3].tolist(),
        },
    }


def _orientation_schema(
    *,
    orientation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build explicit orientation-status metadata for downstream consumers."""
    image_orientation_known = orientation is not None
    return {
        "label_space": {
            "known": True,
            "source": "annotated_chessboard_position",
            "fen_matches_label_space": True,
            "rank_1_known": True,
            "file_a_known": True,
            "e1_square_known": True,
        },
        "image_space": {
            "known": image_orientation_known,
            "board_geometry_known": image_orientation_known,
            "white_side_known": image_orientation_known,
            "e1_pixel_location_known": image_orientation_known,
            "requires_board_annotation": True,
        },
    }


def build_converted_annotations(data: dict[str, Any], raw_root: Path) -> list[ConvertedAnnotation]:
    """Build converted sidecars from loaded ChessReD aggregate annotations."""
    category_to_piece = {
        int(category["id"]): _CATEGORY_NAME_TO_PIECE[category["name"]]
        for category in data["categories"]
        if category["name"] in _CATEGORY_NAME_TO_PIECE
    }
    pieces_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for piece_annotation in data["annotations"]["pieces"]:
        pieces_by_image[int(piece_annotation["image_id"])].append(piece_annotation)

    corners_by_image = {
        int(corner_annotation["image_id"]): corner_annotation
        for corner_annotation in data["annotations"]["corners"]
    }
    split_by_image = _split_lookup(data)

    converted: list[ConvertedAnnotation] = []
    for image_record in data["images"]:
        image_id = int(image_record["id"])
        corner_annotation = corners_by_image.get(image_id)
        piece_annotations = pieces_by_image.get(image_id)
        primary_splits = split_by_image.get(image_id)
        if not piece_annotations or not primary_splits:
            continue

        image_path = _resolve_image_path(raw_root, image_record)
        if image_path is None:
            LOGGER.warning(
                "Skipping image_id=%s file=%s because the image file could not be found",
                image_id,
                image_record["file_name"],
            )
            continue

        pieces = _build_piece_entries(piece_annotations, category_to_piece)
        if not pieces:
            continue

        source_corners = (
            _named_source_corners(corner_annotation)
            if corner_annotation is not None
            else None
        )
        orientation = _orientation_payload(source_corners)
        ordered_corners = (
            orientation["canonical_board_corners"]
            if orientation is not None
            else None
        )

        payload = {
            "fen": _build_fen(pieces),
            "white_turn": None,
            "width": int(image_record["width"]),
            "height": int(image_record["height"]),
            "corners": (
                [
                    ordered_corners["a8"],
                    ordered_corners["h8"],
                    ordered_corners["h1"],
                    ordered_corners["a1"],
                ]
                if ordered_corners is not None
                else None
            ),
            "pieces": pieces,
            "orientation_schema": _orientation_schema(orientation=orientation),
            "orientation": orientation,
            "source": {
                "dataset": data["info"]["description"],
                "image_id": image_id,
                "annotation_corner_id": (
                    int(corner_annotation["id"])
                    if corner_annotation is not None
                    else None
                ),
                "camera": image_record.get("camera"),
                "file_name": image_record["file_name"],
                "path": str(image_path),
                "game_id": int(image_record["game_id"]),
                "move_id": int(image_record["move_id"]),
                "splits": primary_splits,
                "source_corners": source_corners,
                "source_path_hint": image_record.get("path"),
            },
        }
        converted.append(
            ConvertedAnnotation(
                image_path=image_path,
                split=primary_splits[0],
                payload=payload,
            )
        )

    return converted


def write_converted_annotations(
    converted: list[ConvertedAnnotation],
    output_dir: Path,
    *,
    link_images: bool,
    overwrite: bool,
) -> None:
    """Write converted sidecars into split directories."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in converted:
        split_dir = output_dir / item.split
        split_dir.mkdir(parents=True, exist_ok=True)

        destination_image_path = split_dir / item.image_path.name
        destination_json_path = destination_image_path.with_suffix(".json")

        if destination_json_path.exists() and not overwrite:
            LOGGER.info("Skipping existing sidecar %s", destination_json_path)
            continue

        if link_images:
            if destination_image_path.exists() or destination_image_path.is_symlink():
                destination_image_path.unlink()
            destination_image_path.symlink_to(item.image_path.resolve())
        elif not destination_image_path.exists() or overwrite:
            shutil.copy2(item.image_path, destination_image_path)

        destination_json_path.write_text(json.dumps(item.payload))


def convert_annotations(
    annotations_path: Path,
    raw_root: Path,
    output_dir: Path,
    *,
    link_images: bool = True,
    overwrite: bool = False,
) -> list[ConvertedAnnotation]:
    """Convert ChessReD aggregate annotations and write split-organized output."""
    data = json.loads(annotations_path.read_text())
    converted = build_converted_annotations(data, raw_root)
    write_converted_annotations(
        converted,
        output_dir,
        link_images=link_images,
        overwrite=overwrite,
    )
    LOGGER.info("Wrote %s converted annotations to %s", len(converted), output_dir)
    return converted


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert ChessReD aggregate annotations into per-image JSON sidecars."
        )
    )
    add_logging_args(parser)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/chess_boards/raw_2/annotations.json"),
        help="Path to the aggregate ChessReD annotations JSON.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/chess_boards/raw_2"),
        help="Root directory containing the ChessReD images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/chess_boards/raw_2_converted"),
        help="Directory where split-organized images and sidecars will be written.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of symlinking them into the output tree.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sidecars and image links in the output tree.",
    )
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    convert_annotations(
        args.annotations,
        args.raw_root,
        args.output,
        link_images=not args.copy_images,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
