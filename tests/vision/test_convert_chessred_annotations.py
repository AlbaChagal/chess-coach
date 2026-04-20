"""Tests for ChessReD aggregate annotation conversion."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scripts.convert_chessred_annotations import (
    build_converted_annotations,
    convert_annotations,
)


def _sample_dataset() -> dict[str, object]:
    return {
        "info": {"description": "Chess Recognition Dataset (ChessReD)"},
        "images": [
            {
                "file_name": "board.jpg",
                "path": "images/0/board.jpg",
                "camera": "Phone",
                "height": 1000,
                "width": 1000,
                "game_id": 0,
                "move_id": 3,
                "id": 0,
            }
        ],
        "categories": [
            {"id": 0, "name": "white-pawn"},
            {"id": 1, "name": "white-rook"},
            {"id": 5, "name": "white-king"},
            {"id": 6, "name": "black-pawn"},
            {"id": 7, "name": "black-rook"},
            {"id": 12, "name": "empty"},
        ],
        "annotations": {
            "corners": [
                {
                    "image_id": 0,
                    "corners": {
                        "top_left": [100.0, 100.0],
                        "top_right": [900.0, 100.0],
                        "bottom_right": [900.0, 900.0],
                        "bottom_left": [100.0, 900.0],
                    },
                    "id": 77,
                }
            ],
            "pieces": [
                {
                    "image_id": 0,
                    "category_id": 7,
                    "chessboard_position": "a8",
                    "id": 1,
                    "bbox": [110.0, 110.0, 80.0, 80.0],
                },
                {
                    "image_id": 0,
                    "category_id": 7,
                    "chessboard_position": "h8",
                    "id": 2,
                    "bbox": [810.0, 110.0, 80.0, 80.0],
                },
                {
                    "image_id": 0,
                    "category_id": 6,
                    "chessboard_position": "e7",
                    "id": 3,
                    "bbox": [510.0, 210.0, 80.0, 80.0],
                },
                {
                    "image_id": 0,
                    "category_id": 5,
                    "chessboard_position": "e1",
                    "id": 4,
                    "bbox": [510.0, 810.0, 80.0, 80.0],
                },
                {
                    "image_id": 0,
                    "category_id": 1,
                    "chessboard_position": "a1",
                    "id": 5,
                    "bbox": [110.0, 810.0, 80.0, 80.0],
                },
            ],
        },
        "splits": {
            "train": {"image_ids": [0], "n_samples": 1},
            "val": {"image_ids": [], "n_samples": 0},
            "test": {"image_ids": [], "n_samples": 0},
            "chessred2k": {
                "train": {"image_ids": [], "n_samples": 0},
                "val": {"image_ids": [], "n_samples": 0},
                "test": {"image_ids": [0], "n_samples": 1},
            },
        },
    }


def _sample_dataset_without_board_annotation() -> dict[str, object]:
    payload = _sample_dataset()
    payload["images"] = [
        payload["images"][0],
        {
            "file_name": "board_2.jpg",
            "path": "images/1/board_2.jpg",
            "camera": "Phone",
            "height": 1000,
            "width": 1000,
            "game_id": 1,
            "move_id": 4,
            "id": 1,
        },
    ]
    payload["annotations"]["pieces"] = [
        *payload["annotations"]["pieces"],
        {
            "image_id": 1,
            "category_id": 7,
            "chessboard_position": "a8",
            "id": 6,
            "bbox": [110.0, 110.0, 80.0, 80.0],
        },
        {
            "image_id": 1,
            "category_id": 5,
            "chessboard_position": "e1",
            "id": 7,
            "bbox": [510.0, 810.0, 80.0, 80.0],
        },
    ]
    payload["splits"]["train"]["image_ids"] = [0, 1]
    payload["splits"]["train"]["n_samples"] = 2
    return payload


def _sample_dataset_without_piece_boxes() -> dict[str, object]:
    payload = _sample_dataset()
    for piece in payload["annotations"]["pieces"]:
        piece.pop("bbox", None)
    return payload


def test_build_converted_annotations_keeps_current_training_fields(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw_2"
    image_dir = raw_root / "images-2" / "0"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "board.jpg"
    cv2.imwrite(str(image_path), np.zeros((1000, 1000, 3), dtype=np.uint8))

    converted = build_converted_annotations(_sample_dataset(), raw_root)

    assert len(converted) == 1
    payload = converted[0].payload
    assert converted[0].split == "train"
    assert payload["fen"] == "r6r/4p3/8/8/8/8/8/R3K3"
    assert payload["white_turn"] is None
    assert payload["corners"] == [
        [100.0, 100.0],
        [900.0, 100.0],
        [900.0, 900.0],
        [100.0, 900.0],
    ]
    assert payload["orientation"]["white_side"] == "bottom"
    assert payload["orientation"]["rotation_to_white_bottom"] == 0
    assert payload["orientation"]["e1"]["square"] == "e1"
    assert payload["orientation_schema"]["label_space"]["known"] is True
    assert payload["orientation_schema"]["image_space"]["known"] is True
    assert payload["orientation_schema"]["image_space"]["e1_pixel_location_known"] is True
    assert payload["source"]["splits"] == ["train", "chessred2k_test"]
    assert [piece["square"] for piece in payload["pieces"]] == [
        "a8",
        "h8",
        "e7",
        "a1",
        "e1",
    ]


def test_convert_annotations_writes_split_json_and_image_link(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw_2"
    image_dir = raw_root / "images-2" / "0"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "board.jpg"
    cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))

    annotations_path = raw_root / "annotations.json"
    annotations_path.write_text(json.dumps(_sample_dataset()))
    output_dir = tmp_path / "converted"

    converted = convert_annotations(annotations_path, raw_root, output_dir)

    assert len(converted) == 1
    written_image = output_dir / "train" / "board.jpg"
    written_json = written_image.with_suffix(".json")
    assert written_image.exists()
    assert written_json.exists()

    payload = json.loads(written_json.read_text())
    assert payload["fen"] == "r6r/4p3/8/8/8/8/8/R3K3"
    assert payload["orientation_schema"]["label_space"]["fen_matches_label_space"] is True
    assert payload["orientation"]["white_side"] == "bottom"


def test_build_converted_annotations_keeps_piece_only_images_without_corners(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw_2"
    first_dir = raw_root / "images-2" / "0"
    second_dir = raw_root / "images-2" / "1"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    cv2.imwrite(str(first_dir / "board.jpg"), np.zeros((1000, 1000, 3), dtype=np.uint8))
    cv2.imwrite(
        str(second_dir / "board_2.jpg"),
        np.zeros((1000, 1000, 3), dtype=np.uint8),
    )

    converted = build_converted_annotations(
        _sample_dataset_without_board_annotation(),
        raw_root,
    )

    assert len(converted) == 2
    piece_only = next(item for item in converted if item.payload["source"]["image_id"] == 1)
    assert piece_only.payload["fen"] == "r7/8/8/8/8/8/8/4K3"
    assert piece_only.payload["corners"] is None
    assert piece_only.payload["orientation_schema"]["label_space"]["known"] is True
    assert piece_only.payload["orientation_schema"]["image_space"]["known"] is False
    assert (
        piece_only.payload["orientation_schema"]["image_space"]["e1_pixel_location_known"]
        is False
    )
    assert piece_only.payload["orientation"] is None
    assert piece_only.payload["source"]["annotation_corner_id"] is None


def test_build_converted_annotations_keeps_square_labels_without_bboxes(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw_2"
    image_dir = raw_root / "images-2" / "0"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "board.jpg"
    cv2.imwrite(str(image_path), np.zeros((1000, 1000, 3), dtype=np.uint8))

    converted = build_converted_annotations(
        _sample_dataset_without_piece_boxes(),
        raw_root,
    )

    assert len(converted) == 1
    payload = converted[0].payload
    assert payload["fen"] == "r6r/4p3/8/8/8/8/8/R3K3"
    assert all(piece["box"] is None for piece in payload["pieces"])
    assert payload["orientation"]["white_side"] == "bottom"
