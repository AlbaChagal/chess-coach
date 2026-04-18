"""Tests for detector dataset preparation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from chesscoach.vision.detection_dataset import DetectionDataset
from scripts.prepare_detection_dataset import (
    prepare_detection_dataset,
    select_metadata_corners,
)


def test_select_metadata_corners_uses_piece_annotations() -> None:
    payload = {
        "corners": [
            [900, 900],
            [100, 900],
            [100, 100],
            [900, 100],
        ],
        "pieces": [
            {"piece": "q", "square": "a8", "box": [120, 120, 50, 50]},
            {"piece": "Q", "square": "h1", "box": [830, 830, 50, 50]},
        ],
    }

    corners = select_metadata_corners(payload)

    assert np.allclose(
        corners,
        np.array(
            [
                [100, 100],
                [900, 100],
                [900, 900],
                [100, 900],
            ],
            dtype=np.float32,
        ),
    )


def test_prepare_detection_dataset_writes_manifest_and_images(tmp_path: Path) -> None:
    raw_train = tmp_path / "raw" / "train"
    raw_train.mkdir(parents=True)
    image_path = raw_train / "board.jpg"
    cv2.imwrite(str(image_path), np.full((128, 128, 3), 200, dtype=np.uint8))
    image_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "corners": [[20, 20], [108, 20], [108, 108], [20, 108]],
                "pieces": [
                    {"piece": "q", "square": "a8", "box": [20, 20, 20, 20]},
                    {"piece": "Q", "square": "h1", "box": [88, 88, 20, 20]},
                ],
            }
        )
    )

    manifest_path = prepare_detection_dataset(tmp_path / "raw", tmp_path / "prepared")

    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["split"] == "train"
    assert len(records[0]["annotations"]) == 2
    assert len(records[0]["board_corners"]) == 4
    assert (tmp_path / "prepared" / records[0]["image_path"]).exists()


def test_detection_dataset_loads_tensor_targets(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    image_dir = prepared_dir / "images" / "train"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "board.jpg"
    cv2.imwrite(str(image_path), np.full((64, 64, 3), 100, dtype=np.uint8))
    manifest_path = prepared_dir / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "image_path": "images/train/board.jpg",
                "split": "train",
                "width": 64,
                "height": 64,
                "annotations": [
                    {
                        "label": "bQ",
                        "label_index": 11,
                        "square": "a8",
                        "box": [1.0, 2.0, 10.0, 20.0],
                    }
                ],
            }
        )
        + "\n"
    )

    dataset = DetectionDataset(manifest_path, split="train")
    image, target = dataset[0]

    assert image.shape == (3, 64, 64)
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].tolist() == [11]


def test_detection_dataset_resizes_images_and_boxes(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    image_dir = prepared_dir / "images" / "train"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "board.jpg"
    cv2.imwrite(str(image_path), np.full((100, 200, 3), 100, dtype=np.uint8))
    manifest_path = prepared_dir / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "image_path": "images/train/board.jpg",
                "split": "train",
                "width": 200,
                "height": 100,
                "annotations": [
                    {
                        "label": "bQ",
                        "label_index": 11,
                        "square": "a8",
                        "box": [10.0, 20.0, 110.0, 70.0],
                    }
                ],
            }
        )
        + "\n"
    )

    dataset = DetectionDataset(manifest_path, split="train", image_size=50)
    image, target = dataset[0]

    assert image.shape == (3, 50, 50)
    assert target["boxes"].tolist() == [[2.5, 10.0, 27.5, 35.0]]


def test_detection_dataset_augment_flip_remaps_boxes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_dir = tmp_path / "prepared"
    image_dir = prepared_dir / "images" / "train"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "board.jpg"
    cv2.imwrite(str(image_path), np.full((64, 100, 3), 100, dtype=np.uint8))
    manifest_path = prepared_dir / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "image_path": "images/train/board.jpg",
                "split": "train",
                "width": 100,
                "height": 64,
                "annotations": [
                    {
                        "label": "bQ",
                        "label_index": 11,
                        "square": "a8",
                        "box": [10.0, 20.0, 30.0, 50.0],
                    }
                ],
            }
        )
        + "\n"
    )
    random_values = iter([0.0, 0.5, 0.5, 1.0])
    monkeypatch.setattr("chesscoach.vision.detection_dataset.random.random", lambda: next(random_values))
    monkeypatch.setattr(
        "chesscoach.vision.detection_dataset.random.uniform",
        lambda low, high: 1.0 if low >= 0.9 else 0.0,
    )

    dataset = DetectionDataset(manifest_path, split="train", augment=True)
    _, target = dataset[0]

    assert target["boxes"].tolist() == [[70.0, 20.0, 90.0, 50.0]]
