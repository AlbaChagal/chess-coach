"""Tests for board-localizer helpers."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import random
from chesscoach.vision.board_localizer import denormalize_corners, normalize_corners
from chesscoach.vision.board_localizer_dataset import (
    BoardLocalizationDataset,
    _apply_perspective_jitter,
)
from scripts.train_board_localizer import _load_sample_weights


def test_corner_normalization_round_trips() -> None:
    corners = np.array(
        [[10.0, 20.0], [110.0, 20.0], [110.0, 220.0], [10.0, 220.0]],
        dtype=np.float32,
    )

    normalized = normalize_corners(corners, 200, 400)
    restored = denormalize_corners(normalized, 200, 400)

    np.testing.assert_allclose(restored, corners)


def test_perspective_jitter_can_leave_sample_unchanged(monkeypatch) -> None:
    image = np.full((40, 60, 3), 127, dtype=np.uint8)
    corners = np.array(
        [[5.0, 6.0], [55.0, 6.0], [55.0, 34.0], [5.0, 34.0]],
        dtype=np.float32,
    )

    monkeypatch.setattr(random, "random", lambda: 1.0)

    warped_image, warped_corners = _apply_perspective_jitter(image, corners)

    np.testing.assert_array_equal(warped_image, image)
    np.testing.assert_allclose(warped_corners, corners)


def test_load_sample_weights_matches_dataset_ids(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"not-used")
    manifest_path.write_text(
        json.dumps(
            {
                "split": "train",
                "image_path": "sample.png",
                "board_corners": [[1, 1], [9, 1], [9, 9], [1, 9]],
            }
        )
        + "\n"
    )
    dataset = BoardLocalizationDataset(
        manifest_path,
        split="train",
        root=tmp_path,
        image_size=64,
    )
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(
        json.dumps(
            {
                "default_weight": 1.0,
                "samples": {"sample.png": 2.5},
            }
        )
    )

    weights = _load_sample_weights(dataset, weights_path)

    assert weights == [2.5]
