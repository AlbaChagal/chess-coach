"""Tests for board-localizer manifest preparation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scripts.prepare_board_localizer_dataset import prepare_board_localizer_dataset


def test_prepare_board_localizer_dataset_writes_manifest(tmp_path: Path) -> None:
    raw_train = tmp_path / "raw" / "train"
    raw_train.mkdir(parents=True)
    image_path = raw_train / "board.jpg"
    cv2.imwrite(str(image_path), np.full((32, 32, 3), 200, dtype=np.uint8))
    image_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "width": 32,
                "height": 32,
                "corners": [[2, 2], [30, 2], [30, 30], [2, 30]],
                "pieces": [
                    {"piece": "K", "square": "e1", "box": [18, 24, 8, 6]},
                ],
            }
        )
    )

    manifest_path = prepare_board_localizer_dataset(
        tmp_path / "raw",
        tmp_path / "board_localizer",
    )
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]

    assert len(records) == 1
    assert records[0]["split"] == "train"
    assert records[0]["image_path"] == str(image_path.resolve())
    assert records[0]["board_corners"] == [[2.0, 2.0], [30.0, 2.0], [30.0, 30.0], [2.0, 30.0]]
