"""Tests for board-localizer dataset inspection script."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scripts.inspect_board_localizer_dataset import inspect_board_localizer_dataset


def test_inspect_board_localizer_dataset_writes_panels_and_summary(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "board.jpg"
    cv2.imwrite(str(image_path), np.zeros((200, 300, 3), dtype=np.uint8))
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "image_path": str(image_path),
                "split": "train",
                "width": 300,
                "height": 200,
                "board_corners": [[30, 20], [270, 25], [260, 180], [35, 175]],
            }
        )
        + "\n"
    )

    written = inspect_board_localizer_dataset(
        manifest_path,
        tmp_path / "out",
        split="train",
        limit=5,
        seed=0,
        augment=False,
    )

    assert len(written) == 1
    assert written[0].exists()
    assert written[0].with_suffix(".txt").exists()
    raw_summary_path = tmp_path / "out" / "train_summary_raw.json"
    sample_summary_path = tmp_path / "out" / "train_summary_sample_targets.json"
    assert raw_summary_path.exists()
    assert sample_summary_path.exists()
    raw_summary = json.loads(raw_summary_path.read_text())
    sample_summary = json.loads(sample_summary_path.read_text())
    assert raw_summary["count"] == 1
    assert raw_summary["normalized_area_mean"] > 0.0
    assert sample_summary["count"] == 1
    assert sample_summary["normalized_area_mean"] > 0.0
    assert sample_summary["augment_applied"] is False
