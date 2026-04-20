"""Tests for board-annotation inspection script."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scripts.inspect_board_annotations import inspect_board_annotations


def test_inspect_board_annotations_writes_panels_for_cornered_samples(
    tmp_path: Path,
) -> None:
    raw_test = tmp_path / "raw" / "test"
    raw_test.mkdir(parents=True)
    image_path = raw_test / "board.jpg"
    cv2.imwrite(str(image_path), np.zeros((200, 200, 3), dtype=np.uint8))
    image_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "fen": "8/8/8/8/8/8/8/8",
                "corners": [[20, 20], [180, 20], [180, 180], [20, 180]],
                "orientation_schema": {
                    "image_space": {"known": True},
                },
                "orientation": {
                    "white_side": "bottom",
                    "rotation_to_white_bottom": 0,
                },
                "source": {"game_id": 1, "move_id": 2},
            }
        )
    )

    written = inspect_board_annotations(
        tmp_path / "raw",
        tmp_path / "out",
        split="test",
        limit=5,
        seed=0,
    )

    assert len(written) == 1
    assert written[0].exists()
    assert written[0].with_suffix(".txt").exists()
