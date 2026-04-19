"""Tests for raw-board orientation audit helpers."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scripts.audit_board_orientation_labels import (
    _square_from_point,
    audit_board_orientation_labels,
)


def test_square_from_point_uses_canonical_fen_orientation() -> None:
    assert _square_from_point(np.array([64.0, 64.0], dtype=np.float32)) == "a8"
    assert _square_from_point(np.array([960.0, 960.0], dtype=np.float32)) == "h1"


def test_audit_board_orientation_labels_reports_no_mismatches_for_consistent_board(
    monkeypatch,
    tmp_path: Path,
) -> None:
    raw_test = tmp_path / "raw" / "test"
    raw_test.mkdir(parents=True)
    image_path = raw_test / "board.jpg"
    cv2.imwrite(str(image_path), np.zeros((1024, 1024, 3), dtype=np.uint8))
    image_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "corners": [
                    [0.0, 0.0],
                    [1023.0, 0.0],
                    [1023.0, 1023.0],
                    [0.0, 1023.0],
                ],
                "pieces": [
                    {"piece": "K", "square": "a8", "box": [20.0, 20.0, 88.0, 88.0]},
                    {"piece": "Q", "square": "h8", "box": [920.0, 20.0, 88.0, 88.0]},
                    {"piece": "R", "square": "a1", "box": [20.0, 920.0, 88.0, 88.0]},
                    {
                        "piece": "k",
                        "square": "h1",
                        "box": [920.0, 920.0, 88.0, 88.0],
                    },
                ],
            }
        )
    )
    monkeypatch.setattr(
        "scripts.audit_board_orientation_labels.select_metadata_corners",
        lambda payload: np.array(payload["corners"], dtype=np.float32),
    )

    results = audit_board_orientation_labels(tmp_path / "raw", split="test")

    assert len(results) == 1
    assert results[0].mismatches == 0
    assert results[0].total_pieces == 4
