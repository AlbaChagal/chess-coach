"""Tests for hard-example export tooling."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scripts.export_board_localizer_hard_examples import (
    export_board_localizer_hard_examples,
)


def test_export_board_localizer_hard_examples_writes_weights(
    monkeypatch, tmp_path: Path
) -> None:
    image_path = tmp_path / "board.png"
    cv2.imwrite(str(image_path), np.zeros((20, 20, 3), dtype=np.uint8))
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "split": "train",
                "image_path": str(image_path),
                "board_corners": [[1, 1], [18, 1], [18, 18], [1, 18]],
            }
        )
        + "\n"
    )

    class StubLocalizer:
        def __init__(self, checkpoint: Path, image_size: int) -> None:
            self.checkpoint = checkpoint
            self.image_size = image_size

        def detect_corners(self, image: np.ndarray) -> np.ndarray:
            return np.array(
                [[3, 1], [18, 1], [18, 18], [1, 18]],
                dtype=np.float32,
            )

    monkeypatch.setattr(
        "scripts.export_board_localizer_hard_examples.BoardCornerLocalizer",
        StubLocalizer,
    )
    output_path = tmp_path / "weights.json"

    export_board_localizer_hard_examples(
        manifest_path,
        checkpoint=tmp_path / "checkpoint.pt",
        output_path=output_path,
        split="train",
        image_size=640,
        min_weight=1.0,
        max_weight=4.0,
        error_scale_px=20.0,
    )

    payload = json.loads(output_path.read_text())
    assert payload["split"] == "train"
    assert payload["default_weight"] == 1.0
    assert payload["samples"][str(image_path)] == 1.025
