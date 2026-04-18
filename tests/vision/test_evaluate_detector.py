"""Tests for detector evaluation metrics."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from chesscoach.vision.piece_assignment import PieceDetection
from scripts import evaluate_detector as evaluate_module


class _StubDetector:
    def __init__(
        self,
        checkpoint: Path,
        *,
        score_threshold: float = 0.35,
        image_size: int = 640,
    ) -> None:
        _ = checkpoint
        self._score_threshold = score_threshold
        self._image_size = image_size

    def detect(self, warped_board: np.ndarray) -> list[PieceDetection]:
        _ = warped_board
        detections = [
            PieceDetection(label="wQ", score=0.9, box=(0.0, 0.0, 100.0, 120.0)),
            PieceDetection(label="bQ", score=0.3, box=(128.0, 0.0, 228.0, 120.0)),
        ]
        return [detection for detection in detections if detection.score >= self._score_threshold]


def test_evaluate_detector_reports_occupied_accuracy_and_piece_counts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared_dir = tmp_path / "prepared"
    image_dir = prepared_dir / "images" / "val"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "board.jpg"
    cv2.imwrite(str(image_path), np.full((1024, 1024, 3), 100, dtype=np.uint8))
    manifest_path = prepared_dir / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "image_path": "images/val/board.jpg",
                "split": "val",
                "width": 1024,
                "height": 1024,
                "board_corners": [
                    [0.0, 0.0],
                    [1023.0, 0.0],
                    [1023.0, 1023.0],
                    [0.0, 1023.0],
                ],
                "annotations": [
                    {"label": "wQ", "square": "a8", "label_index": 5, "box": [0, 0, 100, 120]},
                    {"label": "bQ", "square": "b8", "label_index": 11, "box": [128, 0, 228, 120]},
                ],
            }
        )
        + "\n"
    )

    monkeypatch.setattr(evaluate_module, "PieceDetector", _StubDetector)
    metrics = evaluate_module.evaluate_detector(
        manifest_path,
        tmp_path / "checkpoint.pt",
        split="val",
        score_threshold=0.35,
    )

    assert metrics["square_accuracy"] == 63 / 64
    assert metrics["occupied_square_accuracy"] == 1 / 2
    assert metrics["boards_at_most_1_error"] == 1.0
    assert metrics["boards_at_most_2_errors"] == 1.0
    assert metrics["avg_predicted_pieces_per_board"] == 1.0
    assert metrics["avg_expected_pieces_per_board"] == 2.0
    assert metrics["avg_assigned_pieces_per_board"] == 1.0
    assert metrics["avg_same_square_rejections_per_board"] == 0.0
    assert metrics["avg_neighbor_duplicate_rejections_per_board"] == 0.0
    assert metrics["avg_missed_pieces_per_board"] == 1.0
    assert metrics["avg_extra_pieces_per_board"] == 0.0
    assert metrics["avg_wrong_label_pieces_per_board"] == 0.0
    assert metrics["per_class"]["wQ"]["precision"] == 1.0
    assert metrics["per_class"]["wQ"]["recall"] == 1.0
    assert metrics["per_class"]["bQ"]["recall"] == 0.0
