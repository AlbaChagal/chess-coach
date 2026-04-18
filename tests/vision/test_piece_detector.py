"""Tests for piece detector postprocessing."""

from __future__ import annotations

from chesscoach.vision.piece_assignment import PieceDetection
from chesscoach.vision.piece_detector import _apply_class_agnostic_nms


def test_class_agnostic_nms_removes_cross_class_overlap() -> None:
    detections = [
        PieceDetection(label="wQ", score=0.95, box=(10.0, 10.0, 110.0, 210.0)),
        PieceDetection(label="bQ", score=0.80, box=(15.0, 15.0, 105.0, 205.0)),
        PieceDetection(label="wR", score=0.70, box=(200.0, 10.0, 260.0, 120.0)),
    ]

    filtered = _apply_class_agnostic_nms(detections)

    assert filtered == [detections[0], detections[2]]


def test_class_agnostic_nms_keeps_non_overlapping_detections() -> None:
    detections = [
        PieceDetection(label="wQ", score=0.95, box=(10.0, 10.0, 110.0, 210.0)),
        PieceDetection(label="bQ", score=0.80, box=(150.0, 15.0, 245.0, 205.0)),
    ]

    filtered = _apply_class_agnostic_nms(detections)

    assert filtered == detections
