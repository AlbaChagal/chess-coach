"""Tests for piece-to-square assignment on warped boards."""

from __future__ import annotations

import numpy as np

from chesscoach.vision.piece_assignment import (
    PieceDetection,
    assign_detections_to_squares,
    assign_detections_to_squares_with_stats,
    assign_detections_via_homography,
)


def test_assign_detections_uses_bottom_center_point() -> None:
    detections = [
        PieceDetection(
            label="wQ",
            score=0.9,
            box=(10.0, 10.0, 110.0, 250.0),
        )
    ]

    grid = assign_detections_to_squares(detections, board_size=1024)

    assert grid[1][0] == "wQ"


def test_assign_detections_prefers_higher_confidence_on_same_square() -> None:
    detections = [
        PieceDetection(label="wQ", score=0.2, box=(0.0, 0.0, 100.0, 120.0)),
        PieceDetection(label="bQ", score=0.9, box=(10.0, 10.0, 90.0, 120.0)),
    ]

    grid = assign_detections_to_squares(detections, board_size=1024)

    assert grid[0][0] == "bQ"


def test_assign_detections_uses_bottom_strip_vote_near_square_edge() -> None:
    detections = [
        PieceDetection(label="wQ", score=0.9, box=(110.0, 0.0, 170.0, 128.0)),
    ]

    grid = assign_detections_to_squares(detections, board_size=1024)

    assert grid[0][1] == "wQ"


def test_assign_detections_suppresses_neighbor_duplicates() -> None:
    detections = [
        PieceDetection(label="wQ", score=0.95, box=(190.0, 10.0, 230.0, 126.0)),
        PieceDetection(label="wQ", score=0.60, box=(210.0, 10.0, 250.0, 126.0)),
    ]

    grid, stats = assign_detections_to_squares_with_stats(
        detections,
        board_size=1024,
    )

    assert grid[0][1] == "wQ"
    assert grid[0][2] == "empty"
    assert stats.accepted_detections == 1
    assert stats.neighbor_duplicate_rejections == 1


def test_assign_detections_keeps_adjacent_different_labels() -> None:
    detections = [
        PieceDetection(label="wQ", score=0.95, box=(190.0, 10.0, 230.0, 126.0)),
        PieceDetection(label="bQ", score=0.90, box=(192.0, 10.0, 232.0, 126.0)),
    ]

    grid, stats = assign_detections_to_squares_with_stats(
        detections,
        board_size=1024,
    )

    assert grid[0][1] == "wQ"
    assert stats.neighbor_duplicate_rejections == 0
    assert stats.same_square_rejections == 1


def test_assign_detections_via_homography_maps_raw_boxes_to_squares() -> None:
    detections = [
        PieceDetection(label="wQ", score=0.9, box=(10.0, 10.0, 110.0, 120.0)),
    ]
    board_corners = np.array(
        [[0.0, 0.0], [1023.0, 0.0], [1023.0, 1023.0], [0.0, 1023.0]],
        dtype=np.float32,
    )

    grid = assign_detections_via_homography(detections, board_corners=board_corners)

    assert grid[0][0] == "wQ"
