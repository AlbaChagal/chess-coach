"""Tests for board-level reranking and diagnostics."""

from __future__ import annotations

from chesscoach.vision.board_postprocess import (
    count_board_errors,
    find_mismatched_squares,
    rerank_board_candidates,
)
from chesscoach.vision.piece_assignment import SquareCandidate


def test_rerank_board_candidates_enforces_single_white_king() -> None:
    square_candidates = {
        (0, 4): [
            SquareCandidate("wK", 0.95, (0.0, 0.0, 0.0, 0.0), 0.0, 0.0, 0, 4, 0.0),
        ],
        (7, 4): [
            SquareCandidate("wK", 0.74, (0.0, 0.0, 0.0, 0.0), 0.0, 0.0, 7, 4, 0.0),
            SquareCandidate("wQ", 0.72, (0.0, 0.0, 0.0, 0.0), 0.0, 0.0, 7, 4, 0.0),
        ],
        (0, 3): [
            SquareCandidate("bK", 0.95, (0.0, 0.0, 0.0, 0.0), 0.0, 0.0, 0, 3, 0.0),
        ],
    }

    grid = rerank_board_candidates(square_candidates)

    assert grid[0][4] == "wK"
    assert grid[7][4] == "wQ"


def test_count_board_errors_breaks_down_failure_modes() -> None:
    expected = [["empty" for _ in range(8)] for _ in range(8)]
    predicted = [["empty" for _ in range(8)] for _ in range(8)]
    expected[0][0] = "wQ"
    expected[0][1] = "bQ"
    predicted[0][1] = "wR"
    predicted[0][2] = "bR"

    assert count_board_errors(expected, predicted) == (3, 1, 1, 1)


def test_find_mismatched_squares_reports_square_names() -> None:
    expected = [["empty" for _ in range(8)] for _ in range(8)]
    predicted = [["empty" for _ in range(8)] for _ in range(8)]
    expected[0][0] = "wQ"
    predicted[0][1] = "bQ"

    assert find_mismatched_squares(expected, predicted) == [
        ("a8", "wQ", "empty"),
        ("b8", "empty", "bQ"),
    ]


def test_rerank_board_candidates_can_choose_empty_for_weak_detection() -> None:
    square_candidates = {
        (0, 4): [
            SquareCandidate("wK", 0.95, (0.0, 0.0, 0.0, 0.0), 0.0, 0.0, 0, 4, 0.0),
        ],
        (7, 4): [
            SquareCandidate("bK", 0.95, (0.0, 0.0, 0.0, 0.0), 0.0, 0.0, 7, 4, 0.0),
        ],
        (4, 4): [
            SquareCandidate("wQ", 0.32, (0.0, 0.0, 0.0, 0.0), 0.0, 0.0, 4, 4, 0.0),
            SquareCandidate("wB", 0.20, (0.0, 0.0, 0.0, 0.0), 0.0, 0.0, 4, 4, 0.0),
        ],
    }

    grid = rerank_board_candidates(square_candidates)

    assert grid[4][4] == "empty"
