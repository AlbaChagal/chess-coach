"""Tests for full-system benchmark debug helpers."""

from __future__ import annotations

from scripts.debug_vision_benchmark import (
    _count_neighbor_square_drifts,
    _fen_to_grid,
    _fen_to_squares,
)


def test_fen_to_squares_expands_piece_placement() -> None:
    squares = _fen_to_squares("8/8/8/3k4/8/8/4K3/8")

    assert len(squares) == 64
    assert squares[27] == "bK"
    assert squares[52] == "wK"


def test_fen_to_grid_returns_eight_rows() -> None:
    grid = _fen_to_grid("8/8/8/3k4/8/8/4K3/8")

    assert len(grid) == 8
    assert all(len(row) == 8 for row in grid)
    assert grid[3][3] == "bK"
    assert grid[6][4] == "wK"


def test_count_neighbor_square_drifts_matches_adjacent_same_label_shift() -> None:
    expected = _fen_to_grid("8/8/8/8/8/8/4P3/8")
    predicted = _fen_to_grid("8/8/8/8/8/5P2/8/8")

    assert _count_neighbor_square_drifts(expected, predicted) == 1
