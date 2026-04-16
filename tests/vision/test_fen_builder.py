"""Unit tests for fen_builder — pure logic, no ML dependencies."""

from __future__ import annotations

import pytest

from chesscoach.vision.fen_builder import build_fen
from chesscoach.vision.types import PieceLabel
from tests.vision.conftest import make_empty_grid, starting_position_grid

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
EMPTY_FEN = "8/8/8/8/8/8/8/8"


def test_starting_position() -> None:
    assert build_fen(starting_position_grid()) == STARTING_FEN


def test_empty_board() -> None:
    assert build_fen(make_empty_grid()) == EMPTY_FEN


def test_all_white_pawns_rank_2() -> None:
    grid = make_empty_grid()
    grid[6] = ["wP"] * 8
    fen = build_fen(grid)
    ranks = fen.split("/")
    assert ranks[6] == "PPPPPPPP"
    assert all(r == "8" for i, r in enumerate(ranks) if i != 6)


def test_all_black_pawns_rank_7() -> None:
    grid = make_empty_grid()
    grid[1] = ["bP"] * 8
    fen = build_fen(grid)
    ranks = fen.split("/")
    assert ranks[1] == "pppppppp"


def test_single_white_king_center() -> None:
    grid = make_empty_grid()
    grid[4][4] = "wK"
    fen = build_fen(grid)
    ranks = fen.split("/")
    assert ranks[4] == "4K3"


def test_single_black_queen_a8() -> None:
    grid = make_empty_grid()
    grid[0][0] = "bQ"
    fen = build_fen(grid)
    ranks = fen.split("/")
    assert ranks[0] == "q7"


def test_run_of_empties_at_end_of_rank() -> None:
    grid = make_empty_grid()
    grid[0][0] = "wR"
    fen = build_fen(grid)
    ranks = fen.split("/")
    assert ranks[0] == "R7"


def test_mixed_rank() -> None:
    grid = make_empty_grid()
    grid[0] = ["wR", "empty", "wB", "empty", "empty", "empty", "wB", "wR"]
    fen = build_fen(grid)
    ranks = fen.split("/")
    assert ranks[0] == "R1B3BR"


def test_all_piece_types_appear() -> None:
    """Each FEN character should appear at least once in the starting position."""
    fen = build_fen(starting_position_grid())
    for char in "rnbqkpRNBQKP":
        assert char in fen, f"Missing piece character: {char!r}"


def test_output_has_seven_slashes() -> None:
    fen = build_fen(make_empty_grid())
    assert fen.count("/") == 7


@pytest.mark.parametrize(
    "label,expected_char",
    [
        ("wP", "P"),
        ("wN", "N"),
        ("wB", "B"),
        ("wR", "R"),
        ("wQ", "Q"),
        ("wK", "K"),
        ("bP", "p"),
        ("bN", "n"),
        ("bB", "b"),
        ("bR", "r"),
        ("bQ", "q"),
        ("bK", "k"),
    ],
)
def test_each_piece_label_maps_correctly(
    label: PieceLabel,
    expected_char: str,
) -> None:
    grid = make_empty_grid()
    grid[0][0] = label
    fen = build_fen(grid)
    assert fen.startswith(expected_char)
