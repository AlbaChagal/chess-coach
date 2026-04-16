"""Shared fixtures for vision tests."""

from __future__ import annotations

import numpy as np
import pytest

from chesscoach.vision.types import PieceLabel, SquareGrid


def make_empty_grid() -> SquareGrid:
    row: list[PieceLabel] = ["empty"] * 8
    return [list(row) for _ in range(8)]


def starting_position_grid() -> SquareGrid:
    """Return the 8×8 grid for the standard chess starting position."""
    back_rank_black: list[PieceLabel] = ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"]
    back_rank_white: list[PieceLabel] = ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]
    empty_row: list[PieceLabel] = ["empty"] * 8
    black_pawns: list[PieceLabel] = ["bP"] * 8
    white_pawns: list[PieceLabel] = ["wP"] * 8
    return [
        back_rank_black,  # rank 8
        black_pawns,  # rank 7
        list(empty_row),
        list(empty_row),
        list(empty_row),
        list(empty_row),
        white_pawns,  # rank 2
        back_rank_white,  # rank 1
    ]


def make_synthetic_board(size: int = 512) -> np.ndarray:
    """Create a synthetic rendered chessboard image (BGR, uint8).

    The board has alternating light/dark squares with a clear black border,
    suitable for testing the board detector.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    sq = size // 8
    light = (240, 217, 181)  # light square color (BGR)
    dark = (181, 136, 99)  # dark square color (BGR)
    for row in range(8):
        for col in range(8):
            color = light if (row + col) % 2 == 0 else dark
            y1, y2 = row * sq, (row + 1) * sq
            x1, x2 = col * sq, (col + 1) * sq
            img[y1:y2, x1:x2] = color
    return img


@pytest.fixture()
def empty_grid() -> SquareGrid:
    return make_empty_grid()


@pytest.fixture()
def starting_grid() -> SquareGrid:
    return starting_position_grid()


@pytest.fixture()
def synthetic_board() -> np.ndarray:
    return make_synthetic_board()
