"""Shared types for the vision component."""

from typing import Literal

PieceLabel = Literal[
    "empty",
    "wP",
    "wN",
    "wB",
    "wR",
    "wQ",
    "wK",
    "bP",
    "bN",
    "bB",
    "bR",
    "bQ",
    "bK",
]

# 8 rows × 8 cols, rank 8 (row 0) → rank 1 (row 7), file a (col 0) → file h (col 7)
SquareGrid = list[list[PieceLabel]]

PIECE_LABELS: list[PieceLabel] = [
    "empty",
    "wP",
    "wN",
    "wB",
    "wR",
    "wQ",
    "wK",
    "bP",
    "bN",
    "bB",
    "bR",
    "bQ",
    "bK",
]
