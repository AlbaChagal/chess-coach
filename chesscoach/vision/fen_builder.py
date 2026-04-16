"""Convert an 8x8 grid of piece labels to a FEN piece-placement string."""

from chesscoach.vision.types import PieceLabel, SquareGrid

# Maps our PieceLabel notation to FEN characters
_LABEL_TO_FEN: dict[PieceLabel, str] = {
    "empty": "",
    "wP": "P",
    "wN": "N",
    "wB": "B",
    "wR": "R",
    "wQ": "Q",
    "wK": "K",
    "bP": "p",
    "bN": "n",
    "bB": "b",
    "bR": "r",
    "bQ": "q",
    "bK": "k",
}


def build_fen(grid: SquareGrid) -> str:
    """Convert an 8×8 piece grid to a FEN piece-placement string.

    Args:
        grid: 8 rows × 8 cols, rank 8 first (index 0), file a first (index 0).

    Returns:
        FEN piece-placement string, e.g.
        ``"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"``.
    """
    ranks: list[str] = []
    for row in grid:
        rank_str = ""
        empty_count = 0
        for label in row:
            fen_char = _LABEL_TO_FEN[label]
            if fen_char == "":
                empty_count += 1
            else:
                if empty_count:
                    rank_str += str(empty_count)
                    empty_count = 0
                rank_str += fen_char
        if empty_count:
            rank_str += str(empty_count)
        ranks.append(rank_str)
    return "/".join(ranks)
