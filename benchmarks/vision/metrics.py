"""Accuracy metrics for the vision pipeline."""

from __future__ import annotations


def _fen_to_squares(fen_placement: str) -> list[str]:
    """Expand a FEN piece-placement string into a flat list of 64 piece chars.

    Empty squares are represented as ``"."`` for easy comparison.

    Args:
        fen_placement: The first FEN field, e.g.
            ``"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"``.

    Returns:
        List of 64 single-character strings.  Rank 8 first, file a first.
    """
    squares: list[str] = []
    for rank in fen_placement.split("/"):
        for char in rank:
            if char.isdigit():
                squares.extend(["."] * int(char))
            else:
                squares.append(char)
    if len(squares) != 64:
        raise ValueError(
            f"FEN placement does not expand to 64 squares: {fen_placement!r}"
        )
    return squares


def square_accuracy(predicted: str, expected: str) -> float:
    """Fraction of the 64 squares that are correctly predicted.

    Args:
        predicted: FEN piece-placement string produced by the pipeline.
        expected:  Ground-truth FEN piece-placement string.

    Returns:
        Float in [0.0, 1.0].
    """
    pred_squares = _fen_to_squares(predicted)
    exp_squares = _fen_to_squares(expected)
    correct = sum(p == e for p, e in zip(pred_squares, exp_squares))
    return correct / 64


def board_accuracy(predicted: str, expected: str) -> bool:
    """Return ``True`` iff *predicted* exactly matches *expected*.

    Args:
        predicted: FEN piece-placement string produced by the pipeline.
        expected:  Ground-truth FEN piece-placement string.
    """
    return predicted.strip() == expected.strip()


def per_piece_accuracy(
    predictions: list[str],
    expected_list: list[str],
) -> dict[str, float]:
    """Per-piece-type accuracy across multiple boards.

    Args:
        predictions: List of predicted FEN placements (one per board).
        expected_list: Matching list of ground-truth FEN placements.

    Returns:
        Dict mapping each piece character (e.g. ``"P"``, ``"k"``) to its
        fraction of correct predictions, ignoring empty squares.
    """
    correct: dict[str, int] = {}
    total: dict[str, int] = {}

    for pred, exp in zip(predictions, expected_list):
        pred_sq = _fen_to_squares(pred)
        exp_sq = _fen_to_squares(exp)
        for p, e in zip(pred_sq, exp_sq):
            if e == ".":
                continue
            total[e] = total.get(e, 0) + 1
            if p == e:
                correct[e] = correct.get(e, 0) + 1

    return {piece: correct.get(piece, 0) / count for piece, count in total.items()}
