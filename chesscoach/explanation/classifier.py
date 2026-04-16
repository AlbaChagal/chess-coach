"""Classify a played chess move by centipawn loss relative to the engine best."""

from __future__ import annotations

import math

from chesscoach.explanation.models import MoveLabel, MoveQuality

# (cp_loss_upper_bound_exclusive, label, emoji)
# Evaluated in order; first threshold where cp_loss < upper_bound wins.
_THRESHOLDS: list[tuple[float, MoveLabel, str]] = [
    (1,    "best",        ""),
    (11,   "good",        ""),
    (51,   "inaccuracy",  "?!"),
    (151,  "mistake",     "?"),
    (math.inf, "blunder", "??"),
]

# Centipawn value assigned to a mate score for comparison purposes.
# Positive = mating side is winning; we use a large number.
_MATE_CP = 10_000


def _cp_value(score_cp: int | None, score_mate: int | None) -> int:
    """Resolve a (cp, mate) pair to a single signed centipawn value."""
    if score_mate is not None:
        # Positive mate = we are mating; negative = opponent is mating.
        return _MATE_CP if score_mate > 0 else -_MATE_CP
    return score_cp if score_cp is not None else 0


def classify_move(
    played_cp: int | None,
    best_cp: int | None,
    played_mate: int | None = None,
    best_mate: int | None = None,
) -> MoveQuality:
    """Return a :class:`MoveQuality` for a move given engine scores.

    Both scores are from the perspective of the side that just moved
    (positive = good for that side).

    Args:
        played_cp: Centipawn score of the position after the played move.
        best_cp: Centipawn score after the engine's best move.
        played_mate: Mate-in-N for the played move (positive = we mate).
        best_mate: Mate-in-N for the best move (positive = we mate).

    Returns:
        A :class:`MoveQuality` with label, cp_loss, and emoji.
    """
    played_val = _cp_value(played_cp, played_mate)
    best_val = _cp_value(best_cp, best_mate)

    # CP loss: how much worse than the best move (clamped to 0).
    cp_loss = max(0, best_val - played_val)

    for upper, label, emoji in _THRESHOLDS:
        if cp_loss < upper:
            return MoveQuality(label=label, cp_loss=cp_loss, emoji=emoji)

    # Unreachable — math.inf catches everything — but keeps type checker happy.
    return MoveQuality(label="blunder", cp_loss=cp_loss, emoji="??")
