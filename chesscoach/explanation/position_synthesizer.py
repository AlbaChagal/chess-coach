"""Helpers for normalizing engine analysis into position-level line models."""

from __future__ import annotations

from chesscoach.analysis.models import MoveAnalysis
from chesscoach.explanation.models import CandidateLine


def normalize_move_analysis(move: MoveAnalysis) -> CandidateLine:
    """Convert one engine analysis result into a normalized candidate line.

    Args:
        move: Engine analysis for a single candidate move.

    Returns:
        A normalized :class:`~chesscoach.explanation.models.CandidateLine`.

    Raises:
        ValueError: If the root move SAN or UCI is missing.
    """
    if not move.move_san or move.move_san == "?":
        raise ValueError("MoveAnalysis is missing a usable SAN root move.")
    if not move.move_uci or move.move_uci == "?":
        raise ValueError("MoveAnalysis is missing a usable UCI root move.")

    return CandidateLine(
        root_move_uci=move.move_uci,
        root_move_san=move.move_san,
        score_cp=move.score_cp,
        score_mate=move.score_mate,
        depth=move.depth,
        continuation_san=list(move.continuation),
        continuation_uci=list(move.continuation_uci),
    )


def normalize_move_analyses(moves: list[MoveAnalysis]) -> list[CandidateLine]:
    """Convert engine candidate moves into normalized candidate lines."""
    return [normalize_move_analysis(move) for move in moves]


def candidate_line_has_aligned_continuations(line: CandidateLine) -> bool:
    """Return whether the SAN and UCI continuation lists are aligned."""
    return len(line.continuation_san) == len(line.continuation_uci)
