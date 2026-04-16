"""Data models for the explanation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from chesscoach.analysis.models import MoveAnalysis

MoveLabel = Literal["brilliant", "best", "good", "inaccuracy", "mistake", "blunder"]


@dataclass(frozen=True)
class MoveQuality:
    """Classification of a played move relative to the engine's best choice."""

    label: MoveLabel
    cp_loss: int  # centipawns lost vs best move (0 for mate situations or equal)
    emoji: str    # "!!" / "!" / "" / "?!" / "?" / "??"


@dataclass(frozen=True)
class TacticInfo:
    """A single tactical motif detected in a position."""

    name: str         # "fork" | "pin" | "skewer" | "hanging_piece" | "discovered_attack" | "check"
    description: str  # Human-readable description, e.g. "Knight on e5 forks king and rook"


@dataclass(frozen=True)
class ExplainedMove:
    """Fully analysed move: classification, tactics, and engine alternatives."""

    fen_before: str
    move_played_san: str
    move_played_uci: str
    quality: MoveQuality
    best_move: MoveAnalysis           # engine's top choice
    alternatives: list[MoveAnalysis]  # remaining top-N engine candidates
    tactics_after_played: list[TacticInfo]  # what the opponent can do after your move
    tactics_after_best: list[TacticInfo]    # what you gain if you play the best move


class ExplanationError(Exception):
    """Raised when the LLM provider call fails."""
