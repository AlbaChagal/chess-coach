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


@dataclass(frozen=True)
class AlternativeExplanation:
    """Summary of why an alternative move is inferior to the best move."""

    move_san: str
    move_uci: str
    score_display: str
    reason: str


@dataclass(frozen=True)
class StructuredExplanation:
    """Typed explanation payload for product and client rendering."""

    summary: str
    what_the_move_does: str
    what_it_threatens: str
    why_it_is_best: str
    why_alternatives_are_worse: str
    alternatives: list[AlternativeExplanation]
    tactical_themes: list[str]


@dataclass(frozen=True)
class PlayedMoveResult:
    """Typed summary of the move the user actually played."""

    move_uci: str
    move_san: str
    quality_label: MoveLabel
    quality_emoji: str
    cp_loss: int
    tactics_after_played: list[str]
    tactics_after_best: list[str]


@dataclass(frozen=True)
class BestMoveComparison:
    """Comparison between the played move and the engine's best move."""

    best_move_uci: str
    best_move_san: str
    best_move_score_display: str
    played_move_uci: str
    played_move_san: str
    played_move_quality: MoveLabel
    cp_loss: int
    why_best_move_is_better: str


@dataclass(frozen=True)
class StructuredPlayedMoveExplanation:
    """Typed coaching payload for evaluating a played move."""

    summary: str
    what_the_move_tried_to_do: str
    what_was_missed: str
    what_changed_after_move: str
    why_best_move_was_better: str
    practical_lesson: str
    tactical_themes: list[str]
    alternatives: list[AlternativeExplanation]


class ExplanationError(Exception):
    """Raised when the LLM provider call fails."""
