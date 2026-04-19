"""Typed request and response models for the coaching pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from chesscoach.analysis.models import MoveAnalysis
from chesscoach.explanation.models import StructuredExplanation

OrientationStatus = Literal["missing_click", "user_marked", "resolved", "failed"]
AnalysisStatus = Literal["success", "skipped", "failed"]
ExplanationStatus = Literal["success", "skipped", "failed"]
CoachingStatus = Literal["success", "partial", "failed"]
UserActionRequired = Literal["white_king_start_click", "side_to_move"]


@dataclass(frozen=True)
class ImageClick:
    """Raw image click coordinates from the UI."""

    x: float
    y: float


@dataclass(frozen=True)
class PipelineWarning:
    """Structured warning for clients and CLI rendering."""

    code: str
    message: str


@dataclass(frozen=True)
class CoachingRequest:
    """Top-level request for the coaching pipeline."""

    image: Path | bytes
    side_to_move: Literal["w", "b"] | None = None
    white_king_start_click: ImageClick | None = None
    castling_rights: str | None = None
    en_passant: str | None = None
    include_explanation: bool = False
    explanation_provider: Literal["anthropic", "openai"] | None = None
    explanation_model: str | None = None
    top_n: int = 3


@dataclass(frozen=True)
class VisionResult:
    """Piece-placement result produced by the vision layer."""

    fen_placement: str | None
    vision_confidence: float | None
    orientation_status: OrientationStatus
    needs_user_confirmation: bool
    white_king_start_click: ImageClick | None
    debug: dict[str, object] | None = None


@dataclass(frozen=True)
class CompletedPosition:
    """Fully specified position ready for engine analysis."""

    fen: str
    fen_placement: str
    side_to_move: str
    castling_rights: str
    en_passant: str
    source: str
    user_confirmed_orientation: bool
    white_king_start_click: ImageClick


@dataclass(frozen=True)
class AnalysisResult:
    """Engine analysis output for a completed position."""

    fen: str
    top_moves: list[MoveAnalysis]
    engine_depth: int | None
    analysis_latency_ms: float | None
    analysis_status: AnalysisStatus


@dataclass(frozen=True)
class ExplanationResult:
    """Optional explanation output for the best engine move."""

    move_uci: str | None
    move_san: str | None
    explanation_text: str | None
    structured_explanation: StructuredExplanation | None
    provider: str | None
    status: ExplanationStatus


@dataclass(frozen=True)
class CoachingResult:
    """End-to-end pipeline result returned to clients."""

    vision: VisionResult
    position: CompletedPosition | None
    analysis: AnalysisResult | None
    explanation: ExplanationResult | None
    status: CoachingStatus
    user_action_required: UserActionRequired | None
    warnings: list[PipelineWarning] = field(default_factory=list)
