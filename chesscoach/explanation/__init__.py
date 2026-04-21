"""Explanation component: position analysis → human-readable coaching text.

Public API::

    from chesscoach.explanation import Explainer, ClaudeProvider, OpenAIProvider

    with ChessEngine() as engine:
        explainer = Explainer(engine, ClaudeProvider())
        text = explainer.explain_move(fen_before, move_uci)
"""

from chesscoach.analysis.models import MoveAnalysis
from chesscoach.explanation.explainer import Explainer
from chesscoach.explanation.models import (
    AlternativeExplanation,
    BestMoveComparison,
    CandidateLine,
    ExplainedMove,
    ExplanationError,
    LineFeature,
    MoveQuality,
    PositionTheme,
    PlayedMoveResult,
    RecurringIdea,
    StructuredExplanation,
    StructuredPositionExplanation,
    StructuredPlayedMoveExplanation,
    TacticInfo,
)
from chesscoach.explanation.position_synthesizer import (
    candidate_line_has_aligned_continuations,
    normalize_move_analyses,
    normalize_move_analysis,
)
from chesscoach.explanation.providers import ClaudeProvider, LLMProvider, OpenAIProvider

__all__ = [
    "ClaudeProvider",
    "AlternativeExplanation",
    "BestMoveComparison",
    "CandidateLine",
    "Explainer",
    "ExplainedMove",
    "ExplanationError",
    "LLMProvider",
    "LineFeature",
    "MoveQuality",
    "OpenAIProvider",
    "PositionTheme",
    "PlayedMoveResult",
    "PositionExplainer",
    "RecurringIdea",
    "StructuredExplanation",
    "StructuredPositionExplanation",
    "StructuredPlayedMoveExplanation",
    "TacticInfo",
    "candidate_line_has_aligned_continuations",
    "normalize_move_analyses",
    "normalize_move_analysis",
]


class PositionExplainer:
    """Backward-compatible stub.

    Use :class:`Explainer` for full functionality.  This class is kept so
    existing code that catches ``NotImplementedError`` continues to work.
    """

    def explain(self, fen: str, moves: list[MoveAnalysis]) -> str:
        """Not implemented — use :class:`Explainer` instead."""
        raise NotImplementedError
