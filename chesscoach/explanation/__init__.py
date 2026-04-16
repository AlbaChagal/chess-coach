"""Explanation component: position analysis → human-readable coaching text.

Public API::

    from chesscoach.explanation import Explainer, ClaudeProvider, OpenAIProvider

    with ChessEngine() as engine:
        explainer = Explainer(engine, ClaudeProvider())
        text = explainer.explain_move(fen_before, move_uci)
"""

from chesscoach.analysis.models import MoveAnalysis
from chesscoach.explanation.explainer import Explainer
from chesscoach.explanation.models import ExplainedMove, ExplanationError, MoveQuality, TacticInfo
from chesscoach.explanation.providers import ClaudeProvider, LLMProvider, OpenAIProvider

__all__ = [
    "ClaudeProvider",
    "Explainer",
    "ExplainedMove",
    "ExplanationError",
    "LLMProvider",
    "MoveQuality",
    "OpenAIProvider",
    "PositionExplainer",
    "TacticInfo",
]


class PositionExplainer:
    """Backward-compatible stub.

    Use :class:`Explainer` for full functionality.  This class is kept so
    existing code that catches ``NotImplementedError`` continues to work.
    """

    def explain(self, fen: str, moves: list[MoveAnalysis]) -> str:
        """Not implemented — use :class:`Explainer` instead."""
        raise NotImplementedError
