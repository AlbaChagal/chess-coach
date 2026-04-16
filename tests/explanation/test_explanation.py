import pytest

from chesscoach.analysis.models import MoveAnalysis
from chesscoach.explanation import PositionExplainer

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
SAMPLE_MOVES = [
    MoveAnalysis("e4", "e2e4", 35, None, 20, ["e5", "Nf3", "Nc6"]),
]


def test_explain_interface():
    """PositionExplainer.explain must accept a FEN str and MoveAnalysis list and return str.
    Currently raises NotImplementedError — update this test when implemented."""
    explainer = PositionExplainer()
    with pytest.raises(NotImplementedError):
        explainer.explain(STARTING_FEN, SAMPLE_MOVES)
