# Explanation component: MoveAnalysis list → human-readable text
# TODO: implement LLM-based explanation of the position and suggested moves

from chesscoach.analysis.models import MoveAnalysis


class PositionExplainer:
    def explain(self, fen: str, moves: list[MoveAnalysis]) -> str:
        raise NotImplementedError
