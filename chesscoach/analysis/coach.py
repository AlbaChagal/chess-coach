from __future__ import annotations

import chess

from chesscoach.analysis.engine import ChessEngine
from chesscoach.analysis.models import MoveAnalysis


class ChessCoach:
    def __init__(self, engine: ChessEngine) -> None:
        self._engine = engine

    def parse_fen(self, fen: str) -> chess.Board:
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise ValueError(f"Invalid FEN: {fen!r}") from exc
        if not board.is_valid():
            raise ValueError(f"Invalid board position in FEN: {fen!r}")
        return board

    def analyze_position(self, fen: str, n: int = 3) -> list[MoveAnalysis]:
        board = self.parse_fen(fen)
        return self._engine.get_best_moves(board, n)

    def format_suggestions(self, fen: str, moves: list[MoveAnalysis]) -> str:
        lines = [f"Top {len(moves)} moves for: {fen}", ""]
        for i, move in enumerate(moves, 1):
            score = move.score_display()
            line_str = " ".join(move.continuation) if move.continuation else "-"
            lines.append(f"{i}. {move.move_san:<6} [{score:>6}]  line: {line_str}")
        return "\n".join(lines)
