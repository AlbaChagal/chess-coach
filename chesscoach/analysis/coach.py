from __future__ import annotations

import logging

import chess

from chesscoach.analysis.engine import ChessEngine
from chesscoach.analysis.models import MoveAnalysis

LOGGER = logging.getLogger(__name__)


class ChessCoach:
    def __init__(self, engine: ChessEngine) -> None:
        self._engine = engine

    def parse_fen(self, fen: str) -> chess.Board:
        LOGGER.debug(f"Parsing FEN: {fen}")
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise ValueError(f"Invalid FEN: {fen!r}") from exc
        if not board.is_valid():
            raise ValueError(f"Invalid board position in FEN: {fen!r}")
        LOGGER.debug(
            f"Parsed FEN successfully. Turn={board.turn} "
            f"fullmove={board.fullmove_number}"
        )
        return board

    def analyze_position(self, fen: str, n: int = 3) -> list[MoveAnalysis]:
        LOGGER.info(f"Analyzing position with top_n={n}")
        board = self.parse_fen(fen)
        moves = self._engine.get_best_moves(board, n)
        LOGGER.info(f"Analysis completed with {len(moves)} candidate moves")
        return moves

    def format_suggestions(self, fen: str, moves: list[MoveAnalysis]) -> str:
        LOGGER.debug(f"Formatting {len(moves)} move suggestions")
        lines = [f"Top {len(moves)} moves for: {fen}", ""]
        for i, move in enumerate(moves, 1):
            score = move.score_display()
            line_str = " ".join(move.continuation) if move.continuation else "-"
            lines.append(f"{i}. {move.move_san:<6} [{score:>6}]  line: {line_str}")
        return "\n".join(lines)
