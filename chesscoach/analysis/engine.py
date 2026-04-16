from __future__ import annotations

import logging

import chess
import chess.engine

from chesscoach.analysis.models import MoveAnalysis

CONTINUATION_MOVES = 3
LOGGER = logging.getLogger(__name__)


class ChessEngine:
    def __init__(self, engine_path: str = "stockfish", depth: int = 20) -> None:
        self._engine_path = engine_path
        self._depth = depth
        self._engine: chess.engine.SimpleEngine | None = None

    def __enter__(self) -> "ChessEngine":
        LOGGER.debug(f"Opening engine at {self._engine_path}")
        self._engine = chess.engine.SimpleEngine.popen_uci(self._engine_path)
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        if self._engine is not None:
            LOGGER.debug(f"Closing engine at {self._engine_path}")
            self._engine.quit()
            self._engine = None

    def get_best_moves(self, board: chess.Board, n: int = 3) -> list[MoveAnalysis]:
        owned = self._engine is None
        if owned:
            LOGGER.debug(f"Opening engine on demand at {self._engine_path}")
            self._engine = chess.engine.SimpleEngine.popen_uci(self._engine_path)
        try:
            LOGGER.info(
                f"Requesting engine analysis depth={self._depth} multipv={n} "
                f"fen={board.fen()}"
            )
            infos = self._engine.analyse(
                board,
                chess.engine.Limit(depth=self._depth),
                multipv=n,
            )
            analyses = [self._info_to_analysis(board, info) for info in infos]
            LOGGER.debug(f"Engine returned {len(analyses)} principal variations")
            return analyses
        finally:
            if owned:
                self.close()

    def _info_to_analysis(
        self, board: chess.Board, info: chess.engine.InfoDict
    ) -> MoveAnalysis:
        pv: list[chess.Move] = info.get("pv", [])
        first_move = pv[0] if pv else None

        move_san = board.san(first_move) if first_move else "?"
        move_uci = first_move.uci() if first_move else "?"

        pov_score = info["score"].relative
        if pov_score.is_mate():
            score_cp = None
            score_mate = pov_score.mate()
        else:
            score_cp = pov_score.score()
            score_mate = None

        depth: int = info.get("depth", 0)
        continuation = self._extract_continuation(board, pv)
        LOGGER.debug(
            f"Converted engine info to analysis move={move_uci} depth={depth} "
            f"score_cp={score_cp} score_mate={score_mate}"
        )

        return MoveAnalysis(
            move_san=move_san,
            move_uci=move_uci,
            score_cp=score_cp,
            score_mate=score_mate,
            depth=depth,
            continuation=continuation,
        )

    @staticmethod
    def _extract_continuation(
        board: chess.Board, pv: list[chess.Move]
    ) -> list[str]:
        # Skip the first move (it's the suggestion itself); show the next N
        remaining = pv[1: 1 + CONTINUATION_MOVES]
        tmp = board.copy()
        # Apply the first move so continuations are from the opponent's POV
        if pv:
            tmp.push(pv[0])
        sans: list[str] = []
        for move in remaining:
            sans.append(tmp.san(move))
            tmp.push(move)
        return sans
