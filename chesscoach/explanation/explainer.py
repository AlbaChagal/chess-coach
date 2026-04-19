"""Main explanation pipeline: analysis → classification → tactics → LLM narration."""

from __future__ import annotations

import logging

import chess

from chesscoach.analysis.engine import ChessEngine
from chesscoach.analysis.models import MoveAnalysis
from chesscoach.explanation.classifier import classify_move
from chesscoach.explanation.models import (
    AlternativeExplanation,
    ExplainedMove,
    ExplanationError,
    StructuredExplanation,
    TacticInfo,
)
from chesscoach.explanation.prompt import build_prompt
from chesscoach.explanation.providers import LLMProvider
from chesscoach.explanation.tactics import detect_tactics

LOGGER = logging.getLogger(__name__)


class Explainer:
    """Full explanation pipeline for a chess move.

    Combines Stockfish analysis, rule-based move classification, tactic
    detection, and LLM narration into a single coherent coaching response.

    Args:
        engine: An open :class:`~chesscoach.analysis.engine.ChessEngine` instance.
        provider: LLM backend — any :class:`~chesscoach.explanation.providers.LLMProvider`.
        top_n: Number of engine alternatives to fetch (default: 3).

    Example::

        with ChessEngine() as engine:
            explainer = Explainer(engine, ClaudeProvider())
            text = explainer.explain_move(fen_before, move_uci)
    """

    def __init__(
        self,
        engine: ChessEngine,
        provider: LLMProvider | None,
        top_n: int = 3,
    ) -> None:
        self._engine = engine
        self._provider = provider
        self._top_n = top_n

    def analyze_position(self, fen_before: str) -> ExplainedMove:
        """Analyze the engine's best move for the given position."""
        board = self._parse_fen(fen_before)
        engine_moves = self._engine.get_best_moves(board, self._top_n)
        if not engine_moves:
            raise ValueError("Engine returned no moves for this position.")

        best_move = engine_moves[0]
        return self.analyze_move(fen_before, best_move.move_uci)

    def build_structured_explanation(
        self,
        explained: ExplainedMove,
    ) -> StructuredExplanation:
        """Convert engine and tactic output into a typed explanation payload."""
        best_move = explained.best_move
        tactical_themes = _collect_tactical_themes(explained.tactics_after_best)
        summary = _build_summary(best_move, explained.tactics_after_best)
        what_the_move_does = _build_what_the_move_does(best_move)
        what_it_threatens = _build_what_it_threatens(explained.tactics_after_best)
        why_it_is_best = _build_why_it_is_best(best_move, explained.alternatives)
        alternatives = _build_alternative_explanations(
            best_move,
            explained.alternatives,
        )
        why_alternatives_are_worse = _build_why_alternatives_are_worse(
            alternatives,
        )
        return StructuredExplanation(
            summary=summary,
            what_the_move_does=what_the_move_does,
            what_it_threatens=what_it_threatens,
            why_it_is_best=why_it_is_best,
            why_alternatives_are_worse=why_alternatives_are_worse,
            alternatives=alternatives,
            tactical_themes=tactical_themes,
        )

    def narrate_explanation(
        self,
        explained: ExplainedMove,
        structured: StructuredExplanation,
    ) -> str:
        """Turn a structured explanation into final coaching text via the LLM."""
        if self._provider is None:
            raise ExplanationError("No explanation provider configured.")
        system, user = build_prompt(explained, structured)
        LOGGER.debug("Calling LLM provider for explanation")
        return self._provider.complete(system, user)

    def analyze_move(self, fen_before: str, move_uci: str) -> ExplainedMove:
        """Run the structural analysis pipeline without calling the LLM.

        This is fast and free — useful when you only need the classification
        and tactics data (e.g. for display without a coaching explanation).

        Args:
            fen_before: FEN string of the position *before* the move.
            move_uci: The move played in UCI format (e.g. ``"e2e4"``).

        Returns:
            A fully populated :class:`ExplainedMove`.

        Raises:
            ValueError: If the FEN is invalid or the move is illegal.
        """
        board = self._parse_fen(fen_before)
        move = self._parse_move(board, move_uci)
        move_san = board.san(move)

        LOGGER.info(
            "Analyzing move=%s fen=%s",
            move_san,
            fen_before[:40] + "...",
        )

        # 1. Engine analysis of the position before the move.
        engine_moves = self._engine.get_best_moves(board, self._top_n)
        if not engine_moves:
            raise ValueError("Engine returned no moves for this position.")
        best_move = engine_moves[0]
        alternatives = engine_moves[1:]

        # 2. Evaluate the position after the played move to get played_cp.
        played_cp, played_mate = self._eval_after(board, move)

        # 3. Classify the move quality.
        quality = classify_move(
            played_cp=played_cp,
            best_cp=best_move.score_cp,
            played_mate=played_mate,
            best_mate=best_move.score_mate,
        )
        LOGGER.debug("Move quality: label=%s cp_loss=%s", quality.label, quality.cp_loss)

        # 4. Detect tactics after the played move (what opponent can do).
        tactics_after_played = detect_tactics(board, move)

        # 5. Detect tactics after the best move (what we could have gained).
        try:
            best_chess_move = chess.Move.from_uci(best_move.move_uci)
            tactics_after_best = detect_tactics(board, best_chess_move)
        except (ValueError, chess.InvalidMoveError):
            tactics_after_best = []

        return ExplainedMove(
            fen_before=fen_before,
            move_played_san=move_san,
            move_played_uci=move_uci,
            quality=quality,
            best_move=best_move,
            alternatives=alternatives,
            tactics_after_played=tactics_after_played,
            tactics_after_best=tactics_after_best,
        )

    def explain_move(self, fen_before: str, move_uci: str) -> str:
        """Full pipeline: analyze → build prompt → LLM → coaching text.

        Args:
            fen_before: FEN string of the position before the move.
            move_uci: The move played in UCI format.

        Returns:
            A coaching explanation as a plain string.

        Raises:
            ValueError: If the FEN is invalid or the move is illegal.
            ExplanationError: If the LLM provider call fails.
        """
        explained = self.analyze_move(fen_before, move_uci)
        structured = self.build_structured_explanation(explained)
        return self.narrate_explanation(explained, structured)

    def explain(self, fen: str, moves: list[MoveAnalysis]) -> str:
        """Explain the top engine move for a position (legacy interface).

        Implements the ``PositionExplainer.explain`` contract. Uses the best
        move from *moves* as the "played" move so the LLM describes why it's
        the correct choice.

        Args:
            fen: FEN string of the current position.
            moves: Engine move suggestions (first entry is used as the best move).

        Returns:
            A coaching explanation as a plain string.

        Raises:
            ValueError: If *moves* is empty or the FEN is invalid.
            ExplanationError: If the LLM provider call fails.
        """
        if not moves:
            raise ValueError("No moves provided to explain.")
        return self.explain_move(fen, moves[0].move_uci)

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _parse_fen(fen: str) -> chess.Board:
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise ValueError(f"Invalid FEN: {fen!r}") from exc
        if not board.is_valid():
            raise ValueError(f"Invalid board position in FEN: {fen!r}")
        return board

    @staticmethod
    def _parse_move(board: chess.Board, move_uci: str) -> chess.Move:
        try:
            move = chess.Move.from_uci(move_uci)
        except (ValueError, chess.InvalidMoveError) as exc:
            raise ValueError(f"Invalid UCI move: {move_uci!r}") from exc
        if move not in board.legal_moves:
            raise ValueError(
                f"Illegal move {move_uci!r} in position {board.fen()!r}"
            )
        return move

    def _eval_after(
        self,
        board: chess.Board,
        move: chess.Move,
    ) -> tuple[int | None, int | None]:
        """Return (score_cp, score_mate) after *move* is played, from the mover's POV."""
        board_after = board.copy()
        board_after.push(move)
        # Terminal positions need special handling — the engine has no moves to analyse.
        if board_after.is_checkmate():
            return None, 1  # mover delivered mate in 1
        analyses = self._engine.get_best_moves(board_after, 1)
        if not analyses:
            return None, None
        # score is from the *next* player's POV — flip to get mover's POV.
        a = analyses[0]
        if a.score_mate is not None:
            return None, -a.score_mate
        if a.score_cp is not None:
            return -a.score_cp, None
        return None, None


def _collect_tactical_themes(tactics: list[TacticInfo]) -> list[str]:
    names = {tactic.name.replace("_", " ") for tactic in tactics}
    return sorted(names)


def _build_summary(best_move: MoveAnalysis, tactics: list[TacticInfo]) -> str:
    if tactics:
        return (
            f"{best_move.move_san} is best because it immediately creates "
            f"{tactics[0].description.lower()}."
        )
    return (
        f"{best_move.move_san} is best because it keeps the strongest evaluation "
        f"and the cleanest continuation."
    )


def _build_what_the_move_does(best_move: MoveAnalysis) -> str:
    if best_move.continuation:
        line = " ".join(best_move.continuation)
        return (
            f"It improves the position while preserving the best continuation: "
            f"{line}."
        )
    return "It improves coordination and keeps the position under control."


def _build_what_it_threatens(tactics: list[TacticInfo]) -> str:
    if tactics:
        return "; ".join(tactic.description for tactic in tactics)
    return "It increases pressure and keeps stronger practical options available."


def _build_why_it_is_best(
    best_move: MoveAnalysis,
    alternatives: list[MoveAnalysis],
) -> str:
    if not alternatives:
        return (
            f"It keeps the engine evaluation at {best_move.score_display()} with no "
            f"equally strong alternative available."
        )
    alternative = alternatives[0]
    gap = _score_gap_text(best_move, alternative)
    return (
        f"It holds {best_move.score_display()}, which is {gap} than "
        f"{alternative.move_san}."
    )


def _build_alternative_explanations(
    best_move: MoveAnalysis,
    alternatives: list[MoveAnalysis],
) -> list[AlternativeExplanation]:
    items: list[AlternativeExplanation] = []
    for alternative in alternatives:
        gap = _score_gap_text(best_move, alternative)
        items.append(
            AlternativeExplanation(
                move_san=alternative.move_san,
                move_uci=alternative.move_uci,
                score_display=alternative.score_display(),
                reason=(
                    f"It is {gap} than {best_move.move_san} and leads to a weaker "
                    f"continuation."
                ),
            )
        )
    return items


def _build_why_alternatives_are_worse(
    alternatives: list[AlternativeExplanation],
) -> str:
    if not alternatives:
        return "No other move reaches the same level of engine confidence here."
    first = alternatives[0]
    return (
        f"{first.move_san} is the closest alternative, but {first.reason.lower()}"
    )


def _score_gap_text(best_move: MoveAnalysis, alternative: MoveAnalysis) -> str:
    if best_move.score_cp is not None and alternative.score_cp is not None:
        gap_cp = abs(best_move.score_cp - alternative.score_cp)
        return f"about {gap_cp / 100:.2f} pawns better"
    if best_move.score_mate is not None and alternative.score_mate is not None:
        if best_move.score_mate == alternative.score_mate:
            return "roughly equal in mating terms"
        return "stronger in mating terms"
    return "stronger"
