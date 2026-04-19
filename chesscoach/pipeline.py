"""End-to-end orchestration for vision, analysis, and optional explanation."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from typing import cast

import chess

from chesscoach.analysis.coach import ChessCoach
from chesscoach.analysis.engine import ChessEngine
from chesscoach.analysis.models import MoveAnalysis
from chesscoach.explanation import (
    ClaudeProvider,
    Explainer,
    LLMProvider,
    OpenAIProvider,
)
from chesscoach.explanation.models import ExplanationError
from chesscoach.pipeline_models import (
    AnalysisResult,
    CoachingRequest,
    CoachingResult,
    CompletedPosition,
    ExplanationResult,
    PipelineWarning,
    VisionResult,
)
from chesscoach.vision import BoardNotFoundError, predict_fen

LOW_CONFIDENCE_WARNING = PipelineWarning(
    code="board_detection_low_confidence",
    message=("The board could not be detected. Please try to upload a clearer image."),
)
EXPLANATION_UNAVAILABLE_WARNING = PipelineWarning(
    code="explanation_skipped_unavailable",
    message="Explanation was skipped because no LLM provider is configured.",
)
EXPLANATION_FAILED_WARNING_CODE = "explanation_failed"
VISION_CONFIDENCE_SUCCESS = 1.0
VISION_CONFIDENCE_FAILURE = 0.0
DEFAULT_EN_PASSANT = "-"
DEFAULT_HALFMOVE_CLOCK = 0
DEFAULT_FULLMOVE_NUMBER = 1


def run_vision(request: CoachingRequest) -> tuple[VisionResult, list[PipelineWarning]]:
    """Run the vision layer and return a typed result plus any warnings."""
    if request.white_king_start_click is None:
        return (
            VisionResult(
                fen_placement=None,
                vision_confidence=None,
                orientation_status="missing_click",
                needs_user_confirmation=True,
                white_king_start_click=None,
            ),
            [],
        )

    try:
        fen_placement = predict_fen(_coerce_image_input(request.image))
    except (BoardNotFoundError, ValueError):
        return (
            VisionResult(
                fen_placement=None,
                vision_confidence=VISION_CONFIDENCE_FAILURE,
                orientation_status="failed",
                needs_user_confirmation=False,
                white_king_start_click=request.white_king_start_click,
            ),
            [LOW_CONFIDENCE_WARNING],
        )

    return (
        VisionResult(
            fen_placement=fen_placement,
            vision_confidence=VISION_CONFIDENCE_SUCCESS,
            orientation_status="user_marked",
            needs_user_confirmation=False,
            white_king_start_click=request.white_king_start_click,
        ),
        [],
    )


def complete_position(
    vision_result: VisionResult,
    request: CoachingRequest,
) -> CompletedPosition | None:
    """Build a complete FEN string from the vision result and user inputs."""
    if vision_result.fen_placement is None:
        return None
    if request.white_king_start_click is None:
        return None
    if request.side_to_move is None:
        return None

    castling_rights, source = _resolve_castling_rights(
        vision_result.fen_placement,
        request.side_to_move,
        request.castling_rights,
    )
    en_passant = request.en_passant or DEFAULT_EN_PASSANT
    fen = (
        f"{vision_result.fen_placement} {request.side_to_move} {castling_rights} "
        f"{en_passant} {DEFAULT_HALFMOVE_CLOCK} {DEFAULT_FULLMOVE_NUMBER}"
    )
    _validate_completed_fen(fen)

    return CompletedPosition(
        fen=fen,
        fen_placement=vision_result.fen_placement,
        side_to_move=request.side_to_move,
        castling_rights=castling_rights,
        en_passant=en_passant,
        source=source,
        user_confirmed_orientation=True,
        white_king_start_click=request.white_king_start_click,
    )


def run_analysis(
    position: CompletedPosition,
    top_n: int = 3,
) -> AnalysisResult:
    """Analyze a completed position with Stockfish."""
    start = time.perf_counter()
    engine = ChessEngine()
    coach = ChessCoach(engine)
    try:
        top_moves = coach.analyze_position(position.fen, n=top_n)
    finally:
        engine.close()
    latency_ms = (time.perf_counter() - start) * 1000
    engine_depth = max((move.depth for move in top_moves), default=None)
    return AnalysisResult(
        fen=position.fen,
        top_moves=top_moves,
        engine_depth=engine_depth,
        analysis_latency_ms=latency_ms,
        analysis_status="success",
    )


def run_explanation(
    position: CompletedPosition,
    analysis: AnalysisResult,
    request: CoachingRequest,
) -> tuple[ExplanationResult, list[PipelineWarning]]:
    """Explain the engine's top move when configured and requested."""
    if not request.include_explanation:
        return (
            ExplanationResult(
                move_uci=None,
                move_san=None,
                explanation_text=None,
                structured_explanation=None,
                provider=None,
                status="skipped",
            ),
            [],
        )
    if not analysis.top_moves:
        return (
            ExplanationResult(
                move_uci=None,
                move_san=None,
                explanation_text=None,
                structured_explanation=None,
                provider=None,
                status="skipped",
            ),
            [
                PipelineWarning(
                    code="explanation_skipped_no_moves",
                    message="Explanation was skipped because analysis returned no moves.",
                )
            ],
        )

    provider_name, provider = _pick_explanation_provider()
    best_move = analysis.top_moves[0]
    if provider is None:
        return (
            ExplanationResult(
                move_uci=best_move.move_uci,
                move_san=best_move.move_san,
                explanation_text=None,
                structured_explanation=None,
                provider=None,
                status="skipped",
            ),
            [EXPLANATION_UNAVAILABLE_WARNING],
        )

    engine = ChessEngine()
    try:
        explainer = Explainer(engine, provider, top_n=request.top_n)
        explanation_text = explainer.explain_move(position.fen, best_move.move_uci)
    except (ExplanationError, ValueError) as exc:
        return (
            ExplanationResult(
                move_uci=best_move.move_uci,
                move_san=best_move.move_san,
                explanation_text=None,
                structured_explanation=None,
                provider=provider_name,
                status="failed",
            ),
            [
                PipelineWarning(
                    code=EXPLANATION_FAILED_WARNING_CODE,
                    message=f"Explanation failed: {exc}",
                )
            ],
        )
    finally:
        engine.close()

    return (
        ExplanationResult(
            move_uci=best_move.move_uci,
            move_san=best_move.move_san,
            explanation_text=explanation_text,
            structured_explanation=None,
            provider=provider_name,
            status="success",
        ),
        [],
    )


def run_coaching_pipeline(request: CoachingRequest) -> CoachingResult:
    """Run the full Milestone 1 coaching pipeline."""
    if request.white_king_start_click is None:
        return CoachingResult(
            vision=VisionResult(
                fen_placement=None,
                vision_confidence=None,
                orientation_status="missing_click",
                needs_user_confirmation=True,
                white_king_start_click=None,
            ),
            position=None,
            analysis=None,
            explanation=None,
            status="partial",
            user_action_required="white_king_start_click",
        )

    vision_result, warnings = run_vision(request)
    if (
        vision_result.fen_placement is None
        or vision_result.vision_confidence != VISION_CONFIDENCE_SUCCESS
    ):
        return CoachingResult(
            vision=vision_result,
            position=None,
            analysis=None,
            explanation=None,
            status="failed",
            user_action_required=None,
            warnings=warnings,
        )

    position = complete_position(vision_result, request)
    if request.side_to_move is None:
        return CoachingResult(
            vision=vision_result,
            position=None,
            analysis=None,
            explanation=None,
            status="partial",
            user_action_required="side_to_move",
            warnings=warnings,
        )
    if position is None:
        return CoachingResult(
            vision=vision_result,
            position=None,
            analysis=None,
            explanation=None,
            status="failed",
            user_action_required=None,
            warnings=warnings,
        )

    analysis = run_analysis(position, top_n=request.top_n)
    explanation, explanation_warnings = run_explanation(position, analysis, request)
    return CoachingResult(
        vision=vision_result,
        position=position,
        analysis=analysis,
        explanation=explanation,
        status="success",
        user_action_required=None,
        warnings=[*warnings, *explanation_warnings],
    )


def coaching_result_to_dict(result: CoachingResult) -> dict[str, Any]:
    """Convert a coaching result into a JSON-serializable dictionary."""
    payload = _serialize_dataclass(result)
    analysis = payload.get("analysis")
    if analysis is not None:
        top_moves = analysis.get("top_moves", [])
        analysis["top_moves"] = [
            {**move, "score_display": _score_display_from_move(move)}
            for move in top_moves
        ]
    return payload


def _coerce_image_input(image: Path | bytes) -> Path | bytes:
    if isinstance(image, Path):
        return image
    return image


def _resolve_castling_rights(
    fen_placement: str,
    side_to_move: str,
    explicit_castling_rights: str | None,
) -> tuple[str, str]:
    if explicit_castling_rights is not None:
        board = chess.Board(
            f"{fen_placement} {side_to_move} {explicit_castling_rights} - 0 1"
        )
        if not board.is_valid():
            raise ValueError(f"Invalid board position in FEN: {fen_placement!r}")
        return explicit_castling_rights, "user"

    inferred = _infer_castling_rights_from_placement(fen_placement)
    return inferred, "heuristic"


def _infer_castling_rights_from_placement(fen_placement: str) -> str:
    board = chess.Board(f"{fen_placement} w - - 0 1")
    rights: list[str] = []
    if board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE):
        if board.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE):
            rights.append("K")
        if board.piece_at(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE):
            rights.append("Q")
    if board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK):
        if board.piece_at(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK):
            rights.append("k")
        if board.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK):
            rights.append("q")
    return "".join(rights) or "-"


def _validate_completed_fen(fen: str) -> None:
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError(f"Invalid board position in FEN: {fen!r}")


def _pick_explanation_provider() -> tuple[str | None, LLMProvider | None]:
    configured: list[tuple[str, LLMProvider]] = []
    if os.getenv("ANTHROPIC_API_KEY"):
        configured.append(("anthropic", ClaudeProvider()))
    if os.getenv("OPENAI_API_KEY"):
        configured.append(("openai", OpenAIProvider()))
    if len(configured) != 1:
        return None, None
    return configured[0]


def _serialize_dataclass(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, MoveAnalysis):
        payload = asdict(value)
        payload["score_display"] = value.score_display()
        return payload
    if is_dataclass(value):
        dataclass_value = cast(Any, value)
        return {
            field_name: _serialize_dataclass(field_value)
            for field_name, field_value in asdict(dataclass_value).items()
        }
    if isinstance(value, list):
        return [_serialize_dataclass(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_dataclass(item) for key, item in value.items()}
    return value


def _score_display_from_move(move: dict[str, Any]) -> str:
    analysis = MoveAnalysis(
        move_san=move["move_san"],
        move_uci=move["move_uci"],
        score_cp=move["score_cp"],
        score_mate=move["score_mate"],
        depth=move["depth"],
        continuation=move["continuation"],
    )
    return analysis.score_display()
