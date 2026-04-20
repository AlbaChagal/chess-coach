"""HTTP API for the mobile-ready ChessCoach backend flow."""

from __future__ import annotations

import base64
import binascii
from typing import Literal, cast

import chess
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from chesscoach.pipeline import (
    complete_position,
    coaching_result_to_dict,
    run_analysis,
    run_coaching_pipeline,
    run_explanation,
    run_vision,
    serialize_pipeline_value,
)
from chesscoach.pipeline_models import (
    CoachingRequest,
    CompletedPosition,
    ImageClick,
    VisionResult,
)


class ApiImageClick(BaseModel):
    """Raw image click coordinates supplied by the client."""

    x: float
    y: float


class VisionApiRequest(BaseModel):
    """Request body for the vision-only endpoint."""

    image_base64: str
    white_king_start_click: ApiImageClick


class CompletePositionApiRequest(BaseModel):
    """Request body for explicit FEN completion."""

    fen_placement: str
    side_to_move: Literal["w", "b"] | None = None
    white_king_start_click: ApiImageClick
    castling_rights: str | None = None
    en_passant: str | None = None


class AnalyzeApiRequest(BaseModel):
    """Request body for engine analysis."""

    fen: str
    top_n: int = Field(default=3, ge=1)


class ExplainApiRequest(BaseModel):
    """Request body for optional explanation generation."""

    fen: str
    played_move_uci: str | None = None
    explanation_provider: Literal["anthropic", "openai"] | None = None
    explanation_model: str | None = None
    top_n: int = Field(default=3, ge=1)


class CoachApiRequest(BaseModel):
    """Request body for the one-shot coaching endpoint."""

    image_base64: str
    side_to_move: Literal["w", "b"] | None = None
    white_king_start_click: ApiImageClick
    castling_rights: str | None = None
    en_passant: str | None = None
    played_move_uci: str | None = None
    include_explanation: bool = False
    explanation_provider: Literal["anthropic", "openai"] | None = None
    explanation_model: str | None = None
    top_n: int = Field(default=3, ge=1)


def create_app() -> FastAPI:
    """Create the FastAPI application for the ChessCoach backend."""
    app = FastAPI(title="ChessCoach API", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        """Return a simple landing page for browser-based local testing."""
        return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>ChessCoach API</title>
    <style>
      :root {
        color-scheme: light;
        --bg: #f4efe2;
        --panel: #fffaf0;
        --ink: #1f1a17;
        --muted: #5c5146;
        --accent: #8b5e34;
        --accent-2: #d8b98a;
      }
      body {
        margin: 0;
        font-family: Georgia, "Times New Roman", serif;
        background: radial-gradient(circle at top, #fff7e8, var(--bg));
        color: var(--ink);
      }
      main {
        max-width: 720px;
        margin: 48px auto;
        padding: 32px;
        background: var(--panel);
        border: 1px solid var(--accent-2);
        border-radius: 18px;
        box-shadow: 0 18px 40px rgba(72, 51, 30, 0.08);
      }
      h1 {
        margin-top: 0;
        margin-bottom: 12px;
      }
      p, li {
        line-height: 1.55;
        color: var(--muted);
      }
      code {
        background: #f2e6d5;
        padding: 2px 6px;
        border-radius: 6px;
      }
      a {
        color: var(--accent);
      }
    </style>
  </head>
  <body>
    <main>
      <h1>ChessCoach API</h1>
      <p>This is the local backend for the mobile-ready ChessCoach flow.</p>
      <p>Useful endpoints:</p>
      <ul>
        <li><a href="/docs">/docs</a> for the interactive API UI</li>
        <li><a href="/health">/health</a> for a quick health check</li>
        <li><code>POST /vision</code>, <code>POST /complete-position</code>, <code>POST /analyze</code>, <code>POST /explain</code>, and <code>POST /coach</code></li>
      </ul>
      <p>If you opened the base URL expecting a UI, the API docs at <a href="/docs">/docs</a> are the right place to start.</p>
    </main>
  </body>
</html>
"""

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return a simple health signal for local testing."""
        return {"status": "ok"}

    @app.post("/vision")
    def vision(payload: VisionApiRequest) -> dict[str, object]:
        """Run the fast vision path and return piece placement plus warnings."""
        request = CoachingRequest(
            image=_decode_image_base64(payload.image_base64),
            white_king_start_click=_to_image_click(payload.white_king_start_click),
        )
        vision_result, warnings = run_vision(request)
        return {
            "status": "success"
            if vision_result.fen_placement is not None
            else "failed",
            "vision": serialize_pipeline_value(vision_result),
            "warnings": serialize_pipeline_value(warnings),
        }

    @app.post("/complete-position")
    def complete_position_endpoint(
        payload: CompletePositionApiRequest,
    ) -> dict[str, object]:
        """Complete a partial board placement into a full FEN."""
        vision_result = VisionResult(
            fen_placement=payload.fen_placement,
            vision_confidence=None,
            orientation_status="resolved",
            needs_user_confirmation=False,
            white_king_start_click=_to_image_click(payload.white_king_start_click),
        )
        request = CoachingRequest(
            image=b"",
            side_to_move=payload.side_to_move,
            white_king_start_click=_to_image_click(payload.white_king_start_click),
            castling_rights=payload.castling_rights,
            en_passant=payload.en_passant,
        )
        if payload.side_to_move is None:
            return {
                "status": "partial",
                "position": None,
                "user_action_required": "side_to_move",
                "warnings": [],
            }
        try:
            position = complete_position(vision_result, request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "status": "success" if position is not None else "failed",
            "position": serialize_pipeline_value(position),
            "user_action_required": None,
            "warnings": [],
        }

    @app.post("/analyze")
    def analyze(payload: AnalyzeApiRequest) -> dict[str, object]:
        """Analyze a completed FEN without requesting explanation text."""
        position = _completed_position_from_fen(payload.fen)
        analysis = run_analysis(position, top_n=payload.top_n)
        return {
            "status": "success",
            "analysis": serialize_pipeline_value(analysis),
        }

    @app.post("/explain")
    def explain(payload: ExplainApiRequest) -> dict[str, object]:
        """Run analysis and optional explanation as a separate backend step."""
        position = _completed_position_from_fen(payload.fen)
        analysis = run_analysis(position, top_n=payload.top_n)
        explanation, warnings = run_explanation(
            position=position,
            analysis=analysis,
            request=CoachingRequest(
                image=b"",
                side_to_move=_literal_side_to_move(position.side_to_move),
                white_king_start_click=position.white_king_start_click,
                played_move_uci=payload.played_move_uci,
                include_explanation=True,
                explanation_provider=payload.explanation_provider,
                explanation_model=payload.explanation_model,
                top_n=payload.top_n,
            ),
        )
        return {
            "status": explanation.status,
            "analysis": serialize_pipeline_value(analysis),
            "explanation": serialize_pipeline_value(explanation),
            "warnings": serialize_pipeline_value(warnings),
        }

    @app.post("/coach")
    def coach(payload: CoachApiRequest) -> dict[str, object]:
        """Run the full image-to-analysis coaching pipeline."""
        result = run_coaching_pipeline(
            CoachingRequest(
                image=_decode_image_base64(payload.image_base64),
                side_to_move=payload.side_to_move,
                white_king_start_click=_to_image_click(payload.white_king_start_click),
                castling_rights=payload.castling_rights,
                en_passant=payload.en_passant,
                played_move_uci=payload.played_move_uci,
                include_explanation=payload.include_explanation,
                explanation_provider=payload.explanation_provider,
                explanation_model=payload.explanation_model,
                top_n=payload.top_n,
            )
        )
        return coaching_result_to_dict(result)

    return app


def _decode_image_base64(image_base64: str) -> bytes:
    """Decode a base64 image string, including optional data-URL prefixes."""
    encoded = image_base64
    if "," in image_base64 and ";base64" in image_base64:
        encoded = image_base64.split(",", maxsplit=1)[1]
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(
            status_code=400, detail="Invalid image_base64 payload."
        ) from exc


def _to_image_click(click: ApiImageClick) -> ImageClick:
    """Convert an API click model into the pipeline dataclass."""
    return ImageClick(x=click.x, y=click.y)


def _completed_position_from_fen(fen: str) -> CompletedPosition:
    """Parse a full FEN string into the pipeline's completed-position model."""
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not board.is_valid():
        raise HTTPException(
            status_code=422, detail=f"Invalid board position in FEN: {fen!r}"
        )
    return CompletedPosition(
        fen=board.fen(),
        fen_placement=board.board_fen(),
        side_to_move="w" if board.turn == chess.WHITE else "b",
        castling_rights=board.castling_xfen(),
        en_passant="-"
        if board.ep_square is None
        else chess.square_name(board.ep_square),
        source="api",
        user_confirmed_orientation=True,
        white_king_start_click=ImageClick(x=0.0, y=0.0),
    )


def _literal_side_to_move(side_to_move: str) -> Literal["w", "b"]:
    """Narrow a parsed side-to-move value to the pipeline request type."""
    if side_to_move not in {"w", "b"}:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid side_to_move value: {side_to_move!r}",
        )
    return cast(Literal["w", "b"], side_to_move)
