"""HTTP API and browser UI for ChessCoach."""

from __future__ import annotations

import base64
import binascii
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, cast

import chess
import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from chesscoach.auth import (
    DEFAULT_SESSION_COOKIE,
    AuthError,
    AuthStore,
    SavedSnapshotRecord,
    UserRecord,
    UserSettingsRecord,
    create_auth_store,
    decode_session_cookie,
    encode_session_cookie,
)
from chesscoach.pipeline import (
    INVALID_BOARD_POSITION_WARNING,
    complete_position,
    coaching_result_to_dict,
    run_analysis,
    run_coaching_pipeline,
    run_explanation,
    run_vision,
    serialize_pipeline_value,
)
from chesscoach.pipeline_models import (
    PipelineWarning,
    CoachingRequest,
    CompletedPosition,
    ImageClick,
    VisionResult,
)
from chesscoach.vision import BoardNotFoundError
from chesscoach.vision.board_detector import detect_board_corners

TEMPLATES_DIR = Path(__file__).with_name("templates")
STATIC_DIR = Path(__file__).with_name("static")
LOW_CONFIDENCE_WARNING = PipelineWarning(
    code="board_detection_low_confidence",
    message="The board could not be detected. Please try to upload a clearer image.",
)


class ApiImageClick(BaseModel):
    """Raw image click coordinates supplied by the client."""

    x: float
    y: float


class VisionApiRequest(BaseModel):
    """Request body for the vision-only endpoint."""

    image_base64: str
    white_king_start_click: ApiImageClick


class DetectBoardApiRequest(BaseModel):
    """Request body for the staged board-detection endpoint."""

    image_base64: str


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


class SignupRequest(BaseModel):
    """Request body for user signup."""

    email: str
    password: str


class LoginRequest(BaseModel):
    """Request body for user login."""

    email: str
    password: str


class SavedSnapshotApiRequest(BaseModel):
    """Request body for creating a saved analysis snapshot."""

    snapshot: dict[str, object]


class SettingsApiRequest(BaseModel):
    """Request body for synced user settings."""

    show_coordinates: bool


def create_app() -> FastAPI:
    """Create the FastAPI application for the ChessCoach backend and UI."""
    app = FastAPI(title="ChessCoach API", version="0.2.0")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.state.auth_store = _initialize_auth_store()
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    static_asset_version = _static_asset_version()

    @app.get("/", response_class=HTMLResponse)
    def root(request: Request) -> RedirectResponse:
        """Send users to the correct entry point based on auth state."""
        user = _current_user_from_request(request, app.state.auth_store)
        if user is None:
            return RedirectResponse("/login", status_code=302)
        return RedirectResponse("/app/analyze", status_code=302)

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return a simple health signal for local testing."""
        return {"status": "ok"}

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request) -> Response:
        """Render the browser login screen."""
        user = _current_user_from_request(request, app.state.auth_store)
        if user is not None:
            return RedirectResponse("/app/analyze", status_code=302)
        return templates.TemplateResponse(
            request=request,
            name="auth.html",
            context={
                "mode": "login",
                "page_title": "Log In",
                "auth_heading": "Improve Your Chess",
                "auth_subheading": "Sign in to access synced analysis and saved positions.",
                "auth_endpoint": "/auth/login",
                "submit_label": "Log In",
                "switch_href": "/signup",
                "switch_label": "Create account",
                "show_confirm_password": False,
                "static_asset_version": static_asset_version,
            },
        )

    @app.get("/signup", response_class=HTMLResponse)
    def signup_page(request: Request) -> Response:
        """Render the browser signup screen."""
        user = _current_user_from_request(request, app.state.auth_store)
        if user is not None:
            return RedirectResponse("/app/analyze", status_code=302)
        return templates.TemplateResponse(
            request=request,
            name="auth.html",
            context={
                "mode": "signup",
                "page_title": "Create Account",
                "auth_heading": "Create Your Account",
                "auth_subheading": "Set up a synced ChessCoach profile with email and password.",
                "auth_endpoint": "/auth/signup",
                "submit_label": "Sign Up",
                "switch_href": "/login",
                "switch_label": "Log in instead",
                "show_confirm_password": True,
                "static_asset_version": static_asset_version,
            },
        )

    @app.post("/auth/signup")
    def signup(payload: SignupRequest) -> JSONResponse:
        """Create a new auth account and start a session."""
        try:
            user = app.state.auth_store.create_user(payload.email, payload.password)
        except AuthError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        response = JSONResponse(
            {
                "status": "success",
                "user": {"id": user.id, "email": user.email},
                "redirect_to": "/app/analyze",
            }
        )
        _set_session_cookie(response, user)
        return response

    @app.post("/auth/login")
    def login(payload: LoginRequest) -> JSONResponse:
        """Authenticate a user and start a session."""
        try:
            user = app.state.auth_store.authenticate_user(
                payload.email, payload.password
            )
        except AuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        response = JSONResponse(
            {
                "status": "success",
                "user": {"id": user.id, "email": user.email},
                "redirect_to": "/app/analyze",
            }
        )
        _set_session_cookie(response, user)
        return response

    @app.get("/auth/me")
    def auth_me(request: Request) -> JSONResponse:
        """Return the authenticated user for session bootstrap."""
        user = _current_user_from_request(request, app.state.auth_store)
        if user is None:
            raise HTTPException(status_code=401, detail="Not authenticated.")
        settings = app.state.auth_store.get_user_settings(user.id)
        return JSONResponse(
            {
                "status": "success",
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "created_at": user.created_at,
                },
                "settings": _serialize_user_settings(settings),
            }
        )

    @app.post("/auth/logout")
    def logout() -> JSONResponse:
        """Clear the auth session cookie."""
        response = JSONResponse({"status": "success", "redirect_to": "/login"})
        response.delete_cookie(DEFAULT_SESSION_COOKIE, path="/")
        return response

    @app.get("/app", response_class=HTMLResponse)
    def app_root(request: Request) -> RedirectResponse:
        """Redirect the authenticated app root to Analyze."""
        if _current_user_from_request(request, app.state.auth_store) is None:
            return _redirect_to_login_response(request)
        return RedirectResponse("/app/analyze", status_code=302)

    @app.get("/app/analyze", response_class=HTMLResponse)
    def analyze_page(request: Request) -> Response:
        """Render the Analyze destination placeholder."""
        user = _current_user_from_request(request, app.state.auth_store)
        if user is None:
            return _redirect_to_login_response(request)
        return templates.TemplateResponse(
            request=request,
            name="app_shell.html",
            context=_shell_context(
                active_tab="analyze",
                page_title="Analyze",
                heading="Analyze",
                subheading="Load a board image, review analysis, and save synced snapshots.",
                user=user,
                body_variant="analyze",
                static_asset_version=static_asset_version,
            ),
        )

    @app.get("/app/saved", response_class=HTMLResponse)
    def saved_page(request: Request) -> Response:
        """Render the Saved destination placeholder."""
        user = _current_user_from_request(request, app.state.auth_store)
        if user is None:
            return _redirect_to_login_response(request)
        return templates.TemplateResponse(
            request=request,
            name="app_shell.html",
            context=_shell_context(
                active_tab="saved",
                page_title="Saved",
                heading="Saved Positions",
                subheading="Open and manage your synced analysis snapshots.",
                user=user,
                body_variant="saved",
                static_asset_version=static_asset_version,
            ),
        )

    @app.get("/app/profile", response_class=HTMLResponse)
    def profile_page(request: Request) -> Response:
        """Render the Profile destination with logout and settings shell."""
        user = _current_user_from_request(request, app.state.auth_store)
        if user is None:
            return _redirect_to_login_response(request)
        return templates.TemplateResponse(
            request=request,
            name="app_shell.html",
            context=_shell_context(
                active_tab="profile",
                page_title="Profile",
                heading="Profile",
                subheading="Manage your account and app display preferences.",
                user=user,
                body_variant="profile",
                static_asset_version=static_asset_version,
            ),
        )

    @app.get("/api/settings")
    def settings(request: Request) -> JSONResponse:
        """Return synced settings for the authenticated user."""
        user = _require_authenticated_user(request, app.state.auth_store)
        settings_record = app.state.auth_store.get_user_settings(user.id)
        return JSONResponse(
            {
                "status": "success",
                "settings": _serialize_user_settings(settings_record),
            }
        )

    @app.post("/api/settings")
    def update_settings(
        request: Request,
        payload: SettingsApiRequest,
    ) -> JSONResponse:
        """Persist synced settings for the authenticated user."""
        user = _require_authenticated_user(request, app.state.auth_store)
        settings_record = app.state.auth_store.update_user_settings(
            user.id,
            show_coordinates=payload.show_coordinates,
        )
        return JSONResponse(
            {
                "status": "success",
                "settings": _serialize_user_settings(settings_record),
            }
        )

    @app.get("/api/saved")
    def list_saved(request: Request) -> JSONResponse:
        """List saved analysis snapshots for the authenticated user."""
        user = _require_authenticated_user(request, app.state.auth_store)
        snapshots = app.state.auth_store.list_saved_snapshots(user.id)
        return JSONResponse(
            {
                "status": "success",
                "snapshots": serialize_pipeline_value(snapshots),
            }
        )

    @app.post("/api/saved")
    def create_saved(
        request: Request,
        payload: SavedSnapshotApiRequest,
    ) -> JSONResponse:
        """Persist a saved analysis snapshot for the authenticated user."""
        user = _require_authenticated_user(request, app.state.auth_store)
        try:
            snapshot = app.state.auth_store.create_saved_snapshot(
                user.id,
                snapshot=payload.snapshot,
            )
        except AuthError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JSONResponse(
            {
                "status": "success",
                "snapshot": _serialize_saved_snapshot_record(snapshot),
            }
        )

    @app.get("/api/saved/{snapshot_id}")
    def get_saved(snapshot_id: int, request: Request) -> JSONResponse:
        """Return one saved snapshot for the authenticated user."""
        user = _require_authenticated_user(request, app.state.auth_store)
        snapshot = app.state.auth_store.get_saved_snapshot(user.id, snapshot_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Saved snapshot not found.")
        return JSONResponse(
            {
                "status": "success",
                "snapshot": _serialize_saved_snapshot_record(snapshot),
            }
        )

    @app.delete("/api/saved/{snapshot_id}")
    def delete_saved(snapshot_id: int, request: Request) -> JSONResponse:
        """Delete one saved snapshot for the authenticated user."""
        user = _require_authenticated_user(request, app.state.auth_store)
        deleted = app.state.auth_store.delete_saved_snapshot(user.id, snapshot_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Saved snapshot not found.")
        return JSONResponse({"status": "success", "deleted": True})

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

    @app.post("/detect-board")
    def detect_board(payload: DetectBoardApiRequest) -> dict[str, object]:
        """Detect board geometry for the staged Analyze UI."""
        bgr = _decode_image_base64_to_bgr(payload.image_base64)
        height, width = bgr.shape[:2]
        try:
            board_corners = detect_board_corners(bgr)
        except (BoardNotFoundError, ValueError):
            return {
                "status": "failed",
                "detection": {
                    "board_corners": None,
                    "confidence": 0.0,
                    "image_width": width,
                    "image_height": height,
                },
                "warnings": serialize_pipeline_value([LOW_CONFIDENCE_WARNING]),
            }
        return {
            "status": "success",
            "detection": {
                "board_corners": serialize_pipeline_value(board_corners.tolist()),
                "confidence": 1.0,
                "image_width": width,
                "image_height": height,
            },
            "warnings": [],
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
        except ValueError:
            return {
                "status": "failed",
                "position": None,
                "user_action_required": None,
                "warnings": serialize_pipeline_value(
                    [INVALID_BOARD_POSITION_WARNING]
                ),
            }
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


def _initialize_auth_store() -> AuthStore:
    store = create_auth_store()
    store.initialize()
    return store


def _set_session_cookie(response: JSONResponse, user: UserRecord) -> None:
    response.set_cookie(
        key=DEFAULT_SESSION_COOKIE,
        value=encode_session_cookie(user),
        httponly=True,
        samesite="lax",
        path="/",
    )


def _current_user_from_request(
    request: Request,
    store: AuthStore,
) -> UserRecord | None:
    cookie_value = request.cookies.get(DEFAULT_SESSION_COOKIE)
    if cookie_value is None:
        return None
    user_id = decode_session_cookie(cookie_value)
    if user_id is None:
        return None
    return store.get_user_by_id(user_id)


def _require_authenticated_user(
    request: Request,
    store: AuthStore,
) -> UserRecord:
    user = _current_user_from_request(request, store)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return user


def _shell_context(
    *,
    active_tab: str,
    page_title: str,
    heading: str,
    subheading: str,
    user: UserRecord,
    body_variant: str,
    static_asset_version: str,
) -> dict[str, Any]:
    return {
        "page_title": page_title,
        "active_tab": active_tab,
        "heading": heading,
        "subheading": subheading,
        "user": user,
        "body_variant": body_variant,
        "static_asset_version": static_asset_version,
    }


def _static_asset_version() -> str:
    """Return a stable cache-busting version for browser UI assets."""
    asset_paths = [
        STATIC_DIR / "app.css",
        STATIC_DIR / "app.js",
    ]
    latest_mtime_ns = max(path.stat().st_mtime_ns for path in asset_paths)
    return str(latest_mtime_ns)


def _serialize_user_settings(settings: UserSettingsRecord) -> dict[str, object]:
    return {
        "show_coordinates": settings.show_coordinates,
    }


def _serialize_saved_snapshot_record(
    snapshot: SavedSnapshotRecord,
) -> dict[str, object]:
    return {
        "id": snapshot.id,
        "created_at": snapshot.created_at,
        "updated_at": snapshot.updated_at,
        "summary": serialize_pipeline_value(snapshot.summary),
        "snapshot": snapshot.snapshot,
    }


def _redirect_to_login_response(request: Request) -> RedirectResponse:
    next_path = request.url.path
    return RedirectResponse(f"/login?next={next_path}", status_code=302)


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


def _decode_image_base64_to_bgr(image_base64: str) -> np.ndarray:
    """Decode a base64 image into an OpenCV BGR array."""
    image_bytes = _decode_image_base64(image_base64)
    try:
        with PILImage.open(BytesIO(image_bytes)) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid image_base64 payload.")
    rgb = np.array(normalized)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


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
