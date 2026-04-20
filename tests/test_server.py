from __future__ import annotations

import base64

import cv2
import pytest
from fastapi.testclient import TestClient

from chesscoach.analysis.models import MoveAnalysis
from chesscoach.explanation.models import (
    BestMoveComparison,
    PlayedMoveResult,
    StructuredPlayedMoveExplanation,
)
from chesscoach.pipeline_models import (
    AnalysisResult,
    CoachingResult,
    CompletedPosition,
    ExplanationResult,
    ImageClick,
    PipelineWarning,
    VisionResult,
)
from chesscoach.server import create_app
from tests.vision.conftest import make_synthetic_board

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
STARTING_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("CHESSCOACH_AUTH_DB", str(tmp_path / "auth.db"))
    monkeypatch.setenv("CHESSCOACH_SESSION_SECRET", "test-session-secret")
    return TestClient(create_app())


def _image_payload() -> str:
    return base64.b64encode(b"fake-image-bytes").decode("ascii")


def _board_payload(board) -> str:
    success, encoded = cv2.imencode(".png", board)
    assert success
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _analysis_result() -> AnalysisResult:
    return AnalysisResult(
        fen=STARTING_FEN,
        top_moves=[
            MoveAnalysis("e4", "e2e4", 35, None, 20, ["e5", "Nf3"]),
            MoveAnalysis("d4", "d2d4", 25, None, 20, ["d5"]),
        ],
        engine_depth=20,
        analysis_latency_ms=12.0,
        analysis_status="success",
    )


def _position() -> CompletedPosition:
    return CompletedPosition(
        fen=STARTING_FEN,
        fen_placement=STARTING_PLACEMENT,
        side_to_move="w",
        castling_rights="KQkq",
        en_passant="-",
        source="heuristic",
        user_confirmed_orientation=True,
        white_king_start_click=ImageClick(x=10.0, y=20.0),
    )


def _signup(client: TestClient, email: str = "user@example.com") -> None:
    response = client.post(
        "/auth/signup",
        json={"email": email, "password": "strongpass"},
    )
    assert response.status_code == 200


def test_root_redirects_to_login_when_logged_out(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"] == "/login"


def test_login_page_renders_auth_ui(client: TestClient) -> None:
    response = client.get("/login")

    assert response.status_code == 200
    assert "Improve Your Chess" in response.text
    assert "data-auth-form" in response.text


def test_signup_creates_user_and_session(client: TestClient) -> None:
    response = client.post(
        "/auth/signup",
        json={"email": "user@example.com", "password": "strongpass"},
    )

    assert response.status_code == 200
    assert response.json()["user"]["email"] == "user@example.com"
    assert "chesscoach_session" in response.cookies


def test_signup_rejects_duplicate_email(client: TestClient) -> None:
    _signup(client)

    response = client.post(
        "/auth/signup",
        json={"email": "user@example.com", "password": "strongpass"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "An account with this email already exists."


def test_login_rejects_invalid_credentials(client: TestClient) -> None:
    _signup(client)
    client.post("/auth/logout")

    response = client.post(
        "/auth/login",
        json={"email": "user@example.com", "password": "wrongpass"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password."


def test_auth_me_requires_session(client: TestClient) -> None:
    response = client.get("/auth/me")

    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated."


def test_auth_me_returns_authenticated_user(client: TestClient) -> None:
    _signup(client)

    response = client.get("/auth/me")

    assert response.status_code == 200
    assert response.json()["user"]["email"] == "user@example.com"


def test_session_persists_across_requests(client: TestClient) -> None:
    _signup(client)

    first = client.get("/app/analyze")
    second = client.get("/auth/me")

    assert first.status_code == 200
    assert "Load a Board Image" in first.text
    assert second.status_code == 200
    assert second.json()["user"]["email"] == "user@example.com"


def test_protected_route_redirects_to_login(client: TestClient) -> None:
    response = client.get("/app/profile", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"] == "/login?next=/app/profile"


def test_authenticated_root_redirects_into_app(client: TestClient) -> None:
    _signup(client)

    response = client.get("/", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"] == "/app/analyze"


def test_profile_page_renders_user_and_logout(client: TestClient) -> None:
    _signup(client)

    response = client.get("/app/profile")

    assert response.status_code == 200
    assert "user@example.com" in response.text
    assert "Log Out" in response.text
    assert "Display Settings" in response.text


def test_logout_clears_session(client: TestClient) -> None:
    _signup(client)

    logout = client.post("/auth/logout")
    redirected = client.get("/app/analyze", follow_redirects=False)

    assert logout.status_code == 200
    assert redirected.status_code == 302
    assert redirected.headers["location"] == "/login?next=/app/analyze"


def test_saved_page_placeholder_renders(client: TestClient) -> None:
    _signup(client)

    response = client.get("/app/saved")

    assert response.status_code == 200
    assert "Saved Snapshots Coming Next" in response.text


def test_analyze_page_renders_phase_two_flow_shell(client: TestClient) -> None:
    _signup(client)

    response = client.get("/app/analyze")

    assert response.status_code == 200
    assert "Load a Board Image" in response.text
    assert "Where did the white king start the game?" in response.text
    assert 'data-detect-endpoint="/detect-board"' in response.text


def test_detect_board_endpoint_returns_corners_for_detectable_board(
    client: TestClient,
) -> None:
    response = client.post(
        "/detect-board",
        json={"image_base64": _board_payload(make_synthetic_board())},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert len(payload["detection"]["board_corners"]) == 4
    assert payload["detection"]["confidence"] == 1.0


def test_detect_board_endpoint_returns_warning_on_failure(client: TestClient) -> None:
    blank = _board_payload(make_synthetic_board() * 0)

    response = client.post("/detect-board", json={"image_base64": blank})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "failed"
    assert payload["detection"]["board_corners"] is None
    assert payload["warnings"][0]["code"] == "board_detection_low_confidence"


def test_vision_endpoint_decodes_image_and_returns_result(
    client: TestClient, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def _run_vision(request):
        captured["image"] = request.image
        captured["click"] = request.white_king_start_click
        return (
            VisionResult(
                fen_placement=STARTING_PLACEMENT,
                vision_confidence=1.0,
                orientation_status="user_marked",
                needs_user_confirmation=False,
                white_king_start_click=request.white_king_start_click,
            ),
            [],
        )

    monkeypatch.setattr("chesscoach.server.run_vision", _run_vision)

    response = client.post(
        "/vision",
        json={
            "image_base64": _image_payload(),
            "white_king_start_click": {"x": 12.0, "y": 34.0},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["vision"]["fen_placement"] == STARTING_PLACEMENT
    assert captured["image"] == b"fake-image-bytes"
    assert captured["click"] == ImageClick(x=12.0, y=34.0)


def test_vision_endpoint_rejects_invalid_base64(client: TestClient) -> None:
    response = client.post(
        "/vision",
        json={
            "image_base64": "not-valid-base64",
            "white_king_start_click": {"x": 1.0, "y": 2.0},
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image_base64 payload."


def test_complete_position_endpoint_returns_partial_when_side_missing(
    client: TestClient,
) -> None:
    response = client.post(
        "/complete-position",
        json={
            "fen_placement": STARTING_PLACEMENT,
            "side_to_move": None,
            "white_king_start_click": {"x": 1.0, "y": 2.0},
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "partial"
    assert response.json()["user_action_required"] == "side_to_move"


def test_complete_position_endpoint_returns_completed_fen(client: TestClient) -> None:
    response = client.post(
        "/complete-position",
        json={
            "fen_placement": STARTING_PLACEMENT,
            "side_to_move": "w",
            "white_king_start_click": {"x": 1.0, "y": 2.0},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["position"]["fen"] == STARTING_FEN


def test_analyze_endpoint_returns_score_display(
    client: TestClient, monkeypatch
) -> None:
    monkeypatch.setattr(
        "chesscoach.server.run_analysis", lambda position, top_n: _analysis_result()
    )

    response = client.post("/analyze", json={"fen": STARTING_FEN, "top_n": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis"]["top_moves"][0]["score_display"] == "+0.35"


def test_explain_endpoint_returns_analysis_and_explanation(
    client: TestClient, monkeypatch
) -> None:
    monkeypatch.setattr(
        "chesscoach.server.run_analysis", lambda position, top_n: _analysis_result()
    )

    def _run_explanation(position, analysis, request):
        assert request.include_explanation is True
        assert request.played_move_uci == "d2d4"
        return (
            ExplanationResult(
                move_uci="d2d4",
                move_san="d4",
                explanation_text=None,
                structured_explanation=StructuredPlayedMoveExplanation(
                    summary="e4 is still cleaner.",
                    what_the_move_tried_to_do="It tries to claim central space.",
                    what_was_missed="It misses the cleaner central option.",
                    what_changed_after_move="Black gets a simpler reply.",
                    why_best_move_was_better="It keeps the best evaluation.",
                    practical_lesson="Compare candidate moves for activity.",
                    alternatives=[],
                    tactical_themes=[],
                ),
                played_move_result=PlayedMoveResult(
                    move_uci="d2d4",
                    move_san="d4",
                    quality_label="inaccuracy",
                    quality_emoji="?!",
                    cp_loss=15,
                    tactics_after_played=[],
                    tactics_after_best=[],
                ),
                comparison=BestMoveComparison(
                    best_move_uci="e2e4",
                    best_move_san="e4",
                    best_move_score_display="+0.35",
                    played_move_uci="d2d4",
                    played_move_san="d4",
                    played_move_quality="inaccuracy",
                    cp_loss=15,
                    why_best_move_is_better="It keeps more space.",
                ),
                provider=None,
                status="success",
            ),
            [
                PipelineWarning(
                    code="explanation_skipped_unavailable",
                    message=(
                        "Explanation was skipped because no LLM provider is configured."
                    ),
                )
            ],
        )

    monkeypatch.setattr("chesscoach.server.run_explanation", _run_explanation)

    response = client.post(
        "/explain",
        json={"fen": STARTING_FEN, "played_move_uci": "d2d4"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["analysis"]["top_moves"][0]["score_display"] == "+0.35"
    assert payload["explanation"]["played_move_result"]["quality_label"] == (
        "inaccuracy"
    )
    assert payload["warnings"][0]["code"] == "explanation_skipped_unavailable"


def test_coach_endpoint_returns_full_pipeline_payload(
    client: TestClient, monkeypatch
) -> None:
    def _run_pipeline(request):
        assert request.image == b"fake-image-bytes"
        return CoachingResult(
            vision=VisionResult(
                fen_placement=STARTING_PLACEMENT,
                vision_confidence=1.0,
                orientation_status="user_marked",
                needs_user_confirmation=False,
                white_king_start_click=ImageClick(x=1.0, y=2.0),
            ),
            position=_position(),
            analysis=_analysis_result(),
            explanation=None,
            status="success",
            user_action_required=None,
            warnings=[],
        )

    monkeypatch.setattr("chesscoach.server.run_coaching_pipeline", _run_pipeline)

    response = client.post(
        "/coach",
        json={
            "image_base64": _image_payload(),
            "side_to_move": "w",
            "white_king_start_click": {"x": 1.0, "y": 2.0},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["position"]["fen"] == STARTING_FEN
    assert payload["analysis"]["top_moves"][0]["score_display"] == "+0.35"
