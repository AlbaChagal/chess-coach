from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from chesscoach.analysis.models import MoveAnalysis
from chesscoach.explanation.models import (
    BestMoveComparison,
    PlayedMoveResult,
    StructuredExplanation,
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


def _oriented_image_payload(width: int, height: int, orientation: int) -> str:
    image = PILImage.new("RGB", (width, height), color=(255, 0, 0))
    exif = PILImage.Exif()
    exif[274] = orientation
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


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


def _saved_snapshot_payload() -> dict[str, object]:
    return {
        "position": {
            "fen": STARTING_FEN,
            "fen_placement": STARTING_PLACEMENT,
            "side_to_move": "w",
            "castling_rights": "KQkq",
            "en_passant": "-",
            "source": "heuristic",
            "user_confirmed_orientation": True,
            "white_king_start_click": {"x": 10.0, "y": 20.0},
        },
        "analysis": {
            "fen": STARTING_FEN,
            "top_moves": [
                {
                    "move_san": "e4",
                    "move_uci": "e2e4",
                    "score_cp": 35,
                    "score_mate": None,
                    "score_display": "+0.35",
                    "depth": 20,
                    "continuation": ["e5", "Nf3"],
                }
            ],
            "engine_depth": 20,
            "analysis_latency_ms": 12.0,
            "analysis_status": "success",
        },
        "explanation": {
            "move_uci": "e2e4",
            "move_san": "e4",
            "explanation_text": "Play e4 to claim central space.",
            "structured_explanation": {
                "summary": "e4 is the cleanest move.",
                "what_the_move_does": "It claims the center.",
                "what_it_threatens": "It opens lines for development.",
                "why_it_is_best": "It keeps the strongest evaluation.",
                "why_alternatives_are_worse": "They yield central control.",
                "alternatives": [],
                "tactical_themes": [],
            },
            "played_move_result": None,
            "comparison": None,
            "provider": "openai",
            "status": "success",
        },
        "explanation_warnings": [],
        "played_move_selection": {
            "from": "e2",
            "to": "e4",
            "promotion": "",
        },
    }


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
    assert '/static/app.css?v=' in response.text
    assert '/static/app.js?v=' in response.text


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
    assert response.json()["settings"]["show_coordinates"] is True


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
    assert 'data-setting-input="showCoordinates"' in response.text
    assert "Coordinates display affects the analysis board" in response.text


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
    assert "Saved Positions" in response.text
    assert 'data-saved-app' in response.text


def test_analyze_page_renders_phase_two_flow_shell(client: TestClient) -> None:
    _signup(client)

    response = client.get("/app/analyze")

    assert response.status_code == 200
    assert "Load a Board Image" in response.text
    assert "Where did the white king start the game?" in response.text
    assert 'data-detect-endpoint="/detect-board"' in response.text
    assert 'data-analyze-endpoint="/analyze"' in response.text
    assert 'data-explain-endpoint="/explain"' in response.text
    assert '/static/app.css?v=' in response.text
    assert '/static/app.js?v=' in response.text
    assert 'data-analysis-board' in response.text
    assert "Board preview updates from the starting position" in response.text
    assert "Continue to Analysis" in response.text
    assert "Top Lines" in response.text
    assert "Explain Best Move" in response.text
    assert "Compare My Move" in response.text
    assert "Show Structured Breakdown" in response.text


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


def test_detect_board_endpoint_applies_exif_orientation(
    client: TestClient, monkeypatch
) -> None:
    captured: dict[str, tuple[int, int]] = {}

    def _detect_board_corners(image):
        captured["shape"] = image.shape[:2]
        return np.array(
            [[0.0, 0.0], [19.0, 0.0], [19.0, 39.0], [0.0, 39.0]],
            dtype=np.float32,
        )

    monkeypatch.setattr("chesscoach.server.detect_board_corners", _detect_board_corners)

    response = client.post(
        "/detect-board",
        json={"image_base64": _oriented_image_payload(40, 20, 6)},
    )

    assert response.status_code == 200
    assert captured["shape"] == (40, 20)
    assert response.json()["status"] == "success"


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


def test_complete_position_endpoint_returns_structured_failure_for_invalid_board(
    client: TestClient,
) -> None:
    response = client.post(
        "/complete-position",
        json={
            "fen_placement": "P7/1p6/1bp5/2kp4/8/8/8/6rP",
            "side_to_move": "w",
            "white_king_start_click": {"x": 1.0, "y": 2.0},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "failed"
    assert payload["position"] is None
    assert payload["warnings"][0]["code"] == "invalid_board_position"


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


def test_explain_endpoint_supports_best_move_mode(
    client: TestClient, monkeypatch
) -> None:
    monkeypatch.setattr(
        "chesscoach.server.run_analysis", lambda position, top_n: _analysis_result()
    )

    def _run_explanation(position, analysis, request):
        assert request.include_explanation is True
        assert request.played_move_uci is None
        return (
            ExplanationResult(
                move_uci="e2e4",
                move_san="e4",
                explanation_text="Play e4 to claim central space.",
                structured_explanation=StructuredExplanation(
                    summary="e4 is the cleanest way to start.",
                    what_the_move_does="It places a pawn in the center.",
                    what_it_threatens="It prepares quick development.",
                    why_it_is_best="It keeps the strongest evaluation.",
                    why_alternatives_are_worse="They concede central control.",
                    alternatives=[],
                    tactical_themes=[],
                ),
                played_move_result=None,
                comparison=None,
                provider="openai",
                status="success",
            ),
            [],
        )

    monkeypatch.setattr("chesscoach.server.run_explanation", _run_explanation)

    response = client.post("/explain", json={"fen": STARTING_FEN})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["explanation"]["move_uci"] == "e2e4"
    assert payload["explanation"]["explanation_text"] == (
        "Play e4 to claim central space."
    )


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


def test_settings_endpoint_requires_auth(client: TestClient) -> None:
    response = client.get("/api/settings")

    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated."


def test_settings_endpoints_round_trip(client: TestClient) -> None:
    _signup(client)

    initial = client.get("/api/settings")
    updated = client.post("/api/settings", json={"show_coordinates": False})
    refreshed = client.get("/api/settings")

    assert initial.status_code == 200
    assert initial.json()["settings"]["show_coordinates"] is True
    assert updated.status_code == 200
    assert updated.json()["settings"]["show_coordinates"] is False
    assert refreshed.status_code == 200
    assert refreshed.json()["settings"]["show_coordinates"] is False


def test_saved_snapshot_endpoints_round_trip(client: TestClient) -> None:
    _signup(client)

    created = client.post("/api/saved", json={"snapshot": _saved_snapshot_payload()})
    snapshot_id = created.json()["snapshot"]["id"]
    listed = client.get("/api/saved")
    fetched = client.get(f"/api/saved/{snapshot_id}")
    deleted = client.delete(f"/api/saved/{snapshot_id}")
    listed_after_delete = client.get("/api/saved")

    assert created.status_code == 200
    assert created.json()["snapshot"]["summary"]["best_move_san"] == "e4"
    assert listed.status_code == 200
    assert len(listed.json()["snapshots"]) == 1
    assert listed.json()["snapshots"][0]["has_explanation"] is True
    assert fetched.status_code == 200
    assert fetched.json()["snapshot"]["snapshot"]["position"]["fen"] == STARTING_FEN
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert listed_after_delete.status_code == 200
    assert listed_after_delete.json()["snapshots"] == []


def test_saved_snapshot_creation_validates_required_fields(client: TestClient) -> None:
    _signup(client)

    response = client.post("/api/saved", json={"snapshot": {"position": {}}})

    assert response.status_code == 422
    assert response.json()["detail"] == "Saved snapshot is missing required analysis data."


def test_saved_snapshot_endpoints_are_user_scoped(client: TestClient) -> None:
    _signup(client, "first@example.com")
    created = client.post("/api/saved", json={"snapshot": _saved_snapshot_payload()})
    snapshot_id = created.json()["snapshot"]["id"]

    client.post("/auth/logout")
    _signup(client, "second@example.com")

    listed = client.get("/api/saved")
    fetched = client.get(f"/api/saved/{snapshot_id}")
    deleted = client.delete(f"/api/saved/{snapshot_id}")

    assert listed.status_code == 200
    assert listed.json()["snapshots"] == []
    assert fetched.status_code == 404
    assert fetched.json()["detail"] == "Saved snapshot not found."
    assert deleted.status_code == 404
    assert deleted.json()["detail"] == "Saved snapshot not found."
