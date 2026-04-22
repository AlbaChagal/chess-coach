from __future__ import annotations

import re
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.request import urlopen

import cv2
import pytest
import uvicorn
from playwright.sync_api import Page, expect

from chesscoach.analysis.models import MoveAnalysis
from chesscoach.explanation.models import (
    BestMoveComparison,
    PlayedMoveResult,
    StructuredExplanation,
    StructuredPlayedMoveExplanation,
)
from chesscoach.pipeline_models import AnalysisResult, ExplanationResult, VisionResult
from chesscoach.server import create_app
from tests.vision.conftest import make_synthetic_board

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
STARTING_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
AFTER_E4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
PIECE_ROUTING_FEN = "4k3/8/8/3p4/2B1P3/8/4N3/4K3 w - - 0 1"


def _pick_free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _analysis_result(fen: str = STARTING_FEN) -> AnalysisResult:
    if fen == AFTER_E4_FEN:
        return AnalysisResult(
            fen=AFTER_E4_FEN,
            top_moves=[
                MoveAnalysis(
                    "c5",
                    "c7c5",
                    18,
                    None,
                    20,
                    ["Nf3", "Nc6"],
                    ["g1f3", "b8c6"],
                ),
                MoveAnalysis(
                    "e5",
                    "e7e5",
                    14,
                    None,
                    20,
                    ["Nf3", "Nc6"],
                    ["g1f3", "b8c6"],
                ),
                MoveAnalysis(
                    "Nf6",
                    "g8f6",
                    11,
                    None,
                    20,
                    ["Nc3", "d5"],
                    ["b1c3", "d7d5"],
                ),
            ],
            engine_depth=20,
            analysis_latency_ms=10.0,
            analysis_status="success",
        )
    return AnalysisResult(
        fen=fen,
        top_moves=[
            MoveAnalysis(
                "e4",
                "e2e4",
                35,
                None,
                20,
                ["e5", "Nf3", "Nc6"],
                ["e7e5", "g1f3", "b8c6"],
            ),
            MoveAnalysis(
                "d4",
                "d2d4",
                22,
                None,
                20,
                ["d5", "c4"],
                ["d7d5", "c2c4"],
            ),
            MoveAnalysis(
                "Nf3",
                "g1f3",
                18,
                None,
                20,
                ["d5", "g3"],
                ["d7d5", "g2g3"],
            ),
        ],
        engine_depth=20,
        analysis_latency_ms=12.0,
        analysis_status="success",
    )


def _piece_routing_analysis_result() -> AnalysisResult:
    return AnalysisResult(
        fen=PIECE_ROUTING_FEN,
        top_moves=[
            MoveAnalysis(
                "Ng3",
                "e2g3",
                42,
                None,
                20,
                ["Kf7"],
                ["e8f7"],
            ),
            MoveAnalysis(
                "Bxd5",
                "c4d5",
                31,
                None,
                20,
                ["Kf8"],
                ["e8f8"],
            ),
            MoveAnalysis(
                "exd5",
                "e4d5",
                25,
                None,
                20,
                ["Kf7"],
                ["e8f7"],
            ),
        ],
        engine_depth=20,
        analysis_latency_ms=8.0,
        analysis_status="success",
    )


@pytest.fixture
def board_image_path(tmp_path: Path) -> Path:
    path = tmp_path / "board.png"
    assert cv2.imwrite(str(path), make_synthetic_board())
    return path


@pytest.fixture
def live_server_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[str]:
    monkeypatch.setenv("CHESSCOACH_AUTH_DB", str(tmp_path / "auth.db"))
    monkeypatch.setenv("CHESSCOACH_SESSION_SECRET", "test-session-secret")

    import chesscoach.server as server_module

    monkeypatch.setattr(
        server_module,
        "detect_board_corners",
        lambda _image: server_module.np.array(
            [[0.0, 0.0], [511.0, 0.0], [511.0, 511.0], [0.0, 511.0]],
            dtype=server_module.np.float32,
        ),
    )

    def _run_vision(request):
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

    monkeypatch.setattr(server_module, "run_vision", _run_vision)
    monkeypatch.setattr(
        server_module,
        "run_analysis",
        lambda position, top_n: _analysis_result(position.fen),
    )

    def _run_explanation(position, analysis, request):
        if request.played_move_uci == "d2d4":
            return (
                ExplanationResult(
                    move_uci="d2d4",
                    move_san="d4",
                    explanation_text="d4 is playable, but it concedes some central flexibility.",
                    structured_explanation=StructuredPlayedMoveExplanation(
                        summary="d4 is solid but a touch less precise here.",
                        what_the_move_tried_to_do="It grabs space and supports c4.",
                        what_was_missed="It does not challenge the center as directly as e4.",
                        what_changed_after_move="Black equalizes a bit more comfortably.",
                        why_best_move_was_better="e4 keeps the initiative cleaner.",
                        practical_lesson="Prefer the move that maximizes central pressure.",
                        tactical_themes=["central control"],
                        alternatives=[],
                    ),
                    played_move_result=PlayedMoveResult(
                        move_uci="d2d4",
                        move_san="d4",
                        quality_label="good",
                        quality_emoji="!",
                        cp_loss=13,
                        tactics_after_played=[],
                        tactics_after_best=[],
                    ),
                    comparison=BestMoveComparison(
                        best_move_uci="e2e4",
                        best_move_san="e4",
                        best_move_score_display="+0.35",
                        played_move_uci="d2d4",
                        played_move_san="d4",
                        played_move_quality="good",
                        cp_loss=13,
                        why_best_move_is_better="It controls more central squares immediately.",
                    ),
                    provider="openai",
                    status="success",
                ),
                [],
            )
        if request.played_move_uci == "g1f3":
            return (
                ExplanationResult(
                    move_uci="g1f3",
                    move_san="Nf3",
                    explanation_text="Nf3 develops smoothly, but it delays the central pawn break.",
                    structured_explanation=StructuredPlayedMoveExplanation(
                        summary="Nf3 is natural development with slightly less bite.",
                        what_the_move_tried_to_do="It develops a knight toward the center.",
                        what_was_missed="White could strike with e4 immediately.",
                        what_changed_after_move="The position stays fine but less forcing.",
                        why_best_move_was_better="e4 claims space before Black settles.",
                        practical_lesson="Natural moves are fine, but timing central breaks matters.",
                        tactical_themes=["development"],
                        alternatives=[],
                    ),
                    played_move_result=PlayedMoveResult(
                        move_uci="g1f3",
                        move_san="Nf3",
                        quality_label="good",
                        quality_emoji="!",
                        cp_loss=17,
                        tactics_after_played=[],
                        tactics_after_best=[],
                    ),
                    comparison=BestMoveComparison(
                        best_move_uci="e2e4",
                        best_move_san="e4",
                        best_move_score_display="+0.35",
                        played_move_uci="g1f3",
                        played_move_san="Nf3",
                        played_move_quality="good",
                        cp_loss=17,
                        why_best_move_is_better="It seizes more space right away.",
                    ),
                    provider="openai",
                    status="success",
                ),
                [],
            )
        return (
            ExplanationResult(
                move_uci="e2e4",
                move_san="e4",
                explanation_text="e4 claims central space and opens lines for development.",
                structured_explanation=StructuredExplanation(
                    summary="e4 is the cleanest central break.",
                    what_the_move_does="It places a pawn in the center and frees both bishop and queen.",
                    what_it_threatens="It prepares quick development and more space.",
                    why_it_is_best="It keeps the strongest engine evaluation.",
                    why_alternatives_are_worse="They are a bit slower to claim central control.",
                    alternatives=[],
                    tactical_themes=["central control"],
                ),
                played_move_result=None,
                comparison=None,
                provider="openai",
                status="success",
            ),
            [],
        )

    monkeypatch.setattr(server_module, "run_explanation", _run_explanation)

    app = create_app()
    port = _pick_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with urlopen(f"{base_url}/health") as response:
                if response.status == 200:
                    break
        except OSError:
            time.sleep(0.1)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("Timed out waiting for live server.")

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def piece_routing_server_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[str]:
    monkeypatch.setenv("CHESSCOACH_AUTH_DB", str(tmp_path / "auth-piece-routing.db"))
    monkeypatch.setenv("CHESSCOACH_SESSION_SECRET", "test-session-secret")

    import chesscoach.server as server_module

    monkeypatch.setattr(
        server_module,
        "detect_board_corners",
        lambda _image: server_module.np.array(
            [[0.0, 0.0], [511.0, 0.0], [511.0, 511.0], [0.0, 511.0]],
            dtype=server_module.np.float32,
        ),
    )

    def _run_vision(request):
        return (
            VisionResult(
                fen_placement="4k3/8/8/3p4/2B1P3/8/4N3/4K3",
                vision_confidence=1.0,
                orientation_status="user_marked",
                needs_user_confirmation=False,
                white_king_start_click=request.white_king_start_click,
            ),
            [],
        )

    monkeypatch.setattr(server_module, "run_vision", _run_vision)
    monkeypatch.setattr(
        server_module,
        "run_analysis",
        lambda position, top_n: _piece_routing_analysis_result(),
    )

    app = create_app()
    port = _pick_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with urlopen(f"{base_url}/health") as response:
                if response.status == 200:
                    break
        except OSError:
            time.sleep(0.1)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("Timed out waiting for live server.")

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)


def _sign_up_and_open_analyze(page: Page, live_server_url: str) -> None:
    page.goto(f"{live_server_url}/signup")
    page.locator('input[name="email"]').fill("user@example.com")
    page.locator('input[name="password"]').fill("strongpass")
    page.locator('input[name="confirm_password"]').fill("strongpass")
    page.locator("[data-auth-submit]").click()
    page.wait_for_url(f"{live_server_url}/app/analyze")
    expect(page.locator('[data-analyze-app]')).to_be_visible()


def _upload_and_detect_board(
    page: Page,
    live_server_url: str,
    board_image_path: Path,
) -> None:
    _sign_up_and_open_analyze(page, live_server_url)
    page.locator("[data-image-input]").set_input_files(str(board_image_path))
    page.locator("[data-detect-button]").click()
    expect(page.get_by_text("Where did the white king start the game?")).to_be_visible()
    expect(page.locator("[data-image-stage]")).to_be_visible()


def _click_square(page: Page, image_locator: str, x_ratio: float, y_ratio: float) -> None:
    box = page.locator(image_locator).bounding_box()
    assert box is not None
    page.locator(image_locator).click(
        position={
            "x": box["width"] * x_ratio,
            "y": box["height"] * y_ratio,
        }
    )


def _square_piece(page: Page, square: str) -> str:
    return page.locator(f"[data-analysis-board] .square-{square} .piece-glyph").inner_text()


def _ready_square_piece(page: Page, square: str) -> str:
    return page.locator(f"[data-ready-board] .square-{square} .piece-glyph").inner_text()


def _capture_browser_errors(page: Page) -> list[str]:
    errors: list[str] = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.on(
        "console",
        lambda msg: errors.append(msg.text) if msg.type == "error" else None,
    )
    return errors


def test_orientation_click_shows_visible_selected_square_feedback(
    page: Page,
    live_server_url: str,
    board_image_path: Path,
) -> None:
    browser_errors = _capture_browser_errors(page)
    _upload_and_detect_board(page, live_server_url, board_image_path)

    _click_square(page, "[data-image-stage]", 0.5625, 0.9375)

    assert browser_errors == []
    expect(page.locator("[data-selection-note]")).to_have_text("Selected square: e1.")
    expect(page.locator("[data-selection-badge]")).to_be_visible()
    expect(page.locator("[data-selection-badge-square]")).to_have_text("e1")
    expect(page.locator("[data-selected-marker]")).not_to_have_attribute("hidden", "")


def test_analysis_flow_renders_board_and_top_lines(
    page: Page,
    live_server_url: str,
    board_image_path: Path,
) -> None:
    _upload_and_detect_board(page, live_server_url, board_image_path)
    expect(page.locator("[data-analysis-source-card]")).to_be_visible()
    expect(page.locator("[data-analysis-source-image]")).to_have_attribute(
        "src",
        re.compile(r"data:image/png;base64,.*"),
    )

    _click_square(page, "[data-image-stage]", 0.5625, 0.9375)
    page.locator("[data-orientation-continue-button]").click()
    expect(page.locator("[data-analysis-source-card]")).to_be_visible()
    page.locator('[data-side-option="w"]').click()
    page.locator("[data-complete-button]").click()
    expect(page.locator("[data-ready-fen]")).to_contain_text(STARTING_FEN)
    expect(page.locator("[data-analysis-source-card]")).to_be_visible()

    page.locator('[data-step-nav="orientation"]').click()
    expect(page.get_by_text("Where did the white king start the game?")).to_be_visible()
    expect(page.locator("[data-selection-badge-square]")).to_have_text("e1")
    expect(page.locator("[data-analysis-source-card]")).to_be_visible()

    page.locator('[data-step-nav="side"]').click()
    expect(page.get_by_text("Who moves next?")).to_be_visible()
    expect(page.locator('[data-side-option="w"]')).to_have_class(
        re.compile(r".*\bactive\b.*")
    )
    expect(page.locator("[data-analysis-source-card]")).to_be_visible()

    page.locator('[data-step-nav="ready"]').click()
    expect(page.locator("[data-ready-fen]")).to_contain_text(STARTING_FEN)
    expect(page.locator("[data-analysis-source-card]")).to_be_visible()

    page.locator('[data-step-nav="upload"]').click()
    expect(page.get_by_role("heading", name="Load a Board Image")).to_be_visible()
    expect(page.locator("[data-image-preview-card]")).to_be_visible()
    expect(page.locator("[data-analysis-source-card]")).to_be_visible()

    page.locator('[data-step-nav="ready"]').click()
    expect(page.locator("[data-ready-fen]")).to_contain_text(STARTING_FEN)

    page.locator("[data-continue-to-analysis-button]").click()
    expect(page.locator("[data-analysis-layout]")).to_be_visible()
    expect(page.locator("[data-line-list] .line-card")).to_have_count(3)
    expect(page.locator("[data-analysis-board] .analysis-square")).to_have_count(64)
    expect(page.locator("[data-analysis-error]")).to_be_hidden()
    expect(page.locator("[data-line-explain-button]")).to_have_count(3)
    expect(page.locator("[data-analysis-source-card]")).to_be_visible()
    expect(page.locator("[data-analysis-source-image]")).to_have_attribute(
        "src",
        re.compile(r"data:image/png;base64,.*"),
    )
    expect(page.locator("[data-analysis-arrow]")).not_to_have_attribute("hidden", "")
    expect(page.locator("[data-analysis-arrow-layer]")).not_to_have_attribute(
        "hidden", ""
    )
    expect(page.locator("[data-analysis-arrow-head]")).not_to_have_attribute(
        "hidden", ""
    )

    page.locator("[data-line-list] .line-card").nth(1).click()
    expect(page.locator("[data-analysis-arrow-head]")).not_to_have_attribute(
        "hidden", ""
    )
    expect(page.locator("[data-analysis-arrow-head]")).to_have_attribute(
        "d",
        re.compile(r".*Z"),
    )


def test_each_top_line_has_its_own_explain_action(
    page: Page,
    live_server_url: str,
    board_image_path: Path,
) -> None:
    _upload_and_detect_board(page, live_server_url, board_image_path)
    _click_square(page, "[data-image-stage]", 0.5625, 0.9375)
    page.locator("[data-orientation-continue-button]").click()
    page.locator('[data-side-option="w"]').click()
    page.locator("[data-complete-button]").click()
    page.locator("[data-continue-to-analysis-button]").click()

    expect(page.locator("[data-line-explain-button]")).to_have_count(3)
    expect(page.locator("[data-line-explain-button=\"0\"]")).to_have_text("Explain e4")
    expect(page.locator("[data-line-explain-button=\"1\"]")).to_have_text("Explain d4")
    expect(page.locator("[data-line-explain-button=\"2\"]")).to_have_text(
        "Explain Nf3"
    )

    page.locator("[data-line-explain-button=\"0\"]").click()
    expect(page.locator("[data-explanation-result]")).to_be_visible()
    expect(page.locator("[data-explanation-move-label]")).to_have_text("e4 (e2e4)")
    expect(page.locator("[data-explanation-text]")).to_have_text(
        "e4 claims central space and opens lines for development."
    )
    expect(page.locator("[data-played-move-result]")).to_be_hidden()

    page.locator("[data-line-explain-button=\"1\"]").click()
    expect(page.locator("[data-explanation-move-label]")).to_have_text("d4 (d2d4)")
    expect(page.locator("[data-explanation-text]")).to_have_text(
        "d4 is playable, but it concedes some central flexibility."
    )
    expect(page.locator("[data-played-move-result]")).to_be_visible()
    expect(page.locator("[data-comparison-summary]")).to_contain_text(
        "e4 (+0.35) was stronger than d4."
    )

    page.locator("[data-line-explain-button=\"2\"]").click()
    expect(page.locator("[data-explanation-move-label]")).to_have_text("Nf3 (g1f3)")
    expect(page.locator("[data-explanation-text]")).to_have_text(
        "Nf3 develops smoothly, but it delays the central pawn break."
    )


def test_analysis_arrows_only_bend_for_knights(
    page: Page,
    piece_routing_server_url: str,
    board_image_path: Path,
) -> None:
    _upload_and_detect_board(page, piece_routing_server_url, board_image_path)

    _click_square(page, "[data-image-stage]", 0.5625, 0.9375)
    page.locator("[data-orientation-continue-button]").click()
    page.locator('[data-side-option="w"]').click()
    page.locator("[data-complete-button]").click()
    expect(page.locator("[data-ready-fen]")).to_contain_text(PIECE_ROUTING_FEN)
    page.locator("[data-continue-to-analysis-button]").click()

    expect(page.locator("[data-analysis-layout]")).to_be_visible()

    page.locator("[data-line-list] .line-card").nth(0).click()
    knight_path = page.locator("[data-analysis-arrow]").get_attribute("d")
    knight_head = page.locator("[data-analysis-arrow-head]").get_attribute("d")
    assert knight_path is not None
    assert knight_head is not None
    assert knight_path.count("L") == 2
    assert knight_head.endswith("Z")

    page.locator("[data-line-list] .line-card").nth(1).click()
    bishop_path = page.locator("[data-analysis-arrow]").get_attribute("d")
    bishop_head = page.locator("[data-analysis-arrow-head]").get_attribute("d")
    assert bishop_path is not None
    assert bishop_head is not None
    assert bishop_path.count("L") == 1
    assert bishop_head.endswith("Z")

    page.locator("[data-line-list] .line-card").nth(2).click()
    pawn_path = page.locator("[data-analysis-arrow]").get_attribute("d")
    pawn_head = page.locator("[data-analysis-arrow-head]").get_attribute("d")
    assert pawn_path is not None
    assert pawn_head is not None
    assert pawn_path.count("L") == 1
    assert pawn_head.endswith("Z")


def test_analysis_playback_moves_board_forward_and_back(
    page: Page,
    live_server_url: str,
    board_image_path: Path,
) -> None:
    browser_errors = _capture_browser_errors(page)
    _upload_and_detect_board(page, live_server_url, board_image_path)

    _click_square(page, "[data-image-stage]", 0.5625, 0.9375)
    page.locator("[data-orientation-continue-button]").click()
    page.locator('[data-side-option="w"]').click()
    page.locator("[data-complete-button]").click()
    expect(page.locator("[data-ready-fen]")).to_contain_text(STARTING_FEN)
    page.locator('[data-step-nav="analysis"]').click()

    page.wait_for_timeout(250)
    assert browser_errors == []
    expect(page.locator("[data-analysis-layout]")).to_be_visible()
    expect(page.locator("[data-analysis-step-note]")).to_have_text("Step 0 of 4")
    expect(page.locator("[data-analysis-arrow-head]")).not_to_have_attribute(
        "hidden", ""
    )
    assert _square_piece(page, "e2") == "♙"
    assert _square_piece(page, "e4") == ""

    page.locator("[data-analysis-next-button]").click()

    expect(page.locator("[data-analysis-step-note]")).to_have_text("Step 1 of 4")
    expect(page.locator("[data-analysis-arrow-head]")).not_to_have_attribute(
        "hidden", ""
    )
    assert _square_piece(page, "e2") == ""
    assert _square_piece(page, "e4") == "♙"

    page.locator("[data-analysis-next-button]").click()

    expect(page.locator("[data-analysis-step-note]")).to_have_text("Step 2 of 4")
    assert _square_piece(page, "e5") == "♟"

    page.locator("[data-analysis-prev-button]").click()

    expect(page.locator("[data-analysis-step-note]")).to_have_text("Step 1 of 4")
    assert _square_piece(page, "e5") == ""
    assert _square_piece(page, "e4") == "♙"

    page.locator("[data-analysis-reset-button]").click()

    expect(page.locator("[data-analysis-step-note]")).to_have_text("Step 0 of 4")
    assert _square_piece(page, "e2") == "♙"
    assert _square_piece(page, "e4") == ""

    page.locator("[data-line-list] .line-card").nth(2).click()
    expect(page.locator("[data-analysis-arrow-head]")).not_to_have_attribute(
        "hidden", ""
    )
    expect(page.locator("[data-analysis-arrow-head]")).to_have_attribute(
        "d",
        re.compile(r".*Z"),
    )


def test_analysis_board_accepts_legal_move_and_reanalyzes_position(
    page: Page,
    live_server_url: str,
    board_image_path: Path,
) -> None:
    _upload_and_detect_board(page, live_server_url, board_image_path)

    _click_square(page, "[data-image-stage]", 0.5625, 0.9375)
    page.locator("[data-orientation-continue-button]").click()
    page.locator('[data-side-option="w"]').click()
    page.locator("[data-complete-button]").click()
    page.locator("[data-continue-to-analysis-button]").click()

    expect(page.locator("[data-line-list] .line-card").first).to_contain_text("1. e4")
    assert _square_piece(page, "e2") == "♙"
    assert _square_piece(page, "e4") == ""

    page.locator("[data-analysis-board] .square-e2").click()
    expect(page.locator("[data-analysis-board] .square-e4")).to_have_class(
        re.compile(r".*\bis-legal-target\b.*")
    )
    page.locator("[data-analysis-board] .square-e4").click()

    expect(page.locator("[data-analysis-step-note]")).to_have_text("Step 0 of 3")
    expect(page.locator("[data-line-list] .line-card").first).to_contain_text("1. c5")
    assert _square_piece(page, "e2") == ""
    assert _square_piece(page, "e4") == "♙"


def test_ready_stage_editor_supports_move_tray_cancel_and_commit(
    page: Page,
    live_server_url: str,
    board_image_path: Path,
) -> None:
    _upload_and_detect_board(page, live_server_url, board_image_path)

    _click_square(page, "[data-image-stage]", 0.5625, 0.9375)
    page.locator("[data-orientation-continue-button]").click()
    page.locator('[data-side-option="w"]').click()
    page.locator("[data-complete-button]").click()
    expect(page.locator("[data-ready-fen]")).to_contain_text(STARTING_FEN)

    page.locator("[data-ready-edit-toggle-button]").click()
    expect(page.locator("[data-ready-editor]")).to_be_visible()
    assert _ready_square_piece(page, "e2") == "♙"
    assert _ready_square_piece(page, "e4") == ""

    page.locator("[data-ready-board] .square-e2").click()
    page.locator("[data-ready-board] .square-e4").click()
    assert _ready_square_piece(page, "e2") == ""
    assert _ready_square_piece(page, "e4") == "♙"

    page.locator('[data-ready-piece-button="clear"]').click()
    page.locator("[data-ready-board] .square-b1").click()
    assert _ready_square_piece(page, "b1") == ""

    page.locator('[data-ready-piece-button="q"]').click()
    page.locator("[data-ready-board] .square-b1").click()
    assert _ready_square_piece(page, "b1") == "♛"

    page.locator("[data-ready-cancel-tool-button]").click()
    expect(page.locator('[data-ready-piece-button="q"]')).not_to_have_class(
        re.compile(r".*\bactive\b.*")
    )

    page.locator("[data-ready-cancel-button]").click()
    expect(page.locator("[data-ready-editor]")).to_be_hidden()

    page.locator("[data-ready-edit-toggle-button]").click()
    assert _ready_square_piece(page, "e2") == "♙"
    assert _ready_square_piece(page, "e4") == ""
    assert _ready_square_piece(page, "b1") == "♘"

    page.locator("[data-ready-board] .square-e2").click()
    page.locator("[data-ready-board] .square-e4").click()
    page.locator("[data-ready-apply-button]").click()

    expect(page.locator("[data-ready-editor]")).to_be_hidden()
    expect(page.locator("[data-ready-fen]")).to_contain_text(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
    )

    page.locator("[data-continue-to-analysis-button]").click()
    expect(page.locator("[data-analysis-layout]")).to_be_visible()
    assert _square_piece(page, "e2") == ""
    assert _square_piece(page, "e4") == "♙"


def test_ready_stage_editor_reset_and_invalid_validation(
    page: Page,
    live_server_url: str,
    board_image_path: Path,
) -> None:
    _upload_and_detect_board(page, live_server_url, board_image_path)

    _click_square(page, "[data-image-stage]", 0.5625, 0.9375)
    page.locator("[data-orientation-continue-button]").click()
    page.locator('[data-side-option="w"]').click()
    page.locator("[data-complete-button]").click()

    page.locator("[data-ready-edit-toggle-button]").click()
    expect(page.locator("[data-ready-editor]")).to_be_visible()

    page.locator('[data-ready-piece-button="clear"]').click()
    page.locator("[data-ready-board] .square-e1").click()
    expect(page.locator("[data-ready-error]")).to_have_text(
        "The board must contain exactly one white king."
    )
    expect(page.locator("[data-ready-apply-button]")).to_be_disabled()

    page.locator("[data-ready-reset-button]").click()
    assert _ready_square_piece(page, "e1") == "♔"
    expect(page.locator("[data-ready-error]")).to_be_hidden()
    expect(page.locator("[data-ready-feedback]")).to_have_text(
        "Draft restored to the detected position."
    )
