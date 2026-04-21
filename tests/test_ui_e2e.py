from __future__ import annotations

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
from chesscoach.pipeline_models import AnalysisResult, ImageClick, VisionResult
from chesscoach.server import create_app
from tests.vision.conftest import make_synthetic_board

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
STARTING_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def _pick_free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _analysis_result() -> AnalysisResult:
    return AnalysisResult(
        fen=STARTING_FEN,
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
        lambda position, top_n: _analysis_result(),
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


def test_orientation_click_shows_visible_selected_square_feedback(
    page: Page,
    live_server_url: str,
    board_image_path: Path,
) -> None:
    _upload_and_detect_board(page, live_server_url, board_image_path)

    _click_square(page, "[data-image-stage]", 0.5625, 0.9375)

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

    _click_square(page, "[data-image-stage]", 0.5625, 0.9375)
    page.locator("[data-orientation-continue-button]").click()
    page.locator('[data-side-option="w"]').click()
    page.locator("[data-complete-button]").click()
    expect(page.locator("[data-ready-fen]")).to_contain_text(STARTING_FEN)

    page.locator("[data-continue-to-analysis-button]").click()
    expect(page.locator("[data-analysis-layout]")).to_be_visible()
    expect(page.locator("[data-line-list] .line-card")).to_have_count(3)
    expect(page.locator("[data-analysis-board] .analysis-square")).to_have_count(64)
    expect(page.locator("[data-analysis-error]")).to_be_hidden()
    expect(page.locator("[data-explain-best-button]")).to_be_visible()
