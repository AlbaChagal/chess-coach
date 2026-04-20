from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_orientation_click_handler_is_bound_to_image_stage() -> None:
    script = (PROJECT_ROOT / "chesscoach/static/app.js").read_text()

    assert 'stage.addEventListener("click", handleStageClick);' in script
    assert 'overlaySvg.addEventListener("click", handleOverlayClick);' not in script


def test_orientation_click_maps_from_rendered_image_bounds() -> None:
    script = (PROJECT_ROOT / "chesscoach/static/app.js").read_text()

    assert "const rect = stageImage.getBoundingClientRect();" in script
    assert "if (state.flipped) {" in script
    assert 'cursor: crosshair;' in (
        PROJECT_ROOT / "chesscoach/static/app.css"
    ).read_text()


def test_analysis_success_path_uses_dedicated_render_error_message() -> None:
    script = (PROJECT_ROOT / "chesscoach/static/app.js").read_text()

    assert 'showError(analysisError, "Unable to run engine analysis right now.");' in (
        script
    )
    assert "renderAfterAnalysisSuccess" not in script


def test_analysis_board_renderer_is_hardened_against_partial_dom_state() -> None:
    script = (PROJECT_ROOT / "chesscoach/static/app.js").read_text()
    styles = (PROJECT_ROOT / "chesscoach/static/app.css").read_text()

    assert (
        "Board preview unavailable right now. Lines and scores are still available."
        in script
    )
    assert "function rebuildBoard(boardElement, fen, orientation, showNotation)" in script
    assert "const squareIndex = (8 - rank) * 8 + fileIndex;" in script
    assert "pieceGlyph.textContent = piece ? PIECE_TO_GLYPH[piece] : \"\";" in script
    assert "grid-template-rows: repeat(8, minmax(0, 1fr));" in styles
    assert "min-height: 280px;" in styles
    assert ".analysis-square" in styles
    assert "min-width: 0;" in styles
    assert "min-height: 0;" in styles


def test_starting_board_preview_is_rendered_from_known_fen() -> None:
    script = (PROJECT_ROOT / "chesscoach/static/app.js").read_text()

    assert "const STARTING_POSITION_FEN =" in script
    assert 'const analysisBoardElement = root.querySelector("[data-analysis-board]");' in (
        script
    )
    assert "function renderPreviewBoard()" in script
    assert "const previewFen =" in script
    assert "state.completedPosition?.fen ||" in script
    assert "STARTING_POSITION_FEN;" in script
    assert "renderPreviewBoard();" in script
    assert "renderAnalysisState();" in script
