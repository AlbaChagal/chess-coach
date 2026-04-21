from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_orientation_click_handler_is_bound_to_image_stage() -> None:
    script = (PROJECT_ROOT / "chesscoach/static/app.js").read_text()

    assert 'stage.addEventListener("click", handleStageClick);' in script
    assert 'overlaySvg.addEventListener("click", handleOverlayClick);' not in script


def test_orientation_click_maps_from_rendered_image_bounds() -> None:
    script = (PROJECT_ROOT / "chesscoach/static/app.js").read_text()
    styles = (PROJECT_ROOT / "chesscoach/static/app.css").read_text()
    template = (PROJECT_ROOT / "chesscoach/templates/app_shell.html").read_text()

    assert "const rect = stageImage.getBoundingClientRect();" in script
    assert "if (state.flipped) {" in script
    assert 'cursor: crosshair;' in styles
    assert "stroke-dasharray: 8 5;" in styles
    assert "filter: drop-shadow(0 0 10px rgba(159, 61, 50, 0.4));" in styles
    assert '.corner-handle {' in styles
    assert 'cursor: grab;' in styles
    assert 'data-selection-badge' in template
    assert 'data-selected-marker' in template
    assert "function setElementVisibility(element, visible)" in script
    assert 'element.removeAttribute("hidden");' in script
    assert 'setElementVisibility(selectionBadge, !!state.selectedSquare);' in script
    assert 'selectedMarkerText.textContent = markerText;' in script
    assert '.selection-badge {' in styles


def test_analysis_success_path_uses_dedicated_render_error_message() -> None:
    script = (PROJECT_ROOT / "chesscoach/static/app.js").read_text()

    assert 'showError(analysisError, "Unable to run engine analysis right now.");' in (
        script
    )
    assert "Unable to update the analysis view right now. Please try again." in script
    assert 'console.error("analysis render failed", _error, payload.analysis);' in script
    assert "renderAfterAnalysisSuccess" not in script


def test_analysis_board_renderer_is_hardened_against_partial_dom_state() -> None:
    script = (PROJECT_ROOT / "chesscoach/static/app.js").read_text()
    styles = (PROJECT_ROOT / "chesscoach/static/app.css").read_text()
    template = (PROJECT_ROOT / "chesscoach/templates/app_shell.html").read_text()

    assert (
        "Board preview unavailable right now. Lines and scores are still available."
        in script
    )
    assert "function normalizeAnalysisResult(result)" in script
    assert 'console.error("analysis playback render failed", error, state.analysis);' in (
        script
    )
    assert 'data-analysis-source-card' in template
    assert 'data-analysis-source-image' in template
    assert "function rebuildBoard(boardElement, fen, orientation, showNotation)" in script
    assert "const squareIndex = (8 - rank) * 8 + fileIndex;" in script
    assert "pieceGlyph.textContent = piece ? PIECE_TO_GLYPH[piece] : \"\";" in script
    assert 'const analysisArrowHead = root.querySelector("[data-analysis-arrow-head]");' in script
    assert 'analysisArrow.setAttribute(' in script
    assert 'analysisArrowHead.setAttribute(' in script
    assert "const baseCenterX = toX - unitX * headLength;" in script
    assert '`M ${fromX} ${fromY} L ${midX} ${midY} L ${baseCenterX} ${baseCenterY}`' in script
    assert '`M ${toX} ${toY} L ${leftX} ${leftY} L ${rightX} ${rightY} Z`' in script
    assert "setElementVisibility(analysisArrow, true);" in script
    assert "setElementVisibility(analysisArrowHead, true);" in script
    assert "setElementVisibility(analysisArrowLayer, true);" in script
    assert "grid-template-rows: repeat(8, minmax(0, 1fr));" in styles
    assert "min-height: 280px;" in styles
    assert ".analysis-square" in styles
    assert ".analysis-arrow-layer [data-analysis-arrow]" in styles
    assert ".analysis-arrow-layer [data-analysis-arrow-head]" in styles
    assert ".analysis-source-card {" in styles
    assert "min-width: 0;" in styles
    assert "min-height: 0;" in styles


def test_starting_board_preview_is_rendered_from_known_fen() -> None:
    script = (PROJECT_ROOT / "chesscoach/static/app.js").read_text()

    assert "const STARTING_POSITION_FEN =" in script
    assert 'const analysisBoardElement = root.querySelector("[data-analysis-board]");' in (
        script
    )
    assert "function renderPreviewBoard()" in script
    assert "let previewFen = state.completedPosition?.fen || STARTING_POSITION_FEN;" in (
        script
    )
    assert "state.completedPosition?.fen ||" in script
    assert "STARTING_POSITION_FEN;" in script
    assert "renderPreviewBoard();" in script
    assert "renderAnalysisState();" in script


def test_reset_flow_buttons_are_bound_globally() -> None:
    script = (PROJECT_ROOT / "chesscoach/static/app.js").read_text()

    assert 'root.querySelectorAll("[data-reset-flow-button]").forEach((button) => {' in (
        script
    )
    assert 'button.addEventListener("click", resetToUpload);' in script


def test_step_pills_are_clickable_navigation_controls() -> None:
    script = (PROJECT_ROOT / "chesscoach/static/app.js").read_text()
    template = (PROJECT_ROOT / "chesscoach/templates/app_shell.html").read_text()
    styles = (PROJECT_ROOT / "chesscoach/static/app.css").read_text()

    assert 'data-step-nav="upload"' in template
    assert 'data-step-nav="analysis"' in template
    assert "function navigateToStep(step)" in script
    assert 'root.querySelectorAll("[data-step-nav]").forEach((button) => {' in script
    assert ".step-pill:hover:not(:disabled)" in styles


def test_orientation_manual_corner_correction_is_wired() -> None:
    script = (PROJECT_ROOT / "chesscoach/static/app.js").read_text()
    template = (PROJECT_ROOT / "chesscoach/templates/app_shell.html").read_text()

    assert 'data-corner-handle="0"' in template
    assert 'data-reset-corners-button' in template
    assert 'board_corners: state.detection?.board_corners?.map((point) => ({' in script
    assert 'stage.addEventListener("pointerdown", handleStagePointerDown);' in script
    assert 'stage.addEventListener("pointermove", handleStagePointerMove);' in script
    assert "function resetBoardCorners()" in script
