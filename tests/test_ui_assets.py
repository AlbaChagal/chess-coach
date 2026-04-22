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
    assert (
        "Selection saved. Continue if this matches where the white king started."
        in script
    )
    assert "Selected square:" not in script
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
    assert "function analysisLineMoves(state)" in script
    assert "function playbackMoves(state)" in script
    assert "const moves = playbackMoves(state);" in script
    assert 'baseFen: "",' in script
    assert "const baseFen = state.analysis.baseFen || state.completedPosition?.fen || \"\";" in (
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
    assert "function shouldUseOrthogonalArrow(piece)" in script
    assert 'return piece === "N" || piece === "n";' in script
    assert "piece: movingPieceAtSquare(boardState, move.from)," in script
    assert "const baseCenterX = toX - unitX * headLength;" in script
    assert '`M ${fromX} ${fromY} L ${midX} ${midY} L ${baseCenterX} ${baseCenterY}`' in script
    assert '`M ${fromX} ${fromY} L ${baseCenterX} ${baseCenterY}`' in script
    assert '`M ${toX} ${toY} L ${leftX} ${leftY} L ${rightX} ${rightY} Z`' in script
    assert "setElementVisibility(analysisArrow, true);" in script
    assert "setElementVisibility(analysisArrowHead, true);" in script
    assert "setElementVisibility(analysisArrowLayer, true);" in script
    assert "setElementVisibility(analysisSourceCard, !!state.imageDataUrl);" in script
    assert "grid-template-rows: repeat(8, minmax(0, 1fr));" in styles
    assert "min-height: 280px;" in styles
    assert ".analysis-square" in styles
    assert ".analysis-arrow-layer [data-analysis-arrow]" in styles
    assert ".analysis-arrow-layer [data-analysis-arrow-head]" in styles
    assert ".analysis-source-card {" in styles
    assert "min-width: 0;" in styles
    assert "min-height: 0;" in styles


def test_top_lines_render_per_line_explain_actions() -> None:
    script = (PROJECT_ROOT / "chesscoach/static/app.js").read_text()
    styles = (PROJECT_ROOT / "chesscoach/static/app.css").read_text()
    template = (PROJECT_ROOT / "chesscoach/templates/app_shell.html").read_text()

    assert 'data-played-move-form' not in template
    assert "Request an explanation for any suggested line or tap the board" in template
    assert "requestedLineIndex: null," in script
    assert "requestedMoveUci: null," in script
    assert 'data-line-explain-button="${index}"' in script
    assert 'data-line-select-button="${index}"' in script
    assert 'requestExplanationForLine(index);' in script
    assert 'requestExplanation("best_move", null, index);' in script
    assert 'requestExplanation("played_move", selectedMove.move_uci, index);' in (
        script
    )
    assert ".line-card-select {" in styles
    assert ".line-card-actions {" in styles
    assert ".line-explain-button {" in styles
    assert ".interactive-board-note {" in styles


def test_interactive_analysis_board_uses_backend_legal_moves_and_move_application() -> None:
    script = (PROJECT_ROOT / "chesscoach/static/app.js").read_text()
    styles = (PROJECT_ROOT / "chesscoach/static/app.css").read_text()
    template = (PROJECT_ROOT / "chesscoach/templates/app_shell.html").read_text()

    assert 'data-legal-moves-endpoint="/legal-moves"' in template
    assert 'data-play-move-endpoint="/play-move"' in template
    assert "interactiveLegalMoves: []," in script
    assert "interactiveLegalMovesFen: \"\"," in script
    assert "interactiveTargetSquares: []," in script
    assert "sessionMoves: []," in script
    assert "history: []," in script
    assert "function ensureInteractiveLegalMoves(fen)" in script
    assert "function handleAnalysisBoardClick(event)" in script
    assert "function applyInteractiveMove(fen, moveUci)" in script
    assert "function createAnalysisSnapshot()" in script
    assert "function applyAnalysisSnapshot(snapshot, stepIndex)" in script
    assert "function snapshotForStep(targetStep)" in script
    assert "function setAnalysisStep(targetStep)" in script
    assert "function matchingSuggestedMoveIndex(moveUci)" in script
    assert "function playbackMoves(state)" in script
    assert "clearAnalysisInteraction();" in script
    assert "state.analysis.interactiveTargetSquares = [" in script
    assert "const suggestedMoveIndex = matchingSuggestedMoveIndex(moveUci);" in script
    assert "state.analysis.activeLineIndex = suggestedMoveIndex;" in script
    assert "createAnalysisSnapshot()" in script
    assert "await analyzeCurrentPosition(" in script
    assert 'body: JSON.stringify({ fen })' in script
    assert 'body: JSON.stringify({' in script
    assert 'move_uci: moveUci,' in script
    assert ".slice(0, state.analysis.stepIndex)" in script
    assert ".concat(moveUci);" in script
    assert "stepIndex: sessionMoves.length," in script
    assert ".analysis-square.is-selected {" in styles
    assert ".analysis-square.is-legal-target::after {" in styles
    assert ".analysis-square.is-legal-capture::after {" in styles
    assert 'square.classList.add("is-legal-capture");' in script
    assert "top: 50%;" in styles
    assert "left: 50%;" in styles
    assert "transform: translate(-50%, -50%);" in styles
    assert "background: rgba(89, 93, 102, 0.34);" in styles
    assert "background: rgba(89, 93, 102, 0.18);" in styles
    assert "z-index: 2;" in styles
    assert "z-index: 3;" in styles


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
    assert "const nextBaseFen = baseFen || state.completedPosition.fen;" in script
    assert "baseFen: nextBaseFen," in script
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


def test_ready_stage_editor_assets_and_validation_are_wired() -> None:
    script = (PROJECT_ROOT / "chesscoach/static/app.js").read_text()
    styles = (PROJECT_ROOT / "chesscoach/static/app.css").read_text()
    template = (PROJECT_ROOT / "chesscoach/templates/app_shell.html").read_text()

    assert 'data-ready-edit-toggle-button' in template
    assert 'data-ready-flip-button' not in template
    assert (
        template.index('data-analysis-board')
        < template.index('data-ready-editor')
        < template.index('data-ready-edit-toggle-button')
    )
    assert 'data-ready-editor' in template
    assert 'data-ready-piece-button="clear"' in template
    assert 'data-ready-cancel-tool-button' in template
    assert 'data-ready-reset-button' in template
    assert 'data-ready-cancel-button' in template
    assert 'data-ready-apply-button' in template
    assert "function createReadyEditorState()" in script
    assert "function boardPerspectiveOrientation()" in script
    assert "function boardPerspectiveButtonLabel()" in script
    assert "function toggleBoardPerspective()" in script
    assert 'state.step === "ready" || state.step === "analysis"' in script
    assert "function validateReadyPlacement(placementText)" in script
    assert ('return "The board must contain exactly one white king.";') in script
    assert ('return "The board must contain exactly one black king.";') in script
    assert (
        'return "Pawns cannot be placed on the first or eighth rank.";'
    ) in script
    assert "function handleReadyBoardClick(event)" in script
    assert "function applyReadyEditor()" in script
    assert "function resetAnalysisForCommittedPosition()" in script
    assert "const analysisMatchesCommittedFen =" in script
    assert "state.analysis.baseFen === state.completedPosition?.fen;" in script
    assert 'analysisFlipButton?.addEventListener("click", toggleBoardPerspective);' in (
        script
    )
    assert 'state.readyEditor.draftPlacement = state.readyEditor.detectedPlacement;' in script
    assert 'state.readyEditor.draftPlacement = payload.position.fen_placement;' in script
    assert "resetAnalysisForCommittedPosition();" in script
    assert template.index('data-ready-editor') < template.index("analysis-playback")
    assert ".ready-editor {" in styles
    assert ".ready-editor-toolbar {" in styles
    assert ".ready-editor-actions {" in styles
    assert ".piece-palette {" in styles
    assert ".piece-chip {" in styles
    assert ".analysis-square.is-selected {" in styles
