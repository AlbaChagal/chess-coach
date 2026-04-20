(function () {
  const ANALYZE_STORAGE_KEY = "chesscoach-analyze-state";
  const SETTINGS_STORAGE_KEY = "chesscoach-ui-settings";
  const PIECE_THEME_URL =
    "https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png";

  function showError(target, message) {
    if (!target) {
      return;
    }
    target.hidden = false;
    target.textContent = message;
  }

  function clearError(target) {
    if (!target) {
      return;
    }
    target.hidden = true;
    target.textContent = "";
  }

  function loadSettings() {
    try {
      const stored = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
      if (!stored) {
        return { showCoordinates: true };
      }
      return { showCoordinates: true, ...JSON.parse(stored) };
    } catch (_error) {
      return { showCoordinates: true };
    }
  }

  function saveSettings(settings) {
    window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  }

  async function handleAuthFormSubmit(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const endpoint = form.dataset.authEndpoint;
    const mode = form.dataset.mode;
    const email = form.elements.namedItem("email").value.trim();
    const password = form.elements.namedItem("password").value;
    const confirmPasswordField = form.elements.namedItem("confirm_password");
    const errorTarget = form.querySelector("[data-auth-error]");
    const submitButton = form.querySelector("[data-auth-submit]");

    clearError(errorTarget);

    if (!email || !password) {
      showError(errorTarget, "Email and password are required.");
      return;
    }

    if (mode === "signup") {
      const confirmPassword = confirmPasswordField.value;
      if (password !== confirmPassword) {
        showError(errorTarget, "Passwords do not match.");
        return;
      }
    }

    submitButton.disabled = true;
    submitButton.textContent = mode === "signup" ? "Signing Up..." : "Logging In...";

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ email, password }),
      });
      const payload = await response.json();
      if (!response.ok) {
        showError(errorTarget, payload.detail || "Something went wrong.");
        return;
      }
      window.location.href = payload.redirect_to || "/app/analyze";
    } catch (_error) {
      showError(errorTarget, "Unable to reach the server. Please try again.");
    } finally {
      submitButton.disabled = false;
      submitButton.textContent = mode === "signup" ? "Sign Up" : "Log In";
    }
  }

  async function handleLogout(event) {
    event.preventDefault();
    const button = event.currentTarget;
    button.disabled = true;
    button.textContent = "Logging Out...";
    try {
      const response = await fetch("/auth/logout", {
        method: "POST",
        credentials: "same-origin",
      });
      const payload = await response.json();
      window.location.href = payload.redirect_to || "/login";
    } catch (_error) {
      button.disabled = false;
      button.textContent = "Log Out";
    }
  }

  function setupSettingsForm() {
    const settings = loadSettings();
    document.querySelectorAll("[data-setting-input]").forEach((input) => {
      if (input.dataset.settingInput === "showCoordinates") {
        input.checked = settings.showCoordinates;
      }
      input.addEventListener("change", () => {
        const nextSettings = {
          ...loadSettings(),
          [input.dataset.settingInput]: input.checked,
        };
        saveSettings(nextSettings);
      });
    });
  }

  function createAnalyzeState() {
    return {
      step: "upload",
      imageFile: null,
      imageDataUrl: "",
      detection: null,
      selectedClick: null,
      selectedSquare: null,
      sideToMove: null,
      completedPosition: null,
      flipped: false,
      analysis: {
        status: "idle",
        result: null,
        activeLineIndex: 0,
        stepIndex: 0,
        flipped: false,
      },
    };
  }

  function restoreAnalyzeState() {
    try {
      const stored = window.sessionStorage.getItem(ANALYZE_STORAGE_KEY);
      if (!stored) {
        return null;
      }
      return JSON.parse(stored);
    } catch (_error) {
      return null;
    }
  }

  function persistAnalyzeState(state) {
    const payload = {
      step: state.step,
      completedPosition: state.completedPosition,
      analysis: state.analysis,
    };
    window.sessionStorage.setItem(ANALYZE_STORAGE_KEY, JSON.stringify(payload));
  }

  function clearAnalyzeStateStorage() {
    window.sessionStorage.removeItem(ANALYZE_STORAGE_KEY);
  }

  function setupAnalyzeFlow(root) {
    const state = createAnalyzeState();
    const restored = restoreAnalyzeState();
    if (restored) {
      state.step = restored.step || state.step;
      state.completedPosition = restored.completedPosition || null;
      state.analysis = {
        ...state.analysis,
        ...(restored.analysis || {}),
      };
      if (!state.completedPosition && state.step !== "upload") {
        state.step = "upload";
      }
    }

    const detectEndpoint = root.dataset.detectEndpoint;
    const visionEndpoint = root.dataset.visionEndpoint;
    const completeEndpoint = root.dataset.completeEndpoint;
    const analyzeEndpoint = root.dataset.analyzeEndpoint;

    const stepSections = Array.from(root.querySelectorAll("[data-step]"));
    const stepPills = Array.from(root.querySelectorAll("[data-step-pill]"));
    const imageInput = root.querySelector("[data-image-input]");
    const cameraInput = root.querySelector("[data-camera-input]");
    const uploadError = root.querySelector("[data-upload-error]");
    const detectError = root.querySelector("[data-detect-error]");
    const completeError = root.querySelector("[data-complete-error]");
    const analysisError = root.querySelector("[data-analysis-error]");
    const previewCard = root.querySelector("[data-image-preview-card]");
    const previewImage = root.querySelector("[data-image-preview]");
    const stageImage = root.querySelector("[data-stage-image]");
    const stage = root.querySelector("[data-image-stage]");
    const overlaySvg = root.querySelector("[data-overlay-svg]");
    const boardOutline = root.querySelector("[data-board-outline]");
    const selectedSquare = root.querySelector("[data-selected-square]");
    const selectedPoint = root.querySelector("[data-selected-point]");
    const selectionNote = root.querySelector("[data-selection-note]");
    const sideNote = root.querySelector("[data-side-note]");
    const detectButton = root.querySelector("[data-detect-button]");
    const resetImageButton = root.querySelector("[data-reset-image-button]");
    const flipButton = root.querySelector("[data-flip-button]");
    const orientationContinueButton = root.querySelector(
      "[data-orientation-continue-button]"
    );
    const completeButton = root.querySelector("[data-complete-button]");
    const sideButtons = Array.from(root.querySelectorAll("[data-side-option]"));
    const readyFen = root.querySelector("[data-ready-fen]");
    const readyCastling = root.querySelector("[data-ready-castling]");
    const readyEnPassant = root.querySelector("[data-ready-en-passant]");
    const continueToAnalysisButton = root.querySelector(
      "[data-continue-to-analysis-button]"
    );
    const analysisLoading = root.querySelector("[data-analysis-loading]");
    const analysisLayout = root.querySelector("[data-analysis-layout]");
    const analysisBoardElement = root.querySelector("[data-analysis-board]");
    const analysisArrowLayer = root.querySelector("[data-analysis-arrow-layer]");
    const analysisArrow = root.querySelector("[data-analysis-arrow]");
    const analysisFlipButton = root.querySelector("[data-analysis-flip-button]");
    const analysisPrevButton = root.querySelector("[data-analysis-prev-button]");
    const analysisNextButton = root.querySelector("[data-analysis-next-button]");
    const analysisResetButton = root.querySelector("[data-analysis-reset-button]");
    const analysisRetryButton = root.querySelector("[data-analysis-retry-button]");
    const analysisStepNote = root.querySelector("[data-analysis-step-note]");
    const lineList = root.querySelector("[data-line-list]");

    let boardWidget = null;
    let boardShowNotation = null;

    function render() {
      stepSections.forEach((section) => {
        section.hidden = section.dataset.step !== state.step;
      });
      stepPills.forEach((pill) => {
        pill.classList.toggle("active", pill.dataset.stepPill === state.step);
      });

      if (previewCard) {
        previewCard.hidden = !state.imageDataUrl;
      }
      if (previewImage && state.imageDataUrl) {
        previewImage.src = state.imageDataUrl;
      }
      if (stageImage && state.imageDataUrl) {
        stageImage.src = state.imageDataUrl;
      }
      if (stage) {
        stage.classList.toggle("flipped", state.flipped);
      }

      detectButton.disabled = !state.imageDataUrl;
      resetImageButton.hidden = !state.imageDataUrl;
      orientationContinueButton.disabled = state.selectedClick === null;
      completeButton.disabled = state.sideToMove === null;

      sideButtons.forEach((button) => {
        button.classList.toggle(
          "active",
          button.dataset.sideOption === state.sideToMove
        );
      });

      selectionNote.textContent = state.selectedSquare
        ? `Selected square: ${state.selectedSquare}.`
        : "No square selected yet.";
      sideNote.textContent = state.sideToMove
        ? `${state.sideToMove === "w" ? "White" : "Black"} to move selected.`
        : "No side selected yet.";

      if (state.detection && state.detection.board_corners) {
        overlaySvg.setAttribute(
          "viewBox",
          `0 0 ${state.detection.image_width} ${state.detection.image_height}`
        );
        boardOutline.setAttribute(
          "points",
          state.detection.board_corners.map((point) => point.join(",")).join(" ")
        );
      }

      selectedSquare.hidden = state.selectedClick === null;
      selectedPoint.hidden = state.selectedClick === null;

      if (state.selectedClick) {
        selectedPoint.setAttribute("cx", String(state.selectedClick.x));
        selectedPoint.setAttribute("cy", String(state.selectedClick.y));
      }

      if (state.selectedClick && state.detection && state.selectedSquare) {
        const polygonPoints = squarePolygonPoints(state);
        selectedSquare.setAttribute(
          "points",
          polygonPoints.map((point) => point.join(",")).join(" ")
        );
      }

      if (state.completedPosition) {
        readyFen.textContent = state.completedPosition.fen;
        readyCastling.textContent = state.completedPosition.castling_rights;
        readyEnPassant.textContent = state.completedPosition.en_passant;
        continueToAnalysisButton.disabled = false;
      } else {
        continueToAnalysisButton.disabled = true;
      }

      renderAnalysisState();
      persistAnalyzeState(state);
    }

    function renderAnalysisState() {
      const analysis = state.analysis;
      analysisLoading.hidden = analysis.status !== "loading";
      analysisLayout.hidden = analysis.status !== "success";
      analysisRetryButton.hidden = analysis.status !== "failed";
      if (analysis.status !== "failed") {
        clearError(analysisError);
      }

      if (analysis.status !== "success" || !analysis.result) {
        return;
      }

      renderLineList();
      renderBoard();
      renderArrow();
      renderPlaybackControls();
    }

    function resetToUpload() {
      state.step = "upload";
      state.imageFile = null;
      state.imageDataUrl = "";
      state.detection = null;
      state.selectedClick = null;
      state.selectedSquare = null;
      state.sideToMove = null;
      state.completedPosition = null;
      state.flipped = false;
      state.analysis = createAnalyzeState().analysis;
      clearError(uploadError);
      clearError(detectError);
      clearError(completeError);
      clearError(analysisError);
      if (imageInput) {
        imageInput.value = "";
      }
      if (cameraInput) {
        cameraInput.value = "";
      }
      clearAnalyzeStateStorage();
      destroyBoard();
      render();
    }

    function handleFileSelection(file) {
      clearError(uploadError);
      if (!file) {
        return;
      }
      if (!file.type.startsWith("image/")) {
        showError(uploadError, "Please choose an image file.");
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        state.imageFile = file;
        state.imageDataUrl = String(reader.result || "");
        state.detection = null;
        state.selectedClick = null;
        state.selectedSquare = null;
        state.sideToMove = null;
        state.completedPosition = null;
        state.flipped = false;
        state.analysis = createAnalyzeState().analysis;
        render();
      };
      reader.readAsDataURL(file);
    }

    async function runBoardDetection() {
      clearError(detectError);
      if (!state.imageDataUrl) {
        showError(uploadError, "Choose an image before continuing.");
        return;
      }
      detectButton.disabled = true;
      detectButton.textContent = "Detecting...";
      try {
        const response = await fetch(detectEndpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "same-origin",
          body: JSON.stringify({ image_base64: state.imageDataUrl }),
        });
        const payload = await response.json();
        if (!response.ok || payload.status !== "success") {
          const warning = payload.warnings && payload.warnings[0];
          showError(
            detectError,
            (warning && warning.message) ||
              "The board could not be detected. Please try to upload a clearer image."
          );
          state.step = "orientation";
          state.detection = payload.detection || null;
          render();
          return;
        }
        state.detection = payload.detection;
        state.selectedClick = null;
        state.selectedSquare = null;
        state.step = "orientation";
        render();
      } catch (_error) {
        showError(
          detectError,
          "Unable to detect the board right now. Please try again."
        );
        state.step = "orientation";
        render();
      } finally {
        detectButton.disabled = !state.imageDataUrl;
        detectButton.textContent = "Detect Board";
      }
    }

    function svgPointFromEvent(event) {
      const point = overlaySvg.createSVGPoint();
      point.x = event.clientX;
      point.y = event.clientY;
      const matrix = overlaySvg.getScreenCTM();
      if (!matrix) {
        return null;
      }
      return point.matrixTransform(matrix.inverse());
    }

    function handleOverlayClick(event) {
      if (!state.detection || !state.detection.board_corners) {
        return;
      }
      clearError(detectError);
      const point = svgPointFromEvent(event);
      if (!point) {
        return;
      }
      const corners = state.detection.board_corners;
      const imageToBoard = solveHomography(corners, [
        [0, 0],
        [8, 0],
        [8, 8],
        [0, 8],
      ]);
      const boardPoint = applyHomography(imageToBoard, [point.x, point.y]);
      const fileIndex = Math.floor(boardPoint[0]);
      const rankIndex = Math.floor(boardPoint[1]);
      if (fileIndex < 0 || fileIndex > 7 || rankIndex < 0 || rankIndex > 7) {
        showError(detectError, "Tap inside the detected board area.");
        return;
      }
      state.selectedClick = { x: point.x, y: point.y };
      state.selectedSquare = squareName(fileIndex, rankIndex);
      render();
    }

    async function completePosition() {
      clearError(completeError);
      if (!state.selectedClick || !state.sideToMove) {
        showError(
          completeError,
          "Finish the orientation and side-to-move steps first."
        );
        return;
      }
      completeButton.disabled = true;
      completeButton.textContent = "Completing...";
      try {
        const visionResponse = await fetch(visionEndpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "same-origin",
          body: JSON.stringify({
            image_base64: state.imageDataUrl,
            white_king_start_click: state.selectedClick,
          }),
        });
        const visionPayload = await visionResponse.json();
        if (!visionResponse.ok || visionPayload.status !== "success") {
          const warning = visionPayload.warnings && visionPayload.warnings[0];
          showError(
            completeError,
            (warning && warning.message) ||
              "The board could not be detected. Please try another image."
          );
          return;
        }
        const completionResponse = await fetch(completeEndpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "same-origin",
          body: JSON.stringify({
            fen_placement: visionPayload.vision.fen_placement,
            side_to_move: state.sideToMove,
            white_king_start_click: state.selectedClick,
          }),
        });
        const completionPayload = await completionResponse.json();
        if (!completionResponse.ok || completionPayload.status !== "success") {
          showError(
            completeError,
            completionPayload.detail || "Unable to complete the position."
          );
          return;
        }
        state.completedPosition = completionPayload.position;
        state.analysis = createAnalyzeState().analysis;
        state.step = "ready";
        render();
      } catch (_error) {
        showError(completeError, "Unable to complete the position right now.");
      } finally {
        completeButton.disabled = state.sideToMove === null;
        completeButton.textContent = "Complete Position";
      }
    }

    async function enterAnalysisMode() {
      if (!state.completedPosition) {
        return;
      }
      if (typeof window.Chessboard === "undefined" || typeof window.Chess === "undefined") {
        showError(
          analysisError,
          "Analysis board assets failed to load. Please refresh and try again."
        );
        state.step = "analysis";
        state.analysis.status = "failed";
        render();
        return;
      }
      state.step = "analysis";
      state.analysis.status = "loading";
      clearError(analysisError);
      render();
      try {
        const response = await fetch(analyzeEndpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "same-origin",
          body: JSON.stringify({
            fen: state.completedPosition.fen,
            top_n: 3,
          }),
        });
        const payload = await response.json();
        if (!response.ok || payload.status !== "success") {
          showError(analysisError, payload.detail || "Unable to run engine analysis.");
          state.analysis.status = "failed";
          render();
          return;
        }
        state.analysis = {
          status: "success",
          result: payload.analysis,
          activeLineIndex: 0,
          stepIndex: 0,
          flipped: false,
        };
        render();
      } catch (_error) {
        showError(analysisError, "Unable to run engine analysis right now.");
        state.analysis.status = "failed";
        render();
      }
    }

    function renderLineList() {
      const analysis = state.analysis;
      const topMoves = analysis.result.top_moves || [];
      lineList.innerHTML = "";
      topMoves.forEach((move, index) => {
        const button = document.createElement("button");
        button.className = "line-card";
        if (index === analysis.activeLineIndex) {
          button.classList.add("active");
        }
        const previewMoves = [move.move_san].concat(move.continuation || []).join(" ");
        button.innerHTML = `
          <div class="line-card-head">
            <span class="line-card-move">${index + 1}. ${move.move_san}</span>
            <span class="line-card-score">${move.score_display}</span>
          </div>
          <p class="line-card-preview">${previewMoves}</p>
        `;
        button.addEventListener("click", () => {
          state.analysis.activeLineIndex = index;
          state.analysis.stepIndex = 0;
          render();
        });
        lineList.appendChild(button);
      });
    }

    function renderBoard() {
      const currentFen = currentPlaybackFen(state);
      const orientation = state.analysis.flipped ? "black" : "white";
      const showNotation = loadSettings().showCoordinates;
      const config = {
        draggable: false,
        position: currentFen,
        orientation,
        showNotation,
        pieceTheme: PIECE_THEME_URL,
      };
      if (boardWidget === null || boardShowNotation !== showNotation) {
        destroyBoard();
        boardWidget = window.Chessboard(analysisBoardElement, config);
        boardShowNotation = showNotation;
      } else {
        boardWidget.orientation(orientation);
        boardWidget.position(currentFen, false);
      }
    }

    function destroyBoard() {
      if (boardWidget && typeof boardWidget.destroy === "function") {
        boardWidget.destroy();
      }
      boardWidget = null;
      boardShowNotation = null;
      if (analysisBoardElement) {
        analysisBoardElement.innerHTML = "";
      }
    }

    function renderPlaybackControls() {
      const moves = currentLineMoves(state);
      analysisPrevButton.disabled = state.analysis.stepIndex === 0;
      analysisNextButton.disabled = state.analysis.stepIndex >= moves.length;
      analysisResetButton.disabled = state.analysis.stepIndex === 0;
      analysisStepNote.textContent = `Step ${state.analysis.stepIndex} of ${moves.length}`;
    }

    function renderArrow() {
      const arrowMove = currentArrowMove(state);
      if (!arrowMove || !boardWidget) {
        analysisArrow.hidden = true;
        analysisArrowLayer.hidden = true;
        return;
      }
      const fromSquare = analysisBoardElement.querySelector(`.square-${arrowMove.from}`);
      const toSquare = analysisBoardElement.querySelector(`.square-${arrowMove.to}`);
      if (!fromSquare || !toSquare) {
        analysisArrow.hidden = true;
        analysisArrowLayer.hidden = true;
        return;
      }
      const boardRect = analysisBoardElement.getBoundingClientRect();
      const fromRect = fromSquare.getBoundingClientRect();
      const toRect = toSquare.getBoundingClientRect();
      analysisArrowLayer.setAttribute(
        "viewBox",
        `0 0 ${boardRect.width} ${boardRect.height}`
      );
      analysisArrow.setAttribute(
        "x1",
        String(fromRect.left - boardRect.left + fromRect.width / 2)
      );
      analysisArrow.setAttribute(
        "y1",
        String(fromRect.top - boardRect.top + fromRect.height / 2)
      );
      analysisArrow.setAttribute(
        "x2",
        String(toRect.left - boardRect.left + toRect.width / 2)
      );
      analysisArrow.setAttribute(
        "y2",
        String(toRect.top - boardRect.top + toRect.height / 2)
      );
      analysisArrow.hidden = false;
      analysisArrowLayer.hidden = false;
    }

    function currentLineMoves(state) {
      if (!state.analysis.result) {
        return [];
      }
      const move = state.analysis.result.top_moves[state.analysis.activeLineIndex];
      if (!move) {
        return [];
      }
      return [move.move_san].concat(move.continuation || []);
    }

    function currentPlaybackFen(state) {
      if (!state.completedPosition) {
        return "start";
      }
      const chess = new window.Chess(state.completedPosition.fen);
      const moves = currentLineMoves(state);
      for (let index = 0; index < state.analysis.stepIndex; index += 1) {
        chess.move(moves[index], { sloppy: true });
      }
      return chess.fen();
    }

    function currentArrowMove(state) {
      if (!state.completedPosition || !state.analysis.result) {
        return null;
      }
      const moves = currentLineMoves(state);
      if (state.analysis.stepIndex >= moves.length) {
        return null;
      }
      const chess = new window.Chess(currentPlaybackFen(state));
      const next = chess.move(moves[state.analysis.stepIndex], { sloppy: true });
      if (!next) {
        return null;
      }
      return next;
    }

    imageInput.addEventListener("change", (event) => {
      handleFileSelection(event.currentTarget.files[0]);
    });
    cameraInput.addEventListener("change", (event) => {
      handleFileSelection(event.currentTarget.files[0]);
    });
    detectButton.addEventListener("click", runBoardDetection);
    resetImageButton.addEventListener("click", resetToUpload);
    root
      .querySelector("[data-back-to-upload-button]")
      .addEventListener("click", () => {
        state.step = "upload";
        render();
      });
    root
      .querySelector("[data-back-to-orientation-button]")
      .addEventListener("click", () => {
        state.step = "orientation";
        render();
      });
    orientationContinueButton.addEventListener("click", () => {
      state.step = "side";
      render();
    });
    flipButton.addEventListener("click", () => {
      state.flipped = !state.flipped;
      render();
    });
    completeButton.addEventListener("click", completePosition);
    continueToAnalysisButton.addEventListener("click", enterAnalysisMode);
    analysisFlipButton.addEventListener("click", () => {
      state.analysis.flipped = !state.analysis.flipped;
      render();
    });
    analysisPrevButton.addEventListener("click", () => {
      state.analysis.stepIndex = Math.max(0, state.analysis.stepIndex - 1);
      render();
    });
    analysisNextButton.addEventListener("click", () => {
      const maxIndex = currentLineMoves(state).length;
      state.analysis.stepIndex = Math.min(maxIndex, state.analysis.stepIndex + 1);
      render();
    });
    analysisResetButton.addEventListener("click", () => {
      state.analysis.stepIndex = 0;
      render();
    });
    analysisRetryButton.addEventListener("click", enterAnalysisMode);
    root.querySelector("[data-start-over-button]").addEventListener("click", resetToUpload);
    overlaySvg.addEventListener("click", handleOverlayClick);

    sideButtons.forEach((button) => {
      button.addEventListener("click", () => {
        state.sideToMove = button.dataset.sideOption;
        clearError(completeError);
        render();
      });
    });

    window.addEventListener("resize", () => {
      if (state.step === "analysis" && state.analysis.status === "success") {
        renderArrow();
      }
    });

    render();
  }

  function squareName(fileIndex, rankIndex) {
    const files = "abcdefgh";
    return `${files[fileIndex]}${8 - rankIndex}`;
  }

  function squarePolygonPoints(state) {
    const corners = state.detection.board_corners;
    const boardToImage = solveHomography(
      [
        [0, 0],
        [8, 0],
        [8, 8],
        [0, 8],
      ],
      corners
    );
    const [fileIndex, rankIndex] = squareIndicesFromName(state.selectedSquare);
    const canonical = [
      [fileIndex, rankIndex],
      [fileIndex + 1, rankIndex],
      [fileIndex + 1, rankIndex + 1],
      [fileIndex, rankIndex + 1],
    ];
    return canonical.map((point) => applyHomography(boardToImage, point));
  }

  function squareIndicesFromName(square) {
    const files = "abcdefgh";
    return [files.indexOf(square[0]), 8 - Number(square[1])];
  }

  function solveHomography(src, dst) {
    const matrix = [];
    const vector = [];
    for (let index = 0; index < 4; index += 1) {
      const [x, y] = src[index];
      const [u, v] = dst[index];
      matrix.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
      vector.push(u);
      matrix.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
      vector.push(v);
    }
    const solution = gaussianSolve(matrix, vector);
    return [
      [solution[0], solution[1], solution[2]],
      [solution[3], solution[4], solution[5]],
      [solution[6], solution[7], 1],
    ];
  }

  function gaussianSolve(matrix, vector) {
    const size = vector.length;
    const augmented = matrix.map((row, index) => row.concat(vector[index]));
    for (let pivot = 0; pivot < size; pivot += 1) {
      let maxRow = pivot;
      for (let row = pivot + 1; row < size; row += 1) {
        if (Math.abs(augmented[row][pivot]) > Math.abs(augmented[maxRow][pivot])) {
          maxRow = row;
        }
      }
      const temp = augmented[pivot];
      augmented[pivot] = augmented[maxRow];
      augmented[maxRow] = temp;
      const pivotValue = augmented[pivot][pivot];
      if (Math.abs(pivotValue) < 1e-9) {
        throw new Error("Cannot solve homography.");
      }
      for (let column = pivot; column <= size; column += 1) {
        augmented[pivot][column] /= pivotValue;
      }
      for (let row = 0; row < size; row += 1) {
        if (row === pivot) {
          continue;
        }
        const factor = augmented[row][pivot];
        for (let column = pivot; column <= size; column += 1) {
          augmented[row][column] -= factor * augmented[pivot][column];
        }
      }
    }
    return augmented.map((row) => row[size]);
  }

  function applyHomography(matrix, point) {
    const [x, y] = point;
    const denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2];
    return [
      (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denominator,
      (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denominator,
    ];
  }

  document.querySelectorAll("[data-auth-form]").forEach((form) => {
    form.addEventListener("submit", handleAuthFormSubmit);
  });

  document.querySelectorAll("[data-logout-button]").forEach((button) => {
    button.addEventListener("click", handleLogout);
  });

  setupSettingsForm();

  document.querySelectorAll("[data-analyze-app]").forEach((root) => {
    setupAnalyzeFlow(root);
  });
})();
