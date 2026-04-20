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

  async function fetchSyncedSettings(endpoint) {
    const response = await fetch(endpoint, {
      method: "GET",
      credentials: "same-origin",
    });
    const payload = await response.json();
    if (!response.ok || payload.status !== "success") {
      throw new Error(payload.detail || "Unable to load settings.");
    }
    const settings = {
      showCoordinates: payload.settings.show_coordinates,
    };
    saveSettings(settings);
    return settings;
  }

  async function persistSyncedSettings(endpoint, settings) {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({
        show_coordinates: settings.showCoordinates,
      }),
    });
    const payload = await response.json();
    if (!response.ok || payload.status !== "success") {
      throw new Error(payload.detail || "Unable to save settings.");
    }
    const nextSettings = {
      showCoordinates: payload.settings.show_coordinates,
    };
    saveSettings(nextSettings);
    return nextSettings;
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

  function createExplanationState() {
    return {
      status: "idle",
      mode: null,
      playedMove: {
        from: "",
        to: "",
        promotion: "",
      },
      result: null,
      warnings: [],
    };
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
      savedSnapshotId: null,
      analysis: {
        status: "idle",
        result: null,
        activeLineIndex: 0,
        stepIndex: 0,
        flipped: false,
      },
      explanation: createExplanationState(),
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
      savedSnapshotId: state.savedSnapshotId,
      analysis: state.analysis,
      explanation: state.explanation,
    };
    window.sessionStorage.setItem(ANALYZE_STORAGE_KEY, JSON.stringify(payload));
  }

  function clearAnalyzeStateStorage() {
    window.sessionStorage.removeItem(ANALYZE_STORAGE_KEY);
  }

  function setupProfileApp(root) {
    const settingsEndpoint = root.dataset.settingsEndpoint;
    const showCoordinatesInput = root.querySelector(
      '[data-setting-input="showCoordinates"]'
    );
    const statusTarget = root.querySelector("[data-settings-status]");
    if (!showCoordinatesInput || !settingsEndpoint) {
      return;
    }

    async function loadProfileSettings() {
      statusTarget.textContent = "Loading synced settings...";
      try {
        const settings = await fetchSyncedSettings(settingsEndpoint);
        showCoordinatesInput.checked = settings.showCoordinates;
        statusTarget.textContent =
          "Coordinates display affects the analysis board and syncs to your account.";
      } catch (_error) {
        const cached = loadSettings();
        showCoordinatesInput.checked = cached.showCoordinates;
        statusTarget.textContent =
          "Unable to load synced settings right now. Showing your local preference.";
      }
    }

    showCoordinatesInput.addEventListener("change", async () => {
      const nextSettings = {
        ...loadSettings(),
        showCoordinates: showCoordinatesInput.checked,
      };
      saveSettings(nextSettings);
      statusTarget.textContent = "Saving setting...";
      try {
        await persistSyncedSettings(settingsEndpoint, nextSettings);
        statusTarget.textContent =
          "Coordinates display affects the analysis board and syncs to your account.";
      } catch (_error) {
        statusTarget.textContent =
          "Unable to sync settings right now. Your local preference was kept.";
      }
    });

    loadProfileSettings();
  }

  function setupSavedApp(root) {
    const savedEndpoint = root.dataset.savedEndpoint;
    const openBase = root.dataset.openBase || "/app/analyze?saved=";
    const loading = root.querySelector("[data-saved-loading]");
    const error = root.querySelector("[data-saved-error]");
    const empty = root.querySelector("[data-saved-empty]");
    const list = root.querySelector("[data-saved-list]");

    async function loadSavedSnapshots() {
      loading.hidden = false;
      list.hidden = true;
      empty.hidden = true;
      clearError(error);
      try {
        const response = await fetch(savedEndpoint, {
          method: "GET",
          credentials: "same-origin",
        });
        const payload = await response.json();
        if (!response.ok || payload.status !== "success") {
          showError(error, payload.detail || "Unable to load saved positions.");
          return;
        }
        renderSavedList(payload.snapshots || []);
      } catch (_error) {
        showError(error, "Unable to load saved positions right now.");
      } finally {
        loading.hidden = true;
      }
    }

    function renderSavedList(snapshots) {
      list.innerHTML = "";
      if (snapshots.length === 0) {
        empty.hidden = false;
        list.hidden = true;
        return;
      }
      empty.hidden = true;
      list.hidden = false;
      snapshots.forEach((snapshot) => {
        const item = document.createElement("a");
        item.className = "saved-card";
        item.href = `${openBase}${snapshot.id}`;
        item.innerHTML = `
          <div class="saved-card-head">
            <div>
              <strong>${snapshot.best_move_san || "Saved Position"}</strong>
              <p>${snapshot.best_move_score_display || "No score available"}</p>
            </div>
            <span>${formatSavedDate(snapshot.updated_at)}</span>
          </div>
          <div class="saved-card-meta">
            <span>${snapshot.side_to_move === "w" ? "White" : "Black"} to move</span>
            <span>${snapshot.has_explanation ? "Explained" : "Analysis only"}</span>
            <span>${snapshot.has_coaching ? "Coached" : "No coaching"}</span>
          </div>
          <code>${snapshot.fen}</code>
        `;
        list.appendChild(item);
      });
    }

    loadSavedSnapshots();
  }

  function setupAnalyzeFlow(root) {
    const state = createAnalyzeState();
    const restored = restoreAnalyzeState();
    if (restored) {
      state.step = restored.step || state.step;
      state.completedPosition = restored.completedPosition || null;
      state.savedSnapshotId = restored.savedSnapshotId || null;
      state.analysis = {
        ...state.analysis,
        ...(restored.analysis || {}),
      };
      state.explanation = {
        ...state.explanation,
        ...(restored.explanation || {}),
        playedMove: {
          ...state.explanation.playedMove,
          ...((restored.explanation && restored.explanation.playedMove) || {}),
        },
      };
      if (!state.completedPosition && state.step !== "upload") {
        state.step = "upload";
      }
    }

    const detectEndpoint = root.dataset.detectEndpoint;
    const visionEndpoint = root.dataset.visionEndpoint;
    const completeEndpoint = root.dataset.completeEndpoint;
    const analyzeEndpoint = root.dataset.analyzeEndpoint;
    const explainEndpoint = root.dataset.explainEndpoint;
    const saveEndpoint = root.dataset.saveEndpoint;
    const settingsEndpoint = root.dataset.settingsEndpoint;
    const savedEndpointBase = root.dataset.savedEndpointBase;
    const requestedSavedId = new URLSearchParams(window.location.search).get("saved");

    const stepSections = Array.from(root.querySelectorAll("[data-step]"));
    const stepPills = Array.from(root.querySelectorAll("[data-step-pill]"));
    const imageInput = root.querySelector("[data-image-input]");
    const cameraInput = root.querySelector("[data-camera-input]");
    const uploadError = root.querySelector("[data-upload-error]");
    const detectError = root.querySelector("[data-detect-error]");
    const completeError = root.querySelector("[data-complete-error]");
    const analysisError = root.querySelector("[data-analysis-error]");
    const saveFeedback = root.querySelector("[data-save-feedback]");
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
    const saveButton = root.querySelector("[data-save-button]");
    const deleteSavedButton = root.querySelector("[data-delete-saved-button]");
    const insightsPanel = root.querySelector("[data-insights-panel]");
    const explainBestButton = root.querySelector("[data-explain-best-button]");
    const playedMoveForm = root.querySelector("[data-played-move-form]");
    const playedFromSelect = root.querySelector("[data-played-from-select]");
    const playedToSelect = root.querySelector("[data-played-to-select]");
    const playedPromotionSelect = root.querySelector(
      "[data-played-promotion-select]"
    );
    const compareMoveButton = root.querySelector("[data-compare-move-button]");
    const explanationLoading = root.querySelector("[data-explanation-loading]");
    const explanationError = root.querySelector("[data-explanation-error]");
    const explanationWarnings = root.querySelector("[data-explanation-warnings]");
    const explanationResult = root.querySelector("[data-explanation-result]");
    const explanationSummaryBlock = root.querySelector(
      "[data-explanation-summary-block]"
    );
    const explanationMoveLabel = root.querySelector("[data-explanation-move-label]");
    const explanationText = root.querySelector("[data-explanation-text]");
    const playedMoveResult = root.querySelector("[data-played-move-result]");
    const playedMoveLabel = root.querySelector("[data-played-move-label]");
    const playedMoveQuality = root.querySelector("[data-played-move-quality]");
    const playedMoveLoss = root.querySelector("[data-played-move-loss]");
    const bestMoveComparison = root.querySelector("[data-best-move-comparison]");
    const comparisonSummary = root.querySelector("[data-comparison-summary]");
    const structuredDetails = root.querySelector("[data-structured-details]");
    const structuredContent = root.querySelector("[data-structured-content]");

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

      saveButton.hidden = state.savedSnapshotId !== null;
      deleteSavedButton.hidden = state.savedSnapshotId === null;
      playedFromSelect.value = state.explanation.playedMove.from;
      playedToSelect.value = state.explanation.playedMove.to;
      playedPromotionSelect.value = state.explanation.playedMove.promotion;

      renderAnalysisState();
      renderExplanationState();
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

    function renderExplanationState() {
      const explanation = state.explanation;
      const hasAnalysis = state.analysis.status === "success" && !!state.analysis.result;
      insightsPanel.hidden = !hasAnalysis;
      explanationLoading.hidden = explanation.status !== "loading";
      explanationResult.hidden =
        explanation.status !== "success" || explanation.result === null;
      if (explanation.status !== "failed") {
        clearError(explanationError);
      }

      explainBestButton.disabled = explanation.status === "loading";
      compareMoveButton.disabled =
        explanation.status === "loading" || playedMoveUci() === null;

      renderWarnings(explanationWarnings, explanation.warnings || []);

      if (explanation.status !== "success" || !explanation.result) {
        explanationSummaryBlock.hidden = true;
        playedMoveResult.hidden = true;
        bestMoveComparison.hidden = true;
        structuredDetails.hidden = true;
        return;
      }

      const result = explanation.result;
      const moveLabel =
        result.move_san && result.move_uci
          ? `${result.move_san} (${result.move_uci})`
          : result.move_san || result.move_uci || "Best move";
      explanationSummaryBlock.hidden = false;
      explanationMoveLabel.textContent = moveLabel;

      if (result.explanation_text) {
        explanationText.hidden = false;
        explanationText.textContent = result.explanation_text;
      } else {
        explanationText.hidden = true;
        explanationText.textContent = "";
      }

      if (result.played_move_result) {
        playedMoveResult.hidden = false;
        playedMoveLabel.textContent =
          `${result.played_move_result.move_san} (${result.played_move_result.move_uci})`;
        playedMoveQuality.textContent =
          `${result.played_move_result.quality_emoji} ` +
          `${result.played_move_result.quality_label}`;
        playedMoveLoss.textContent = String(result.played_move_result.cp_loss);
      } else {
        playedMoveResult.hidden = true;
      }

      if (result.comparison) {
        bestMoveComparison.hidden = false;
        comparisonSummary.textContent =
          `${result.comparison.best_move_san} ` +
          `(${result.comparison.best_move_score_display}) was stronger than ` +
          `${result.comparison.played_move_san}. ` +
          `${result.comparison.why_best_move_is_better}`;
      } else {
        bestMoveComparison.hidden = true;
        comparisonSummary.textContent = "";
      }

      renderStructuredExplanation(result.structured_explanation);
    }

    function renderWarnings(target, warnings) {
      target.innerHTML = "";
      target.hidden = warnings.length === 0;
      warnings.forEach((warning) => {
        const item = document.createElement("div");
        item.className = "warning-card";
        item.innerHTML = `<strong>${warning.code}</strong><p>${warning.message}</p>`;
        target.appendChild(item);
      });
    }

    function renderStructuredExplanation(structured) {
      structuredContent.innerHTML = "";
      structuredDetails.open = false;
      if (!structured) {
        structuredDetails.hidden = true;
        return;
      }
      structuredDetails.hidden = false;

      appendStructuredSection("Summary", structured.summary);
      if ("what_the_move_does" in structured) {
        appendStructuredSection("What The Move Does", structured.what_the_move_does);
        appendStructuredSection("What It Threatens", structured.what_it_threatens);
        appendStructuredSection("Why It Is Best", structured.why_it_is_best);
        appendStructuredSection(
          "Why Alternatives Are Worse",
          structured.why_alternatives_are_worse
        );
      } else {
        appendStructuredSection(
          "What The Move Tried To Do",
          structured.what_the_move_tried_to_do
        );
        appendStructuredSection("What Was Missed", structured.what_was_missed);
        appendStructuredSection(
          "What Changed After Move",
          structured.what_changed_after_move
        );
        appendStructuredSection(
          "Why Best Move Was Better",
          structured.why_best_move_was_better
        );
        appendStructuredSection("Practical Lesson", structured.practical_lesson);
      }

      if (structured.tactical_themes && structured.tactical_themes.length > 0) {
        appendStructuredList("Tactical Themes", structured.tactical_themes);
      }
      if (structured.alternatives && structured.alternatives.length > 0) {
        appendStructuredAlternatives("Alternatives", structured.alternatives);
      }
    }

    function appendStructuredSection(title, body) {
      const section = document.createElement("section");
      section.className = "structured-section";
      section.innerHTML = `<h4>${title}</h4><p class="insight-text">${body}</p>`;
      structuredContent.appendChild(section);
    }

    function appendStructuredList(title, items) {
      const section = document.createElement("section");
      section.className = "structured-section";
      const heading = document.createElement("h4");
      heading.textContent = title;
      const list = document.createElement("ul");
      list.className = "feature-list structured-list";
      items.forEach((item) => {
        const listItem = document.createElement("li");
        listItem.textContent = item;
        list.appendChild(listItem);
      });
      section.appendChild(heading);
      section.appendChild(list);
      structuredContent.appendChild(section);
    }

    function appendStructuredAlternatives(title, alternatives) {
      const section = document.createElement("section");
      section.className = "structured-section";
      const heading = document.createElement("h4");
      heading.textContent = title;
      section.appendChild(heading);
      alternatives.forEach((alternative) => {
        const card = document.createElement("div");
        card.className = "alternative-card";
        card.innerHTML = `
          <div class="line-card-head">
            <span class="line-card-move">${alternative.move_san}</span>
            <span class="line-card-score">${alternative.score_display}</span>
          </div>
          <p class="insight-text">${alternative.reason}</p>
        `;
        section.appendChild(card);
      });
      structuredContent.appendChild(section);
    }

    function resetToUpload() {
      const nextState = createAnalyzeState();
      state.step = nextState.step;
      state.imageFile = nextState.imageFile;
      state.imageDataUrl = nextState.imageDataUrl;
      state.detection = nextState.detection;
      state.selectedClick = nextState.selectedClick;
      state.selectedSquare = nextState.selectedSquare;
      state.sideToMove = nextState.sideToMove;
      state.completedPosition = nextState.completedPosition;
      state.flipped = nextState.flipped;
      state.savedSnapshotId = nextState.savedSnapshotId;
      state.analysis = nextState.analysis;
      state.explanation = nextState.explanation;
      clearError(uploadError);
      clearError(detectError);
      clearError(completeError);
      clearError(analysisError);
      clearError(explanationError);
      clearError(saveFeedback);
      if (imageInput) {
        imageInput.value = "";
      }
      if (cameraInput) {
        cameraInput.value = "";
      }
      if (requestedSavedId) {
        window.history.replaceState({}, "", "/app/analyze");
      }
      clearAnalyzeStateStorage();
      destroyBoard();
      render();
    }

    function handleFileSelection(file) {
      clearError(uploadError);
      clearError(saveFeedback);
      if (!file) {
        return;
      }
      if (!file.type.startsWith("image/")) {
        showError(uploadError, "Please choose an image file.");
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        const nextState = createAnalyzeState();
        state.imageFile = file;
        state.imageDataUrl = String(reader.result || "");
        state.detection = nextState.detection;
        state.selectedClick = nextState.selectedClick;
        state.selectedSquare = nextState.selectedSquare;
        state.sideToMove = nextState.sideToMove;
        state.completedPosition = nextState.completedPosition;
        state.flipped = nextState.flipped;
        state.savedSnapshotId = nextState.savedSnapshotId;
        state.analysis = nextState.analysis;
        state.explanation = nextState.explanation;
        state.step = "upload";
        render();
      };
      reader.readAsDataURL(file);
    }

    async function runBoardDetection() {
      clearError(detectError);
      clearError(saveFeedback);
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
      clearError(saveFeedback);
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
        state.savedSnapshotId = null;
        state.analysis = createAnalyzeState().analysis;
        state.explanation = createExplanationState();
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
      if (
        typeof window.Chessboard === "undefined" ||
        typeof window.Chess === "undefined"
      ) {
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
      state.savedSnapshotId = null;
      state.analysis.status = "loading";
      clearError(analysisError);
      clearError(saveFeedback);
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
        state.explanation = createExplanationState();
        render();
      } catch (_error) {
        showError(analysisError, "Unable to run engine analysis right now.");
        state.analysis.status = "failed";
        render();
      }
    }

    async function requestExplanation(mode) {
      if (!state.completedPosition) {
        return;
      }
      const playedMove = mode === "played_move" ? playedMoveUci() : null;
      if (mode === "played_move" && playedMove === null) {
        showError(
          explanationError,
          "Choose the played move squares before requesting coaching."
        );
        return;
      }

      state.explanation.status = "loading";
      state.explanation.mode = mode;
      state.explanation.result = null;
      state.explanation.warnings = [];
      clearError(explanationError);
      render();
      try {
        const response = await fetch(explainEndpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "same-origin",
          body: JSON.stringify({
            fen: state.completedPosition.fen,
            played_move_uci: playedMove,
            top_n: 3,
          }),
        });
        const payload = await response.json();
        if (
          !response.ok ||
          (payload.status !== "success" && payload.status !== "skipped")
        ) {
          showError(
            explanationError,
            payload.detail || "Unable to generate insight right now."
          );
          state.explanation.status = "failed";
          render();
          return;
        }

        state.explanation.status = "success";
        state.explanation.result = payload.explanation;
        state.explanation.warnings = payload.warnings || [];
        render();
      } catch (_error) {
        showError(
          explanationError,
          "Unable to generate insight right now. Please try again."
        );
        state.explanation.status = "failed";
        render();
      }
    }

    async function saveCurrentSnapshot() {
      clearError(saveFeedback);
      if (!state.completedPosition || !state.analysis.result) {
        showError(saveFeedback, "Analyze a position before saving it.");
        return;
      }
      saveButton.disabled = true;
      saveButton.textContent = "Saving...";
      try {
        const response = await fetch(saveEndpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "same-origin",
          body: JSON.stringify({
            snapshot: {
              position: state.completedPosition,
              analysis: state.analysis.result,
              explanation:
                state.explanation.status === "success"
                  ? state.explanation.result
                  : null,
              explanation_warnings:
                state.explanation.status === "success"
                  ? state.explanation.warnings
                  : [],
              played_move_selection: state.explanation.playedMove,
            },
          }),
        });
        const payload = await response.json();
        if (!response.ok || payload.status !== "success") {
          showError(saveFeedback, payload.detail || "Unable to save this position.");
          return;
        }
        state.savedSnapshotId = payload.snapshot.id;
        saveFeedback.hidden = false;
        saveFeedback.textContent = "Position saved.";
        render();
      } catch (_error) {
        showError(saveFeedback, "Unable to save this position right now.");
      } finally {
        saveButton.disabled = false;
        saveButton.textContent = "Save Position";
      }
    }

    async function deleteCurrentSnapshot() {
      if (state.savedSnapshotId === null) {
        return;
      }
      if (!window.confirm("Delete this saved position?")) {
        return;
      }
      deleteSavedButton.disabled = true;
      deleteSavedButton.textContent = "Deleting...";
      clearError(saveFeedback);
      try {
        const response = await fetch(
          `${savedEndpointBase}/${state.savedSnapshotId}`,
          {
            method: "DELETE",
            credentials: "same-origin",
          }
        );
        const payload = await response.json();
        if (!response.ok || payload.status !== "success") {
          showError(
            saveFeedback,
            payload.detail || "Unable to delete this saved position."
          );
          return;
        }
        clearAnalyzeStateStorage();
        window.location.href = "/app/saved";
      } catch (_error) {
        showError(saveFeedback, "Unable to delete this saved position right now.");
      } finally {
        deleteSavedButton.disabled = false;
        deleteSavedButton.textContent = "Delete Saved Position";
      }
    }

    async function loadSavedSnapshot(snapshotId) {
      state.step = "analysis";
      state.analysis.status = "loading";
      clearError(analysisError);
      clearError(saveFeedback);
      render();
      try {
        const response = await fetch(`${savedEndpointBase}/${snapshotId}`, {
          method: "GET",
          credentials: "same-origin",
        });
        const payload = await response.json();
        if (!response.ok || payload.status !== "success") {
          showError(
            analysisError,
            payload.detail || "Unable to load the saved position."
          );
          state.analysis.status = "failed";
          render();
          return;
        }
        hydrateFromSavedSnapshot(payload.snapshot);
      } catch (_error) {
        showError(analysisError, "Unable to load the saved position right now.");
        state.analysis.status = "failed";
        render();
      }
    }

    function hydrateFromSavedSnapshot(record) {
      const snapshot = record.snapshot;
      state.step = "analysis";
      state.savedSnapshotId = record.id;
      state.completedPosition = snapshot.position || null;
      state.analysis = {
        status: "success",
        result: snapshot.analysis,
        activeLineIndex: 0,
        stepIndex: 0,
        flipped: false,
      };
      state.explanation = createExplanationState();
      if (snapshot.played_move_selection) {
        state.explanation.playedMove = {
          ...state.explanation.playedMove,
          ...snapshot.played_move_selection,
        };
      }
      if (snapshot.explanation) {
        state.explanation.status = "success";
        state.explanation.result = snapshot.explanation;
        state.explanation.warnings = snapshot.explanation_warnings || [];
      }
      render();
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

    function playedMoveUci() {
      const { from, to, promotion } = state.explanation.playedMove;
      if (!from || !to) {
        return null;
      }
      return `${from}${to}${promotion || ""}`;
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
    saveButton.addEventListener("click", saveCurrentSnapshot);
    deleteSavedButton.addEventListener("click", deleteCurrentSnapshot);
    explainBestButton.addEventListener("click", () => {
      requestExplanation("best_move");
    });
    playedMoveForm.addEventListener("submit", (event) => {
      event.preventDefault();
      requestExplanation("played_move");
    });
    playedFromSelect.addEventListener("change", () => {
      state.explanation.playedMove.from = playedFromSelect.value;
      clearError(explanationError);
      render();
    });
    playedToSelect.addEventListener("change", () => {
      state.explanation.playedMove.to = playedToSelect.value;
      clearError(explanationError);
      render();
    });
    playedPromotionSelect.addEventListener("change", () => {
      state.explanation.playedMove.promotion = playedPromotionSelect.value;
      clearError(explanationError);
      render();
    });
    root
      .querySelector("[data-start-over-button]")
      .addEventListener("click", resetToUpload);
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

    populateSquareSelect(playedFromSelect);
    populateSquareSelect(playedToSelect);

    fetchSyncedSettings(settingsEndpoint)
      .catch(() => loadSettings())
      .finally(() => {
        if (requestedSavedId) {
          loadSavedSnapshot(requestedSavedId);
          return;
        }
        render();
      });
  }

  function populateSquareSelect(select) {
    squareOptions().forEach((square) => {
      const option = document.createElement("option");
      option.value = square;
      option.textContent = square;
      select.appendChild(option);
    });
  }

  function squareOptions() {
    const files = "abcdefgh";
    const squares = [];
    for (let rank = 8; rank >= 1; rank -= 1) {
      for (let fileIndex = 0; fileIndex < files.length; fileIndex += 1) {
        squares.push(`${files[fileIndex]}${rank}`);
      }
    }
    return squares;
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

  function formatSavedDate(isoText) {
    try {
      return new Date(isoText).toLocaleString();
    } catch (_error) {
      return isoText;
    }
  }

  document.querySelectorAll("[data-auth-form]").forEach((form) => {
    form.addEventListener("submit", handleAuthFormSubmit);
  });

  document.querySelectorAll("[data-logout-button]").forEach((button) => {
    button.addEventListener("click", handleLogout);
  });

  document.querySelectorAll("[data-profile-app]").forEach((root) => {
    setupProfileApp(root);
  });

  document.querySelectorAll("[data-saved-app]").forEach((root) => {
    setupSavedApp(root);
  });

  document.querySelectorAll("[data-analyze-app]").forEach((root) => {
    setupAnalyzeFlow(root);
  });
})();
