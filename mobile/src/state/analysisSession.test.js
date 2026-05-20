import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  branchWithAnalysis,
  createAnalysisSession,
  playSuggestedMove,
  previousStep,
  selectLine,
  resetSession
} from "./analysisSession.js";

const START =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

const startAnalysis = {
  top_moves: [
    {
      move_san: "e4",
      move_uci: "e2e4",
      continuation_uci: ["e7e5"]
    },
    {
      move_san: "d4",
      move_uci: "d2d4",
      continuation_uci: ["d7d5"]
    }
  ]
};

const branchAnalysis = {
  top_moves: [
    {
      move_san: "d5",
      move_uci: "d7d5",
      continuation_uci: []
    }
  ]
};

describe("analysis session", () => {
  it("treats a suggested move as line playback", () => {
    const session = createAnalysisSession({
      rootFen: START,
      analysis: startAnalysis
    });
    const next = playSuggestedMove(session, "e2e4");
    const continued = playSuggestedMove(next, "e7e5");

    assert.equal(next.stepIndex, 1);
    assert.equal(next.selectedLineIndex, 0);
    assert.equal(next.frames.length, 2);
    assert.match(next.currentFen, /4P3/);
    assert.equal(next.analysis, startAnalysis);
    assert.equal(continued.stepIndex, 2);
    assert.equal(continued.frames.length, 3);
    assert.match(continued.currentFen, /4p3/);
  });

  it("lets the selected line change without losing the current board", () => {
    const session = createAnalysisSession({
      rootFen: START,
      analysis: startAnalysis
    });
    const next = selectLine(session, 1);

    assert.equal(next.currentFen, START);
    assert.equal(next.selectedLineIndex, 1);
    assert.equal(next.stepIndex, 0);
  });

  it("restores branched play when stepping backward or resetting", () => {
    const session = createAnalysisSession({
      rootFen: START,
      analysis: startAnalysis
    });
    const branch = branchWithAnalysis(session, {
      moveUci: "e2e3",
      nextFen: "rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
      analysis: branchAnalysis
    });
    const continued = playSuggestedMove(branch, "d7d5");

    assert.equal(branch.stepIndex, 0);
    assert.equal(branch.frames.length, 2);
    assert.equal(continued.stepIndex, 1);
    assert.equal(continued.frames.length, 3);

    const previous = previousStep(continued);
    assert.equal(previous.stepIndex, 0);
    assert.equal(previous.currentFen, branch.currentFen);
    assert.equal(previous.analysis, branchAnalysis);

    const reset = resetSession(continued);
    assert.equal(reset.currentFen, START);
    assert.equal(reset.stepIndex, 0);
    assert.equal(reset.frames.length, 1);
    assert.equal(reset.analysis, startAnalysis);
  });
});
