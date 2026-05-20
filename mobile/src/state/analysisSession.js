import { applyUciMoveToFen } from "../utils/chess.js";

function currentFrame(session) {
  if (!session.frames || session.frames.length === 0) {
    return null;
  }
  return session.frames[session.frames.length - 1];
}

function sessionFromFrames(rootFen, frames) {
  if (!frames || frames.length === 0) {
    return {
      rootFen,
      currentFen: rootFen,
      analysis: null,
      selectedLineIndex: 0,
      stepIndex: 0,
      frames: []
    };
  }
  const current = frames[frames.length - 1];
  return {
    rootFen,
    currentFen: current.fen,
    analysis: current.analysis,
    selectedLineIndex: current.selectedLineIndex,
    stepIndex: current.stepIndex,
    frames
  };
}

function createFrame({ fen, analysis, selectedLineIndex = 0, stepIndex = 0 }) {
  return {
    fen,
    analysis,
    selectedLineIndex,
    stepIndex
  };
}

export function createAnalysisSession({ rootFen, analysis }) {
  return sessionFromFrames(
    rootFen,
    [createFrame({ fen: rootFen, analysis })]
  );
}

export function selectLine(session, selectedLineIndex) {
  const frame = currentFrame(session);
  if (!frame) {
    return session;
  }
  const nextFrames = session.frames.slice(0, -1).concat([
    createFrame({
      fen: frame.fen,
      analysis: frame.analysis,
      selectedLineIndex,
      stepIndex: 0
    })
  ]);
  return sessionFromFrames(session.rootFen, nextFrames);
}

export function currentLineMoves(session) {
  const frame = currentFrame(session);
  if (!frame) {
    return [];
  }
  const topMoves = frame.analysis?.top_moves || [];
  const line = topMoves[frame.selectedLineIndex];
  if (!line) {
    return [];
  }
  return [line.move_uci].concat(line.continuation_uci || []);
}

export function playbackFen(session) {
  return currentFrame(session)?.fen || session.rootFen;
}

export function suggestedMoveIndex(session, moveUci) {
  const frame = currentFrame(session);
  if (!frame) {
    return -1;
  }
  const currentMoves = currentLineMoves(session);
  if (currentMoves[frame.stepIndex] === moveUci) {
    return frame.selectedLineIndex;
  }
  if (frame.stepIndex !== 0) {
    return -1;
  }
  return (frame.analysis?.top_moves || []).findIndex(
    (move) => move.move_uci === moveUci
  );
}

export function playSuggestedMove(session, moveUci, nextFen = null) {
  const matchIndex = suggestedMoveIndex(session, moveUci);
  if (matchIndex < 0) {
    return session;
  }
  const frame = currentFrame(session);
  if (!frame) {
    return session;
  }
  const playedFrame = createFrame({
    fen: nextFen || applyUciMoveToFen(frame.fen, moveUci),
    analysis: frame.analysis,
    selectedLineIndex: matchIndex,
    stepIndex: frame.stepIndex + 1
  });
  return sessionFromFrames(session.rootFen, session.frames.concat([playedFrame]));
}

export function branchWithAnalysis(session, { moveUci, nextFen, analysis }) {
  const frame = currentFrame(session);
  if (!frame) {
    return session;
  }
  const nextFrame = createFrame({
    fen: nextFen || applyUciMoveToFen(frame.fen, moveUci),
    analysis,
    selectedLineIndex: 0,
    stepIndex: 0
  });
  return sessionFromFrames(session.rootFen, session.frames.concat([nextFrame]));
}

export function previousStep(session) {
  if (!session.frames || session.frames.length <= 1) {
    return session;
  }
  return sessionFromFrames(session.rootFen, session.frames.slice(0, -1));
}

export function resetSession(session) {
  if (!session.frames || session.frames.length === 0) {
    return session;
  }
  return sessionFromFrames(session.rootFen, session.frames.slice(0, 1));
}

export function moveToStep(session, stepIndex) {
  if (!session.frames || session.frames.length === 0) {
    return session;
  }
  if (stepIndex <= 0) {
    return resetSession(session);
  }
  if (stepIndex >= session.frames.length - 1) {
    return session;
  }
  return sessionFromFrames(session.rootFen, session.frames.slice(0, stepIndex + 1));
}
