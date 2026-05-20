import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, it } from "node:test";

import { ANALYSIS_ARROW_OPACITY, isPawnPiece } from "./chess.js";

const chessBoardSource = readFileSync(
  new URL("../components/ChessBoard.js", import.meta.url),
  "utf8"
);
const chessPieceIconSource = readFileSync(
  new URL("../components/ChessPieceIcon.js", import.meta.url),
  "utf8"
);

describe("chess glyphs", () => {
  it("uses the downloaded Cburnett piece images for board rendering", () => {
    assert.match(chessPieceIconSource, /wK\.png/);
    assert.match(chessPieceIconSource, /wQ\.png/);
    assert.match(chessPieceIconSource, /wR\.png/);
    assert.match(chessPieceIconSource, /wB\.png/);
    assert.match(chessPieceIconSource, /wN\.png/);
    assert.match(chessPieceIconSource, /wP\.png/);
    assert.match(chessPieceIconSource, /bK\.png/);
    assert.match(chessPieceIconSource, /bQ\.png/);
    assert.match(chessPieceIconSource, /bR\.png/);
    assert.match(chessPieceIconSource, /bB\.png/);
    assert.match(chessPieceIconSource, /bN\.png/);
    assert.match(chessPieceIconSource, /bP\.png/);
  });
});

describe("chess helpers", () => {
  it("detects pawn pieces for the board fallback", () => {
    assert.equal(isPawnPiece("P"), true);
    assert.equal(isPawnPiece("p"), true);
    assert.equal(isPawnPiece("Q"), false);
  });

  it("uses 70% opacity for the analysis arrow", () => {
    assert.equal(ANALYSIS_ARROW_OPACITY, 0.7);
  });
});

describe("mobile board rendering regressions", () => {
  it("renders board pieces with the shared image piece set", () => {
    assert.match(chessBoardSource, /import \{ ChessPieceIcon \} from "\.\/ChessPieceIcon";/);
    assert.match(chessBoardSource, /return <ChessPieceIcon piece=\{piece\} size=\{34\} \/>;/);
    assert.match(chessPieceIconSource, /const PIECE_IMAGES = \{/);
    assert.match(chessPieceIconSource, /<Image/);
  });

  it("uses a slimmer rounded arrow that keeps the shaft visible", () => {
    assert.match(chessBoardSource, /const ARROW_HEAD_LENGTH = 14;/);
    assert.match(chessBoardSource, /const ARROW_HEAD_HALF_WIDTH = 8;/);
    assert.match(chessBoardSource, /const ARROW_SHAFT_HALF_WIDTH = 4\.5;/);
    assert.match(chessBoardSource, /<Circle/);
    assert.match(chessBoardSource, /points=\{head\.shaftPoints\}/);
    assert.match(chessBoardSource, /points=\{head\.headPoints\}/);
    assert.doesNotMatch(chessBoardSource, /<Line/);
  });
});
