import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  getBoardDisplayBounds,
  getNearestSquare,
  getSquareCenterPoint,
  getSquareFromBoardPoint,
  getSquareFromBoardLocation
} from "./BoardGeometry.js";

describe("board geometry", () => {
  const corners = [
    { x: 0, y: 0 },
    { x: 80, y: 0 },
    { x: 80, y: 80 },
    { x: 0, y: 80 }
  ];

  it("projects the e1 square center into the raw image", () => {
    assert.deepEqual(getSquareCenterPoint(corners, "e1"), { x: 45, y: 75 });
  });

  it("finds the nearest square for a raw point", () => {
    assert.equal(getNearestSquare(corners, { x: 45, y: 75 }), "e1");
  });

  it("maps touch coordinates into the correct square", () => {
    const bounds = getBoardDisplayBounds(corners, 80, 80, 160, 160);
    assert.equal(getSquareFromBoardLocation(bounds, 90, 150), "e1");
  });

  it("maps raw image points into detected board squares", () => {
    assert.equal(getSquareFromBoardPoint(corners, { x: 45, y: 75 }), "e1");
  });

  it("returns null for raw image points outside the detected board", () => {
    assert.equal(getSquareFromBoardPoint(corners, { x: 90, y: 75 }), null);
  });
});
