import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  getContainedImageBounds,
  nextClockwiseCornerIndex,
  getPrecisionOverlayGeometry,
  getSourcePointFromContainedPoint,
  getSourcePointFromPrecisionTap
} from "./cornerPrecision.js";

describe("corner precision helpers", () => {
  it("maps contained image presses back to source coordinates", () => {
    const bounds = getContainedImageBounds(
      { width: 300, height: 300 },
      1200,
      600
    );

    assert.ok(bounds);
    const point = getSourcePointFromContainedPoint(150, 150, bounds, 1200, 600);

    assert.deepEqual(point, { x: 600, y: 300 });
  });

  it("returns null when the press falls outside the contained image", () => {
    const bounds = getContainedImageBounds(
      { width: 300, height: 300 },
      1200,
      600
    );

    assert.ok(bounds);
    const point = getSourcePointFromContainedPoint(10, 10, bounds, 1200, 600);

    assert.equal(point, null);
  });

  it("maps precision overlay taps back to the original image", () => {
    const geometry = getPrecisionOverlayGeometry({
      focusPoint: { x: 600, y: 300 },
      imageWidth: 1200,
      imageHeight: 600,
      viewportWidth: 240,
      viewportHeight: 240,
      zoomScale: 3
    });

    assert.ok(geometry);
    const point = getSourcePointFromPrecisionTap(
      120,
      120,
      geometry,
      1200,
      600
    );

    assert.deepEqual(point, { x: 600, y: 300 });
  });

  it("clamps precision taps to the image edges", () => {
    const geometry = getPrecisionOverlayGeometry({
      focusPoint: { x: 10, y: 10 },
      imageWidth: 1200,
      imageHeight: 600,
      viewportWidth: 240,
      viewportHeight: 240,
      zoomScale: 3
    });

    assert.ok(geometry);
    const point = getSourcePointFromPrecisionTap(0, 0, geometry, 1200, 600);

    assert.deepEqual(point, { x: 0, y: 0 });
  });

  it("returns null when precision taps fall outside the rendered image", () => {
    const geometry = getPrecisionOverlayGeometry({
      focusPoint: { x: 20, y: 20 },
      imageWidth: 40,
      imageHeight: 40,
      viewportWidth: 200,
      viewportHeight: 200,
      zoomScale: 2
    });

    assert.ok(geometry);
    const point = getSourcePointFromPrecisionTap(10, 10, geometry, 40, 40);

    assert.equal(point, null);
  });

  it("advances corners clockwise for the guided placement flow", () => {
    assert.equal(nextClockwiseCornerIndex(0), 1);
    assert.equal(nextClockwiseCornerIndex(1), 2);
    assert.equal(nextClockwiseCornerIndex(2), 3);
    assert.equal(nextClockwiseCornerIndex(3), 0);
  });
});
