import { squareName, squareToIndices } from "../utils/chess.js";

export const BOARD_SIZE = 8;

/**
 * Clamps a numeric value to an inclusive range.
 *
 * @param {number} value - Value to clamp.
 * @param {number} min - Lower bound.
 * @param {number} max - Upper bound.
 * @returns {number} The clamped value.
 */
export function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

/**
 * Linearly interpolates between two values.
 *
 * @param {number} start - Starting value.
 * @param {number} end - Ending value.
 * @param {number} fraction - Interpolation fraction.
 * @returns {number} The interpolated value.
 */
export function lerp(start, end, fraction) {
  return start + (end - start) * fraction;
}

/**
 * Linearly interpolates between two points.
 *
 * @param {{x: number, y: number}} start - Start point.
 * @param {{x: number, y: number}} end - End point.
 * @param {number} fraction - Interpolation fraction.
 * @returns {{x: number, y: number}} The interpolated point.
 */
export function lerpPoint(start, end, fraction) {
  return {
    x: lerp(start.x, end.x, fraction),
    y: lerp(start.y, end.y, fraction)
  };
}

/**
 * Scales a point from raw image coordinates into display coordinates.
 *
 * @param {{x: number, y: number}} point - Raw image point.
 * @param {number} imageWidth - Source image width.
 * @param {number} imageHeight - Source image height.
 * @param {number} displayWidth - Display width.
 * @param {number} displayHeight - Display height.
 * @returns {{x: number, y: number}} The scaled point.
 */
export function toDisplayPoint(
  point,
  imageWidth,
  imageHeight,
  displayWidth,
  displayHeight
) {
  return {
    x: (point.x / imageWidth) * displayWidth,
    y: (point.y / imageHeight) * displayHeight
  };
}

/**
 * Projects a point inside the detected board quadrilateral.
 *
 * Corners are ordered top-left, top-right, bottom-right, bottom-left.
 *
 * @param {Array<{x: number, y: number}>} corners - Board corner points.
 * @param {number} fileFraction - Horizontal fraction across the board.
 * @param {number} rankFraction - Vertical fraction across the board.
 * @returns {{x: number, y: number}} Projected point.
 */
export function projectBoardPoint(corners, fileFraction, rankFraction) {
  const topEdge = lerpPoint(corners[0], corners[1], fileFraction);
  const bottomEdge = lerpPoint(corners[3], corners[2], fileFraction);
  return lerpPoint(topEdge, bottomEdge, rankFraction);
}

/**
 * Returns the raw image center point for a board square.
 *
 * @param {Array<{x: number, y: number}>} corners - Board corner points.
 * @param {string} square - Square name such as `e1`.
 * @returns {{x: number, y: number}} Raw image center for the square.
 */
export function getSquareCenterPoint(corners, square) {
  const [row, file] = squareToIndices(square);
  return projectBoardPoint(
    corners,
    (file + 0.5) / BOARD_SIZE,
    (row + 0.5) / BOARD_SIZE
  );
}

/**
 * Returns the raw image polygon for a board square.
 *
 * @param {Array<{x: number, y: number}>} corners - Board corner points.
 * @param {string} square - Square name such as `e1`.
 * @returns {Array<{x: number, y: number}>} Square polygon points.
 */
export function getSquarePolygon(corners, square) {
  const [row, file] = squareToIndices(square);
  return [
    projectBoardPoint(corners, file / BOARD_SIZE, row / BOARD_SIZE),
    projectBoardPoint(corners, (file + 1) / BOARD_SIZE, row / BOARD_SIZE),
    projectBoardPoint(
      corners,
      (file + 1) / BOARD_SIZE,
      (row + 1) / BOARD_SIZE
    ),
    projectBoardPoint(corners, file / BOARD_SIZE, (row + 1) / BOARD_SIZE)
  ];
}

/**
 * Returns the display bounds of the detected board.
 *
 * @param {Array<{x: number, y: number}>} corners - Board corner points.
 * @param {number} imageWidth - Source image width.
 * @param {number} imageHeight - Source image height.
 * @param {number} displayWidth - Display width.
 * @param {number} displayHeight - Display height.
 * @returns {{left: number, top: number, width: number, height: number}}
 *   Board bounds in display coordinates.
 */
export function getBoardDisplayBounds(
  corners,
  imageWidth,
  imageHeight,
  displayWidth,
  displayHeight
) {
  const displayCorners = corners.map((corner) =>
    toDisplayPoint(corner, imageWidth, imageHeight, displayWidth, displayHeight)
  );
  const xs = displayCorners.map((corner) => corner.x);
  const ys = displayCorners.map((corner) => corner.y);
  const left = Math.min(...xs);
  const top = Math.min(...ys);
  const right = Math.max(...xs);
  const bottom = Math.max(...ys);
  return {
    left,
    top,
    width: right - left,
    height: bottom - top
  };
}

/**
 * Returns the square label nearest to a raw image point.
 *
 * @param {Array<{x: number, y: number}>} corners - Board corner points.
 * @param {{x: number, y: number}} point - Raw image point.
 * @returns {string} The nearest square name.
 */
export function getNearestSquare(corners, point) {
  let nearestSquare = "e1";
  let nearestDistance = Number.POSITIVE_INFINITY;

  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let file = 0; file < BOARD_SIZE; file += 1) {
      const square = squareName(file, row);
      const center = getSquareCenterPoint(corners, square);
      const distance = (center.x - point.x) ** 2 + (center.y - point.y) ** 2;
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestSquare = square;
      }
    }
  }

  return nearestSquare;
}

/**
 * Returns the square label for a point inside the board bounds.
 *
 * The mapping assumes a regular 8x8 board selection overlay.
 *
 * @param {{x: number, y: number, width: number, height: number}} bounds -
 *   Board bounds in display coordinates.
 * @param {number} locationX - Pointer x coordinate.
 * @param {number} locationY - Pointer y coordinate.
 * @returns {string | null} The square under the pointer or null.
 */
export function getSquareFromBoardLocation(bounds, locationX, locationY) {
  const relativeX = locationX - bounds.left;
  const relativeY = locationY - bounds.top;
  if (
    relativeX < 0 ||
    relativeY < 0 ||
    relativeX > bounds.width ||
    relativeY > bounds.height
  ) {
    return null;
  }

  const file = clamp(Math.floor((relativeX / bounds.width) * BOARD_SIZE), 0, 7);
  const row = clamp(Math.floor((relativeY / bounds.height) * BOARD_SIZE), 0, 7);
  return squareName(file, row);
}

/**
 * Returns the detected board square containing a raw image point.
 *
 * @param {Array<{x: number, y: number}>} corners - Board corner points.
 * @param {{x: number, y: number}} point - Raw image point.
 * @returns {string | null} The square containing the point or null.
 */
export function getSquareFromBoardPoint(corners, point) {
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let file = 0; file < BOARD_SIZE; file += 1) {
      const square = squareName(file, row);
      if (isPointInPolygon(point, getSquarePolygon(corners, square))) {
        return square;
      }
    }
  }

  return null;
}

function isPointInPolygon(point, polygon) {
  let hasPositive = false;
  let hasNegative = false;

  for (let index = 0; index < polygon.length; index += 1) {
    const start = polygon[index];
    const end = polygon[(index + 1) % polygon.length];
    const crossProduct =
      (end.x - start.x) * (point.y - start.y) -
      (end.y - start.y) * (point.x - start.x);

    if (crossProduct > 0) {
      hasPositive = true;
    } else if (crossProduct < 0) {
      hasNegative = true;
    }

    if (hasPositive && hasNegative) {
      return false;
    }
  }

  return true;
}
