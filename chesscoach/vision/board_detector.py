"""Detect a chessboard in an image and warp it to a top-down 512×512 view.

Detection strategy (in order of preference):
1. OpenCV ``findChessboardCorners`` — fast, very reliable on rendered/digital boards.
2. RANSAC grid fitting — fits all 9 horizontal + 9 vertical lines of the board
   grid, then derives a robust homography. Handles physical boards with
   perspective distortion, shadows, and background clutter.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

BOARD_SIZE = 512
_SQUARE_SIZE = BOARD_SIZE // 8

# Number of grid lines in each direction (8 squares → 9 lines)
_GRID_LINES = 9
LOGGER = logging.getLogger(__name__)


class BoardNotFoundError(Exception):
    """Raised when no chessboard can be detected in the image."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_board(image: np.ndarray) -> np.ndarray:
    """Detect a chessboard in *image* and return a warped 512×512 top-down view.

    Args:
        image: BGR image as a numpy array (H×W×3, uint8).

    Returns:
        Warped 512×512 BGR array with the board filling the frame, rank 8 at
        the top and file a at the left (white-at-bottom orientation assumed).

    Raises:
        BoardNotFoundError: If no chessboard grid can be reliably detected.
    """
    LOGGER.debug(f"Detecting board in image with shape={image.shape}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = _find_board_corners(gray)
    LOGGER.debug(f"Detected board corners: {corners.tolist()}")
    return _warp(image, corners)


def split_into_squares(board: np.ndarray) -> list[list[np.ndarray]]:
    """Split a 512×512 warped board into 8×8 individual square images.

    Args:
        board: 512×512 BGR array returned by :func:`detect_board`.

    Returns:
        8×8 list of 64×64 BGR arrays.  Row 0 = rank 8, col 0 = file a.
    """
    LOGGER.debug(f"Splitting warped board with shape={board.shape} into squares")
    squares: list[list[np.ndarray]] = []
    for row in range(8):
        rank_squares: list[np.ndarray] = []
        for col in range(8):
            y1 = row * _SQUARE_SIZE
            y2 = (row + 1) * _SQUARE_SIZE
            x1 = col * _SQUARE_SIZE
            x2 = (col + 1) * _SQUARE_SIZE
            rank_squares.append(board[y1:y2, x1:x2])
        squares.append(rank_squares)
    return squares


# ---------------------------------------------------------------------------
# Corner detection
# ---------------------------------------------------------------------------


def _find_board_corners(gray: np.ndarray) -> np.ndarray:
    """Return the four outer corners as a (4, 2) float32 array.

    Order: top-left, top-right, bottom-right, bottom-left.
    Tries ``findChessboardCorners`` first, falls back to RANSAC grid fitting.
    """
    # Fast-path: works perfectly for digital/rendered boards
    result = cv2.findChessboardCorners(
        gray,
        (7, 7),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    found: bool = result[0]
    raw_corners: np.ndarray | None = result[1]
    if found and raw_corners is not None:
        LOGGER.debug("Board detected with findChessboardCorners")
        inner = raw_corners.reshape(-1, 2)
        return _outer_corners_from_inner(inner, gray.shape[::-1])

    contour_corners = _contour_board_corners(gray)
    if contour_corners is not None:
        LOGGER.debug("Board detected with contour-based perimeter fallback")
        return contour_corners

    # Robust fallback: RANSAC grid fitting for physical boards
    LOGGER.debug("findChessboardCorners failed; falling back to RANSAC grid fitting")
    return _ransac_grid_corners(gray)


def _contour_board_corners(gray: np.ndarray) -> np.ndarray | None:
    """Try to detect the board using its outer quadrilateral contour."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_h, image_w = gray.shape
    image_center = np.array([image_w / 2, image_h / 2], dtype=np.float32)
    min_area = image_w * image_h * 0.08
    best_quad: np.ndarray | None = None
    best_score = float("-inf")

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        quad = approx.reshape(4, 2).astype(np.float32)
        ordered = _order_corners(quad)
        center = ordered.mean(axis=0)
        center_penalty = np.linalg.norm(center - image_center)
        score = area - center_penalty * 250
        if score > best_score:
            best_score = score
            best_quad = ordered

    return best_quad


def _outer_corners_from_inner(
    inner: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Extrapolate four outer board corners from the 7×7 inner corner grid."""
    # inner[0] = top-left, inner[6] = top-right, inner[42] = bottom-left
    tl_inner = inner[0]
    tr_inner = inner[6]
    bl_inner = inner[42]

    col_step = (tr_inner - tl_inner) / 6
    row_step = (bl_inner - tl_inner) / 6

    tl = tl_inner - col_step - row_step
    tr = tr_inner + col_step - row_step
    br = tr_inner + col_step + 7 * row_step
    bl = tl_inner - col_step + 7 * row_step

    w, h = image_size
    clip = lambda p: np.clip(p, 0, [w, h])  # noqa: E731
    return np.array([clip(tl), clip(tr), clip(br), clip(bl)], dtype=np.float32)


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Return corners ordered as top-left, top-right, bottom-right, bottom-left."""
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmin(diffs)]
    bottom_left = points[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


# ---------------------------------------------------------------------------
# RANSAC grid fitting (physical-board fallback)
# ---------------------------------------------------------------------------


def _ransac_grid_corners(gray: np.ndarray) -> np.ndarray:
    """Detect board corners using probabilistic Hough + RANSAC grid fitting.

    Steps:
    1. Canny edge detection.
    2. ``HoughLinesP`` (probabilistic) to find line segments.
    3. Cluster into horizontal and vertical groups.
    4. RANSAC: for each direction, keep the 9 lines that best form an evenly
       spaced parallel grid.
    5. Use the resulting 9×9 = 81 intersection points as source correspondences
       and ``findHomography`` (RANSAC) to compute a robust warp.

    Raises:
        BoardNotFoundError: If insufficient lines are detected.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3)

    h, w = gray.shape
    min_length = min(h, w) // 6  # segments shorter than this are noise

    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=min_length,
        maxLineGap=20,
    )
    if raw is None or len(raw) < 4:
        raise BoardNotFoundError(
            "Not enough line segments found to detect a chessboard."
        )
    LOGGER.debug(f"Detected {len(raw)} raw Hough line segments")

    h_segs, v_segs = _split_segments(raw.reshape(-1, 4))
    LOGGER.debug(
        f"Split Hough segments into {len(h_segs)} horizontal and "
        f"{len(v_segs)} vertical candidates"
    )

    if len(h_segs) < _GRID_LINES or len(v_segs) < _GRID_LINES:
        raise BoardNotFoundError(
            f"Too few lines: {len(h_segs)} horizontal, {len(v_segs)} vertical "
            f"(need at least {_GRID_LINES} each)."
        )

    h_positions = _ransac_grid_positions(_seg_positions(h_segs, axis=1), _GRID_LINES)
    v_positions = _ransac_grid_positions(_seg_positions(v_segs, axis=0), _GRID_LINES)

    # Build 81 source → destination point correspondences
    src_pts = np.array(
        [[vx, hy] for hy in h_positions for vx in v_positions],
        dtype=np.float32,
    )
    # Map 9×9 grid to a normalised 8×8 board square
    spacing = BOARD_SIZE / 8
    dst_pts = np.array(
        [
            [col * spacing, row * spacing]
            for row in range(_GRID_LINES)
            for col in range(_GRID_LINES)
        ],
        dtype=np.float32,
    )

    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if homography is None or mask is None or int(mask.sum()) < 20:
        raise BoardNotFoundError(
            "Could not compute a reliable homography from detected grid lines."
        )
    LOGGER.debug(f"Computed board homography with {int(mask.sum())} inliers")

    # Extract the four outer corners by inverting the homography
    inv = np.linalg.inv(homography)
    corner_dst = np.array([[0, 0], [BOARD_SIZE, 0], [BOARD_SIZE, BOARD_SIZE], [0, BOARD_SIZE]], dtype=np.float32)
    corner_src = cv2.perspectiveTransform(corner_dst.reshape(1, -1, 2), inv)
    return corner_src.reshape(4, 2).astype(np.float32)


def _split_segments(
    segments: np.ndarray,
    angle_thresh_deg: float = 15.0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split line segments into horizontal and vertical groups."""
    h_segs: list[np.ndarray] = []
    v_segs: list[np.ndarray] = []
    thresh = np.deg2rad(angle_thresh_deg)
    for seg in segments:
        x1, y1, x2, y2 = seg
        angle = abs(np.arctan2(y2 - y1, x2 - x1))
        if angle < thresh or angle > np.pi - thresh:
            # nearly horizontal
            h_segs.append(seg)
        elif abs(angle - np.pi / 2) < thresh:
            # nearly vertical
            v_segs.append(seg)
    return h_segs, v_segs


def _seg_positions(segs: list[np.ndarray], axis: int) -> np.ndarray:
    """Return the midpoint coordinate for each segment along *axis*."""
    arr = np.array(segs, dtype=np.float32)
    return (arr[:, axis] + arr[:, axis + 2]) / 2


def _ransac_grid_positions(
    positions: np.ndarray,
    n_lines: int,
    n_trials: int = 200,
) -> np.ndarray:
    """Pick *n_lines* evenly-spaced positions from *positions* using RANSAC.

    For each trial, randomly pick 2 positions to define a start + spacing,
    project the expected grid positions, count inliers, keep the best set.

    Returns:
        Sorted array of *n_lines* float positions.

    Raises:
        BoardNotFoundError: If no consistent grid is found.
    """
    best_inliers: np.ndarray | None = None
    best_count = 0
    inlier_tol = (positions.max() - positions.min()) / (n_lines * 4)

    rng = np.random.default_rng(42)

    for _ in range(n_trials):
        idx = rng.choice(len(positions), 2, replace=False)
        a, b = positions[idx]
        if abs(a - b) < 1e-3:
            continue
        spacing = abs(a - b)
        origin = min(a, b)
        expected = origin + np.arange(n_lines) * spacing
        inliers = np.array(
            [p for p in positions if np.min(np.abs(expected - p)) < inlier_tol]
        )
        if len(inliers) > best_count:
            best_count = len(inliers)
            best_inliers = inliers

    if best_inliers is None or best_count < n_lines:
        raise BoardNotFoundError(
            f"Could not fit {n_lines} evenly-spaced grid lines "
            f"(best inlier count: {best_count})."
        )

    # Cluster inliers into n_lines bins and return their means
    best_inliers_sorted = np.sort(best_inliers)
    clusters = np.array_split(best_inliers_sorted, n_lines)
    return np.array([c.mean() for c in clusters if len(c) > 0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Warp helpers
# ---------------------------------------------------------------------------


def _warp(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Perspective-warp *image* so that *corners* map to a 512×512 square."""
    dst = np.array(
        [
            [0, 0],
            [BOARD_SIZE - 1, 0],
            [BOARD_SIZE - 1, BOARD_SIZE - 1],
            [0, BOARD_SIZE - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, dst)
    LOGGER.debug(f"Warping board with transform matrix={matrix.tolist()}")
    return cv2.warpPerspective(image, matrix, (BOARD_SIZE, BOARD_SIZE))
