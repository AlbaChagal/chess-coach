"""Map piece detections on a warped board to chess squares."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from chesscoach.vision.board_detector import (
    BOARD_SIZE,
    DEFAULT_WARP_MARGIN_RATIO,
    canonical_board_bounds,
    canonical_board_corners,
)
from chesscoach.vision.types import PieceLabel, SquareGrid


@dataclass(frozen=True, slots=True)
class PieceDetection:
    """A detected piece on a warped board image."""

    label: PieceLabel
    score: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class AssignmentStats:
    """Diagnostics for square assignment postprocessing."""

    raw_detections: int = 0
    accepted_detections: int = 0
    same_square_rejections: int = 0
    neighbor_duplicate_rejections: int = 0


@dataclass(frozen=True, slots=True)
class SquareCandidate:
    """Detection candidate assigned to a board square in board space."""

    label: PieceLabel
    score: float
    box: tuple[float, float, float, float]
    bottom_center_x: float
    bottom_center_y: float
    row: int
    col: int
    center_distance: float


_NEIGHBOR_DUPLICATE_DISTANCE_SQUARES = 0.30
_FOOT_STRIP_TOP_RATIO = 0.72
_FOOT_STRIP_X_RATIOS = (0.2, 0.4, 0.5, 0.6, 0.8)
_FOOT_STRIP_Y_RATIOS = (0.78, 0.88, 0.96)


def _empty_grid() -> SquareGrid:
    return [["empty" for _ in range(8)] for _ in range(8)]


def _square_indices(
    x: float,
    y: float,
    *,
    board_origin_x: float,
    board_origin_y: float,
    square_size: float,
) -> tuple[int, int]:
    col = min(max(int((x - board_origin_x) / square_size), 0), 7)
    row = min(max(int((y - board_origin_y) / square_size), 0), 7)
    return row, col


def _project_detection_candidates(
    detections: list[PieceDetection],
    *,
    board_origin_x: float,
    board_origin_y: float,
    square_size: float,
) -> list[SquareCandidate]:
    candidates: list[SquareCandidate] = []
    for detection in detections:
        candidates.append(
            _candidate_from_points(
                detection.label,
                detection.score,
                detection.box,
                _foot_strip_points(detection.box),
                board_origin_x=board_origin_x,
                board_origin_y=board_origin_y,
                square_size=square_size,
            )
        )
    return candidates


def _foot_strip_points(box: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = box
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    foot_top = y1 + height * _FOOT_STRIP_TOP_RATIO
    points: list[tuple[float, float]] = []
    for x_ratio in _FOOT_STRIP_X_RATIOS:
        x = x1 + width * x_ratio
        for y_ratio in _FOOT_STRIP_Y_RATIOS:
            y = max(foot_top, y1 + height * y_ratio)
            points.append((x, min(y, y2)))
    return points


def _vote_square_from_points(
    points: list[tuple[float, float]],
    *,
    board_origin_x: float,
    board_origin_y: float,
    square_size: float,
) -> tuple[int, int, float, float]:
    counts: dict[tuple[int, int], int] = {}
    sums: dict[tuple[int, int], tuple[float, float]] = {}
    for x, y in points:
        square = _square_indices(
            x,
            y,
            board_origin_x=board_origin_x,
            board_origin_y=board_origin_y,
            square_size=square_size,
        )
        counts[square] = counts.get(square, 0) + 1
        sum_x, sum_y = sums.get(square, (0.0, 0.0))
        sums[square] = (sum_x + x, sum_y + y)

    best_square = max(
        counts,
        key=lambda square: (
            counts[square],
            sums[square][1] / counts[square],
        ),
    )
    sum_x, sum_y = sums[best_square]
    count = counts[best_square]
    return best_square[0], best_square[1], sum_x / count, sum_y / count


def _candidate_from_points(
    label: PieceLabel,
    score: float,
    box: tuple[float, float, float, float],
    points: list[tuple[float, float]],
    *,
    board_origin_x: float,
    board_origin_y: float,
    square_size: float,
) -> SquareCandidate:
    row, col, anchor_x, anchor_y = _vote_square_from_points(
        points,
        board_origin_x=board_origin_x,
        board_origin_y=board_origin_y,
        square_size=square_size,
    )
    square_center_x = board_origin_x + (col + 0.5) * square_size
    square_center_y = board_origin_y + (row + 0.5) * square_size
    center_distance = math.hypot(
        anchor_x - square_center_x,
        anchor_y - square_center_y,
    )
    return SquareCandidate(
        label=label,
        score=score,
        box=box,
        bottom_center_x=anchor_x,
        bottom_center_y=anchor_y,
        row=row,
        col=col,
        center_distance=center_distance,
    )


def _suppress_neighbor_duplicates(
    candidates: list[SquareCandidate],
    *,
    square_size: float,
) -> tuple[list[SquareCandidate], int]:
    accepted: list[SquareCandidate] = []
    duplicate_rejections = 0
    max_distance = square_size * _NEIGHBOR_DUPLICATE_DISTANCE_SQUARES

    for candidate in sorted(
        candidates,
        key=lambda item: (item.score, -item.center_distance),
        reverse=True,
    ):
        is_duplicate = False
        for incumbent in accepted:
            if candidate.label != incumbent.label:
                continue
            distance = math.hypot(
                candidate.bottom_center_x - incumbent.bottom_center_x,
                candidate.bottom_center_y - incumbent.bottom_center_y,
            )
            if distance > max_distance:
                continue
            if max(abs(candidate.row - incumbent.row), abs(candidate.col - incumbent.col)) > 1:
                continue
            is_duplicate = True
            break
        if is_duplicate:
            duplicate_rejections += 1
            continue
        accepted.append(candidate)

    return accepted, duplicate_rejections


def collect_square_candidates(
    detections: list[PieceDetection],
    *,
    board_size: int,
    board_origin_x: float = 0.0,
    board_origin_y: float = 0.0,
    board_extent: float | None = None,
) -> tuple[dict[tuple[int, int], list[SquareCandidate]], AssignmentStats]:
    """Project detections to per-square candidate lists with diagnostics."""
    actual_board_extent = board_extent if board_extent is not None else float(board_size)
    square_size = actual_board_extent / 8
    candidates = _project_detection_candidates(
        detections,
        board_origin_x=board_origin_x,
        board_origin_y=board_origin_y,
        square_size=square_size,
    )
    filtered_candidates, neighbor_duplicate_rejections = _suppress_neighbor_duplicates(
        candidates,
        square_size=square_size,
    )
    candidate_map: dict[tuple[int, int], list[SquareCandidate]] = {}
    for candidate in filtered_candidates:
        square = (candidate.row, candidate.col)
        candidate_map.setdefault(square, []).append(candidate)

    same_square_rejections = 0
    for square_candidates in candidate_map.values():
        square_candidates.sort(
            key=lambda item: (item.score, -item.center_distance),
            reverse=True,
        )
        same_square_rejections += max(0, len(square_candidates) - 1)

    return candidate_map, AssignmentStats(
        raw_detections=len(detections),
        accepted_detections=len(candidate_map),
        same_square_rejections=same_square_rejections,
        neighbor_duplicate_rejections=neighbor_duplicate_rejections,
    )


def assign_detections_to_squares_with_stats(
    detections: list[PieceDetection],
    *,
    board_size: int,
    board_origin_x: float = 0.0,
    board_origin_y: float = 0.0,
    board_extent: float | None = None,
) -> tuple[SquareGrid, AssignmentStats]:
    """Assign detections to squares and return assignment diagnostics."""
    grid = _empty_grid()
    candidate_map, stats = collect_square_candidates(
        detections,
        board_size=board_size,
        board_origin_x=board_origin_x,
        board_origin_y=board_origin_y,
        board_extent=board_extent,
    )
    for (row, col), square_candidates in candidate_map.items():
        grid[row][col] = square_candidates[0].label

    return grid, stats


def collect_square_candidates_via_homography(
    detections: list[PieceDetection],
    *,
    board_corners: np.ndarray,
    board_size: int = BOARD_SIZE,
    margin_ratio: float = DEFAULT_WARP_MARGIN_RATIO,
) -> tuple[dict[tuple[int, int], list[SquareCandidate]], AssignmentStats]:
    """Project raw-image detections to board-space square candidates."""
    board_origin_x, board_origin_y, board_extent = canonical_board_bounds(
        board_size,
        margin_ratio=margin_ratio,
    )
    square_size = board_extent / 8
    destination_corners = canonical_board_corners(
        board_size,
        margin_ratio=margin_ratio,
    )
    homography = cv2.getPerspectiveTransform(
        board_corners.astype(np.float32),
        destination_corners.astype(np.float32),
    )
    candidates: list[SquareCandidate] = []
    for detection in detections:
        foot_points = np.array(
            [[list(point) for point in _foot_strip_points(detection.box)]],
            dtype=np.float32,
        )
        projected_points = cv2.perspectiveTransform(foot_points, homography)[0]
        projected_box = (
            float(min(point[0] for point in projected_points)),
            float(min(point[1] for point in projected_points)),
            float(max(point[0] for point in projected_points)),
            float(max(point[1] for point in projected_points)),
        )
        candidates.append(
            _candidate_from_points(
                detection.label,
                detection.score,
                projected_box,
                [(float(point[0]), float(point[1])) for point in projected_points],
                board_origin_x=board_origin_x,
                board_origin_y=board_origin_y,
                square_size=square_size,
            )
        )

    filtered_candidates, neighbor_duplicate_rejections = _suppress_neighbor_duplicates(
        candidates,
        square_size=square_size,
    )
    candidate_map: dict[tuple[int, int], list[SquareCandidate]] = {}
    for candidate in filtered_candidates:
        candidate_map.setdefault((candidate.row, candidate.col), []).append(candidate)
    same_square_rejections = 0
    for square_candidates in candidate_map.values():
        square_candidates.sort(
            key=lambda item: (item.score, -item.center_distance),
            reverse=True,
        )
        same_square_rejections += max(0, len(square_candidates) - 1)
    return candidate_map, AssignmentStats(
        raw_detections=len(detections),
        accepted_detections=len(candidate_map),
        same_square_rejections=same_square_rejections,
        neighbor_duplicate_rejections=neighbor_duplicate_rejections,
    )


def assign_detections_to_squares(
    detections: list[PieceDetection],
    *,
    board_size: int,
    board_origin_x: float = 0.0,
    board_origin_y: float = 0.0,
    board_extent: float | None = None,
) -> SquareGrid:
    """Assign detections to squares using the piece base point.

    The bottom-center of a piece box is a better proxy for the occupied square
    than the box center because tall pieces extend into neighboring squares.

    Args:
        detections: Piece detections on a warped board.
        board_size: Warped board canvas dimension in pixels.
        board_origin_x: Left edge of the actual board inside the canvas.
        board_origin_y: Top edge of the actual board inside the canvas.
        board_extent: Side length of the actual board inside the canvas.

    Returns:
        8x8 label grid in rank-8-to-rank-1 order.
    """
    grid, _ = assign_detections_to_squares_with_stats(
        detections,
        board_size=board_size,
        board_origin_x=board_origin_x,
        board_origin_y=board_origin_y,
        board_extent=board_extent,
    )
    return grid


def assign_detections_via_homography_with_stats(
    detections: list[PieceDetection],
    *,
    board_corners: np.ndarray,
    board_size: int = BOARD_SIZE,
    margin_ratio: float = DEFAULT_WARP_MARGIN_RATIO,
) -> tuple[SquareGrid, AssignmentStats]:
    """Assign raw-image detections to squares and return diagnostics."""
    grid = _empty_grid()
    candidate_map, stats = collect_square_candidates_via_homography(
        detections,
        board_corners=board_corners,
        board_size=board_size,
        margin_ratio=margin_ratio,
    )
    for (row, col), square_candidates in candidate_map.items():
        grid[row][col] = square_candidates[0].label
    return grid, stats


def assign_detections_via_homography(
    detections: list[PieceDetection],
    *,
    board_corners: np.ndarray,
    board_size: int = BOARD_SIZE,
    margin_ratio: float = DEFAULT_WARP_MARGIN_RATIO,
) -> SquareGrid:
    """Assign raw-image detections to squares using board homography."""
    grid, _ = assign_detections_via_homography_with_stats(
        detections,
        board_corners=board_corners,
        board_size=board_size,
        margin_ratio=margin_ratio,
    )
    return grid
