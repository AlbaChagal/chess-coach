"""Board-level reranking and diagnostics for piece assignments."""

from __future__ import annotations

from itertools import product

from chesscoach.vision.piece_assignment import SquareCandidate
from chesscoach.vision.types import PieceLabel, SquareGrid

_NON_KING_SOFT_LIMITS: dict[PieceLabel, int] = {
    "wP": 8,
    "bP": 8,
    "wQ": 2,
    "bQ": 2,
    "wR": 3,
    "bR": 3,
    "wB": 3,
    "bB": 3,
    "wN": 3,
    "bN": 3,
}
_KING_PENALTY = 2.5
_SOFT_COUNT_PENALTY = 0.75
_EMPTY_BIAS = 0.12
_LOW_CONFIDENCE_OCCUPIED_PENALTY = 0.7
_MAX_UNCERTAIN_SQUARES = 6
_MAX_ALTERNATIVES_PER_SQUARE = 3
_UNCERTAIN_GAP_THRESHOLD = 0.18
_LOW_CONFIDENCE_THRESHOLD = 0.55
_ALLOW_EMPTY_THRESHOLD = 0.78


def empty_grid() -> SquareGrid:
    """Return an all-empty board grid."""
    return [["empty" for _ in range(8)] for _ in range(8)]


def grid_from_candidates(
    square_candidates: dict[tuple[int, int], list[SquareCandidate]],
) -> SquareGrid:
    """Build a greedy board grid from per-square candidates."""
    grid = empty_grid()
    for (row, col), candidates in square_candidates.items():
        if candidates:
            grid[row][col] = candidates[0].label
    return grid


def count_board_errors(
    expected: SquareGrid,
    predicted: SquareGrid,
) -> tuple[int, int, int, int]:
    """Return total, missed, extra, and wrong-label board errors."""
    total_errors = 0
    missed = 0
    extra = 0
    wrong_label = 0
    for row in range(8):
        for col in range(8):
            expected_label = expected[row][col]
            predicted_label = predicted[row][col]
            if expected_label == predicted_label:
                continue
            total_errors += 1
            if expected_label != "empty" and predicted_label == "empty":
                missed += 1
            elif expected_label == "empty" and predicted_label != "empty":
                extra += 1
            elif expected_label != "empty" and predicted_label != "empty":
                wrong_label += 1
    return total_errors, missed, extra, wrong_label


def find_mismatched_squares(
    expected: SquareGrid,
    predicted: SquareGrid,
) -> list[tuple[str, PieceLabel, PieceLabel]]:
    """Return board mismatches as square name, expected label, predicted label."""
    mismatches: list[tuple[str, PieceLabel, PieceLabel]] = []
    for row in range(8):
        for col in range(8):
            expected_label = expected[row][col]
            predicted_label = predicted[row][col]
            if expected_label == predicted_label:
                continue
            square = f"{chr(ord('a') + col)}{8 - row}"
            mismatches.append((square, expected_label, predicted_label))
    return mismatches


def rerank_board_candidates(
    square_candidates: dict[tuple[int, int], list[SquareCandidate]],
) -> SquareGrid:
    """Rerank uncertain squares using simple chess piece-count constraints."""
    base_grid = grid_from_candidates(square_candidates)
    uncertain_squares = _select_uncertain_squares(square_candidates)
    if not uncertain_squares:
        return base_grid

    best_grid = base_grid
    best_score = _score_grid(base_grid, square_candidates)

    option_sets = [
        _square_options(square_candidates[square])
        for square in uncertain_squares
    ]
    for option_combo in product(*option_sets):
        candidate_grid = [row[:] for row in base_grid]
        for square, candidate in zip(uncertain_squares, option_combo):
            row, col = square
            candidate_grid[row][col] = candidate.label
        candidate_score = _score_grid(candidate_grid, square_candidates)
        if candidate_score > best_score:
            best_score = candidate_score
            best_grid = candidate_grid

    return best_grid


def _select_uncertain_squares(
    square_candidates: dict[tuple[int, int], list[SquareCandidate]],
) -> list[tuple[int, int]]:
    ranked_squares: list[tuple[float, tuple[int, int]]] = []
    for square, candidates in square_candidates.items():
        if len(candidates) < 2:
            continue
        top_candidate = candidates[0]
        second_candidate = candidates[1]
        gap = top_candidate.score - second_candidate.score
        if gap > _UNCERTAIN_GAP_THRESHOLD and top_candidate.score > _LOW_CONFIDENCE_THRESHOLD:
            continue
        ranked_squares.append((gap, square))

    ranked_squares.sort(key=lambda item: item[0])
    return [square for _, square in ranked_squares[:_MAX_UNCERTAIN_SQUARES]]


def _square_options(candidates: list[SquareCandidate]) -> list[SquareCandidate]:
    options = candidates[: _MAX_ALTERNATIVES_PER_SQUARE]
    top_candidate = candidates[0]
    if top_candidate.score <= _ALLOW_EMPTY_THRESHOLD:
        options = [
            SquareCandidate(
                label="empty",
                score=max(0.0, _EMPTY_BIAS - top_candidate.score),
                box=top_candidate.box,
                bottom_center_x=top_candidate.bottom_center_x,
                bottom_center_y=top_candidate.bottom_center_y,
                row=top_candidate.row,
                col=top_candidate.col,
                center_distance=top_candidate.center_distance,
            ),
            *options,
        ]
    return options


def _score_grid(
    grid: SquareGrid,
    square_candidates: dict[tuple[int, int], list[SquareCandidate]],
) -> float:
    score = 0.0
    piece_counts: dict[PieceLabel, int] = {label: 0 for label in _NON_KING_SOFT_LIMITS}
    white_kings = 0
    black_kings = 0

    for row in range(8):
        for col in range(8):
            label = grid[row][col]
            candidate_score = _EMPTY_BIAS if label == "empty" else 0.0
            for candidate in square_candidates.get((row, col), []):
                if candidate.label == label:
                    candidate_score = candidate.score
                    break
            score += candidate_score
            if label != "empty" and candidate_score < _ALLOW_EMPTY_THRESHOLD:
                score -= (
                    (_ALLOW_EMPTY_THRESHOLD - candidate_score)
                    * _LOW_CONFIDENCE_OCCUPIED_PENALTY
                )
            if label == "wK":
                white_kings += 1
            elif label == "bK":
                black_kings += 1
            elif label in piece_counts:
                piece_counts[label] += 1

    score -= abs(1 - white_kings) * _KING_PENALTY
    score -= abs(1 - black_kings) * _KING_PENALTY
    for label, limit in _NON_KING_SOFT_LIMITS.items():
        overflow = max(0, piece_counts[label] - limit)
        score -= overflow * _SOFT_COUNT_PENALTY
    return score
