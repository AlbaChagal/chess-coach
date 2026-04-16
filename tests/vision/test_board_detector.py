"""Unit tests for board_detector — uses synthetic images, no real photos."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from chesscoach.vision.board_detector import (
    BOARD_SIZE,
    BoardNotFoundError,
    detect_board,
    split_into_squares,
)
from tests.vision.conftest import make_synthetic_board


def make_perspective_board(angle_deg: float = 15.0) -> np.ndarray:
    """Apply a mild perspective warp to a synthetic board to simulate camera angle."""
    board = make_synthetic_board(size=512)
    h, w = board.shape[:2]
    shift = int(w * np.tan(np.deg2rad(angle_deg)))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[shift, 0], [w - shift, 0], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(board, matrix, (w, h))


def test_detect_board_returns_correct_shape(synthetic_board: np.ndarray) -> None:
    result = detect_board(synthetic_board)
    assert result.shape == (BOARD_SIZE, BOARD_SIZE, 3)


def test_detect_board_returns_uint8(synthetic_board: np.ndarray) -> None:
    result = detect_board(synthetic_board)
    assert result.dtype == np.uint8


def test_blank_image_raises_board_not_found() -> None:
    blank = np.zeros((512, 512, 3), dtype=np.uint8)
    with pytest.raises(BoardNotFoundError):
        detect_board(blank)


def test_solid_color_image_raises_board_not_found() -> None:
    solid = np.full((256, 256, 3), 200, dtype=np.uint8)
    with pytest.raises(BoardNotFoundError):
        detect_board(solid)


def test_split_into_squares_count(synthetic_board: np.ndarray) -> None:
    warped = detect_board(synthetic_board)
    squares = split_into_squares(warped)
    assert len(squares) == 8
    assert all(len(row) == 8 for row in squares)


def test_split_into_squares_shape(synthetic_board: np.ndarray) -> None:
    warped = detect_board(synthetic_board)
    sq_size = BOARD_SIZE // 8
    squares = split_into_squares(warped)
    for row in squares:
        for sq in row:
            assert sq.shape == (sq_size, sq_size, 3)


def test_detect_board_perspective_warped() -> None:
    """Board with mild perspective warp should still be detected."""
    board = make_perspective_board(angle_deg=10.0)
    result = detect_board(board)
    assert result.shape == (BOARD_SIZE, BOARD_SIZE, 3)
    assert result.dtype == np.uint8


def test_detect_board_with_gaussian_noise(synthetic_board: np.ndarray) -> None:
    """Board with added Gaussian noise should still be detected."""
    noise = np.random.default_rng(0).integers(0, 30, synthetic_board.shape, dtype=np.uint8)
    noisy = np.clip(synthetic_board.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    result = detect_board(noisy)
    assert result.shape == (BOARD_SIZE, BOARD_SIZE, 3)
