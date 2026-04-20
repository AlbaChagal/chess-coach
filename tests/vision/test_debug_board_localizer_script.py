"""Tests for board-localizer debug helpers."""

from __future__ import annotations

import numpy as np

from scripts.debug_board_localizer import (
    _fit_width,
    _max_corner_error,
    _mean_corner_error,
    _stack_debug_panel,
)


def test_mean_corner_error_returns_average_distance() -> None:
    expected = np.array(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        dtype=np.float32,
    )
    predicted = np.array(
        [[1.0, 0.0], [10.0, 2.0], [13.0, 10.0], [0.0, 14.0]],
        dtype=np.float32,
    )

    assert _mean_corner_error(expected, predicted) == 2.5


def test_max_corner_error_returns_largest_distance() -> None:
    expected = np.array(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        dtype=np.float32,
    )
    predicted = np.array(
        [[0.0, 0.0], [10.0, 0.0], [16.0, 10.0], [0.0, 13.0]],
        dtype=np.float32,
    )

    assert _max_corner_error(expected, predicted) == 6.0


def test_fit_width_preserves_requested_width() -> None:
    image = np.zeros((20, 40, 3), dtype=np.uint8)

    resized = _fit_width(image, 80)

    assert resized.shape == (40, 80, 3)


def test_stack_debug_panel_combines_three_views() -> None:
    image = np.zeros((50, 80, 3), dtype=np.uint8)

    panel = _stack_debug_panel(
        raw_overlay=image,
        expected_warp=image,
        predicted_warp=image,
        mean_error=12.5,
        max_error=24.0,
    )

    assert panel.shape[1] == 2700
    assert panel.shape[0] >= 90
