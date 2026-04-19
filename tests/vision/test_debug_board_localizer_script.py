"""Tests for board-localizer debug helpers."""

from __future__ import annotations

import numpy as np

from scripts.debug_board_localizer import _max_corner_error, _mean_corner_error


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
