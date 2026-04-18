"""Tests for board-detector evaluation helpers."""

from __future__ import annotations

import numpy as np

from scripts.evaluate_board_detector import (
    bucket_geometry_status,
    max_corner_error,
    mean_corner_error,
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

    assert mean_corner_error(expected, predicted) == 2.5


def test_max_corner_error_returns_largest_distance() -> None:
    expected = np.array(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        dtype=np.float32,
    )
    predicted = np.array(
        [[0.0, 0.0], [10.0, 0.0], [16.0, 10.0], [0.0, 13.0]],
        dtype=np.float32,
    )

    assert max_corner_error(expected, predicted) == 6.0


def test_bucket_geometry_status_handles_not_found() -> None:
    assert (
        bucket_geometry_status(
            None,
            bad_geometry_threshold_px=20.0,
        )
        == "board_not_found"
    )


def test_bucket_geometry_status_handles_bad_and_good_geometry() -> None:
    assert (
        bucket_geometry_status(
            24.0,
            bad_geometry_threshold_px=20.0,
        )
        == "bad_geometry"
    )
    assert (
        bucket_geometry_status(
            12.0,
            bad_geometry_threshold_px=20.0,
        )
        == "good_geometry"
    )
