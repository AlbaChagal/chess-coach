"""Tests for board-detector evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scripts.evaluate_board_detector import (
    _has_usable_board_corners,
    bucket_geometry_status,
    evaluate_board_detector,
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


def test_has_usable_board_corners_requires_four_corners() -> None:
    assert _has_usable_board_corners({"corners": [[1], [2], [3], [4]]}) is True
    assert _has_usable_board_corners({"corners": None}) is False
    assert _has_usable_board_corners({}) is False


def test_evaluate_board_detector_skips_samples_without_corners(
    monkeypatch,
    tmp_path: Path,
) -> None:
    raw_test = tmp_path / "raw" / "test"
    raw_test.mkdir(parents=True)

    image_path = raw_test / "board.jpg"
    cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))
    image_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "corners": None,
                "pieces": [
                    {"piece": "K", "square": "e1", "box": None},
                ],
            }
        )
    )

    monkeypatch.setattr(
        "scripts.evaluate_board_detector.detect_board_corners",
        lambda image: np.array(
            [[1.0, 1.0], [30.0, 1.0], [30.0, 30.0], [1.0, 30.0]],
            dtype=np.float32,
        ),
    )

    diagnostics = evaluate_board_detector(
        tmp_path / "raw",
        split="test",
        bad_geometry_threshold_px=20.0,
        overlay_output_dir=None,
        overlay_limit=0,
    )

    assert diagnostics == []
