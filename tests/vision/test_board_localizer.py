"""Tests for board-localizer helpers."""

from __future__ import annotations

import numpy as np
import random

from chesscoach.vision.board_localizer import denormalize_corners, normalize_corners
from chesscoach.vision.board_localizer_dataset import _apply_perspective_jitter


def test_corner_normalization_round_trips() -> None:
    corners = np.array(
        [[10.0, 20.0], [110.0, 20.0], [110.0, 220.0], [10.0, 220.0]],
        dtype=np.float32,
    )

    normalized = normalize_corners(corners, 200, 400)
    restored = denormalize_corners(normalized, 200, 400)

    np.testing.assert_allclose(restored, corners)


def test_perspective_jitter_can_leave_sample_unchanged(monkeypatch) -> None:
    image = np.full((40, 60, 3), 127, dtype=np.uint8)
    corners = np.array(
        [[5.0, 6.0], [55.0, 6.0], [55.0, 34.0], [5.0, 34.0]],
        dtype=np.float32,
    )

    monkeypatch.setattr(random, "random", lambda: 1.0)

    warped_image, warped_corners = _apply_perspective_jitter(image, corners)

    np.testing.assert_array_equal(warped_image, image)
    np.testing.assert_allclose(warped_corners, corners)
