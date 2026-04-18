"""Tests for board-localizer helpers."""

from __future__ import annotations

import numpy as np

from chesscoach.vision.board_localizer import denormalize_corners, normalize_corners


def test_corner_normalization_round_trips() -> None:
    corners = np.array(
        [[10.0, 20.0], [110.0, 20.0], [110.0, 220.0], [10.0, 220.0]],
        dtype=np.float32,
    )

    normalized = normalize_corners(corners, 200, 400)
    restored = denormalize_corners(normalized, 200, 400)

    np.testing.assert_allclose(restored, corners)
