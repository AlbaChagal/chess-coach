"""Regression tests for square preparation label parsing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_squares import _read_fen_placement, _select_metadata_corners


def test_read_fen_placement_supports_fen_sidecars(tmp_path: Path) -> None:
    image_path = tmp_path / "board.png"
    image_path.write_bytes(b"")
    image_path.with_suffix(".fen").write_text("8/8/8/8/8/8/8/8 w - - 0 1\n")

    assert _read_fen_placement(image_path) == "8/8/8/8/8/8/8/8"


def test_read_fen_placement_supports_json_sidecars(tmp_path: Path) -> None:
    image_path = tmp_path / "board.png"
    image_path.write_bytes(b"")
    image_path.with_suffix(".json").write_text(
        json.dumps({"fen": "8/8/8/8/8/8/8/8 w - - 0 1"})
    )

    assert _read_fen_placement(image_path) == "8/8/8/8/8/8/8/8"


def test_read_fen_placement_rejects_invalid_json_payload(tmp_path: Path) -> None:
    image_path = tmp_path / "board.png"
    image_path.write_bytes(b"")
    image_path.with_suffix(".json").write_text(json.dumps({"fen": None}))

    with pytest.raises(ValueError, match="Missing 'fen' string"):
        _read_fen_placement(image_path)


def test_select_metadata_corners_infers_board_orientation() -> None:
    payload = {
        "corners": [
            [900, 900],
            [100, 900],
            [100, 100],
            [900, 100],
        ],
        "pieces": [
            {"square": "a8", "box": [120, 120, 40, 40]},
            {"square": "h1", "box": [840, 840, 40, 40]},
        ],
    }

    corners = _select_metadata_corners(payload)

    assert corners is not None
    assert np.allclose(
        corners,
        np.array(
            [
                [100, 100],
                [900, 100],
                [900, 900],
                [100, 900],
            ],
            dtype=np.float32,
        ),
    )
