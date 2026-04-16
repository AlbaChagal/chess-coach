"""Regression tests for square preparation label parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare_squares import _read_fen_placement


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
