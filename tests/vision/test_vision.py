"""Backward-compatibility test: BoardVision public interface."""

from pathlib import Path

import pytest

from chesscoach.vision import BoardVision


def test_fen_from_image_raises_on_missing_file() -> None:
    """BoardVision.fen_from_image raises when the file cannot be read."""
    vision = BoardVision()
    with pytest.raises((ValueError, Exception)):
        vision.fen_from_image(Path("nonexistent_board.png"))
