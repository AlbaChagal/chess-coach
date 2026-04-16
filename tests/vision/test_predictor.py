"""Integration and unit tests for the predict_fen pipeline."""

from __future__ import annotations

import re

import cv2
import numpy as np
import pytest
from PIL import Image as PILImage

from chesscoach.vision import BoardNotFoundError, predict_fen
from chesscoach.vision.piece_classifier import PieceClassifier
from tests.vision.conftest import make_synthetic_board

_FEN_RANK_RE = re.compile(r"^[1-8pPnNbBrRqQkK]+$")
_FEN_PLACEMENT_RE = re.compile(r"^([1-8pPnNbBrRqQkK]+/){7}[1-8pPnNbBrRqQkK]+$")


def _board_to_bytes(board: np.ndarray) -> bytes:
    success, buf = cv2.imencode(".png", board)
    assert success
    return buf.tobytes()


def _board_to_pil(board: np.ndarray) -> PILImage.Image:
    rgb = cv2.cvtColor(board, cv2.COLOR_BGR2RGB)
    return PILImage.fromarray(rgb)


@pytest.fixture()
def stub() -> PieceClassifier:
    return PieceClassifier()


@pytest.fixture()
def board_bytes() -> bytes:
    return _board_to_bytes(make_synthetic_board())


@pytest.fixture()
def board_pil() -> PILImage.Image:
    return _board_to_pil(make_synthetic_board())


# --- output format ---


def test_returns_valid_fen_format_from_bytes(
    board_bytes: bytes,
    stub: PieceClassifier,
) -> None:
    fen = predict_fen(board_bytes, stub)
    assert _FEN_PLACEMENT_RE.match(fen), f"Invalid FEN: {fen!r}"


def test_returns_valid_fen_format_from_pil(
    board_pil: PILImage.Image,
    stub: PieceClassifier,
) -> None:
    fen = predict_fen(board_pil, stub)
    assert _FEN_PLACEMENT_RE.match(fen), f"Invalid FEN: {fen!r}"


def test_returns_valid_fen_format_from_path(
    tmp_path: pytest.TempPathFactory,
    stub: PieceClassifier,
) -> None:
    board = make_synthetic_board()
    img_path = tmp_path / "board.png"  # type: ignore[operator]
    cv2.imwrite(str(img_path), board)
    fen = predict_fen(img_path, stub)
    assert _FEN_PLACEMENT_RE.match(fen), f"Invalid FEN: {fen!r}"


def test_stub_produces_all_empty_fen(
    board_bytes: bytes,
    stub: PieceClassifier,
) -> None:
    """With stub classifier every square is 'empty', so every rank is '8'."""
    fen = predict_fen(board_bytes, stub)
    assert fen == "8/8/8/8/8/8/8/8"


def test_output_has_seven_slashes(board_bytes: bytes, stub: PieceClassifier) -> None:
    fen = predict_fen(board_bytes, stub)
    assert fen.count("/") == 7


# --- error handling ---


def test_invalid_bytes_raises_value_error(stub: PieceClassifier) -> None:
    with pytest.raises(ValueError):
        predict_fen(b"not an image", stub)


def test_blank_image_raises_board_not_found(stub: PieceClassifier) -> None:
    blank = np.zeros((256, 256, 3), dtype=np.uint8)
    blank_bytes = _board_to_bytes(blank)
    with pytest.raises(BoardNotFoundError):
        predict_fen(blank_bytes, stub)


# --- BoardVision wrapper ---


def test_board_vision_fen_from_image(tmp_path: pytest.TempPathFactory) -> None:
    from chesscoach.vision import BoardVision

    board = make_synthetic_board()
    img_path = tmp_path / "board.png"  # type: ignore[operator]
    cv2.imwrite(str(img_path), board)

    vision = BoardVision()
    fen = vision.fen_from_image(img_path)
    assert _FEN_PLACEMENT_RE.match(fen)
