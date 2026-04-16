"""Unit tests for PieceClassifier (two-stage occupancy + piece model)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from chesscoach.vision.piece_classifier import PieceClassifier
from chesscoach.vision.types import PIECE_LABELS


@pytest.fixture()
def stub() -> PieceClassifier:
    return PieceClassifier()  # no checkpoints → stub mode


# ---------------------------------------------------------------------------
# Stub mode
# ---------------------------------------------------------------------------


def test_stub_returns_empty_for_any_square(stub: PieceClassifier) -> None:
    square = np.zeros((64, 64, 3), dtype=np.uint8)
    assert stub.classify(square) == "empty"


def test_stub_handles_various_sizes(stub: PieceClassifier) -> None:
    for size in (32, 64, 100, 128):
        square = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        assert stub.classify(square) == "empty"


def test_stub_is_deterministic(stub: PieceClassifier) -> None:
    square = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    results = {stub.classify(square) for _ in range(5)}
    assert len(results) == 1


def test_stub_returns_valid_piece_label(stub: PieceClassifier) -> None:
    square = np.zeros((64, 64, 3), dtype=np.uint8)
    assert stub.classify(square) in PIECE_LABELS


# ---------------------------------------------------------------------------
# Partial checkpoint raises
# ---------------------------------------------------------------------------


def test_only_occupancy_checkpoint_raises() -> None:
    with pytest.raises(ValueError, match="Both"):
        PieceClassifier(occupancy_checkpoint=Path("occ.pt"))


def test_only_piece_checkpoint_raises() -> None:
    with pytest.raises(ValueError, match="Both"):
        PieceClassifier(piece_checkpoint=Path("piece.pt"))


def test_missing_checkpoint_file_raises() -> None:
    with pytest.raises(Exception):
        PieceClassifier(
            occupancy_checkpoint=Path("/nonexistent/occ.pt"),
            piece_checkpoint=Path("/nonexistent/piece.pt"),
        )


# ---------------------------------------------------------------------------
# Two-stage wiring with mock models
# ---------------------------------------------------------------------------


def _make_mock_model(argmax_class: int, num_classes: int) -> MagicMock:
    """Build a mock nn.Module that returns fixed logits producing *argmax_class*."""
    logits = torch.zeros(1, num_classes)
    logits[0, argmax_class] = 10.0

    mock = MagicMock()
    mock.return_value = logits
    mock.eval.return_value = mock
    mock.to.return_value = mock
    return mock


def _classifier_with_mocks(
    occupancy_class: int,
    piece_class: int,
) -> PieceClassifier:
    """Return a PieceClassifier whose internal models are mocked."""
    clf = PieceClassifier.__new__(PieceClassifier)
    clf._stub = False
    clf._device = torch.device("cpu")
    clf._occupancy_model = _make_mock_model(occupancy_class, 2)
    clf._piece_model = _make_mock_model(piece_class, 12)
    return clf


def test_two_stage_empty_square() -> None:
    """Occupancy model says empty (class 0) → result is 'empty'."""
    clf = _classifier_with_mocks(occupancy_class=0, piece_class=0)
    square = np.zeros((64, 64, 3), dtype=np.uint8)
    assert clf.classify(square) == "empty"


def test_two_stage_occupied_white_pawn() -> None:
    """Occupancy=occupied (1), piece=0 (wP) → result is 'wP'."""
    clf = _classifier_with_mocks(occupancy_class=1, piece_class=0)
    square = np.zeros((64, 64, 3), dtype=np.uint8)
    assert clf.classify(square) == "wP"


def test_two_stage_occupied_black_king() -> None:
    """Occupancy=occupied (1), piece=11 (bK, last in list) → result is 'bK'."""
    from chesscoach.vision.piece_classifier import _PIECE_LABELS_NO_EMPTY

    last_idx = len(_PIECE_LABELS_NO_EMPTY) - 1
    clf = _classifier_with_mocks(occupancy_class=1, piece_class=last_idx)
    square = np.zeros((64, 64, 3), dtype=np.uint8)
    assert clf.classify(square) == _PIECE_LABELS_NO_EMPTY[last_idx]


def test_two_stage_piece_model_not_called_when_empty() -> None:
    """Piece model must NOT be invoked if occupancy model says empty."""
    clf = _classifier_with_mocks(occupancy_class=0, piece_class=0)
    square = np.zeros((64, 64, 3), dtype=np.uint8)
    clf.classify(square)
    clf._piece_model.assert_not_called()
