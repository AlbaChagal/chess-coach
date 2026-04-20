"""Integration and unit tests for the predict_fen pipeline."""

from __future__ import annotations

import re
from io import BytesIO

import cv2
import numpy as np
import pytest
from PIL import Image as PILImage

from chesscoach.vision import BoardNotFoundError, predict_fen
from chesscoach.vision.board_detector import (
    BOARD_SIZE,
    DEFAULT_WARP_MARGIN_RATIO,
    canonical_board_corners,
)
from chesscoach.vision.piece_classifier import PieceClassifier
from chesscoach.vision.piece_detector import PieceDetector
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


def _oriented_jpeg_bytes(width: int, height: int, orientation: int) -> bytes:
    image = PILImage.new("RGB", (width, height), color=(255, 0, 0))
    exif = PILImage.Exif()
    exif[274] = orientation
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


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


def test_detector_stub_produces_all_empty_fen(board_bytes: bytes) -> None:
    fen = predict_fen(board_bytes, PieceDetector())
    assert fen == "8/8/8/8/8/8/8/8"


def test_predict_fen_uses_default_board_localizer_when_available(
    monkeypatch,
    board_bytes: bytes,
) -> None:
    from chesscoach.vision.piece_assignment import PieceDetection

    class _Detector:
        def detect(self, image: np.ndarray) -> list[PieceDetection]:
            _ = image
            return []

    localizer_calls: list[np.ndarray] = []

    class _Localizer:
        def detect_corners(self, image: np.ndarray) -> np.ndarray:
            localizer_calls.append(image)
            return np.array(
                [[0.0, 0.0], [255.0, 0.0], [255.0, 255.0], [0.0, 255.0]],
                dtype=np.float32,
            )

    from chesscoach.vision import predictor as predictor_module

    monkeypatch.setattr(
        predictor_module,
        "_get_default_board_localizer",
        lambda: _Localizer(),
    )

    fen = predictor_module.predict_fen(board_bytes, _Detector())

    assert fen == "8/8/8/8/8/8/8/8"
    assert len(localizer_calls) == 1


def test_get_default_detector_uses_default_checkpoint_when_present(
    monkeypatch,
) -> None:
    from chesscoach.vision import predictor as predictor_module

    created: dict[str, object] = {}

    class _Detector:
        pass

    class _Checkpoint:
        def exists(self) -> bool:
            return True

    monkeypatch.setattr(predictor_module, "_default_detector", None)
    monkeypatch.setattr(
        predictor_module,
        "_DEFAULT_DETECTOR_CHECKPOINT",
        _Checkpoint(),
    )

    def _piece_detector(checkpoint=None):
        created["checkpoint"] = checkpoint
        return _Detector()

    monkeypatch.setattr(predictor_module, "PieceDetector", _piece_detector)

    detector = predictor_module._get_default_detector()

    assert isinstance(detector, _Detector)
    assert created["checkpoint"] == predictor_module._DEFAULT_DETECTOR_CHECKPOINT


@pytest.mark.parametrize("rotation_shift", [0, 1, 2, 3])
def test_orients_board_corners_from_white_king_click(rotation_shift: int) -> None:
    from chesscoach.vision import predictor as predictor_module

    board_corners = np.array(
        [[0.0, 0.0], [255.0, 0.0], [255.0, 255.0], [0.0, 255.0]],
        dtype=np.float32,
    )
    rotated_corners = np.roll(board_corners, shift=rotation_shift, axis=0)
    destination_corners = canonical_board_corners(
        BOARD_SIZE,
        margin_ratio=DEFAULT_WARP_MARGIN_RATIO,
    )
    inverse_homography = cv2.getPerspectiveTransform(
        destination_corners.astype(np.float32),
        rotated_corners.astype(np.float32),
    )
    e1_center = np.array([[predictor_module._e1_square_center()]], dtype=np.float32)
    click_point = cv2.perspectiveTransform(e1_center, inverse_homography)[0][0]

    oriented = predictor_module._orient_board_corners_from_white_king_click(
        board_corners,
        (float(click_point[0]), float(click_point[1])),
    )

    assert np.allclose(oriented, rotated_corners)


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


def test_to_bgr_applies_exif_orientation_for_bytes() -> None:
    from chesscoach.vision.predictor import _to_bgr

    oriented = _oriented_jpeg_bytes(width=40, height=20, orientation=6)

    bgr = _to_bgr(oriented)

    assert bgr.shape[:2] == (40, 20)


# --- BoardVision wrapper ---


def test_board_vision_fen_from_image(tmp_path: pytest.TempPathFactory) -> None:
    from chesscoach.vision import BoardVision

    board = make_synthetic_board()
    img_path = tmp_path / "board.png"  # type: ignore[operator]
    cv2.imwrite(str(img_path), board)

    vision = BoardVision()
    fen = vision.fen_from_image(img_path)
    assert _FEN_PLACEMENT_RE.match(fen)
