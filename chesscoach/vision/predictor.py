"""Top-level pipeline: image bytes/path/PIL → FEN piece-placement string."""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Protocol, TypeGuard, cast

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps

from chesscoach.vision.board_detector import (
    BOARD_SIZE,
    DEFAULT_WARP_MARGIN_RATIO,
    canonical_board_bounds,
    canonical_board_corners,
    detect_board_corners,
    detect_board,
    split_into_squares,
    warp_board_from_corners,
)
from chesscoach.vision.board_localizer import BoardCornerLocalizer
from chesscoach.vision.board_postprocess import rerank_board_candidates
from chesscoach.vision.fen_builder import build_fen
from chesscoach.vision.piece_assignment import (
    PieceDetection,
    collect_square_candidates_via_homography,
)
from chesscoach.vision.piece_detector import PieceDetector
from chesscoach.vision.types import PieceLabel, SquareGrid

_default_detector: PieceDetector | None = None
_default_board_localizer: BoardCornerLocalizer | None = None
_default_board_localizer_initialized = False
_DEFAULT_DETECTOR_CHECKPOINT = Path("models/piece_detector.pt")
_DEFAULT_BOARD_LOCALIZER_CHECKPOINT = Path("models/board_localizer.pt")
LOGGER = logging.getLogger(__name__)


class _LegacyClassifier(Protocol):
    def classify(
        self,
        occupancy_square: np.ndarray,
        piece_square: np.ndarray | None = None,
    ) -> PieceLabel: ...


class _DetectorClassifier(Protocol):
    def detect(self, image: np.ndarray) -> list[PieceDetection]: ...


def _get_default_detector() -> PieceDetector:
    global _default_detector
    if _default_detector is None:
        if _DEFAULT_DETECTOR_CHECKPOINT.exists():
            LOGGER.info(
                f"Initializing default piece detector from "
                f"{_DEFAULT_DETECTOR_CHECKPOINT}"
            )
            _default_detector = PieceDetector(_DEFAULT_DETECTOR_CHECKPOINT)
        else:
            LOGGER.info(
                f"No default piece detector checkpoint found at "
                f"{_DEFAULT_DETECTOR_CHECKPOINT}; falling back to stub detector."
            )
            _default_detector = PieceDetector()
    return _default_detector


def _get_default_board_localizer() -> BoardCornerLocalizer | None:
    global _default_board_localizer
    global _default_board_localizer_initialized
    if _default_board_localizer_initialized:
        return _default_board_localizer

    _default_board_localizer_initialized = True
    if not _DEFAULT_BOARD_LOCALIZER_CHECKPOINT.exists():
        LOGGER.info(
            f"No default board localizer checkpoint found at "
            f"{_DEFAULT_BOARD_LOCALIZER_CHECKPOINT}; falling back to classical "
            f"board detection."
        )
        return None

    LOGGER.info(
        f"Initializing default board localizer from "
        f"{_DEFAULT_BOARD_LOCALIZER_CHECKPOINT}"
    )
    _default_board_localizer = BoardCornerLocalizer(
        _DEFAULT_BOARD_LOCALIZER_CHECKPOINT
    )
    return _default_board_localizer


def _to_bgr(image: bytes | Path | PILImage.Image) -> np.ndarray:
    """Convert any supported input type to a BGR numpy array."""
    if isinstance(image, (bytes, bytearray)):
        LOGGER.debug("Decoding board image from raw bytes")
        try:
            with PILImage.open(BytesIO(image)) as pil_image:
                return _pil_to_bgr(pil_image)
        except (OSError, ValueError) as exc:
            raise ValueError("Could not decode image bytes.") from exc

    if isinstance(image, Path):
        LOGGER.debug(f"Loading board image from path: {image}")
        try:
            with PILImage.open(image) as pil_image:
                return _pil_to_bgr(pil_image)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read image file: {image}") from exc

    # PIL.Image
    LOGGER.debug("Converting board image from PIL.Image")
    return _pil_to_bgr(image)


def _pil_to_bgr(image: PILImage.Image) -> np.ndarray:
    """Convert a PIL image to BGR, applying EXIF orientation first."""
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    rgb = np.array(normalized)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def predict_fen(
    image: bytes | Path | PILImage.Image,
    classifier: PieceDetector | _LegacyClassifier | _DetectorClassifier | None = None,
    board_localizer: BoardCornerLocalizer | None = None,
    white_king_start_click: tuple[float, float] | None = None,
) -> str:
    """Detect the chess position in *image* and return a FEN piece-placement string.

    Only the piece-placement field is returned (the first FEN segment, e.g.
    ``"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"``).  The caller is
    responsible for appending the remaining FEN fields (active color, castling
    rights, etc.) which cannot be inferred from a single image.

    Args:
        image: The board image as raw bytes, a file :class:`~pathlib.Path`, or
            a :class:`PIL.Image.Image`.
        classifier: A detector or the legacy per-square classifier. Defaults to
            a stub detector (returns no detections) when no checkpoint has been
            loaded.

    Returns:
        FEN piece-placement string (rank 8 first, rank 1 last).

    Raises:
        :exc:`~chesscoach.vision.board_detector.BoardNotFoundError`: If no
            chessboard can be located in the image.
        :exc:`ValueError`: If the image cannot be decoded.
    """
    if classifier is None:
        classifier = _get_default_detector()

    LOGGER.info("Starting FEN prediction")
    bgr = _to_bgr(image)
    if _is_detector_classifier(classifier):
        if board_localizer is None:
            board_localizer = _get_default_board_localizer()
        board_corners = (
            board_localizer.detect_corners(bgr)
            if board_localizer is not None
            else detect_board_corners(bgr)
        )
        board_corners = _orient_board_corners_from_white_king_click(
            board_corners,
            white_king_start_click,
        )
        detections = classifier.detect(bgr)
        square_candidates, _ = collect_square_candidates_via_homography(
            detections,
            board_corners=board_corners,
            board_size=BOARD_SIZE,
            margin_ratio=DEFAULT_WARP_MARGIN_RATIO,
        )
        grid = rerank_board_candidates(square_candidates)
        fen = build_fen(grid)
        LOGGER.info(f"Finished detector-based FEN prediction: {fen}")
        return fen

    warped = (
        warp_board_from_corners(
            bgr,
            _orient_board_corners_from_white_king_click(
                detect_board_corners(bgr),
                white_king_start_click,
            ),
        )
        if white_king_start_click is not None
        else detect_board(bgr)
    )
    return _predict_with_legacy_classifier(warped, cast(_LegacyClassifier, classifier))


def _is_detector_classifier(
    classifier: PieceDetector | _LegacyClassifier | _DetectorClassifier,
) -> TypeGuard[PieceDetector | _DetectorClassifier]:
    """Return whether *classifier* exposes detector-style image inference."""
    return hasattr(classifier, "detect")


def _orient_board_corners_from_white_king_click(
    board_corners: np.ndarray,
    white_king_start_click: tuple[float, float] | None,
) -> np.ndarray:
    """Rotate board corners so the clicked white king start square maps to e1."""
    if white_king_start_click is None:
        return board_corners

    click_point = np.array([[white_king_start_click]], dtype=np.float32)
    target_center = np.array(_e1_square_center(), dtype=np.float32)
    candidate_corners = [
        np.roll(board_corners, shift=shift, axis=0).astype(np.float32)
        for shift in range(4)
    ]
    best_corners = candidate_corners[0]
    best_distance = float("inf")

    for corners in candidate_corners:
        homography = cv2.getPerspectiveTransform(
            corners.astype(np.float32),
            canonical_board_corners(
                BOARD_SIZE,
                margin_ratio=DEFAULT_WARP_MARGIN_RATIO,
            ),
        )
        projected_click = cv2.perspectiveTransform(click_point, homography)[0][0]
        distance = float(np.linalg.norm(projected_click - target_center))
        if distance < best_distance:
            best_distance = distance
            best_corners = corners

    return best_corners


def _e1_square_center() -> tuple[float, float]:
    """Return the canonical center point of e1 on the warped board canvas."""
    board_origin_x, board_origin_y, board_extent = canonical_board_bounds(
        BOARD_SIZE,
        margin_ratio=DEFAULT_WARP_MARGIN_RATIO,
    )
    square_size = board_extent / 8
    center_x = board_origin_x + (4 + 0.5) * square_size
    center_y = board_origin_y + (7 + 0.5) * square_size
    return center_x, center_y


def _predict_with_legacy_classifier(
    warped: np.ndarray,
    classifier: _LegacyClassifier,
) -> str:
    """Fallback path for the legacy square-classifier pipeline."""
    occupancy_squares = split_into_squares(warped, context_scale=1.0)
    piece_squares = split_into_squares(
        warped,
        crop_width_scale=1.5,
        crop_height_scale=2.4,
        center_y_offset_scale=-0.45,
    )

    grid: SquareGrid = []
    for row_idx, (occ_row, piece_row) in enumerate(
        zip(occupancy_squares, piece_squares)
    ):
        rank_labels: list[PieceLabel] = [
            classifier.classify(occ_square, piece_square)
            for occ_square, piece_square in zip(occ_row, piece_row)
        ]
        LOGGER.debug(f"Predicted rank {row_idx} labels: {rank_labels}")
        grid.append(rank_labels)

    fen = build_fen(grid)
    LOGGER.info(f"Finished legacy FEN prediction: {fen}")
    return fen
