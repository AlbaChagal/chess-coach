"""Piece detector operating on a warped board image."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms

from chesscoach.torch_utils import select_device
from chesscoach.vision.piece_assignment import PieceDetection
from chesscoach.vision.types import PIECE_LABELS, PieceLabel

_DETECTOR_LABELS: list[PieceLabel] = [label for label in PIECE_LABELS if label != "empty"]
_INDEX_TO_LABEL: dict[int, PieceLabel] = {
    idx: label for idx, label in enumerate(_DETECTOR_LABELS, start=1)
}
_LABEL_TO_INDEX: dict[PieceLabel, int] = {
    label: idx for idx, label in _INDEX_TO_LABEL.items()
}
_DEFAULT_SCORE_THRESHOLD = 0.35
_CLASS_AGNOSTIC_NMS_IOU_THRESHOLD = 0.35
DEFAULT_DETECTOR_IMAGE_SIZE = 640
LOGGER = logging.getLogger(__name__)
DETECTOR_ARCHITECTURE = "fasterrcnn_mobilenet_v3_large_fpn"


def detector_num_classes() -> int:
    """Return the number of detector classes including background."""
    return len(_DETECTOR_LABELS) + 1


def detector_label_to_index(label: PieceLabel) -> int:
    """Map a piece label to its detector class index."""
    return _LABEL_TO_INDEX[label]


def detector_index_to_label(index: int) -> PieceLabel:
    """Map a detector class index back to a piece label."""
    return _INDEX_TO_LABEL[index]


def build_piece_detector(*, pretrained_backbone: bool = True) -> torch.nn.Module:
    """Build the detector architecture used for training and inference."""
    weights = (
        FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        if pretrained_backbone
        else None
    )
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        detector_num_classes(),
    )
    return model


def select_detector_device() -> torch.device:
    """Return a stable device choice for torchvision detection models."""
    device = select_device()
    if device.type == "mps":
        LOGGER.info(
            "Using cpu for piece detection because torchvision detection on mps "
            "is unreliable and often much slower than expected."
        )
        return torch.device("cpu")
    return device


def _image_to_tensor(board: np.ndarray) -> torch.Tensor:
    """Convert a warped BGR board image to a normalized tensor."""
    rgb = cv2.cvtColor(board, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor


def _resize_board_for_detector(
    board: np.ndarray,
    *,
    image_size: int,
) -> tuple[np.ndarray, float, float]:
    """Resize a warped board to the detector working size."""
    height, width = board.shape[:2]
    if height == image_size and width == image_size:
        return board, 1.0, 1.0
    resized = cv2.resize(board, (image_size, image_size))
    return resized, width / image_size, height / image_size


def _apply_class_agnostic_nms(
    detections: list[PieceDetection],
    *,
    iou_threshold: float = _CLASS_AGNOSTIC_NMS_IOU_THRESHOLD,
) -> list[PieceDetection]:
    """Suppress overlapping detections regardless of predicted class."""
    if len(detections) < 2:
        return detections

    boxes = torch.tensor([detection.box for detection in detections], dtype=torch.float32)
    scores = torch.tensor(
        [detection.score for detection in detections],
        dtype=torch.float32,
    )
    kept_indices = nms(boxes, scores, iou_threshold).tolist()
    return [detections[index] for index in kept_indices]


class PieceDetector:
    """Detect chess pieces on a warped board image.

    Args:
        checkpoint: Optional checkpoint path for a trained detector.
        score_threshold: Minimum confidence score to keep a detection.
    """

    def __init__(
        self,
        checkpoint: Path | None = None,
        *,
        score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
        image_size: int = DEFAULT_DETECTOR_IMAGE_SIZE,
    ) -> None:
        self._stub = checkpoint is None
        self._device = select_detector_device()
        self._score_threshold = score_threshold
        self._image_size = image_size
        LOGGER.info(
            f"Initializing PieceDetector stub={self._stub} "
            f"device={self._device} score_threshold={score_threshold:.2f} "
            f"image_size={image_size}"
        )
        self._model: torch.nn.Module | None = None

        if checkpoint is None:
            return

        model = build_piece_detector(pretrained_backbone=False).to(self._device)
        state = torch.load(str(checkpoint), map_location=self._device)
        model.load_state_dict(state)
        model.eval()
        self._model = model
        LOGGER.info(f"Loaded piece detector checkpoint={checkpoint}")

    def detect(self, warped_board: np.ndarray) -> list[PieceDetection]:
        """Run piece detection on a warped board image."""
        if self._stub:
            LOGGER.debug("Stub detector returning no detections")
            return []
        if self._model is None:
            raise RuntimeError("PieceDetector is not initialized with a model.")

        resized_board, scale_x, scale_y = _resize_board_for_detector(
            warped_board,
            image_size=self._image_size,
        )
        tensor = _image_to_tensor(resized_board).to(self._device)
        with torch.no_grad():
            prediction = self._model([tensor])[0]

        detections: list[PieceDetection] = []
        boxes = prediction["boxes"].detach().cpu().tolist()
        labels = prediction["labels"].detach().cpu().tolist()
        scores = prediction["scores"].detach().cpu().tolist()
        for box, label_idx, score in zip(boxes, labels, scores):
            if score < self._score_threshold or label_idx == 0:
                continue
            x1, y1, x2, y2 = box
            detections.append(
                PieceDetection(
                    label=detector_index_to_label(int(label_idx)),
                    score=float(score),
                    box=(
                        float(x1 * scale_x),
                        float(y1 * scale_y),
                        float(x2 * scale_x),
                        float(y2 * scale_y),
                    ),
                )
            )

        detections = _apply_class_agnostic_nms(detections)
        LOGGER.debug(f"Piece detector produced {len(detections)} detections")
        return detections
