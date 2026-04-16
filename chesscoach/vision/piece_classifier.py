"""Two-stage piece classifier: occupancy then piece type.

Stage 1 — OccupancyModel (ResNet-18, 2 classes):
    Is this square empty or occupied?

Stage 2 — PieceModel (ResNet-18, 12 classes):
    Which piece (type + colour) is on this occupied square?

Running both as ResNet-18 keeps the dependency surface small while matching
chesscog's reported per-square accuracy on physical boards (~99.96 / 100%).

Stub mode (no checkpoints supplied):
    Stage 1 always returns "empty" — the full pipeline can be wired and tested
    without any trained weights.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from chesscoach.torch_utils import select_device
from chesscoach.vision.types import PIECE_LABELS, PieceLabel

# chesscog uses 100×100 inputs with contextual padding — we match that size
_INPUT_SIZE = 100
_NUM_PIECE_CLASSES = 12  # wP wN wB wR wQ wK bP bN bB bR bQ bK (no "empty")
_PIECE_LABELS_NO_EMPTY: list[PieceLabel] = [lbl for lbl in PIECE_LABELS if lbl != "empty"]
LOGGER = logging.getLogger(__name__)

_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((_INPUT_SIZE, _INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _build_resnet(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _load_model(checkpoint_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    LOGGER.info(
        f"Loading ResNet-18 checkpoint path={checkpoint_path} "
        f"num_classes={num_classes} device={device}"
    )
    model = _build_resnet(num_classes).to(device)
    state = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    LOGGER.debug(f"Loaded model architecture: {model}")
    return model


def _infer(model: nn.Module, square: np.ndarray, device: torch.device) -> int:
    """Run a forward pass and return the argmax class index."""
    rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
    tensor: torch.Tensor = _TRANSFORM(rgb)  # type: ignore[assignment]
    batch = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(batch)
    pred = int(logits.argmax(dim=1).item())
    LOGGER.debug(f"Inference result device={device} predicted_class={pred}")
    return pred


class PieceClassifier:
    """Two-stage piece classifier for a single board square.

    Stage 1 (occupancy): is the square empty or occupied?
    Stage 2 (piece type): which of the 12 pieces is it?

    When both checkpoints are ``None`` the classifier runs in **stub mode**:
    every square is classified as ``"empty"``.  This allows the full pipeline
    to be wired and tested before real model weights are available.

    Args:
        occupancy_checkpoint: Path to a ``.pt`` file for the occupancy ResNet-18
            (2-class: empty / occupied).  Pass ``None`` for stub mode.
        piece_checkpoint: Path to a ``.pt`` file for the piece ResNet-18
            (12-class: wP..bK).  Pass ``None`` for stub mode.
    """

    def __init__(
        self,
        occupancy_checkpoint: Path | None = None,
        piece_checkpoint: Path | None = None,
    ) -> None:
        self._stub = occupancy_checkpoint is None and piece_checkpoint is None
        self._device = select_device()
        LOGGER.info(
            f"Initializing PieceClassifier stub={self._stub} device={self._device}"
        )

        if not self._stub:
            if occupancy_checkpoint is None or piece_checkpoint is None:
                raise ValueError(
                    "Both occupancy_checkpoint and piece_checkpoint must be provided "
                    "together, or both omitted for stub mode."
                )
            self._occupancy_model = _load_model(occupancy_checkpoint, 2, self._device)
            self._piece_model = _load_model(piece_checkpoint, _NUM_PIECE_CLASSES, self._device)

    def classify(self, square: np.ndarray) -> PieceLabel:
        """Classify a single board square.

        Args:
            square: BGR numpy array of any size (resized to 100×100 internally).

        Returns:
            A :data:`~chesscoach.vision.types.PieceLabel` such as ``"wK"``
            or ``"empty"``.
        """
        if self._stub:
            LOGGER.debug("Stub classifier returning empty square")
            return "empty"

        # Stage 1: occupancy
        occupancy_idx = _infer(self._occupancy_model, square, self._device)
        LOGGER.debug(f"Occupancy prediction class={occupancy_idx}")
        if occupancy_idx == 0:  # class 0 = empty
            return "empty"

        # Stage 2: piece type
        piece_idx = _infer(self._piece_model, square, self._device)
        label = _PIECE_LABELS_NO_EMPTY[piece_idx]
        LOGGER.debug(f"Piece prediction class={piece_idx} label={label}")
        return label
