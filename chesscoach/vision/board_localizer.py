"""Learned board-corner localization for raw chessboard images."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from chesscoach.torch_utils import select_device

LOGGER = logging.getLogger(__name__)
DEFAULT_BOARD_LOCALIZER_IMAGE_SIZE = 512
BOARD_LOCALIZER_ARCHITECTURE = "resnet18"


class _BoardCornerRegressor(nn.Module):
    """ResNet backbone with sigmoid-normalized corner regression head."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self._backbone = backbone

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return normalized corner coordinates in ``[0, 1]``."""
        return torch.sigmoid(self._backbone(inputs))


def build_board_localizer(*, pretrained_backbone: bool = True) -> nn.Module:
    """Build the board-corner regression model used for training and inference."""
    weights = (
        models.ResNet18_Weights.IMAGENET1K_V1
        if pretrained_backbone
        else None
    )
    backbone = models.resnet18(weights=weights)
    backbone.fc = nn.Linear(backbone.fc.in_features, 8)
    return _BoardCornerRegressor(backbone)


def select_board_localizer_device() -> torch.device:
    """Return the preferred torch device for board-corner regression."""
    return select_device()


def normalize_corners(corners: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalize pixel-space corners into ``[0, 1]`` image coordinates."""
    scales = np.array([width, height], dtype=np.float32)
    return corners.astype(np.float32) / scales


def denormalize_corners(
    corners: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Convert normalized corners back into pixel-space image coordinates."""
    scales = np.array([width, height], dtype=np.float32)
    return corners.astype(np.float32) * scales


def _build_transform(image_size: int) -> transforms.Compose:
    """Return the shared image preprocessing pipeline."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class BoardCornerLocalizer:
    """Predict ordered outer board corners from a raw image.

    Args:
        checkpoint: Trained localizer checkpoint path.
        image_size: Inference resize used by the regressor.
    """

    def __init__(
        self,
        checkpoint: Path,
        *,
        image_size: int = DEFAULT_BOARD_LOCALIZER_IMAGE_SIZE,
    ) -> None:
        self._device = select_board_localizer_device()
        self._image_size = image_size
        self._transform = _build_transform(image_size)
        LOGGER.info(
            f"Initializing BoardCornerLocalizer device={self._device} "
            f"image_size={image_size}"
        )
        model = build_board_localizer(pretrained_backbone=False).to(self._device)
        state = torch.load(str(checkpoint), map_location=self._device)
        model.load_state_dict(state)
        model.eval()
        self._model = model
        LOGGER.info(f"Loaded board localizer checkpoint={checkpoint}")

    def detect_corners(self, image: np.ndarray) -> np.ndarray:
        """Predict ordered board corners for a raw BGR image."""
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor: torch.Tensor = self._transform(rgb)  # type: ignore[assignment]
        batch = tensor.unsqueeze(0).to(self._device)
        with torch.no_grad():
            prediction = self._model(batch).squeeze(0).detach().cpu().numpy()
        normalized = prediction.reshape(4, 2).clip(0.0, 1.0)
        return denormalize_corners(normalized, width, height)
