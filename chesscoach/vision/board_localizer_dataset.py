"""Dataset helpers for learned board-corner localization."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from chesscoach.vision.board_localizer import normalize_corners

Split = Literal["train", "val", "test"]
_MAX_BRIGHTNESS_SHIFT = 16.0
_MIN_CONTRAST_SCALE = 0.9
_MAX_CONTRAST_SCALE = 1.1
_BLUR_PROBABILITY = 0.15


def _apply_color_jitter(image: np.ndarray) -> np.ndarray:
    """Apply mild brightness and contrast jitter."""
    alpha = random.uniform(_MIN_CONTRAST_SCALE, _MAX_CONTRAST_SCALE)
    beta = random.uniform(-_MAX_BRIGHTNESS_SHIFT, _MAX_BRIGHTNESS_SHIFT)
    jittered = image.astype(np.float32) * alpha + beta
    return np.clip(jittered, 0, 255).astype(np.uint8)


def _apply_blur(image: np.ndarray) -> np.ndarray:
    """Apply a mild blur augmentation."""
    return cv2.GaussianBlur(image, (3, 3), 0)


def _augment_localizer_sample(image: np.ndarray) -> np.ndarray:
    """Apply image-only augmentations safe for corner regression."""
    augmented = _apply_color_jitter(image)
    if random.random() < _BLUR_PROBABILITY:
        augmented = _apply_blur(augmented)
    return augmented


def _build_transform(image_size: int) -> transforms.Compose:
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


class BoardLocalizationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load raw images and normalized board corners from a JSONL manifest."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        split: Split,
        root: Path | None = None,
        image_size: int,
        augment: bool = False,
    ) -> None:
        self._root = root or manifest_path.parent
        self._transform = _build_transform(image_size)
        self._augment = augment
        self._records: list[dict[str, Any]] = []
        for line in manifest_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["split"] == split:
                self._records.append(record)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self._records[idx]
        image_path = self._root / record["image_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read board-localizer image: {image_path}")
        if self._augment:
            image = _augment_localizer_sample(image)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor: torch.Tensor = self._transform(rgb)  # type: ignore[assignment]
        corners = np.array(record["board_corners"], dtype=np.float32)
        normalized = normalize_corners(corners, image.shape[1], image.shape[0]).reshape(-1)
        target = torch.tensor(normalized, dtype=torch.float32)
        return tensor, target
