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
_PERSPECTIVE_JITTER_PROBABILITY = 0.0
_MAX_CORNER_JITTER_RATIO = 0.06
_TRANSLATION_JITTER_PROBABILITY = 0.7
_MAX_CANVAS_EXPANSION_RATIO = 0.35


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


def _apply_perspective_jitter(
    image: np.ndarray,
    corners: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a mild perspective warp and remap the corner targets."""
    if random.random() >= _PERSPECTIVE_JITTER_PROBABILITY:
        return image, corners

    height, width = image.shape[:2]
    source = np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float32,
    )
    max_dx = width * _MAX_CORNER_JITTER_RATIO
    max_dy = height * _MAX_CORNER_JITTER_RATIO
    destination = source + np.array(
        [
            [random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy)],
            [random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy)],
            [random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy)],
            [random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy)],
        ],
        dtype=np.float32,
    )
    destination[:, 0] = np.clip(destination[:, 0], 0.0, width - 1.0)
    destination[:, 1] = np.clip(destination[:, 1], 0.0, height - 1.0)

    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
    )
    remapped_corners = cv2.perspectiveTransform(
        corners.reshape(1, -1, 2),
        matrix,
    ).reshape(-1, 2)
    remapped_corners[:, 0] = np.clip(remapped_corners[:, 0], 0.0, width - 1.0)
    remapped_corners[:, 1] = np.clip(remapped_corners[:, 1], 0.0, height - 1.0)
    return warped, remapped_corners.astype(np.float32)


def _apply_translation_jitter(
    image: np.ndarray,
    corners: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate the image within a larger replicated canvas and remap corners."""
    if random.random() >= _TRANSLATION_JITTER_PROBABILITY:
        return image, corners

    height, width = image.shape[:2]
    max_pad_x = max(int(round(width * _MAX_CANVAS_EXPANSION_RATIO)), 1)
    max_pad_y = max(int(round(height * _MAX_CANVAS_EXPANSION_RATIO)), 1)
    pad_left = random.randint(0, max_pad_x)
    pad_right = random.randint(0, max_pad_x)
    pad_top = random.randint(0, max_pad_y)
    pad_bottom = random.randint(0, max_pad_y)
    expanded = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REPLICATE,
    )

    available_x = expanded.shape[1] - width
    available_y = expanded.shape[0] - height
    offset_x = random.randint(0, available_x) if available_x > 0 else 0
    offset_y = random.randint(0, available_y) if available_y > 0 else 0
    translated = expanded[offset_y : offset_y + height, offset_x : offset_x + width]

    remapped_corners = corners.copy()
    remapped_corners[:, 0] += float(pad_left - offset_x)
    remapped_corners[:, 1] += float(pad_top - offset_y)
    remapped_corners[:, 0] = np.clip(remapped_corners[:, 0], 0.0, width - 1.0)
    remapped_corners[:, 1] = np.clip(remapped_corners[:, 1], 0.0, height - 1.0)
    return translated, remapped_corners.astype(np.float32)


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


class BoardLocalizationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
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

    def sample_ids(self) -> list[str]:
        """Return stable sample ids for weighting and diagnostics."""
        return [str(record["image_path"]) for record in self._records]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self._records[idx]
        image_path = self._root / record["image_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read board-localizer image: {image_path}")
        corners = np.array(record["board_corners"], dtype=np.float32)
        if self._augment:
            image, corners = _apply_translation_jitter(image, corners)
            image, corners = _apply_perspective_jitter(image, corners)
            image = _augment_localizer_sample(image)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor: torch.Tensor = self._transform(rgb)  # type: ignore[assignment]
        normalized = normalize_corners(corners, image.shape[1], image.shape[0]).reshape(-1)
        target = torch.tensor(normalized, dtype=torch.float32)
        size = torch.tensor(
            [float(image.shape[1]), float(image.shape[0])],
            dtype=torch.float32,
        )
        return tensor, target, size
