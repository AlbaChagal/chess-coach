"""Dataset helpers for piece detection on board images."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

Split = Literal["train", "val", "test"]
_SCALE_JITTER_MIN = 0.9
_SCALE_JITTER_MAX = 1.1
_MAX_BRIGHTNESS_SHIFT = 16.0
_MIN_CONTRAST_SCALE = 0.9
_MAX_CONTRAST_SCALE = 1.1
_HORIZONTAL_FLIP_PROBABILITY = 0.5
_BLUR_PROBABILITY = 0.15


def _apply_horizontal_flip(
    image: np.ndarray,
    boxes: torch.Tensor,
) -> tuple[np.ndarray, torch.Tensor]:
    """Flip the sample horizontally and remap bounding boxes."""
    flipped = np.ascontiguousarray(image[:, ::-1])
    width = image.shape[1]
    remapped_boxes = boxes.clone()
    remapped_boxes[:, 0] = width - boxes[:, 2]
    remapped_boxes[:, 2] = width - boxes[:, 0]
    return flipped, remapped_boxes


def _apply_color_jitter(image: np.ndarray) -> np.ndarray:
    """Apply mild brightness and contrast jitter."""
    alpha = random.uniform(_MIN_CONTRAST_SCALE, _MAX_CONTRAST_SCALE)
    beta = random.uniform(-_MAX_BRIGHTNESS_SHIFT, _MAX_BRIGHTNESS_SHIFT)
    jittered = image.astype(np.float32) * alpha + beta
    return np.clip(jittered, 0, 255).astype(np.uint8)


def _apply_scale_jitter(
    image: np.ndarray,
    boxes: torch.Tensor,
) -> tuple[np.ndarray, torch.Tensor]:
    """Apply isotropic scale jitter while keeping the original canvas size."""
    scale = random.uniform(_SCALE_JITTER_MIN, _SCALE_JITTER_MAX)
    if abs(scale - 1.0) < 1e-3:
        return image, boxes

    height, width = image.shape[:2]
    scaled_width = max(int(round(width * scale)), 1)
    scaled_height = max(int(round(height * scale)), 1)
    scaled_image = cv2.resize(image, (scaled_width, scaled_height))
    scaled_boxes = boxes.clone()
    scaled_boxes[:, [0, 2]] *= scale
    scaled_boxes[:, [1, 3]] *= scale

    if scale >= 1.0:
        offset_x = (scaled_width - width) // 2
        offset_y = (scaled_height - height) // 2
        cropped_image = scaled_image[offset_y : offset_y + height, offset_x : offset_x + width]
        scaled_boxes[:, [0, 2]] -= float(offset_x)
        scaled_boxes[:, [1, 3]] -= float(offset_y)
        scaled_boxes[:, [0, 2]] = scaled_boxes[:, [0, 2]].clamp(0.0, float(width))
        scaled_boxes[:, [1, 3]] = scaled_boxes[:, [1, 3]].clamp(0.0, float(height))
        return cropped_image, scaled_boxes

    canvas = np.zeros_like(image)
    offset_x = (width - scaled_width) // 2
    offset_y = (height - scaled_height) // 2
    canvas[offset_y : offset_y + scaled_height, offset_x : offset_x + scaled_width] = scaled_image
    scaled_boxes[:, [0, 2]] += float(offset_x)
    scaled_boxes[:, [1, 3]] += float(offset_y)
    return canvas, scaled_boxes


def _apply_blur(image: np.ndarray) -> np.ndarray:
    """Apply a mild blur augmentation."""
    return cv2.GaussianBlur(image, (3, 3), 0)


def _augment_detection_sample(
    image: np.ndarray,
    boxes: torch.Tensor,
) -> tuple[np.ndarray, torch.Tensor]:
    """Apply box-safe augmentations to a detector training sample."""
    augmented_image = image
    augmented_boxes = boxes

    if random.random() < _HORIZONTAL_FLIP_PROBABILITY:
        augmented_image, augmented_boxes = _apply_horizontal_flip(
            augmented_image,
            augmented_boxes,
        )
    augmented_image = _apply_color_jitter(augmented_image)
    augmented_image, augmented_boxes = _apply_scale_jitter(
        augmented_image,
        augmented_boxes,
    )
    if random.random() < _BLUR_PROBABILITY:
        augmented_image = _apply_blur(augmented_image)
    return augmented_image, augmented_boxes


class DetectionDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """Load detector images and targets from a JSONL manifest."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        split: Split,
        root: Path | None = None,
        image_size: int | None = None,
        augment: bool = False,
    ) -> None:
        self._root = root or manifest_path.parent
        self._image_size = image_size
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        record = self._records[idx]
        image_path = self._root / record["image_path"]
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read detector image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        boxes = torch.tensor(
            [annotation["box"] for annotation in record["annotations"]],
            dtype=torch.float32,
        )
        if self._augment:
            rgb, boxes = _augment_detection_sample(rgb, boxes)
        if self._image_size is not None:
            height, width = rgb.shape[:2]
            rgb = cv2.resize(rgb, (self._image_size, self._image_size))
            scale_x = self._image_size / width
            scale_y = self._image_size / height
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        labels = torch.tensor(
            [annotation["label_index"] for annotation in record["annotations"]],
            dtype=torch.int64,
        )
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        return image, target


def detection_collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Collate detection samples into the list-based format torchvision expects."""
    images, targets = zip(*batch)
    return list(images), list(targets)
