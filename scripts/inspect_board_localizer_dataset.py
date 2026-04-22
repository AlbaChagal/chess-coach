"""Inspect board-localizer training samples and dataset geometry."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from chesscoach.logging_utils import add_logging_args, configure_logging
from chesscoach.vision.board_detector import BOARD_SIZE, warp_board_from_corners
from chesscoach.vision.board_localizer import normalize_corners
from chesscoach.vision.board_localizer_dataset import (
    _apply_perspective_jitter,
    _apply_translation_jitter,
    _augment_localizer_sample,
)

LOGGER = logging.getLogger(__name__)
_PANEL_BACKGROUND = (18, 18, 18)
_TEXT_COLOR = (240, 240, 240)
_MUTED_TEXT_COLOR = (170, 170, 170)
_TARGET_COLOR = (255, 0, 255)


def _load_manifest_records(manifest_path: Path, split: str) -> list[dict[str, Any]]:
    """Load manifest records for one split."""
    records: list[dict[str, Any]] = []
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["split"] == split:
            records.append(record)
    if not records:
        raise ValueError(f"No board-localizer records found for split={split}")
    return records


def _draw_polygon(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Return a copy of an image with labeled target corners."""
    annotated = image.copy()
    points = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        annotated,
        [points],
        isClosed=True,
        color=_TARGET_COLOR,
        thickness=3,
    )
    for index, point in enumerate(corners.astype(np.int32)):
        cv2.circle(annotated, tuple(point), 6, _TARGET_COLOR, -1)
        cv2.putText(
            annotated,
            str(index),
            (int(point[0]) + 8, int(point[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            _TARGET_COLOR,
            2,
            cv2.LINE_AA,
        )
    return annotated


def _fit_width(image: np.ndarray, width: int) -> np.ndarray:
    """Resize an image to a target width while preserving aspect ratio."""
    current_height, current_width = image.shape[:2]
    if current_width == width:
        return image
    scale = width / current_width
    resized_height = max(1, int(round(current_height * scale)))
    return cv2.resize(image, (width, resized_height))


def _quadrilateral_area(corners: np.ndarray) -> float:
    """Return the area of a four-corner polygon."""
    x_coords = corners[:, 0]
    y_coords = corners[:, 1]
    cross_terms = x_coords * np.roll(y_coords, -1)
    reverse_cross_terms = y_coords * np.roll(x_coords, -1)
    return float(0.5 * abs(np.sum(cross_terms - reverse_cross_terms)))


def _edge_lengths(corners: np.ndarray) -> list[float]:
    """Return ordered edge lengths of a quadrilateral."""
    deltas = corners - np.roll(corners, -1, axis=0)
    return [float(np.linalg.norm(delta)) for delta in deltas]


def _corner_center(corners: np.ndarray) -> tuple[float, float]:
    """Return the quadrilateral center as the mean of its corners."""
    center = corners.mean(axis=0)
    return float(center[0]), float(center[1])


def _geometry_summary(
    corners: np.ndarray,
    width: int,
    height: int,
) -> dict[str, float | list[float]]:
    """Summarize board geometry in normalized coordinates."""
    normalized = normalize_corners(corners, width, height)
    normalized_corners = normalized.reshape(4, 2)
    normalized_area = _quadrilateral_area(normalized_corners)
    center_x, center_y = _corner_center(normalized_corners)
    return {
        "normalized_area": normalized_area,
        "normalized_center_x": center_x,
        "normalized_center_y": center_y,
        "normalized_edge_lengths": _edge_lengths(normalized_corners),
    }


def _apply_sample_augmentation(
    image: np.ndarray,
    corners: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same stochastic augmentations used during training."""
    random.seed(seed)
    augmented_image, augmented_corners = _apply_translation_jitter(image, corners)
    augmented_image, augmented_corners = _apply_perspective_jitter(
        augmented_image,
        augmented_corners,
    )
    augmented_image = _augment_localizer_sample(augmented_image)
    return augmented_image, augmented_corners


def _build_info_lines(
    *,
    image_path: Path,
    raw_summary: dict[str, float | list[float]],
    augmented_summary: dict[str, float | list[float]],
    augmented: bool,
) -> list[str]:
    """Build text lines for one inspection sample."""
    mode = "augmented" if augmented else "raw"
    return [
        f"image: {image_path.name}",
        f"mode: {mode}",
        (
            "raw normalized area="
            f"{float(raw_summary['normalized_area']):.4f} "
            "center=("
            f"{float(raw_summary['normalized_center_x']):.4f}, "
            f"{float(raw_summary['normalized_center_y']):.4f})"
        ),
        (
            "raw normalized edges="
            f"{[round(value, 4) for value in raw_summary['normalized_edge_lengths']]}"
        ),
        (
            "target normalized area="
            f"{float(augmented_summary['normalized_area']):.4f} "
            "center=("
            f"{float(augmented_summary['normalized_center_x']):.4f}, "
            f"{float(augmented_summary['normalized_center_y']):.4f})"
        ),
        (
            "target normalized edges="
            f"{[round(value, 4) for value in augmented_summary['normalized_edge_lengths']]}"
        ),
    ]


def _build_panel(
    image_path: Path,
    raw_image: np.ndarray,
    target_image: np.ndarray,
    raw_corners: np.ndarray,
    target_corners: np.ndarray,
    *,
    augmented: bool,
) -> tuple[np.ndarray, list[str], dict[str, float | list[float]]]:
    """Build one visual inspection panel and return geometry details."""
    raw_summary = _geometry_summary(raw_corners, raw_image.shape[1], raw_image.shape[0])
    target_summary = _geometry_summary(
        target_corners,
        target_image.shape[1],
        target_image.shape[0],
    )
    raw_panel = _fit_width(raw_image, 700)
    target_panel = _fit_width(_draw_polygon(target_image, target_corners), 700)
    warp = warp_board_from_corners(target_image, target_corners)
    warp_panel = cv2.resize(warp, (BOARD_SIZE, BOARD_SIZE))
    warp_panel = _fit_width(warp_panel, 700)

    top_height = max(raw_panel.shape[0], target_panel.shape[0], warp_panel.shape[0])
    info_lines = _build_info_lines(
        image_path=image_path,
        raw_summary=raw_summary,
        augmented_summary=target_summary,
        augmented=augmented,
    )
    panel_height = top_height + 220
    panel_width = raw_panel.shape[1] + target_panel.shape[1] + warp_panel.shape[1]
    panel = np.full((panel_height, panel_width, 3), _PANEL_BACKGROUND, dtype=np.uint8)
    left_end = raw_panel.shape[1]
    middle_end = left_end + target_panel.shape[1]
    panel[: raw_panel.shape[0], :left_end] = raw_panel
    panel[: target_panel.shape[0], left_end:middle_end] = target_panel
    panel[: warp_panel.shape[0], middle_end : middle_end + warp_panel.shape[1]] = (
        warp_panel
    )

    text_y = top_height + 28
    for line in info_lines:
        cv2.putText(
            panel,
            line,
            (18, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            _TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
        text_y += 28
    cv2.putText(
        panel,
        "left: raw image    middle: model target image with target corners    right: warp from target corners",
        (18, text_y + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        _MUTED_TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    return panel, info_lines, target_summary


def _aggregate_split_stats(
    records: list[dict[str, Any]],
) -> dict[str, float | int | list[float]]:
    """Compute split-level geometry statistics from manifest records."""
    areas: list[float] = []
    center_xs: list[float] = []
    center_ys: list[float] = []
    all_edge_lengths: list[list[float]] = []
    for record in records:
        corners = np.array(record["board_corners"], dtype=np.float32)
        width = int(record["width"])
        height = int(record["height"])
        summary = _geometry_summary(corners, width, height)
        areas.append(float(summary["normalized_area"]))
        center_xs.append(float(summary["normalized_center_x"]))
        center_ys.append(float(summary["normalized_center_y"]))
        all_edge_lengths.append(
            [float(value) for value in summary["normalized_edge_lengths"]]
        )

    edge_array = np.array(all_edge_lengths, dtype=np.float32)
    return {
        "count": len(records),
        "normalized_area_mean": float(np.mean(areas)),
        "normalized_area_min": float(np.min(areas)),
        "normalized_area_max": float(np.max(areas)),
        "normalized_center_x_mean": float(np.mean(center_xs)),
        "normalized_center_y_mean": float(np.mean(center_ys)),
        "normalized_edge_length_mean": edge_array.mean(axis=0).round(6).tolist(),
        "normalized_edge_length_min": edge_array.min(axis=0).round(6).tolist(),
        "normalized_edge_length_max": edge_array.max(axis=0).round(6).tolist(),
    }


def inspect_board_localizer_dataset(
    manifest_path: Path,
    output_dir: Path,
    *,
    split: str,
    limit: int,
    seed: int,
    augment: bool,
) -> list[Path]:
    """Write board-localizer inspection panels and a split summary."""
    records = _load_manifest_records(manifest_path, split)
    randomizer = random.Random(seed)
    selected_records = randomizer.sample(records, k=min(limit, len(records)))
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _aggregate_split_stats(records)
    summary_path = output_dir / f"{split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    written: list[Path] = []
    for index, record in enumerate(selected_records, start=1):
        image_path = Path(record["image_path"])
        raw_image = cv2.imread(str(image_path))
        if raw_image is None:
            raise FileNotFoundError(f"Could not read board-localizer image: {image_path}")
        raw_corners = np.array(record["board_corners"], dtype=np.float32)
        if augment:
            target_image, target_corners = _apply_sample_augmentation(
                raw_image,
                raw_corners,
                seed=seed + index,
            )
        else:
            target_image, target_corners = raw_image.copy(), raw_corners.copy()
        panel, info_lines, target_summary = _build_panel(
            image_path,
            raw_image,
            target_image,
            raw_corners,
            target_corners,
            augmented=augment,
        )
        output_path = output_dir / f"{index:03d}_{image_path.stem}.jpg"
        cv2.imwrite(str(output_path), panel)
        info_path = output_dir / f"{index:03d}_{image_path.stem}.txt"
        info_payload = {
            "image_path": str(image_path),
            "split": split,
            "augment": augment,
            "info_lines": info_lines,
            "target_summary": target_summary,
        }
        info_path.write_text(json.dumps(info_payload, indent=2) + "\n")
        written.append(output_path)

    LOGGER.info(
        "Board-localizer dataset inspection written count=%s split=%s output=%s augment=%s seed=%s",
        len(written),
        split,
        output_dir,
        augment,
        seed,
    )
    return written


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect board-localizer samples and split geometry."
    )
    add_logging_args(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply the same translation/perspective/image augmentations used in training.",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    inspect_board_localizer_dataset(
        args.manifest,
        args.output,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        augment=args.augment,
    )


if __name__ == "__main__":
    main()
