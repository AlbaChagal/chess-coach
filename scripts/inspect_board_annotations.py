"""Render visual audits for raw board annotations and converted geometry."""

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

LOGGER = logging.getLogger(__name__)
_IMAGE_PATTERNS = ("*.jpg", "*.png")
_PANEL_BACKGROUND = (18, 18, 18)
_TEXT_COLOR = (240, 240, 240)
_ANNOTATION_COLOR = (0, 255, 255)


def _iter_split_images(split_dir: Path) -> list[Path]:
    """Return sorted image paths for a split directory."""
    image_paths = [
        image_path
        for pattern in _IMAGE_PATTERNS
        for image_path in split_dir.glob(pattern)
    ]
    return sorted(image_paths)


def _load_json_payload(image_path: Path) -> dict[str, Any]:
    """Load a sidecar JSON payload for an image."""
    payload = json.loads(image_path.with_suffix(".json").read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload for {image_path}")
    return payload


def _has_usable_corners(payload: dict[str, Any]) -> bool:
    """Return whether a payload contains four board corners."""
    raw_corners = payload.get("corners")
    return isinstance(raw_corners, list) and len(raw_corners) == 4


def _draw_polygon(
    image: cv2.typing.MatLike,
    corners: np.ndarray,
    *,
    color: tuple[int, int, int],
) -> None:
    """Draw a labeled polygon on an image."""
    points = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=3)
    for index, point in enumerate(corners.astype(np.int32)):
        cv2.circle(image, tuple(point), 6, color, -1)
        cv2.putText(
            image,
            str(index),
            (int(point[0]) + 8, int(point[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def _fit_width(image: np.ndarray, width: int) -> np.ndarray:
    """Resize an image to a target width, preserving aspect ratio."""
    current_height, current_width = image.shape[:2]
    if current_width == width:
        return image
    scale = width / current_width
    resized_height = max(1, int(round(current_height * scale)))
    return cv2.resize(image, (width, resized_height))


def _build_info_lines(payload: dict[str, Any], image_path: Path) -> list[str]:
    """Build text lines describing a sample."""
    orientation_schema = payload.get("orientation_schema") or {}
    image_space = orientation_schema.get("image_space") or {}
    orientation = payload.get("orientation") or {}
    source = payload.get("source") or {}
    lines = [
        f"image: {image_path.name}",
        f"fen: {payload.get('fen', '')}",
        f"corners_known: {image_space.get('known', False)}",
        f"white_side: {orientation.get('white_side', 'unknown')}",
        f"rotation_to_white_bottom: {orientation.get('rotation_to_white_bottom', 'n/a')}",
        f"game_id: {source.get('game_id', 'n/a')}  move_id: {source.get('move_id', 'n/a')}",
    ]
    return lines


def _build_panel(
    image_path: Path,
    payload: dict[str, Any],
) -> np.ndarray:
    """Build a side-by-side inspection panel for one sample."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Missing image: {image_path}")

    corners = np.array(payload["corners"], dtype=np.float32)
    annotated = image.copy()
    _draw_polygon(annotated, corners, color=_ANNOTATION_COLOR)

    warped = warp_board_from_corners(image, corners)
    warped = cv2.resize(warped, (BOARD_SIZE, BOARD_SIZE))

    left = _fit_width(annotated, 1000)
    right = _fit_width(warped, 1000)
    panel_height = max(left.shape[0], right.shape[0]) + 190
    panel_width = left.shape[1] + right.shape[1]
    panel = np.full((panel_height, panel_width, 3), _PANEL_BACKGROUND, dtype=np.uint8)
    panel[: left.shape[0], : left.shape[1]] = left
    panel[: right.shape[0], left.shape[1] : left.shape[1] + right.shape[1]] = right

    info_lines = _build_info_lines(payload, image_path)
    text_y = max(left.shape[0], right.shape[0]) + 28
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
        "left: raw image with annotated corners    right: warped board from those corners",
        (18, text_y + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )
    return panel


def inspect_board_annotations(
    input_dir: Path,
    output_dir: Path,
    *,
    split: str,
    limit: int,
    seed: int,
) -> list[Path]:
    """Write inspection panels for a random sample of boards with corners."""
    split_dir = input_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    candidates: list[Path] = []
    for image_path in _iter_split_images(split_dir):
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            continue
        payload = _load_json_payload(image_path)
        if _has_usable_corners(payload):
            candidates.append(image_path)

    if not candidates:
        raise ValueError(f"No images with usable corners found under {split_dir}")

    randomizer = random.Random(seed)
    selected = randomizer.sample(candidates, k=min(limit, len(candidates)))
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for index, image_path in enumerate(selected, start=1):
        payload = _load_json_payload(image_path)
        panel = _build_panel(image_path, payload)
        output_path = output_dir / f"{index:03d}_{image_path.stem}.jpg"
        cv2.imwrite(str(output_path), panel)
        written.append(output_path)

        sidecar_path = output_dir / f"{index:03d}_{image_path.stem}.txt"
        sidecar_path.write_text(
            "\n".join(_build_info_lines(payload, image_path)) + "\n"
        )

    LOGGER.info(
        "Board annotation inspection written count=%s split=%s output=%s seed=%s",
        len(written),
        split,
        output_dir,
        seed,
    )
    return written


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Write visual audits for board-corner annotations."
    )
    add_logging_args(parser)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    inspect_board_annotations(
        args.input,
        args.output,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
