"""Prepare a board-corner localization manifest from raw board annotations."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from chesscoach.logging_utils import add_logging_args, configure_logging

try:
    from scripts.prepare_detection_dataset import select_metadata_corners
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from scripts.prepare_detection_dataset import select_metadata_corners

LOGGER = logging.getLogger(__name__)
_IMAGE_PATTERNS = ("*.jpg", "*.png")


def _load_json_payload(image_path: Path) -> dict[str, Any]:
    payload = json.loads(image_path.with_suffix(".json").read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload for {image_path}")
    return payload


def prepare_board_localizer_dataset(input_dir: Path, output_dir: Path) -> Path:
    """Prepare raw-image board-corner regression annotations from raw data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    with manifest_path.open("w") as manifest_file:
        for split in ("train", "val", "test"):
            split_dir = input_dir / split
            if not split_dir.exists():
                LOGGER.warning(f"Split directory not found: {split_dir}")
                continue
            image_paths = sorted(
                image_path
                for pattern in _IMAGE_PATTERNS
                for image_path in split_dir.glob(pattern)
            )
            LOGGER.info(
                f"Preparing board-localizer split={split} images={len(image_paths)}"
            )
            for image_path in image_paths:
                json_path = image_path.with_suffix(".json")
                if not json_path.exists():
                    continue
                payload = _load_json_payload(image_path)
                ordered_corners = select_metadata_corners(payload)
                record = {
                    "image_path": str(image_path.resolve()),
                    "split": split,
                    "width": payload.get("width"),
                    "height": payload.get("height"),
                    "board_corners": ordered_corners.tolist(),
                }
                manifest_file.write(json.dumps(record) + "\n")

    LOGGER.info(f"Board-localizer manifest written to {manifest_path}")
    return manifest_path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare a board-localizer manifest from raw annotations."
    )
    add_logging_args(parser)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/chess_boards/board_localizer"),
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    prepare_board_localizer_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
