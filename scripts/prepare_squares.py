"""Convert raw chesscog board images + FEN labels into labeled 100×100 square images.

For each board image the script:
1. Runs :func:`~chesscoach.vision.board_detector.detect_board` to produce a
   warped 512×512 top-down view.
2. Splits it into 64 squares.
3. Labels each square from the FEN string.
4. Saves the 100×100 square image under ``output/{label}/``.
5. Records the sample in a CSV manifest (``output/squares.csv``).

Boards that fail detection are skipped and logged to stderr.

Usage::

    uv run python scripts/prepare_squares.py \\
        --input  data/chesscog/raw \\
        --output data/chesscog/squares

Expected input layout (chesscog convention)::

    data/chesscog/raw/
        train/
            <id>.jpg   (board image)
            <id>.fen   (FEN string, first line used)
        val/
            ...
        test/
            ...

Output layout::

    data/chesscog/squares/
        wP/  wN/  wB/  wR/  wQ/  wK/
        bP/  bN/  bB/  bR/  bQ/  bK/
        empty/
        squares.csv     # columns: image_path, label, split
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

from chesscoach.vision.board_detector import BoardNotFoundError, detect_board, split_into_squares
from chesscoach.vision.fen_builder import _LABEL_TO_FEN  # reuse the label mapping
from chesscoach.vision.types import PIECE_LABELS, PieceLabel

_SQUARE_SIZE = 100

# Invert the label→FEN char map to get FEN char→label
_FEN_TO_LABEL: dict[str, PieceLabel] = {
    v: k for k, v in _LABEL_TO_FEN.items() if v != ""
}
_FEN_TO_LABEL["."] = "empty"


def _fen_to_grid(fen_placement: str) -> list[list[PieceLabel]]:
    """Expand a FEN piece-placement string into an 8×8 label grid."""
    grid: list[list[PieceLabel]] = []
    for rank_str in fen_placement.split("/"):
        row: list[PieceLabel] = []
        for ch in rank_str:
            if ch.isdigit():
                row.extend(["empty"] * int(ch))  # type: ignore[arg-type]
            else:
                label = _FEN_TO_LABEL.get(ch)
                if label is None:
                    raise ValueError(f"Unknown FEN character: {ch!r}")
                row.append(label)
        if len(row) != 8:
            raise ValueError(f"Rank does not have 8 squares: {rank_str!r}")
        grid.append(row)
    if len(grid) != 8:
        raise ValueError(f"FEN does not have 8 ranks: {fen_placement!r}")
    return grid


def _read_fen_placement(image_path: Path) -> str | None:
    """Read the piece-placement FEN for a board image sidecar."""
    fen_path = image_path.with_suffix(".fen")
    if fen_path.exists():
        return fen_path.read_text().strip().split()[0]

    json_path = image_path.with_suffix(".json")
    if not json_path.exists():
        return None

    payload = json.loads(json_path.read_text())
    fen = payload.get("fen")
    if not isinstance(fen, str) or not fen.strip():
        raise ValueError(f"Missing 'fen' string in {json_path}")
    return fen.strip().split()[0]


def prepare_squares(input_dir: Path, output_dir: Path) -> None:
    """Convert raw board images to labeled square images.

    Args:
        input_dir: Root of the raw chesscog dataset (contains train/val/test).
        output_dir: Root of the output square image dataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in PIECE_LABELS:
        (output_dir / label).mkdir(exist_ok=True)

    csv_path = output_dir / "squares.csv"
    splits = ["train", "val", "test"]

    n_saved = 0
    n_skipped = 0

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_path", "label", "split"])
        writer.writeheader()

        for split in splits:
            split_dir = input_dir / split
            if not split_dir.exists():
                print(f"  [WARN] split directory not found: {split_dir}", file=sys.stderr)
                continue

            image_paths = sorted(split_dir.glob("*.jpg")) + sorted(split_dir.glob("*.png"))
            for img_path in tqdm(image_paths, desc=split):
                try:
                    fen_placement = _read_fen_placement(img_path)
                except ValueError as exc:
                    print(f"  [SKIP] bad label file for {img_path.name}: {exc}", file=sys.stderr)
                    n_skipped += 1
                    continue

                if fen_placement is None:
                    print(
                        f"  [SKIP] no .fen or .json label file for {img_path.name}",
                        file=sys.stderr,
                    )
                    n_skipped += 1
                    continue

                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    print(f"  [SKIP] cannot read image: {img_path}", file=sys.stderr)
                    n_skipped += 1
                    continue

                try:
                    warped = detect_board(bgr)
                except BoardNotFoundError as exc:
                    print(f"  [SKIP] board not found in {img_path.name}: {exc}", file=sys.stderr)
                    n_skipped += 1
                    continue

                try:
                    grid = _fen_to_grid(fen_placement)
                except ValueError as exc:
                    print(f"  [SKIP] bad FEN in {img_path.name}: {exc}", file=sys.stderr)
                    n_skipped += 1
                    continue

                squares = split_into_squares(warped)
                for row_idx, (sq_row, label_row) in enumerate(zip(squares, grid)):
                    for col_idx, (sq, label) in enumerate(zip(sq_row, label_row)):
                        resized = cv2.resize(sq, (_SQUARE_SIZE, _SQUARE_SIZE))
                        sq_filename = f"{img_path.stem}_r{row_idx}c{col_idx}.jpg"
                        sq_path = output_dir / label / sq_filename
                        cv2.imwrite(str(sq_path), resized)
                        writer.writerow(
                            {
                                "image_path": str(sq_path.relative_to(output_dir)),
                                "label": label,
                                "split": split,
                            }
                        )
                        n_saved += 1

    print(f"\nDone. Saved {n_saved} squares, skipped {n_skipped} boards.")
    print(f"Manifest written to {csv_path}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert chesscog board images into labeled 100×100 square images."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Root of the raw chesscog dataset (contains train/val/test sub-dirs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/chesscog/squares"),
        help="Output directory for square images and manifest CSV.",
    )
    args = parser.parse_args(argv)
    prepare_squares(args.input, args.output)


if __name__ == "__main__":
    main()
