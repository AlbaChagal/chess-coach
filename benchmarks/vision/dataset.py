"""Load a labeled image dataset for benchmarking the vision pipeline."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BoardSample:
    """A single labeled sample: board image path + expected FEN placement."""

    image_path: Path
    fen_placement: str


def load_csv(path: Path) -> list[BoardSample]:
    """Load a dataset from a CSV file with ``image_path`` and ``fen`` columns.

    Args:
        path: Path to the CSV file.

    Returns:
        List of :class:`BoardSample` instances.

    Raises:
        ValueError: If a required column is missing.
    """
    samples: list[BoardSample] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {
            "image_path",
            "fen",
        }.issubset(set(reader.fieldnames)):
            raise ValueError(
                "CSV must have 'image_path' and 'fen' columns. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            fen_placement = row["fen"].split()[0]  # take only piece-placement field
            samples.append(
                BoardSample(
                    image_path=Path(row["image_path"]),
                    fen_placement=fen_placement,
                )
            )
    return samples


def load_json(path: Path) -> list[BoardSample]:
    """Load a dataset from a JSON file (list of ``{image_path, fen}`` objects).

    Args:
        path: Path to the JSON file.

    Returns:
        List of :class:`BoardSample` instances.
    """
    data: list[dict[str, str]] = json.loads(path.read_text())
    return [
        BoardSample(
            image_path=Path(item["image_path"]),
            fen_placement=item["fen"].split()[0],
        )
        for item in data
    ]
