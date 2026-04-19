"""Tests for suspicious-board filtering utility."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.filter_suspicious_orientation_boards import (
    _filter_manifest,
    suspicious_board_paths,
)


def test_suspicious_board_paths_filters_by_mismatch_threshold(tmp_path: Path) -> None:
    audit_report = tmp_path / "audit.json"
    audit_report.write_text(
        json.dumps(
            {
                "boards": [
                    {"image_path": "/tmp/a.png", "mismatches": 0},
                    {"image_path": "/tmp/b.png", "mismatches": 2},
                ]
            }
        )
    )

    suspicious = suspicious_board_paths(audit_report)

    assert suspicious == ["/tmp/b.png"]


def test_filter_manifest_removes_suspicious_records(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({"image_path": "/tmp/a.png", "split": "train"}) + "\n"
        + json.dumps({"image_path": "/tmp/b.png", "split": "train"}) + "\n"
    )
    output_path = tmp_path / "filtered.jsonl"

    kept, removed = _filter_manifest(
        manifest_path,
        output_path,
        {"/tmp/b.png"},
    )

    assert (kept, removed) == (1, 1)
    lines = output_path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["image_path"] == "/tmp/a.png"
