"""Tests for detector/localizer training script manifest resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import train_board_localizer, train_detector


def test_train_detector_resolve_manifest_uses_explicit_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"

    resolved = train_detector._resolve_manifest_path(
        manifest_path=manifest_path,
        raw_input=None,
        prepared_output=None,
    )

    assert resolved == manifest_path


def test_train_detector_resolve_manifest_prepares_from_raw_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_input = tmp_path / "raw"
    prepared_output = tmp_path / "prepared"
    expected_manifest = prepared_output / "manifest.jsonl"

    monkeypatch.setattr(
        train_detector,
        "prepare_detection_dataset",
        lambda raw, output: expected_manifest,
    )

    resolved = train_detector._resolve_manifest_path(
        manifest_path=None,
        raw_input=raw_input,
        prepared_output=prepared_output,
    )

    assert resolved == expected_manifest


def test_train_board_localizer_resolve_manifest_uses_explicit_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.jsonl"

    resolved = train_board_localizer._resolve_manifest_path(
        manifest_path=manifest_path,
        raw_input=None,
        prepared_output=None,
    )

    assert resolved == manifest_path


def test_train_board_localizer_resolve_manifest_prepares_from_raw_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_input = tmp_path / "raw"
    prepared_output = tmp_path / "prepared"
    expected_manifest = prepared_output / "manifest.jsonl"

    monkeypatch.setattr(
        train_board_localizer,
        "prepare_board_localizer_dataset",
        lambda raw, output: expected_manifest,
    )

    resolved = train_board_localizer._resolve_manifest_path(
        manifest_path=None,
        raw_input=raw_input,
        prepared_output=prepared_output,
    )

    assert resolved == expected_manifest


def test_train_scripts_require_manifest_or_raw_input() -> None:
    with pytest.raises(ValueError, match="Either manifest_path or raw_input"):
        train_detector._resolve_manifest_path(
            manifest_path=None,
            raw_input=None,
            prepared_output=None,
        )

    with pytest.raises(ValueError, match="Either manifest_path or raw_input"):
        train_board_localizer._resolve_manifest_path(
            manifest_path=None,
            raw_input=None,
            prepared_output=None,
        )
