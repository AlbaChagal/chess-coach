"""Tests for detector/localizer training script manifest resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

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


def test_edge_length_features_returns_ordered_board_edge_lengths() -> None:
    corners = torch.tensor([[0.0, 0.0, 4.0, 0.0, 4.0, 3.0, 0.0, 3.0]])

    edge_lengths = train_board_localizer._edge_length_features(corners)

    assert torch.allclose(edge_lengths, torch.tensor([[4.0, 3.0, 4.0, 3.0]]))


def test_board_localizer_loss_is_zero_when_predictions_match_targets() -> None:
    criterion = torch.nn.SmoothL1Loss()
    corners = torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]])

    loss = train_board_localizer._board_localizer_loss(corners, corners, criterion)

    assert loss.item() == pytest.approx(0.0)


def test_board_localizer_loss_penalizes_inset_quadrilateral() -> None:
    criterion = torch.nn.SmoothL1Loss()
    targets = torch.tensor([[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])
    predictions = torch.tensor([[1.0, 1.0, 9.0, 1.0, 9.0, 9.0, 1.0, 9.0]])

    corner_only_loss = criterion(predictions, targets).item()
    combined_loss = train_board_localizer._board_localizer_loss(
        predictions,
        targets,
        criterion,
    ).item()

    assert combined_loss > corner_only_loss


def test_quadrilateral_area_returns_expected_value() -> None:
    corners = torch.tensor([[0.0, 0.0, 4.0, 0.0, 4.0, 3.0, 0.0, 3.0]])

    area = train_board_localizer._quadrilateral_area(corners)

    assert torch.allclose(area, torch.tensor([12.0]))


def test_board_localizer_loss_components_include_area_penalty() -> None:
    criterion = torch.nn.SmoothL1Loss()
    targets = torch.tensor([[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])
    predictions = torch.tensor([[1.0, 1.0, 9.0, 1.0, 9.0, 9.0, 1.0, 9.0]])

    loss_components = train_board_localizer._board_localizer_loss_components(
        predictions,
        targets,
        criterion,
    )

    assert loss_components["corner"].item() > 0.0
    assert loss_components["edge"].item() > 0.0
    assert loss_components["area"].item() > 0.0
    assert loss_components["total"].item() > loss_components["corner"].item()
