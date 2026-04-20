"""Tests for consolidated vision evaluation script helpers."""

from __future__ import annotations

from pathlib import Path

from scripts import evaluate_vision_models
from scripts.evaluate_vision_models import (
    _parse_model_spec,
    _resolve_benchmark_dataset,
    _summary_row,
)


def test_parse_model_spec_requires_name_and_detector() -> None:
    spec = _parse_model_spec(
        "name=baseline,detector=models/piece_detector.pt,localizer=models/board_localizer.pt"
    )

    assert spec.name == "baseline"
    assert spec.detector_checkpoint == Path("models/piece_detector.pt")
    assert spec.board_localizer_checkpoint == Path("models/board_localizer.pt")


def test_summary_row_formats_present_and_missing_metrics() -> None:
    row = _summary_row(
        {
            "model": {"name": "baseline"},
            "detector": {"board_accuracy": 0.5, "square_accuracy": 0.9},
            "localizer": None,
            "e2e_detector_only": {"board_accuracy": 0.25, "square_accuracy": 0.75},
            "e2e_with_localizer": None,
        }
    )

    assert row == {
        "model": "baseline",
        "det_board": "0.5000",
        "det_sq": "0.9000",
        "loc_mean_px": "-",
        "loc_leq20": "-",
        "e2e_det_board": "0.2500",
        "e2e_det_sq": "0.7500",
        "e2e_loc_board": "-",
        "e2e_loc_sq": "-",
    }


def test_resolve_benchmark_dataset_uses_explicit_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "benchmark.csv"

    resolved = _resolve_benchmark_dataset(
        benchmark_dataset=dataset_path,
        benchmark_input=None,
        benchmark_output=None,
    )

    assert resolved == dataset_path


def test_resolve_benchmark_dataset_prepares_from_raw_input(
    monkeypatch, tmp_path: Path
) -> None:
    raw_input = tmp_path / "raw"
    output_path = tmp_path / "benchmark.csv"

    monkeypatch.setattr(
        evaluate_vision_models,
        "prepare_benchmark_dataset",
        lambda input_dir, output: output_path,
    )

    resolved = _resolve_benchmark_dataset(
        benchmark_dataset=None,
        benchmark_input=raw_input,
        benchmark_output=output_path,
    )

    assert resolved == output_path
