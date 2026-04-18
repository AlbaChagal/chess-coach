"""Tests for end-to-end vision benchmark wiring."""

from __future__ import annotations

from pathlib import Path

from benchmarks.vision.dataset import BoardSample
from benchmarks.vision import evaluate as evaluate_module


class _DetectorStub:
    def __init__(self, checkpoint: Path | None = None) -> None:
        _ = checkpoint


def test_run_evaluation_uses_predict_fen_output(monkeypatch) -> None:
    samples = [
        BoardSample(
            image_path=Path("board.png"),
            fen_placement="8/8/8/8/8/8/8/8",
            split="test",
        )
    ]
    classifier = _DetectorStub()

    monkeypatch.setattr(
        evaluate_module,
        "predict_fen",
        lambda path, model, board_localizer=None: "8/8/8/8/8/8/8/8",
    )

    results = evaluate_module.run_evaluation(samples, classifier)

    assert results["n_boards"] == 1
    assert results["board_accuracy"] == 1.0
    assert results["square_accuracy"] == 1.0
    assert results["failure_breakdown"] == {}


def test_run_evaluation_reports_board_detection_failures(monkeypatch) -> None:
    samples = [
        BoardSample(
            image_path=Path("missing.png"),
            fen_placement="8/8/8/8/8/8/8/8",
            split="test",
        )
    ]
    classifier = _DetectorStub()

    def _raise(*args, **kwargs):
        raise evaluate_module.BoardNotFoundError("not found")

    monkeypatch.setattr(evaluate_module, "predict_fen", _raise)

    results = evaluate_module.run_evaluation(samples, classifier)

    assert results["n_boards"] == 0
    assert results["n_errors"] == 1
    assert results["failure_breakdown"] == {"board_not_found": 1}


def test_main_prefers_detector_checkpoint(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("[]")
    detector_calls: list[Path] = []

    monkeypatch.setattr(evaluate_module, "_load_dataset", lambda path: [])
    monkeypatch.setattr(
        evaluate_module,
        "run_evaluation",
        lambda samples, classifier, board_localizer=None: {
            "n_boards": 0,
            "n_errors": 0,
            "board_accuracy": 0.0,
            "square_accuracy": 0.0,
            "per_piece_accuracy": {},
        },
    )
    monkeypatch.setattr(
        evaluate_module,
        "PieceDetector",
        lambda checkpoint=None: detector_calls.append(checkpoint) or _DetectorStub(checkpoint),
    )
    monkeypatch.setattr(
        evaluate_module,
        "BoardCornerLocalizer",
        lambda checkpoint: _DetectorStub(checkpoint),
    )
    monkeypatch.setattr(evaluate_module, "configure_logging", lambda level: None)

    evaluate_module.main(
        [
            "--dataset",
            str(dataset_path),
            "--detector-checkpoint",
            "models/piece_detector.pt",
        ]
    )

    assert detector_calls == [Path("models/piece_detector.pt")]


def test_main_filters_dataset_by_split(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("[]")
    seen_samples: list[BoardSample] = []

    monkeypatch.setattr(
        evaluate_module,
        "_load_dataset",
        lambda path: [
            BoardSample(Path("train.png"), "8/8/8/8/8/8/8/8", "train"),
            BoardSample(Path("test.png"), "8/8/8/8/8/8/8/8", "test"),
        ],
    )
    monkeypatch.setattr(
        evaluate_module,
        "run_evaluation",
        lambda samples, classifier, board_localizer=None: (
            seen_samples.extend(samples)
            or {
                "n_boards": 0,
                "n_errors": 0,
                "board_accuracy": 0.0,
                "square_accuracy": 0.0,
                "per_piece_accuracy": {},
                "failure_breakdown": {},
            }
        ),
    )
    monkeypatch.setattr(evaluate_module, "PieceDetector", lambda checkpoint=None: _DetectorStub(checkpoint))
    monkeypatch.setattr(
        evaluate_module,
        "BoardCornerLocalizer",
        lambda checkpoint: _DetectorStub(checkpoint),
    )
    monkeypatch.setattr(evaluate_module, "configure_logging", lambda level: None)

    evaluate_module.main(
        [
            "--dataset",
            str(dataset_path),
            "--split",
            "test",
        ]
    )

    assert seen_samples == [BoardSample(Path("test.png"), "8/8/8/8/8/8/8/8", "test")]


def test_main_passes_optional_board_localizer(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("[]")
    localizer_calls: list[tuple[Path, int]] = []
    seen_localizers: list[object | None] = []

    monkeypatch.setattr(evaluate_module, "_load_dataset", lambda path: [])
    monkeypatch.setattr(
        evaluate_module,
        "run_evaluation",
        lambda samples, classifier, board_localizer=None: (
            seen_localizers.append(board_localizer)
            or {
                "n_boards": 0,
                "n_errors": 0,
                "board_accuracy": 0.0,
                "square_accuracy": 0.0,
                "per_piece_accuracy": {},
                "failure_breakdown": {},
            }
        ),
    )
    monkeypatch.setattr(evaluate_module, "PieceDetector", lambda checkpoint=None: _DetectorStub(checkpoint))
    monkeypatch.setattr(
        evaluate_module,
        "BoardCornerLocalizer",
        lambda checkpoint, image_size=512: localizer_calls.append((checkpoint, image_size))
        or object(),
    )
    monkeypatch.setattr(evaluate_module, "configure_logging", lambda level: None)

    evaluate_module.main(
        [
            "--dataset",
            str(dataset_path),
            "--board-localizer-checkpoint",
            "models/board_localizer.pt",
            "--board-localizer-image-size",
            "640",
        ]
    )

    assert localizer_calls == [(Path("models/board_localizer.pt"), 640)]
    assert len(seen_localizers) == 1
