"""Tests for score-threshold sweep utility."""

from __future__ import annotations

from pathlib import Path

from scripts import sweep_vision_score_threshold as sweep_module


class _DetectorStub:
    def __init__(
        self,
        checkpoint: Path,
        *,
        score_threshold: float,
        image_size: int,
    ) -> None:
        self.checkpoint = checkpoint
        self.score_threshold = score_threshold
        self.image_size = image_size


def test_best_result_prefers_board_accuracy_then_square_accuracy() -> None:
    best = sweep_module._best_result(
        [
            {"score_threshold": 0.05, "board_accuracy": 0.65, "square_accuracy": 0.98},
            {"score_threshold": 0.08, "board_accuracy": 0.68, "square_accuracy": 0.97},
            {"score_threshold": 0.10, "board_accuracy": 0.68, "square_accuracy": 0.99},
        ]
    )

    assert best["score_threshold"] == 0.10


def test_sweep_score_thresholds_instantiates_detector_per_threshold(
    monkeypatch, tmp_path: Path
) -> None:
    detector_calls: list[tuple[Path, float, int]] = []

    monkeypatch.setattr(
        sweep_module,
        "_load_dataset",
        lambda path: [],
    )
    monkeypatch.setattr(
        sweep_module,
        "run_evaluation",
        lambda samples, detector, board_localizer=None: {
            "n_boards": 0,
            "n_errors": 0,
            "board_accuracy": detector.score_threshold,
            "square_accuracy": 1.0,
            "per_piece_accuracy": {},
            "failure_breakdown": {},
        },
    )
    monkeypatch.setattr(
        sweep_module,
        "PieceDetector",
        lambda checkpoint, score_threshold, image_size: (
            detector_calls.append((checkpoint, score_threshold, image_size))
            or _DetectorStub(
                checkpoint,
                score_threshold=score_threshold,
                image_size=image_size,
            )
        ),
    )
    monkeypatch.setattr(
        sweep_module,
        "BoardCornerLocalizer",
        lambda checkpoint, image_size=512: object(),
    )

    results = sweep_module.sweep_score_thresholds(
        dataset_path=tmp_path / "dataset.csv",
        detector_checkpoint=Path("models/piece_detector.pt"),
        board_localizer_checkpoint=Path("models/board_localizer.pt"),
        split="test",
        thresholds=[0.05, 0.10],
        image_size=800,
        board_localizer_image_size=640,
    )

    assert detector_calls == [
        (Path("models/piece_detector.pt"), 0.05, 800),
        (Path("models/piece_detector.pt"), 0.10, 800),
    ]
    assert [result["score_threshold"] for result in results] == [0.05, 0.10]


def test_sweep_score_thresholds_parallel_path(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sweep_module,
        "_load_dataset",
        lambda path: [],
    )

    class _Future:
        def __init__(self, value: dict[str, object]) -> None:
            self._value = value

        def result(self) -> dict[str, object]:
            return self._value

    class _Executor:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = max_workers
            self.futures: list[_Future] = []

        def __enter__(self) -> _Executor:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def submit(self, fn, **kwargs):  # type: ignore[no-untyped-def]
            future = _Future(fn(**kwargs))
            self.futures.append(future)
            return future

    monkeypatch.setattr(
        sweep_module,
        "ProcessPoolExecutor",
        _Executor,
    )
    monkeypatch.setattr(
        sweep_module,
        "as_completed",
        lambda futures: list(futures),
    )
    monkeypatch.setattr(
        sweep_module,
        "_evaluate_threshold",
        lambda **kwargs: {
            "score_threshold": kwargs["score_threshold"],
            "n_boards": 0,
            "n_errors": 0,
            "board_accuracy": kwargs["score_threshold"],
            "square_accuracy": 1.0,
            "per_piece_accuracy": {},
            "failure_breakdown": {},
        },
    )

    results = sweep_module.sweep_score_thresholds(
        dataset_path=tmp_path / "dataset.csv",
        detector_checkpoint=Path("models/piece_detector.pt"),
        board_localizer_checkpoint=Path("models/board_localizer.pt"),
        split="test",
        thresholds=[0.10, 0.05],
        image_size=800,
        board_localizer_image_size=640,
        jobs=2,
    )

    assert [result["score_threshold"] for result in results] == [0.05, 0.10]
