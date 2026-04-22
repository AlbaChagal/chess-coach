"""Tests for explanation benchmark evaluation helpers and CLI."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.explanation.dataset import (
    BenchmarkCase,
    ExpectedExplanationSignals,
    ExpectedIdea,
)
from benchmarks.explanation import evaluate as evaluate_module
from chesscoach.explanation.models import (
    PositionTheme,
    RecurringIdea,
    StructuredPositionExplanation,
)
from scripts import evaluate_explanations


class _ExplainerStub:
    def __init__(self, engine: object, provider: object | None) -> None:
        _ = engine
        _ = provider

    def analyze_position_theme(self, fen_before: str) -> PositionTheme:
        _ = fen_before
        return PositionTheme(
            summary="The top candidate lines share an emphasis on f4 break.",
            recurring_ideas=[
                RecurringIdea(
                    kind="pawn_break",
                    label="f4 break",
                    evidence_lines=[0, 1],
                    support=2 / 3,
                    description="This line includes the f4 pawn advance.",
                )
            ],
            side_to_move_plan="The strong lines aim to support f4 break.",
            opponent_counterplay=(
                "No clear shared counterplay signal is visible from the current line features."
            ),
            critical_decision="The main decision is how to support f4 break.",
            best_move_role="The best move supports f4 break more directly.",
            line_divergence_summary=(
                "The top lines share some ideas, but the execution differs."
            ),
        )

    def build_structured_position_explanation(
        self,
        fen_before: str,
    ) -> StructuredPositionExplanation:
        _ = fen_before
        return StructuredPositionExplanation(
            position_summary="The top candidate lines share an emphasis on f4 break.",
            main_ideas=["The strong lines repeatedly build toward f4 break."],
            shared_plan="The strong lines aim to support f4 break.",
            why_the_best_move_fits="The best move supports f4 break more directly.",
            what_all_good_lines_have_in_common=(
                "The good lines share f4 break, but the execution differs."
            ),
            what_to_watch_out_for=(
                "No clear shared counterplay signal is visible from the current line features."
            ),
            candidate_move_roles=["f4 is the most direct route toward f4 break."],
        )


class _EngineStub:
    def __enter__(self) -> "_EngineStub":
        return self

    def __exit__(self, *args: object) -> None:
        _ = args


def test_evaluate_case_matches_expected_signals() -> None:
    case = BenchmarkCase(
        id="case-1",
        fen="8/8/8/8/8/8/8/8 w - - 0 1",
        top_n=3,
        expected=ExpectedExplanationSignals(
            shared_ideas=[ExpectedIdea(kind="pawn_break", label="f4 break")],
            divergence="partial_convergence",
            best_move_role_contains=["more directly"],
            shared_plan_contains=["f4 break"],
            counterplay_contains=[],
            counterplay_fallback_ok=True,
        ),
    )

    result = evaluate_module.evaluate_case(case, _ExplainerStub(_EngineStub(), None))

    assert result.passed is True
    assert result.shared_idea_matches is True
    assert result.divergence_matches is True


def test_summarize_results_builds_aggregate_metrics() -> None:
    report = evaluate_module.summarize_results(
        [
            evaluate_module.BenchmarkCaseResult(
                id="case-1",
                passed=True,
                shared_idea_matches=True,
                divergence_matches=True,
                best_move_role_matches=True,
                shared_plan_matches=True,
                counterplay_matches=True,
                actual_shared_ideas=["pawn_break:f4 break"],
                actual_divergence="partial_convergence",
                actual_best_move_role="The best move supports f4 break more directly.",
                actual_shared_plan="The strong lines aim to support f4 break.",
                actual_counterplay="No clear shared counterplay signal is visible from the current line features.",
                actual_summary="The top candidate lines share an emphasis on f4 break.",
            )
        ]
    )

    assert report["n_cases"] == 1
    assert report["pass_rate"] == 1.0


def test_write_report_writes_json(tmp_path: Path) -> None:
    output_path = tmp_path / "report.json"
    report = {"n_cases": 1, "cases": []}

    evaluate_module.write_report(report, output_path)

    assert json.loads(output_path.read_text()) == report


def test_main_loads_cases_and_writes_output(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "positions.jsonl"
    output_path = tmp_path / "report.json"
    seen_datasets: list[Path] = []
    written_reports: list[tuple[dict[str, object], Path]] = []

    monkeypatch.setattr(
        evaluate_explanations,
        "load_jsonl",
        lambda path: seen_datasets.append(path) or [],
    )
    monkeypatch.setattr(
        evaluate_explanations,
        "evaluate_cases",
        lambda cases: {"n_cases": len(cases), "cases": []},
    )
    monkeypatch.setattr(
        evaluate_explanations,
        "write_report",
        lambda report, path: written_reports.append((report, path)),
    )
    monkeypatch.setattr(
        evaluate_explanations,
        "print_report",
        lambda report: None,
    )
    monkeypatch.setattr(evaluate_explanations, "configure_logging", lambda level: None)

    evaluate_explanations.main(
        [
            "--dataset",
            str(dataset_path),
            "--output",
            str(output_path),
        ]
    )

    assert seen_datasets == [dataset_path]
    assert written_reports == [({"n_cases": 0, "cases": []}, output_path)]
