"""Deterministic evaluation helpers for the explanation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import logging
from typing import Protocol

from benchmarks.explanation.dataset import BenchmarkCase
from chesscoach.analysis.engine import ChessEngine
from chesscoach.explanation import Explainer
from chesscoach.explanation.models import PositionTheme, StructuredPositionExplanation

LOGGER = logging.getLogger(__name__)


class _ExplainerProtocol(Protocol):
    """Minimal explainer interface required by the evaluation harness."""

    def analyze_position_theme(self, fen_before: str) -> PositionTheme:
        """Return a synthesized position theme."""
        ...

    def build_structured_position_explanation(
        self,
        fen_before: str,
    ) -> StructuredPositionExplanation:
        """Return a structured position explanation."""
        ...


@dataclass(frozen=True)
class BenchmarkCaseResult:
    """Evaluation result for one explanation benchmark case."""

    id: str
    passed: bool
    shared_idea_matches: bool
    divergence_matches: bool
    best_move_role_matches: bool
    shared_plan_matches: bool
    counterplay_matches: bool
    actual_shared_ideas: list[str]
    actual_divergence: str
    actual_best_move_role: str
    actual_shared_plan: str
    actual_counterplay: str
    actual_summary: str


def evaluate_cases(cases: list[BenchmarkCase]) -> dict[str, object]:
    """Evaluate deterministic explanation outputs for benchmark cases."""
    results: list[BenchmarkCaseResult] = []
    with ChessEngine() as engine:
        explainer = Explainer(engine, provider=None)
        for case in cases:
            results.append(evaluate_case(case, explainer))
    return summarize_results(results)


def evaluate_case(
    case: BenchmarkCase,
    explainer: _ExplainerProtocol,
) -> BenchmarkCaseResult:
    """Evaluate one benchmark case using the deterministic explanation stack."""
    theme = explainer.analyze_position_theme(case.fen)
    structured = explainer.build_structured_position_explanation(case.fen)
    actual_shared_ideas = [f"{idea.kind}:{idea.label}" for idea in theme.recurring_ideas]
    expected_shared_ideas = {
        f"{idea.kind}:{idea.label}" for idea in case.expected.shared_ideas
    }
    actual_divergence = classify_divergence(theme.line_divergence_summary)
    shared_idea_matches = set(actual_shared_ideas) == expected_shared_ideas
    divergence_matches = (
        case.expected.divergence is None or actual_divergence == case.expected.divergence
    )
    best_move_role_matches = _contains_all(
        theme.best_move_role,
        case.expected.best_move_role_contains,
    )
    shared_plan_matches = _contains_all(
        structured.shared_plan,
        case.expected.shared_plan_contains,
    )
    counterplay_matches = _counterplay_matches(
        actual_counterplay=structured.what_to_watch_out_for,
        expected_contains=case.expected.counterplay_contains,
        fallback_ok=case.expected.counterplay_fallback_ok,
    )
    passed = all(
        [
            shared_idea_matches,
            divergence_matches,
            best_move_role_matches,
            shared_plan_matches,
            counterplay_matches,
        ]
    )
    return BenchmarkCaseResult(
        id=case.id,
        passed=passed,
        shared_idea_matches=shared_idea_matches,
        divergence_matches=divergence_matches,
        best_move_role_matches=best_move_role_matches,
        shared_plan_matches=shared_plan_matches,
        counterplay_matches=counterplay_matches,
        actual_shared_ideas=actual_shared_ideas,
        actual_divergence=actual_divergence,
        actual_best_move_role=theme.best_move_role,
        actual_shared_plan=structured.shared_plan,
        actual_counterplay=structured.what_to_watch_out_for,
        actual_summary=structured.position_summary,
    )


def summarize_results(results: list[BenchmarkCaseResult]) -> dict[str, object]:
    """Build aggregate and per-case evaluation output."""
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    return {
        "n_cases": total,
        "n_passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "shared_idea_match_rate": _match_rate(results, "shared_idea_matches"),
        "divergence_match_rate": _match_rate(results, "divergence_matches"),
        "best_move_role_match_rate": _match_rate(results, "best_move_role_matches"),
        "shared_plan_match_rate": _match_rate(results, "shared_plan_matches"),
        "counterplay_match_rate": _match_rate(results, "counterplay_matches"),
        "cases": [asdict(result) for result in results],
    }


def write_report(report: dict[str, object], output_path: Path) -> None:
    """Write an evaluation report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))


def classify_divergence(summary: str) -> str:
    """Normalize a divergence summary into a benchmark-friendly label."""
    normalized = summary.lower()
    if "share the same plan" in normalized:
        return "strong_convergence"
    if "share some ideas" in normalized:
        return "partial_convergence"
    return "distinct_plans"


def print_report(report: dict[str, object]) -> None:
    """Print a concise human-readable evaluation report."""
    LOGGER.info("=== Explanation Benchmark ===")
    LOGGER.info("Cases evaluated: %s", report["n_cases"])
    LOGGER.info("Cases passed: %s", report["n_passed"])
    LOGGER.info("Pass rate: %.1f%%", report["pass_rate"] * 100)  # type: ignore[operator]
    LOGGER.info(
        "Shared idea match rate: %.1f%%",
        report["shared_idea_match_rate"] * 100,  # type: ignore[operator]
    )
    LOGGER.info(
        "Divergence match rate: %.1f%%",
        report["divergence_match_rate"] * 100,  # type: ignore[operator]
    )
    LOGGER.info(
        "Best move role match rate: %.1f%%",
        report["best_move_role_match_rate"] * 100,  # type: ignore[operator]
    )
    failures = [
        case for case in report["cases"]  # type: ignore[index]
        if not case["passed"]  # type: ignore[index]
    ]
    if failures:
        LOGGER.info("Failures:")
        for failure in failures:
            LOGGER.info(
                "  %s shared_ideas=%s divergence=%s",
                failure["id"],
                failure["actual_shared_ideas"],
                failure["actual_divergence"],
            )


def _match_rate(results: list[BenchmarkCaseResult], field_name: str) -> float:
    """Compute a rounded match rate for a boolean result field."""
    if not results:
        return 0.0
    matches = sum(1 for result in results if getattr(result, field_name))
    return round(matches / len(results), 4)


def _contains_all(text: str, expected_substrings: list[str]) -> bool:
    """Return whether all expected substrings are present in text."""
    normalized = text.lower()
    return all(expected.lower() in normalized for expected in expected_substrings)


def _counterplay_matches(
    *,
    actual_counterplay: str,
    expected_contains: list[str],
    fallback_ok: bool,
) -> bool:
    """Return whether counterplay text matches the expected signal."""
    if expected_contains:
        return _contains_all(actual_counterplay, expected_contains)
    if fallback_ok:
        return "no clear shared counterplay signal" in actual_counterplay.lower()
    return True
