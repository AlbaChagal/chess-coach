"""Load and validate explanation benchmark cases."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExpectedIdea:
    """Expected recurring idea annotation for a benchmark case."""

    kind: str
    label: str


@dataclass(frozen=True)
class ExpectedExplanationSignals:
    """Expected high-level explanation signals for one benchmark case."""

    shared_ideas: list[ExpectedIdea] = field(default_factory=list)
    divergence: str | None = None
    best_move_role_contains: list[str] = field(default_factory=list)
    shared_plan_contains: list[str] = field(default_factory=list)
    counterplay_contains: list[str] = field(default_factory=list)
    counterplay_fallback_ok: bool = False


@dataclass(frozen=True)
class BenchmarkCase:
    """One explanation benchmark case."""

    id: str
    fen: str
    top_n: int
    expected: ExpectedExplanationSignals
    notes: str | None = None


def load_jsonl(path: Path) -> list[BenchmarkCase]:
    """Load explanation benchmark cases from a JSONL file."""
    LOGGER.debug("Loading explanation benchmark cases from %s", path)
    cases: list[BenchmarkCase] = []
    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            cases.append(_parse_case(payload, line_number))
    LOGGER.info("Loaded %s explanation benchmark cases from %s", len(cases), path)
    return cases


def _parse_case(payload: object, line_number: int) -> BenchmarkCase:
    """Convert one raw JSONL record into a typed benchmark case."""
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark line {line_number} must be a JSON object.")
    case_id = _require_str(payload, "id", line_number)
    fen = _require_str(payload, "fen", line_number)
    top_n = _require_int(payload, "top_n", line_number)
    expected_payload = payload.get("expected")
    if not isinstance(expected_payload, dict):
        raise ValueError(f"Benchmark line {line_number} must include an expected object.")
    notes = payload.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError(f"Benchmark line {line_number} notes must be a string.")
    return BenchmarkCase(
        id=case_id,
        fen=fen,
        top_n=top_n,
        expected=_parse_expected(expected_payload, line_number),
        notes=notes,
    )


def _parse_expected(
    payload: dict[str, object],
    line_number: int,
) -> ExpectedExplanationSignals:
    """Convert the expected section of a benchmark record."""
    shared_ideas_payload = payload.get("shared_ideas", [])
    if not isinstance(shared_ideas_payload, list):
        raise ValueError(
            f"Benchmark line {line_number} expected.shared_ideas must be a list."
        )
    shared_ideas = [
        _parse_expected_idea(item, line_number) for item in shared_ideas_payload
    ]
    divergence = payload.get("divergence")
    if divergence is not None and not isinstance(divergence, str):
        raise ValueError(
            f"Benchmark line {line_number} expected.divergence must be a string."
        )
    return ExpectedExplanationSignals(
        shared_ideas=shared_ideas,
        divergence=divergence,
        best_move_role_contains=_parse_string_list(
            payload.get("best_move_role_contains", []),
            field_name="best_move_role_contains",
            line_number=line_number,
        ),
        shared_plan_contains=_parse_string_list(
            payload.get("shared_plan_contains", []),
            field_name="shared_plan_contains",
            line_number=line_number,
        ),
        counterplay_contains=_parse_string_list(
            payload.get("counterplay_contains", []),
            field_name="counterplay_contains",
            line_number=line_number,
        ),
        counterplay_fallback_ok=_parse_bool(
            payload.get("counterplay_fallback_ok", False),
            field_name="counterplay_fallback_ok",
            line_number=line_number,
        ),
    )


def _parse_expected_idea(payload: object, line_number: int) -> ExpectedIdea:
    """Convert one expected idea object."""
    if not isinstance(payload, dict):
        raise ValueError(
            f"Benchmark line {line_number} expected.shared_ideas entries must be objects."
        )
    kind = _require_str(payload, "kind", line_number)
    label = _require_str(payload, "label", line_number)
    return ExpectedIdea(kind=kind, label=label)


def _require_str(payload: dict[str, object], key: str, line_number: int) -> str:
    """Require a string value from a raw payload."""
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Benchmark line {line_number} field {key!r} must be a string.")
    return value


def _require_int(payload: dict[str, object], key: str, line_number: int) -> int:
    """Require an integer value from a raw payload."""
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Benchmark line {line_number} field {key!r} must be an integer.")
    return value


def _parse_string_list(
    value: object,
    *,
    field_name: str,
    line_number: int,
) -> list[str]:
    """Parse a list of strings from a raw benchmark payload."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(
            f"Benchmark line {line_number} expected.{field_name} must be a list of strings."
        )
    return list(value)


def _parse_bool(value: object, *, field_name: str, line_number: int) -> bool:
    """Parse a boolean from a raw benchmark payload."""
    if not isinstance(value, bool):
        raise ValueError(
            f"Benchmark line {line_number} expected.{field_name} must be a boolean."
        )
    return value
