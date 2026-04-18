"""Qualitative regression script for the explanation module.

Runs 10 hand-crafted positions through analyze_move() and prints a
human-readable report for manual review.  No LLM calls are made by
default; pass --llm to also generate coaching text (requires
ANTHROPIC_API_KEY or OPENAI_API_KEY to be set).

Usage:
    uv run python scripts/explain_regression.py
    uv run python scripts/explain_regression.py --llm          # Claude Haiku
    uv run python scripts/explain_regression.py --llm openai   # GPT-4o-mini
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from dataclasses import dataclass

# Ensure the project root is importable when run as a script.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from chesscoach.analysis.engine import ChessEngine
from chesscoach.explanation import Explainer
from chesscoach.explanation.models import ExplainedMove


# ---------------------------------------------------------------------------
# Dataset: 10 positions covering the main scenarios
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Example:
    label: str
    description: str
    fen: str
    move_uci: str
    expect_quality: str   # rough expected label for reviewer
    expect_tactics: list[str]  # tactic names that should appear


EXAMPLES: list[Example] = [
    Example(
        label="01-opening-quiet",
        description="Standard opening pawn push (1.e4) — no immediate threats.",
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        move_uci="e2e4",
        expect_quality="best/good",
        expect_tactics=[],
    ),
    Example(
        label="02-checkmate-scholars",
        description="Scholar's mate — Qxf7# delivers checkmate.",
        fen="r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 3",
        move_uci="h5f7",
        expect_quality="best",
        expect_tactics=["check"],
    ),
    Example(
        label="03-blunder-hanging-rook",
        description=(
            "White king walks away (Ke1-f1) abandoning Rd1 to Qxd1. "
            "Rook is undefended and attacked by the black queen."
        ),
        fen="3qk3/8/8/8/8/8/8/3RK3 w - - 0 1",
        move_uci="e1f1",
        expect_quality="blunder",
        expect_tactics=["hanging_piece"],
    ),
    Example(
        label="04-knight-fork-best",
        description=(
            "Nd5-f6+ forks the black king on g8 and rook on d7. "
            "Best move — should score high with fork + check."
        ),
        fen="6k1/3r4/8/3N4/8/8/8/4K3 w - - 0 1",
        move_uci="d5f6",
        expect_quality="best",
        expect_tactics=["fork", "check"],
    ),
    Example(
        label="05-missed-fork",
        description=(
            "Same position as #04 but white plays Nd5-e3, missing the fork. "
            "Should be classified as a mistake/inaccuracy."
        ),
        fen="6k1/3r4/8/3N4/8/8/8/4K3 w - - 0 1",
        move_uci="d5e3",
        expect_quality="mistake/inaccuracy",
        expect_tactics=[],
    ),
    Example(
        label="06-discovered-attack",
        description=(
            "Knight on b6 moves to d5, unblocking bishop on a5 which "
            "now attacks the rook on d8 along the a5-d8 diagonal."
        ),
        fen="3rk3/8/1N6/B7/8/8/8/4K3 w - - 0 1",
        move_uci="b6d5",
        expect_quality="best/good",
        expect_tactics=["discovered_attack"],
    ),
    Example(
        label="07-pin-creation",
        description=(
            "Bishop b2 moves to c3, pinning the black knight on f6 "
            "against the black king on h8 along the c3-h8 diagonal."
        ),
        fen="7k/8/5n2/8/8/8/1B6/4K3 w - - 0 1",
        move_uci="b2c3",
        expect_quality="best/good",
        expect_tactics=["pin"],
    ),
    Example(
        label="08-skewer-rook",
        description=(
            "Ra1-a8+ skewers the black king on h8: king must move, "
            "then Rxa7 or Rxh7 wins the queen."
        ),
        fen="7k/7q/8/8/8/8/8/R3K3 w - - 0 1",
        move_uci="a1a8",
        expect_quality="best",
        expect_tactics=["check", "skewer"],
    ),
    Example(
        label="09-queen-blunder-pawn",
        description=(
            "White queen walks to d4 which is simultaneously attacked "
            "by black pawns on c5 and e5.  Clear blunder — queen hangs."
        ),
        fen="4k3/8/8/2p1p3/8/8/8/Q3K3 w - - 0 1",
        move_uci="a1d4",
        expect_quality="blunder",
        expect_tactics=["hanging_piece"],
    ),
    Example(
        label="10-endgame-king-advance",
        description=(
            "King-and-rook endgame: white king advances Ke1-d2, "
            "supporting the rook and tightening the net around the black king. "
            "Quiet positional improvement — no tactics expected."
        ),
        fen="8/8/8/8/3k4/8/8/R3K3 w - - 0 1",
        move_uci="e1d2",
        expect_quality="best/good",
        expect_tactics=[],
    ),
]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_SEP = "─" * 68
_WIDE = "═" * 68


def _fmt_score(cp: int | None, mate: int | None) -> str:
    if mate is not None:
        sign = "+" if mate > 0 else ""
        return f"M{sign}{mate}"
    if cp is not None:
        pawns = cp / 100
        sign = "+" if pawns >= 0 else ""
        return f"{sign}{pawns:.2f}"
    return "?"


def _fmt_tactics(tactics: list) -> str:  # type: ignore[type-arg]
    if not tactics:
        return "none"
    return ", ".join(f"{t.name} ({t.description})" for t in tactics)


def _fmt_continuation(moves: list[str]) -> str:
    if not moves:
        return ""
    return "  line: " + " ".join(moves[:4])


def _print_result(
    idx: int,
    ex: Example,
    result: ExplainedMove,
    explanation: str | None,
) -> None:
    print(_SEP)
    print(f" #{idx:02d}  [{ex.label}]")
    print(f"      {ex.description}")
    print()
    print(f"  FEN      : {ex.fen}")
    print(f"  Move     : {result.move_played_san}  ({result.move_played_uci})")
    print(
        f"  Quality  : {result.quality.label} {result.quality.emoji}"
        f"  (cp loss: {result.quality.cp_loss})"
    )
    print()
    print(f"  Best move: {result.best_move.move_san}"
          f"  ({_fmt_score(result.best_move.score_cp, result.best_move.score_mate)})"
          f"{_fmt_continuation(result.best_move.continuation)}")
    if result.alternatives:
        for alt in result.alternatives:
            print(f"           : {alt.move_san}"
                  f"  ({_fmt_score(alt.score_cp, alt.score_mate)})"
                  f"{_fmt_continuation(alt.continuation)}")
    print()
    print(f"  Tactics (after played) : {_fmt_tactics(result.tactics_after_played)}")
    print(f"  Tactics (after best)   : {_fmt_tactics(result.tactics_after_best)}")
    print()
    print(f"  Expected quality  : {ex.expect_quality}")
    print(f"  Expected tactics  : {ex.expect_tactics or 'none'}")

    if explanation:
        print()
        wrapped = textwrap.fill(explanation, width=64, initial_indent="  ", subsequent_indent="  ")
        print(f"  LLM explanation:\n{wrapped}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm",
        nargs="?",
        const="claude",
        choices=["claude", "openai"],
        metavar="PROVIDER",
        help="Also generate LLM coaching text.  PROVIDER is 'claude' (default) or 'openai'.",
    )
    parser.add_argument(
        "--examples",
        nargs="+",
        metavar="LABEL",
        help="Run only these example labels (e.g. 04-knight-fork-best).",
    )
    return parser.parse_args()


def _make_provider(provider_name: str) -> object:
    """Build an LLM provider, raising clearly if the key is missing."""
    if provider_name == "openai":
        from chesscoach.explanation.providers import OpenAIProvider

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
            sys.exit(1)
        return OpenAIProvider(api_key=key)

    from chesscoach.explanation.providers import ClaudeProvider

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    return ClaudeProvider(api_key=key)


def main() -> None:
    args = _parse_args()

    examples = EXAMPLES
    if args.examples:
        labels = set(args.examples)
        examples = [e for e in EXAMPLES if e.label in labels]
        if not examples:
            print(f"No examples matched: {args.examples}", file=sys.stderr)
            sys.exit(1)

    provider = _make_provider(args.llm) if args.llm else None

    print(_WIDE)
    print("  Explanation Module — Qualitative Regression Report")
    print(f"  {len(examples)} examples  |  LLM: {'yes (' + str(args.llm) + ')' if provider else 'no'}")
    print(_WIDE)

    errors: list[tuple[str, Exception]] = []

    with ChessEngine() as engine:
        # provider is either None or a real LLMProvider — pass a dummy when absent
        dummy_or_real = provider if provider is not None else _DummyProvider()
        explainer = Explainer(engine, dummy_or_real)  # type: ignore[arg-type]

        for idx, ex in enumerate(examples, start=1):
            try:
                result = explainer.analyze_move(ex.fen, ex.move_uci)
                explanation: str | None = None
                if provider is not None:
                    explanation = explainer.explain_move(ex.fen, ex.move_uci)
                _print_result(idx, ex, result, explanation)
            except Exception as exc:  # noqa: BLE001
                errors.append((ex.label, exc))
                print(_SEP)
                print(f" #{idx:02d}  [{ex.label}]  ERROR: {exc}")
                print()

    print(_WIDE)
    if errors:
        print(f"  {len(errors)} example(s) failed:")
        for label, exc in errors:
            print(f"    {label}: {exc}")
    else:
        print(f"  All {len(examples)} examples completed successfully.")
    print(_WIDE)


class _DummyProvider:
    """No-op provider so Explainer can be constructed without an LLM."""

    def complete(self, system: str, user: str) -> str:  # noqa: ARG002
        return ""


if __name__ == "__main__":
    main()
