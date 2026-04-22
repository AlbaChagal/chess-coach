"""Evaluate deterministic explanation outputs on a curated benchmark set."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.explanation.dataset import load_jsonl
from benchmarks.explanation.evaluate import evaluate_cases, print_report, write_report
from chesscoach.logging_utils import add_logging_args, configure_logging

DEFAULT_DATASET = Path("benchmarks/explanation/positions.jsonl")
DEFAULT_OUTPUT = Path("benchmarks/explanation/results/latest.json")


def main(argv: list[str] | None = None) -> None:
    """Entry point for the explanation benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic explanation outputs on benchmark cases."
    )
    add_logging_args(parser)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the explanation benchmark JSONL dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write a JSON report.",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    cases = load_jsonl(args.dataset)
    report = evaluate_cases(cases)
    print_report(report)
    if args.output is not None:
        write_report(report, args.output)


if __name__ == "__main__":
    main()
