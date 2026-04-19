"""List and exclude suspicious boards from orientation-audit results."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from chesscoach.logging_utils import add_logging_args, configure_logging

LOGGER = logging.getLogger(__name__)


def _load_audit_report(audit_report: Path) -> list[dict[str, Any]]:
    payload = json.loads(audit_report.read_text())
    boards = payload.get("boards")
    if not isinstance(boards, list):
        raise ValueError(f"Invalid audit report: {audit_report}")
    return [board for board in boards if isinstance(board, dict)]


def suspicious_board_paths(
    audit_report: Path,
    *,
    max_mismatches: int = 0,
) -> list[str]:
    """Return suspicious board image paths from an orientation-audit report."""
    boards = _load_audit_report(audit_report)
    suspicious = [
        str(board["image_path"])
        for board in boards
        if int(board.get("mismatches", 0)) > max_mismatches
    ]
    return sorted(suspicious)


def _filter_manifest(
    manifest_path: Path,
    output_path: Path,
    excluded_paths: set[str],
) -> tuple[int, int]:
    kept = 0
    removed = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open() as src, output_path.open("w") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            image_path = record.get("image_path")
            if isinstance(image_path, str) and image_path in excluded_paths:
                removed += 1
                continue
            dst.write(json.dumps(record) + "\n")
            kept += 1
    return kept, removed


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="List or exclude suspicious boards from orientation audits."
    )
    add_logging_args(parser)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=0,
        dest="max_mismatches",
        help="Keep boards with mismatches <= this threshold.",
    )
    parser.add_argument(
        "--output-list",
        type=Path,
        default=None,
        dest="output_list",
        help="Optional path to write excluded image paths.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        nargs="*",
        default=[],
        help="Optional manifest.jsonl files to filter.",
    )
    parser.add_argument(
        "--filtered-output-dir",
        type=Path,
        default=None,
        dest="filtered_output_dir",
        help="Directory for filtered manifest copies.",
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    suspicious = suspicious_board_paths(
        args.audit_report,
        max_mismatches=args.max_mismatches,
    )
    LOGGER.info(
        f"Suspicious boards from {args.audit_report}: count={len(suspicious)}"
    )
    for image_path in suspicious:
        LOGGER.info(f"  {image_path}")

    if args.output_list is not None:
        args.output_list.parent.mkdir(parents=True, exist_ok=True)
        args.output_list.write_text("\n".join(suspicious) + ("\n" if suspicious else ""))
        LOGGER.info(f"Wrote suspicious board list to {args.output_list}")

    if args.manifest:
        if args.filtered_output_dir is None:
            raise ValueError("--filtered-output-dir is required when --manifest is used")
        excluded_paths = set(suspicious)
        for manifest_path in args.manifest:
            output_path = args.filtered_output_dir / manifest_path.name
            kept, removed = _filter_manifest(manifest_path, output_path, excluded_paths)
            LOGGER.info(
                f"Filtered manifest {manifest_path} -> {output_path} "
                f"kept={kept} removed={removed}"
            )


if __name__ == "__main__":
    main()
