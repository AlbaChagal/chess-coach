"""Helpers for normalizing engine analysis into position-level line models."""

from __future__ import annotations

import re

from chesscoach.analysis.models import MoveAnalysis
from chesscoach.explanation.models import CandidateLine, LineFeature

_CHECK_OR_MATE_SUFFIXES = ("+", "#")
_PIECE_SAN_PREFIXES = ("N", "B", "R", "Q")
_SAN_ANNOTATION_PATTERN = re.compile(r"[+#?!]+$")


def normalize_move_analysis(move: MoveAnalysis) -> CandidateLine:
    """Convert one engine analysis result into a normalized candidate line.

    Args:
        move: Engine analysis for a single candidate move.

    Returns:
        A normalized :class:`~chesscoach.explanation.models.CandidateLine`.

    Raises:
        ValueError: If the root move SAN or UCI is missing.
    """
    if not move.move_san or move.move_san == "?":
        raise ValueError("MoveAnalysis is missing a usable SAN root move.")
    if not move.move_uci or move.move_uci == "?":
        raise ValueError("MoveAnalysis is missing a usable UCI root move.")

    return CandidateLine(
        root_move_uci=move.move_uci,
        root_move_san=move.move_san,
        score_cp=move.score_cp,
        score_mate=move.score_mate,
        depth=move.depth,
        continuation_san=list(move.continuation),
        continuation_uci=list(move.continuation_uci),
    )


def normalize_move_analyses(moves: list[MoveAnalysis]) -> list[CandidateLine]:
    """Convert engine candidate moves into normalized candidate lines."""
    return [normalize_move_analysis(move) for move in moves]


def candidate_line_has_aligned_continuations(line: CandidateLine) -> bool:
    """Return whether the SAN and UCI continuation lists are aligned."""
    return len(line.continuation_san) == len(line.continuation_uci)


def extract_line_features(line: CandidateLine) -> list[LineFeature]:
    """Extract deterministic features from one normalized candidate line."""
    features = _extract_features_from_move(
        move_san=line.root_move_san,
        ply_index=0,
        move_uci=line.root_move_uci,
    )
    has_aligned_continuations = candidate_line_has_aligned_continuations(line)
    for index, move_san in enumerate(line.continuation_san, start=1):
        continuation_uci = None
        if has_aligned_continuations:
            continuation_uci = line.continuation_uci[index - 1]
        features.extend(
            _extract_features_from_move(
                move_san=move_san,
                ply_index=index,
                move_uci=continuation_uci,
            )
        )
    return features


def extract_line_features_for_lines(
    lines: list[CandidateLine],
) -> list[list[LineFeature]]:
    """Extract deterministic features for multiple normalized candidate lines."""
    return [extract_line_features(line) for line in lines]


def _extract_features_from_move(
    *,
    move_san: str,
    ply_index: int,
    move_uci: str | None,
) -> list[LineFeature]:
    """Build line features from a single SAN move."""
    normalized_san = _normalize_san(move_san)
    features: list[LineFeature] = []
    if normalized_san in {"O-O", "O-O-O"}:
        features.append(
            LineFeature(
                kind="king_safety",
                label=_king_safety_label(normalized_san),
                ply_index=ply_index,
                move_uci=move_uci,
                move_san=move_san,
                description=f"This line includes {_king_safety_phrase(normalized_san)}.",
            )
        )
    if _is_pawn_san(normalized_san):
        square = _destination_square_from_pawn_san(normalized_san)
        features.append(
            LineFeature(
                kind="pawn_break",
                label=f"{square} break",
                ply_index=ply_index,
                move_uci=move_uci,
                move_san=move_san,
                description=f"This line includes the {square} pawn advance.",
            )
        )
    if _is_piece_san(normalized_san):
        label = _piece_improvement_label(normalized_san)
        features.append(
            LineFeature(
                kind="piece_improvement",
                label=label,
                ply_index=ply_index,
                move_uci=move_uci,
                move_san=move_san,
                description=f"The line includes {label.lower()}.",
            )
        )
    if _is_check_or_mate_san(move_san):
        motif = "mate threat" if "#" in move_san else "check"
        features.append(
            LineFeature(
                kind="tactical_motif",
                label=motif,
                ply_index=ply_index,
                move_uci=move_uci,
                move_san=move_san,
                description=f"The move {move_san} gives {motif}.",
            )
        )
    return features


def _normalize_san(move_san: str) -> str:
    """Strip SAN suffix annotations that do not affect move classification."""
    return _SAN_ANNOTATION_PATTERN.sub("", move_san)


def _is_check_or_mate_san(move_san: str) -> bool:
    """Return whether the SAN move is marked as check or mate."""
    return move_san.endswith(_CHECK_OR_MATE_SUFFIXES)


def _is_pawn_san(move_san: str) -> bool:
    """Return whether a SAN move represents a pawn move."""
    if not move_san:
        return False
    if move_san.startswith(("O", "K")):
        return False
    return move_san[0] in "abcdefgh"


def _is_piece_san(move_san: str) -> bool:
    """Return whether a SAN move represents a non-king piece move."""
    if not move_san:
        return False
    return move_san.startswith(_PIECE_SAN_PREFIXES)


def _destination_square_from_pawn_san(move_san: str) -> str:
    """Return the destination square from a pawn SAN move."""
    return move_san[-2:]


def _king_safety_label(move_san: str) -> str:
    """Return a concise label for explicit castling moves."""
    if move_san == "O-O":
        return "kingside castling"
    return "queenside castling"


def _king_safety_phrase(move_san: str) -> str:
    """Return a short phrase for castling descriptions."""
    if move_san == "O-O":
        return "kingside castling"
    return "queenside castling"


def _piece_improvement_label(move_san: str) -> str:
    """Return a concise development label for piece moves."""
    return f"{move_san} development"
