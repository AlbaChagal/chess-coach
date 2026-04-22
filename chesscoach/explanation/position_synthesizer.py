"""Helpers for normalizing engine analysis into position-level line models."""

from __future__ import annotations

import re
from dataclasses import dataclass

from chesscoach.analysis.models import MoveAnalysis
from chesscoach.explanation.models import (
    CandidateLine,
    IdeaKind,
    LineFeature,
    PositionTheme,
    RecurringIdea,
    StructuredPositionExplanation,
)

_CHECK_OR_MATE_SUFFIXES = ("+", "#")
_PIECE_SAN_PREFIXES = ("N", "B", "R", "Q")
_SAN_ANNOTATION_PATTERN = re.compile(r"[+#?!]+$")
_NO_COUNTERPLAY_TEXT = (
    "No clear shared counterplay signal is visible from the current line features."
)


@dataclass(frozen=True)
class _FeatureGroup:
    """Grouped cross-line evidence for a recurring feature key."""

    kind: IdeaKind
    label: str
    evidence_lines: list[int]
    earliest_ply_indices: list[int]
    description: str


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


def synthesize_recurring_ideas(
    lines: list[CandidateLine],
    features_by_line: list[list[LineFeature]] | None = None,
) -> list[RecurringIdea]:
    """Find ideas that recur across candidate lines."""
    if not lines:
        return []
    normalized_features = _resolve_features_by_line(lines, features_by_line)
    grouped_features = _group_features_by_recurrence_key(normalized_features)
    recurring_ideas = _build_recurring_ideas(grouped_features, len(lines))
    return recurring_ideas


def synthesize_position_theme(
    lines: list[CandidateLine],
    features_by_line: list[list[LineFeature]] | None = None,
) -> PositionTheme:
    """Build a position-level summary from candidate lines and line features."""
    if not lines:
        raise ValueError("Position theme synthesis requires at least one line.")
    normalized_features = _resolve_features_by_line(lines, features_by_line)
    recurring_ideas = synthesize_recurring_ideas(lines, normalized_features)
    divergence_class = _classify_line_divergence(recurring_ideas, len(lines))
    return PositionTheme(
        summary=_build_position_summary(recurring_ideas, divergence_class),
        recurring_ideas=recurring_ideas,
        side_to_move_plan=_build_side_to_move_plan(
            lines,
            normalized_features,
            recurring_ideas,
            divergence_class,
        ),
        opponent_counterplay=_build_opponent_counterplay(
            normalized_features,
            recurring_ideas,
        ),
        critical_decision=_build_critical_decision(recurring_ideas, divergence_class),
        best_move_role=_build_best_move_role(
            lines,
            normalized_features,
            recurring_ideas,
            divergence_class,
        ),
        line_divergence_summary=_build_line_divergence_summary(divergence_class),
    )


def build_structured_position_explanation(
    lines: list[CandidateLine],
    theme: PositionTheme | None = None,
    features_by_line: list[list[LineFeature]] | None = None,
) -> StructuredPositionExplanation:
    """Convert synthesized line data into a product-facing position explanation."""
    if not lines:
        raise ValueError("Structured position explanation requires at least one line.")
    normalized_features = _resolve_features_by_line(lines, features_by_line)
    resolved_theme = theme or synthesize_position_theme(lines, normalized_features)
    return StructuredPositionExplanation(
        position_summary=resolved_theme.summary,
        main_ideas=_build_main_ideas(lines, resolved_theme, normalized_features),
        shared_plan=resolved_theme.side_to_move_plan,
        why_the_best_move_fits=resolved_theme.best_move_role,
        what_all_good_lines_have_in_common=_build_commonality_text(resolved_theme),
        what_to_watch_out_for=resolved_theme.opponent_counterplay,
        candidate_move_roles=_build_candidate_move_roles(
            lines,
            normalized_features,
            resolved_theme,
        ),
    )


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


def _resolve_features_by_line(
    lines: list[CandidateLine],
    features_by_line: list[list[LineFeature]] | None,
) -> list[list[LineFeature]]:
    """Return line features, validating provided inputs when present."""
    if features_by_line is None:
        return extract_line_features_for_lines(lines)
    if len(features_by_line) != len(lines):
        raise ValueError("Feature list count must match line count.")
    return features_by_line


def _group_features_by_recurrence_key(
    features_by_line: list[list[LineFeature]],
) -> dict[tuple[IdeaKind, str], _FeatureGroup]:
    """Group features by exact recurrence key with per-line deduplication."""
    grouped: dict[tuple[IdeaKind, str], _FeatureGroup] = {}
    for line_index, line_features in enumerate(features_by_line):
        line_seen: set[tuple[IdeaKind, str]] = set()
        ordered_features = sorted(
            line_features,
            key=lambda feature: (
                feature.ply_index if feature.ply_index is not None else 10**6,
                feature.kind,
                feature.label,
            ),
        )
        for feature in ordered_features:
            key = (feature.kind, feature.label)
            if key in line_seen:
                continue
            line_seen.add(key)
            if key not in grouped:
                grouped[key] = _FeatureGroup(
                    kind=feature.kind,
                    label=feature.label,
                    evidence_lines=[line_index],
                    earliest_ply_indices=[
                        feature.ply_index if feature.ply_index is not None else 10**6
                    ],
                    description=feature.description,
                )
                continue
            group = grouped[key]
            grouped[key] = _FeatureGroup(
                kind=group.kind,
                label=group.label,
                evidence_lines=[*group.evidence_lines, line_index],
                earliest_ply_indices=[
                    *group.earliest_ply_indices,
                    feature.ply_index if feature.ply_index is not None else 10**6,
                ],
                description=group.description,
            )
    return grouped


def _build_recurring_ideas(
    grouped_features: dict[tuple[IdeaKind, str], _FeatureGroup],
    line_count: int,
) -> list[RecurringIdea]:
    """Promote grouped features into recurring ideas using conservative rules."""
    if line_count <= 1:
        return []
    minimum_support_lines = 2 if line_count >= 3 else line_count
    ideas: list[RecurringIdea] = []
    for group in grouped_features.values():
        if len(group.evidence_lines) < minimum_support_lines:
            continue
        support = len(group.evidence_lines) / line_count
        ideas.append(
            RecurringIdea(
                kind=group.kind,
                label=group.label,
                evidence_lines=group.evidence_lines,
                support=support,
                description=group.description,
            )
        )
    return sorted(
        ideas,
        key=lambda idea: (
            -idea.support,
            _average_earliest_ply(grouped_features[(idea.kind, idea.label)]),
            idea.label,
        ),
    )


def _average_earliest_ply(group: _FeatureGroup) -> float:
    """Return the average earliest ply index for a grouped feature."""
    return sum(group.earliest_ply_indices) / len(group.earliest_ply_indices)


def _classify_line_divergence(
    recurring_ideas: list[RecurringIdea],
    line_count: int,
) -> str:
    """Classify whether lines converge on one plan or diverge."""
    if line_count <= 1:
        return "distinct_plans"
    if not recurring_ideas:
        return "distinct_plans"
    if recurring_ideas[0].support >= 0.75:
        return "strong_convergence"
    return "partial_convergence"


def _build_position_summary(
    recurring_ideas: list[RecurringIdea],
    divergence_class: str,
) -> str:
    """Build a short position summary from recurring ideas."""
    if not recurring_ideas:
        return "The top candidate lines suggest distinct plans rather than one shared idea."
    top_idea = recurring_ideas[0]
    if divergence_class == "strong_convergence":
        return f"The strongest lines all build toward {top_idea.label}."
    return f"The top candidate lines share an emphasis on {top_idea.label}."


def _build_side_to_move_plan(
    lines: list[CandidateLine],
    features_by_line: list[list[LineFeature]],
    recurring_ideas: list[RecurringIdea],
    divergence_class: str,
) -> str:
    """Describe what the side to move appears to be trying to achieve."""
    if recurring_ideas:
        if len(recurring_ideas) >= 2:
            return (
                f"The strong lines aim to combine {recurring_ideas[0].label} with "
                f"{recurring_ideas[1].label}."
            )
        return f"The strong lines aim to support {recurring_ideas[0].label}."
    best_line_features = features_by_line[0]
    if best_line_features:
        return (
            "The best line emphasizes "
            f"{best_line_features[0].label}, but the top options do not yet show "
            "one shared plan."
        )
    return (
        f"The best line starts with {lines[0].root_move_san}, but the top options "
        "do not yet show one shared plan."
    )


def _build_opponent_counterplay(
    features_by_line: list[list[LineFeature]],
    recurring_ideas: list[RecurringIdea],
) -> str:
    """Describe the opponent's visible resources conservatively."""
    tactical_ideas = [
        idea for idea in recurring_ideas if idea.kind == "tactical_motif"
    ]
    if tactical_ideas:
        return (
            "The candidate lines repeatedly feature "
            f"{tactical_ideas[0].label}, so forcing play needs attention."
        )
    continuation_tactical = any(
        feature.kind == "tactical_motif" and (feature.ply_index or 0) > 0
        for line_features in features_by_line
        for feature in line_features
    )
    if continuation_tactical:
        return "The continuations show forcing play that must be respected."
    return _NO_COUNTERPLAY_TEXT


def _build_critical_decision(
    recurring_ideas: list[RecurringIdea],
    divergence_class: str,
) -> str | None:
    """Describe the main strategic split only when the lines diverge."""
    if divergence_class == "strong_convergence":
        return None
    if not recurring_ideas:
        return (
            "The main decision is which plan to prioritize, because the top lines "
            "do not yet converge on one shared idea."
        )
    return (
        "The main decision is how to balance "
        f"{recurring_ideas[0].label} against the other candidate plans."
    )


def _build_best_move_role(
    lines: list[CandidateLine],
    features_by_line: list[list[LineFeature]],
    recurring_ideas: list[RecurringIdea],
    divergence_class: str,
) -> str:
    """Describe what the best move contributes relative to the shared plan."""
    best_line = lines[0]
    best_features = features_by_line[0]
    if recurring_ideas:
        top_idea = recurring_ideas[0]
        best_feature = _find_feature_by_label(best_features, top_idea.label)
        if best_feature is not None and _is_earliest_supporting_feature(
            top_idea,
            best_feature,
            features_by_line,
        ):
            return (
                f"The best move supports {top_idea.label} more directly than the "
                "other leading candidates."
            )
        if best_feature is not None:
            return (
                f"The best move keeps the shared {top_idea.label} plan intact."
            )
    if best_features:
        return (
            f"The best move starts with {best_features[0].label}, making it the "
            "engine's preferred route."
        )
    if divergence_class == "distinct_plans":
        return (
            f"The best move {best_line.root_move_san} is the engine's preferred "
            "route among otherwise distinct candidate plans."
        )
    return f"The best move {best_line.root_move_san} fits the shared plan most directly."


def _build_line_divergence_summary(divergence_class: str) -> str:
    """Return a concise summary of line convergence vs divergence."""
    if divergence_class == "strong_convergence":
        return (
            "The top lines mostly share the same plan and differ by move order or timing."
        )
    if divergence_class == "partial_convergence":
        return "The top lines share some ideas, but the execution differs."
    return "The top lines point to distinct plans rather than one common approach."


def _build_main_ideas(
    lines: list[CandidateLine],
    theme: PositionTheme,
    features_by_line: list[list[LineFeature]],
) -> list[str]:
    """Build short list-shaped teaching points for the position."""
    if theme.recurring_ideas:
        main_ideas = [
            f"The strong lines repeatedly build toward {idea.label}."
            for idea in theme.recurring_ideas[:3]
        ]
        if (
            theme.critical_decision is not None
            and len(main_ideas) < 3
            and theme.line_divergence_summary
        ):
            main_ideas.append(theme.line_divergence_summary)
        return main_ideas
    best_line_features = features_by_line[0]
    if best_line_features:
        return [
            f"The best line starts with {best_line_features[0].label}.",
            "The top candidates do not yet converge on one shared plan.",
        ]
    return [
        f"The best line starts with {lines[0].root_move_san}.",
        "The top candidates do not yet converge on one shared plan.",
    ]


def _build_commonality_text(theme: PositionTheme) -> str:
    """Summarize what the candidate lines have in common."""
    if not theme.recurring_ideas:
        return "The top lines do not share one strong common idea yet."
    if len(theme.recurring_ideas) >= 2:
        return (
            f"The good lines all support {theme.recurring_ideas[0].label} and "
            f"{theme.recurring_ideas[1].label}, even if they reach them by "
            "different move orders."
        )
    if theme.line_divergence_summary.startswith("The top lines mostly share"):
        return (
            f"The good lines all support {theme.recurring_ideas[0].label}, even if "
            "they differ by move order or timing."
        )
    return (
        f"The good lines share {theme.recurring_ideas[0].label}, but the "
        "execution differs."
    )


def _build_candidate_move_roles(
    lines: list[CandidateLine],
    features_by_line: list[list[LineFeature]],
    theme: PositionTheme,
) -> list[str]:
    """Build one concise role sentence per candidate move."""
    return [
        _build_candidate_move_role(
            line,
            line_features,
            theme,
            is_best_move=index == 0,
        )
        for index, (line, line_features) in enumerate(zip(lines, features_by_line))
    ]


def _build_candidate_move_role(
    line: CandidateLine,
    line_features: list[LineFeature],
    theme: PositionTheme,
    *,
    is_best_move: bool,
) -> str:
    """Build a concise role sentence for one candidate move."""
    if theme.recurring_ideas:
        top_idea = theme.recurring_ideas[0]
        idea_feature = _find_feature_by_label(line_features, top_idea.label)
        if idea_feature is not None:
            if (idea_feature.ply_index or 0) == 0:
                if is_best_move:
                    return (
                        f"{line.root_move_san} is the most direct route toward "
                        f"{top_idea.label}."
                    )
                return f"{line.root_move_san} supports {top_idea.label} directly."
            return (
                f"{line.root_move_san} supports the same {top_idea.label} plan "
                "more gradually."
            )
    if line_features:
        return f"{line.root_move_san} emphasizes {line_features[0].label}."
    return f"{line.root_move_san} is one of the engine's candidate plans."


def _find_feature_by_label(
    features: list[LineFeature],
    label: str,
) -> LineFeature | None:
    """Return the first feature with the requested label."""
    for feature in features:
        if feature.label == label:
            return feature
    return None


def _is_earliest_supporting_feature(
    idea: RecurringIdea,
    best_feature: LineFeature,
    features_by_line: list[list[LineFeature]],
) -> bool:
    """Return whether the best-line feature appears earlier than other support."""
    best_ply = best_feature.ply_index if best_feature.ply_index is not None else 10**6
    for line_index in idea.evidence_lines:
        feature = _find_feature_by_label(features_by_line[line_index], idea.label)
        if feature is None:
            continue
        ply_index = feature.ply_index if feature.ply_index is not None else 10**6
        if ply_index < best_ply:
            return False
    return True
