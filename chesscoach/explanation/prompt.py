"""Build LLM prompts from typed explanation payloads."""

from __future__ import annotations

from chesscoach.explanation.models import (
    CandidateLine,
    ExplainedMove,
    PositionContext,
    StructuredPositionExplanation,
    StructuredExplanation,
    StructuredPlayedMoveExplanation,
    TacticInfo,
)

_POSITION_SYSTEM_PROMPT = """\
You are a practical chess coach explaining a chess position.
Write 3-5 short sentences in plain language.
Explain what the position is about before focusing on the best move.
Mention the main plan shared by the strong candidate lines when relevant.
End with a practical takeaway.
Do not mention hidden engine process, prompt structure, or uncertainty.\
"""

_BEST_MOVE_SYSTEM_PROMPT = """\
You are a practical chess coach explaining the best move in context.
Write 3-5 short sentences in plain language.
Start with the position idea, then explain why the best move fits that plan.
Briefly mention what the good lines have in common and why alternatives are less precise.
Do not mention hidden engine process, prompt structure, or uncertainty.\
"""

_PLAYED_MOVE_SYSTEM_PROMPT = """\
You are a practical chess coach comparing a played move to the best move.
Write 3-5 short sentences in plain language.
Explain what the position demanded, what shared idea was missed, and why the best move fit the position better.
End with a practical lesson.
Do not mention hidden engine process, prompt structure, or uncertainty.\
"""


def _tactics_text(tactics: list[TacticInfo], *, prefix: str) -> str:
    if not tactics:
        return f"{prefix}: None detected."
    items = "; ".join(t.description for t in tactics)
    return f"{prefix}: {items}"


def build_best_move_prompt(
    explained: ExplainedMove,
    structured: StructuredExplanation,
) -> tuple[str, str]:
    """Return a ``(system, user)`` prompt pair for the LLM.

    Args:
        explained: The fully analysed move.
        structured: The typed structured explanation payload.

    Returns:
        A tuple of ``(system_prompt, user_prompt)`` strings.
    """
    best = explained.best_move
    best_score = best.score_display()
    best_line = " ".join(best.continuation) if best.continuation else "—"
    alternatives = (
        "\n".join(
            (
                f"- {alternative.move_san} ({alternative.score_display}): "
                f"{alternative.reason}"
            )
            for alternative in structured.alternatives
        )
        if structured.alternatives
        else "- None"
    )

    tactics_played = _tactics_text(
        explained.tactics_after_played,
        prefix="Tactical motifs after the best move",
    )
    tactics_best = _tactics_text(
        explained.tactics_after_best,
        prefix="What the best move enables",
    )
    position_context = _position_context_text(structured.position_context)

    user = (
        f"Position (FEN): {explained.fen_before}\n"
        f"Best move: {best.move_san} ({best_score}) — line: {best_line}\n"
        f"Summary: {structured.summary}\n"
        f"What the move does: {structured.what_the_move_does}\n"
        f"What it threatens: {structured.what_it_threatens}\n"
        f"Why it is best: {structured.why_it_is_best}\n"
        f"Why alternatives are worse: {structured.why_alternatives_are_worse}\n"
        f"Alternatives:\n{alternatives}\n"
        f"\n"
        f"{position_context}"
        f"{tactics_best}\n"
        f"{tactics_played}\n"
        f"\n"
        "Write a concise coaching explanation of the position first, then explain "
        "why this best move fits that bigger idea and why the alternatives fall short."
    )

    return _BEST_MOVE_SYSTEM_PROMPT, user


def build_position_prompt(
    structured: StructuredPositionExplanation,
    *,
    fen_before: str,
    candidate_lines: list[CandidateLine] | None = None,
) -> tuple[str, str]:
    """Return a ``(system, user)`` prompt pair for position-level narration."""
    candidate_roles = "\n".join(
        f"- {role}" for role in structured.candidate_move_roles
    )
    candidate_line_summary = _candidate_line_summary(candidate_lines)
    main_ideas = "\n".join(f"- {idea}" for idea in structured.main_ideas)
    user = (
        f"Position (FEN): {fen_before}\n"
        f"Position summary: {structured.position_summary}\n"
        f"Main ideas:\n{main_ideas}\n"
        f"Shared plan: {structured.shared_plan}\n"
        f"Why the best move fits: {structured.why_the_best_move_fits}\n"
        f"What all good lines have in common: "
        f"{structured.what_all_good_lines_have_in_common}\n"
        f"What to watch out for: {structured.what_to_watch_out_for}\n"
        f"Candidate move roles:\n{candidate_roles}\n"
        f"{candidate_line_summary}"
        f"\n"
        "Write a concise coaching explanation of the position. Explain what the "
        "position is about first, then explain how the best move fits that bigger idea."
    )

    return _POSITION_SYSTEM_PROMPT, user


def build_played_move_prompt(
    explained: ExplainedMove,
    structured: StructuredPlayedMoveExplanation,
) -> tuple[str, str]:
    """Return a ``(system, user)`` prompt pair for played-move coaching."""
    best = explained.best_move
    best_score = best.score_display()
    best_line = " ".join(best.continuation) if best.continuation else "—"
    alternatives = (
        "\n".join(
            (
                f"- {alternative.move_san} ({alternative.score_display}): "
                f"{alternative.reason}"
            )
            for alternative in structured.alternatives
        )
        if structured.alternatives
        else "- None"
    )
    tactics_played = _tactics_text(
        explained.tactics_after_played,
        prefix="What changed after the played move",
    )
    tactics_best = _tactics_text(
        explained.tactics_after_best,
        prefix="What the best move would have created",
    )
    position_context = _position_context_text(structured.position_context)
    user = (
        f"Position (FEN): {explained.fen_before}\n"
        f"Played move: {explained.move_played_san}\n"
        f"Best move: {best.move_san} ({best_score}) — line: {best_line}\n"
        f"Summary: {structured.summary}\n"
        f"What the move tried to do: {structured.what_the_move_tried_to_do}\n"
        f"What was missed: {structured.what_was_missed}\n"
        f"What changed after the move: {structured.what_changed_after_move}\n"
        f"Why the best move was better: {structured.why_best_move_was_better}\n"
        f"Practical lesson: {structured.practical_lesson}\n"
        f"Alternatives:\n{alternatives}\n"
        f"\n"
        f"{position_context}"
        f"{tactics_best}\n"
        f"{tactics_played}\n"
        f"\n"
        "Write a concise coaching explanation that compares the played move to "
        "the best move, explains what the position demanded, and ends with a practical lesson."
    )

    return _PLAYED_MOVE_SYSTEM_PROMPT, user


def _position_context_text(position_context: PositionContext | None) -> str:
    if position_context is None:
        return ""
    return (
        f"Position summary: {position_context.position_summary}\n"
        f"Shared plan: {position_context.shared_plan}\n"
        "What the good lines have in common: "
        f"{position_context.what_all_good_lines_have_in_common}\n"
        f"What to watch out for: {position_context.what_to_watch_out_for}\n"
    )


def _candidate_line_summary(candidate_lines: list[CandidateLine] | None) -> str:
    if not candidate_lines:
        return ""
    items = "\n".join(
        f"- {line.root_move_san} ({_score_display(line)})"
        for line in candidate_lines
    )
    return f"Candidate lines:\n{items}\n"


def _score_display(line: CandidateLine) -> str:
    if line.score_mate is not None:
        sign = "+" if line.score_mate > 0 else ""
        return f"#{sign}{line.score_mate}" if line.score_mate > 0 else f"#-{abs(line.score_mate)}"
    if line.score_cp is not None:
        return f"{line.score_cp / 100:+.2f}"
    return "?"
