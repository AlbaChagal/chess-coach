"""Build LLM prompts from typed explanation payloads."""

from __future__ import annotations

from chesscoach.explanation.models import (
    ExplainedMove,
    StructuredExplanation,
    StructuredPlayedMoveExplanation,
    TacticInfo,
)

_SYSTEM_PROMPT = """\
You are a practical chess coach explaining the best move in a position.
Write 3-5 short sentences in plain language.
Start with the core move idea, then cover the threat or plan it creates.
Briefly compare the best move to the next alternatives.
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
        f"{tactics_best}\n"
        f"{tactics_played}\n"
        f"\n"
        "Write a concise coaching explanation of why this best move is strong, "
        "what practical idea it creates, and why the alternatives fall short."
    )

    return _SYSTEM_PROMPT, user


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
        f"{tactics_played}\n"
        f"{tactics_best}\n"
        f"\n"
        "Write a concise coaching explanation that compares the played move to "
        "the best move, explains the consequence, and ends with a practical lesson."
    )
    return _SYSTEM_PROMPT, user
