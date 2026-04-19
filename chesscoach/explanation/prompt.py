"""Build LLM prompts from a structured explanation payload."""

from __future__ import annotations

from chesscoach.explanation.models import ExplainedMove, StructuredExplanation, TacticInfo

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


def build_prompt(
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
