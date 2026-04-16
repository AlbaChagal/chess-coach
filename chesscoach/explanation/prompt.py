"""Build LLM prompts from a structured ExplainedMove."""

from __future__ import annotations

from chesscoach.explanation.models import ExplainedMove, TacticInfo

_SYSTEM_PROMPT = """\
You are an encouraging chess coach explaining moves to an amateur player.
Be clear and educational. Use plain, conversational language.
Keep explanations to 2-4 sentences. Do not use engine notation jargon.
Focus on the key idea the player missed or executed well.\
"""


def _tactics_text(tactics: list[TacticInfo], *, prefix: str) -> str:
    if not tactics:
        return f"{prefix}: None detected."
    items = "; ".join(t.description for t in tactics)
    return f"{prefix}: {items}"


def build_prompt(explained: ExplainedMove) -> tuple[str, str]:
    """Return a ``(system, user)`` prompt pair for the LLM.

    Args:
        explained: The fully analysed move.

    Returns:
        A tuple of ``(system_prompt, user_prompt)`` strings.
    """
    quality = explained.quality
    best = explained.best_move

    quality_str = (
        f"{quality.label.capitalize()} {quality.emoji}".strip()
        + f", -{quality.cp_loss / 100:.2f} pawns"
        if quality.cp_loss > 0
        else quality.label.capitalize()
    )

    best_score = best.score_display()
    best_line = " ".join(best.continuation) if best.continuation else "—"

    tactics_played = _tactics_text(
        explained.tactics_after_played,
        prefix="What the opponent can do after your move",
    )
    tactics_best = _tactics_text(
        explained.tactics_after_best,
        prefix="What the best move enables for you",
    )

    user = (
        f"Position (FEN): {explained.fen_before}\n"
        f"Move played: {explained.move_played_san} ({quality_str})\n"
        f"Best move was: {best.move_san} ({best_score}) — line: {best_line}\n"
        f"\n"
        f"{tactics_played}\n"
        f"{tactics_best}\n"
        f"\n"
        f"Explain what went wrong (or right) and what the player should understand."
    )

    return _SYSTEM_PROMPT, user
