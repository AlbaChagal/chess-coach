"""LLM provider implementations: Claude (Anthropic) and OpenAI.

Usage::

    provider = ClaudeProvider()   # uses ANTHROPIC_API_KEY env var
    text = provider.complete(system="You are helpful.", user="Hello!")

    provider = OpenAIProvider()   # uses OPENAI_API_KEY env var
    text = provider.complete(system="You are helpful.", user="Hello!")
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from chesscoach.explanation.models import ExplanationError

LOGGER = logging.getLogger(__name__)


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal interface for an LLM text-completion backend."""

    def complete(self, system: str, user: str) -> str:
        """Send *system* + *user* messages and return the assistant reply.

        Args:
            system: System prompt setting the assistant's persona/task.
            user: User message describing what to explain.

        Returns:
            The assistant's reply as a plain string.

        Raises:
            ExplanationError: If the API call fails for any reason.
        """
        ...


class ClaudeProvider:
    """Uses the Anthropic SDK to call Claude models.

    Args:
        model: Claude model ID (default: ``claude-haiku-4-5-20251001``).
        api_key: Anthropic API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
        max_tokens: Maximum tokens in the response (default: 300).
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        max_tokens: int = 300,
    ) -> None:
        import anthropic  # lazy import — optional at module level

        self._model = model
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key)  # uses env var if None

    def complete(self, system: str, user: str) -> str:
        """Call the Claude API and return the text response."""
        LOGGER.debug("ClaudeProvider.complete model=%s max_tokens=%s", self._model, self._max_tokens)
        try:

            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            block = response.content[0]
            text = block.text  # type: ignore[union-attr]
            LOGGER.debug("ClaudeProvider received %s chars", len(text))
            return text
        except Exception as exc:
            raise ExplanationError(f"Claude API call failed: {exc}") from exc


class OpenAIProvider:
    """Uses the OpenAI SDK to call OpenAI chat models.

    Args:
        model: OpenAI model ID (default: ``gpt-4o-mini``).
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
        max_tokens: Maximum tokens in the response (default: 300).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        max_tokens: int = 300,
    ) -> None:
        import openai  # lazy import — optional at module level

        self._model = model
        self._max_tokens = max_tokens
        self._client = openai.OpenAI(api_key=api_key)  # uses env var if None

    def complete(self, system: str, user: str) -> str:
        """Call the OpenAI chat completions API and return the text response."""
        LOGGER.debug("OpenAIProvider.complete model=%s max_tokens=%s", self._model, self._max_tokens)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = response.choices[0].message.content or ""
            LOGGER.debug("OpenAIProvider received %s chars", len(text))
            return text
        except Exception as exc:
            raise ExplanationError(f"OpenAI API call failed: {exc}") from exc
