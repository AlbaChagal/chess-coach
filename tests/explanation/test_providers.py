"""Tests for LLM provider implementations (mocked API calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chesscoach.explanation.models import ExplanationError
from chesscoach.explanation.providers import ClaudeProvider, LLMProvider, OpenAIProvider


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_claude_provider_satisfies_llm_provider_protocol() -> None:
    assert isinstance(ClaudeProvider, type)
    # runtime_checkable Protocol check (uses __init__ stub, not real API key)
    with patch("anthropic.Anthropic"):
        provider = ClaudeProvider(api_key="test-key")
    assert isinstance(provider, LLMProvider)


def test_openai_provider_satisfies_llm_provider_protocol() -> None:
    with patch("openai.OpenAI"):
        provider = OpenAIProvider(api_key="test-key")
    assert isinstance(provider, LLMProvider)


# ---------------------------------------------------------------------------
# ClaudeProvider — happy path
# ---------------------------------------------------------------------------


def test_claude_provider_returns_text_on_success() -> None:
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Great move!")]

    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        instance = mock_anthropic_cls.return_value
        instance.messages.create.return_value = mock_response

        provider = ClaudeProvider(api_key="test-key")
        result = provider.complete("system", "user")

    assert result == "Great move!"


def test_claude_provider_passes_correct_model() -> None:
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="ok")]

    with patch("anthropic.Anthropic") as mock_cls:
        instance = mock_cls.return_value
        instance.messages.create.return_value = mock_response

        provider = ClaudeProvider(model="claude-haiku-4-5-20251001", api_key="test-key")
        provider.complete("sys", "usr")
        call_kwargs = instance.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"


def test_claude_provider_passes_system_and_user() -> None:
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="ok")]

    with patch("anthropic.Anthropic") as mock_cls:
        instance = mock_cls.return_value
        instance.messages.create.return_value = mock_response

        provider = ClaudeProvider(api_key="test-key")
        provider.complete("my system", "my user")
        call_kwargs = instance.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "my system"
        assert call_kwargs["messages"][0]["content"] == "my user"


# ---------------------------------------------------------------------------
# ClaudeProvider — error path
# ---------------------------------------------------------------------------


def test_claude_provider_raises_explanation_error_on_failure() -> None:
    with patch("anthropic.Anthropic") as mock_cls:
        instance = mock_cls.return_value
        instance.messages.create.side_effect = RuntimeError("network error")

        provider = ClaudeProvider(api_key="test-key")
        with pytest.raises(ExplanationError):
            provider.complete("sys", "usr")


# ---------------------------------------------------------------------------
# OpenAIProvider — happy path
# ---------------------------------------------------------------------------


def test_openai_provider_returns_text_on_success() -> None:
    mock_message = MagicMock()
    mock_message.content = "Nice fork!"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("openai.OpenAI") as mock_cls:
        instance = mock_cls.return_value
        instance.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(api_key="test-key")
        result = provider.complete("system", "user")

    assert result == "Nice fork!"


def test_openai_provider_passes_correct_model() -> None:
    mock_message = MagicMock()
    mock_message.content = "ok"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("openai.OpenAI") as mock_cls:
        instance = mock_cls.return_value
        instance.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(model="gpt-4o-mini", api_key="test-key")
        provider.complete("sys", "usr")
        call_kwargs = instance.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"


def test_openai_provider_passes_system_and_user_roles() -> None:
    mock_message = MagicMock()
    mock_message.content = "ok"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("openai.OpenAI") as mock_cls:
        instance = mock_cls.return_value
        instance.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(api_key="test-key")
        provider.complete("my system", "my user")
        messages = instance.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "my system"}
        assert messages[1] == {"role": "user", "content": "my user"}


# ---------------------------------------------------------------------------
# OpenAIProvider — error path
# ---------------------------------------------------------------------------


def test_openai_provider_raises_explanation_error_on_failure() -> None:
    with patch("openai.OpenAI") as mock_cls:
        instance = mock_cls.return_value
        instance.chat.completions.create.side_effect = RuntimeError("timeout")

        provider = OpenAIProvider(api_key="test-key")
        with pytest.raises(ExplanationError):
            provider.complete("sys", "usr")
