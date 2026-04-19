"""Settings / LLM provider resolution for DeepSeek, OpenAI, and Volcengine Ark."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agent.llm import build_chat_llm, build_quick_llm, build_reason_llm
from src.agent.settings import Settings


def test_from_env_prefers_ark_when_only_ark_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("ARK_API_KEY", "ak-test")
    monkeypatch.setenv("ARK_MODEL", "deepseek-v3-2-251201")
    s = Settings.from_env()
    assert s.llm_provider == "ark"
    assert s.ark_base_url == "https://ark.cn-beijing.volces.com/api/v3"


def test_from_env_llm_provider_ark_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ark")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds")
    monkeypatch.setenv("OPENAI_API_KEY", "oa")
    monkeypatch.setenv("ARK_API_KEY", "ak")
    monkeypatch.setenv("ARK_MODEL", "m")
    s = Settings.from_env()
    assert s.llm_provider == "ark"


def test_build_chat_llm_ark_requires_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ark")
    monkeypatch.setenv("ARK_API_KEY", "k")
    monkeypatch.delenv("ARK_MODEL", raising=False)
    s = Settings(
        llm_provider="ark",
        deepseek_api_key="",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        openai_api_key="",
        openai_base_url="",
        openai_model="gpt-4o",
        ark_api_key="k",
        ark_base_url="https://ark.cn-beijing.volces.com/api/v3",
        ark_model="",
    )
    with pytest.raises(RuntimeError, match="ARK_MODEL"):
        build_chat_llm(s)


def test_build_chat_llm_ark_calls_chat_openai() -> None:
    s = Settings(
        llm_provider="ark",
        deepseek_api_key="",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        openai_api_key="",
        openai_base_url="",
        openai_model="gpt-4o",
        ark_api_key="ak",
        ark_base_url="https://ark.cn-beijing.volces.com/api/v3",
        ark_model="deepseek-v3-2-251201",
    )
    with patch("src.agent.llm.ChatOpenAI") as mock_cls:
        build_chat_llm(s)
    mock_cls.assert_called_once()
    _, kwargs = mock_cls.call_args
    assert kwargs["model"] == "deepseek-v3-2-251201"
    assert kwargs["api_key"] == "ak"
    assert kwargs["base_url"] == "https://ark.cn-beijing.volces.com/api/v3"
    assert kwargs["temperature"] == 0


def test_build_reason_llm_prefers_reason_model() -> None:
    s = Settings(
        llm_provider="openai",
        deepseek_api_key="",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        openai_api_key="oa",
        openai_base_url="",
        openai_model="gpt-4o",
        ark_api_key="",
        ark_base_url="https://ark.cn-beijing.volces.com/api/v3",
        ark_model="",
        reason_model="gpt-5.4",
        quick_model="gpt-5.1-codex-mini",
    )
    with patch("src.agent.llm.ChatOpenAI") as mock_cls:
        build_reason_llm(s)
    _, kwargs = mock_cls.call_args
    assert kwargs["model"] == "gpt-5.4"


def test_build_quick_llm_prefers_quick_model() -> None:
    s = Settings(
        llm_provider="openai",
        deepseek_api_key="",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        openai_api_key="oa",
        openai_base_url="",
        openai_model="gpt-4o",
        ark_api_key="",
        ark_base_url="https://ark.cn-beijing.volces.com/api/v3",
        ark_model="",
        reason_model="gpt-5.4",
        quick_model="gpt-5.1-codex-mini",
    )
    with patch("src.agent.llm.ChatOpenAI") as mock_cls:
        build_quick_llm(s)
    _, kwargs = mock_cls.call_args
    assert kwargs["model"] == "gpt-5.1-codex-mini"
