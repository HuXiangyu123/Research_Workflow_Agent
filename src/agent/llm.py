from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.agent.settings import Settings


def build_deepseek_chat(settings: Settings) -> ChatOpenAI:
    if not settings.deepseek_api_key:
        raise RuntimeError(
            "缺少 DEEPSEEK_API_KEY。请在 .env 中配置，或从 .env.example 复制后填写。"
        )

    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
    )

