from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.agent.settings import Settings


def build_chat_llm(settings: Settings, max_tokens: int = 8192) -> ChatOpenAI:
    """按 Settings.llm_provider 构造 ChatOpenAI（DeepSeek / OpenAI 兼容 / 火山 Ark）。"""
    common_kwargs = {
        "max_tokens": max_tokens,
        "timeout": settings.llm_timeout_s,
        "max_retries": settings.llm_max_retries,
    }

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError(
                "已选择 OpenAI 兼容后端（LLM_PROVIDER=openai 或仅配置了 OPENAI_API_KEY），"
                "但未设置 OPENAI_API_KEY。请在 .env 中填写。"
            )
        kwargs: dict = {
            "model": settings.openai_model,
            "api_key": settings.openai_api_key,
            "temperature": 0,
            **common_kwargs,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return ChatOpenAI(**kwargs)

    if settings.llm_provider == "ark":
        if not settings.ark_api_key:
            raise RuntimeError(
                "已选择火山方舟（LLM_PROVIDER=ark 或仅配置了 ARK_API_KEY），"
                "但未设置 ARK_API_KEY。请在 .env 中填写。"
            )
        if not settings.ark_model:
            raise RuntimeError(
                "使用方舟时请设置 ARK_MODEL（控制台中的推理接入点 ID / 模型名，"
                "例如 deepseek-v3-2-251201）。"
            )
        return ChatOpenAI(
            model=settings.ark_model,
            api_key=settings.ark_api_key,
            base_url=settings.ark_base_url,
            temperature=0,
            **common_kwargs,
        )

    if not settings.deepseek_api_key:
        raise RuntimeError(
            "缺少 DEEPSEEK_API_KEY。请在 .env 中配置，或改用 OpenAI / 方舟："
            "OPENAI_API_KEY + LLM_PROVIDER=openai，或 ARK_API_KEY + ARK_MODEL + LLM_PROVIDER=ark。"
        )
    is_reasoner = "reasoner" in settings.deepseek_model.lower()
    ds_kwargs: dict = {
        "model": settings.deepseek_model,
        "api_key": settings.deepseek_api_key,
        "base_url": settings.deepseek_base_url,
        **common_kwargs,
    }
    if not is_reasoner:
        ds_kwargs["temperature"] = 0
    return ChatOpenAI(**ds_kwargs)


def build_deepseek_chat(settings: Settings, max_tokens: int = 8192) -> ChatOpenAI:
    """兼容旧名：与 build_chat_llm 相同。"""
    return build_chat_llm(settings, max_tokens=max_tokens)
