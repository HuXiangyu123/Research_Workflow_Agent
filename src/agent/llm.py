from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.agent.settings import Settings


def _model_for_purpose(settings: Settings, purpose: str) -> str:
    if settings.llm_provider == "openai":
        if purpose == "reason":
            return settings.reason_model or settings.openai_model
        if purpose == "quick":
            return settings.quick_model or settings.openai_model
        return settings.openai_model or settings.reason_model or settings.quick_model

    if settings.llm_provider == "ark":
        if purpose == "reason":
            return settings.reason_model or settings.ark_model
        if purpose == "quick":
            return settings.quick_model or settings.ark_model
        return settings.ark_model or settings.reason_model or settings.quick_model

    if purpose == "reason":
        return settings.reason_model or settings.deepseek_model
    if purpose == "quick":
        return settings.quick_model or settings.deepseek_model
    return settings.deepseek_model or settings.reason_model or settings.quick_model


def _build_llm(settings: Settings, *, model: str, max_tokens: int) -> ChatOpenAI:
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
            "model": model,
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
            model=model or settings.ark_model,
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
    chosen_model = model or settings.deepseek_model
    is_reasoner = "reasoner" in chosen_model.lower()
    ds_kwargs: dict = {
        "model": chosen_model,
        "api_key": settings.deepseek_api_key,
        "base_url": settings.deepseek_base_url,
        **common_kwargs,
    }
    if not is_reasoner:
        ds_kwargs["temperature"] = 0
    return ChatOpenAI(**ds_kwargs)


def build_chat_llm(settings: Settings, max_tokens: int = 8192) -> ChatOpenAI:
    """按 Settings.llm_provider 构造默认 ChatOpenAI。"""
    return _build_llm(settings, model=_model_for_purpose(settings, "chat"), max_tokens=max_tokens)


def build_deepseek_chat(settings: Settings, max_tokens: int = 8192) -> ChatOpenAI:
    """兼容旧名：与 build_chat_llm 相同。"""
    return build_chat_llm(settings, max_tokens=max_tokens)


def build_reason_llm(
    settings: Settings,
    max_tokens: int = 8192,
    timeout_s: int | None = None,
) -> ChatOpenAI:
    """Compatibility alias for reasoning-oriented calls.

    The active provider/model routing still lives in ``Settings``. ``timeout_s``
    is accepted because older research nodes pass longer timeouts for batch
    extraction/drafting.
    """
    model = _model_for_purpose(settings, "reason")
    if timeout_s is None:
        return _build_llm(settings, model=model, max_tokens=max_tokens)
    tuned = Settings(
        llm_provider=settings.llm_provider,
        deepseek_api_key=settings.deepseek_api_key,
        deepseek_base_url=settings.deepseek_base_url,
        deepseek_model=settings.deepseek_model,
        openai_api_key=settings.openai_api_key,
        openai_base_url=settings.openai_base_url,
        openai_model=settings.openai_model,
        reason_model=settings.reason_model,
        quick_model=settings.quick_model,
        ark_api_key=settings.ark_api_key,
        ark_base_url=settings.ark_base_url,
        ark_model=settings.ark_model,
        searxng_base_url=settings.searxng_base_url,
        database_url=settings.database_url,
        llm_timeout_s=timeout_s,
        llm_max_retries=settings.llm_max_retries,
    )
    return _build_llm(tuned, model=model, max_tokens=max_tokens)


def build_quick_llm(settings: Settings, max_tokens: int = 4096) -> ChatOpenAI:
    """Build a lightweight model for planning/tool-selection passes."""
    return _build_llm(settings, model=_model_for_purpose(settings, "quick"), max_tokens=max_tokens)
