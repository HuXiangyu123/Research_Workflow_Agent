from __future__ import annotations

from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class Settings:
    """LLM 配置：DeepSeek 官方、OpenAI 兼容端，或火山方舟 Ark（OpenAI Chat 兼容）。"""

    llm_provider: str  # "deepseek" | "openai" | "ark"
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    ark_api_key: str
    ark_base_url: str
    ark_model: str

    # SearXNG 配置（新增）
    searxng_base_url: str = field(default="")

    # 数据库配置（新增，用于 PostgreSQL 连接）
    database_url: str = field(default="")
    llm_timeout_s: int = field(default=45)
    llm_max_retries: int = field(default=1)

    @staticmethod
    def from_env() -> "Settings":
        deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        ark_key = os.getenv("ARK_API_KEY", "").strip()
        provider = os.getenv("LLM_PROVIDER", "").strip().lower()

        if provider == "openai":
            llm_provider = "openai"
        elif provider == "ark":
            llm_provider = "ark"
        elif provider == "deepseek":
            llm_provider = "deepseek"
        elif openai_key and not deepseek_key and not ark_key:
            llm_provider = "openai"
        elif ark_key and not deepseek_key and not openai_key:
            llm_provider = "ark"
        else:
            llm_provider = "deepseek"

        return Settings(
            llm_provider=llm_provider,
            deepseek_api_key=deepseek_key,
            deepseek_base_url=os.getenv(
                "DEEPSEEK_API_BASE", "https://api.deepseek.com"
            ).strip(),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip(),
            openai_api_key=openai_key,
            openai_base_url=os.getenv("OPENAI_API_BASE", "").strip(),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o").strip(),
            ark_api_key=ark_key,
            ark_base_url=os.getenv(
                "ARK_API_BASE", "https://ark.cn-beijing.volces.com/api/v3"
            ).strip(),
            ark_model=os.getenv("ARK_MODEL", "").strip(),
            searxng_base_url=os.getenv("SEARXNG_BASE_URL", "http://127.0.0.1:8080").strip(),
            database_url=os.getenv("DATABASE_URL", "").strip(),
            llm_timeout_s=int(os.getenv("LLM_TIMEOUT_S", "45").strip()),
            llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "1").strip()),
        )
