"""Config models — Phase 4: ExecutionMode, Phase4Config."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, ConfigDict, Field


class ExecutionMode(str, Enum):
    LEGACY = "legacy"
    HYBRID = "hybrid"
    V2 = "v2"


class AgentMode(str, Enum):
    AUTO = "auto"
    PLANNER = "planner"
    RETRIEVER = "retriever"
    ANALYST = "analyst"
    REVIEWER = "reviewer"


class SupervisorMode(str, Enum):
    GRAPH = "graph"
    LLM_HANDOFF = "llm_handoff"


class NodeBackendMode(str, Enum):
    LEGACY = "legacy"
    V2 = "v2"
    AUTO = "auto"


class AgentParadigm(str, Enum):
    """Agent 设计模式枚举 — 用于面试故事和技术文档。"""

    PLAN_AND_EXECUTE = "plan_and_execute"
    REACT = "react"
    REFLEXION = "reflexion"
    TAG = "tag"  # Tool-Augmented Generation
    REASONING_VIA_ARTIFACTS = "reasoning_via_artifacts"
    HIERARCHICAL = "hierarchical"


class NodeBackendConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Current research graph nodes
    clarify: NodeBackendMode = NodeBackendMode.LEGACY
    search_plan: NodeBackendMode = NodeBackendMode.AUTO
    search: NodeBackendMode = NodeBackendMode.AUTO
    extract: NodeBackendMode = NodeBackendMode.LEGACY
    draft: NodeBackendMode = NodeBackendMode.AUTO
    review: NodeBackendMode = NodeBackendMode.AUTO
    persist_artifacts: NodeBackendMode = NodeBackendMode.LEGACY

    # Deprecated aliases kept for backward compatibility with older Phase 4 UIs.
    plan_search: NodeBackendMode | None = None
    search_corpus: NodeBackendMode | None = None
    extract_cards: NodeBackendMode | None = None
    synthesize: NodeBackendMode | None = None
    revise: NodeBackendMode | None = None
    write_report: NodeBackendMode | None = None

    def mode_for(self, node_name: str) -> NodeBackendMode:
        alias_map: dict[str, tuple[NodeBackendMode | None, NodeBackendMode, NodeBackendMode]] = {
            "clarify": (None, self.clarify, NodeBackendMode.LEGACY),
            "search_plan": (self.plan_search, self.search_plan, NodeBackendMode.AUTO),
            "search": (self.search_corpus, self.search, NodeBackendMode.AUTO),
            "extract": (self.extract_cards, self.extract, NodeBackendMode.LEGACY),
            "draft": (self.synthesize or self.write_report, self.draft, NodeBackendMode.AUTO),
            "review": (self.revise, self.review, NodeBackendMode.AUTO),
            "persist_artifacts": (None, self.persist_artifacts, NodeBackendMode.LEGACY),
        }
        deprecated_mode, current_mode, default_mode = alias_map.get(
            node_name,
            (None, NodeBackendMode.AUTO, NodeBackendMode.AUTO),
        )
        if deprecated_mode is not None and current_mode == default_mode:
            return deprecated_mode
        return current_mode


class Phase4Config(BaseModel):
    """Phase 4 全局配置：控制 legacy/hybrid/v2 执行模式、MCP/Skills/Replan 开关。"""

    model_config = ConfigDict(extra="ignore")

    execution_mode: ExecutionMode = ExecutionMode.HYBRID
    agent_mode: AgentMode = AgentMode.AUTO
    supervisor_mode: SupervisorMode = SupervisorMode.GRAPH
    enable_mcp: bool = True
    enable_skills: bool = True
    enable_replan: bool = True
    auto_fill: bool = False
    node_backends: NodeBackendConfig = Field(default_factory=NodeBackendConfig)


# ─── 模型配置 ─────────────────────────────────────────────────────────────────


class LLMProviderConfig(BaseModel):
    """单个 LLM Provider 的配置。"""

    model_config = ConfigDict(extra="forbid")

    provider: str = "deepseek"  # "deepseek" | "openai" | "ark"
    api_key_set: bool = False  # 不暴露真实 key，只告知是否已配置
    base_url: str = ""
    default_model: str = ""


class ModelConfig(BaseModel):
    """
    前端可编辑的模型配置。

    用途：
    - 前端展示当前模型配置
    - 用户切换推理/快速模型
    - 提交任务时带上选择的模型（用于任务级覆盖）
    """

    model_config = ConfigDict(extra="forbid")

    # 当前 provider 信息（只读展示）
    current_provider: str = "deepseek"

    # 可用 provider 列表（只读）
    available_providers: list[LLMProviderConfig] = Field(default_factory=list)

    # 用户选择的模型（可编辑）
    reason_model: str = ""  # 推理模型
    quick_model: str = ""  # 快速模型

    # 可用模型列表（根据 provider 动态）
    available_reason_models: list[str] = Field(default_factory=list)
    available_quick_models: list[str] = Field(default_factory=list)

    # provider 支持的模型列表（固定预定义）
    provider_models: dict[str, list[str]] = Field(default_factory=lambda: {
        "deepseek": ["deepseek-chat", "deepseek-reasoner"],
        "openai": ["gpt-5.4", "gpt-5.1-codex-mini", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "ark": ["deepseek-v3-2-251201", "doubao-pro-32k"],
    })
