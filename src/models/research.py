"""Research 相关的数据模型（SearchPlan、Memory、ResearchState）。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ─── SearchPlan ────────────────────────────────────────────────────────────────


class CoverageStrategy(str, Enum):
    BROAD = "broad"      # 广泛覆盖，快速迭代
    FOCUSED = "focused"  # 精准覆盖，慢速精炼
    HYBRID = "hybrid"    # 先广后精


class DedupStrategy(str, Enum):
    EXACT = "exact"       # 完全去重
    SEMANTIC = "semantic" # 语义去重
    NONE = "none"         # 不去重


class SearchQueryGroup(BaseModel):
    group_id: str = Field(..., description="查询组唯一 ID")
    queries: list[str] = Field(..., description="该组内的查询列表")
    intent: str = Field(..., description="查询意图描述（broad/focused/background）")
    priority: int = Field(default=1, ge=1, le=3, description="优先级（1=最高）")
    expected_hits: int = Field(default=20, description="期望返回数量")
    notes: str = Field(default="", description="备注")


class SearchPlan(BaseModel):
    """SearchPlanAgent 的输出：从 ResearchBrief 推导出的可执行搜索计划。"""

    schema_version: str = Field(default="v1", description="Schema 版本")
    plan_goal: str = Field(..., description="搜索计划的核心目标")
    coverage_strategy: CoverageStrategy = Field(default=CoverageStrategy.HYBRID)
    query_groups: list[SearchQueryGroup] = Field(default_factory=list)
    source_preferences: list[str] = Field(
        default_factory=lambda: ["arxiv", "semantic_scholar", "google_scholar"],
        description="优先搜索的来源"
    )
    dedup_strategy: DedupStrategy = Field(default=DedupStrategy.SEMANTIC)
    rerank_required: bool = Field(default=True, description="是否需要对候选结果进行重排序")
    max_candidates_per_query: int = Field(default=30, ge=1, le=100)
    requires_local_corpus: bool = Field(default=False, description="是否需要查询本地语料")
    coverage_notes: str = Field(default="", description="覆盖率说明")
    planner_warnings: list[str] = Field(default_factory=list)
    followup_search_seeds: list[str] = Field(
        default_factory=list, description="后续搜索种子词"
    )
    followup_needed: bool = Field(default=False, description="是否需要后续搜索")

    class Config:
        use_enum_values = True


# ─── Working Memory ────────────────────────────────────────────────────────────


class SearchPlannerMemory(BaseModel):
    """SearchPlanAgent 的工作记忆。"""

    attempted_queries: list[str] = Field(default_factory=list)
    query_to_hits: dict[str, int] = Field(default_factory=dict)
    empty_queries: list[str] = Field(default_factory=list)
    high_noise_queries: list[str] = Field(default_factory=list)
    subquestion_coverage_map: dict[str, list[str]] = Field(default_factory=dict)
    source_usage_stats: dict[str, int] = Field(default_factory=dict)
    planner_reflections: list[str] = Field(default_factory=list)
    iteration_count: int = Field(default=0)
    remaining_budget: int = Field(default=10, ge=0)
    last_action: str = Field(default="init", description="上一步动作（search/expand/rewrite/stop）")
    last_hits: int = Field(default=0)


# ─── SearchPlanAgent 结果 ─────────────────────────────────────────────────────


class SearchPlanResult(BaseModel):
    """SearchPlanAgent 的返回结果（含工作记忆快照）。"""

    plan: SearchPlan
    memory: SearchPlannerMemory
    warnings: list[str] = Field(default_factory=list)
    raw_model_output: Optional[str] = Field(default=None, description="原始 LLM 输出（调试用）")


# ─── ResearchState（Phase1 定义）───────────────────────────────────────────────


class ArtifactRef(BaseModel):
    artifact_id: str
    workspace_id: str
    task_id: str
    artifact_type: str  # "research_brief" | "search_plan" | "paper_card"
    uri_or_path: str
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class ResearchState(BaseModel):
    """Research workflow 的统一状态。"""

    task_id: str
    workspace_id: str
    user_input: dict = Field(default_factory=dict)
    uploaded_sources: list[str] = Field(default_factory=list)
    brief: Optional["ResearchBrief"] = None
    search_plan: Optional[SearchPlan] = None
    paper_cards: list["PaperCard"] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    current_stage: str = "created"
    error: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}


# ─── 导入兼容（避免循环 import）───────────────────────────────────────────────


def _lazy_import():
    from src.models.paper import PaperCard
    from src.research.research_brief import ResearchBrief
    return ResearchBrief, PaperCard
