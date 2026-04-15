"""Skills models — Phase 4: SkillManifest, SkillRun."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.models.agent import AgentRole


class SkillBackend(str, Enum):
    LOCAL_GRAPH = "local_graph"
    LOCAL_FUNCTION = "local_function"
    MCP_PROMPT = "mcp_prompt"
    MCP_TOOLCHAIN = "mcp_toolchain"


class SkillVisibility(str, Enum):
    AUTO = "auto"
    EXPLICIT = "explicit"
    BOTH = "both"


class SkillManifest(BaseModel):
    """Skills 注册表条目（对应前端 Skill Palette）。"""

    model_config = ConfigDict(extra="forbid")

    skill_id: str
    name: str
    description: str
    backend: SkillBackend
    visibility: SkillVisibility = SkillVisibility.BOTH
    default_agent: AgentRole
    tags: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_artifact_type: str | None = None
    backend_ref: str  # e.g. "mcp:academic.search" or "graph:paper_plan_builder"


class SkillRunRequest(BaseModel):
    """执行 skill 的请求。"""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    task_id: str | None = None
    skill_id: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    preferred_agent: AgentRole | None = None


class SkillRunResponse(BaseModel):
    """Skill 执行结果。"""

    model_config = ConfigDict(extra="forbid")

    skill_run_id: str = Field(default_factory=lambda: f"skr_{uuid4().hex[:12]}")
    workspace_id: str
    task_id: str | None = None
    skill_id: str
    backend: SkillBackend
    output_artifact_ids: list[str] = Field(default_factory=list)
    trace_refs: list[str] = Field(default_factory=list)
    summary: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
