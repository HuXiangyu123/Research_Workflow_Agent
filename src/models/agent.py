"""Agent models — Phase 4: AgentRole, AgentRun, Replan."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.models.config import AgentMode


class AgentRole(str, Enum):
    SUPERVISOR = "supervisor"
    PLANNER = "planner"
    RETRIEVER = "retriever"
    ANALYST = "analyst"
    REVIEWER = "reviewer"


class AgentVisibility(str, Enum):
    AUTO = "auto"
    EXPLICIT = "explicit"
    BOTH = "both"


class AgentDescriptor(BaseModel):
    """单个 agent 角色描述（用于目录注册）。"""

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    role: AgentRole
    title: str
    description: str
    visibility: AgentVisibility = AgentVisibility.BOTH
    supported_skills: list[str] = Field(default_factory=list)
    supported_nodes: list[str] = Field(default_factory=list)


class AgentRunRequest(BaseModel):
    """请求运行指定角色的 agent。"""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    task_id: str | None = None
    role: AgentRole | None = None
    mode: AgentMode = AgentMode.AUTO
    node_name: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    preferred_skill_id: str | None = None


class AgentRunResponse(BaseModel):
    """Agent 运行结果。"""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: f"agr_{uuid4().hex[:12]}")
    workspace_id: str
    task_id: str | None = None
    role: AgentRole
    selected_skill_id: str | None = None
    output_artifact_ids: list[str] = Field(default_factory=list)
    trace_refs: list[str] = Field(default_factory=list)
    collaboration_trace: list[dict[str, Any]] = Field(default_factory=list)
    summary: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ReplanTrigger(str, Enum):
    REVIEWER = "reviewer"
    RETRIEVER = "retriever"
    USER = "user"


class ReplanRequest(BaseModel):
    """请求重新规划（reviewer 指出 coverage gap 或用户显式触发）。"""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    task_id: str
    trigger: ReplanTrigger
    reason: str
    target_stage: str = "search_plan"
    inputs: dict[str, Any] = Field(default_factory=dict)


class ReplanResponse(BaseModel):
    """重新规划结果。"""

    model_config = ConfigDict(extra="forbid")

    replan_id: str = Field(default_factory=lambda: f"rpl_{uuid4().hex[:12]}")
    workspace_id: str
    task_id: str
    trigger: ReplanTrigger
    target_stage: str
    output_artifact_ids: list[str] = Field(default_factory=list)
    trace_refs: list[str] = Field(default_factory=list)
    collaboration_trace: list[dict[str, Any]] = Field(default_factory=list)
    summary: str | None = None
