"""Workspace models — Phase 3: ArtifactType, WorkspaceArtifact, ArtifactRef."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


# ─── Artifact Type ─────────────────────────────────────────────────────────────


class ArtifactType(str, Enum):
    """Workspace 中支持的 artifact 类型。"""

    BRIEF = "brief"
    SEARCH_PLAN = "search_plan"
    PAPER_CARD = "paper_card"
    RAG_RESULT = "rag_result"
    COMPARISON_MATRIX = "comparison_matrix"
    REPORT_OUTLINE = "report_outline"
    REPORT_DRAFT = "report_draft"
    REVIEW_FEEDBACK = "review_feedback"
    NODE_TRACE = "node_trace"
    TOOL_TRACE = "tool_trace"
    EVAL_REPORT = "eval_report"
    UPLOAD = "upload"
    TASK_LOG = "task_log"
    RAW_INPUT = "raw_input"


# ─── Artifact Ref ─────────────────────────────────────────────────────────────


class ArtifactRef(BaseModel):
    """轻量引用，用于列表展示。"""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    artifact_type: ArtifactType
    title: str


# ─── Workspace Artifact ────────────────────────────────────────────────────────


class WorkspaceArtifact(BaseModel):
    """
    Workspace 中的研究产物。

    与 ResearchState.artifacts 的区别：
    - ResearchState.artifacts 是运行时状态中的引用（轻量）
    - WorkspaceArtifact 是持久化的完整记录（包含 content_ref、metadata）
    """

    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(default_factory=lambda: f"art_{uuid4().hex[:12]}")
    workspace_id: str
    task_id: str | None = None
    artifact_type: ArtifactType
    title: str
    status: str = "ready"  # "pending" | "ready" | "failed"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by_node: str | None = None
    content_ref: str | None = None  # 存储路径或外部 URL
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Workspace Summary ─────────────────────────────────────────────────────────


class WorkspaceSummary(BaseModel):
    """Workspace 概览（面板视图）。"""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    status: str
    current_stage: str | None = None
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
