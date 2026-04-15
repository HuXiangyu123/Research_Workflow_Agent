"""Trace models — Phase 3: NodeRun, ToolRun, TraceEvent."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


# ─── Run Status ───────────────────────────────────────────────────────────────


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


# ─── Node Run ────────────────────────────────────────────────────────────────


class NodeRun(BaseModel):
    """
    记录一次 graph node 执行。

    生命周期：
    1. node 进入 → status=RUNNING, started_at=now
    2. node 执行中 → 可追加 warning_messages
    3. node 完成 → status=SUCCEEDED/FAILED, ended_at=now, duration_ms=computed
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: f"nr_{uuid4().hex[:12]}")
    task_id: str
    workspace_id: str
    node_name: str
    stage: str
    status: RunStatus = RunStatus.PENDING
    started_at: datetime | None = None
    ended_at: datetime | None = None
    input_artifact_ids: list[str] = Field(default_factory=list)
    output_artifact_ids: list[str] = Field(default_factory=list)
    warning_messages: list[str] = Field(default_factory=list)
    error_message: str | None = None
    duration_ms: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Tool Run ───────────────────────────────────────────────────────────────


class ToolRun(BaseModel):
    """
    记录一次工具调用。

    嵌套在 NodeRun 内部（通过 parent_run_id 关联）。
    """

    model_config = ConfigDict(extra="forbid")

    tool_run_id: str = Field(default_factory=lambda: f"tr_{uuid4().hex[:12]}")
    parent_run_id: str
    task_id: str
    workspace_id: str
    node_name: str
    tool_name: str
    status: RunStatus = RunStatus.PENDING
    started_at: datetime | None = None
    ended_at: datetime | None = None
    input_summary: dict[str, Any] = Field(default_factory=dict)
    output_summary: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    duration_ms: int | None = None


# ─── Trace Event Type ────────────────────────────────────────────────────────


class TraceEventType(str, Enum):
    TASK_CREATED = "task_created"
    NODE_STARTED = "node_started"
    NODE_FINISHED = "node_finished"
    NODE_FAILED = "node_failed"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    TOOL_FAILED = "tool_failed"
    ARTIFACT_SAVED = "artifact_saved"
    REVIEW_GENERATED = "review_generated"
    WARNING = "warning"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    STAGE_CHANGED = "stage_changed"


# ─── Trace Event ────────────────────────────────────────────────────────────


class TraceEvent(BaseModel):
    """
    流式 SSE 事件。

    用于实时推送 node/tool 状态变化到前端。
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: f"evt_{uuid4().hex[:12]}")
    task_id: str
    workspace_id: str
    run_id: str | None = None
    tool_run_id: str | None = None
    event_type: TraceEventType
    ts: datetime = Field(default_factory=datetime.utcnow)
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_sse_dict(self) -> dict:
        return {
            "event": self.event_type.value,
            "data": self.model_dump(mode="json"),
        }
