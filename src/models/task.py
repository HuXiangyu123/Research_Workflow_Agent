from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskRecord(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    input_type: str = "arxiv"
    input_value: str = ""
    report_mode: Literal["draft", "full"] = "draft"
    source_type: Literal["arxiv", "pdf", "research"] = "arxiv"
    paper_type: Literal["regular", "survey"] | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    draft_markdown: str | None = None
    full_markdown: str | None = None
    result_markdown: str | None = None
    brief: dict[str, Any] | None = None
    search_plan: dict[str, Any] | None = None
    current_stage: str | None = None
    report_context_snapshot: str | None = None
    followup_hints: list[str] = Field(default_factory=list)
    chat_history: list[dict] = Field(default_factory=list)
    chat_summary: str | None = None
    error: str | None = None
    node_events: list[dict] = Field(default_factory=list)
