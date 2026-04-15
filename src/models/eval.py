"""Eval models — Phase 3: InternalEvalReport, EvalScope, EvalMetric."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


# ─── Eval Scope ──────────────────────────────────────────────────────────────


class EvalScope(str, Enum):
    """Internal eval 支持的范围。"""

    RETRIEVAL = "retrieval"
    REVIEWER = "reviewer"
    WORKFLOW = "workflow"


# ─── Eval Metric ─────────────────────────────────────────────────────────────


class EvalMetric(BaseModel):
    """单个评测指标。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float | int | None = None
    note: str | None = None


# ─── Eval Case Result ───────────────────────────────────────────────────────


class EvalCaseResult(BaseModel):
    """单条评测用例的结果。"""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    scope: EvalScope
    passed: bool
    metrics: list[EvalMetric] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ─── Internal Eval Report ───────────────────────────────────────────────────


class InternalEvalReport(BaseModel):
    """
    Internal eval runner 的输出。

    支持 retrieval / reviewer / workflow 三个维度的评测。
    """

    model_config = ConfigDict(extra="forbid")

    report_id: str = Field(default_factory=lambda: f"ieval_{uuid4().hex[:12]}")
    eval_set: str
    scopes: list[EvalScope]
    summary: dict[str, Any] = Field(default_factory=dict)
    case_results: list[EvalCaseResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
