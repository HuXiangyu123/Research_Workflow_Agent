"""Review models — Phase 3: ReviewFeedback, CoverageGap, ClaimSupport, RevisionAction."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


# ─── Severity & Category ────────────────────────────────────────────────────


class ReviewSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    BLOCKER = "blocker"


class ReviewCategory(str, Enum):
    COVERAGE_GAP = "coverage_gap"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    CITATION_REACHABILITY = "citation_reachability"
    DUPLICATION = "duplication"
    CONSISTENCY = "consistency"


# ─── Coverage Gap ─────────────────────────────────────────────────────────────


class CoverageGap(BaseModel):
    """指出某个 sub-question 或 topic 没有被充分覆盖。"""

    model_config = ConfigDict(extra="forbid")

    sub_question_id: str | None = None
    missing_topics: list[str] = Field(default_factory=list)
    missing_papers: list[str] = Field(default_factory=list)
    note: str | None = None


# ─── Claim Support ────────────────────────────────────────────────────────────


class ClaimSupport(BaseModel):
    """评估单个 claim 是否有 evidence 支撑。"""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim_text: str
    supported: bool
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    note: str | None = None


# ─── Revision Action ──────────────────────────────────────────────────────────


class RevisionActionType(str, Enum):
    RESEARCH_MORE = "research_more"
    REWRITE_SECTION = "rewrite_section"
    FIX_CITATION = "fix_citation"
    DROP_CLAIM = "drop_claim"
    MERGE_DUPLICATE = "merge_duplicate"
    REVISE_DRAFT = "revise_draft"  # 补充：重新生成草稿


class RevisionAction(BaseModel):
    """由 ReviewFeedback 触发的具体修改动作。"""

    model_config = ConfigDict(extra="forbid")

    action_type: RevisionActionType
    target: str
    reason: str
    priority: int = Field(default=1, ge=1, le=3)


# ─── Review Issue ────────────────────────────────────────────────────────────


class ReviewIssue(BaseModel):
    """ReviewFeedback 中的单个问题。"""

    model_config = ConfigDict(extra="forbid")

    issue_id: str = Field(default_factory=lambda: f"issue_{uuid4().hex[:10]}")
    severity: ReviewSeverity
    category: ReviewCategory
    target: str
    summary: str
    evidence_refs: list[str] = Field(default_factory=list)


# ─── Review Feedback ────────────────────────────────────────────────────────


class ReviewFeedback(BaseModel):
    """
    Reviewer 的顶层输出。

    职责：检查 coverage gap、unsupported claims、citation reachability、
    duplication/consistency，并在必要时触发二轮检索或局部修订。
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "v1"
    review_id: str = Field(default_factory=lambda: f"review_{uuid4().hex[:12]}")
    task_id: str
    workspace_id: str
    passed: bool
    issues: list[ReviewIssue] = Field(default_factory=list)
    coverage_gaps: list[CoverageGap] = Field(default_factory=list)
    claim_supports: list[ClaimSupport] = Field(default_factory=list)
    revision_actions: list[RevisionAction] = Field(default_factory=list)
    summary: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
