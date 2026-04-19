from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

from src.models.paper import EvidenceBundle, NormalizedDocument
from src.models.report import DraftReport, FinalReport, ReportFrame, ResolvedReport, VerifiedReport


class NodeStatus(BaseModel):
    node: str
    status: Literal["pending", "running", "done", "limited", "failed", "skipped"]
    started_at: str | None = None
    ended_at: str | None = None
    duration_ms: int | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    tokens_delta: int = 0
    repair_triggered: bool = False


class AgentState(TypedDict):
    task_id: str
    workspace_id: str
    raw_input: str
    source_type: Literal["arxiv", "pdf", "research"]
    report_mode: Literal["draft", "full"]
    research_depth: Literal["plan", "full"]
    interaction_mode: Literal["interactive", "non_interactive"]
    paper_type: Literal["regular", "survey"] | None
    brief: dict[str, Any] | None
    search_plan: dict[str, Any] | None
    search_plan_warnings: list[str]
    rag_result: dict[str, Any] | None
    paper_cards: list[dict[str, Any]]
    compression_result: dict[str, Any] | None
    taxonomy: dict[str, Any] | None
    review_feedback: dict[str, Any] | None
    review_passed: bool | None
    artifacts_created: list[dict[str, Any]]
    artifact_count: int
    current_stage: str | None
    arxiv_id: str | None
    pdf_text: str | None
    source_manifest: dict | None
    normalized_doc: NormalizedDocument | None
    evidence: EvidenceBundle | None
    report_frame: ReportFrame | None
    draft_report: DraftReport | None
    resolved_report: ResolvedReport | None
    verified_report: VerifiedReport | None
    final_report: FinalReport | None
    draft_markdown: str | None
    full_markdown: str | None
    followup_hints: list[str]
    awaiting_followup: bool
    followup_resolution: dict[str, Any] | None
    tokens_used: Annotated[int, operator.add]
    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    degradation_mode: Literal["normal", "limited", "safe_abort"]
    node_statuses: dict[str, NodeStatus]
