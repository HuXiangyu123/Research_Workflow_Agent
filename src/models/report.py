from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Citation(BaseModel):
    label: str
    url: str
    reason: str
    source_tier: Literal["A", "B", "C", "D"] | None = None
    reachable: bool | None = None
    fetched_content: str | None = None


class ClaimSupport(BaseModel):
    claim_id: str
    citation_label: str
    support_status: Literal["supported", "partial", "unsupported", "unverifiable"]
    evidence_excerpt: str | None = None
    reason: str | None = None
    judge_confidence: float | None = None


class Claim(BaseModel):
    id: str
    text: str
    citation_labels: list[str]
    supports: list[ClaimSupport] = []
    overall_status: Literal["grounded", "partial", "ungrounded", "abstained"] = "ungrounded"


class GroundingStats(BaseModel):
    total_claims: int
    grounded: int
    partial: int
    ungrounded: int
    abstained: int
    tier_a_ratio: float
    tier_b_ratio: float


class DraftReport(BaseModel):
    sections: dict[str, str]
    claims: list[Claim]
    citations: list[Citation]


class ReportFrame(BaseModel):
    title: str
    paper_type: Literal["regular", "survey"]
    mode: Literal["draft", "full"]
    sections: dict[str, str]
    outline: dict[str, list[str]] | None = None
    claims: list[Claim]
    citations: list[Citation]


class ResolvedReport(BaseModel):
    sections: dict[str, str]
    claims: list[Claim]
    citations: list[Citation]


class VerifiedReport(BaseModel):
    sections: dict[str, str]
    claims: list[Claim]
    citations: list[Citation]


class FinalReport(BaseModel):
    sections: dict[str, str]
    claims: list[Claim]
    citations: list[Citation]
    grounding_stats: GroundingStats
    report_confidence: Literal["high", "limited", "low"]
