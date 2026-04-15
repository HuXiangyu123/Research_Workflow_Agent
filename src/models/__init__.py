"""Domain models for the literature report agent."""

from src.models.paper import (
    EvidenceBundle,
    NormalizedDocument,
    PaperMetadata,
    RagResult,
    WebResult,
)
from src.models.report import (
    Citation,
    Claim,
    ClaimSupport,
    DraftReport,
    FinalReport,
    GroundingStats,
    ResolvedReport,
    VerifiedReport,
)
from src.models.task import TaskStatus
from src.research.research_brief import (
    AmbiguityItem,
    ClarifyInput,
    ClarifyResult,
    ResearchBrief,
)

__all__ = [
    "AmbiguityItem",
    "Citation",
    "Claim",
    "ClaimSupport",
    "ClarifyInput",
    "ClarifyResult",
    "DraftReport",
    "EvidenceBundle",
    "FinalReport",
    "GroundingStats",
    "NormalizedDocument",
    "PaperMetadata",
    "RagResult",
    "ResearchBrief",
    "ResolvedReport",
    "TaskStatus",
    "VerifiedReport",
    "WebResult",
]
