"""RAG Eval Module — 检索系统性能评测（模块 7）。"""

from src.eval.rag.models import (
    # Core case
    RagEvalCase,
    GoldPaper,
    GoldEvidence,
    GoldClaim,
    # Strategy
    RetrievalStrategy,
    RetrievalStrategyEnum,
    STRATEGIES,
    # Results
    EvalCaseResult,
    StrategyMetrics,
    StrategyComparisonReport,
    RagEvalReport,
    # Layer results
    PaperRetrievalMetrics,
    PaperRankingMetrics,
    EvidenceRetrievalMetrics,
    CitationGroundingMetrics,
)
from src.eval.rag.metrics import (
    # Core functions
    compute_all_metrics,
    compute_paper_retrieval_metrics,
    compute_paper_ranking_metrics,
    compute_evidence_retrieval_metrics,
    compute_citation_grounding_metrics,
    # Metric helpers
    ndcg,
    mrr,
    ap,
    map_k,
)
from src.eval.rag.matchers import (
    # Matchers
    loose_match,
    strict_match,
    paper_match,
    get_matcher,
    # Utilities
    token_overlap_ratio,
    text_similarity,
    normalize_section,
    section_overlap,
    SECTION_ALIASES,
)
from src.eval.rag.runner import RagEvalRunner
from src.eval.rag.report import RagEvalReporter

__all__ = [
    # Core case
    "RagEvalCase",
    "GoldPaper",
    "GoldEvidence",
    "GoldClaim",
    # Strategy
    "RetrievalStrategy",
    "RetrievalStrategyEnum",
    "STRATEGIES",
    # Results
    "EvalCaseResult",
    "StrategyMetrics",
    "StrategyComparisonReport",
    "RagEvalReport",
    # Layer results
    "PaperRetrievalMetrics",
    "PaperRankingMetrics",
    "EvidenceRetrievalMetrics",
    "CitationGroundingMetrics",
    # Metrics functions
    "compute_all_metrics",
    "compute_paper_retrieval_metrics",
    "compute_paper_ranking_metrics",
    "compute_evidence_retrieval_metrics",
    "compute_citation_grounding_metrics",
    "ndcg",
    "mrr",
    "ap",
    "map_k",
    # Matchers
    "loose_match",
    "strict_match",
    "paper_match",
    "get_matcher",
    "token_overlap_ratio",
    "text_similarity",
    "normalize_section",
    "section_overlap",
    "SECTION_ALIASES",
    # Runner
    "RagEvalRunner",
    "RagEvalReporter",
]

