"""Corpus Search 模块 — Paper-level Retrieval（模块 4 + 模块 5）。"""

from src.corpus.search.deduper import DedupedCandidate, DedupInfo, PaperDeduper
from src.corpus.search.candidate_builder import CandidateBuilder, PaperCandidate, ScoreBreakdown
from src.corpus.search.retrievers.models import (
    InitialPaperCandidates,
    MergedCandidate,
    RetrievalPath,
    RetrievalTrace,
    RecallEvidence,
    MatchedQuery,
)
from src.corpus.search.retrievers.paper_retriever import PaperRetriever
from src.corpus.search.retrievers.chunk_retriever import ChunkRetriever
from src.corpus.search.reranker import CrossEncoderReranker, RerankResult

from src.corpus.search.models import EvidenceChunk, ScoreBreakdown
from src.corpus.search.evidence_typer import EvidenceTyper
from src.corpus.search.result_builder import RagResultBuilder

__all__ = [
    # Models
    "EvidenceChunk",
    "ScoreBreakdown",
    # Retrievers
    "PaperRetriever",
    "ChunkRetriever",
    # Module 5
    "InitialPaperCandidates",
    "MergedCandidate",
    "RetrievalPath",
    "RetrievalTrace",
    "RecallEvidence",
    "MatchedQuery",
    "PaperDeduper",
    "DedupedCandidate",
    "DedupInfo",
    "CandidateBuilder",
    "PaperCandidate",
    "CrossEncoderReranker",
    "RerankResult",
    # Module 6
    "EvidenceTyper",
    "RagResultBuilder",
]
