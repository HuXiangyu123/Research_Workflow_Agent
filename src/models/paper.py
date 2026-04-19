from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RerankLog:
    """Rerank 日志。"""
    stage: str = ""   # "paper_rerank" / "chunk_rerank"
    model: str = ""
    candidates_count: int = 0
    top_k: int = 0


@dataclass
class RagResult:
    """
    结构化 RAG 检索结果（模块 6 输出）。

    替代字符串拼接，作为后续 workflow 的正式 artifact。
    原始版本见 corpus/store/repository.py 的 EvidenceSearchResult。
    """
    # Legacy chunk-level fields used by the report graph evidence bundle.
    text: str = ""
    doc_id: str = ""
    score: float = 0.0

    # 检索上下文
    query: str = ""
    sub_questions: list[str] = field(default_factory=list)
    rag_strategy: str = "keyword+dense+rrf"  # 记录本次检索策略

    # 检索对象
    paper_candidates: list = field(default_factory=list)   # PaperCandidate[]
    evidence_chunks: list = field(default_factory=list)    # EvidenceChunk[]

    # 检索轨迹
    retrieval_trace: list = field(default_factory=list)    # RetrievalTrace[]
    dedup_log: list = field(default_factory=list)          # DedupInfo[]
    rerank_log: list = field(default_factory=list)        # RerankLog[]

    # 覆盖度注释
    coverage_notes: list[str] = field(default_factory=list)
    total_papers: int = 0
    total_chunks: int = 0

    # 时间戳
    retrieved_at: str = ""  # ISO format


# ── Legacy compatibility (for existing code that expects pydantic models) ─────
# 以下为向后兼容保留，后续逐步迁移


@dataclass
class PaperMetadata:
    """论文元数据。"""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    pdf_url: str | None = None
    published: str | None = None


@dataclass
class WebResult:
    """Web 检索结果。"""
    url: str = ""
    text: str = ""
    status_code: int = 0


@dataclass
class EvidenceBundle:
    """Evidence 检索结果包。"""
    rag_results: list[RagResult] = field(default_factory=list)
    web_results: list[WebResult] = field(default_factory=list)


@dataclass
class NormalizedDocument:
    """规范化后的论文文档。"""
    metadata: PaperMetadata = field(default_factory=PaperMetadata)
    document_text: str = ""
    document_sections: dict[str, str] = field(default_factory=dict)
    source_manifest: dict = field(default_factory=dict)
