"""Search 模块数据模型 — 模块 4 核心数据结构。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Retrieval Path
# ---------------------------------------------------------------------------


class RetrievalPath(str, Enum):
    """召回路径。"""

    KEYWORD_TITLE = "keyword_title"       # BM25 on title
    KEYWORD_ABSTRACT = "keyword_abstract"  # BM25 on abstract
    KEYWORD_COARSE = "keyword_coarse"     # BM25 on coarse chunk text
    DENSE_COARSE = "dense_coarse"         # Dense retrieval on coarse chunks
    DENSE_TITLE = "dense_title"           # Dense retrieval on title embedding
    METADATA_FILTER = "metadata_filter"   # Metadata filter only (no vector/text)


# ---------------------------------------------------------------------------
# MatchedQuery — 子问题命中信息
# ---------------------------------------------------------------------------


@dataclass
class MatchedQuery:
    """一条 query 命中的信息。"""

    query_text: str
    path: RetrievalPath
    rank: int          # 在该路径的排名
    score: float
    is_main_query: bool = False   # 是否是主 query（而非子问题）
    sub_question_id: Optional[str] = None


# ---------------------------------------------------------------------------
# RecallEvidence — 单条召回证据
# ---------------------------------------------------------------------------


@dataclass
class RecallEvidence:
    """来自单路召回的具体证据（chunk 级别）。"""

    chunk_id: str
    doc_id: str
    canonical_id: str
    section: str
    text: str
    score: float
    path: RetrievalPath
    page_start: int = 1
    page_end: int = 1
    token_count: int = 0


# ---------------------------------------------------------------------------
# MergedCandidate — 单篇合并后候选
# ---------------------------------------------------------------------------


@dataclass
class MergedCandidate:
    """
    单篇论文的合并候选。

    来自多路召回合并后、去重前的高召回候选。
    """

    # 论文标识
    doc_id: str
    canonical_id: Optional[str] = None

    # 召回证据（保留来源，用于后续解释）
    matched_queries: list[MatchedQuery] = field(default_factory=list)
    recall_evidence: list[RecallEvidence] = field(default_factory=list)

    # 原始分数（RRF 等融合后）
    rrf_score: float = 0.0
    keyword_score: float = 0.0
    dense_score: float = 0.0

    # 命中的 sub-questions（用于 reviewer 判断覆盖完整性）
    matched_sub_question_ids: list[str] = field(default_factory=list)

    # 文档元数据（补全用）
    title: str = ""
    abstract: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    source_uri: str = ""

    @property
    def is_from_keyword(self) -> bool:
        return any(m.path in (
            RetrievalPath.KEYWORD_TITLE,
            RetrievalPath.KEYWORD_ABSTRACT,
            RetrievalPath.KEYWORD_COARSE,
        ) for m in self.matched_queries)

    @property
    def is_from_dense(self) -> bool:
        return any(m.path in (
            RetrievalPath.DENSE_COARSE,
            RetrievalPath.DENSE_TITLE,
        ) for m in self.matched_queries)

    @property
    def recall_paths(self) -> list[str]:
        return list(set(m.path.value for m in self.matched_queries))

    def add_evidence(self, evidence: RecallEvidence) -> None:
        self.recall_evidence.append(evidence)
        dense_paths = {RetrievalPath.DENSE_COARSE, RetrievalPath.DENSE_TITLE}
        kw_paths = {
            RetrievalPath.KEYWORD_TITLE,
            RetrievalPath.KEYWORD_ABSTRACT,
            RetrievalPath.KEYWORD_COARSE,
        }
        self.dense_score = max(
            self.dense_score,
            max((e.score for e in self.recall_evidence if e.path in dense_paths), default=0.0),
        )
        self.keyword_score = max(
            self.keyword_score,
            max((e.score for e in self.recall_evidence if e.path in kw_paths), default=0.0),
        )


# ---------------------------------------------------------------------------
# RetrievalTrace — 单次检索轨迹
# ---------------------------------------------------------------------------


@dataclass
class RetrievalTrace:
    """
    一次检索的完整轨迹。

    记录每条候选是怎么被召回来的，供模块 7 (eval) 追溯根因。
    """

    query: str
    sub_question_id: Optional[str] = None
    retrieval_path: RetrievalPath = RetrievalPath.KEYWORD_COARSE
    target_index: str = "coarse"       # coarse / title / abstract
    filter_summary: str = ""            # 过滤条件摘要
    top_k_requested: int = 0
    returned_doc_ids: list[str] = field(default_factory=list)
    returned_chunk_ids: list[str] = field(default_factory=list)
    returned_count: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# InitialPaperCandidates — 模块 4 最终输出
# ---------------------------------------------------------------------------


@dataclass
class InitialPaperCandidates:
    """
    模块 4 的最终产物：高召回论文候选池。

    包含：
    - 多路召回的粗候选（可能有重复）
    - 每条候选的来源记录
    - 完整检索轨迹
    """

    query: str
    sub_question_id: Optional[str] = None

    # 候选论文列表
    candidates: list[MergedCandidate] = field(default_factory=list)

    # 检索轨迹
    traces: list[RetrievalTrace] = field(default_factory=list)

    # 统计
    total_candidates: int = 0
    keyword_candidates: int = 0
    dense_candidates: int = 0
    both_candidates: int = 0        # 两路都命中的
    sub_question_coverage: int = 0   # 覆盖了多少个子问题

    timestamp: float = field(default_factory=time.time)

    def build_summary(self) -> dict:
        """生成统计摘要。"""
        self.keyword_candidates = sum(1 for c in self.candidates if c.is_from_keyword)
        self.dense_candidates = sum(1 for c in self.candidates if c.is_from_dense)
        self.both_candidates = sum(1 for c in self.candidates if c.is_from_keyword and c.is_from_dense)
        self.sub_question_coverage = len(set(
            sq_id for c in self.candidates for sq_id in c.matched_sub_question_ids
        ))
        self.total_candidates = len(self.candidates)
        return {
            "total": self.total_candidates,
            "keyword_only": self.keyword_candidates - self.both_candidates,
            "dense_only": self.dense_candidates - self.both_candidates,
            "both_channels": self.both_candidates,
            "sub_question_coverage": self.sub_question_coverage,
        }

    def top_by_rrf(self, top_k: int = 20) -> list[MergedCandidate]:
        """按 RRF 分数取 top-K。"""
        return sorted(self.candidates, key=lambda c: c.rrf_score, reverse=True)[:top_k]
