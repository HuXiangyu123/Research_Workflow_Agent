"""Module 6 Data Models — EvidenceChunk and supporting types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScoreBreakdown:
    """Evidence chunk 分数明细。"""
    keyword_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0


@dataclass
class EvidenceChunk:
    """
    Evidence 检索结果（模块 6 输出）。

    表示在已选论文内部检索到的具体证据块。
    """
    chunk_id: str = ""
    paper_id: str = ""               # doc_id
    canonical_id: str = ""

    # 内容
    text: str = ""
    section: str = ""
    page_start: int = 1
    page_end: int = 1

    # 检索分数
    scores: ScoreBreakdown = field(default_factory=ScoreBreakdown)

    # Evidence typing
    support_type: str = "claim_support"  # method/result/background/limitation/claim_support

    # 来源追踪
    matched_query: str = ""
    sub_question_id: str = ""
    chunk_path: str = "keyword"  # keyword / dense / hybrid

    # 兼容属性（forward-compatible）
    keyword_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0

    def __post_init__(self):
        # 同步 scores 和顶层属性
        if self.scores:
            self.keyword_score = self.scores.keyword_score
            self.dense_score = self.scores.dense_score
            self.rrf_score = self.scores.rrf_score
        else:
            self.scores = ScoreBreakdown(
                keyword_score=self.keyword_score,
                dense_score=self.dense_score,
                rrf_score=self.rrf_score,
            )