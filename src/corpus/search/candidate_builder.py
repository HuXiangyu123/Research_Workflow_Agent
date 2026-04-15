"""Candidate Builder — 构建最终 Top-K PaperCandidate。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── ScoreBreakdown ─────────────────────────────────────────────────────────────


@dataclass
class ScoreBreakdown:
    """分数明细。"""
    rrf_score: float = 0.0
    keyword_score: float = 0.0
    dense_score: float = 0.0
    rerank_score: Optional[float] = None
    final_score: float = 0.0


# ── PaperCandidate ─────────────────────────────────────────────────────────────


@dataclass
class PaperCandidate:
    """
    最终论文候选（模块 5 输出）。

    这是系统内用于后续 workflow 的标准论文候选格式。
    """
    paper_id: str = ""
    canonical_id: str = ""

    # 元数据
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    abstract: Optional[str] = None

    # 来源
    source_refs: list[str] = field(default_factory=list)
    primary_doc_id: str = ""

    # 分数
    scores: ScoreBreakdown = field(default_factory=ScoreBreakdown)

    # 追踪
    matched_queries: list[str] = field(default_factory=list)
    matched_paths: list[str] = field(default_factory=list)
    why_retrieved: str = ""


# ── CandidateBuilder ───────────────────────────────────────────────────────────


class CandidateBuilder:
    """将 DedupedCandidate 转换为 PaperCandidate。"""

    def build(
        self,
        deduped: list,
        top_k: int = 20,
    ) -> list[PaperCandidate]:
        """
        构建最终 Top-K PaperCandidate。

        Args:
            deduped: 去重后的候选列表
            top_k: 返回数量

        Returns:
            Top-K PaperCandidate 列表
        """
        # 按 final_score 降序排列（没有 final_score 则用 rrf_score）
        sorted_candidates = sorted(
            deduped,
            key=lambda c: getattr(c, "final_score", None) or c.rrf_score,
            reverse=True,
        )[:top_k]

        return [self._to_paper_candidate(c) for c in sorted_candidates]

    def _to_paper_candidate(self, c) -> PaperCandidate:
        """将 DedupedCandidate 转换为 PaperCandidate。"""
        final = getattr(c, "final_score", None) or c.rrf_score
        rerank_score = getattr(c, "rerank_score", None)

        return PaperCandidate(
            paper_id=c.primary_doc_id,
            canonical_id=c.canonical_id,
            title=c.title or "",
            authors=c.authors or [],
            year=c.year,
            venue=c.venue,
            abstract=c.abstract,
            source_refs=getattr(c, "dedup_info", None) and c.dedup_info.source_refs or [],
            primary_doc_id=c.primary_doc_id,
            scores=ScoreBreakdown(
                rrf_score=c.rrf_score,
                keyword_score=getattr(c, "keyword_score", 0.0) or 0.0,
                dense_score=getattr(c, "dense_score", 0.0) or 0.0,
                rerank_score=rerank_score,
                final_score=final,
            ),
            matched_queries=[
                getattr(mq, "query_text", "") or ""
                for mq in getattr(c, "matched_queries", [])
            ],
            matched_paths=[
                getattr(p, "value", str(p)) for p in getattr(c, "matched_paths", [])
            ],
            why_retrieved=self._build_why(c),
        )

    def _build_why(self, c) -> str:
        """生成 why_retrieved 描述。"""
        paths = [
            getattr(p, "value", str(p))
            for p in getattr(c, "matched_paths", [])
        ]
        path_str = ", ".join(paths) if paths else "unknown"
        rrf = c.rrf_score
        rerank = getattr(c, "rerank_score", None)
        if rerank is not None:
            return f"Matched via {path_str} (RRF={rrf:.3f}, Rerank={rerank:.3f})"
        return f"Matched via {path_str} (RRF={rrf:.3f})"
