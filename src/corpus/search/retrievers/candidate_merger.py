"""Candidate Merger — 多路召回结果合并 + RRF 融合 + source attribution。"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from src.corpus.search.retrievers.models import (
    MergedCandidate,
    RecallEvidence,
    RetrievalPath,
    RetrievalTrace,
)

logger = logging.getLogger(__name__)


class CandidateMerger:
    """
    多路召回结果合并器。

    职责：
    1. 收集 keyword / dense 各路召回
    2. 按 doc_id 归并（同一篇论文的不同召回来源合并）
    3. RRF 融合计算综合分数
    4. 补全文档元数据（title / abstract / authors 等）
    5. 生成检索轨迹
    """

    def __init__(self, db_session):
        self._db = db_session

    def merge(
        self,
        query: str,
        keyword_evidence: list[RecallEvidence],
        dense_evidence: list[RecallEvidence],
        sub_question_id: str | None = None,
        rrf_k: int = 60,
        top_k: int = 20,
    ) -> tuple[list[MergedCandidate], list[RetrievalTrace]]:
        """
        合并 keyword + dense 召回结果。

        Args:
            query: 检索 query
            keyword_evidence: keyword 召回的 chunk evidence
            dense_evidence: dense 召回的 chunk evidence
            sub_question_id: 当前处理的子问题 ID（可选）
            rrf_k: RRF 融合参数
            top_k: 返回多少个候选

        Returns:
            (merged_candidates, traces)
        """
        traces: list[RetrievalTrace] = []
        candidates_by_doc: dict[str, MergedCandidate] = {}

        # ── Keyword 融合（RRF over keyword paths）───────────────────────────────
        kw_scores: dict[str, float] = defaultdict(float)
        for rank, ev in enumerate(keyword_evidence):
            kw_scores[ev.chunk_id] += 1.0 / (rrf_k + rank + 1)

        # ── Dense 融合（RRF over dense paths）─────────────────────────────────
        dense_scores: dict[str, float] = defaultdict(float)
        for rank, ev in enumerate(dense_evidence):
            dense_scores[ev.chunk_id] += 1.0 / (rrf_k + rank + 1)

        # ── 收集所有 chunk_id → doc_id 映射 ─────────────────────────────────
        all_chunk_ids = set(kw_scores) | set(dense_scores)
        doc_id_map = self._resolve_doc_ids(list(all_chunk_ids))

        # ── 构建 MergedCandidate ─────────────────────────────────────────────
        for chunk_id, kw_score in kw_scores.items():
            doc_id = doc_id_map.get(chunk_id, chunk_id)
            if doc_id not in candidates_by_doc:
                candidates_by_doc[doc_id] = MergedCandidate(doc_id=doc_id)

            cand = candidates_by_doc[doc_id]
            # 找到对应的 evidence
            ev = next((e for e in keyword_evidence if e.chunk_id == chunk_id), None)
            if ev:
                cand.add_evidence(ev)

            # keyword RRF score
            kw_in_cand = kw_score + cand.rrf_score
            cand.rrf_score = kw_in_cand

        for chunk_id, dense_score in dense_scores.items():
            doc_id = doc_id_map.get(chunk_id, chunk_id)
            if doc_id not in candidates_by_doc:
                candidates_by_doc[doc_id] = MergedCandidate(doc_id=doc_id)

            cand = candidates_by_doc[doc_id]
            ev = next((e for e in dense_evidence if e.chunk_id == chunk_id), None)
            if ev:
                cand.add_evidence(ev)

            # dense RRF contribution
            cand.rrf_score += dense_score

        # ── 合并同一 doc_id 的 keyword + dense ────────────────────────────────
        merged: list[MergedCandidate] = list(candidates_by_doc.values())

        # ── 全局 RRF 再融合（keyword vs dense）────────────────────────────────
        if keyword_evidence and dense_evidence:
            merged = self._cross_fuse(merged, keyword_evidence, dense_evidence, rrf_k)

        # ── 截断并按分数排序 ────────────────────────────────────────────────
        merged = sorted(merged, key=lambda c: c.rrf_score, reverse=True)[:top_k]

        # ── 补全元数据 ──────────────────────────────────────────────────────
        self._enrich_metadata(merged)

        # ── 生成 Trace ───────────────────────────────────────────────────────
        if keyword_evidence:
            traces.append(
                RetrievalTrace(
                    query=query,
                    sub_question_id=sub_question_id,
                    retrieval_path=RetrievalPath.KEYWORD_COARSE,
                    target_index="coarse",
                    filter_summary="",
                    top_k_requested=len(keyword_evidence),
                    returned_doc_ids=[e.doc_id for e in keyword_evidence],
                    returned_chunk_ids=[e.chunk_id for e in keyword_evidence],
                    returned_count=len(keyword_evidence),
                )
            )
        if dense_evidence:
            traces.append(
                RetrievalTrace(
                    query=query,
                    sub_question_id=sub_question_id,
                    retrieval_path=RetrievalPath.DENSE_COARSE,
                    target_index="coarse",
                    filter_summary="",
                    top_k_requested=len(dense_evidence),
                    returned_doc_ids=[e.doc_id for e in dense_evidence],
                    returned_chunk_ids=[e.chunk_id for e in dense_evidence],
                    returned_count=len(dense_evidence),
                )
            )

        return merged, traces

    def _resolve_doc_ids(self, chunk_ids: list[str]) -> dict[str, str]:
        """
        根据 chunk_ids 解析 doc_id 映射。

        优先从 candidates_by_doc 的 evidence 中获取，缺失时查 DB。
        """
        if not chunk_ids:
            return {}

        from src.db.models import CoarseChunk

        rows = (
            self._db.query(CoarseChunk.coarse_chunk_id, CoarseChunk.doc_id)
            .filter(CoarseChunk.coarse_chunk_id.in_(chunk_ids))
            .all()
        )
        return {row[0]: str(row[1]) for row in rows}

    def _cross_fuse(
        self,
        candidates: list[MergedCandidate],
        keyword_evidence: list[RecallEvidence],
        dense_evidence: list[RecallEvidence],
        k: int,
    ) -> list[MergedCandidate]:
        """
        全局 RRF：keyword 和 dense 各算一路，一起排名。

        keyword 和 dense 各占一路，对每篇论文：
        - 若有 keyword 命中，取其 keyword 最优排名
        - 若有 dense 命中，取其 dense 最优排名
        - 两者 RRF 分数相加
        """
        # 找每篇论文在 keyword 和 dense 中的最优排名
        kw_best: dict[str, tuple[int, float]] = {}
        for rank, ev in enumerate(keyword_evidence):
            if ev.doc_id not in kw_best or rank < kw_best[ev.doc_id][0]:
                kw_best[ev.doc_id] = (rank, ev.score)

        dense_best: dict[str, tuple[int, float]] = {}
        for rank, ev in enumerate(dense_evidence):
            if ev.doc_id not in dense_best or rank < dense_best[ev.doc_id][0]:
                dense_best[ev.doc_id] = (rank, ev.score)

        for c in candidates:
            kw_rank, kw_score = kw_best.get(c.doc_id, (None, 0.0))
            dense_rank, dense_score = dense_best.get(c.doc_id, (None, 0.0))

            c.rrf_score = 0.0
            if kw_rank is not None:
                c.rrf_score += 1.0 / (k + kw_rank + 1)
            if dense_rank is not None:
                c.rrf_score += 1.0 / (k + dense_rank + 1)

            c.keyword_score = kw_score
            c.dense_score = dense_score

        return candidates

    def _enrich_metadata(self, candidates: list[MergedCandidate]) -> None:
        """补全文档元数据（title / abstract / authors / year 等）。"""
        if not candidates:
            return

        doc_ids = list({c.doc_id for c in candidates})
        from src.db.models import Document

        rows = (
            self._db.query(Document)
            .filter(Document.doc_id.in_(doc_ids))
            .all()
        )
        meta_map: dict[str, dict] = {}
        for row in rows:
            meta_map[row.doc_id] = {
                "title": row.title or "",
                "abstract": getattr(row, "summary", None) or "",
                "authors": getattr(row, "authors", None) or "",
                "year": self._parse_year(getattr(row, "published_date", None)),
                "venue": getattr(row, "venue", None),
                "canonical_id": getattr(row, "canonical_id", None) or "",
                "source_uri": getattr(row, "source_uri", "") or "",
                "source_type": getattr(row, "source_type", "") or "",
            }

        for c in candidates:
            m = meta_map.get(c.doc_id, {})
            c.title = m.get("title", "")
            c.abstract = m.get("abstract")
            authors_str = m.get("authors", "")
            c.authors = (
                [a.strip() for a in authors_str.split(",") if a.strip()]
                if authors_str
                else []
            )
            c.year = m.get("year")
            c.venue = m.get("venue")
            c.canonical_id = m.get("canonical_id", "") or c.canonical_id
            c.source_uri = m.get("source_uri", "")
            c.source_type = m.get("source_type", "")

    @staticmethod
    def _parse_year(date_str) -> int | None:
        if not date_str:
            return None
        s = str(date_str)
        if len(s) >= 4 and s[:4].isdigit():
            return int(s[:4])
        return None
