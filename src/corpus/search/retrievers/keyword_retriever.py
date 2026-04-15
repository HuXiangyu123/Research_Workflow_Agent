"""Keyword Retriever — BM25 关键词召回（title / abstract / coarse chunks）。"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

from src.corpus.search.retrievers.models import (
    RecallEvidence,
    RetrievalPath,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class KeywordRetriever:
    """
    Paper-level 关键词检索器。

    召回对象：
    - title（论文标题）
    - abstract（摘要）
    - coarse_chunk.text（粗粒度块正文）

    适合抓取：模型名 / benchmark 名 / task name / author / venue / year /
    query 中的强术语。
    """

    def __init__(self, db_session: "Session"):
        """
        Args:
            db_session: SQLAlchemy session（PostgreSQL）
        """
        self._db = db_session

    # ── 主入口 ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        path: RetrievalPath = RetrievalPath.KEYWORD_COARSE,
        top_k: int = 30,
        doc_ids: list[str] | None = None,
        filters: Optional[dict] = None,
    ) -> list[RecallEvidence]:
        """
        关键词检索。

        Args:
            query: 检索词
            path: 召回路径（决定在哪个字段检索）
            top_k: 返回数量
            doc_ids: 可选，限定特定 doc_ids
            filters: 可选，SQL filter 字典（year_range, source_type）

        Returns:
            RecallEvidence 列表
        """
        start = time.time()

        try:
            if path == RetrievalPath.KEYWORD_TITLE:
                return self._search_title(query, top_k, doc_ids, filters)
            elif path == RetrievalPath.KEYWORD_ABSTRACT:
                return self._search_abstract(query, top_k, doc_ids, filters)
            elif path == RetrievalPath.KEYWORD_COARSE:
                return self._search_coarse(query, top_k, doc_ids, filters)
            else:
                logger.warning(f"[KeywordRetriever] 不支持的 path: {path}")
                return []
        except Exception as e:
            logger.error(f"[KeywordRetriever] search 失败（path={path}）：{e}")
            return []

    # ── 过滤器辅助方法 ───────────────────────────────────────────────────────────

    def _apply_document_filters(
        self,
        q,
        filters: Optional[dict],
        doc_model,
    ):
        """对 Document 字段应用 filters。"""
        from sqlalchemy import Integer, func

        if filters is None:
            return q
        if filters.get("year_range"):
            y_min, y_max = filters["year_range"]
            if y_min is not None:
                q = q.filter(
                    func.substring(doc_model.published_date, 1, 4)
                    .cast(Integer) >= y_min
                )
            if y_max is not None:
                q = q.filter(
                    func.substring(doc_model.published_date, 1, 4)
                    .cast(Integer) <= y_max
                )
        if filters.get("source_type"):
            q = q.filter(doc_model.source_type.in_(filters["source_type"]))
        return q

    # ── Title 检索 ────────────────────────────────────────────────────────────

    def _search_title(
        self,
        query: str,
        top_k: int,
        doc_ids: list[str] | None = None,
        filters: Optional[dict] = None,
    ) -> list[RecallEvidence]:
        """在论文标题上做 BM25 检索。"""
        from sqlalchemy import func, text
        from src.db.models import Document

        q = self._db.query(
            Document.doc_id,
            Document.canonical_id,
            Document.title,
            Document.source_uri,
            Document.published_date,
            Document.source_type,
            func.ts_rank(
                func.to_tsvector("english", Document.title),
                func.plainto_tsquery("english", query),
            ).label("score"),
        ).filter(
            func.to_tsvector("english", Document.title).op("@@")(
                func.plainto_tsquery("english", query)
            )
        )

        q = self._apply_document_filters(q, filters, Document)

        if doc_ids:
            q = q.filter(Document.doc_id.in_(doc_ids))

        rows = q.order_by(text("score DESC")).limit(top_k).all()
        return [
            RecallEvidence(
                chunk_id=row[0],          # 用 doc_id 作为 chunk_id
                doc_id=str(row[0]),
                canonical_id=str(row[1] or ""),
                section="title",
                text=str(row[2] or "")[:500],
                score=float(row[6]) if len(row) > 6 else 0.0,
                path=RetrievalPath.KEYWORD_TITLE,
            )
            for row in rows
        ]

    # ── Abstract 检索 ────────────────────────────────────────────────────────

    def _search_abstract(
        self,
        query: str,
        top_k: int,
        doc_ids: list[str] | None = None,
        filters: Optional[dict] = None,
    ) -> list[RecallEvidence]:
        """在摘要字段上做 BM25 检索。"""
        from sqlalchemy import func, text
        from src.db.models import Document

        q = self._db.query(
            Document.doc_id,
            Document.canonical_id,
            Document.title,
            Document.summary,
            Document.source_uri,
            Document.published_date,
            Document.source_type,
            func.ts_rank(
                func.to_tsvector("english", Document.summary),
                func.plainto_tsquery("english", query),
            ).label("score"),
        ).filter(
            Document.summary.isnot(None),
            func.to_tsvector("english", Document.summary).op("@@")(
                func.plainto_tsquery("english", query)
            ),
        )

        q = self._apply_document_filters(q, filters, Document)

        if doc_ids:
            q = q.filter(Document.doc_id.in_(doc_ids))

        rows = q.order_by(text("score DESC")).limit(top_k).all()
        return [
            RecallEvidence(
                chunk_id=row[0],
                doc_id=str(row[0]),
                canonical_id=str(row[1] or ""),
                section="abstract",
                text=str(row[3] or "")[:500],
                score=float(row[7]) if len(row) > 7 else 0.0,
                path=RetrievalPath.KEYWORD_ABSTRACT,
            )
            for row in rows
        ]

    # ── Coarse Chunk 检索 ───────────────────────────────────────────────────

    def _search_coarse(
        self,
        query: str,
        top_k: int,
        doc_ids: list[str] | None = None,
        filters: Optional[dict] = None,
    ) -> list[RecallEvidence]:
        """在粗粒度 chunk text 上做 BM25 检索。"""
        from sqlalchemy import func, text
        from src.db.models import CoarseChunk, Document

        q = self._db.query(
            CoarseChunk.coarse_chunk_id,
            CoarseChunk.doc_id,
            CoarseChunk.canonical_id,
            CoarseChunk.section,
            CoarseChunk.text,
            CoarseChunk.page_start,
            CoarseChunk.page_end,
            CoarseChunk.token_count,
            Document.title,
            Document.source_uri,
            Document.published_date,
            Document.source_type,
            func.ts_rank(
                func.to_tsvector("english", CoarseChunk.text),
                func.plainto_tsquery("english", query),
            ).label("score"),
        ).join(Document, CoarseChunk.doc_id == Document.doc_id).filter(
            func.to_tsvector("english", CoarseChunk.text).op("@@")(
                func.plainto_tsquery("english", query)
            )
        )

        q = self._apply_document_filters(q, filters, Document)

        if doc_ids:
            q = q.filter(CoarseChunk.doc_id.in_(doc_ids))

        rows = q.order_by(text("score DESC")).limit(top_k).all()
        return [
            RecallEvidence(
                chunk_id=str(row[0]),
                doc_id=str(row[1]),
                canonical_id=str(row[2] or ""),
                section=str(row[3] or "unknown"),
                text=str(row[4] or "")[:500],
                score=float(row[12]) if len(row) > 12 else 0.0,
                path=RetrievalPath.KEYWORD_COARSE,
                page_start=int(row[5]) if row[5] else 1,
                page_end=int(row[6]) if row[6] else 1,
                token_count=int(row[7]) if row[7] else 0,
            )
            for row in rows
        ]

    # ── 多路径并行召回 ────────────────────────────────────────────────────────

    def search_all_paths(
        self,
        query: str,
        top_k: int = 20,
        doc_ids: list[str] | None = None,
        filters: Optional[dict] = None,
    ) -> dict[RetrievalPath, list[RecallEvidence]]:
        """
        并行在 title / abstract / coarse 上召回。

        Args:
            query: 检索词
            top_k: 返回数量
            doc_ids: 可选，限定特定 doc_ids
            filters: 可选，SQL filter 字典（year_range, source_type）

        Returns:
            {path: [RecallEvidence, ...]}
        """
        title_results = self._search_title(query, top_k, doc_ids, filters)
        abstract_results = self._search_abstract(query, top_k, doc_ids, filters)
        coarse_results = self._search_coarse(query, top_k, doc_ids, filters)

        return {
            RetrievalPath.KEYWORD_TITLE: title_results,
            RetrievalPath.KEYWORD_ABSTRACT: abstract_results,
            RetrievalPath.KEYWORD_COARSE: coarse_results,
        }
