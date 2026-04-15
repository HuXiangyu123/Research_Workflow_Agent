"""Keyword Index — PostgreSQL BM25 全文检索（支持 title / abstract / chunk text）。"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.corpus.models import CoarseChunk, FineChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search Result
# ---------------------------------------------------------------------------


@dataclass
class KeywordSearchResult:
    """关键词检索结果。"""

    id: str
    text: str
    score: float          # BM25 rank score
    doc_id: str
    canonical_id: str
    section: str
    page_start: int = 1
    page_end: int = 1
    metadata: dict | None = None


# ---------------------------------------------------------------------------
# Base Keyword Index
# ---------------------------------------------------------------------------


class KeywordIndex(ABC):
    """关键词/BM25 全文检索抽象基类。"""

    @abstractmethod
    def search(
        self,
        query: str,
        fields: list[str] | None = None,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[KeywordSearchResult]:
        """
        关键词/BM25 检索。

        Args:
            query: 搜索词
            fields: 检索字段（默认 ["text"]）
            top_k: 返回数量
            filters: 过滤条件（doc_id / canonical_id / section 等）

        Returns:
            KeywordSearchResult 列表
        """
        ...

    @abstractmethod
    def index_chunk(self, chunk: Any) -> None:
        """索引单个 chunk。"""
        ...

    @abstractmethod
    def index_chunks(self, chunks: list[Any]) -> int:
        """批量索引 chunks。"""
        ...


# ---------------------------------------------------------------------------
# PostgreSQL BM25 Keyword Index
# ---------------------------------------------------------------------------


class PGKeywordIndex(KeywordIndex):
    """
    基于 PostgreSQL ts_rank 的 BM25 全文检索。

    复用了现有的 HybridSearcher.search_bm25() 逻辑。
    后续可替换为 Elasticsearch / OpenSearch 实现。

    支持检索字段：
    - text（chunk 正文）
    - title（通过 join document）
    - abstract（通过 join document）
    """

    def __init__(self, db_session=None):
        """
        Args:
            db_session: 可选的 SQLAlchemy session
        """
        self._session = db_session

    def set_session(self, session) -> None:
        """设置 DB session（可后续注入）。"""
        self._session = session

    def search(
        self,
        query: str,
        fields: list[str] | None = None,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[KeywordSearchResult]:
        """
        PostgreSQL BM25 检索。

        支持 filter：
        - doc_id: 限定文档
        - canonical_id: 限定论文
        - section: 限定章节
        """
        if not self._session:
            logger.warning("[PGKeywordIndex] 无 DB session，无法检索")
            return []

        from sqlalchemy import func, text
        from src.db.models import Chunk, Document

        fields = fields or ["text"]

        try:
            # 构建 tsvector 检索
            base_query = (
                self._session.query(
                    Chunk.chunk_id,
                    Chunk.text,
                    Chunk.doc_id,
                    Chunk.page_start,
                    Chunk.page_end,
                    Document.title,
                    Document.source_uri,
                    func.ts_rank(
                        func.to_tsvector("english", Chunk.text),
                        func.plainto_tsquery("english", query),
                    ).label("score"),
                )
                .join(Document, Chunk.doc_id == Document.doc_id)
                .filter(
                    func.to_tsvector("english", Chunk.text).op("@@")(
                        func.plainto_tsquery("english", query)
                    )
                )
            )

            # 应用 filters
            if filters:
                if "doc_id" in filters:
                    base_query = base_query.filter(Chunk.doc_id == filters["doc_id"])
                if "canonical_id" in filters:
                    base_query = base_query.filter(Document.canonical_id == filters["canonical_id"])

            base_query = base_query.order_by(text("score DESC")).limit(top_k)
            rows = base_query.all()

            results = []
            for row in rows:
                results.append(
                    KeywordSearchResult(
                        id=str(row[0]),
                        text=row[1][:500],  # 截断
                        score=float(row[6]) if len(row) > 6 else 0.0,
                        doc_id=str(row[2]) if row[2] else "",
                        canonical_id="",
                        section="",
                        page_start=int(row[3]) if row[3] else 1,
                        page_end=int(row[4]) if row[4] else 1,
                    )
                )
            return results

        except Exception as e:
            logger.error(f"[PGKeywordIndex] search 失败：{e}")
            return []

    def index_chunk(self, chunk: Any) -> None:
        """索引单个 chunk（写入 PostgreSQL chunks 表）。"""
        if not self._session:
            return

        from src.db.models import Chunk

        orm = Chunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            section_path=[chunk.section] if hasattr(chunk, "section") else [],
            page_start=getattr(chunk, "page_start", 1),
            page_end=getattr(chunk, "page_end", 1),
            char_start=getattr(chunk, "char_start", 0),
            char_end=getattr(chunk, "char_end", 0),
            text=chunk.text,
            text_hash=self._text_hash(chunk.text),
            len_chars=len(chunk.text),
        )
        self._session.merge(orm)
        self._session.flush()

    def index_chunks(self, chunks: list[Any]) -> int:
        """批量索引 chunks。"""
        for chunk in chunks:
            self.index_chunk(chunk)
        return len(chunks)

    @staticmethod
    def _text_hash(text: str) -> str:
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:24]
