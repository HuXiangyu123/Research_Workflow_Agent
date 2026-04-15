"""Chunk Store — PostgreSQL 存储 CoarseChunk / FineChunk 及父子关系。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.corpus.models import CoarseChunk, FineChunk

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class ChunkStore:
    """
    Chunk 持久化存储（PostgreSQL）。

    职责：
    - upsert_coarse_chunks：批量写入/更新 CoarseChunk
    - upsert_fine_chunks：批量写入/更新 FineChunk
    - get_coarse_chunks_by_doc：获取文档的所有 coarse chunks
    - get_fine_chunks_by_coarse：获取 coarse chunk 的所有 fine 子块
    - get_fine_chunks_by_doc：获取文档的所有 fine chunks
    - delete_by_doc：删除文档的所有 chunks
    """

    def __init__(self, db_session: "Session"):
        self._db = db_session

    def upsert_coarse(self, chunks: list[CoarseChunk]) -> list[str]:
        """批量 upsert CoarseChunk，返回 chunk_id 列表。"""
        from src.db.models import CoarseChunk as ORMCoarseChunk

        ids: list[str] = []
        for c in chunks:
            orm = ORMCoarseChunk(
                coarse_chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                canonical_id=c.canonical_id or "",
                section=c.section,
                section_level=c.section_level,
                page_start=c.page_start,
                page_end=c.page_end,
                char_start=c.char_start,
                char_end=c.char_end,
                text=c.text,
                text_hash=self._text_hash(c.text),
                token_count=c.token_count,
                order_idx=c.order,
                meta_info=c.meta_info,
            )
            self._db.merge(orm)
            ids.append(c.chunk_id)

        self._db.flush()
        logger.debug(f"[ChunkStore] upserted {len(ids)} coarse chunks")
        return ids

    def upsert_fine(self, chunks: list[FineChunk]) -> list[str]:
        """批量 upsert FineChunk，返回 chunk_id 列表。"""
        from src.db.models import FineChunk as ORMFineChunk

        ids: list[str] = []
        for c in chunks:
            orm = ORMFineChunk(
                fine_chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                canonical_id=c.canonical_id or "",
                coarse_chunk_id=c.parent_coarse_chunk_id,
                section=c.section,
                page_start=c.page_start,
                page_end=c.page_end,
                char_start=c.char_start,
                char_end=c.char_end,
                text=c.text,
                text_hash=self._text_hash(c.text),
                token_count=c.token_count,
                order_idx=c.order,
                meta_info=c.meta_info,
            )
            self._db.merge(orm)
            ids.append(c.chunk_id)

        self._db.flush()
        logger.debug(f"[ChunkStore] upserted {len(ids)} fine chunks")
        return ids

    def get_coarse_by_doc(self, doc_id: str) -> list[CoarseChunk]:
        """获取文档的所有 coarse chunks。"""
        from src.db.models import CoarseChunk as ORM

        rows = (
            self._db.query(ORM)
            .filter(ORM.doc_id == doc_id)
            .order_by(ORM.order_idx)
            .all()
        )
        return [self._orm_to_coarse(r) for r in rows]

    def get_fine_by_coarse(self, coarse_chunk_id: str) -> list[FineChunk]:
        """获取 coarse chunk 的所有 fine 子块。"""
        from src.db.models import FineChunk as ORM

        rows = (
            self._db.query(ORM)
            .filter(ORM.coarse_chunk_id == coarse_chunk_id)
            .order_by(ORM.order_idx)
            .all()
        )
        return [self._orm_to_fine(r) for r in rows]

    def get_fine_by_doc(self, doc_id: str) -> list[FineChunk]:
        """获取文档的所有 fine chunks。"""
        from src.db.models import FineChunk as ORM

        rows = (
            self._db.query(ORM)
            .filter(ORM.doc_id == doc_id)
            .order_by(ORM.order_idx)
            .all()
        )
        return [self._orm_to_fine(r) for r in rows]

    def delete_by_doc(self, doc_id: str) -> None:
        """删除文档的所有 coarse + fine chunks（cascade 由 DB 负责）。"""
        from src.db.models import CoarseChunk as ORMCoarse

        self._db.query(ORMCoarse).filter(ORMCoarse.doc_id == doc_id).delete(
            synchronize_session="fetch"
        )
        self._db.flush()
        logger.debug(f"[ChunkStore] deleted chunks for doc_id={doc_id}")

    def count_coarse(self, doc_id: str) -> int:
        from src.db.models import CoarseChunk as ORM
        return self._db.query(ORM).filter(ORM.doc_id == doc_id).count()

    def count_fine(self, doc_id: str) -> int:
        from src.db.models import FineChunk as ORM
        return self._db.query(ORM).filter(ORM.doc_id == doc_id).count()

    def _orm_to_coarse(self, orm) -> CoarseChunk:
        return CoarseChunk(
            chunk_id=orm.coarse_chunk_id,
            doc_id=orm.doc_id,
            canonical_id=orm.canonical_id,
            section=orm.section,
            section_level=orm.section_level,
            page_start=orm.page_start,
            page_end=orm.page_end,
            char_start=orm.char_start,
            char_end=orm.char_end,
            text=orm.text,
            token_count=orm.token_count,
            order=orm.order_idx,
            meta_info=orm.meta_info or {},
        )

    def _orm_to_fine(self, orm) -> FineChunk:
        return FineChunk(
            chunk_id=orm.fine_chunk_id,
            doc_id=orm.doc_id,
            canonical_id=orm.canonical_id,
            parent_coarse_chunk_id=orm.coarse_chunk_id,
            section=orm.section,
            page_start=orm.page_start,
            page_end=orm.page_end,
            char_start=orm.char_start,
            char_end=orm.char_end,
            text=orm.text,
            token_count=orm.token_count,
            order=orm.order_idx,
            meta_info=orm.meta_info or {},
        )

    @staticmethod
    def _text_hash(text: str) -> str:
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:24]
