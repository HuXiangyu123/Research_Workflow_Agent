"""元数据库访问层（PostgreSQL + SQLAlchemy）。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from src.db import get_db_session
from src.db.models import Chunk as _ORMChunk
from src.db.models import Document as _ORMDocument

if TYPE_CHECKING:
    from src.corpus.models import Chunk, DocumentMeta


class MetaDB:
    """PostgreSQL 元数据库访问接口（兼容旧 SQLite 版本）。"""

    def upsert_document(self, doc: "DocumentMeta") -> None:
        with get_db_session() as s:
            orm = _ORMDocument(
                doc_id=doc.doc_id,
                source_id=doc.source_id,
                source_uri=doc.source_uri,
                title=doc.title,
                authors=doc.authors,
                published_date=doc.published_date,
                summary=doc.summary,
                added_at=doc.added_at,
                updated_at=doc.updated_at,
                status=doc.status,
                error=doc.error,
            )
            s.merge(orm)

    def get_document(self, doc_id: str) -> Optional["DocumentMeta"]:
        from src.corpus.models import DocumentMeta as _DM

        with get_db_session() as s:
            orm = s.get(_ORMDocument, doc_id)
            if orm is None:
                return None
            return _DM(
                doc_id=orm.doc_id,
                source_id=orm.source_id,
                source_uri=orm.source_uri,
                title=orm.title,
                authors=orm.authors,
                published_date=orm.published_date,
                summary=orm.summary,
                added_at=orm.added_at,
                updated_at=orm.updated_at,
                status=orm.status,
                error=orm.error,
            )

    def insert_chunks(self, chunks: list["Chunk"]) -> None:
        with get_db_session() as s:
            orm_chunks = []
            for ch in chunks:
                orm_chunks.append(
                    _ORMChunk(
                        chunk_id=ch.chunk_id,
                        doc_id=ch.doc_id,
                        section_path=ch.section_path,
                        page_start=ch.span.page_start,
                        page_end=ch.span.page_end,
                        char_start=ch.span.char_start,
                        char_end=ch.span.char_end,
                        text=ch.text,
                        text_hash=ch.text_hash,
                        len_chars=ch.len_chars,
                    )
                )
            for oc in orm_chunks:
                s.merge(oc)

    def clear_chunks_for_doc(self, doc_id: str) -> None:
        with get_db_session() as s:
            s.query(_ORMChunk).filter(_ORMChunk.doc_id == doc_id).delete(
                synchronize_session="fetch"
            )
