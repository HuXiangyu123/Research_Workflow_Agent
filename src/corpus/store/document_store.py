"""Document Store — PostgreSQL 存储论文级元数据。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from src.corpus.models import StandardizedDocument

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class DocumentStore:
    """
    论文级元数据存储（PostgreSQL）。

    职责：
    - upsert_document：写入/更新 Document 元数据
    - get_document：通过 doc_id / canonical_id 查询
    - list_by_canonical：列出同一 canonical paper 的所有文档版本
    - delete：删除文档（cascade 删除 chunks）
    """

    def __init__(self, db_session: "Session"):
        self._db = db_session

    def upsert(self, doc: StandardizedDocument) -> str:
        """
        写入或更新 Document 元数据。

        Returns:
            doc_id
        """
        from src.db.models import Document

        authors_str = ", ".join(doc.authors) if doc.authors else None
        title_str = doc.title or "Unknown"
        source_type = None
        if doc.source_ref:
            source_type = doc.source_ref.source_type.value if hasattr(doc.source_ref.source_type, 'value') else str(doc.source_ref.source_type)

        orm = Document(
            doc_id=doc.doc_id,
            source_id=doc.source_ref.source_id if doc.source_ref else doc.doc_id,
            canonical_id=doc.canonical_id,
            source_uri=doc.source_ref.uri_or_path if doc.source_ref else "",
            source_type=source_type,
            title=title_str,
            authors=authors_str,
            published_date=str(doc.year) if doc.year else None,
            summary=doc.abstract,
            status="ready",
            parse_quality=doc.parse_quality_score,
        )
        self._db.merge(orm)
        self._db.flush()
        logger.debug(f"[DocumentStore] upsert doc_id={doc.doc_id}, canonical_id={doc.canonical_id}")
        return doc.doc_id

    def get(self, doc_id: str) -> Optional[StandardizedDocument]:
        """通过 doc_id 查询 Document。"""
        from src.db.models import Document

        orm = self._db.get(Document, doc_id)
        if orm is None:
            return None
        return self._orm_to_doc(orm)

    def get_by_canonical(self, canonical_id: str) -> list[StandardizedDocument]:
        """通过 canonical_id 列出所有文档版本。"""
        from src.db.models import Document

        rows = (
            self._db.query(Document)
            .filter(Document.canonical_id == canonical_id)
            .all()
        )
        return [self._orm_to_doc(r) for r in rows]

    def delete(self, doc_id: str) -> bool:
        """删除 Document（级联删除 chunks）。"""
        from src.db.models import Document

        row = self._db.get(Document, doc_id)
        if row is None:
            return False
        self._db.delete(row)
        self._db.flush()
        logger.debug(f"[DocumentStore] deleted doc_id={doc_id}")
        return True

    def list_by_workspace(
        self, workspace_id: str | None, limit: int = 100, offset: int = 0
    ) -> list[StandardizedDocument]:
        """
        列出 workspace 下的文档。

        Note: 当前 Document 表暂无 workspace_id 字段，
        后续扩展时可通过 canonical_paper.workspace_id join 实现。
        """
        from src.db.models import Document

        query = self._db.query(Document).filter(Document.status == "ready")
        if workspace_id:
            # 未来扩展：join canonical_papers
            pass
        rows = query.offset(offset).limit(limit).all()
        return [self._orm_to_doc(r) for r in rows]

    def _orm_to_doc(self, orm) -> StandardizedDocument:
        """将 ORM 对象转换为 StandardizedDocument。"""
        authors = []
        if orm.authors:
            authors = [a.strip() for a in orm.authors.split(",") if a.strip()]

        year = None
        if orm.published_date:
            try:
                year = int(orm.published_date[:4])
            except (ValueError, TypeError):
                pass

        return StandardizedDocument(
            doc_id=orm.doc_id,
            workspace_id=None,
            canonical_id=orm.canonical_id,
            title=orm.title or "Unknown",
            authors=authors,
            year=year,
            abstract=orm.summary,
            ingest_status=orm.status,
            parse_quality_score=orm.parse_quality or 0.0,
        )
