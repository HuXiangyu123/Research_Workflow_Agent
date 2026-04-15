"""
Chunk Linker — 建立 CoarseChunk ↔ FineChunk 父子关系 + 持久化到 PostgreSQL。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.corpus.models import CoarseChunk, FineChunk

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunk Linker
# ---------------------------------------------------------------------------


class ChunkLinker:
    """
    建立 CoarseChunk ↔ FineChunk 父子关系，并持久化到 PostgreSQL。

    职责：
    1. 验证 FineChunk.parent_coarse_chunk_id 指向有效 CoarseChunk
    2. 将 CoarseChunk[] + FineChunk[] 批量写入 PostgreSQL
    3. 建立 parent-child 索引
    """

    def link_and_persist(
        self,
        coarse_chunks: list[CoarseChunk],
        fine_chunks: list[FineChunk],
        db_session: "Session | None" = None,
    ) -> tuple[list[str], list[str]]:
        """
        建立父子关系并持久化。

        Args:
            coarse_chunks: CoarseChunk 列表
            fine_chunks: FineChunk 列表
            db_session: 可选的 SQLAlchemy session

        Returns:
            (coarse_chunk_ids, fine_chunk_ids)：写入成功的主键列表
        """
        # 1. 验证父子关系
        coarse_id_set = {c.chunk_id for c in coarse_chunks}
        orphan_fines = [
            f for f in fine_chunks
            if f.parent_coarse_chunk_id not in coarse_id_set
        ]
        if orphan_fines:
            logger.warning(
                f"发现 {len(orphan_fines)} 个 FineChunk 无父 CoarseChunk，跳过持久化"
            )

        # 过滤 orphan
        valid_fines = [
            f for f in fine_chunks
            if f.parent_coarse_chunk_id in coarse_id_set
        ]

        coarse_ids: list[str] = []
        fine_ids: list[str] = []

        # 2. 持久化（若 db_session 可用）
        if db_session:
            coarse_ids, fine_ids = self._persist_to_db(
                coarse_chunks, valid_fines, db_session
            )
        else:
            # 仅生成 ID
            coarse_ids = [c.chunk_id for c in coarse_chunks]
            fine_ids = [f.chunk_id for f in valid_fines]

        # 3. 建立反向索引（coarse → fine_ids）
        coarse_to_fines: dict[str, list[str]] = {}
        for f in valid_fines:
            coarse_to_fines.setdefault(f.parent_coarse_chunk_id, []).append(f.chunk_id)

        logger.info(
            f"[ChunkLinker] 写入完成：{len(coarse_ids)} coarse chunks, "
            f"{len(fine_ids)} fine chunks"
        )

        return coarse_ids, fine_ids

    def _persist_to_db(
        self,
        coarse_chunks: list[CoarseChunk],
        fine_chunks: list[FineChunk],
        db_session: "Session",
    ) -> tuple[list[str], list[str]]:
        """批量写入 PostgreSQL。"""
        from src.db.models import CoarseChunk as ORMCoarseChunk
        from src.db.models import FineChunk as ORMFineChunk

        coarse_ids: list[str] = []
        fine_ids: list[str] = []

        # 批量 upsert coarse chunks
        orm_coarses = []
        for c in coarse_chunks:
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
                text_hash=_text_hash(c.text),
                token_count=c.token_count,
                order_idx=c.order,
                meta_info=c.meta_info,
            )
            orm_coarses.append(orm)
            coarse_ids.append(c.chunk_id)

        for oc in orm_coarses:
            db_session.merge(oc)

        # 批量 upsert fine chunks
        orm_fines = []
        for f in fine_chunks:
            orm = ORMFineChunk(
                fine_chunk_id=f.chunk_id,
                doc_id=f.doc_id,
                canonical_id=f.canonical_id or "",
                coarse_chunk_id=f.parent_coarse_chunk_id,
                section=f.section,
                page_start=f.page_start,
                page_end=f.page_end,
                char_start=f.char_start,
                char_end=f.char_end,
                text=f.text,
                text_hash=_text_hash(f.text),
                token_count=f.token_count,
                order_idx=f.order,
                meta_info=f.meta_info,
            )
            orm_fines.append(orm)
            fine_ids.append(f.chunk_id)

        for of in orm_fines:
            db_session.merge(of)

        db_session.flush()
        return coarse_ids, fine_ids


def _text_hash(text: str) -> str:
    import hashlib
    return hashlib.md5(text.encode()).hexdigest()[:24]
