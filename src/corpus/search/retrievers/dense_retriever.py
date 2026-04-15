"""Dense Retriever — Milvus 向量召回（coarse chunk / title embedding）。"""

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


class DenseRetriever:
    """
    Paper-level 向量检索器。

    召回对象：
    - coarse_chunk 向量（在 coarse_vector_index 上）
    - title 向量（在 title 专属 index 上，未来扩展）

    适合抓取：同义表达 / 描述型 query / 用户没有精确复述术语的情况 /
    语义上相近的方法路线。
    """

    def __init__(
        self,
        db_session: "Session",
        milvus_index,  # MilvusVectorIndex instance
        collection_coarse: str = "coarse_chunks",
    ):
        """
        Args:
            db_session: SQLAlchemy session
            milvus_index: MilvusVectorIndex 实例
            collection_coarse: coarse chunk collection 名
        """
        self._db = db_session
        self._milvus = milvus_index
        self._collection_coarse = collection_coarse

    def search(
        self,
        query_vector: list[float],
        path: RetrievalPath = RetrievalPath.DENSE_COARSE,
        top_k: int = 30,
        filters: dict | None = None,
    ) -> list[RecallEvidence]:
        """
        向量检索。

        Args:
            query_vector: 查询向量
            path: 召回路径
            top_k: 返回数量
            filters: Milvus filter 表达式

        Returns:
            RecallEvidence 列表
        """
        start = time.time()

        if self._milvus is None:
            logger.warning("[DenseRetriever] Milvus 未连接，无法向量检索")
            return []

        try:
            collection_name = self._collection_coarse
            if path == RetrievalPath.DENSE_TITLE:
                # 未来可扩展为 title embedding collection
                collection_name = self._collection_coarse

            results = self._milvus.search(
                name=collection_name,
                query_vector=query_vector,
                top_k=top_k,
                filters=filters,
            )

            evidence_list: list[RecallEvidence] = []
            for rank, r in enumerate(results):
                evidence_list.append(
                    RecallEvidence(
                        chunk_id=r.id,
                        doc_id=r.doc_id,
                        canonical_id=getattr(r, "canonical_id", "") or "",
                        section=getattr(r, "section", "unknown") or "unknown",
                        text=r.text[:500] if r.text else "",
                        score=r.score,
                        path=path,
                        page_start=r.page_start if hasattr(r, "page_start") else 1,
                        page_end=r.page_end if hasattr(r, "page_end") else 1,
                        token_count=r.token_count if hasattr(r, "token_count") else 0,
                    )
                )

            duration = (time.time() - start) * 1000
            logger.debug(
                f"[DenseRetriever] path={path.value} top_k={top_k} "
                f"returned={len(evidence_list)} duration={duration:.1f}ms"
            )
            return evidence_list

        except Exception as e:
            logger.error(f"[DenseRetriever] search 失败（path={path}）：{e}")
            return []

    def search_coarse(
        self,
        query_vector: list[float],
        top_k: int = 30,
        filters: dict | None = None,
    ) -> list[RecallEvidence]:
        """在 coarse chunk 向量上检索（快捷入口）。"""
        return self.search(
            query_vector=query_vector,
            path=RetrievalPath.DENSE_COARSE,
            top_k=top_k,
            filters=filters,
        )
