"""
CorpusRepository — 统一访问层。

封装 DocumentStore + ChunkStore + VectorIndex + KeywordIndex + MetadataIndex，
对上层（模块 4/5）提供统一检索 API。

使用示例：
    from src.corpus.store import CorpusRepository
    repo = CorpusRepository()
    repo.index_document(doc, coarse_chunks, fine_chunks)
    results = repo.search_papers(query="multi-agent system", top_k=10)
    evidence = repo.search_evidence(query="attention mechanism", doc_ids=[...])
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from src.corpus.models import (
    CoarseChunk,
    DocumentMeta,
    FineChunk,
    StandardizedDocument,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

# ---------------------------------------------------------------------------
# Search Result
# ---------------------------------------------------------------------------


@dataclass
class PaperSearchResult:
    """
    Paper-level 检索结果。

    来自：coarse chunk hybrid retrieval
    """

    doc_id: str
    canonical_id: str
    title: str
    abstract: str | None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    coarse_chunk_id: str = ""
    coarse_text: str = ""
    coarse_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0
    section: str = "unknown"
    page_start: int = 1
    page_end: int = 1
    source_uri: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class EvidenceSearchResult:
    """
    Evidence-level 检索结果。

    来自：fine chunk retrieval
    """

    fine_chunk_id: str
    coarse_chunk_id: str
    doc_id: str
    canonical_id: str
    text: str
    score: float
    section: str
    page_start: int = 1
    page_end: int = 1
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CorpusRepository
# ---------------------------------------------------------------------------


class CorpusRepository:
    """
    统一检索与写入 API。

    封装了：
    - DocumentStore（论文元数据）
    - ChunkStore（coarse + fine chunks）
    - VectorIndex（Milvus，向量检索）
    - KeywordIndex（PostgreSQL BM25，关键词检索）
    - MetadataIndex（结构化过滤）
    """

    def __init__(
        self,
        db_session: "Session | None" = None,
        milvus_host: str = "127.0.0.1",
        milvus_port: int = 19530,
        embedding_dim: int = EMBEDDING_DIM,
        collection_coarse: str = "coarse_chunks",
        collection_fine: str = "fine_chunks",
    ):
        """
        Args:
            db_session: SQLAlchemy session（PostgreSQL）
            milvus_host: Milvus 服务器地址
            milvus_port: Milvus 端口
            embedding_dim: embedding 向量维度
            collection_coarse: Milvus coarse chunk collection 名
            collection_fine: Milvus fine chunk collection 名
        """
        self._db = db_session
        self._milvus_host = milvus_host
        self._milvus_port = milvus_port
        self._embedding_dim = embedding_dim
        self._collection_coarse = collection_coarse
        self._collection_fine = collection_fine

        # 延迟初始化（连接 Milvus 需要显式调用 connect()）
        self._doc_store = None
        self._chunk_store = None
        self._keyword_index = None
        self._metadata_index = None
        self._vector_index = None

    # ── 连接管理 ──────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """连接所有存储后端。"""
        if self._db is None:
            from src.db.engine import get_session_factory

            self._db = get_session_factory()()

        if self._doc_store is None:
            self._doc_store = _DocumentStore(self._db)
            self._chunk_store = _ChunkStore(self._db)
            self._keyword_index = _PGKeywordIndex(self._db)
            self._metadata_index = _MetadataIndex(self._db)

        # Milvus（允许外部注入，注入后跳过初始化）
        if self._vector_index is None:
            try:
                from src.corpus.store.vector_index import MilvusVectorIndex

                self._vector_index = MilvusVectorIndex(
                    host=self._milvus_host,
                    port=self._milvus_port,
                    alias="corpus_repo",
                )
                self._vector_index.connect()
                # 创建/加载 collection
                self._vector_index.create_collection(
                    self._collection_coarse, dim=self._embedding_dim,
                    description="Coarse chunks (paper-level retrieval)",
                )
                self._vector_index.create_collection(
                    self._collection_fine, dim=self._embedding_dim,
                    description="Fine chunks (evidence-level retrieval)",
                )
                logger.info("[CorpusRepository] Milvus connected OK")
            except Exception as e:
                logger.warning(f"[CorpusRepository] Milvus 连接失败（继续运行）：{e}")
                self._vector_index = None

    # ── 索引写入 ─────────────────────────────────────────────────────────────

    def index_document(
        self,
        doc: StandardizedDocument,
        coarse_chunks: list[CoarseChunk],
        fine_chunks: list[FineChunk],
        embeddings: dict[str, list[float]] | None = None,
    ) -> dict:
        """
        将 Document + Chunks 写入所有存储层。

        Args:
            doc: StandardizedDocument（Module 1 输出）
            coarse_chunks: CoarseChunk 列表（Module 2 输出）
            fine_chunks: FineChunk 列表（Module 2 输出）
            embeddings: {chunk_id: embedding_vector} 字典

        Returns:
            {"doc_id": ..., "coarse_indexed": N, "fine_indexed": M}
        """
        self._ensure_connected()

        stats = {
            "doc_id": doc.doc_id,
            "coarse_indexed": 0,
            "fine_indexed": 0,
            "errors": [],
        }

        try:
            # 1. Document Store
            self._doc_store.upsert(doc)

            # 2. Chunk Store（PostgreSQL）
            if coarse_chunks:
                self._chunk_store.upsert_coarse(coarse_chunks)
                stats["coarse_indexed"] = len(coarse_chunks)
            if fine_chunks:
                self._chunk_store.upsert_fine(fine_chunks)
                stats["fine_indexed"] = len(fine_chunks)

            # 3. Keyword Index（PostgreSQL BM25）
            if coarse_chunks:
                self._keyword_index.index_chunks(coarse_chunks)
            if fine_chunks:
                self._keyword_index.index_chunks(fine_chunks)

            # 4. Vector Index（Milvus）
            if self._vector_index and embeddings:
                self._index_vectors(coarse_chunks, fine_chunks, embeddings)

            logger.info(
                f"[CorpusRepository] Indexed doc={doc.doc_id} "
                f"coarse={stats['coarse_indexed']} fine={stats['fine_indexed']}"
            )
        except Exception as e:
            logger.error(f"[CorpusRepository] index_document 失败：{e}")
            stats["errors"].append(str(e))

        return stats

    def _index_vectors(
        self,
        coarse_chunks: list[CoarseChunk],
        fine_chunks: list[FineChunk],
        embeddings: dict[str, list[float]],
    ) -> None:
        """将 chunks + embeddings 写入 Milvus。"""
        from src.corpus.store.vector_index import VectorRecord

        coarse_records = []
        for c in coarse_chunks:
            vec = embeddings.get(c.chunk_id)
            if not vec:
                continue
            coarse_records.append(
                VectorRecord(
                    id=c.chunk_id,
                    vector=vec,
                    text=c.text[:8192],
                    doc_id=c.doc_id,
                    canonical_id=c.canonical_id or "",
                    section=c.section,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    token_count=c.token_count,
                    metadata={"order": c.order, "section_level": c.section_level},
                )
            )
        if coarse_records:
            self._vector_index.upsert(self._collection_coarse, coarse_records)

        fine_records = []
        for c in fine_chunks:
            vec = embeddings.get(c.chunk_id)
            if not vec:
                continue
            fine_records.append(
                VectorRecord(
                    id=c.chunk_id,
                    vector=vec,
                    text=c.text[:8192],
                    doc_id=c.doc_id,
                    canonical_id=c.canonical_id or "",
                    section=c.section,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    token_count=c.token_count,
                    metadata={"coarse_order": c.meta_info.get("coarse_order", 0)},
                )
            )
        if fine_records:
            self._vector_index.upsert(self._collection_fine, fine_records)

    # ── Paper-level 检索 ─────────────────────────────────────────────────────

    def search_papers(
        self,
        query: str,
        embedding: list[float] | None = None,
        top_k: int = 10,
        year_range: tuple[int, int] | None = None,
        source_type: str | None = None,
        workspace_id: str | None = None,
        hybrid: bool = True,
    ) -> list[PaperSearchResult]:
        """
        Paper-level 检索（coarse chunk hybrid retrieval）。

        策略：
        1. 获取候选 doc_id（metadata filter）
        2. 向量检索（Milvus）+ 关键词检索（BM25）
        3. RRF 融合
        4. 去重（canonical_id）
        5. 补全 Document 元数据

        Returns:
            list[PaperSearchResult]
        """
        self._ensure_connected()

        from src.corpus.store.vector_index import VectorSearchResult as _VecRes

        vector_results: list[_VecRes] = []
        keyword_results: list[dict] = []

        if embedding and self._vector_index:
            try:
                # 向量检索
                filters = {}
                if year_range:
                    filters["year"] = year_range
                vector_results = self._vector_index.search(
                    self._collection_coarse,
                    query_vector=embedding,
                    top_k=top_k * 3,  # 多取一些用于融合
                    filters=filters,
                )
            except Exception as e:
                logger.warning(f"[CorpusRepository] 向量检索失败：{e}")

        if hybrid:
            try:
                # 关键词检索
                keyword_results = self._keyword_index.search_coarse(
                    query, top_k=top_k * 3
                )
            except Exception as e:
                logger.warning(f"[CorpusRepository] BM25 检索失败：{e}")

        # RRF 融合
        fused = self._rrf_fuse(
            vector_results=vector_results,
            keyword_results=keyword_results,
            k=60,
            top_k=top_k,
        )

        # 补全元数据
        return self._enrich_paper_results(fused)

    def _rrf_fuse(
        self,
        vector_results: list,
        keyword_results: list,
        k: int = 60,
        top_k: int = 10,
    ) -> list[dict]:
        """
        倒数排名融合（Reciprocal Rank Fusion）。

        Returns:
            list of {chunk_id, rrf_score, source: "vector"|"keyword"}
        """
        scores: dict[str, float] = {}

        for rank, r in enumerate(vector_results):
            chunk_id = r.id
            if chunk_id not in scores:
                scores[chunk_id] = 0.0
            scores[chunk_id] += 1.0 / (k + rank + 1)

        for rank, r in enumerate(keyword_results):
            chunk_id = r["chunk_id"]
            if chunk_id not in scores:
                scores[chunk_id] = 0.0
            scores[chunk_id] += 1.0 / (k + rank + 1)

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {"chunk_id": cid, "rrf_score": score}
            for cid, score in sorted_scores[:top_k]
        ]

    def _enrich_paper_results(
        self, fused: list[dict]
    ) -> list[PaperSearchResult]:
        """融合结果补全 Document 元数据。"""
        if not fused:
            return []

        chunk_ids = [item["chunk_id"] for item in fused]
        doc_ids = list(set(item["doc_id"] for item in fused if hasattr(item, "doc_id")))

        # 查询文档元数据
        docs_meta = self._doc_store.batch_get(doc_ids)

        results = []
        for item in fused:
            chunk_id = item["chunk_id"]
            # 从粗粒度 chunk 补全信息
            doc = docs_meta.get(item.get("doc_id", ""))
            results.append(
                PaperSearchResult(
                    doc_id=item.get("doc_id", ""),
                    canonical_id=item.get("canonical_id", ""),
                    title=doc.title if doc else item.get("section", "Unknown"),
                    abstract=doc.abstract if doc else None,
                    authors=doc.authors if doc else [],
                    year=doc.year if doc else None,
                    coarse_chunk_id=chunk_id,
                    coarse_text=item.get("text", ""),
                    coarse_score=item.get("score", 0.0),
                    combined_score=item.get("rrf_score", 0.0),
                    section=item.get("section", "unknown"),
                    page_start=item.get("page_start", 1),
                    page_end=item.get("page_end", 1),
                    source_uri="",
                )
            )

        return results

    # ── Evidence-level 检索 ─────────────────────────────────────────────────

    def search_evidence(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        canonical_id: str | None = None,
        embedding: list[float] | None = None,
        sections: list[str] | None = None,
        top_k: int = 10,
        hybrid: bool = True,
    ) -> list[EvidenceSearchResult]:
        """
        Evidence-level 检索（fine chunk retrieval）。

        在指定 doc_ids 或 canonical_id 范围内检索 evidence chunks。
        """
        self._ensure_connected()

        from src.corpus.store.vector_index import VectorSearchResult as _VecRes

        filters = {}
        if doc_ids:
            filters["doc_id"] = doc_ids  # Milvus 支持 IN 过滤
        if canonical_id:
            filters["canonical_id"] = canonical_id

        vector_results: list[_VecRes] = []
        if embedding and self._vector_index:
            try:
                vector_results = self._vector_index.search(
                    self._collection_fine,
                    query_vector=embedding,
                    top_k=top_k * 2,
                    filters=filters if filters else None,
                )
            except Exception as e:
                logger.warning(f"[CorpusRepository] fine chunk 向量检索失败：{e}")

        # 去重（取 top_k）
        seen: set[str] = set()
        unique_results: list[_VecRes] = []
        for r in vector_results:
            if r.id not in seen:
                seen.add(r.id)
                unique_results.append(r)
                if len(unique_results) >= top_k:
                    break

        return [
            EvidenceSearchResult(
                fine_chunk_id=r.id,
                coarse_chunk_id=r.metadata.get("coarse_chunk_id", ""),
                doc_id=r.doc_id,
                canonical_id=r.canonical_id,
                text=r.text[:500],
                score=r.score,
                section=r.section,
                page_start=r.page_start,
                page_end=r.page_end,
                token_count=r.token_count,
                metadata=r.metadata,
            )
            for r in unique_results
        ]

    # ── 模块 4：Paper-level Retrieval（新版）──────────────────────────────────

    def search_papers_ex(
        self,
        query: str,
        embedding: list[float] | None = None,
        sub_questions: list[dict] | None = None,
        year_range: tuple[int, int] | None = None,
        sources: list[str] | None = None,
        venues: list[str] | None = None,
        workspace_id: str | None = None,
        keyword_top_k: int = 30,
        dense_top_k: int = 30,
        top_k: int = 20,
        rrf_k: int = 60,
    ):
        """
        模块 4 入口：Paper-level Retrieval（使用新版 search 模块）。

        流水线：
            Query Preparation → Hybrid Recall（keyword + dense）→ Candidate Merge → InitialPaperCandidates

        Args:
            query: 主检索 query
            embedding: 可选，dense 检索用的 query embedding
            sub_questions: SearchPlan 中的子问题列表
            year_range: 可选，(min_year, max_year)
            sources: 可选，限定来源类型
            venues: 可选，限定会议/期刊
            workspace_id: 可选，限定 workspace
            keyword_top_k: keyword 召回数量
            dense_top_k: dense 召回数量
            top_k: 返回的候选论文数量
            rrf_k: RRF 融合参数

        Returns:
            InitialPaperCandidates（高召回论文候选池）
        """
        from src.corpus.search.retrievers.paper_retriever import PaperRetriever

        self._ensure_connected()

        retriever = PaperRetriever(
            db_session=self._db,
            milvus_index=self._vector_index,
            collection_coarse=self._collection_coarse,
        )

        return retriever.search(
            query=query,
            embedding=embedding,
            sub_questions=sub_questions,
            year_range=year_range,
            sources=sources,
            venues=venues,
            workspace_id=workspace_id,
            keyword_top_k=keyword_top_k,
            dense_top_k=dense_top_k,
            top_k=top_k,
            rrf_k=rrf_k,
        )

    # ── 辅助方法 ────────────────────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        if self._doc_store is None:
            self.connect()

    def get_document(self, doc_id: str) -> StandardizedDocument | None:
        """获取单个 Document。"""
        self._ensure_connected()
        return self._doc_store.get(doc_id)

    def get_coarse_chunks(self, doc_id: str) -> list[CoarseChunk]:
        """获取文档的所有 coarse chunks。"""
        self._ensure_connected()
        return self._chunk_store.get_coarse_by_doc(doc_id)

    def get_fine_chunks(self, doc_id: str) -> list[FineChunk]:
        """获取文档的所有 fine chunks。"""
        self._ensure_connected()
        return self._chunk_store.get_fine_by_doc(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """删除文档及其 chunks。"""
        self._ensure_connected()

        self._chunk_store.delete_by_doc(doc_id)
        self._doc_store.delete(doc_id)

        # 从 Milvus 删除向量
        if self._vector_index:
            try:
                # 获取对应的 chunk_ids
                coarse = self._chunk_store.get_coarse_by_doc(doc_id)
                fine = self._chunk_store.get_fine_by_doc(doc_id)
                ids = [c.chunk_id for c in coarse] + [f.chunk_id for f in fine]
                if ids:
                    self._vector_index.delete_by_id(self._collection_coarse, ids)
                    self._vector_index.delete_by_id(self._collection_fine, ids)
            except Exception as e:
                logger.warning(f"[CorpusRepository] Milvus 删除失败：{e}")

        return True

    def stats(self) -> dict:
        """返回各存储层的统计信息。"""
        self._ensure_connected()

        stats: dict[str, Any] = {}

        try:
            stats["coarse_vector_count"] = self._vector_index.count(self._collection_coarse) if self._vector_index else -1
            stats["fine_vector_count"] = self._vector_index.count(self._collection_fine) if self._vector_index else -1
        except Exception:
            stats["coarse_vector_count"] = -1
            stats["fine_vector_count"] = -1

        try:
            from src.db.models import Document, CoarseChunk, FineChunk
            with self._db.no_autoflush:
                stats["document_count"] = self._db.query(Document).count()
                stats["coarse_chunk_count"] = self._db.query(CoarseChunk).count()
                stats["fine_chunk_count"] = self._db.query(FineChunk).count()
        except Exception:
            stats["document_count"] = -1
            stats["coarse_chunk_count"] = -1
            stats["fine_chunk_count"] = -1

        return stats


# ---------------------------------------------------------------------------
# Internal simple wrappers (lazy initialization)
# ---------------------------------------------------------------------------


class _DocumentStore:
    """延迟初始化的 DocumentStore。"""

    def __init__(self, db):
        self._inner = None
        self._db = db

    def _get(self):
        if self._inner is None:
            from src.corpus.store.document_store import DocumentStore
            self._inner = DocumentStore(self._db)
        return self._inner

    def upsert(self, doc): return self._get().upsert(doc)
    def get(self, doc_id): return self._get().get(doc_id)
    def delete(self, doc_id): return self._get().delete(doc_id)
    def batch_get(self, doc_ids):
        results = {}
        for did in doc_ids:
            doc = self._get().get(did)
            if doc:
                results[did] = doc
        return results


class _ChunkStore:
    def __init__(self, db):
        self._inner = None
        self._db = db

    def _get(self):
        if self._inner is None:
            from src.corpus.store.chunk_store import ChunkStore
            self._inner = ChunkStore(self._db)
        return self._inner

    def upsert_coarse(self, chunks): return self._get().upsert_coarse(chunks)
    def upsert_fine(self, chunks): return self._get().upsert_fine(chunks)
    def get_coarse_by_doc(self, doc_id): return self._get().get_coarse_by_doc(doc_id)
    def get_fine_by_doc(self, doc_id): return self._get().get_fine_by_doc(doc_id)
    def delete_by_doc(self, doc_id): self._get().delete_by_doc(doc_id)


class _PGKeywordIndex:
    def __init__(self, db):
        self._inner = None
        self._db = db

    def _get(self):
        if self._inner is None:
            from src.corpus.store.keyword_index import PGKeywordIndex
            self._inner = PGKeywordIndex(self._db)
        return self._inner

    def index_chunks(self, chunks): return self._get().index_chunks(chunks)

    def search_coarse(self, query, top_k=10):
        """搜索 coarse chunks。"""
        return self._get().search(query, fields=["text"], top_k=top_k)

    def search_fine(self, query, top_k=10):
        return self._get().search(query, fields=["text"], top_k=top_k)


class _MetadataIndex:
    def __init__(self, db):
        self._inner = None
        self._db = db

    def _get(self):
        if self._inner is None:
            from src.corpus.store.metadata_index import MetadataIndex
            self._inner = MetadataIndex(self._db)
        return self._inner
