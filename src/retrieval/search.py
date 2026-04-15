"""混合检索器：PostgreSQL BM25 + FAISS 向量（RRF 融合）。"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any

import faiss
import numpy as np
from sqlalchemy import func, text

from src.db import get_db_session
from src.db.models import Chunk, Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.getcwd(), "data")
VECTOR_INDEX_DIR = os.path.join(DATA_DIR, "indexes", "vector")
FAISS_INDEX_FILE = os.path.join(VECTOR_INDEX_DIR, "faiss.index")
CHUNK_ID_MAP_FILE = os.path.join(VECTOR_INDEX_DIR, "chunk_ids.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"


class HybridSearcher:
    """混合检索：PostgreSQL BM25（FTS5 via SQLAlchemy）+ FAISS 向量 + RRF 融合。"""

    def __init__(self):
        self._faiss_index: faiss.IndexFlat | None = None
        self._chunk_ids: list[str] | None = None
        self._model = None
        self._load_vector_index()

    def _load_vector_index(self) -> None:
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(CHUNK_ID_MAP_FILE):
            logger.info("Loading FAISS vector index...")
            self._faiss_index = faiss.read_index(FAISS_INDEX_FILE)
            with open(CHUNK_ID_MAP_FILE, "rb") as f:
                self._chunk_ids = pickle.load(f)
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(MODEL_NAME)
                logger.info(f"Vector index loaded with {len(self._chunk_ids)} chunks.")
            except ImportError:
                logger.warning("sentence-transformers not installed; vector search unavailable.")
                self._model = None
        else:
            logger.warning("FAISS index not found. Only BM25 available.")

    def search_bm25(self, query: str, k: int = 50) -> list[dict[str, Any]]:
        """使用 PostgreSQL ts_rank 做 BM25 排序。"""
        with get_db_session() as s:
            # 手动构造 FTS 查询（SQLite 兼容语法 → PostgreSQL 适配）
            ranked = (
                s.query(
                    Chunk.chunk_id,
                    Chunk.text,
                    func.ts_rank(
                        func.to_tsvector("english", Chunk.text), func.plainto_tsquery("english", query)
                    ).label("rank"),
                    Chunk.doc_id,
                    Document.title,
                    Document.source_uri,
                )
                .join(Document, Chunk.doc_id == Document.doc_id)
                .filter(
                    func.to_tsvector("english", Chunk.text).op("@@")(
                        func.plainto_tsquery("english", query)
                    )
                )
                .order_by(text("rank DESC"))
                .limit(k)
            )
            return [
                {
                    "chunk_id": row[0],
                    "text": row[1],
                    "score": float(row[2]) if row[2] else 0.0,
                    "doc_id": row[3],
                    "title": row[4],
                    "source_uri": row[5],
                }
                for row in ranked.all()
            ]

    def search_vector(self, query: str, k: int = 50) -> list[dict[str, Any]]:
        """使用 FAISS 向量索引做语义检索。"""
        if not self._faiss_index or not self._model or self._chunk_ids is None:
            return []

        embedding = self._model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(embedding)

        scores, indices = self._faiss_index.search(embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self._chunk_ids):
                continue
            results.append({"chunk_id": self._chunk_ids[idx], "score": float(score)})
        return results

    def _rrf_fusion(
        self, results_dict: dict[str, dict[str, float]], k: int = 60
    ) -> list[dict[str, Any]]:
        """倒数排名融合（Reciprocal Rank Fusion）。"""
        final_scores: dict[str, float] = {}
        for _method, results in results_dict.items():
            for rank, item in enumerate(results):
                chunk_id = item["chunk_id"]
                score = 1.0 / (k + rank + 1)
                if chunk_id not in final_scores:
                    final_scores[chunk_id] = 0.0
                final_scores[chunk_id] += score

        sorted_chunks = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"chunk_id": cid, "score": score} for cid, score in sorted_chunks]

    def _hydrate_chunks(self, chunk_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将 chunk_id 列表补全为完整字段。"""
        if not chunk_scores:
            return []

        chunk_ids = [item["chunk_id"] for item in chunk_scores]
        with get_db_session() as s:
            rows = (
                s.query(Chunk, Document.title, Document.source_uri)
                .join(Document, Chunk.doc_id == Document.doc_id)
                .filter(Chunk.chunk_id.in_(chunk_ids))
                .all()
            )
            chunk_map = {
                row[0].chunk_id: {
                    "chunk_id": row[0].chunk_id,
                    "text": row[0].text,
                    "doc_id": row[0].doc_id,
                    "section_path": row[0].section_path,
                    "page_start": row[0].page_start,
                    "page_end": row[0].page_end,
                    "title": row[1],
                    "source_uri": row[2],
                }
                for row in rows
            }

        hydrated = []
        for item in chunk_scores:
            cid = item["chunk_id"]
            if cid in chunk_map:
                data = chunk_map[cid]
                data["score"] = item["score"]
                hydrated.append(data)

        return hydrated

    def search(
        self, query: str, top_k: int = 12, bm25_k: int = 60, vec_k: int = 60
    ) -> list[dict[str, Any]]:
        """混合搜索：BM25 + 向量 + RRF 融合。"""
        bm25_res = self.search_bm25(query, bm25_k)
        vec_res = self.search_vector(query, vec_k)

        vec_score_map = {r["chunk_id"]: r["score"] for r in vec_res}

        fused = self._rrf_fusion({"bm25": bm25_res, "vector": vec_res}, k=60)
        top_fused = fused[:top_k]

        # 补全 BM25 结果中的 score 字段
        bm25_score_map = {r["chunk_id"]: r["score"] for r in bm25_res}
        for item in top_fused:
            cid = item["chunk_id"]
            item["bm25_score"] = bm25_score_map.get(cid, 0.0)
            item["vec_score"] = vec_score_map.get(cid, 0.0)

        return self._hydrate_chunks(top_fused)


_searcher: HybridSearcher | None = None


def get_searcher() -> HybridSearcher:
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher()
    return _searcher
