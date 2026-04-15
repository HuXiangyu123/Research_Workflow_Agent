"""ChunkRetriever — 在已选论文内检索 fine chunks（模块 6）。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.corpus.store import ChunkStore
    from src.corpus.search.models import EvidenceChunk

logger = logging.getLogger(__name__)

# ── RRF k 参数 ────────────────────────────────────────────────────────────────
RRF_K = 40


class ChunkRetriever:
    """
    在已选论文内检索 fine chunks。

    流水线：
        1. Scope restriction — 只在指定 paper_ids 内检索
        2. Keyword search — BM25（基于 text）
        3. Dense search — Milvus 向量检索
        4. RRF fusion — 合并两路结果
        5. Filter — 去噪（太短/太长/chunk）

    使用方式：
        retriever = ChunkRetriever(chunk_store=chunk_store, milvus_client=milvus)
        results = retriever.retrieve(
            paper_ids=["doc-1", "doc-2"],
            query="multi-agent system architecture",
            sub_questions=["how do agents communicate?"],
            top_k_per_paper=10,
            top_k_global=50,
        )
    """

    def __init__(
        self,
        chunk_store: "ChunkStore | None" = None,
        milvus_client=None,
        embedding_model=None,
    ):
        """
        初始化检索器。

        Args:
            chunk_store: ChunkStore 实例（用于获取 fine chunks）
            milvus_client: Milvus 客户端（用于 dense search）
            embedding_model: SentenceTransformer 模型（可选，无则跳过 dense）
        """
        self._chunk_store = chunk_store
        self._milvus_client = milvus_client
        self._embedding_model = embedding_model

    # ── 主入口 ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        paper_ids: list[str],
        query: str,
        sub_questions: list[str] | None = None,
        top_k_per_paper: int = 10,
        top_k_global: int = 50,
    ) -> list["EvidenceChunk"]:
        """
        在指定论文列表内检索 evidence chunks。

        Args:
            paper_ids: 论文 doc_ids（来自模块 5 的 PaperCandidate）
            query: 主检索 query
            sub_questions: 子问题列表（额外检索范围）
            top_k_per_paper: 每篇论文最多召回的 chunks
            top_k_global: 全局最多召回的 chunks 总数

        Returns:
            EvidenceChunk 列表（按 RRF score 降序）
        """
        if not paper_ids:
            return []

        sub_questions = sub_questions or []

        # Step 1: 收集 fine chunks
        all_chunks = self._collect_chunks(paper_ids)
        if not all_chunks:
            logger.warning(f"[ChunkRetriever] paper_ids={paper_ids} 内无 fine chunks")
            return []

        logger.info(
            f"[ChunkRetriever] query='{query[:30]}' "
            f"papers={len(paper_ids)} chunks={len(all_chunks)}"
        )

        # Step 2: Keyword search
        keyword_results = self._keyword_search(query, all_chunks, top_k=top_k_global)
        keyword_results = self._keyword_search_sub_questions(
            sub_questions, all_chunks, top_k=top_k_per_paper, base_results=keyword_results
        )

        # Step 3: Dense search
        dense_results = []
        if self._embedding_model is not None:
            dense_results = self._dense_search(query, all_chunks, top_k=top_k_global)
            dense_results = self._dense_search_sub_questions(
                sub_questions, all_chunks, top_k=top_k_per_paper, base_results=dense_results
            )

        # Step 4: RRF merge
        merged = self._rrf_merge(keyword_results, dense_results)

        # Step 5: Filter
        filtered = self._filter_chunks(merged)

        # Step 6: Attach query metadata and trim
        for r in filtered:
            r.matched_query = query

        return filtered[:top_k_global]

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _collect_chunks(self, paper_ids: list[str]) -> list:
        """收集指定论文的所有 fine chunks。"""
        if self._chunk_store is None:
            return []

        all_chunks = []
        for pid in paper_ids:
            try:
                chunks = self._chunk_store.get_fine_by_doc(pid)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"[ChunkRetriever] 获取 doc_id={pid} chunks 失败：{e}")
        return all_chunks

    def _keyword_search(
        self,
        query: str,
        chunks: list,
        top_k: int,
    ) -> list["EvidenceChunk"]:
        """BM25 keyword search。"""
        if not chunks:
            return []

        try:
            import rank_bm25
        except ImportError:
            logger.warning("[ChunkRetriever] rank_bm25 未安装，跳过 keyword 路径")
            return []

        # Tokenize
        tokenized_chunks = []
        for c in chunks:
            text = getattr(c, "text", "") or ""
            tokens = text.lower().split()
            tokenized_chunks.append((c, tokens))

        corpus = [tokens for _, tokens in tokenized_chunks]
        bm25 = rank_bm25.BM25Okapi(corpus)
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        results = []
        for i, (c, _) in enumerate(tokenized_chunks):
            if scores[i] > 0:
                results.append(self._chunk_to_result(c, keyword_score=float(scores[i])))

        results.sort(key=lambda r: r.keyword_score, reverse=True)
        return results[:top_k]

    def _keyword_search_sub_questions(
        self,
        sub_questions: list[str],
        chunks: list,
        top_k: int,
        base_results: list["EvidenceChunk"],
    ) -> list["EvidenceChunk"]:
        """对子问题做 keyword search 并合并到 base_results。"""
        for sq in sub_questions:
            sq_text = sq if isinstance(sq, str) else getattr(sq, "id", str(sq))
            sq_results = self._keyword_search(sq_text, chunks, top_k=top_k)
            for sq_r in sq_results:
                sq_r.sub_question_id = sq_text
                base_results.append(sq_r)
        return base_results

    def _dense_search(
        self,
        query: str,
        chunks: list,
        top_k: int,
    ) -> list["EvidenceChunk"]:
        """Milvus 向量相似度检索。"""
        if not chunks or self._embedding_model is None:
            return []

        try:
            import numpy as np
        except ImportError:
            return []

        # Encode query
        try:
            query_embedding = self._embedding_model.encode([query])
            if hasattr(query_embedding, "numpy"):
                query_embedding = query_embedding.numpy()
            query_embedding = np.array(query_embedding).flatten()
        except Exception as e:
            logger.warning(f"[ChunkRetriever] embedding 失败：{e}")
            return []

        results = []
        for c in chunks:
            chunk_vec = getattr(c, "embedding", None)
            if chunk_vec is None:
                continue
            try:
                vec = np.array(chunk_vec)
                score = float(np.dot(query_embedding, vec) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(vec) + 1e-8
                ))
                if score > 0:
                    r = self._chunk_to_result(c, dense_score=score)
                    results.append(r)
            except Exception:
                pass

        results.sort(key=lambda r: r.dense_score, reverse=True)
        return results[:top_k]

    def _dense_search_sub_questions(
        self,
        sub_questions: list[str],
        chunks: list,
        top_k: int,
        base_results: list["EvidenceChunk"],
    ) -> list["EvidenceChunk"]:
        """对子问题做 dense search 并合并到 base_results。"""
        for sq in sub_questions:
            sq_text = sq if isinstance(sq, str) else getattr(sq, "id", str(sq))
            sq_results = self._dense_search(sq_text, chunks, top_k=top_k)
            for sq_r in sq_results:
                sq_r.sub_question_id = sq_text
                base_results.append(sq_r)
        return base_results

    def _rrf_merge(
        self,
        keyword_results: list["EvidenceChunk"],
        dense_results: list["EvidenceChunk"],
        k: int = RRF_K,
    ) -> list["EvidenceChunk"]:
        """Reciprocal Rank Fusion 合并两路检索结果。"""
        # Build rank maps
        k_rank: dict[str, int] = {}
        d_rank: dict[str, int] = {}

        for rank, r in enumerate(keyword_results):
            k_rank[r.chunk_id] = rank + 1
        for rank, r in enumerate(dense_results):
            d_rank[r.chunk_id] = rank + 1

        # Get all chunk_ids
        all_ids = set(k_rank.keys()) | set(d_rank.keys())

        # Compute RRF scores
        merged_map: dict[str, EvidenceChunk] = {}
        for rid in keyword_results:
            merged_map[rid.chunk_id] = rid
        for rid in dense_results:
            if rid.chunk_id not in merged_map:
                merged_map[rid.chunk_id] = rid

        rrf_scores: dict[str, float] = {}
        for cid in all_ids:
            kr = k_rank.get(cid, k + 1)
            dr = d_rank.get(cid, k + 1)
            rrf_scores[cid] = (1 / (k + kr)) + (1 / (k + dr))

        # Update RRF scores and sort
        for cid, rrf in rrf_scores.items():
            r = merged_map[cid]
            r.rrf_score = rrf
            r.chunk_path = self._determine_path(cid, k_rank, d_rank)

        return sorted(merged_map.values(), key=lambda r: r.rrf_score, reverse=True)

    def _determine_path(
        self,
        chunk_id: str,
        k_rank: dict[str, int],
        d_rank: dict[str, int],
    ) -> str:
        """判断 chunk 来自哪条检索路径。"""
        has_k = chunk_id in k_rank
        has_d = chunk_id in d_rank
        if has_k and has_d:
            return "hybrid"
        elif has_k:
            return "keyword"
        else:
            return "dense"

    def _filter_chunks(
        self,
        results: list["EvidenceChunk"],
        min_text_len: int = 50,
        max_text_len: int = 2000,
    ) -> list["EvidenceChunk"]:
        """过滤噪声 chunks（太短/太长）。"""
        filtered = []
        for r in results:
            text_len = len(r.text)
            if text_len < min_text_len or text_len > max_text_len:
                continue
            filtered.append(r)
        return filtered

    def _chunk_to_result(
        self,
        chunk,
        keyword_score: float = 0.0,
        dense_score: float = 0.0,
    ) -> "EvidenceChunk":
        """将 FineChunk 转换为 EvidenceChunk。"""
        from src.corpus.search.models import EvidenceChunk, ScoreBreakdown

        scores = ScoreBreakdown(
            keyword_score=keyword_score,
            dense_score=dense_score,
        )
        return EvidenceChunk(
            chunk_id=getattr(chunk, "chunk_id", ""),
            paper_id=getattr(chunk, "doc_id", ""),
            canonical_id=getattr(chunk, "canonical_id", "") or "",
            text=getattr(chunk, "text", "") or "",
            section=getattr(chunk, "section", "") or "",
            page_start=getattr(chunk, "page_start", 1),
            page_end=getattr(chunk, "page_end", 1),
            scores=scores,
            keyword_score=keyword_score,
            dense_score=dense_score,
        )
