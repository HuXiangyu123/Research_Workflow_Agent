"""Paper Retriever — 模块 4 统一检索入口。

调度 keyword / dense / filters 三路召回，输出 InitialPaperCandidates。
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

from src.corpus.search.retrievers.models import (
    InitialPaperCandidates,
    MergedCandidate,
    RetrievalPath,
    RetrievalTrace,
)
from src.corpus.search.retrievers.query_prep import (
    SearchInput,
    PreparedQuery,
    prepare_search,
)
from src.corpus.search.retrievers.keyword_retriever import KeywordRetriever
from src.corpus.search.retrievers.dense_retriever import DenseRetriever
from src.corpus.search.retrievers.filter_compiler import FilterCompiler
from src.corpus.search.retrievers.candidate_merger import CandidateMerger
from src.corpus.search.retrievers.trace_builder import TraceBuilder

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from src.api.routes.corpus_search import SearchFilters

logger = logging.getLogger(__name__)

# Default RRF k parameter
RRF_K = 40


class PaperRetriever:
    """
    Paper-level Retrieval 统一入口。

    流水线：
        SearchInput
            ↓
        Query Preparation（提取 main + sub-question queries）
            ↓
        Hybrid Recall（并行 keyword + dense + filters）
            ↓
        Candidate Merge（RRF 融合 + doc_id 归并）
            ↓
        InitialPaperCandidates（高召回论文候选池）
    """

    def __init__(
        self,
        db_session: "Session",
        milvus_index=None,  # MilvusVectorIndex instance, optional
        collection_coarse: str = "coarse_chunks",
        embedding_model=None,  # SentenceTransformer model, optional
    ):
        """
        Args:
            db_session: SQLAlchemy session
            milvus_index: MilvusVectorIndex 实例（可选，无则跳过 dense 召回）
            collection_coarse: coarse chunk collection 名
            embedding_model: SentenceTransformer 模型（可选，无则跳过 dense 召回）
        """
        self._db = db_session
        self._milvus = milvus_index
        self._collection_coarse = collection_coarse
        self._embedding_model = embedding_model

        # 子组件（延迟初始化）
        self._keyword_retriever: KeywordRetriever | None = None
        self._dense_retriever: DenseRetriever | None = None
        self._filter_compiler = FilterCompiler()
        self._merger: CandidateMerger | None = None

    def _ensure_components(self) -> None:
        """延迟初始化子组件。"""
        if self._keyword_retriever is None:
            self._keyword_retriever = KeywordRetriever(self._db)
        if self._dense_retriever is None and self._milvus is not None:
            self._dense_retriever = DenseRetriever(
                db_session=self._db,
                milvus_index=self._milvus,
                collection_coarse=self._collection_coarse,
            )
        if self._merger is None:
            self._merger = CandidateMerger(self._db)

    def search(
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
        rrf_k: int = RRF_K,
        filters: Optional["SearchFilters"] = None,
        recall_top_k: int = 100,
    ) -> InitialPaperCandidates:
        """
        Paper-level 检索主入口。

        Args:
            query: 主检索 query
            embedding: 可选，dense 检索用的 query embedding
            sub_questions: SearchPlan 中的子问题列表
            year_range: 可选，(min_year, max_year)
            sources: 可选，限定来源类型 ["arxiv", "local_pdf", ...]
            venues: 可选，限定会议/期刊
            workspace_id: 可选，限定 workspace
            keyword_top_k: keyword 召回数量
            dense_top_k: dense 召回数量
            top_k: 返回的候选论文数量
            rrf_k: RRF 融合参数
            filters: Pydantic SearchFilters 模型，优先级高于散参
            recall_top_k: 统一控制各路召回量

        Returns:
            InitialPaperCandidates（高召回论文候选池）
        """
        # 如果传了 filters 对象，优先用它覆盖散参
        if filters is not None:
            if filters.year_range is not None:
                year_range = filters.year_range
            if filters.source_type:
                sources = filters.source_type
            if filters.venue:
                venues = filters.venue

        start = time.time()
        self._ensure_components()

        # 1. Query Preparation
        search_input = prepare_search(
            query=query,
            sub_questions=sub_questions,
            year_range=year_range,
            sources=sources,
            venues=venues,
            workspace_id=workspace_id,
        )
        queries, filters = search_input.prepare()

        # 2. Compile filters
        compiled = self._filter_compiler.compile(filters)

        # 3. Hybrid Recall（主 query）
        main_query = queries[0]  # 第一个是主 query
        kw_evidence, dense_evidence, traces = self._recall_for_query(
            main_query,
            compiled,
            keyword_top_k=recall_top_k,
            dense_top_k=recall_top_k,
            embedding=embedding,
        )

        # 4. 对每个子问题分别召回并合并
        sub_candidates: list[MergedCandidate] = list(kw_evidence)  # 归并后再说
        for sq_query in queries[1:]:
            sq_kw, sq_dense, sq_traces = self._recall_for_query(
                sq_query,
                compiled,
                keyword_top_k=recall_top_k,
                dense_top_k=recall_top_k,
                embedding=embedding,
            )
            traces.extend(sq_traces)

            # 合并到总候选（跨 query 归并）
            sub_candidates = self._merge_sub_candidates(
                sub_candidates,
                sq_kw,
                sq_dense,
                sq_query.sub_question_id,
                rrf_k,
            )

        # 5. Merge 主 query + sub questions
        all_candidates: list[MergedCandidate] = []
        if kw_evidence or dense_evidence:
            merged, main_traces = self._merger.merge(
                query=main_query.text,
                keyword_evidence=kw_evidence,
                dense_evidence=dense_evidence,
                sub_question_id=None,
                rrf_k=rrf_k,
                top_k=top_k * 2,  # 多取一些供跨 query 归并
            )
            all_candidates = merged

        # 6. 截断 + 统计
        all_candidates = sorted(
            all_candidates, key=lambda c: c.rrf_score, reverse=True
        )[:top_k]

        # 7. 补全元数据
        self._merger._enrich_metadata(all_candidates)

        # 8. 构建输出
        result = InitialPaperCandidates(
            query=query,
            candidates=all_candidates,
            traces=traces,
        )
        summary = result.build_summary()

        duration = (time.time() - start) * 1000
        logger.info(
            f"[PaperRetriever] query='{query[:50]}' candidates={result.total_candidates} "
            f"kw={summary['keyword_only']} dense={summary['dense_only']} "
            f"both={summary['both_channels']} duration={duration:.0f}ms"
        )

        return result

    def _recall_for_query(
        self,
        query: PreparedQuery,
        compiled_filters,
        keyword_top_k: int,
        dense_top_k: int,
        embedding: list[float] | None,
    ) -> tuple[list, list, list[RetrievalTrace]]:
        """对单条 PreparedQuery 执行 hybrid recall。"""
        kw_evidence: list = []
        dense_evidence: list = []
        traces: list[RetrievalTrace] = []
        start = time.time()

        # Keyword recall（多路径并行）
        kw_results = self._keyword_retriever.search_all_paths(
            query=query.text,
            top_k=keyword_top_k,
            filters=compiled_filters.milvus_filter,
        )
        for path, ev_list in kw_results.items():
            if ev_list:
                kw_evidence.extend(ev_list)
                traces.append(
                    RetrievalTrace(
                        query=query.text,
                        sub_question_id=query.sub_question_id,
                        retrieval_path=path,
                        target_index="coarse" if "coarse" in path.value else "title",
                        filter_summary=compiled_filters.filter_summary,
                        top_k_requested=keyword_top_k,
                        returned_doc_ids=[e.doc_id for e in ev_list],
                        returned_chunk_ids=[e.chunk_id for e in ev_list],
                        returned_count=len(ev_list),
                        duration_ms=(time.time() - start) * 1000,
                    )
                )

        # Dense recall
        if embedding and self._dense_retriever:
            try:
                dense_results = self._dense_retriever.search_coarse(
                    query_vector=embedding,
                    top_k=dense_top_k,
                    filters=compiled_filters.milvus_filter,
                )
                if dense_results:
                    dense_evidence = dense_results
                    traces.append(
                        RetrievalTrace(
                            query=query.text,
                            sub_question_id=query.sub_question_id,
                            retrieval_path=RetrievalPath.DENSE_COARSE,
                            target_index="coarse",
                            filter_summary=compiled_filters.filter_summary,
                            top_k_requested=dense_top_k,
                            returned_doc_ids=[e.doc_id for e in dense_results],
                            returned_chunk_ids=[e.chunk_id for e in dense_results],
                            returned_count=len(dense_results),
                            duration_ms=(time.time() - start) * 1000,
                        )
                    )
            except Exception as e:
                logger.warning(f"[PaperRetriever] dense recall 失败：{e}")

        return kw_evidence, dense_evidence, traces

    def _merge_sub_candidates(
        self,
        existing: list,
        kw_evidence: list,
        dense_evidence: list,
        sub_question_id: str | None,
        rrf_k: int,
    ) -> list:
        """将子问题的召回结果归并到已有候选列表。"""
        if not kw_evidence and not dense_evidence:
            return existing

        # 对新召回打分并合并
        new_merged, _ = self._merger.merge(
            query="",
            keyword_evidence=kw_evidence,
            dense_evidence=dense_evidence,
            sub_question_id=sub_question_id,
            rrf_k=rrf_k,
            top_k=50,
        )

        # 归并到现有
        existing_by_doc = {c.doc_id: c for c in existing}
        for nc in new_merged:
            if nc.doc_id in existing_by_doc:
                # 合并 matched_queries 和 matched_sub_question_ids
                existing_c = existing_by_doc[nc.doc_id]
                existing_c.rrf_score += nc.rrf_score
                if sub_question_id and sub_question_id not in existing_c.matched_sub_question_ids:
                    existing_c.matched_sub_question_ids.append(sub_question_id)
            else:
                if sub_question_id:
                    nc.matched_sub_question_ids.append(sub_question_id)
                existing.append(nc)

        return existing
