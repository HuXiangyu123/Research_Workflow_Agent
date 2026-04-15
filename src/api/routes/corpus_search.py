"""Corpus Search API — 模块 4 paper-level retrieval 端点。"""
from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/corpus", tags=["corpus"])


# ── 请求模型 ──────────────────────────────────────────────────────────────────


class SearchFilters(BaseModel):
    """元数据过滤条件。"""
    year_range: Optional[tuple[int, int]] = Field(
        default=None,
        description="年份范围，如 (2020, 2025)",
    )
    source_type: Optional[list[str]] = Field(
        default=None,
        description="来源类型：arxiv / uploaded_pdf / online_url",
    )
    venue: Optional[list[str]] = Field(
        default=None,
        description="会议/期刊名称",
    )
    canonical_id: Optional[list[str]] = Field(
        default=None,
        description="限定特定论文 canonical_id 列表",
    )


class CorpusSearchRequest(BaseModel):
    """POST /corpus/search 请求体。"""
    query: str = Field(..., min_length=1, max_length=2000)
    sub_questions: list[str] = Field(
        default_factory=list,
        description="SearchPlan 中的子问题列表",
    )
    filters: Optional[SearchFilters] = Field(default=None)
    top_k: int = Field(default=100, ge=1, le=500, description="返回的候选论文数量")
    recall_top_k: int = Field(
        default=100, ge=1, le=500,
        description="各召回路径返回的数量（keyword + dense）",
    )
    embedding: Optional[list[float]] = Field(
        default=None,
        description="可选的 query embedding（无则自动生成）",
    )
    # 模块 5 参数
    enable_rerank: bool = Field(
        default=True,
        description="是否启用本地 Cross-Encoder Rerank",
    )
    rerank_top_m: int = Field(
        default=50, ge=5, le=200,
        description="进入 rerank 的候选数量",
    )


# ── 响应模型 ──────────────────────────────────────────────────────────────────


class RecallResultResponse(BaseModel):
    """单条召回结果。"""
    chunk_id: str
    doc_id: str
    canonical_id: Optional[str] = None
    score: float
    path: str
    section: str = ""
    text: str = ""


class MergedCandidateResponse(BaseModel):
    """合并后候选论文。"""
    doc_id: str
    canonical_id: Optional[str] = None
    title: str = ""
    authors: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    abstract: Optional[str] = None
    source_type: str = ""
    rrf_score: float = 0.0
    keyword_score: Optional[float] = None
    dense_score: Optional[float] = None
    recall_results: list[RecallResultResponse] = Field(default_factory=list)


class RetrievalTraceResponse(BaseModel):
    """检索轨迹。"""
    query: str
    sub_question_id: Optional[str] = None
    retrieval_path: str
    target_index: str
    filter_summary: str
    top_k_requested: int
    returned_doc_ids: list[str]
    returned_chunk_ids: list[str]
    returned_count: int
    duration_ms: float
    error: Optional[str] = None


# ── 模块 5 新响应模型 ─────────────────────────────────────────────────────────


class DedupInfoResponse(BaseModel):
    """去重信息。"""
    is_canonical_representative: bool = True
    merged_doc_ids: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)


class ScoreBreakdownResponse(BaseModel):
    """分数明细。"""
    rrf_score: float = 0.0
    keyword_score: float = 0.0
    dense_score: float = 0.0
    rerank_score: Optional[float] = None
    final_score: float = 0.0


class PaperCandidateResponse(BaseModel):
    """
    最终论文候选响应（模块 5 输出）。

    这是新版响应模型，替代 MergedCandidateResponse。
    包含 dedup 信息和完整的分数 breakdown。
    """
    paper_id: str = ""
    canonical_id: str = ""
    title: str = ""
    authors: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    abstract: Optional[str] = None
    source_refs: list[str] = Field(default_factory=list)
    primary_doc_id: str = ""
    scores: ScoreBreakdownResponse = Field(default_factory=ScoreBreakdownResponse)
    matched_queries: list[str] = Field(default_factory=list)
    matched_paths: list[str] = Field(default_factory=list)
    why_retrieved: str = ""
    dedup_info: DedupInfoResponse = Field(default_factory=DedupInfoResponse)


class CorpusSearchResponse(BaseModel):
    """POST /corpus/search 响应体。"""
    # 模块 5 结果（PaperCandidate）
    candidates: list[PaperCandidateResponse]
    # 兼容旧版（MergedCandidate）
    legacy_candidates: list[MergedCandidateResponse] = Field(default_factory=list)
    trace: list[RetrievalTraceResponse]
    total_candidates: int = 0
    merged_count: int = 0
    rerank_count: int = 0  # 实际进入 rerank 的数量
    duration_ms: float = 0.0


# ── 路由处理函数 ──────────────────────────────────────────────────────────────


def _path_to_str(path) -> str:
    """将 RetrievalPath Enum 或字符串转换为字符串。"""
    if hasattr(path, "value"):
        return path.value
    return str(path)


@router.post("/search", response_model=CorpusSearchResponse)
async def corpus_search(req: CorpusSearchRequest) -> CorpusSearchResponse:
    """
    Paper-level 检索：高召回论文候选池 + 检索轨迹。

    流水线：
        query preparation → hybrid recall → candidate merge → trace build
    """
    start = time.time()

    # 构建 sub_questions 格式（与 search_papers_ex 兼容）
    sub_question_list: list[dict] = [
        {"id": f"sq-{i}", "text": sq}
        for i, sq in enumerate(req.sub_questions)
    ] if req.sub_questions else []

    # 提取 filters
    year_range = req.filters.year_range if req.filters else None
    sources = req.filters.source_type if req.filters else None
    venues = req.filters.venue if req.filters else None

    try:
        from src.corpus.store.repository import CorpusRepository
        repo = CorpusRepository()
    except Exception as e:
        logger.exception("[/corpus/search] CorpusRepository 初始化失败")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"CorpusRepository 初始化失败：{e}",
        )

    try:
        result = repo.search_papers_ex(
            query=req.query,
            sub_questions=sub_question_list,
            year_range=year_range,
            sources=sources,
            venues=venues,
            keyword_top_k=req.recall_top_k,
            dense_top_k=req.recall_top_k,
            top_k=req.top_k,
            embedding=req.embedding,
        )
    except Exception as e:
        logger.exception(f"[/corpus/search] 检索失败：{e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    # ── 模块 5：Dedup + Rerank ──────────────────────────────────────────────
    dedup_count = 0
    rerank_count = 0
    final_candidates: list[PaperCandidateResponse] = []

    if result.candidates:
        from src.corpus.search.deduper import PaperDeduper
        from src.corpus.search.candidate_builder import CandidateBuilder

        # 1. Canonical Dedup
        deduper = PaperDeduper(db_session=None)
        deduped = deduper.dedup(result.candidates)
        dedup_count = len(deduped)

        # 2. RRF Budget Trim
        rerank_pool = sorted(
            deduped, key=lambda c: c.rrf_score, reverse=True
        )[:req.rerank_top_m]

        # 3. Cross-Encoder Rerank
        if req.enable_rerank:
            from src.corpus.search.reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker()
            reranked = reranker.rerank_with_fusion(
                query=req.query,
                candidates=rerank_pool,
                fusion_weights=(0.4, 0.6),
                top_n=req.rerank_top_m,
            )
        else:
            # 跳过 rerank，退化为 RRF 排序
            for c in rerank_pool:
                c.final_score = c.rrf_score
            reranked = rerank_pool

        rerank_count = len(reranked)

        # 4. 构建最终 PaperCandidate
        builder = CandidateBuilder()
        paper_candidates = builder.build(deduped, top_k=req.top_k)

        # 转换为响应模型
        final_candidates = [
            PaperCandidateResponse(
                paper_id=pc.paper_id,
                canonical_id=pc.canonical_id,
                title=pc.title,
                authors=", ".join(pc.authors) if pc.authors else None,
                year=pc.year,
                venue=pc.venue,
                abstract=pc.abstract,
                source_refs=pc.source_refs,
                primary_doc_id=pc.primary_doc_id,
                scores=ScoreBreakdownResponse(
                    rrf_score=pc.scores.rrf_score,
                    keyword_score=pc.scores.keyword_score,
                    dense_score=pc.scores.dense_score,
                    rerank_score=pc.scores.rerank_score,
                    final_score=pc.scores.final_score,
                ),
                matched_queries=pc.matched_queries,
                matched_paths=pc.matched_paths,
                why_retrieved=pc.why_retrieved,
                dedup_info=DedupInfoResponse(
                    is_canonical_representative=True,
                    merged_doc_ids=[],
                    source_refs=pc.source_refs,
                ),
            )
            for pc in paper_candidates
        ]

    # 转换 MergedCandidate → MergedCandidateResponse（兼容旧版）
    candidates = [
        MergedCandidateResponse(
            doc_id=c.doc_id,
            canonical_id=c.canonical_id or None,
            title=c.title,
            authors=", ".join(c.authors) if c.authors else None,
            year=c.year,
            venue=c.venue,
            abstract=c.abstract,
            source_type=getattr(c, "source_uri", "") or "",
            rrf_score=c.rrf_score,
            keyword_score=c.keyword_score or None,
            dense_score=c.dense_score or None,
            recall_results=[
                RecallResultResponse(
                    chunk_id=ev.chunk_id,
                    doc_id=ev.doc_id,
                    canonical_id=ev.canonical_id or None,
                    score=ev.score,
                    path=_path_to_str(ev.path),
                    section=ev.section,
                    text=ev.text[:500] if ev.text else "",
                )
                for ev in c.recall_evidence
            ],
        )
        for c in result.candidates
    ]

    # 转换 RetrievalTrace → RetrievalTraceResponse
    traces = [
        RetrievalTraceResponse(
            query=t.query,
            sub_question_id=t.sub_question_id,
            retrieval_path=_path_to_str(t.retrieval_path),
            target_index=t.target_index,
            filter_summary=t.filter_summary,
            top_k_requested=t.top_k_requested,
            returned_doc_ids=t.returned_doc_ids,
            returned_chunk_ids=t.returned_chunk_ids,
            returned_count=t.returned_count,
            duration_ms=t.duration_ms,
            error=t.error,
        )
        for t in result.traces
    ]

    return CorpusSearchResponse(
        candidates=final_candidates,    # 模块 5 结果
        legacy_candidates=candidates,    # 兼容旧版
        trace=traces,
        total_candidates=result.total_candidates,
        merged_count=dedup_count,
        rerank_count=rerank_count,
        duration_ms=(time.time() - start) * 1000,
    )
