"""Corpus Evidence API — 模块 6 fine chunk evidence retrieval 端点。"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/corpus", tags=["corpus"])


# ── 请求模型 ──────────────────────────────────────────────────────────────────


class EvidenceSearchRequest(BaseModel):
    """POST /corpus/evidence 请求体。"""
    # 论文范围（必须）
    paper_ids: list[str] = Field(
        ...,
        description="限定检索的论文 doc_ids 列表（来自 /corpus/search 的结果）",
    )
    canonical_ids: list[str] = Field(
        default_factory=list,
        description="可选的 canonical_id 列表（用于过滤同一论文簇）",
    )
    # 检索内容
    query: str = Field(..., min_length=1, max_length=2000)
    sub_questions: list[str] = Field(
        default_factory=list,
        description="SearchPlan 中的子问题列表",
    )
    # 检索参数
    top_k_per_paper: int = Field(
        default=10, ge=1, le=50,
        description="每篇论文最多召回的 chunks 数量",
    )
    top_k_global: int = Field(
        default=50, ge=1, le=200,
        description="全局最多召回的 chunks 总数",
    )
    enable_typing: bool = Field(
        default=True,
        description="是否启用 evidence typing",
    )


# ── 响应模型 ──────────────────────────────────────────────────────────────────


class ChunkScoreResponse(BaseModel):
    """Chunk 分数明细。"""
    keyword_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0


class EvidenceChunkResponse(BaseModel):
    """Evidence chunk 响应。"""
    chunk_id: str = ""
    paper_id: str = ""
    canonical_id: str = ""
    text: str = ""
    section: str = ""
    page_start: int = 1
    page_end: int = 1
    scores: ChunkScoreResponse = Field(default_factory=ChunkScoreResponse)
    support_type: str = "claim_support"
    matched_query: str = ""
    sub_question_id: str = ""
    chunk_path: str = "keyword"


class RagResultResponse(BaseModel):
    """RagResult 响应（精简版）。"""
    query: str = ""
    sub_questions: list[str] = Field(default_factory=list)
    rag_strategy: str = "keyword+dense+rrf+evidence_typing"
    total_papers: int = 0
    total_chunks: int = 0
    coverage_notes: list[str] = Field(default_factory=list)
    retrieved_at: str = ""


class EvidenceSearchResponse(BaseModel):
    """POST /corpus/evidence 响应体。"""
    rag_result: RagResultResponse
    chunks: list[EvidenceChunkResponse]
    total_chunks: int = 0
    duration_ms: float = 0.0


# ── 路由处理函数 ──────────────────────────────────────────────────────────────


@router.post("/evidence", response_model=EvidenceSearchResponse)
async def corpus_evidence(
    req: EvidenceSearchRequest,
) -> EvidenceSearchResponse:
    """
    Fine chunk evidence retrieval：在已选论文内检索 evidence chunks。

    流水线：
        scope restriction → fine chunk recall (keyword+dense)
        → RRF merge → evidence typing → RagResult build

    Args:
        req: EvidenceSearchRequest，包含 paper_ids、query、检索参数等

    Returns:
        EvidenceSearchResponse，包含检索到的 chunks 和 RagResult
    """
    start = time.time()

    # 校验 paper_ids
    if not req.paper_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="paper_ids 不能为空",
        )

    try:
        # Step 1: 初始化 ChunkStore
        from src.corpus.store import ChunkStore
        chunk_store = ChunkStore()

        # Step 2: ChunkRetriever — fine chunk evidence retrieval
        from src.corpus.search.retrievers.chunk_retriever import ChunkRetriever

        retriever = ChunkRetriever(
            chunk_store=chunk_store,
            milvus_client=None,
            embedding_model=None,
        )
        chunk_results = retriever.retrieve(
            paper_ids=req.paper_ids,
            query=req.query,
            sub_questions=req.sub_questions,
            top_k_per_paper=req.top_k_per_paper,
            top_k_global=req.top_k_global,
        )

        # Step 3: Evidence Typing（可选）
        coverage_notes: list[str] = []
        if req.enable_typing and chunk_results:
            from src.corpus.search.evidence_typer import EvidenceTyper
            typer = EvidenceTyper()
            typer.annotate_chunks(chunk_results)

            # 生成覆盖度报告
            seen_types = {getattr(c, "support_type", "claim_support") for c in chunk_results}
            for stype in ["method", "result"]:
                if stype not in seen_types:
                    coverage_notes.append(f"NOTE: 缺少 {stype} 类型 evidence，可能影响报告完整性")

        if len(chunk_results) < 5:
            coverage_notes.append(
                f"NOTE: evidence chunks 较少（{len(chunk_results)}），覆盖率可能不足"
            )

        # Step 4: RagResultBuilder
        from src.corpus.search.result_builder import RagResultBuilder

        builder = RagResultBuilder()
        rag_result = (
            builder
            .with_query(req.query)
            .with_sub_questions(req.sub_questions)
            .with_evidence_chunks(chunk_results)
            .with_rag_strategy("keyword+dense+rrf+evidence_typing")
            .build()
        )

        # Step 5: 构建响应
        chunk_responses = [
            EvidenceChunkResponse(
                chunk_id=r.chunk_id,
                paper_id=r.paper_id,
                canonical_id=r.canonical_id,
                text=r.text[:1000] if r.text else "",
                section=r.section,
                page_start=r.page_start,
                page_end=r.page_end,
                scores=ChunkScoreResponse(
                    keyword_score=r.scores.keyword_score,
                    dense_score=r.scores.dense_score,
                    rrf_score=r.scores.rrf_score,
                ),
                support_type=getattr(r, "support_type", "claim_support"),
                matched_query=getattr(r, "matched_query", ""),
                sub_question_id=getattr(r, "sub_question_id", ""),
                chunk_path=getattr(r, "chunk_path", "keyword"),
            )
            for r in chunk_results
        ]

        # 合并 coverage_notes
        if rag_result and hasattr(rag_result, "coverage_notes"):
            coverage_notes = list(rag_result.coverage_notes) + coverage_notes

        retrieved_at = ""
        if rag_result and hasattr(rag_result, "retrieved_at"):
            retrieved_at = rag_result.retrieved_at or ""

        rag_result_response = RagResultResponse(
            query=req.query,
            sub_questions=req.sub_questions,
            rag_strategy="keyword+dense+rrf+evidence_typing",
            total_papers=len(req.paper_ids),
            total_chunks=len(chunk_results),
            coverage_notes=coverage_notes,
            retrieved_at=retrieved_at,
        )

        duration_ms = (time.time() - start) * 1000

        return EvidenceSearchResponse(
            rag_result=rag_result_response,
            chunks=chunk_responses,
            total_chunks=len(chunk_results),
            duration_ms=duration_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[/corpus/evidence] evidence 检索失败：{e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
