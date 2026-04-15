"""Evals API — RAG 评测端点（模块 7）。"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evals", tags=["evals"])


# ── 请求模型 ──────────────────────────────────────────────────────────────────


class RagEvalRunRequest(BaseModel):
    """POST /evals/rag/run 请求体。"""
    case_source: str = Field(
        default="smoke",
        description="评测用例来源：smoke / regression / full / inline",
    )
    case_ids: list[str] = Field(
        default_factory=list,
        description="只运行指定的 case_ids（空=运行全部）",
    )
    strategies: list[str] = Field(
        default_factory=lambda: ["hybrid_basic"],
        description="要评测的策略列表",
    )
    comparison_mode: bool = Field(
        default=True,
        description="是否启用策略比较模式",
    )
    verbose: bool = Field(
        default=False,
        description="是否包含完整 artifacts（会增加响应大小）",
    )
    description: str = Field(
        default="",
        description="评测描述",
    )


# ── 响应模型 ──────────────────────────────────────────────────────────────────


class MetricSummaryResponse(BaseModel):
    """指标摘要（用于 API 响应）。"""
    paper_recall_50: float = 0.0
    paper_mrr: float = 0.0
    evidence_recall_25: float = 0.0
    grounding_score: float = 0.0


class CaseResultResponse(BaseModel):
    """单条 case 结果。"""
    case_id: str
    strategy: str
    success: bool
    duration_ms: float
    metrics: MetricSummaryResponse
    errors: list[str] = Field(default_factory=list)


class StrategyMetricsResponse(BaseModel):
    """策略聚合指标。"""
    strategy_name: str
    total_cases: int = 0
    success_cases: int = 0
    avg_paper_recall_50: float = 0.0
    avg_paper_mrr: float = 0.0
    avg_evidence_recall_25: float = 0.0
    avg_grounding_score: float = 0.0


class RagEvalRunResponse(BaseModel):
    """POST /evals/rag/run 响应体。"""
    report_id: str = ""
    description: str = ""
    generated_at: str = ""
    total_cases: int = 0
    total_errors: int = 0
    success_rate: float = 0.0
    overall_metrics: MetricSummaryResponse
    strategy_comparison: Optional[dict] = None
    case_results: list[CaseResultResponse] = Field(default_factory=list)
    duration_ms: float = 0.0


class EvalCaseListItem(BaseModel):
    """评测用例列表项。"""
    case_id: str
    query: str
    source: str
    gold_paper_count: int = 0
    gold_evidence_count: int = 0
    notes: str = ""


class EvalCaseListResponse(BaseModel):
    """GET /evals/rag/cases 响应。"""
    cases: list[EvalCaseListItem]
    total: int = 0


# ── 路由处理函数 ──────────────────────────────────────────────────────────────


@router.post("/rag/run", response_model=RagEvalRunResponse)
async def run_rag_eval(req: RagEvalRunRequest) -> RagEvalRunResponse:
    """
    运行 RAG 评测。

    支持多策略对比，返回四层指标报告。
    """
    import time
    from src.eval.rag.runner import RagEvalRunner

    start = time.time()

    try:
        runner = RagEvalRunner()
        report = runner.run(
            cases=req.case_source,
            strategies=req.strategies,
            comparison_mode=req.comparison_mode,
            description=req.description,
            verbose=req.verbose,
            case_ids=req.case_ids or None,
        )
    except Exception as e:
        logger.exception("[/evals/rag/run] 评测执行失败")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    # 构建响应
    case_results = [
        CaseResultResponse(
            case_id=r.case_id,
            strategy=r.strategy,
            success=r.is_success(),
            duration_ms=r.duration_ms,
            metrics=MetricSummaryResponse(
                paper_recall_50=(
                    r.paper_retrieval.paper_recall_50
                    if r.paper_retrieval else 0.0
                ),
                paper_mrr=(
                    r.paper_retrieval.paper_mrr
                    if r.paper_retrieval else 0.0
                ),
                evidence_recall_25=(
                    r.evidence_retrieval.evidence_recall_25
                    if r.evidence_retrieval else 0.0
                ),
                grounding_score=(
                    r.citation_grounding.grounding_score
                    if r.citation_grounding else 0.0
                ),
            ),
            errors=r.errors,
        )
        for r in report.case_results
    ]

    strategy_comparison_dict = None
    if report.strategy_comparison:
        sc = report.strategy_comparison
        strategy_comparison_dict = {
            "strategies": sc.strategies,
            "winner": sc.winner,
            "best_per_metric": sc.best_per_metric,
            "per_strategy": {
                name: StrategyMetricsResponse(
                    strategy_name=m.strategy_name,
                    total_cases=m.total_cases,
                    success_cases=m.success_cases,
                    avg_paper_recall_50=m.avg_paper_recall_50,
                    avg_paper_mrr=m.avg_paper_mrr,
                    avg_evidence_recall_25=m.avg_evidence_recall_25,
                    avg_grounding_score=m.avg_grounding_score,
                ).model_dump()
                for name, m in sc.per_strategy_metrics.items()
            },
        }

    duration_ms = (time.time() - start) * 1000

    return RagEvalRunResponse(
        report_id=report.report_id or "",
        description=report.description,
        generated_at=report.generated_at,
        total_cases=report.total_cases,
        total_errors=report.total_errors,
        success_rate=report.success_rate,
        overall_metrics=MetricSummaryResponse(
            paper_recall_50=report.overall_paper_recall_50,
            paper_mrr=report.overall_paper_mrr,
            evidence_recall_25=report.overall_evidence_recall_25,
            grounding_score=report.overall_grounding_score,
        ),
        strategy_comparison=strategy_comparison_dict,
        case_results=case_results,
        duration_ms=duration_ms,
    )


@router.get("/rag/cases", response_model=EvalCaseListResponse)
async def list_eval_cases(source: str = "smoke") -> EvalCaseListResponse:
    """
    返回可用评测用例列表。

    Query 参数：
        source: smoke / regression / full
    """
    from src.eval.rag.runner import RagEvalRunner

    try:
        runner = RagEvalRunner()
        cases = runner.load_cases(source)
    except Exception as e:
        logger.error(f"[/evals/rag/cases] 加载用例失败：{e}")
        cases = []

    items = [
        EvalCaseListItem(
            case_id=c.case_id,
            query=c.query[:100] + ("..." if len(c.query) > 100 else ""),
            source=c.source,
            gold_paper_count=len(c.gold_papers),
            gold_evidence_count=len(c.gold_evidence),
            notes=c.notes,
        )
        for c in cases
    ]

    return EvalCaseListResponse(cases=items, total=len(items))
