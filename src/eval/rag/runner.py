"""Runner — RAG 评测执行器（模块 7）。"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from src.eval.rag.models import (
    RagEvalCase,
    RagEvalReport,
    EvalCaseResult,
    STRATEGIES,
    RetrievalStrategy,
    GoldPaper,
    GoldEvidence,
    GoldClaim,
)
from src.eval.rag.metrics import compute_all_metrics
from src.eval.rag.report import RagEvalReporter

logger = logging.getLogger(__name__)


class RagEvalRunner:
    """
    RAG 评测执行器。

    负责：
    1. 加载评测用例（从文件或内联）
    2. 对每个 case 执行指定策略
    3. 调用 metrics 计算四层指标
    4. 生成报告

    使用方式：
        runner = RagEvalRunner(
            retriever=paper_retriever,
            deduper=paper_deduper,
            reranker=reranker,
            chunk_retriever=chunk_retriever,
        )
        report = runner.run(
            cases="smoke",
            strategies=["hybrid_basic", "keyword_only"],
            comparison_mode=True,
        )
    """

    def __init__(
        self,
        retriever=None,           # PaperRetriever
        deduper=None,             # PaperDeduper
        reranker=None,            # CrossEncoderReranker
        chunk_retriever=None,     # ChunkRetriever
        cases_dir: str | Path | None = None,
    ):
        self._retriever = retriever
        self._deduper = deduper
        self._reranker = reranker
        self._chunk_retriever = chunk_retriever
        self._cases_dir = Path(cases_dir) if cases_dir else self._default_cases_dir()
        self._reporter = RagEvalReporter()

    def _default_cases_dir(self) -> Path:
        """获取默认评测用例目录。"""
        return Path(__file__).parent.parent.parent.parent / "tests" / "eval" / "cases"

    # ── 公开 API ────────────────────────────────────────────────────────────

    def load_cases(
        self,
        source: str | list[RagEvalCase] | Path,
    ) -> list[RagEvalCase]:
        """
        加载评测用例。

        Args:
            source: "smoke" / "regression" / "full" / list[RagEvalCase] / Path

        Returns:
            RagEvalCase 列表
        """
        if isinstance(source, list):
            return source
        if isinstance(source, Path):
            return self._load_from_file(source)
        # Named preset
        preset_map = {
            "smoke": "phase2_smoke.jsonl",
            "regression": "scifact_regression.jsonl",
            "scifact_regression": "scifact_regression.jsonl",
            "full": "phase2_full.jsonl",
        }
        filename = preset_map.get(source, source)
        filepath = self._cases_dir / filename
        return self._load_from_file(filepath)

    def run(
        self,
        cases: str | list[RagEvalCase],
        strategies: list[str] | None = None,
        comparison_mode: bool = True,
        description: str = "",
        verbose: bool = False,
        case_ids: list[str] | None = None,
    ) -> RagEvalReport:
        """
        执行 RAG 评测。

        Args:
            cases: 评测用例来源（"smoke" / "regression" / list）
            strategies: 策略列表（None = ["hybrid_basic"]）
            comparison_mode: 是否启用策略比较
            description: 报告描述
            verbose: 是否包含完整 artifacts
            case_ids: 只运行指定 case

        Returns:
            RagEvalReport
        """
        strategies = strategies or ["hybrid_basic"]
        eval_cases = self.load_cases(cases)
        if case_ids:
            eval_cases = [c for c in eval_cases if c.case_id in case_ids]

        if not eval_cases:
            logger.warning("[RagEvalRunner] 没有找到评测用例")
            return RagEvalReport(description=description)

        all_results: list[EvalCaseResult] = []

        for strategy_name in strategies:
            strategy = STRATEGIES.get(strategy_name)
            if strategy is None:
                logger.warning(f"[RagEvalRunner] 未知策略 '{strategy_name}'，跳过")
                continue

            logger.info(
                f"[RagEvalRunner] 运行策略 '{strategy_name}'，"
                f"cases={len(eval_cases)}"
            )

            for case in eval_cases:
                try:
                    result = self.run_single_case(case, strategy, verbose=verbose)
                    result.strategy = strategy_name
                    all_results.append(result)
                except Exception as e:
                    logger.exception(f"[RagEvalRunner] case={case.case_id} 执行失败：{e}")
                    error_result = EvalCaseResult(
                        case_id=case.case_id,
                        strategy=strategy_name,
                        errors=[str(e)],
                        duration_ms=0.0,
                    )
                    all_results.append(error_result)

        # 生成报告
        report = self._reporter.build_report(
            case_results=all_results,
            strategies=strategies,
            comparison_mode=comparison_mode,
            description=description,
        )

        logger.info(
            f"[RagEvalRunner] 评测完成：{report.total_cases} cases，"
            f"{report.total_errors} errors"
        )
        return report

    def run_single_case(
        self,
        case: RagEvalCase,
        strategy: str | RetrievalStrategy,
        verbose: bool = False,
    ) -> EvalCaseResult:
        """
        执行单条评测用例。

        这是模块 7 的核心执行路径。
        """
        if isinstance(strategy, str):
            strategy = STRATEGIES.get(strategy)
            if strategy is None:
                return EvalCaseResult(
                    case_id=case.case_id,
                    strategy=str(strategy),
                    errors=[f"Unknown strategy: {strategy}"],
                )

        start = time.time()

        # Step 1: Paper Retrieval（模块 4）
        predicted_papers = self._retrieve_papers(case, strategy)

        # Step 2: Dedup + Rerank（模块 5）
        deduped = self._dedup_and_rank(case, predicted_papers, strategy)

        # Step 3: Evidence Retrieval（模块 6）
        predicted_chunks = self._retrieve_evidence(case, deduped, strategy)

        # Step 4: Citation / Grounding
        citations = []  # 简化：citation reachability 需要 resolved_report

        # Step 5: 计算四层指标
        result = compute_all_metrics(
            predicted_papers=deduped,
            predicted_chunks=predicted_chunks,
            citations=citations,
            gold=case,
        )

        result.case_id = case.case_id
        result.strategy = strategy.name
        result.duration_ms = (time.time() - start) * 1000

        return result

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _retrieve_papers(
        self,
        case: RagEvalCase,
        strategy: RetrievalStrategy,
    ) -> list:
        """Paper retrieval（模块 4）。"""
        if self._retriever is None:
            logger.warning("[RagEvalRunner] retriever 未配置，返回空列表")
            return []

        try:
            return self._retriever.search(
                query=case.query,
                sub_questions=case.sub_questions,
                recall_top_k=strategy.recall_top_k,
                keyword_weight=strategy.keyword_weight,
                dense_weight=strategy.dense_weight,
            )
        except Exception as e:
            logger.error(f"[RagEvalRunner] paper retrieval 失败：{e}")
            return []

    def _dedup_and_rank(
        self,
        case: RagEvalCase,
        predicted_papers: list,
        strategy: RetrievalStrategy,
    ) -> list:
        """Dedup + Rerank（模块 5）。"""
        # Dedup
        if self._deduper:
            try:
                deduped = self._deduper.dedup(predicted_papers)
            except Exception as e:
                logger.warning(f"[RagEvalRunner] dedup 失败：{e}")
                deduped = predicted_papers
        else:
            deduped = predicted_papers

        # Rerank
        if strategy.rerank_enabled and self._reranker:
            try:
                reranked = self._reranker.rerank_with_fusion(
                    query=case.query,
                    candidates=deduped,
                    fusion_weights=(
                        strategy.fusion_weights_rrf,
                        strategy.fusion_weights_rerank,
                    ),
                )
                return reranked
            except Exception as e:
                logger.warning(f"[RagEvalRunner] rerank 失败：{e}")

        # 退化为 RRF 排序
        return sorted(deduped, key=lambda c: c.rrf_score, reverse=True)

    def _retrieve_evidence(
        self,
        case: RagEvalCase,
        deduped_papers: list,
        strategy: RetrievalStrategy,
    ) -> list:
        """Evidence retrieval（模块 6）。"""
        if not strategy.evidence_recall_enabled:
            return []
        if self._chunk_retriever is None:
            return []

        paper_ids = []
        for p in deduped_papers:
            pid = getattr(p, "primary_doc_id", None) or getattr(p, "doc_id", None)
            if pid:
                paper_ids.append(pid)

        if not paper_ids:
            return []

        try:
            return self._chunk_retriever.retrieve(
                paper_ids=paper_ids,
                query=case.query,
                sub_questions=[{"id": f"sq-{i}", "text": sq} for i, sq in enumerate(case.sub_questions)],
                top_k_global=strategy.evidence_top_k,
            )
        except Exception as e:
            logger.warning(f"[RagEvalRunner] evidence retrieval 失败：{e}")
            return []

    def _load_from_file(self, filepath: Path) -> list[RagEvalCase]:
        """从 JSONL 文件加载评测用例。"""
        if not filepath.exists():
            logger.warning(f"[RagEvalRunner] 评测用例文件不存在：{filepath}")
            return []

        cases = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    cases.append(self._dict_to_case(data))
        except Exception as e:
            logger.error(f"[RagEvalRunner] 加载评测用例失败：{e}")

        return cases

    def _dict_to_case(self, data: dict) -> RagEvalCase:
        """将 dict 转换为 RagEvalCase。"""
        gold_papers = [
            GoldPaper(
                title=p.get("title", ""),
                canonical_id=p.get("canonical_id", ""),
                arxiv_id=p.get("arxiv_id", ""),
                expected_rank=p.get("expected_rank", 0),
            )
            for p in data.get("gold_papers", [])
        ]

        gold_evidence = [
            GoldEvidence(
                paper_title=e.get("paper_title", ""),
                expected_section=e.get("expected_section", ""),
                text_hint=e.get("text_hint", ""),
                sub_question_id=e.get("sub_question_id", ""),
                expected_support_type=e.get("expected_support_type", ""),
            )
            for e in data.get("gold_evidence", [])
        ]

        gold_claims = [
            GoldClaim(
                claim_text=c.get("claim_text", ""),
                supported_by_paper=c.get("supported_by_paper", ""),
                supported_by_evidence_section=c.get("supported_by_evidence_section", ""),
            )
            for c in data.get("gold_claims", [])
        ]

        return RagEvalCase(
            case_id=data.get("case_id", ""),
            query=data.get("query", ""),
            sub_questions=data.get("sub_questions", []),
            gold_papers=gold_papers,
            gold_evidence=gold_evidence,
            gold_claims=gold_claims,
            recall_top_k=data.get("recall_top_k", 100),
            rerank_top_m=data.get("rerank_top_m", 50),
            evidence_top_k=data.get("evidence_top_k", 50),
            source=data.get("source", "manual"),
            notes=data.get("notes", ""),
        )
