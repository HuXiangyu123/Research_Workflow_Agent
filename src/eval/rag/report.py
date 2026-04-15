"""Report — RAG 评测报告生成（模块 7）。"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from src.eval.rag.models import (
    RagEvalReport,
    EvalCaseResult,
    StrategyComparisonReport,
    StrategyMetrics,
)

logger = logging.getLogger(__name__)


def _mean(values: list[float]) -> float:
    """计算平均值。"""
    return sum(values) / len(values) if values else 0.0


class RagEvalReporter:
    """
    RAG 评测报告生成器。

    使用方式：
        reporter = RagEvalReporter()
        report = reporter.build_report(
            case_results=[...],
            strategies=["hybrid_basic", "keyword_only"],
            description="Phase 2 smoke test",
        )
    """

    def build_report(
        self,
        case_results: list[EvalCaseResult],
        strategies: list[str] | None = None,
        comparison_mode: bool = True,
        description: str = "",
    ) -> RagEvalReport:
        """
        构建完整评测报告。

        Args:
            case_results: 所有 case 的执行结果
            strategies: 要评测的策略列表
            comparison_mode: 是否生成策略比较报告
            description: 报告描述

        Returns:
            RagEvalReport
        """
        report_id = self._generate_report_id()
        now = self._now_iso()

        # 聚合 case 统计
        failed_cases = [r.case_id for r in case_results if not r.is_success()]
        total_errors = len(failed_cases)

        # 过滤成功 case 计算全局指标
        success_results = [r for r in case_results if r.is_success()]
        overall_pr50 = _mean([r.paper_retrieval.paper_recall_50 for r in success_results if r.paper_retrieval])
        overall_mrr = _mean([r.paper_retrieval.paper_mrr for r in success_results if r.paper_retrieval])
        overall_er25 = _mean([r.evidence_retrieval.evidence_recall_25 for r in success_results if r.evidence_retrieval])
        overall_gs = _mean([r.citation_grounding.grounding_score for r in success_results if r.citation_grounding])

        # 策略比较
        strategy_comparison: StrategyComparisonReport | None = None
        if comparison_mode and strategies:
            strategy_comparison = self._build_comparison_report(
                case_results, strategies
            )

        report = RagEvalReport(
            report_id=report_id,
            description=description,
            generated_at=now,
            case_results=case_results,
            failed_cases=failed_cases,
            total_cases=len(case_results),
            total_errors=total_errors,
            total_duration_ms=sum(r.duration_ms for r in case_results),
            success_rate=len(success_results) / max(len(case_results), 1),
            overall_paper_recall_50=overall_pr50,
            overall_paper_mrr=overall_mrr,
            overall_evidence_recall_25=overall_er25,
            overall_grounding_score=overall_gs,
            strategy_comparison=strategy_comparison,
        )

        logger.info(
            f"[RagEvalReporter] 报告生成完毕：{report_id}，"
            f"cases={report.total_cases} errors={report.total_errors}"
        )
        return report

    def _build_comparison_report(
        self,
        case_results: list[EvalCaseResult],
        strategies: list[str],
    ) -> StrategyComparisonReport:
        """构建多策略比较报告。"""
        # 按策略分组
        by_strategy: dict[str, list[EvalCaseResult]] = {}
        for r in case_results:
            s = r.strategy or "hybrid_basic"
            by_strategy.setdefault(s, []).append(r)

        # 计算每个策略的聚合指标
        per_strategy: dict[str, StrategyMetrics] = {}
        for strategy_name in strategies:
            results = by_strategy.get(strategy_name, [])
            metrics = StrategyMetrics(strategy_name=strategy_name)
            metrics.aggregate(results)
            per_strategy[strategy_name] = metrics

        # 找出每个指标的最优策略
        best_per_metric = self._find_best_per_metric(per_strategy)

        # 综合 winner（基于 evidence_recall_25 + grounding_score）
        winner = self._determine_winner(per_strategy)

        return StrategyComparisonReport(
            strategies=strategies,
            per_strategy_metrics=per_strategy,
            best_per_metric=best_per_metric,
            winner=winner,
            generated_at=self._now_iso(),
        )

    def _find_best_per_metric(
        self,
        per_strategy: dict[str, StrategyMetrics],
    ) -> dict[str, str]:
        """找出每个指标的最优策略。"""
        metrics_to_compare = [
            ("avg_paper_recall_50", "paper_recall_50"),
            ("avg_paper_mrr", "paper_mrr"),
            ("avg_evidence_recall_25", "evidence_recall_25"),
            ("avg_grounding_score", "grounding_score"),
            ("avg_evidence_precision_25", "evidence_precision"),
        ]

        best: dict[str, str] = {}
        for attr, metric_name in metrics_to_compare:
            best_val = -1.0
            best_strategy = ""
            for name, m in per_strategy.items():
                val = getattr(m, attr, 0.0)
                if val > best_val:
                    best_val = val
                    best_strategy = name
            if best_strategy:
                best[metric_name] = best_strategy
        return best

    def _determine_winner(
        self,
        per_strategy: dict[str, StrategyMetrics],
    ) -> str:
        """基于 evidence_recall_25 + grounding_score 确定 winner。"""
        scores: dict[str, float] = {}
        for name, m in per_strategy.items():
            score = (
                0.5 * m.avg_evidence_recall_25
                + 0.5 * m.avg_grounding_score
            )
            scores[name] = score

        if not scores:
            return ""
        return max(scores, key=scores.get)

    def generate_markdown(self, report: RagEvalReport) -> str:
        """
        将报告生成为 Markdown 格式（便于在 UI 中展示）。

        Args:
            report: RagEvalReport

        Returns:
            Markdown 格式的报告文本
        """
        lines = [
            f"# RAG Eval Report: {report.report_id}",
            "",
            f"**生成时间**: {report.generated_at}",
            f"**描述**: {report.description}",
            f"**Case 总数**: {report.total_cases}",
            f"**成功**: {report.total_cases - report.total_errors} ({report.success_rate:.1%})",
            f"**失败**: {report.total_errors}",
            "",
            "## 整体指标",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| Paper Recall@50 | {report.overall_paper_recall_50:.3f} |",
            f"| Paper MRR | {report.overall_paper_mrr:.3f} |",
            f"| Evidence Recall@25 | {report.overall_evidence_recall_25:.3f} |",
            f"| Grounding Score | {report.overall_grounding_score:.3f} |",
            "",
        ]

        # 策略比较
        if report.strategy_comparison:
            sc = report.strategy_comparison
            lines += [
                "## 策略比较",
                "",
                f"**Winner**: `{sc.winner}`",
                "",
                "| 策略 | Recall@50 | MRR | Ev Recall@25 | Grounding |",
                "|------|-----------|-----|------------|-----------|",
            ]
            for name, m in sc.per_strategy_metrics.items():
                winner_marker = "🏆" if name == sc.winner else ""
                lines.append(
                    f"| {name} {winner_marker} | "
                    f"{m.avg_paper_recall_50:.3f} | "
                    f"{m.avg_paper_mrr:.3f} | "
                    f"{m.avg_evidence_recall_25:.3f} | "
                    f"{m.avg_grounding_score:.3f} |"
                )

            lines += ["", "### 各指标最优策略", ""]
            for metric, best in sc.best_per_metric.items():
                lines.append(f"- **{metric}**: `{best}`")

        # Case 详情
        if report.case_results:
            lines += ["", "## Case 详情", ""]
            for r in report.case_results:
                status = "✅" if r.is_success() else "❌"
                lines += [
                    f"### {status} {r.case_id}",
                    f"**策略**: `{r.strategy}`",
                    "",
                    f"- Paper Recall@50: {r.paper_retrieval.paper_recall_50:.3f}",
                    f"- Paper MRR: {r.paper_retrieval.paper_mrr:.3f}",
                    f"- Evidence Recall@25: {r.evidence_retrieval.evidence_recall_25:.3f}",
                    f"- Grounding Score: {r.citation_grounding.grounding_score:.3f}",
                    f"- 耗时: {r.duration_ms:.1f}ms",
                    "",
                ]
                if r.errors:
                    lines.append(f"**错误**: {', '.join(r.errors)}")
                    lines.append("")

        return "\n".join(lines)

    def _generate_report_id(self) -> str:
        """生成唯一报告 ID。"""
        return f"rag-eval-{uuid.uuid4().hex[:8]}"

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
