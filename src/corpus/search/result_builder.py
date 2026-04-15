"""RagResultBuilder — 构建结构化 RagResult（模块 6）。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.paper import RagResult, RerankLog

logger = logging.getLogger(__name__)


class RagResultBuilder:
    """
    将检索结果构建为结构化 RagResult。

    使用 builder 模式，便于逐步组装复杂对象。

    使用方式：
        builder = RagResultBuilder()
        result = (
            builder
            .with_query("multi-agent systems")
            .with_sub_questions([...])
            .with_paper_candidates([...])
            .with_evidence_chunks([...])
            .with_traces([...])
            .with_dedup_logs([...])
            .with_rerank_logs([...])
            .build()
        )
    """

    def __init__(self):
        self._query: str = ""
        self._sub_questions: list[str] = []
        self._paper_candidates: list = []
        self._evidence_chunks: list = []
        self._traces: list = []
        self._dedup_logs: list = []
        self._rerank_logs: list = []
        self._rag_strategy: str = "keyword+dense+rrf+evidence_typing"

    # ── Builder 接口 ──────────────────────────────────────────────────────────

    def with_query(self, query: str) -> "RagResultBuilder":
        """设置检索查询。"""
        self._query = query
        return self

    def with_sub_questions(self, sub_questions: list[str]) -> "RagResultBuilder":
        """设置子问题列表。"""
        self._sub_questions = sub_questions
        return self

    def with_paper_candidates(
        self, candidates: list
    ) -> "RagResultBuilder":
        """设置候选论文列表。"""
        self._paper_candidates = candidates
        return self

    def with_evidence_chunks(
        self, chunks: list
    ) -> "RagResultBuilder":
        """设置证据块列表。"""
        self._evidence_chunks = chunks
        return self

    def with_traces(self, traces: list) -> "RagResultBuilder":
        """设置检索轨迹列表。"""
        self._traces = traces
        return self

    def with_dedup_logs(self, logs: list) -> "RagResultBuilder":
        """设置去重日志列表。"""
        self._dedup_logs = logs
        return self

    def with_rerank_logs(self, logs: list) -> "RagResultBuilder":
        """设置重排日志列表。"""
        self._rerank_logs = logs
        return self

    def with_rag_strategy(self, strategy: str) -> "RagResultBuilder":
        """设置 RAG 策略名称。"""
        self._rag_strategy = strategy
        return self

    def build(self) -> "RagResult":
        """
        构建最终的 RagResult。

        Returns:
            RagResult: 结构化检索结果，如果导入失败则返回 None
        """
        coverage_notes = self._generate_coverage_notes()

        try:
            from src.models.paper import RagResult
        except ImportError:
            logger.error("[RagResultBuilder] 无法导入 RagResult，请确认模型已定义")
            return None

        result = RagResult(
            query=self._query,
            sub_questions=self._sub_questions,
            rag_strategy=self._rag_strategy,
            paper_candidates=self._paper_candidates,
            evidence_chunks=self._evidence_chunks,
            retrieval_trace=self._traces,
            dedup_log=self._dedup_logs,
            rerank_log=self._rerank_logs,
            coverage_notes=coverage_notes,
            total_papers=len(self._paper_candidates),
            total_chunks=len(self._evidence_chunks),
            retrieved_at=self._now_iso(),
        )

        logger.info(
            f"[RagResultBuilder] built RagResult: "
            f"papers={result.total_papers} chunks={result.total_chunks}"
        )
        return result

    # ── 内部方法 ─────────────────────────────────────────────────────────────

    def _generate_coverage_notes(self) -> list[str]:
        """
        生成覆盖度注释。

        检查以下情况：
        - 无 evidence chunks 时警告
        - 缺少 method/result 类型 evidence
        - evidence 数量不足
        """
        notes: list[str] = []

        if not self._evidence_chunks:
            notes.append("WARNING: 无检索到任何 evidence chunks")
            return notes

        # 检查各 support_type 覆盖
        seen_types: set[str] = set()
        for chunk in self._evidence_chunks:
            stype = getattr(chunk, "support_type", "claim_support")
            if stype:
                seen_types.add(stype)

        for stype in ["method", "result"]:
            if stype not in seen_types:
                notes.append(
                    f"NOTE: 缺少 {stype} 类型 evidence，可能影响报告完整性"
                )

        # 检查是否有足够 chunks
        if len(self._evidence_chunks) < 5:
            notes.append(
                f"NOTE: evidence chunks 较少（{len(self._evidence_chunks)}），覆盖率可能不足"
            )

        return notes

    def _now_iso(self) -> str:
        """
        返回当前 UTC 时间 ISO 格式字符串。

        Returns:
            str: ISO 格式时间，如 "2026-04-10T12:00:00Z"
        """
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
