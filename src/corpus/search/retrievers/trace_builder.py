"""Trace Builder — 记录 retrieval trace，供模块 7 (eval) 追溯根因。"""

from __future__ import annotations

import logging
from typing import Optional

from src.corpus.search.retrievers.models import (
    InitialPaperCandidates,
    RetrievalTrace,
    RetrievalPath,
)

logger = logging.getLogger(__name__)


class TraceBuilder:
    """
    RetrievalTrace 构建器。

    职责：
    - 在每次召回后生成 RetrievalTrace
    - 汇总多条 trace 成 InitialPaperCandidates.traces
    - 提供追溯方法，帮助 reviewer 判断 coverage gap 根因
    """

    def __init__(self):
        self._traces: list[RetrievalTrace] = []

    def add_trace(self, trace: RetrievalTrace) -> None:
        """记录一条 trace。"""
        self._traces.append(trace)

    def add_trace_from_result(
        self,
        query: str,
        path: RetrievalPath,
        sub_question_id: str | None,
        target_index: str,
        returned_doc_ids: list[str],
        returned_chunk_ids: list[str],
        top_k_requested: int,
        duration_ms: float,
        filter_summary: str = "",
        error: str | None = None,
    ) -> None:
        """快捷方法：从执行结果直接构建 trace。"""
        self.add_trace(
            RetrievalTrace(
                query=query,
                sub_question_id=sub_question_id,
                retrieval_path=path,
                target_index=target_index,
                filter_summary=filter_summary,
                top_k_requested=top_k_requested,
                returned_doc_ids=returned_doc_ids,
                returned_chunk_ids=returned_chunk_ids,
                returned_count=len(returned_doc_ids),
                duration_ms=duration_ms,
                error=error,
            )
        )

    def build(self) -> list[RetrievalTrace]:
        """返回所有记录的 trace。"""
        return list(self._traces)

    def diagnose_coverage_gap(
        self,
        returned_doc_ids: list[str],
        expected_doc_ids: list[str] | None = None,
    ) -> dict:
        """
        诊断 coverage gap 的根因。

        基于已记录的 traces，判断：
        - 是 query 本身没覆盖到？
        - 还是 dense recall 漏了？
        - 还是 keyword recall 压根没命中？
        - 还是 filter 太紧？
        """
        kw_paths = {p for p in self._traces if "keyword" in p.retrieval_path.value}
        dense_paths = {p for p in self._traces if "dense" in p.retrieval_path.value}

        diagnostics = {
            "total_traces": len(self._traces),
            "has_keyword_trace": len(kw_paths) > 0,
            "has_dense_trace": len(dense_paths) > 0,
            "total_returned_docs": len(returned_doc_ids),
            "keyword_paths": [str(p.retrieval_path.value) for p in kw_paths],
            "dense_paths": [str(p.retrieval_path.value) for p in dense_paths],
            "likely_gap_reasons": [],
        }

        # 检查各路径是否返回了数据
        kw_returned = sum(p.returned_count for p in kw_paths)
        dense_returned = sum(p.returned_count for p in dense_paths)

        if kw_returned == 0 and dense_returned > 0:
            diagnostics["likely_gap_reasons"].append(
                "keyword recall 没有命中任何结果，但 dense 有结果"
            )
        if dense_returned == 0 and kw_returned > 0:
            diagnostics["likely_gap_reasons"].append(
                "dense recall 没有命中任何结果，但 keyword 有结果"
            )
        if kw_returned == 0 and dense_returned == 0:
            diagnostics["likely_gap_reasons"].append(
                "keyword 和 dense 都没有返回结果，可能是 filter 太紧或 query 不匹配"
            )

        # 检查 filter 是否过紧
        tight_filters = [p for p in self._traces if p.filter_summary and "year" in p.filter_summary]
        if tight_filters:
            diagnostics["likely_gap_reasons"].append(
                "year filter 可能过紧，导致部分论文被过滤"
            )

        # 检查各路径的返回数量
        for trace in self._traces:
            if trace.error:
                diagnostics["likely_gap_reasons"].append(
                    f"trace 出错（path={trace.retrieval_path.value}）：{trace.error}"
                )

        return diagnostics
