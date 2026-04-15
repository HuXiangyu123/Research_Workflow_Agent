"""Query Preparation — 将 query / SearchPlan / SubQuestion 转成结构化检索查询。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

logger = __import__("logging").getLogger(__name__)


# ---------------------------------------------------------------------------
# PreparedQuery — 单条待检索的查询
# ---------------------------------------------------------------------------


@dataclass
class PreparedQuery:
    """一条可执行的检索查询。"""

    text: str
    is_main: bool = True
    sub_question_id: Optional[str] = None
    sub_question_text: Optional[str] = None


# ---------------------------------------------------------------------------
# PreparedFilters — 结构化过滤条件
# ---------------------------------------------------------------------------


@dataclass
class PreparedFilters:
    """
    结构化过滤条件，兼容多种存储层。

    支持字段：
    - year_range: (min, max)
    - sources: list of source_type
    - venues: list of venue
    - workspace_id: str
    - canonical_ids: list[str]  # 用于限定特定论文
    """

    year_min: Optional[int] = None
    year_max: Optional[int] = None
    sources: list[str] = field(default_factory=list)
    venues: list[str] = field(default_factory=list)
    workspace_id: Optional[str] = None
    canonical_ids: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return (
            self.year_min is None
            and self.year_max is None
            and not self.sources
            and not self.venues
            and not self.workspace_id
            and not self.canonical_ids
        )

    def to_milvus_filter(self) -> dict | None:
        """编译成 Milvus filter 表达式字典。"""
        if self.is_empty():
            return None
        parts: dict = {}
        if self.year_min is not None or self.year_max is not None:
            # Milvus 的 year 字段需要在 schema 里存在
            # 此处返回 None 由调用方决定如何处理
            pass
        if self.canonical_ids:
            parts["canonical_id"] = self.canonical_ids
        return parts if parts else None


# ---------------------------------------------------------------------------
# SearchInput — 统一的检索输入
# ---------------------------------------------------------------------------


@dataclass
class SearchInput:
    """
    模块 4 检索的统一输入。

    支持：
    - 裸 query（字符串）
    - 带 sub_questions 的 SearchPlan
    - 带 filters 的扩展输入
    """

    # 主查询
    query: str

    # 子问题（来自 SearchPlan）
    sub_questions: list[dict] = field(default_factory=list)

    # 元数据过滤
    filters: Optional[PreparedFilters] = None

    # 召回参数
    keyword_top_k: int = 30
    dense_top_k: int = 30

    def prepare(self) -> tuple[list[PreparedQuery], PreparedFilters]:
        """
        生成待执行的检索查询列表和过滤条件。

        Returns:
            (queries, filters)：
            - queries: 主 query + 子问题 query 列表
            - filters: 结构化过滤条件
        """
        queries: list[PreparedQuery] = []

        # 主 query
        queries.append(
            PreparedQuery(
                text=self.query,
                is_main=True,
            )
        )

        # 子问题 queries
        for sq in self.sub_questions:
            sq_id = sq.get("id") or sq.get("sub_question_id")
            sq_text = sq.get("text") or sq.get("query") or sq.get("question")
            if sq_text and sq_id:
                queries.append(
                    PreparedQuery(
                        text=sq_text,
                        is_main=False,
                        sub_question_id=sq_id,
                        sub_question_text=sq_text,
                    )
                )

        return queries, self.filters or PreparedFilters()


# ---------------------------------------------------------------------------
# Query Preparation — 主函数
# ---------------------------------------------------------------------------


def prepare_search(
    query: str,
    sub_questions: list[dict] | None = None,
    year_range: tuple[int, int] | None = None,
    sources: list[str] | None = None,
    venues: list[str] | None = None,
    workspace_id: str | None = None,
    **kwargs,
) -> SearchInput:
    """
    快捷构造 SearchInput 并返回 PreparedQuery 列表。

    使用示例：
        input = prepare_search(
            query="multi-agent systems in scientific literature review",
            sub_questions=[
                {"id": "sq1", "text": "how do they retrieve papers"},
                {"id": "sq2", "text": "how do they verify citations"},
            ],
            year_range=(2020, 2025),
            sources=["arxiv", "local_pdf"],
        )
        queries, filters = input.prepare()
    """
    filters = PreparedFilters(
        year_min=year_range[0] if year_range else None,
        year_max=year_range[1] if year_range else None,
        sources=sources or [],
        venues=venues or [],
        workspace_id=workspace_id,
    )
    return SearchInput(
        query=query,
        sub_questions=sub_questions or [],
        filters=filters,
    )
