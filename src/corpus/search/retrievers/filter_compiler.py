"""Filter Compiler — 将 API filters 编译成各存储层的可执行条件。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from src.corpus.search.retrievers.query_prep import PreparedFilters
from src.corpus.store.metadata_index import MetadataFilter, FilterCondition, FilterOp

logger = logging.getLogger(__name__)


@dataclass
class CompiledFilters:
    """
    编译后的过滤条件，可分发到各存储层。

    Attributes:
        milvus_filter: Milvus filter 表达式字典
        sql_predicates: SQLAlchemy where 子句 (predicates_str, params)
        doc_ids_filter: 若过滤器太宽，返回全部 doc_ids 列表
        filter_summary: 人类可读的过滤条件摘要
    """

    milvus_filter: dict | None = None
    sql_predicates: tuple[str, list] | None = None   # (predicates_str, params)
    doc_ids: list[str] | None = None
    filter_summary: str = ""


class FilterCompiler:
    """
    将结构化 API filters 编译成：

    - Milvus filter 表达式（用于向量检索前过滤）
    - SQLAlchemy predicates（用于 PostgreSQL 查询）
    - 可选：全部 doc_ids 列表（用于小规模过滤）
    """

    def compile(
        self,
        filters: PreparedFilters,
        milvus_schema_fields: list[str] | None = None,
    ) -> CompiledFilters:
        """
        编译 PreparedFilters。

        Args:
            filters: 结构化过滤条件
            milvus_schema_fields: Milvus schema 中存在的字段名（用于校验）

        Returns:
            CompiledFilters
        """
        summary_parts: list[str] = []
        milvus_parts: dict = {}
        sql_parts: list[str] = []
        sql_params: list[Any] = []

        # Year range
        if filters.year_min is not None or filters.year_max is not None:
            if filters.year_min is not None:
                sql_parts.append("published_date >= %s")
                sql_params.append(str(filters.year_min))
                summary_parts.append(f"year >= {filters.year_min}")
            if filters.year_max is not None:
                sql_parts.append("published_date <= %s")
                sql_params.append(str(filters.year_max))
                summary_parts.append(f"year <= {filters.year_max}")

        # Source types
        if filters.sources:
            if len(filters.sources) == 1:
                sql_parts.append("source_type = %s")
                sql_params.append(filters.sources[0])
            else:
                ph = ", ".join(["%s"] * len(filters.sources))
                sql_parts.append(f"source_type IN ({ph})")
                sql_params.extend(filters.sources)
            summary_parts.append(f"sources={filters.sources}")

        # Venues
        if filters.venues:
            if len(filters.venues) == 1:
                sql_parts.append("venue = %s")
                sql_params.append(filters.venues[0])
            else:
                ph = ", ".join(["%s"] * len(filters.venues))
                sql_parts.append(f"venue IN ({ph})")
                sql_params.extend(filters.venues)
            summary_parts.append(f"venues={filters.venues}")

        # Build Milvus-compatible filter parts
        # Note: year_range / sources / venues 过滤通过 KeywordRetriever SQL 查询实现，
        # MilvusVectorIndex 仅支持精确字段过滤（canonical_id）
        milvus_parts: dict = {}
        if filters.canonical_ids:
            ph = ", ".join(["%s"] * len(filters.canonical_ids))
            sql_parts.append(f"canonical_id IN ({ph})")
            sql_params.extend(filters.canonical_ids)
            milvus_parts["canonical_id"] = filters.canonical_ids
            summary_parts.append(f"canonical_ids={len(filters.canonical_ids)} 篇")

        sql_predicates = (" AND ".join(sql_parts), sql_params) if sql_parts else None

        return CompiledFilters(
            milvus_filter=milvus_parts if milvus_parts else None,
            sql_predicates=sql_predicates,
            filter_summary="; ".join(summary_parts) if summary_parts else "无过滤",
        )

    def compile_for_sqlalchemy(
        self, filters: PreparedFilters
    ) -> tuple[str | None, list[Any]]:
        """
        仅编译 SQLAlchemy predicates（快捷方法）。

        Returns:
            (predicates_str, params) 或 (None, [])
        """
        compiled = self.compile(filters)
        return (
            (compiled.sql_predicates[0], compiled.sql_predicates[1])
            if compiled.sql_predicates
            else (None, [])
        )
