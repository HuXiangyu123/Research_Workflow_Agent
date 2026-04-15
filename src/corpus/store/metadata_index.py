"""Metadata Index — 结构化过滤层（workspace / year / venue / section 等）。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filter Operators
# ---------------------------------------------------------------------------


class FilterOp(str, Enum):
    """过滤操作符。"""

    EQ = "eq"          # 等于
    NE = "ne"          # 不等于
    IN = "in"          # 在列表中
    NOT_IN = "not_in"  # 不在列表中
    GTE = "gte"        # 大于等于
    LTE = "lte"        # 小于等于
    GT = "gt"          # 大于
    LT = "lt"          # 小于
    CONTAINS = "contains"  # 字符串包含


# ---------------------------------------------------------------------------
# Filter Condition
# ---------------------------------------------------------------------------


@dataclass
class FilterCondition:
    """单个过滤条件。"""

    field: str
    op: FilterOp
    value: Any

    def to_dict(self) -> dict:
        return {"field": self.field, "op": self.op.value, "value": self.value}


@dataclass
class MetadataFilter:
    """
    结构化过滤条件。

    支持字段：
    - paper-level：year / venue / source_type / canonical_id
    - chunk-level：section / chunk_kind / token_count
    - doc-level：doc_id / ingest_status
    """

    conditions: list[FilterCondition] = field(default_factory=list)

    # ── 快捷构造 ──────────────────────────────────────────────────────────────

    @classmethod
    def year_range(
        cls, min_year: int | None = None, max_year: int | None = None
    ) -> "MetadataFilter":
        cfgs = []
        if min_year is not None:
            cfgs.append(FilterCondition("year", FilterOp.GTE, min_year))
        if max_year is not None:
            cfgs.append(FilterCondition("year", FilterOp.LTE, max_year))
        return cls(conditions=cfgs)

    @classmethod
    def section_eq(cls, section: str) -> "MetadataFilter":
        return cls(conditions=[FilterCondition("section", FilterOp.EQ, section)])

    @classmethod
    def doc_ids(cls, doc_ids: list[str]) -> "MetadataFilter":
        return cls(conditions=[FilterCondition("doc_id", FilterOp.IN, doc_ids)])

    @classmethod
    def canonical_id(cls, canonical_id: str) -> "MetadataFilter":
        return cls(conditions=[FilterCondition("canonical_id", FilterOp.EQ, canonical_id)])

    @classmethod
    def source_type(cls, source_type: str) -> "MetadataFilter":
        return cls(conditions=[FilterCondition("source_type", FilterOp.EQ, source_type)])

    @classmethod
    def sections(cls, sections: list[str]) -> "MetadataFilter":
        return cls(conditions=[FilterCondition("section", FilterOp.IN, sections)])

    def add(self, condition: FilterCondition) -> "MetadataFilter":
        self.conditions.append(condition)
        return self

    def merge(self, other: "MetadataFilter") -> "MetadataFilter":
        return MetadataFilter(conditions=self.conditions + other.conditions)

    def to_sql_predicates(self) -> tuple[str, list]:
        """
        转换为 (predicates_str, params) 元组，供 SQLAlchemy 使用。

        Example:
            predicates, params = filt.to_sql_predicates()
            session.query(Document).filter(text(predicates), *params)
        """
        parts: list[str] = []
        params: list[Any] = []

        for c in self.conditions:
            if c.op == FilterOp.EQ:
                parts.append(f"{c.field} = %s"); params.append(c.value)
            elif c.op == FilterOp.NE:
                parts.append(f"{c.field} != %s"); params.append(c.value)
            elif c.op == FilterOp.IN:
                ph = ", ".join(["%s"] * len(c.value))
                parts.append(f"{c.field} IN ({ph})"); params.extend(c.value)
            elif c.op == FilterOp.NOT_IN:
                ph = ", ".join(["%s"] * len(c.value))
                parts.append(f"{c.field} NOT IN ({ph})"); params.extend(c.value)
            elif c.op == FilterOp.GTE:
                parts.append(f"{c.field} >= %s"); params.append(c.value)
            elif c.op == FilterOp.LTE:
                parts.append(f"{c.field} <= %s"); params.append(c.value)
            elif c.op == FilterOp.GT:
                parts.append(f"{c.field} > %s"); params.append(c.value)
            elif c.op == FilterOp.LT:
                parts.append(f"{c.field} < %s"); params.append(c.value)
            elif c.op == FilterOp.CONTAINS:
                parts.append(f"{c.field} LIKE %s"); params.append(f"%{c.value}%")

        return " AND ".join(parts), params

    def is_empty(self) -> bool:
        return len(self.conditions) == 0


# ---------------------------------------------------------------------------
# Metadata Index
# ---------------------------------------------------------------------------


class MetadataIndex:
    """
    元数据过滤与聚合层。

    职责：
    - 编译 MetadataFilter → SQL where 子句
    - 提供聚合统计（year distribution、source distribution）
    """

    def __init__(self, db_session: "Session | None" = None):
        self._db = db_session

    def set_session(self, session: "Session") -> None:
        self._db = session

    def filter_documents(
        self, filt: MetadataFilter, limit: int = 100, offset: int = 0
    ) -> list[str]:
        """
        根据 MetadataFilter 过滤，返回匹配的 doc_id 列表。
        """
        if not self._db:
            logger.warning("[MetadataIndex] 无 DB session")
            return []

        from sqlalchemy import text
        from src.db.models import Document

        if filt.is_empty():
            rows = (
                self._db.query(Document.doc_id)
                .offset(offset).limit(limit).all()
            )
            return [r[0] for r in rows]

        predicates, params = filt.to_sql_predicates()
        if not predicates:
            return []

        rows = (
            self._db.query(Document.doc_id)
            .filter(text(predicates), *params)
            .offset(offset).limit(limit).all()
        )
        return [r[0] for r in rows]

    def count_documents(self, filt: MetadataFilter) -> int:
        """统计符合条件的 Document 数量。"""
        if not self._db:
            return 0

        from sqlalchemy import func, text
        from src.db.models import Document

        if filt.is_empty():
            return self._db.query(func.count(Document.doc_id)).scalar() or 0

        predicates, params = filt.to_sql_predicates()
        if not predicates:
            return 0

        return (
            self._db.query(func.count(Document.doc_id))
            .filter(text(predicates), *params)
            .scalar() or 0
        )

    def aggregate_years(self) -> dict[int, int]:
        """按年份统计文档数量。"""
        if not self._db:
            return {}

        from sqlalchemy import func
        from src.db.models import Document

        try:
            rows = (
                self._db.query(
                    Document.published_date,
                    func.count(Document.doc_id).label("count"),
                )
                .filter(Document.published_date.isnot(None))
                .group_by(Document.published_date)
                .all()
            )
            result: dict[int, int] = {}
            for row in rows:
                date_str = str(row[0] or "")[:4]
                if date_str.isdigit():
                    year = int(date_str)
                    result[year] = result.get(year, 0) + int(row[1] or 0)
            return result
        except Exception:
            return {}

    def aggregate_source_types(self) -> dict[str, int]:
        """按来源类型统计文档数量。"""
        if not self._db:
            return {}

        from sqlalchemy import func
        from src.db.models import Document

        try:
            rows = (
                self._db.query(
                    Document.source_type,
                    func.count(Document.doc_id).label("count"),
                )
                .filter(Document.source_type.isnot(None))
                .group_by(Document.source_type)
                .all()
            )
            return {str(r[0] or "unknown"): int(r[1] or 0) for r in rows}
        except Exception:
            return {}
