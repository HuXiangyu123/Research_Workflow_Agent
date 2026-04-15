"""Paper Deduper — 按 canonical_id 做论文级去重归并。"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ── DedupInfo ────────────────────────────────────────────────────────────────


@dataclass
class DedupInfo:
    """去重信息。"""
    is_canonical_representative: bool = True   # 是否是该 canonical_id 的主版本
    merged_doc_ids: list[str] = field(default_factory=list)  # 被归并的其他 doc_ids
    source_refs: list[str] = field(default_factory=list)   # 所有来源


# ── DedupedCandidate ────────────────────────────────────────────────────────


@dataclass
class DedupedCandidate:
    """
    去重后的单篇论文候选。

    同一论文的多来源版本（PDF / arXiv / conference）被归并为一条。
    """
    canonical_id: str
    merged_doc_ids: list[str] = field(default_factory=list)
    primary_doc_id: str = ""

    # 元数据
    title: str = ""
    abstract: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None

    # 模块 4 分数（RRF 融合后）
    rrf_score: float = 0.0
    keyword_score: float = 0.0
    dense_score: float = 0.0

    # 来源追踪
    matched_queries: list = field(default_factory=list)
    matched_paths: list = field(default_factory=list)
    recall_evidence: list = field(default_factory=list)

    # 去重信息
    dedup_info: DedupInfo = field(default_factory=DedupInfo)

    # Rerank 分数（后续填充）
    rerank_score: Optional[float] = None
    final_score: float = 0.0


# ── PaperDeduper ────────────────────────────────────────────────────────────


class PaperDeduper:
    """
    Canonical Dedup：对候选池按 canonical_id 做论文级归并。

    同一论文的不同来源（PDF / arXiv / conference / journal）合并为单一候选。
    """

    def __init__(self, db_session: "Session | None" = None):
        self._db = db_session

    def dedup(
        self,
        candidates: list,
    ) -> list[DedupedCandidate]:
        """
        对候选列表做 canonical dedup。

        Args:
            candidates: 模块 4 输出的 MergedCandidate 列表

        Returns:
            DedupedCandidate 列表（按 canonical_id 归并）
        """
        # 1. 按 canonical_id（或 doc_id）分组
        groups: dict[str, list] = defaultdict(list)
        for c in candidates:
            key = c.canonical_id if c.canonical_id else c.doc_id
            groups[key].append(c)

        # 2. 对每个 group 做聚合
        deduped: list[DedupedCandidate] = []
        for key, group in groups.items():
            deduped.append(self._merge_group(group))

        # 3. 填充缺失的 canonical_id（从 DB 回填）
        self._fill_missing_canonical_ids(deduped)

        return deduped

    def _merge_group(self, group: list) -> DedupedCandidate:
        """
        将同一 canonical_id 下的多条候选合并为一条 DedupedCandidate。

        合并策略：
        - 选择 RRF score 最高的候选作为主候选（canonical_representative）
        - 收集所有 source_refs
        - 收集所有 matched_queries（去重）
        - 收集所有 recall_paths
        - 聚合 keyword_score / dense_score（取各路最高分）
        - 收集所有 recall_evidence（合并）
        """
        # 选主候选（RRF 分数最高的那个）
        primary = max(group, key=lambda c: c.rrf_score)

        # 收集所有 doc_ids
        merged_doc_ids = [c.doc_id for c in group]
        other_doc_ids = [c.doc_id for c in group if c.doc_id != primary.doc_id]

        # 收集所有来源
        source_refs = []
        for c in group:
            if getattr(c, "source_uri", None):
                source_refs.append(c.source_uri)
            elif getattr(c, "source_refs", None):
                source_refs.extend(c.source_refs)

        deduped = DedupedCandidate(
            canonical_id=primary.canonical_id if primary.canonical_id else primary.doc_id,
            merged_doc_ids=merged_doc_ids,
            primary_doc_id=primary.doc_id,
            # 元数据（取主候选的）
            title=primary.title or "",
            abstract=primary.abstract,
            authors=primary.authors or [],
            year=primary.year,
            venue=primary.venue,
            # 分数（各路取最高）
            rrf_score=primary.rrf_score,
            keyword_score=max((c.keyword_score for c in group), default=0.0),
            dense_score=max((c.dense_score for c in group), default=0.0),
            # 去重信息
            dedup_info=DedupInfo(
                is_canonical_representative=True,
                merged_doc_ids=other_doc_ids,
                source_refs=source_refs,
            ),
        )

        # 合并 matched_queries（去重）
        seen_queries: set = set()
        for c in group:
            for mq in getattr(c, "matched_queries", []):
                key = (getattr(mq, "query_text", "") or "", getattr(mq, "path", ""))
                key_str = str(key[0]) + str(key[1])
                if key_str not in seen_queries:
                    deduped.matched_queries.append(mq)
                    seen_queries.add(key_str)

        # 合并 matched_paths
        paths = set()
        for c in group:
            for mq in getattr(c, "matched_queries", []):
                path = getattr(mq, "path", None)
                if path:
                    paths.add(path)
        deduped.matched_paths = list(paths)

        # 合并 recall_evidence（每个 path 取最优 top-3）
        deduped.recall_evidence = self._merge_evidence(group, top_n=3)

        return deduped

    def _merge_evidence(
        self,
        group: list,
        top_n: int = 3,
    ) -> list:
        """从 group 中收集最好的 recall evidence（每个 path 取 top-N）。"""
        from collections import defaultdict
        from src.corpus.search.retrievers.models import RetrievalPath

        by_path: dict = defaultdict(list)
        for c in group:
            for ev in getattr(c, "recall_evidence", []):
                ev_path = getattr(ev, "path", None)
                if ev_path:
                    by_path[ev_path].append(ev)

        merged = []
        for path, evs in by_path.items():
            top_evs = sorted(evs, key=lambda e: getattr(e, "score", 0.0), reverse=True)[:top_n]
            merged.extend(top_evs)
        return merged

    def _fill_missing_canonical_ids(
        self, deduped: list[DedupedCandidate]
    ) -> None:
        """
        填充缺失的 canonical_id（仅 doc_id 无法映射 canonical_id 的情况）。

        当 deduped.canonical_id == primary_doc_id 且无 DB 映射时，
        保留现状（无法进一步归并）。
        """
        # 目前不做额外处理：canonical_id 缺失 = 论文在 DB 中未被归一化
        # 在 ingest 阶段已有 CanonicalPaper 表管理，这里不重复逻辑
        pass
