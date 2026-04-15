"""Canonicalizer — 论文身份归并：同一论文多个来源 → 唯一 canonical_id。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.corpus.models import (
    CanonicalKey,
    SourceRef,
    SourceType,
    StandardizedDocument,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Merge Decision
# ---------------------------------------------------------------------------


@dataclass
class MergeDecision:
    """归并决策结果。"""

    decision: str  # "auto_merge" | "candidate" | "keep_separate" | "same_paper"
    confidence: float  # 0.0~1.0
    reason: str
    merged_canonical_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Canonicalizer
# ---------------------------------------------------------------------------


class Canonicalizer:
    """
    为每篇论文生成唯一 canonical_id，并处理同论文多来源归并。

    归并优先级：
    1. DOI 完全一致 → 高置信度同论文，自动合并
    2. arXiv ID 一致 → 高置信度同论文，自动合并
    3. title 高相似 + first author 一致 + year 接近 → 候选，需确认
    4. 同标题但 venue/year 差异明显 → 视为版本关系，不直接覆盖
    """

    def __init__(self, db_session=None):
        """
        初始化 Canonicalizer。

        Args:
            db_session: 可选的 SQLAlchemy session，用于查询已有 canonical papers。
        """
        self._db = db_session

    def build_key(
        self,
        title: str,
        authors: list[str],
        year: int | None,
        doi: str | None = None,
        arxiv_id: str | None = None,
        venue: str | None = None,
    ) -> CanonicalKey:
        """
        从元数据构建 CanonicalKey。
        """
        first_author_surname = ""
        if authors:
            # 取第一个作者的 surname（最后一个词）
            parts = authors[0].strip().split()
            first_author_surname = parts[-1] if parts else ""

        return CanonicalKey(
            normalized_title=title.strip().lower(),
            first_author_surname=first_author_surname.lower(),
            year=year or 0,
            doi=doi,
            arxiv_id=arxiv_id,
            venue=venue,
        )

    def decide_merge(
        self,
        new_key: CanonicalKey,
        existing_key: CanonicalKey,
    ) -> MergeDecision:
        """
        判断 new_key 和 existing_key 是否应归并。
        """
        # 1. DOI 完全一致 → 最高置信度
        if new_key.doi and existing_key.doi:
            if new_key.doi.strip() == existing_key.doi.strip():
                return MergeDecision(
                    decision="auto_merge",
                    confidence=0.99,
                    reason="DOI 完全一致",
                )

        # 2. arXiv ID 一致
        if new_key.arxiv_id and existing_key.arxiv_id:
            # 去掉版本号比较（2301.12345v1 == 2301.12345v2）
            new_id = _strip_arxiv_version(new_key.arxiv_id)
            existing_id = _strip_arxiv_version(existing_key.arxiv_id)
            if new_id == existing_id:
                return MergeDecision(
                    decision="auto_merge",
                    confidence=0.95,
                    reason="arXiv ID 一致",
                    merged_canonical_id=f"canon_{new_key.to_hash()}",
                )

        # 3. title + author + year 综合判断
        title_sim = _title_similarity(
            new_key.normalized_title, existing_key.normalized_title
        )
        author_match = new_key.first_author_surname == existing_key.first_author_surname
        year_match = new_key.year == existing_key.year

        if title_sim >= 0.9 and author_match:
            # 高相似标题 + 作者匹配
            if year_match:
                return MergeDecision(
                    decision="auto_merge",
                    confidence=0.85,
                    reason=f"标题相似度 {title_sim:.2f} + 作者匹配 + 年份一致",
                    merged_canonical_id=f"canon_{new_key.to_hash()}",
                )
            else:
                # 年份差 1 年，可能是修订版
                if abs(new_key.year - existing_key.year) <= 1:
                    return MergeDecision(
                        decision="same_paper",
                        confidence=0.75,
                        reason=f"标题相似度高 + 作者匹配 + 年份接近（差 {abs(new_key.year - existing_key.year)} 年）",
                        merged_canonical_id=f"canon_{new_key.to_hash()}",
                    )

        if title_sim >= 0.7 and author_match:
            # 中等相似
            if year_match:
                return MergeDecision(
                    decision="candidate",
                    confidence=0.6,
                    reason=f"标题相似度 {title_sim:.2f} + 作者匹配（需人工确认）",
                )
            else:
                return MergeDecision(
                    decision="candidate",
                    confidence=0.5,
                    reason=f"标题相似但年份差异（需人工确认）",
                )

        # 4. 同标题但 venue/year 差异明显 → 视为不同版本
        if title_sim >= 0.95:
            return MergeDecision(
                decision="same_paper",
                confidence=0.5,
                reason="标题高度相似但 venue/year 差异，可能是不同版本",
            )

        # 5. 差异太大，不归并
        return MergeDecision(
            decision="keep_separate",
            confidence=0.1,
            reason="论文差异较大，保持独立",
        )

    def find_existing_canonical(
        self,
        new_key: CanonicalKey,
    ) -> tuple[Optional[str], MergeDecision]:
        """
        在数据库中查找与 new_key 匹配已有 canonical_id。

        Returns:
            (canonical_id, MergeDecision)：若找到匹配的，返回其 canonical_id 和决策
        """
        if not self._db:
            # 无数据库会话，只能用 key 本身生成 ID
            return None, MergeDecision(
                decision="keep_separate",
                confidence=0.0,
                reason="无数据库会话，无法查找已有论文",
            )

        # 查询所有已有 canonical papers
        # （这里用接口约定，实际 DB 操作在 db.py 层）
        return None, MergeDecision(
            decision="keep_separate",
            confidence=0.0,
            reason="DB 查询接口待接入",
        )

    def canonicalize_document(
        self,
        doc: StandardizedDocument,
    ) -> StandardizedDocument:
        """
        为 StandardizedDocument 赋予 canonical_id。

        流程：
        1. 构建 CanonicalKey
        2. 查询已有 canonical papers（若 DB 可用）
        3. 决定归并策略
        4. 赋予/复用 canonical_id
        """
        if not doc.canonical_key:
            doc.canonical_key = self.build_key(
                title=doc.title,
                authors=doc.authors,
                year=doc.year,
                doi=doc.doi,
                arxiv_id=doc.arxiv_id,
                venue=doc.venue,
            )

        canonical_id, decision = self.find_existing_canonical(doc.canonical_key)

        if decision.decision in ("auto_merge", "same_paper") and decision.merged_canonical_id:
            doc.canonical_id = decision.merged_canonical_id
        elif canonical_id:
            doc.canonical_id = canonical_id
        else:
            # 新论文，生成新的 canonical_id
            doc.canonical_id = f"canon_{doc.canonical_key.to_hash()}"

        doc.warnings.append(f"canonicalize: {decision.reason} (conf={decision.confidence:.2f})")
        doc.ingest_status = "canonicalized"
        return doc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_arxiv_version(arxiv_id: str) -> str:
    """去掉 arXiv ID 的版本号（v1, v2 等）。"""
    import re
    return re.sub(r"v\d+$", "", arxiv_id.strip()).strip()


def _title_similarity(title1: str, title2: str) -> float:
    """
    计算两个标题的相似度（简单词集合 Jaccard）。
    """
    if not title1 or not title2:
        return 0.0

    # 标准化：去掉标点、转小写、分词
    def tokenize(t: str) -> set[str]:
        import re
        words = re.sub(r"[^\w\s]", " ", t.lower()).split()
        # 过滤停用词
        stopwords = {
            "a", "an", "the", "of", "for", "in", "on", "to", "and", "or",
            "with", "from", "by", "at", "is", "are", "was", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "can",
            "this", "that", "these", "those", "it", "its",
        }
        return {w for w in words if w not in stopwords and len(w) > 1}

    set1, set2 = tokenize(title1), tokenize(title2)
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0
