"""EvidenceTyper — 基于 section name 的轻量 evidence typing（模块 6）。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.corpus.search.models import EvidenceChunk
    from src.corpus.search.retrievers.chunk_retriever import ChunkSearchResult

logger = logging.getLogger(__name__)


# ── Section-to-Type 映射规则 ───────────────────────────────────────────────────

SUPPORT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "method": [
        "method", "methodology", "approach", "propose", "proposed",
        "architecture", "model", "algorithm", "framework",
        "technique", "implementation", "design",
    ],
    "result": [
        "result", "results", "experiment", "experiments", "evaluation",
        "performance", "benchmark", "accuracy", "score", "dataset",
        "comparison", "baseline", "ablation", "empirical",
    ],
    "background": [
        "introduction", "background", "related work", "related works",
        "survey", "prior", "previous", "motivation", "overview",
    ],
    "limitation": [
        "limitation", "limitations", "weakness", "weaknesses",
        "failure", "drawback", "cannot", "does not scale",
        "challenge", "future work",
    ],
}


class EvidenceTyper:
    """
    基于 section name 的轻量 evidence typing。

    速度极快，适合大规模 evidence 标注。
    规则：扫描 section 名称中的关键词，匹配即返回对应类型。
    未匹配任何规则 → "claim_support"（默认 fallback）。

    使用方式：
        typer = EvidenceTyper()
        support_type = typer.infer_support_type("Experimental Results and Analysis")
        # → "result"

        # 批量标注
        for chunk in evidence_chunks:
            chunk.support_type = typer.infer_support_type(chunk.section)
    """

    def __init__(self):
        # 预编译：合并关键词，加速匹配
        self._compiled: list[tuple[str, frozenset[str]]] = [
            (stype, frozenset(kws))
            for stype, kws in SUPPORT_TYPE_KEYWORDS.items()
        ]

    def infer_support_type(self, section: str | None) -> str:
        """
        根据 section 名称推断 support_type。

        Args:
            section: 章节标题（如 "Experimental Results"）

        Returns:
            support_type: method / result / background / limitation / claim_support
        """
        if not section:
            return "claim_support"

        section_lower = section.lower()
        for stype, kws in self._compiled:
            if any(kw in section_lower for kw in kws):
                return stype

        return "claim_support"

    def type_chunk(self, chunk) -> str:
        """推断单个 chunk 的 support_type。"""
        section = getattr(chunk, "section", None) or ""
        return self.infer_support_type(section)

    def type_chunks(self, chunks: list) -> list[str]:
        """批量推断 chunks 的 support_type。"""
        return [self.type_chunk(c) for c in chunks]

    def annotate_chunks(self, chunks: list) -> None:
        """
        直接在传入的 chunks 对象上设置 support_type 属性。

        原地修改，无返回值。
        """
        for chunk in chunks:
            stype = self.type_chunk(chunk)
            if hasattr(chunk, "support_type"):
                chunk.support_type = stype
            else:
                # 对于 dataclass-like 对象
                try:
                    object.__setattr__(chunk, "support_type", stype)
                except (AttributeError, TypeError):
                    pass