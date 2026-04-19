"""Extract Compression Node — 在 extract 和 draft 之间插入上下文压缩。

设计文档：docs/features_oncoming/context-compression-for-report-generation.md

压缩管道：
    extract_node → extract_compression_node → draft_node

该节点负责：
1. 构建 Taxonomy（论文分类）
2. 压缩论文摘要（build_compressed_abstracts）
3. 构建 Per-Section Evidence Pool
"""

from __future__ import annotations

import logging
from typing import Any

from src.models.compression import CompressionResult
from src.research.services.compression import compress_paper_cards
from src.tasking.trace_wrapper import get_trace_store, trace_node

logger = logging.getLogger(__name__)


@trace_node(node_name="extract_compression", stage="compress", store=get_trace_store())
def extract_compression_node(state: dict) -> dict:
    """
    Extract Compression 节点 — 上下文压缩。

    输入：state.paper_cards, state.brief
    输出：state.compression_result (CompressionResult)

    该节点在 extract_node 和 draft_node 之间运行，
    将原始 paper_cards（约 30k chars）压缩为：
    - taxonomy (~3k chars)
    - compressed_cards (~4k chars)
    - evidence_pools (~60k chars，按 section 分配）

    总体压缩率约 87%。
    """
    paper_cards = state.get("paper_cards", [])
    brief = state.get("brief")

    if not paper_cards:
        logger.warning("[extract_compression_node] no paper_cards, skipping")
        return {"compression_result": None}

    # 如果没有 brief，从 state 中构建一个简单的
    if not brief:
        brief = None

    try:
        # 执行压缩
        result = compress_paper_cards(paper_cards, brief)

        logger.info(
            "[extract_compression_node] compressed %d papers → taxonomy=%d categories, "
            "compressed_cards=%d, compression_ratio=%.1f%%",
            len(paper_cards),
            len(result.taxonomy.categories),
            len(result.compressed_cards),
            result.compression_stats.get("compression_ratio", 0) * 100,
        )

        # 将 CompressionResult 序列化为 dict 存入 state
        return {
            "compression_result": result.model_dump(),
            "taxonomy": result.taxonomy.model_dump(),
        }

    except Exception as exc:
        logger.exception("[extract_compression_node] compression failed: %s", exc)
        return {
            "compression_result": None,
            "warnings": [f"Compression failed: {exc}"],
        }
