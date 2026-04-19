"""上下文压缩服务 — 报告生成前的上下文压缩核心实现。

设计文档：docs/features_oncoming/context-compression-for-report-generation.md

压缩管道：
    extract_node → extract_compression_node → draft_node

压缩算法：
    A. build_taxonomy — 论文分类压缩
    B. build_compressed_abstracts — 论文摘要压缩
    C. build_evidence_pools — Section 级 Evidence 池
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.models.compression import (
    CompressionResult,
    CompressedCard,
    EvidencePool,
    PoolEntry,
    Taxonomy,
    TaxonomyCategory,
)
from src.research.research_brief import ResearchBrief

logger = logging.getLogger(__name__)

# 最大保留论文数（安全硬上限）
MAX_COMPRESSION_CARDS = 28
# 预算式保留证据，避免固定前 N 截断
COMPRESSION_CHAR_BUDGET = 42000
# 每个 section 的 token 预算（估算 1 token ≈ 2 chars）
SECTION_TOKEN_BUDGETS = {
    "introduction": 8000,
    "background": 6000,
    "taxonomy": 8000,
    "methods": 10000,
    "datasets": 4000,
    "evaluation": 6000,
    "discussion": 5000,
    "future_work": 4000,
    "conclusion": 2000,
}


def compress_paper_cards(
    paper_cards: list[dict[str, Any]],
    brief: ResearchBrief | dict[str, Any] | None,
) -> CompressionResult:
    """
    主入口：将 paper_cards 压缩为 taxonomy + compressed_cards + evidence_pools。

    Args:
        paper_cards: extract_node 输出的论文卡片列表
        brief: ClarifyAgent 生成的 ResearchBrief（用于 taxonomy 构建）

    Returns:
        CompressionResult: 包含 taxonomy、compressed_cards、evidence_pools
    """
    if not paper_cards:
        return CompressionResult()

    # 预算式选择卡片，优先保留正文证据，同时保证方法/主题多样性。
    cards_to_process = _select_cards_for_compression(
        paper_cards,
        max_cards=MAX_COMPRESSION_CARDS,
        char_budget=COMPRESSION_CHAR_BUDGET,
    )

    # Step 1: 构建 Taxonomy
    taxonomy = _build_taxonomy(cards_to_process, brief)

    # Step 2: 压缩论文摘要
    compressed = _build_compressed_abstracts(cards_to_process, taxonomy)

    # Step 3: 构建 Per-Section Evidence Pool
    pools = _build_evidence_pools(compressed, taxonomy, brief)

    # 统计信息
    original_chars = sum(len(_card_to_text(c)) for c in cards_to_process)
    compressed_chars = sum(len(_card_to_text(c)) for c in compressed)
    compression_ratio = (
        1 - compressed_chars / original_chars if original_chars > 0 else 0
    )

    stats = {
        "original_cards": len(paper_cards),
        "processed_cards": len(cards_to_process),
        "original_chars": original_chars,
        "compressed_chars": compressed_chars,
        "compression_ratio": round(compression_ratio, 3),
        "selection_mode": "budgeted_progressive",
    }

    return CompressionResult(
        taxonomy=taxonomy,
        compressed_cards=compressed,
        evidence_pools=pools,
        compression_stats=stats,
    )


def _select_cards_for_compression(
    paper_cards: list[dict[str, Any]],
    *,
    max_cards: int,
    char_budget: int,
) -> list[dict[str, Any]]:
    """Select cards progressively instead of taking a fixed prefix.

    The selector keeps strong full-text cards first, but it also round-robins
    across lightweight topical buckets so the downstream draft sees broader
    evidence coverage.
    """
    if len(paper_cards) <= max_cards:
        return list(paper_cards)

    buckets: dict[str, list[dict[str, Any]]] = {}
    for card in sorted(
        paper_cards,
        key=lambda item: (
            1 if item.get("fulltext_available") else 0,
            float(item.get("combined_score") or item.get("score") or 0.0),
            len(str(item.get("summary") or item.get("abstract") or "")),
        ),
        reverse=True,
    ):
        bucket = _compression_bucket(card)
        buckets.setdefault(bucket, []).append(card)

    ordered_bucket_names = sorted(
        buckets.keys(),
        key=lambda name: (
            0 if name == "fulltext" else 1,
            -len(buckets[name]),
            name,
        ),
    )

    selected: list[dict[str, Any]] = []
    used_chars = 0
    while ordered_bucket_names and len(selected) < max_cards:
        progressed = False
        for bucket in list(ordered_bucket_names):
            queue = buckets.get(bucket, [])
            if not queue:
                ordered_bucket_names.remove(bucket)
                continue

            card = queue.pop(0)
            card_chars = len(_card_to_text(card))
            would_exceed = used_chars + card_chars > char_budget
            if selected and would_exceed and len(selected) >= min(12, max_cards):
                continue

            selected.append(card)
            used_chars += card_chars
            progressed = True
            if len(selected) >= max_cards:
                break

        if not progressed:
            break

    return selected or paper_cards[:max_cards]


def _compression_bucket(card: dict[str, Any]) -> str:
    if card.get("fulltext_available"):
        return "fulltext"

    methods = card.get("methods") or []
    if isinstance(methods, list) and methods:
        return f"method:{str(methods[0]).lower()[:32]}"

    datasets = card.get("datasets") or []
    if isinstance(datasets, list) and datasets:
        return f"dataset:{str(datasets[0]).lower()[:32]}"

    title = str(card.get("title") or "").lower()
    if "survey" in title:
        return "title:survey"
    if "agent" in title:
        return "title:agent"
    if "medical" in title or "clinical" in title:
        return "title:medical"
    return "misc"


def _build_taxonomy(
    cards: list[dict[str, Any]],
    brief: ResearchBrief | dict[str, Any] | None,
) -> Taxonomy:
    """用 LLM 将论文按技术路线/子领域分类。"""
    from src.agent.llm import build_reason_llm
    from src.agent.settings import get_settings
    from langchain_core.messages import HumanMessage, SystemMessage

    cards_text = _render_cards_for_taxonomy(cards)
    topic_value = ""
    if isinstance(brief, dict):
        topic_value = str(brief.get("topic") or "").strip()
    elif brief is not None:
        topic_value = str(getattr(brief, "topic", "") or "").strip()
    topic_hint = f"\nResearch topic: {topic_value}" if topic_value else ""

    SYSTEM = (
        "You are a paper taxonomy expert. Given a list of research papers, "
        "organize them into a hierarchical taxonomy based on technical approaches, "
        "sub-domains, or research paradigms. "
        "The output MUST be strictly valid JSON (no markdown code blocks).\n\n"
        "OUTPUT SCHEMA:\n"
        '  "categories": [\n'
        "    {\n"
        '      "name": "Category Name",\n'
        '      "description": "Brief description of this category",\n'
        '      "papers": ["Paper title 1", "Paper title 2", ...],\n'
        '      "key_characteristics": ["characteristic 1", ...],\n'
        '      "shared_insights": ["insight from multiple papers", ...],\n'
        '      "conflicts": ["conflicting findings between papers", ...]\n'
        "    },\n"
        "    ...\n"
        "  ],\n"
        '  "cross_category_themes": ["theme spanning multiple categories", ...],\n'
        '  "timeline": ["earlier development", "later development", ...],\n'
        '  "key_papers": ["Essential paper title 1", ...]\n'
        "}"
    )

    USER = (
        f"Below are summaries for {len(cards)} research papers:\n"
        f"{cards_text}\n"
        f"{topic_hint}\n\n"
        "Organize them into a taxonomy and return JSON only."
    )

    try:
        settings = get_settings()
        llm = build_reason_llm(settings, max_tokens=8192, timeout_s=180)
        response = llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=USER)])
        content = response.content if hasattr(response, "content") else str(response)
        data = _extract_json(content)
        if data:
            cats = [
                TaxonomyCategory(
                    name=c.get("name", ""),
                    description=c.get("description", ""),
                    papers=c.get("papers", []),
                    key_characteristics=c.get("key_characteristics", []),
                    shared_insights=c.get("shared_insights", []),
                    conflicts=c.get("conflicts", []),
                )
                for c in data.get("categories", [])
            ]
            return Taxonomy(
                categories=cats,
                cross_category_themes=data.get("cross_category_themes", []),
                timeline=data.get("timeline", []),
                key_papers=data.get("key_papers", []),
            )
    except Exception as exc:
        logger.warning("[_build_taxonomy] LLM failed: %s, using fallback", exc)

    # Fallback: 简单按论文数量分组
    return _build_taxonomy_fallback(cards)


def _build_compressed_abstracts(
    cards: list[dict[str, Any]],
    taxonomy: Taxonomy,
) -> list[CompressedCard]:
    """用 LLM 将每张卡片压缩到 ~300 chars。"""
    from src.agent.llm import build_reason_llm
    from src.agent.settings import get_settings
    from langchain_core.messages import HumanMessage, SystemMessage

    if len(cards) > 15:
        # 分批处理避免 token 超限
        compressed: list[CompressedCard] = []
        for i in range(0, len(cards), 15):
            batch = cards[i : i + 15]
            compressed.extend(_compress_batch(batch, taxonomy))
        return compressed

    return _compress_batch(cards, taxonomy)


def _compress_batch(
    batch: list[dict[str, Any]],
    taxonomy: Taxonomy,
) -> list[CompressedCard]:
    """压缩一批卡片。"""
    from src.agent.llm import build_reason_llm
    from src.agent.settings import get_settings
    from langchain_core.messages import HumanMessage, SystemMessage

    taxonomy_hints = ""
    if taxonomy.categories:
        taxonomy_hints = "Taxonomy hints:\n"
        for cat in taxonomy.categories[:5]:
            taxonomy_hints += f"- {cat.name}: {', '.join(cat.papers[:3])}\n"

    SYSTEM = (
        "You are a research paper summarizer. Compress each paper abstract to its core claim. "
        "The output MUST be strictly valid JSON array (no markdown code blocks).\n\n"
        "OUTPUT SCHEMA:\n"
        '  [\n'
        "    {\n"
        '      "title": "Paper Title",\n'
        '      "arxiv_id": "xxxx.xxxxx",\n'
        '      "core_claim": "One-sentence core finding",\n'
        '      "method_type": "e.g. Reinforcement Learning, Knowledge Graph, Transformer",\n'
        '      "key_result": "Key numerical result if available",\n'
        '      "role_in_taxonomy": "Category this paper belongs to",\n'
        '      "connections": ["Connection to other papers", ...]\n'
        "    },\n"
        "    ...\n"
        "  ]"
    )

    batch_text = _render_cards_for_compression(batch)

    USER = (
        f"Compress the following {len(batch)} papers into structured cards:\n"
        f"{batch_text}\n\n"
        f"{taxonomy_hints}\n"
        "Return a JSON array only."
    )

    try:
        settings = get_settings()
        llm = build_reason_llm(settings, max_tokens=4096, timeout_s=120)
        response = llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=USER)])
        content = response.content if hasattr(response, "content") else str(response)
        data = _extract_json(content)
        if data and isinstance(data, list):
            return [
                CompressedCard(
                    title=c.get("title", ""),
                    arxiv_id=c.get("arxiv_id", ""),
                    core_claim=c.get("core_claim", ""),
                    method_type=c.get("method_type", ""),
                    key_result=c.get("key_result", ""),
                    role_in_taxonomy=c.get("role_in_taxonomy", ""),
                    connections=c.get("connections", []),
                )
                for c in data
            ]
    except Exception as exc:
        logger.warning("[_compress_batch] LLM failed: %s, using fallback", exc)

    # Fallback: 简单截取
    return [
        CompressedCard(
            title=c.get("title", ""),
            arxiv_id=c.get("arxiv_id", ""),
            core_claim=_truncate(c.get("summary", c.get("abstract", "")[:300])),
            method_type=c.get("keywords", ["unknown"])[0]
            if c.get("keywords")
            else "unknown",
            key_result="",
            role_in_taxonomy="",
            connections=[],
        )
        for c in batch
    ]


def _build_evidence_pools(
    compressed: list[CompressedCard],
    taxonomy: Taxonomy,
    brief: ResearchBrief | dict[str, Any] | None,
) -> dict[str, EvidencePool]:
    """
    按 section 分配 evidence pool。

    策略：
    1. 识别每篇论文在不同 section 的相关性（通过 taxonomy 匹配）
    2. 中心论文（被多个分类引用）→ 所有相关 section 都分配 evidence
    3. 边缘论文（仅属于一个分类）→ 只在相关 section 出现
    """
    pools: dict[str, EvidencePool] = {}

    # 默认 sections
    sections = [
        "introduction",
        "background",
        "taxonomy",
        "methods",
        "datasets",
        "evaluation",
        "discussion",
        "future_work",
        "conclusion",
    ]

    # 为每个 section 创建 pool
    for section in sections:
        pools[section] = EvidencePool(
            section=section,
            token_budget=SECTION_TOKEN_BUDGETS.get(section, 5000),
            papers=[],
        )

    # 简单策略：均匀分配（每篇论文在所有 section 均匀出现）
    # 更好的策略：通过 taxonomy 匹配决定分配
    per_section_cards = 5  # 每个 section 最多 5 篇论文
    for i, card in enumerate(compressed):
        # 确定该卡片应该出现在哪些 section
        target_sections = _get_target_sections(card, taxonomy, sections)
        for section in target_sections[:per_section_cards]:
            if section in pools:
                pools[section].papers.append(
                    PoolEntry(
                        card=card,
                        allocated_chars=300,
                        focus_aspect=_get_focus_aspect(card, section),
                    )
                )

    return pools


def _get_target_sections(
    card: CompressedCard,
    taxonomy: Taxonomy,
    sections: list[str],
) -> list[str]:
    """根据论文角色和 taxonomy 确定目标 sections。"""
    role = card.role_in_taxonomy.lower()
    method = card.method_type.lower()

    # 方法类论文 → methods, evaluation, discussion
    if "method" in method or "architecture" in method:
        return ["taxonomy", "methods", "evaluation", "discussion"]
    # 数据集类论文 → datasets, evaluation
    if "dataset" in role or "benchmark" in role:
        return ["datasets", "evaluation"]
    # 综述/分析类论文 → introduction, discussion, future_work
    if "survey" in role or "review" in role:
        return ["introduction", "discussion", "future_work"]
    # 通用论文 → taxonomy, methods
    return ["taxonomy", "methods", "discussion"]


def _get_focus_aspect(card: CompressedCard, section: str) -> str:
    """根据 section 确定关注该论文的哪个方面。"""
    aspect_map = {
        "introduction": "research motivation and background",
        "background": "core concepts and task definition",
        "taxonomy": "taxonomy position and method family",
        "methods": card.core_claim,
        "datasets": "datasets and experimental setup",
        "evaluation": card.key_result or "comparative performance evidence",
        "discussion": "strengths and limitations",
        "future_work": "open problems and future directions",
        "conclusion": "main contribution",
    }
    return aspect_map.get(section, card.core_claim)


def _render_cards_for_taxonomy(cards: list[dict[str, Any]]) -> str:
    """渲染卡片列表用于 taxonomy 构建。"""
    lines = []
    for i, c in enumerate(cards, 1):
        title = c.get("title", "Unknown")
        summary = c.get("summary", c.get("abstract", ""))[:800]
        authors = c.get("authors", [])
        author_str = ", ".join(authors[:3]) if authors else "Unknown"
        lines.append(f"[{i}] {title}\nAuthors: {author_str}\nSummary: {summary}\n")
    return "\n".join(lines)


def _render_cards_for_compression(cards: list[dict[str, Any]]) -> str:
    """渲染卡片列表用于压缩。"""
    lines = []
    for c in cards:
        title = c.get("title", "Unknown")
        summary = c.get("summary", c.get("abstract", ""))[:600]
        keywords = c.get("keywords", [])
        kw_str = ", ".join(keywords[:5]) if keywords else ""
        lines.append(f"Title: {title}\nKeywords: {kw_str}\nAbstract: {summary}\n")
    return "\n---\n".join(lines)


def _card_to_text(card: dict[str, Any] | CompressedCard) -> str:
    """将原始卡片或压缩卡片转为文本（用于统计）。"""
    parts = [
        _card_field(card, "title"),
        _card_field(card, "summary") or _card_field(card, "abstract"),
        _card_field(card, "methods") or _card_field(card, "method_type"),
        _card_field(card, "datasets") or _card_field(card, "key_result"),
        _card_field(card, "core_claim"),
        _card_field(card, "role_in_taxonomy"),
        _card_field(card, "connections"),
    ]
    return " ".join(part for part in (_stringify_card_value(p) for p in parts) if part)


def _card_field(card: dict[str, Any] | CompressedCard, key: str) -> Any:
    if isinstance(card, dict):
        return card.get(key)
    return getattr(card, key, None)


def _stringify_card_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        return " ".join(
            part
            for part in (_stringify_card_value(v) for v in value.values())
            if part
        )
    if isinstance(value, (list, tuple, set)):
        return " ".join(
            part
            for part in (_stringify_card_value(item) for item in value)
            if part
        )
    return str(value)


def _truncate(text: str, max_chars: int = 300) -> str:
    """截断文本。"""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _extract_json(content: str) -> Any:
    """从 LLM 输出中提取 JSON。"""
    content = content.strip()
    # 尝试去掉 markdown code block
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:] if lines[0].startswith("```") else lines)
        if content.endswith("```"):
            content = content[:-3]
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # 尝试提取 JSON 对象或数组
        import re

        # 匹配 JSON 对象
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        # 匹配 JSON 数组
        match = re.search(r"\[[\s\S]*\]", content)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None


def _build_taxonomy_fallback(cards: list[dict[str, Any]]) -> Taxonomy:
    """当 LLM 失败时的 fallback taxonomy。"""
    # 简单按论文数量均分
    n = len(cards)
    if n == 0:
        return Taxonomy()

    # 假设 1 个通用类别
    titles = [c.get("title", "") for c in cards[:10]]
    return Taxonomy(
        categories=[
            TaxonomyCategory(
                name="General Research",
                description="Papers that do not clearly belong to a narrower subcategory",
                papers=titles,
                key_characteristics=[],
                shared_insights=[],
                conflicts=[],
            )
        ],
        cross_category_themes=[],
        timeline=[],
        key_papers=titles[:3] if titles else [],
    )
