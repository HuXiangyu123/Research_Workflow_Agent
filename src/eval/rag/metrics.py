"""Metrics — 四层指标计算（模块 7）。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.eval.rag.models import (
        RagEvalCase,
        EvalCaseResult,
        PaperRetrievalMetrics,
        PaperRankingMetrics,
        EvidenceRetrievalMetrics,
        CitationGroundingMetrics,
        RetrievalStrategy,
    )
    from src.corpus.search.models import EvidenceChunk
    from src.corpus.search.deduper import DedupedCandidate

logger = logging.getLogger(__name__)

# ── NDCG helpers ──────────────────────────────────────────────────────────────


def _dcg(relevance: list[int | float]) -> float:
    """
    计算 DCG（Differential Cumulative Gain）。

    DCG[i] = rel[i] / log2(i+2)，其中 i 从 0 开始计数。
    """
    total = 0.0
    for i, rel in enumerate(relevance):
        total += rel / _log2(i + 2)
    return total


def _idcg(relevance: list[int | float]) -> float:
    """
    计算 Ideal DCG：按 relevance 降序排列后的 DCG。

    这是理论最优排序下的 DCG 值。
    """
    sorted_rel = sorted(relevance, reverse=True)
    return _dcg(sorted_rel)


def _log2(n: int) -> float:
    """计算 log2，安全处理 n <= 0 的情况。"""
    import math
    return math.log2(n) if n > 0 else 0.0


def ndcg(relevance: list[int | float], k: int | None = None) -> float:
    """
    计算 NDCG@k（Normalized Discounted Cumulative Gain）。

    NDCG@k = DCG@k / IDCG@k

    Args:
        relevance: 排序列表，每项 1=relevant, 0=not relevant
        k: 截断位置，None 表示计算全部

    Returns:
        NDCG@k 值，范围 [0, 1]
    """
    if k is not None:
        relevance = relevance[:k]
    dcg_val = _dcg(relevance)
    idcg_val = _idcg(relevance)
    if idcg_val == 0:
        return 0.0
    return dcg_val / idcg_val


def mrr(relevance: list[int | float]) -> float:
    """
    计算 MRR（Mean Reciprocal Rank）。

    MRR = 1 / position_of_first_relevant_item

    Returns:
        第一个相关项位置的倒数，无相关项则返回 0.0
    """
    for i, rel in enumerate(relevance):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def ap(relevance: list[int | float]) -> float:
    """
    计算 AP（Average Precision）。

    AP = Σ (P@k × rel@k) / #relevant

    其中 P@k = relevant_items_up_to_k / k

    Args:
        relevance: 排序列表，每项 1=relevant, 0=not relevant

    Returns:
        平均精确率，范围 [0, 1]
    """
    relevant_count = sum(1 for r in relevance if r > 0)
    if relevant_count == 0:
        return 0.0
    total = 0.0
    correct = 0
    for i, rel in enumerate(relevance):
        if rel > 0:
            correct += 1
            total += correct / (i + 1)
    return total / relevant_count


def map_k(relevance: list[int | float], k: int | None = None) -> float:
    """
    计算 MAP@k（Mean Average Precision@k）。

    Args:
        relevance: 排序列表
        k: 截断位置

    Returns:
        MAP@k 值
    """
    if k is not None:
        relevance = relevance[:k]
    return ap(relevance)


# ── Layer 1: Paper Retrieval Metrics ─────────────────────────────────────────


def compute_paper_retrieval_metrics(
    predicted_papers: list,  # list of DedupedCandidate or paper dict
    gold: "RagEvalCase",
    k: int = 10,
) -> "PaperRetrievalMetrics":
    """
    计算 Layer 1 指标：Paper Retrieval（论文召回）。

    评估检索系统召回目标论文的能力。

    Args:
        predicted_papers: 检索返回的论文列表（按得分降序）
        gold: gold 标准
        k: 评测的 top-k 截断位置（默认 10）

    Returns:
        PaperRetrievalMetrics 对象
    """
    from src.eval.rag.models import PaperRetrievalMetrics

    gold_ids = gold.gold_paper_ids()
    gold_titles_lower = gold.gold_paper_titles()

    # 构建 relevance 列表（1=gold, 0=not gold）
    relevance_at_k = []
    predicted_ids = []
    for p in predicted_papers:
        pid = _get_paper_id(p)
        title = (_get_paper_title(p) or "").lower()
        predicted_ids.append(pid)

        if pid in gold_ids or title in gold_titles_lower:
            relevance_at_k.append(1)
        else:
            relevance_at_k.append(0)

    # 统计 gold 召回情况
    total_gold = len(gold_ids)
    recalled = sum(relevance_at_k)
    recalled_10 = sum(relevance_at_k[:10])
    recalled_50 = sum(relevance_at_k[:50])
    recalled_100 = sum(relevance_at_k[:100])

    return PaperRetrievalMetrics(
        paper_recall_10=recalled_10 / max(total_gold, 1),
        paper_recall_50=recalled_50 / max(total_gold, 1),
        paper_recall_100=recalled_100 / max(total_gold, 1),
        paper_mrr=mrr(relevance_at_k),
        paper_ndcg_10=ndcg(relevance_at_k, k=10),
        paper_ndcg_50=ndcg(relevance_at_k, k=50),
        paper_map_10=map_k(relevance_at_k, k=10),
        total_gold_papers=total_gold,
        retrieved_gold_papers=recalled,
    )


# ── Layer 2: Paper Ranking Metrics ───────────────────────────────────────────


def compute_paper_ranking_metrics(
    predicted_papers: list,
    gold: "RagEvalCase",
    baseline_rrf_scores: list[float] | None = None,
) -> "PaperRankingMetrics":
    """
    计算 Layer 2 指标：Paper Ranking（论文排序质量）。

    评估 dedup + rerank 后的排序质量。

    Args:
        predicted_papers: 去重+重排后的论文列表
        gold: gold 标准
        baseline_rrf_scores: 可选的 RRF baseline 分数（用于计算 rerank 改进）

    Returns:
        PaperRankingMetrics 对象
    """
    from src.eval.rag.models import PaperRankingMetrics

    gold_ids = gold.gold_paper_ids()
    gold_titles_lower = gold.gold_paper_titles()

    relevance = []
    for p in predicted_papers:
        pid = _get_paper_id(p)
        title = (_get_paper_title(p) or "").lower()
        relevance.append(1 if (pid in gold_ids or title in gold_titles_lower) else 0)

    deduped_count = len(predicted_papers)
    original_count = deduped_count  # 简化：实际需要传入原始数量

    return PaperRankingMetrics(
        ranking_ndcg_10=ndcg(relevance, k=10),
        ranking_ndcg_50=ndcg(relevance, k=50),
        ranking_mrr=mrr(relevance),
        dedup_precision=1.0,  # 简化：需要传入 dedup_info
        rerank_improvement=0.0,  # 需要对比 baseline
        deduped_count=deduped_count,
        original_count=original_count,
    )


# ── Layer 3: Evidence Retrieval Metrics ──────────────────────────────────────


def compute_evidence_retrieval_metrics(
    predicted_chunks: list,  # list of EvidenceChunk or ChunkSearchResult
    gold: "RagEvalCase",
    matcher_fn=None,
) -> "EvidenceRetrievalMetrics":
    """
    计算 Layer 3 指标：Evidence Retrieval（证据块召回）。

    评估在已选论文内部检索具体证据的能力。

    Args:
        predicted_chunks: 检索到的 evidence chunks
        gold: gold 标准
        matcher_fn: 匹配函数，默认使用 loose_match

    Returns:
        EvidenceRetrievalMetrics 对象
    """
    from src.eval.rag.models import EvidenceRetrievalMetrics

    if matcher_fn is None:
        from src.eval.rag.matchers import loose_match
        matcher_fn = loose_match

    gold_evidence = gold.gold_evidence
    total_gold = len(gold_evidence)

    if total_gold == 0:
        return EvidenceRetrievalMetrics(total_gold_evidence=0)

    # 对每个 predicted chunk，判断它匹配了哪个 gold evidence
    matched_gold_indices: set[int] = set()
    for chunk in predicted_chunks:
        for i, ge in enumerate(gold_evidence):
            if i not in matched_gold_indices and matcher_fn(chunk, ge):
                matched_gold_indices.add(i)
                break

    recalled = len(matched_gold_indices)

    # Section hit 分析
    section_hits = 0
    type_correct = 0
    for i, ge in enumerate(gold_evidence):
        if i in matched_gold_indices:
            section_hits += 1
            # 检查 predicted chunk 的 section 是否命中
            for chunk in predicted_chunks:
                chunk_section = _get_chunk_section(chunk) or ""
                if _section_overlap(chunk_section, ge.expected_section):
                    type_correct += 1
                    break

    # 计算 coverage
    predicted_sections = [_get_chunk_section(c) for c in predicted_chunks]
    method_hit = any(_section_overlap(s, "method") for s in predicted_sections)
    result_hit = any(_section_overlap(s, "result") for s in predicted_sections)
    gold_has_method = any(
        _section_overlap(ge.expected_section, "method") for ge in gold_evidence
    )
    gold_has_result = any(
        _section_overlap(ge.expected_section, "result") for ge in gold_evidence
    )

    return EvidenceRetrievalMetrics(
        evidence_recall_10=min(recalled / max(total_gold, 1), 1.0),
        evidence_recall_25=min(recalled / max(total_gold, 1), 1.0),
        evidence_recall_50=min(recalled / max(total_gold, 1), 1.0),
        evidence_precision_10=recalled / max(len(predicted_chunks[:10]), 1),
        evidence_precision_25=recalled / max(len(predicted_chunks[:25]), 1),
        support_span_hit_rate=section_hits / max(total_gold, 1),
        evidence_type_accuracy=type_correct / max(recalled, 1) if recalled > 0 else 0.0,
        method_coverage=1.0 if (not gold_has_method or method_hit) else 0.0,
        result_coverage=1.0 if (not gold_has_result or result_hit) else 0.0,
        total_gold_evidence=total_gold,
        retrieved_gold_evidence=recalled,
    )


# ── Layer 4: Citation / Grounding Metrics ─────────────────────────────────────


def compute_citation_grounding_metrics(
    citations: list[dict],  # [{url, reachable, claim_text, supported}]
    gold: "RagEvalCase",
) -> "CitationGroundingMetrics":
    """
    计算 Layer 4 指标：Citation / Grounding（引用支撑）。

    评估引用可达性和声明支撑能力。

    Args:
        citations: 从 verified_report 或 resolved_report 中提取的 citation 列表
        gold: gold 标准

    Returns:
        CitationGroundingMetrics 对象
    """
    from src.eval.rag.models import CitationGroundingMetrics

    if not citations:
        return CitationGroundingMetrics()

    total_citations = len(citations)
    reachable = sum(1 for c in citations if c.get("reachable", False))

    total_claims = len(gold.gold_claims)
    supported = 0
    for gc in gold.gold_claims:
        if gc.claim_text:
            supported += 1  # 简化：gold claim 假设全部支撑

    unsupported = max(0, total_claims - supported)
    grounding = (supported - unsupported) / max(total_claims, 1)

    return CitationGroundingMetrics(
        citation_reachability=reachable / max(total_citations, 1),
        supported_claim_rate=supported / max(total_claims, 1),
        unsupported_claim_rate=unsupported / max(total_claims, 1),
        coverage_gap_count=max(0, len(gold.sub_questions) - supported),
        grounding_score=max(0.0, grounding),
        total_citations=total_citations,
        reachable_citations=reachable,
        total_claims=total_claims,
        supported_claims=supported,
    )


# ── 工具函数 ────────────────────────────────────────────────────────────────


def _get_paper_id(p) -> str:
    """
    从 paper 对象中提取 canonical_id 或 doc_id。

    支持 DedupedCandidate、dict 等多种格式。
    """
    if hasattr(p, "canonical_id"):
        return getattr(p, "canonical_id", "") or ""
    if hasattr(p, "primary_doc_id"):
        return getattr(p, "primary_doc_id", "") or ""
    if isinstance(p, dict):
        return p.get("canonical_id", "") or p.get("doc_id", "") or ""
    return ""


def _get_paper_title(p) -> str:
    """从 paper 对象中提取 title。"""
    if hasattr(p, "title"):
        return getattr(p, "title", "") or ""
    if isinstance(p, dict):
        return p.get("title", "") or ""
    return ""


def _get_chunk_section(chunk) -> str:
    """从 chunk 对象中提取 section。"""
    if hasattr(chunk, "section"):
        return getattr(chunk, "section", "") or ""
    if isinstance(chunk, dict):
        return chunk.get("section", "") or ""
    return ""


def _section_overlap(section1: str, section2: str) -> bool:
    """
    检查两个 section 名称是否语义重叠。

    重叠判断：
    1. 完全包含关系
    2. Token 级别重叠（非停用词至少重叠 1 个）
    """
    if not section1 or not section2:
        return False
    s1_tokens = set(section1.lower().split())
    s2_tokens = set(section2.lower().split())
    overlap = s1_tokens & s2_tokens
    # 重叠至少一个非停用词 token
    stopwords = {"the", "and", "of", "in", "a", "to", "for", "with", "on", "by"}
    significant_overlap = overlap - stopwords
    return len(significant_overlap) >= 1


# ── 统一入口 ────────────────────────────────────────────────────────────────


def compute_all_metrics(
    predicted_papers: list,
    predicted_chunks: list,
    citations: list[dict],
    gold: "RagEvalCase",
    baseline_rrf_scores: list[float] | None = None,
) -> "EvalCaseResult":
    """
    计算单条 case 的所有四层指标。

    这是模块 7 的主要对外接口。

    Args:
        predicted_papers: 检索到的论文列表
        predicted_chunks: 检索到的 evidence chunks
        citations: 引用列表（从 verified_report 中提取）
        gold: gold 标准
        baseline_rrf_scores: 可选的 RRF baseline（用于计算 rerank 改进）

    Returns:
        EvalCaseResult 对象，包含所有四层指标
    """
    from src.eval.rag.models import EvalCaseResult

    paper_retrieval = compute_paper_retrieval_metrics(predicted_papers, gold)
    paper_ranking = compute_paper_ranking_metrics(
        predicted_papers, gold, baseline_rrf_scores
    )
    evidence_retrieval = compute_evidence_retrieval_metrics(predicted_chunks, gold)
    citation_grounding = compute_citation_grounding_metrics(citations, gold)

    predicted_ids = [_get_paper_id(p) for p in predicted_papers]
    predicted_sections = [_get_chunk_section(c) for c in predicted_chunks]

    return EvalCaseResult(
        case_id=gold.case_id,
        strategy="hybrid_basic",
        paper_retrieval=paper_retrieval,
        paper_ranking=paper_ranking,
        evidence_retrieval=evidence_retrieval,
        citation_grounding=citation_grounding,
        predicted_paper_ids=predicted_ids,
        predicted_evidence_sections=predicted_sections,
    )
