"""Evidence Matchers — Evidence 匹配函数（模块 7）。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.corpus.search.models import EvidenceChunk
    from src.eval.rag.models import GoldEvidence


def loose_match(chunk: "EvidenceChunk | dict", gold: "GoldEvidence") -> bool:
    """
    宽松匹配：判断 predicted chunk 是否匹配 gold evidence。

    宽松策略：
    1. paper_id 完全匹配
    2. chunk 文本包含 gold evidence 关键片段（>= 5 个连续 token 匹配）
    3. section 语义重叠

    Args:
        chunk: 预测的 evidence chunk
        gold: gold 标准

    Returns:
        True if matched, False otherwise
    """
    # 检查 paper_id 匹配
    chunk_paper_id = _get_chunk_paper_id(chunk)
    if chunk_paper_id and gold.paper_id:
        if chunk_paper_id == gold.paper_id:
            # paper_id 匹配后，检查内容或 section
            chunk_text = _get_chunk_text(chunk) or ""
            gold_text = gold.text or ""

            # 检查 token 重叠
            if _token_overlap_ratio(chunk_text, gold_text) >= 0.3:
                return True

            # 检查 section 重叠
            chunk_section = _get_chunk_section(chunk) or ""
            if _section_overlap(chunk_section, gold.expected_section):
                return True

    # fallback：检查纯文本匹配（跨 paper）
    chunk_text = _get_chunk_text(chunk) or ""
    gold_text = gold.text or ""

    if _token_overlap_ratio(chunk_text, gold_text) >= 0.5:
        return True

    return False


def strict_match(chunk: "EvidenceChunk | dict", gold: "GoldEvidence") -> bool:
    """
    严格匹配：要求 paper_id + section + 高文本重叠率。

    严格策略：
    1. paper_id 必须匹配
    2. section 语义重叠
    3. token 重叠率 >= 70%
    """
    # 检查 paper_id
    chunk_paper_id = _get_chunk_paper_id(chunk)
    if not chunk_paper_id or not gold.paper_id:
        return False
    if chunk_paper_id != gold.paper_id:
        return False

    # 检查 section 重叠
    chunk_section = _get_chunk_section(chunk) or ""
    if not _section_overlap(chunk_section, gold.expected_section):
        return False

    # 检查 token 重叠率
    chunk_text = _get_chunk_text(chunk) or ""
    gold_text = gold.text or ""

    if _token_overlap_ratio(chunk_text, gold_text) >= 0.7:
        return True

    return False


def fuzzy_match(
    chunk: "EvidenceChunk | dict",
    gold: "GoldEvidence",
    threshold: float = 0.4,
) -> bool:
    """
    模糊匹配：基于 token 重叠率的可配置匹配。

    Args:
        chunk: 预测的 evidence chunk
        gold: gold 标准
        threshold: 匹配阈值（默认 0.4）
    """
    chunk_text = _get_chunk_text(chunk) or ""
    gold_text = gold.text or ""

    ratio = _token_overlap_ratio(chunk_text, gold_text)
    return ratio >= threshold


def paper_match(predicted_paper, gold_paper: "GoldPaper") -> bool:
    """
    判断 predicted paper 是否匹配 gold paper。

    匹配策略：
    1. canonical_id 完全匹配
    2. title 完全匹配（忽略大小写）
    3. title 模糊匹配（token 重叠率 >= 0.7）

    Args:
        predicted_paper: 预测的论文对象
        gold_paper: gold 标准

    Returns:
        True if matched
    """
    pred_id = _get_paper_id(predicted_paper)
    pred_title = (_get_paper_title(predicted_paper) or "").lower()
    gold_id = gold_paper.paper_id
    gold_title = (gold_paper.title or "").lower()

    # ID 完全匹配
    if pred_id and gold_id and pred_id == gold_id:
        return True

    # Title 完全匹配
    if pred_title and gold_title and pred_title == gold_title:
        return True

    # Title 模糊匹配
    ratio = _token_overlap_ratio(pred_title, gold_title)
    if ratio >= 0.7:
        return True

    return False


def get_matcher(mode: str = "loose"):
    """
    获取匹配器函数。

    Args:
        mode: 匹配模式，"loose" | "strict" | "fuzzy"

    Returns:
        匹配函数
    """
    if mode == "strict":
        return strict_match
    elif mode == "fuzzy":
        return fuzzy_match
    else:
        return loose_match


def token_overlap_ratio(text1: str, text2: str) -> float:
    """
    公开的 token 重叠率计算函数。

    重叠率 = |tokens1 ∩ tokens2| / min(|tokens1|, |tokens2|)
    """
    return _token_overlap_ratio(text1, text2)


def text_similarity(text1: str, text2: str) -> float:
    """
    文本相似度计算（Jaccard 相似度）。

    相似度 = |tokens1 ∩ tokens2| / |tokens1 ∪ tokens2|
    """
    if not text1 or not text2:
        return 0.0

    tokens1 = _tokenize(text1)
    tokens2 = _tokenize(text2)

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    return intersection / union if union > 0 else 0.0


# ── Section 标准化 ───────────────────────────────────────────────────────────


# Section 别名映射（将不同表述归一化到统一形式）
SECTION_ALIASES: dict[str, str] = {
    # Abstract
    "abstract": "abstract",
    # Introduction
    "introduction": "introduction",
    "intro": "introduction",
    # Method
    "method": "method",
    "methods": "method",
    "methodology": "method",
    "experimental setup": "method",
    "experimental design": "method",
    "experiment": "method",
    # Result
    "result": "result",
    "results": "result",
    "experimental results": "result",
    "evaluation results": "result",
    # Discussion
    "discussion": "discussion",
    "discussions": "discussion",
    # Conclusion
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "concluding remarks": "conclusion",
    # Related work
    "related work": "related work",
    "related works": "related work",
    "related work and background": "related work",
    # Background
    "background": "background",
    # Limitation
    "limitation": "limitation",
    "limitations": "limitation",
    "future work": "limitation",
}


def normalize_section(section: str) -> str:
    """
    将 section 名称标准化。

    Args:
        section: 原始 section 名称

    Returns:
        标准化后的 section 名称
    """
    if not section:
        return ""

    section_lower = section.lower().strip()

    # 精确匹配别名
    if section_lower in SECTION_ALIASES:
        return SECTION_ALIASES[section_lower]

    # 检查是否是某个标准 section 的子串
    for alias, normalized in SECTION_ALIASES.items():
        if alias in section_lower or section_lower in alias:
            return normalized

    # 无法标准化，返回原始值（小写）
    return section_lower


def section_overlap(section1: str, section2: str) -> bool:
    """
    检查两个 section 名称是否重叠（使用标准化后的名称）。

    Args:
        section1: 第一个 section
        section2: 第二个 section

    Returns:
        True if overlapped
    """
    norm1 = normalize_section(section1)
    norm2 = normalize_section(section2)
    return _section_overlap(norm1, norm2)


# ── 工具函数 ────────────────────────────────────────────────────────────────


def _get_paper_id(paper) -> str:
    """从 paper 对象提取 paper_id。"""
    if hasattr(paper, "paper_id"):
        return getattr(paper, "paper_id", "") or ""
    if isinstance(paper, dict):
        return paper.get("paper_id", "") or ""
    return ""


def _get_paper_title(paper) -> str:
    """从 paper 对象提取 title。"""
    if hasattr(paper, "title"):
        return getattr(paper, "title", "") or ""
    if isinstance(paper, dict):
        return paper.get("title", "") or ""
    return ""


def _get_chunk_paper_id(chunk) -> str:
    """从 chunk 中提取 paper_id。"""
    if hasattr(chunk, "paper_id"):
        return getattr(chunk, "paper_id", "") or ""
    if isinstance(chunk, dict):
        return chunk.get("paper_id", "") or ""
    return ""


def _get_chunk_text(chunk) -> str:
    """从 chunk 中提取文本。"""
    if hasattr(chunk, "text"):
        return getattr(chunk, "text", "") or ""
    if isinstance(chunk, dict):
        return chunk.get("text", "") or ""
    return ""


def _get_chunk_section(chunk) -> str:
    """从 chunk 中提取 section。"""
    if hasattr(chunk, "section"):
        return getattr(chunk, "section", "") or ""
    if isinstance(chunk, dict):
        return chunk.get("section", "") or ""
    return ""


def _tokenize(text: str) -> set[str]:
    """将文本分词为 token 集合（小写）。"""
    if not text:
        return set()
    # 简单分词：按空白和标点分割
    tokens = re.findall(r'\w+', text.lower())
    return set(tokens)


def _token_overlap_ratio(text1: str, text2: str) -> float:
    """
    计算两个文本的 token 重叠率。

    重叠率 = |tokens1 ∩ tokens2| / min(|tokens1|, |tokens2|)
    """
    if not text1 or not text2:
        return 0.0

    tokens1 = _tokenize(text1)
    tokens2 = _tokenize(text2)

    if not tokens1 or not tokens2:
        return 0.0

    overlap = len(tokens1 & tokens2)
    min_size = min(len(tokens1), len(tokens2))

    return overlap / min_size if min_size > 0 else 0.0


def _section_overlap(section1: str, section2: str) -> bool:
    """
    检查两个 section 名称是否语义重叠。

    重叠判断：
    1. 完全包含（包含关系）
    2. Token 级别重叠（非停用词至少重叠 1 个）
    """
    if not section1 or not section2:
        return False

    s1_lower = section1.lower()
    s2_lower = section2.lower()

    # 完全包含关系
    if s1_lower in s2_lower or s2_lower in s1_lower:
        return True

    # Token 级别重叠
    s1_tokens = set(s1_lower.split())
    s2_tokens = set(s2_lower.split())

    stopwords = {"the", "and", "of", "in", "a", "to", "for", "with", "on", "by", "an", "is"}
    significant_overlap = (s1_tokens & s2_tokens) - stopwords

    return len(significant_overlap) >= 1


def _contains_continuous_match(text: str, pattern: str, min_length: int = 5) -> bool:
    """
    检查 text 中是否包含 pattern 的连续匹配。

    连续匹配定义：连续 >= min_length 个 token 完全匹配。

    Args:
        text: 待搜索文本
        pattern: 匹配模式
        min_length: 最小连续匹配 token 数

    Returns:
        True if continuous match found
    """
    if not text or not pattern:
        return False

    # 将两个文本分词
    text_tokens = re.findall(r'\w+', text.lower())
    pattern_tokens = re.findall(r'\w+', pattern.lower())

    if len(pattern_tokens) < min_length:
        # pattern 太短，检查是否有连续子序列
        return pattern.lower() in text.lower()

    # 滑动窗口检查连续匹配
    for i in range(len(text_tokens) - len(pattern_tokens) + 1):
        window = text_tokens[i:i + len(pattern_tokens)]
        if window == pattern_tokens:
            return True

    return False
