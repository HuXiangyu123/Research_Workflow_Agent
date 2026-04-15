"""SearchPlanAgent 工具集（调用 SearXNG + 本地语料库）。"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any

from langchain_core.tools import tool

# ─── SearXNG 配置 ────────────────────────────────────────────────────────────

SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
_SEARXNG_TIMEOUT = 15  # seconds


def _searxng_search(
    query: str,
    *,
    categories: str | None = None,
    engines: str | None = None,
    language: str = "en",
    time_range: str | None = None,
    safesearch: int = 0,
    max_results: int = 10,
) -> dict[str, Any]:
    """调用 SearXNG JSON API 并统一返回格式。"""
    params: dict[str, Any] = {
        "q": query,
        "format": "json",
        "language": language,
        "safesearch": safesearch,
    }
    if categories:
        params["categories"] = categories
    if engines:
        params["engines"] = engines
    if time_range:
        params["time_range"] = time_range

    try:
        encoded = urllib.parse.urlencode(params)
        url = f"{SEARXNG_BASE_URL}/search?{encoded}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "PaperReader/1.0"},
        )
        with urllib.request.urlopen(req, timeout=_SEARXNG_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = data.get("results", [])
        suggestions = data.get("suggestions", [])

        hits = []
        for r in results[:max_results]:
            hits.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "engine": r.get("engine", ""),
                "publishedDate": r.get("publishedDate"),
                "score": r.get("score"),
            })

        return {
            "ok": True,
            "query": query,
            "total_results": len(results),
            "hits": hits,
            "suggestions": suggestions,
        }
    except urllib.error.HTTPError as e:
        return {"ok": False, "query": query, "error": f"HTTP {e.code}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "query": query, "error": str(exc)}


# ─── 工具 1：通用网络搜索（通过 SearXNG） ────────────────────────────────────


@tool
def search_arxiv(query: str, top_k: int = 10) -> str:
    """
    在学术资源中搜索论文和相关资料（arXiv、Semantic Scholar、Google Scholar 等）。
    适用于寻找论文、学术报告、技术博客。
    返回结构化的搜索结果列表。
    """
    result = _searxng_search(
        query,
        # SearXNG 多引擎并发查询容易超时，优先使用本地可用的 arXiv 单引
        engines="arxiv",
        max_results=top_k,
    )
    if not result["ok"]:
        return f"搜索失败：{result.get('error', '未知错误')}"

    lines = [f"查询：「{query}」 共找到 {result['total_results']} 条结果："]
    for i, h in enumerate(result["hits"], 1):
        lines.append(
            f"[{i}] {h['title']}\n"
            f"    链接：{h['url']}\n"
            f"    来源：{h['engine']} | 摘要：{h['content'][:200]}"
        )
    if result["suggestions"]:
        lines.append(f"\n建议词：{', '.join(result['suggestions'])}")
    return "\n".join(lines)


# ─── 工具 2：本地语料库检索 ───────────────────────────────────────────────────


@tool
def search_local_corpus(query: str, top_k: int = 8) -> str:
    """
    在已 ingestion 的本地 PDF 语料库中搜索相关段落。
    使用 BM25 + 向量混合检索，返回来自本地文档的精确匹配内容。
    """
    try:
        from src.retrieval.search import get_searcher

        searcher = get_searcher()
        results = searcher.search(query, top_k=top_k)

        if not results:
            return "本地语料库中未找到相关内容。"

        lines = [f"本地检索：「{query}」 共 {len(results)} 条匹配："]
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] {r.get('title', 'Unknown')} "
                f"(Page {r.get('page_start', '?')})\n"
                f"    来源：{r.get('source_uri', '')}\n"
                f"    内容：{r.get('text', '')[:300]}"
            )
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"本地检索出错：{exc}"


# ─── 工具 3：元数据搜索 ───────────────────────────────────────────────────────


@tool
def search_metadata_only(query: str, top_k: int = 10) -> str:
    """
    仅搜索论文元数据（标题、作者、年份、摘要），不返回正文内容。
    适用于快速了解某个方向有哪些论文，而不深入内容。
    """
    result = _searxng_search(
        query,
        engines="arxiv",  # 仅使用本地可用的 arXiv 引擎，避免并发超时
        max_results=top_k,
    )
    if not result["ok"]:
        return f"搜索失败：{result.get('error', '未知错误')}"

    lines = [f"元数据检索：「{query}」 共 {result['total_results']} 条："]
    for i, h in enumerate(result["hits"], 1):
        lines.append(
            f"[{i}] {h['title']}\n"
            f"    链接：{h['url']}\n"
            f"    来源引擎：{h['engine']}"
        )
    return "\n".join(lines)


# ─── 工具 4：关键词扩展 ───────────────────────────────────────────────────────


@tool
def expand_keywords(topic: str, focus_dimension: str = "methods") -> str:
    """
    根据给定主题和关注维度，扩展生成相关关键词列表。
    focus_dimension 可选：methods（方法）、datasets（数据集）、applications（应用）、benchmarks（基准）。
    返回扩展后的关键词和同义词列表。
    """
    # 关键词扩展通过调用 LLM 完成
    from src.agent.llm import build_chat_llm
    from src.agent.settings import Settings

    settings = Settings.from_env()
    llm = build_chat_llm(settings, max_tokens=2048)

    prompt = (
        f"为研究主题「{topic}」在「{focus_dimension}」维度上扩展关键词和同义词。\n"
        "请列出 10-15 个相关术语（中英文均可），包括：\n"
        "1. 核心术语\n2. 相关子领域术语\n3. 常用缩写\n"
        "格式：\n- term: ...\n- term: ...\n"
        "不要解释，直接输出列表。"
    )
    from langchain_core.messages import HumanMessage, SystemMessage

    resp = llm.invoke([SystemMessage(content="你是一个关键词扩展专家。"), HumanMessage(content=prompt)])
    return resp.content if hasattr(resp, "content") else str(resp)


# ─── 工具 5：Query 重写 ───────────────────────────────────────────────────────


@tool
def rewrite_query(query: str, mode: str = "precise") -> str:
    """
    将自然语言查询重写为更适合检索的形式。
    mode 可选：precise（精确化）、broader（扩展）、alternative（换一种说法）。
    返回重写后的查询。
    """
    from src.agent.llm import build_chat_llm
    from src.agent.settings import Settings

    settings = Settings.from_env()
    llm = build_chat_llm(settings, max_tokens=1024)

    mode_map = {
        "precise": "将查询变得更精确、具体，去除歧义，加入技术术语",
        "broader": "将查询扩展到更广泛的领域，使用上位词和近义词",
        "alternative": "用完全不同的表述方式重写，保持语义等价",
    }
    instruction = mode_map.get(mode, mode_map["precise"])

    prompt = (
        f"原始查询：「{query}」\n"
        f"重写模式：{instruction}\n\n"
        f"请生成 1 个符合上述模式的重写查询。直接输出查询文本，不要解释。"
    )
    from langchain_core.messages import HumanMessage, SystemMessage

    resp = llm.invoke([SystemMessage(content="你是一个检索查询优化专家。"), HumanMessage(content=prompt)])
    return resp.content if hasattr(resp, "content") else str(resp)


# ─── 工具 6：合并重复查询 ────────────────────────────────────────────────────


@tool
def merge_duplicate_queries(query_list: list[str]) -> str:
    """
    分析一组查询，将语义重复的合并为一条，最终输出去重后的查询列表。
    返回 JSON 格式的合并结果。
    """
    from src.agent.llm import build_chat_llm
    from src.agent.settings import Settings

    settings = Settings.from_env()
    llm = build_chat_llm(settings, max_tokens=2048)

    prompt = (
        f"以下是一组候选检索查询：\n" + "\n".join(f"- {q}" for q in query_list) + "\n\n"
        "请分析这些查询，将语义相同或高度重叠的合并为一条。\n"
        "输出格式（严格 JSON）：\n"
        '{"merged": ["合并后查询1", "合并后查询2", ...], "rationale": "简要说明合并逻辑"}'
    )
    from langchain_core.messages import HumanMessage, SystemMessage

    resp = llm.invoke([SystemMessage(content="你是一个检索查询优化专家。"), HumanMessage(content=prompt)])
    return resp.content if hasattr(resp, "content") else str(resp)


# ─── 工具 7：命中摘要 ─────────────────────────────────────────────────────────


@tool
def summarize_hits(results: str) -> str:
    """
    对一组搜索结果进行摘要归纳，返回：覆盖主题、高质量结果、低质量结果、缺失角度。
    输入为原始搜索结果文本，输出为结构化摘要。
    """
    from src.agent.llm import build_chat_llm
    from src.agent.settings import Settings

    settings = Settings.from_env()
    llm = build_chat_llm(settings, max_tokens=2048)

    prompt = (
        f"以下是搜索结果：\n{results}\n\n"
        "请对上述搜索结果进行摘要分析，输出以下内容（严格 JSON）：\n"
        "{\n"
        '  "covered_topics": ["已覆盖主题1", ...],\n'
        '  "high_quality_hits": ["高质量结果标题1", ...],\n'
        '  "low_quality_hits": ["低质量/噪声结果标题1", ...],\n'
        '  "missing_angles": ["缺失角度1", ...]\n'
        "}"
    )
    from langchain_core.messages import HumanMessage, SystemMessage

    resp = llm.invoke([SystemMessage(content="你是一个信息分析专家。"), HumanMessage(content=prompt)])
    return resp.content if hasattr(resp, "content") else str(resp)


# ─── 工具 8：覆盖率评估 ───────────────────────────────────────────────────────


@tool
def estimate_subquestion_coverage(
    results: str, sub_questions: list[str]
) -> str:
    """
    评估搜索结果对子问题的覆盖程度。
    输入：原始搜索结果文本 + 子问题列表。
    输出：每个子问题的覆盖评分（0-1）和简短说明。
    """
    from src.agent.llm import build_chat_llm
    from src.agent.settings import Settings

    settings = Settings.from_env()
    llm = build_chat_llm(settings, max_tokens=2048)

    questions_str = "\n".join(f"- {q}" for q in sub_questions)
    prompt = (
        f"搜索结果：\n{results}\n\n"
        f"待评估的子问题：\n{questions_str}\n\n"
        "请评估每个子问题的搜索覆盖程度，输出格式（严格 JSON）：\n"
        "{\n"
        '  "coverage": [\n'
        '    {"question": "子问题1", "score": 0.0-1.0, "reason": "..."},\n'
        "    ...\n"
        "  ]\n"
        "}"
    )
    from langchain_core.messages import HumanMessage, SystemMessage

    resp = llm.invoke([SystemMessage(content="你是一个信息分析专家。"), HumanMessage(content=prompt)])
    return resp.content if hasattr(resp, "content") else str(resp)


# ─── 工具 9：噪声/稀疏检测 ───────────────────────────────────────────────────


@tool
def detect_sparse_or_noisy_queries(results: str) -> str:
    """
    检测搜索结果中的稀疏查询和噪声。
    输入为原始搜索结果文本。
    输出为：稀疏查询列表、噪声结果列表、改进建议。
    """
    from src.agent.llm import build_chat_llm
    from src.agent.settings import Settings

    settings = Settings.from_env()
    llm = build_chat_llm(settings, max_tokens=2048)

    prompt = (
        f"搜索结果：\n{results}\n\n"
        "请分析这些搜索结果并输出（严格 JSON）：\n"
        "{\n"
        '  "sparse_queries": ["导致稀疏结果的查询1", ...],\n'
        '  "noisy_results": ["噪声结果标题1", ...],\n'
        '  "improvement_suggestions": ["改进建议1", ...]\n'
        "}"
    )
    from langchain_core.messages import HumanMessage, SystemMessage

    resp = llm.invoke([SystemMessage(content="你是一个信息分析专家。"), HumanMessage(content=prompt)])
    return resp.content if hasattr(resp, "content") else str(resp)
