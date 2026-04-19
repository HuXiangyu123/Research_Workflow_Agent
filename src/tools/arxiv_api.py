"""arXiv API 工具 — 直接调用 arXiv 原生 API 获取 paper metadata（绕过 SearXNG）。"""

from __future__ import annotations

import logging
import re
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

ARXIV_API_BASE = "https://export.arxiv.org/api/query"
_ARXIV_TIMEOUT = 20  # seconds per request
_ARXIV_TIMEOUT_DIRECT = 60  # seconds for direct search (arXiv 在部分网络环境下响应较慢)
_MAX_IDS_PER_REQUEST = 50  # arXiv API 单次最多 50 个 ID


def _extract_arxiv_id_from_url(url: str) -> str | None:
    """从 URL 中提取 arXiv ID（支持多种格式）。"""
    patterns = [
        r"arxiv\.org/(?:abs|pdf)/([a-z\-]+(?:\.[a-z\-]+)?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?",
        r"export\.arxiv\.org/abs/([a-z\-]+(?:\.[a-z\-]+)?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?",
        r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return _strip_version(m.group(1))
    return None


def _strip_version(aid: str) -> str:
    """去除 arXiv ID 的版本后缀（如 2301.01234v1 → 2301.01234）。"""
    return re.sub(r"v\d+$", "", aid)


def _extract_arxiv_id_from_text(text: str) -> str | None:
    """从文本中提取 arXiv ID（支持多种格式）。"""
    patterns = [
        r"arXiv:\s*([a-z\-]+(?:\.[a-z\-]+)?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?",
        r"arxiv\.org/\S*?([a-z\-]+(?:\.[a-z\-]+)?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?",
        r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            aid = _strip_version(m.group(1))
            if re.match(r"([a-z\-]+(?:\.[a-z\-]+)?/\d{7}|\d{4}\.\d{4,5})$", aid):
                return aid
    return None


def _parse_arxiv_entry(entry_text: str, arxiv_id: str) -> dict[str, Any]:
    """解析单篇 arXiv 条目的 XML 文本。"""
    title_m = re.search(r"<title>\s*(.+?)\s*</title>", entry_text, re.DOTALL)
    summary_m = re.search(r"<summary>\s*(.+?)\s*</summary>", entry_text, re.DOTALL)
    authors = re.findall(r"<name>(.+?)</name>", entry_text)
    published_m = re.search(r"<published>(.+?)</published>", entry_text)
    updated_m = re.search(r"<updated>(.+?)</updated>", entry_text)
    comment_m = re.search(r"<arxiv:comment[^>]*>(.+?)</arxiv:comment>", entry_text, re.DOTALL)
    journal_m = re.search(r"<arxiv:journal_ref[^>]*>(.+?)</arxiv:journal_ref>", entry_text, re.DOTALL)
    doi_m = re.search(r"<arxiv:doi>(.+?)</arxiv:doi>", entry_text)
    categories = re.findall(r"<category term=\"([^\"]+)\"", entry_text)

    title = title_m.group(1).replace("\n", " ").strip() if title_m else ""
    abstract = summary_m.group(1).replace("\n", " ").strip() if summary_m else ""
    published = published_m.group(1)[:10] if published_m else None  # 取日期部分

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "authors": [a.strip() for a in authors],
        "published_date": published,
        "updated_date": updated_m.group(1)[:10] if updated_m else None,
        "categories": categories,
        "comment": comment_m.group(1).strip() if comment_m else None,
        "journal_ref": journal_m.group(1).strip() if journal_m else None,
        "doi": doi_m.group(1).strip() if doi_m else None,
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
    }


def fetch_arxiv_papers_by_ids(arxiv_ids: list[str]) -> dict[str, dict[str, Any]]:
    """
    批量从 arXiv API 获取 paper metadata。

    支持最多 _MAX_IDS_PER_REQUEST 个 ID（单次请求）。
    自动按批次拆分，自动过滤空 ID 和重复 ID。

    Returns:
        dict: {arxiv_id -> metadata_dict}，不包含获取失败的 ID
    """
    # 去重 + 过滤空值
    unique_ids = list(dict.fromkeys(aid.strip() for aid in arxiv_ids if aid and aid.strip()))

    if not unique_ids:
        return {}

    results: dict[str, dict[str, Any]] = {}

    # 按批次处理
    for i in range(0, len(unique_ids), _MAX_IDS_PER_REQUEST):
        batch = unique_ids[i : i + _MAX_IDS_PER_REQUEST]
        batch_results = _fetch_arxiv_batch(batch)
        results.update(batch_results)

    return results


def _fetch_arxiv_batch(arxiv_ids: list[str]) -> dict[str, dict[str, Any]]:
    """单次请求获取一批 arXiv paper metadata（含熔断保护）。"""
    if not arxiv_ids:
        return {}

    from src.agent.circuit_breaker import get_breaker

    breaker = get_breaker("arxiv")

    if not breaker.can_execute():
        logger.warning("[arXiv API] circuit OPEN, skipping fetch for %d IDs", len(arxiv_ids))
        return {}

    id_list_str = ",".join(arxiv_ids)
    params = urllib.parse.urlencode({"id_list": id_list_str, "start": 0, "max_results": len(arxiv_ids)})
    url = f"{ARXIV_API_BASE}?{params}"

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "PaperReader/1.0 (mailto:support@paperreader.ai)",
                "Accept": "application/atom+xml",
            },
        )
        with urllib.request.urlopen(req, timeout=_ARXIV_TIMEOUT) as resp:
            xml_text = resp.read().decode("utf-8")
        breaker.record_success()
    except Exception as exc:
        logger.warning("[arXiv API] fetch failed for %d IDs: %s", len(arxiv_ids), exc)
        breaker.record_failure()
        return {}

    # 解析 XML
    # 每篇论文条目以 <entry> 标签包裹
    results: dict[str, dict[str, Any]] = {}
    entries = re.split(r"<entry>", xml_text)
    entries = entries[1:]  # 去掉 XML 头部

    for entry in entries:
        # 提 arXiv ID（从 id 字段，如 https://arxiv.org/abs/2301.01234）
        id_m = re.search(
            r"<id>(https?://arxiv\.org/abs/(([a-z\-]+(?:\.[a-z\-]+)?/\d{7})|(\d{4}\.\d{4,5}))(?:v\d+)?)</id>",
            entry,
        )
        if not id_m:
            continue
        arxiv_id = _strip_version(id_m.group(2))

        meta = _parse_arxiv_entry(entry, arxiv_id)
        results[arxiv_id] = meta

    logger.debug("[arXiv API] fetched %d/%d papers", len(results), len(arxiv_ids))
    return results


def enrich_search_results_with_arxiv(
    candidates: list[dict[str, Any]],
    max_workers: int = 8,
) -> list[dict[str, Any]]:
    """
    为主搜索结果（来自 SearXNG）补充 arXiv metadata。

    策略：
    1. 优先从 URL 提取 arXiv ID（最准确）
    2. fallback：从 content/text 提取
    3. 对所有唯一 arXiv ID 批量调用 arXiv API
    4. 将 metadata 合并到 candidates

    Returns:
         enriched candidates（已去重）
    """
    # ── Step 1：提取所有 arXiv ID ──────────────────────────────────
    enriched: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    # 先去重，同时收集所有候选的 arXiv ID
    deduped_candidates: list[dict[str, Any]] = []
    for cand in candidates:
        url = cand.get("url", "")
        content = cand.get("content", "") or cand.get("abstract", "")
        aid = _strip_version(str(cand.get("arxiv_id") or "").strip()) if cand.get("arxiv_id") else None
        if not aid:
            aid = _extract_arxiv_id_from_url(url)
        if not aid:
            aid = _extract_arxiv_id_from_text(content)
        if not aid:
            aid = _extract_arxiv_id_from_text(cand.get("title", ""))

        key = aid or url or cand.get("title", "")
        if key and key not in seen_ids:
            seen_ids.add(key)
            cand["_arxiv_id"] = aid
            deduped_candidates.append(cand)

    # ── Step 2：收集所有唯一 arXiv ID ─────────────────────────────
    all_ids = [c.get("_arxiv_id") for c in deduped_candidates if c.get("_arxiv_id")]
    all_ids = list(dict.fromkeys(all_ids))  # 保持顺序去重

    if not all_ids:
        logger.warning("[enrich_arxiv] no arXiv IDs found in %d candidates", len(candidates))
        return deduped_candidates

    # ── Step 3：批量获取 arXiv metadata ────────────────────────────
    meta_map: dict[str, dict[str, Any]] = {}
    if len(all_ids) <= 10:
        # 小批量直接同步请求
        meta_map = fetch_arxiv_papers_by_ids(all_ids)
    else:
        # 大批量并行请求（每块 50 个）
        with ThreadPoolExecutor(max_workers=min(max_workers, (len(all_ids) + 49) // 50)) as pool:
            futures = {}
            for i in range(0, len(all_ids), _MAX_IDS_PER_REQUEST):
                batch = all_ids[i : i + _MAX_IDS_PER_REQUEST]
                futures[pool.submit(fetch_arxiv_papers_by_ids, batch)] = batch
            for future in as_completed(futures):
                try:
                    meta_map.update(future.result())
                except Exception:
                    pass

    # ── Step 4：合并 metadata 到 candidates ────────────────────────
    for cand in deduped_candidates:
        aid = cand.get("_arxiv_id")
        if aid:
            cand["arxiv_id"] = aid
            cand.setdefault("pdf_url", f"https://arxiv.org/pdf/{aid}.pdf")
            if not cand.get("url"):
                cand["url"] = f"https://arxiv.org/abs/{aid}"
        if aid and aid in meta_map:
            meta = meta_map[aid]
            cand["title"] = meta.get("title") or cand.get("title", "")
            cand["abstract"] = meta.get("abstract") or cand.get("abstract", "") or cand.get("content", "")
            cand["authors"] = meta.get("authors", [])
            cand["published_date"] = meta.get("published_date")
            cand["categories"] = meta.get("categories", [])
            cand["doi"] = meta.get("doi")
            cand["comment"] = meta.get("comment")
            cand["url"] = meta.get("url", cand.get("url", ""))
            cand["pdf_url"] = meta.get("pdf_url")
            cand["arxiv_id"] = aid
        else:
            # 没有 API metadata 时，做 fallback title 修复
            title = cand.get("title", "")
            if not title or "search_query=" in title or "arXiv Query:" in title:
                content = cand.get("content", "") or ""
                first_line = content.strip().split("\n")[0].strip()
                first_line = re.sub(r"^(arXiv:|arxiv:|Title:|标题：)\s*", "", first_line, flags=re.IGNORECASE)
                if 5 < len(first_line) < 300:
                    cand["title"] = first_line
                elif aid:
                    cand["title"] = f"arXiv:{aid}"

        cand.pop("_arxiv_id", None)
        cand.pop("_meta_fetched", None)
        enriched.append(cand)

    logger.info(
        "[enrich_arxiv] %d candidates → %d unique, %d enriched with arXiv metadata",
        len(candidates), len(deduped_candidates), len(meta_map),
    )
    return enriched


# ── 默认年份过滤（现代 AI 研究大多在 2020 年以后）──
DEFAULT_YEAR_FILTER = "2020"
# 噪声关键词：标题含这些词的论文与研究方向无关（multi-agent system 等旧论文）
_NOISE_TITLE_KEYWORDS = [
    "multi-agent system", "multi agent system", "agent-based simulation",
    "multi-agent reinforcement learning", "distributed agent",
    "agent simulation", "mobile agent", "software agent",
    "intelligent agent", "cooperative agent",
]


def _get_arxiv_direct_breaker() -> "CircuitBreaker":
    """获取 arXiv direct 搜索的熔断器。"""
    from src.agent.circuit_breaker import get_breaker
    return get_breaker("arxiv", "direct")


def _is_noisy_paper(entry: dict) -> bool:
    """判断是否为噪声论文（标题含噪声关键词或过老）。"""
    title = (entry.get("title") or "").lower()
    published = entry.get("published_date", "")[:4]

    # 过老论文（2020 年以前，且标题含 agent）
    if published and published < "2020":
        if "agent" in title and "llm" not in title and "language model" not in title:
            return True

    # 标题含噪声关键词
    for noise in _NOISE_TITLE_KEYWORDS:
        if noise in title:
            return True

    return False


def search_arxiv_direct(
    query: str,
    max_results: int = 10,
    year_filter: str | None = None,
    filter_noise: bool = True,
) -> list[dict[str, Any]]:
    """
    直接通过 arXiv API 搜索论文（不需要 SearXNG）。

    支持：
    - 关键词搜索（all 字段）
    - 时间过滤（submittedDate）：默认 2020 年（现代 AI 研究大多在 2020 年以后）
    - 噪声过滤：自动过滤标题含 "multi-agent system" 等旧论文
    - 排序（relevance / lastUpdatedDate / submittedDate）

    Returns:
        list of paper dicts with full metadata
    """
    # ── 构造 URL ────────────────────────────────────────────────────────────
    # 策略：先组合完整 search_query，再整体 encode，避免二次编码
    #       例：query="benchmark" → all:benchmark AND submittedDate:[2022 TO 2026]
    #       → all%3Abenchmark+AND+submittedDate%3A%5B2022+TO+2026%5D
    effective_year = year_filter if year_filter else DEFAULT_YEAR_FILTER

    # 噪声查询增加检索数量，过滤后保留
    q_lower = query.lower().strip()
    is_noise_query = q_lower in {"agent", "ai agent", "aiagent", "agents"}
    effective_max = max_results * 3 if is_noise_query else max_results

    # 完整 search_query 字符串（raw，未编码）
    raw_search_query = f"all:{query}"
    if effective_year:
        raw_search_query += f" AND submittedDate:[{effective_year} TO {datetime.now(timezone.utc).year}]"

    # 整体 URL-encode，避免手写 f-string 引入空格/编码不一致问题
    encoded_sq = urllib.parse.quote(raw_search_query, safe="")
    url = f"{ARXIV_API_BASE}?start=0&max_results={effective_max}&sortBy=relevance&sortOrder=descending&search_query={encoded_sq}"

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "PaperReader/1.0 (mailto:support@paperreader.ai)",
                "Accept": "application/atom+xml",
            },
        )
        # 使用熔断保护
        breaker = _get_arxiv_direct_breaker()
        if not breaker.can_execute():
            logger.warning("[arXiv direct search] circuit OPEN, skipping query: %r", query)
            return []
        try:
            with urllib.request.urlopen(req, timeout=_ARXIV_TIMEOUT_DIRECT) as resp:
                xml_text = resp.read().decode("utf-8")
            breaker.record_success()
        except Exception as exc:
            breaker.record_failure()
            raise
    except Exception as exc:
        logger.warning("[arXiv direct search] failed for query %r: %s", query, exc)
        return []

    entries = re.split(r"<entry>", xml_text)
    entries = entries[1:]

    results: list[dict[str, Any]] = []
    for entry in entries:
        id_m = re.search(r"<id>https?://arxiv\.org/abs/(\d+\.\d+)", entry)
        if not id_m:
            continue
        aid = _strip_version(id_m.group(1))
        meta = _parse_arxiv_entry(entry, aid)

        # 噪声过滤
        if filter_noise and _is_noisy_paper(meta):
            logger.debug("[arXiv direct] filtered noisy paper: %s", meta.get("title", "")[:80])
            continue

        results.append(meta)

    logger.info(
        "[arXiv direct search] query=%r → %d raw, %d after noise filter (year≥%s)",
        query, len(results) + (effective_max - len(results)), len(results), effective_year,
    )
    # 返回时截断到 max_results
    return results[:max_results]
