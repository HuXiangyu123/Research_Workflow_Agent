"""DeepXiv client wrapper built on top of ``deepxiv_sdk.Reader``.

Project notes:
- The Python SDK does not auto-persist a token for this server process.
- The CLI can auto-register an anonymous token, but backend code should read
  ``DEEPXIV_TOKEN`` explicitly from the environment.
- Official auth supports either ``Authorization: Bearer`` or ``?token=...``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── DeepXiv Reader 初始化 ────────────────────────────────────────────────────

DEFAULT_DEEPXIV_BASE_URL = "https://data.rag.ac.cn"
DEFAULT_TIMEOUT_S = 60
DEFAULT_MAX_RETRIES = 3

_reader = None
_reader_init_ok = False
_reader_init_error: str | None = None
_reader_config_snapshot: tuple[str | None, str, int, int] | None = None


def _current_reader_config() -> tuple[str | None, str, int, int]:
    token: str | None = None
    base_url = DEFAULT_DEEPXIV_BASE_URL

    try:
        from src.agent.settings import get_settings

        settings = get_settings()
        token = settings.deepxiv_token.strip() or None
        base_url = settings.deepxiv_base_url.strip() or DEFAULT_DEEPXIV_BASE_URL
    except Exception:
        # Fallback for isolated scripts that import this module outside the app.
        import os

        token = (os.getenv("DEEPXIV_TOKEN", "") or "").strip() or None
        base_url = (os.getenv("DEEPXIV_BASE_URL", "") or "").strip() or DEFAULT_DEEPXIV_BASE_URL

    return token, base_url, DEFAULT_TIMEOUT_S, DEFAULT_MAX_RETRIES


def _normalize_authors(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _normalize_published_date(item: dict[str, Any]) -> str:
    value = (
        item.get("publish_at")
        or item.get("published")
        or item.get("published_date")
        or item.get("modified_at")
        or ""
    )
    return str(value)[:10]


def _normalize_trending_days(days: int) -> int:
    if days <= 7:
        return 7
    if days <= 14:
        return 14
    return 30


def _init_reader() -> Any:
    """延迟初始化 DeepXiv Reader，并在配置变更时重建实例。"""
    global _reader, _reader_init_ok, _reader_init_error, _reader_config_snapshot

    token, base_url, timeout_s, max_retries = _current_reader_config()
    snapshot = (token, base_url, timeout_s, max_retries)

    if _reader is not None and _reader_config_snapshot == snapshot:
        return _reader

    try:
        from deepxiv_sdk import Reader

        _reader = Reader(
            token=token,
            base_url=base_url,
            timeout=timeout_s,
            max_retries=max_retries,
        )
        _reader_init_ok = True
        _reader_init_error = None
        _reader_config_snapshot = snapshot
        logger.info(
            "[DeepXiv] Reader initialized successfully (base_url=%s, token=%s)",
            base_url,
            "set" if token else "unset",
        )
        return _reader
    except Exception as e:
        _reader_init_error = str(e)
        _reader = None
        _reader_config_snapshot = None
        logger.warning("[DeepXiv] Reader init failed: %s (will use fallback)", e)
        return None


def is_available() -> bool:
    """检查 DeepXiv 是否可用。"""
    if _reader is None:
        _init_reader()
    token, _, _, _ = _current_reader_config()
    return _reader_init_ok and bool(token)


# ── Search API ─────────────────────────────────────────────────────────────

def search_papers(
    query: str,
    *,
    size: int = 10,
    date_from: str | None = None,
    categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    DeepXiv 关键词搜索 arXiv 论文。

    参数：
        query：搜索关键词
        size：返回数量（默认 10，最大 100）
        date_from：起始日期，如 "2024-01-01"
        categories：限定 cs.AI / cs.CL / cs.LG 等

    返回：
        [{arxiv_id, title, abstract, authors, published_date, categories}, ...]
    """
    reader = _init_reader()
    if reader is None:
        return []

    try:
        response = reader.search(
            query,
            size=min(size, 100),
            categories=categories,
            date_from=date_from,
        )
        if isinstance(response, dict):
            results = response.get("results") or response.get("papers") or response.get("items") or []
        else:
            results = response or []

        papers = []
        for item in (results or []):
            arxiv_id = str(item.get("arxiv_id", "") or item.get("id", ""))
            if arxiv_id:
                arxiv_id = arxiv_id.strip()
            papers.append({
                "arxiv_id": arxiv_id,
                "title": str(item.get("title", "") or "").strip(),
                "abstract": str(item.get("abstract", "") or "").strip(),
                "authors": _normalize_authors(item.get("authors") or item.get("author_names")),
                "published_date": _normalize_published_date(item),
                "categories": item.get("categories", []),
                "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                "pdf_url": str(item.get("src_url") or f"https://arxiv.org/pdf/{arxiv_id}.pdf") if arxiv_id else "",
                "citation_count": int(item.get("citation") or item.get("citations") or 0),
                "score": item.get("score"),
                "_source": "deepxiv",
            })
        logger.info("[DeepXiv] search(%r) → %d papers", query, len(papers))
        return papers
    except Exception as e:
        logger.warning("[DeepXiv] search failed for %r: %s", query, e)
        return []


# ── Paper Brief API ─────────────────────────────────────────────────────────

def get_paper_brief(arxiv_id: str) -> dict[str, Any] | None:
    """
    获取单篇论文的 brief 信息（TLDR + keywords + GitHub URL）。

    DeepXiv brief 包含：
    - title, tldr, keywords, citations, github_url

    比 raw abstract 更结构化，是 progressive reading 第一步。
    """
    reader = _init_reader()
    if reader is None:
        return None

    try:
        brief = reader.brief(arxiv_id)
        if not brief:
            return None

        result = {
            "arxiv_id": arxiv_id,
            "title": brief.get("title") or "",
            "tldr": brief.get("tldr") or brief.get("abstract") or "",
            "keywords": brief.get("keywords", []),
            "github_url": brief.get("github_url") or "",
            "num_citations": brief.get("citations") or brief.get("num_citations") or 0,
            "num_references": brief.get("num_references") or 0,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": brief.get("src_url") or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "published_date": str(brief.get("publish_at") or "")[:10],
        }
        logger.debug("[DeepXiv] brief(%s) → %s", arxiv_id, result.get("title", "")[:50])
        return result
    except Exception as e:
        logger.warning("[DeepXiv] brief failed for %s: %s", arxiv_id, e)
        return None


def get_paper_head(arxiv_id: str) -> dict[str, Any] | None:
    """
    获取论文结构和 token 分布（Progressive reading 第二步）。

    DeepXiv head 包含：
    - title, authors, sections（list[dict]）, token_count
    """
    reader = _init_reader()
    if reader is None:
        return None

    try:
        head = reader.head(arxiv_id)
        if not head:
            return None

        return {
            "arxiv_id": arxiv_id,
            "title": head.get("title") or "",
            "abstract": head.get("abstract") or "",
            "authors": _normalize_authors(head.get("authors")),
            "sections": head.get("sections", []),
            "total_tokens": head.get("token_count") or head.get("total_tokens") or 0,
            "token_count": head.get("token_count") or head.get("total_tokens") or 0,
            "categories": head.get("categories", []),
            "published_date": str(head.get("publish_at") or "")[:10],
            "pdf_url": head.get("src_url") or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        }
    except Exception as e:
        logger.warning("[DeepXiv] head failed for %s: %s", arxiv_id, e)
        return None


def get_paper_section(arxiv_id: str, section: str) -> str | None:
    """
    读取论文特定章节（Progressive reading 第三步）。

    section 参数如 "Introduction", "Method", "Experiments", "Results", "Conclusion"
    """
    reader = _init_reader()
    if reader is None:
        return None

    try:
        text = reader.section(arxiv_id, section)
        return text if text else None
    except Exception as e:
        logger.warning("[DeepXiv] section(%s, %s) failed: %s", arxiv_id, section, e)
        return None


# ── Trending / Popularity API ─────────────────────────────────────────────

def get_trending_papers(
    days: int = 7,
    *,
    size: int = 30,
    categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    获取近期热门论文（按社交信号热度排序）。

    DeepXiv 策略：不依赖关键词匹配，按 trending 热度发现论文。
    用于 discovery 阶段，发现用户可能不知道的相关热文。
    """
    reader = _init_reader()
    if reader is None:
        return []

    try:
        normalized_days = _normalize_trending_days(days)
        response = reader.trending(days=normalized_days, limit=min(size, 100))
        if isinstance(response, dict):
            results = response.get("results") or response.get("papers") or response.get("items") or []
        else:
            results = response or []
        papers = []
        for item in (results or []):
            arxiv_id = str(item.get("arxiv_id", "") or item.get("id", "")).strip()
            papers.append({
                "arxiv_id": arxiv_id,
                "title": str(item.get("title", "") or "").strip(),
                "abstract": str(item.get("abstract", "") or "").strip(),
                "authors": _normalize_authors(item.get("authors") or item.get("author_names")),
                "published_date": _normalize_published_date(item),
                "categories": item.get("categories", []),
                "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                "pdf_url": str(item.get("src_url") or f"https://arxiv.org/pdf/{arxiv_id}.pdf") if arxiv_id else "",
                "citation_count": int(item.get("citation") or item.get("citations") or 0),
                "_source": "deepxiv_trending",
            })
        logger.info(
            "[DeepXiv] trending(days=%d → normalized=%d) → %d papers",
            days,
            normalized_days,
            len(papers),
        )
        return papers
    except Exception as e:
        logger.warning("[DeepXiv] trending failed: %s", e)
        return []


def get_paper_popularity(arxiv_id: str) -> dict[str, Any] | None:
    """获取论文的社交传播指标（views, tweets, likes, replies）。"""
    reader = _init_reader()
    if reader is None:
        return None

    try:
        if hasattr(reader, "social_impact"):
            pop = reader.social_impact(arxiv_id)
        elif hasattr(reader, "popularity"):
            pop = reader.popularity(arxiv_id)
        else:
            raise AttributeError("DeepXiv Reader has neither social_impact nor popularity")
        return pop if pop else None
    except Exception as e:
        logger.warning("[DeepXiv] popularity(%s) failed: %s", arxiv_id, e)
        return None


# ── Semantic Scholar Metadata ──────────────────────────────────────────────

def get_semantic_scholar(sc_id: str) -> dict[str, Any] | None:
    """通过 Semantic Scholar ID 获取丰富元信息（citations, references, fieldsOfStudy）。"""
    reader = _init_reader()
    if reader is None:
        return None

    try:
        meta = reader.semantic_scholar(sc_id)
        return meta if meta else None
    except Exception as e:
        logger.warning("[DeepXiv] semantic_scholar(%s) failed: %s", sc_id, e)
        return None


# ── Batch API ──────────────────────────────────────────────────────────────

def batch_get_briefs(
    arxiv_ids: list[str],
    *,
    max_workers: int = 4,
    delay_per_request: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """
    批量获取 paper briefs（并行，带速率限制）。

    避免对 DeepXiv API 造成过大压力，每次请求间隔 delay_per_request 秒。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, dict[str, Any]] = {}

    def _fetch_one(aid: str) -> tuple[str, dict[str, Any] | None]:
        time.sleep(delay_per_request)
        return aid, get_paper_brief(aid)

    limited_ids = arxiv_ids[:50]  # 最多 50 个，避免 API 配额耗尽
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, aid): aid for aid in limited_ids}
        for future in as_completed(futures):
            try:
                aid, brief = future.result()
                if brief:
                    results[aid] = brief
            except Exception:
                pass

    logger.info("[DeepXiv] batch_briefs(%d) → %d fetched", len(arxiv_ids), len(results))
    return results
