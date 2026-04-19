from __future__ import annotations

import html
import re

from src.models.report import Citation, DraftReport, ResolvedReport
from src.verification.source_tiers import classify_url
from src.verification.reachability import check_url_reachable_sync


def _fetch_content_snippet(url: str, max_chars: int = 2000) -> str | None:
    """Fetch text content from a URL for evidence verification."""
    try:
        import httpx

        with httpx.Client(timeout=10, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            text = resp.text[:max_chars]
            return text if text.strip() else None
    except Exception:
        return None


def _looks_like_html(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("<!doctype html", "<html", "<head", "<body", "<meta "))


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_html(text: str) -> str:
    no_scripts = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    no_tags = re.sub(r"(?is)<[^>]+>", " ", no_scripts)
    return _clean_text(html.unescape(no_tags))


def _extract_arxiv_abstract(html_text: str) -> str | None:
    match = re.search(
        r'(?is)<blockquote[^>]*class="[^"]*abstract[^"]*"[^>]*>(.*?)</blockquote>',
        html_text,
    )
    if not match:
        match = re.search(
            r"(?is)<blockquote[^>]*class='[^']*abstract[^']*'[^>]*>(.*?)</blockquote>",
            html_text,
        )
    if not match:
        return None

    abstract = _strip_html(match.group(1))
    abstract = re.sub(r"(?i)^abstract:\s*", "", abstract).strip()
    return abstract if len(abstract) >= 80 else None


def _sanitize_fetched_content(content: str | None, url: str) -> str | None:
    text = _clean_text(content or "")
    if not text:
        return None

    if not _looks_like_html(text):
        return text

    if "arxiv.org/abs/" in (url or ""):
        abstract = _extract_arxiv_abstract(content or "")
        if abstract:
            return abstract

    stripped = _strip_html(content or "")
    if not stripped:
        return None
    if stripped.lower().startswith("abstract page for arxiv paper") and len(stripped) < 240:
        return None
    return stripped if len(stripped) >= 80 else None


def resolve_citations(state: dict) -> dict:
    """Resolve each citation: classify tier, check reachability, fetch content."""
    draft: DraftReport | None = state.get("draft_report")
    if not draft:
        return {"warnings": ["resolve_citations: no draft_report, skipping"]}

    resolved_citations: list[Citation] = []
    warnings: list[str] = []

    for cit in draft.citations:
        tier = classify_url(cit.url)
        reachable = check_url_reachable_sync(cit.url)

        existing_content = _sanitize_fetched_content(cit.fetched_content, cit.url)
        fetched = None
        if reachable and not existing_content:
            fetched = _fetch_content_snippet(cit.url)
        resolved_content = existing_content or _sanitize_fetched_content(fetched, cit.url)

        resolved_cit = cit.model_copy(
            update={
                "source_tier": tier,
                "reachable": reachable,
                "fetched_content": resolved_content,
            }
        )
        resolved_citations.append(resolved_cit)

        if not reachable:
            warnings.append(
                f"resolve_citations: unreachable URL {cit.label} ({cit.url})"
            )

    resolved = ResolvedReport(
        sections=dict(draft.sections),
        claims=list(draft.claims),
        citations=resolved_citations,
    )

    result: dict = {"resolved_report": resolved}
    if warnings:
        result["warnings"] = warnings
    return result
