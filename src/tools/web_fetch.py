from __future__ import annotations

from typing import Any

import httpx
from langchain_core.tools import tool


@tool
def fetch_webpage_text(url: str, max_chars: int = 8000) -> dict[str, Any]:
    """抓取网页文本（简单清洗），用于补充细节与引用回链。"""
    max_chars = max(500, min(int(max_chars), 20000))
    headers = {"User-Agent": "agent-build/0.1 (+https://example.local)"}
    with httpx.Client(follow_redirects=True, timeout=20, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()

    text = resp.text
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[:max_chars] + " ..."
    return {"url": url, "text": text}

