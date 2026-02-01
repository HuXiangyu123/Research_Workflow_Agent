from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


_DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    "output",
    "data",
}

_DEFAULT_EXCLUDE_FILES = {
    ".env",
    ".env.example",
}


def _is_probably_binary(data: bytes) -> bool:
    if not data:
        return False
    if b"\x00" in data:
        return True
    return False


def _safe_resolve(root: Path, path_str: str) -> Path | None:
    try:
        p = (root / path_str).resolve()
    except Exception:
        return None
    try:
        p.relative_to(root.resolve())
    except Exception:
        return None
    return p


def _should_skip_path(p: Path) -> bool:
    name = p.name
    if name in _DEFAULT_EXCLUDE_FILES:
        return True
    parts = set(p.parts)
    if parts & _DEFAULT_EXCLUDE_DIRS:
        return True
    return False


@tool
def read_local_file(path: str, max_chars: int = 8000) -> dict[str, Any]:
    """读取本仓库内的文本文件片段（自动避免 .env / output / data 等敏感路径）。"""
    root = Path(os.getcwd()).resolve()
    p = _safe_resolve(root, path)
    if p is None or not p.exists() or not p.is_file() or _should_skip_path(p):
        return {"ok": False, "error": "path not allowed or not found"}

    try:
        data = p.read_bytes()
    except Exception as e:
        return {"ok": False, "error": f"read failed: {e}"}

    if _is_probably_binary(data):
        return {"ok": False, "error": "binary file not supported"}

    text = data.decode("utf-8", errors="replace")
    text = text[: max(100, min(int(max_chars), 20000))]
    rel = str(p.relative_to(root))
    return {"ok": True, "path": rel, "text": text}


@tool
def search_local_files(query: str, max_files: int = 5, max_chars_per_file: int = 1200) -> dict[str, Any]:
    """在本仓库内做轻量文本搜索，返回命中文件片段（避免 .env / output / data 等）。"""
    q = (query or "").strip()
    if not q:
        return {"ok": False, "error": "empty query"}

    root = Path(os.getcwd()).resolve()
    results: list[dict[str, Any]] = []

    max_files = max(1, min(int(max_files), 10))
    max_chars_per_file = max(200, min(int(max_chars_per_file), 4000))

    for p in root.rglob("*"):
        if len(results) >= max_files:
            break
        if not p.is_file() or _should_skip_path(p):
            continue
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".exe", ".dll"}:
            continue
        try:
            data = p.read_bytes()
        except Exception:
            continue
        if _is_probably_binary(data):
            continue
        text = data.decode("utf-8", errors="ignore")
        idx = text.lower().find(q.lower())
        if idx == -1:
            continue
        start = max(0, idx - max_chars_per_file // 2)
        snippet = text[start : start + max_chars_per_file]
        rel = str(p.relative_to(root))
        results.append({"path": rel, "snippet": snippet})

    return {"ok": True, "query": q, "results": results}

