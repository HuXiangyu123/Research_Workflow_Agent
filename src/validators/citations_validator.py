from __future__ import annotations


def has_citations_section(text: str) -> bool:
    lowered = text.lower()
    return ("引用" in text) or ("references" in lowered) or ("citations" in lowered)

