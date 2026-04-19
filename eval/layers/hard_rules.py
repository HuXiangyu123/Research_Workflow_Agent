from __future__ import annotations

import re


SECTION_ALIAS_GROUPS = {
    "title": {"title", "标题"},
    "overview": {
        "abstract",
        "abstract and motivation",
        "core contributions",
        "paper information",
        "论文信息",
        "核心贡献",
    },
    "methods": {
        "methods",
        "methods review",
        "method overview",
        "方法概述",
    },
    "evidence": {
        "results",
        "experiments",
        "experiments and results",
        "evaluation",
        "datasets and benchmarks",
        "关键实验",
    },
    "discussion": {
        "discussion",
        "discussion and future directions",
        "limitations",
        "challenges",
        "challenges and limitations",
        "局限性",
    },
}


def _normalize_heading(heading: str) -> str:
    return re.sub(r"\s+", " ", heading.strip().lower())


def _extract_headings(report_md: str) -> set[str]:
    headings: set[str] = set()
    for line in report_md.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# "):
            headings.add("title")
        elif stripped.startswith("## "):
            headings.add(_normalize_heading(stripped[3:]))
    return headings


def check_structure(report_md: str, required_sections: set[str] | None = None) -> dict:
    """Check whether the markdown report satisfies the minimum section structure."""
    found = _extract_headings(report_md)

    if required_sections is not None:
        required_set = {_normalize_heading(section) for section in required_sections}
        missing = required_set - found
        return {"pass": len(missing) == 0, "missing": list(missing), "found": list(found)}

    missing = [
        name
        for name, aliases in SECTION_ALIAS_GROUPS.items()
        if not found.intersection({_normalize_heading(alias) for alias in aliases})
    ]
    return {"pass": len(missing) == 0, "missing": missing, "found": list(found)}


def check_citation_format(report_md: str, required: bool = True) -> dict:
    """Check citations section exists and has proper format."""
    if not required:
        return {"pass": True, "has_section": True, "citation_count": 0, "skipped": True}

    has_section = bool(
        re.search(r"^##\s+引用", report_md, re.MULTILINE)
        or re.search(r"^##\s+参考文献", report_md, re.MULTILINE)
        or re.search(r"^##\s+References", report_md, re.MULTILINE)
    )

    citation_lines = re.findall(r"^\s*-\s+\[", report_md, re.MULTILINE)
    return {
        "pass": has_section and len(citation_lines) > 0,
        "has_section": has_section,
        "citation_count": len(citation_lines),
    }


def check_must_include(report_md: str, keywords: list[str]) -> dict:
    """Check all required keywords appear in report (case-insensitive, supports Chinese/English)."""
    report_lower = report_md.lower()
    results = {}
    for kw in keywords:
        results[kw] = kw in report_md or kw.lower() in report_lower
    return {"pass": all(results.values()), "keywords": results}


def check_min_citations(report_md: str, min_count: int) -> dict:
    """Check minimum citation count."""
    url_pattern = r"https?://\S+"
    urls = re.findall(url_pattern, report_md)
    return {"pass": len(urls) >= min_count, "found": len(urls), "required": min_count}


def check_cost_guard(tokens_used: int, max_tokens: int = 50000) -> dict:
    """Check token usage is within budget."""
    return {"pass": tokens_used <= max_tokens, "tokens_used": tokens_used, "max": max_tokens}


def run_layer1(report_md: str, case: dict, tokens_used: int = 0) -> dict:
    """Run all Layer 1 checks. Returns {pass: bool, checks: {...}}."""
    checks: dict[str, dict] = {}

    if case.get("expect_error"):
        error_indicators = ["无法生成报告", "error", "failed", "失败"]
        report_lower = report_md.lower()
        is_error = any(
            indicator in report_md if indicator in {"无法生成报告", "失败"} else indicator in report_lower
            for indicator in error_indicators
        )
        checks["error_handled"] = {
            "pass": is_error,
            "indicators_found": [
                indicator
                for indicator in error_indicators
                if (
                    indicator in report_md
                    if indicator in {"无法生成报告", "失败"}
                    else indicator in report_lower
                )
            ],
        }
        return {"pass": is_error, "checks": checks}

    checks["structure"] = check_structure(report_md, case.get("sections"))

    min_citations = case.get("min_citations", 1)
    checks["citation_format"] = check_citation_format(report_md, required=(min_citations > 0))

    if "must_include" in case:
        checks["must_include"] = check_must_include(report_md, case["must_include"])

    if "min_citations" in case:
        checks["min_citations"] = check_min_citations(report_md, case["min_citations"])

    checks["cost_guard"] = check_cost_guard(tokens_used)

    all_pass = all(check["pass"] for check in checks.values())
    return {"pass": all_pass, "checks": checks}
