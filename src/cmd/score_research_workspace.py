from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.research.graph.nodes.search import _extract_anchor_groups


PLACEHOLDER_PATTERNS = [
    r"\[Research Topic\]",
    r"\[Dataset [^\]]+\]",
    r"\[Example [^\]]+\]",
    r"\[Adjacent Field [^\]]+\]",
    r"\bTBD\b",
    r"\bTODO\b",
]


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _duplicate_paragraph_ratio(markdown: str) -> float:
    paragraphs = [
        re.sub(r"\s+", " ", p).strip()
        for p in re.split(r"\n\s*\n", markdown)
        if len(re.sub(r"\s+", " ", p).strip()) >= 80
    ]
    if not paragraphs:
        return 0.0
    counts = Counter(paragraphs)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    return duplicates / len(paragraphs)


def _candidate_year(candidate: dict[str, Any]) -> int | None:
    for key in ("published_year", "year"):
        value = candidate.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.search(r"20\d{2}", value)
            if match:
                return int(match.group(0))
    published_date = str(candidate.get("published_date") or "")
    match = re.search(r"20\d{2}", published_date)
    if match:
        return int(match.group(0))
    return None


def _year_bounds(brief: dict[str, Any], search_plan: dict[str, Any]) -> tuple[int | None, int | None]:
    text = " ".join(
        part
        for part in (
            str(brief.get("time_range") or "").strip(),
            str(search_plan.get("plan_goal") or "").strip(),
        )
        if part
    )
    years = [int(item) for item in re.findall(r"20\d{2}", text)]
    if not years:
        return (None, None)
    if len(years) == 1:
        return (years[0], None)
    return (min(years), max(years))


def _year_in_range(year: int | None, bounds: tuple[int | None, int | None]) -> bool:
    start, end = bounds
    if year is None:
        return True
    if start is not None and year < start:
        return False
    if end is not None and year > end:
        return False
    return True


def _supported_ratio(review_feedback: dict[str, Any] | None, claim_verification: dict[str, Any] | None) -> float:
    candidate_stats = [
        (claim_verification or {}).get("grounding_stats", {}),
        (review_feedback or {}).get("grounding_stats", {}),
        ((review_feedback or {}).get("claim_verification") or {}).get("grounding_stats", {}),
    ]
    for stats in candidate_stats:
        if isinstance(stats, dict) and stats.get("supported_ratio") is not None:
            return float(stats["supported_ratio"])
        if isinstance(stats, dict) and stats.get("total"):
            total = float(stats.get("total") or 0.0)
            grounded = float(stats.get("grounded", 0.0) or 0.0)
            partial = float(stats.get("partial", 0.0) or 0.0)
            if total > 0:
                return (grounded + partial) / total

    feedback = review_feedback or {}
    text = json.dumps(feedback, ensure_ascii=False)
    match = re.search(r"(\d+)\s*/\s*(\d+)\s+claims are currently ungrounded", text)
    if match:
        ungrounded = int(match.group(1))
        total = int(match.group(2))
        if total > 0:
            return max(0.0, 1.0 - (ungrounded / total))
    return 0.0


def _grounding_ratios(review_feedback: dict[str, Any] | None, claim_verification: dict[str, Any] | None) -> tuple[float, float]:
    candidate_stats = [
        (claim_verification or {}).get("grounding_stats", {}),
        (review_feedback or {}).get("grounding_stats", {}),
        ((review_feedback or {}).get("claim_verification") or {}).get("grounding_stats", {}),
    ]
    for stats in candidate_stats:
        if not isinstance(stats, dict):
            continue
        total = float(stats.get("total") or 0.0)
        grounded = float(stats.get("grounded", 0.0) or 0.0)
        partial = float(stats.get("partial", 0.0) or 0.0)
        grounded_ratio = float(stats.get("grounded_ratio", grounded / total if total else 0.0) or 0.0)
        supported_ratio = float(
            stats.get("supported_ratio", (grounded + partial) / total if total else 0.0) or 0.0
        )
        if total > 0 or grounded_ratio > 0 or supported_ratio > 0:
            return grounded_ratio, supported_ratio
    supported_ratio = _supported_ratio(review_feedback, claim_verification)
    return 0.0, supported_ratio


def _has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _off_topic(candidate: dict[str, Any], *, strict_core: bool) -> bool:
    diagnostics = candidate.get("relevance_diagnostics", {}) if isinstance(candidate, dict) else {}
    penalties = set(diagnostics.get("penalties", [])) if isinstance(diagnostics, dict) else set()
    matched = set(diagnostics.get("matched_groups", [])) if isinstance(diagnostics, dict) else set()
    if penalties.intersection(
        {
            "off_topic_core_intent",
            "governance_without_clinical_scope",
            "component_method_without_agent_scope",
        }
    ):
        return True
    if strict_core and len(matched.intersection({"agent", "medical", "multimodal_or_imaging", "diagnosis_or_triage"})) < 3:
        return True
    return not matched


def score_task_dir(task_dir: Path) -> dict[str, Any]:
    brief = _load_json(task_dir / "brief.json") or {}
    search_plan = _load_json(task_dir / "search_plan.json") or {}
    rag_result = _load_json(task_dir / "rag_result.json") or {}
    review_feedback = _load_json(task_dir / "review_feedback.json") or {}
    claim_verification = _load_json(task_dir / "claim_verification.json") or {}
    report = (task_dir / "report.md").read_text(encoding="utf-8") if (task_dir / "report.md").is_file() else ""

    candidates = rag_result.get("paper_candidates", []) if isinstance(rag_result, dict) else []
    anchors = _extract_anchor_groups(brief, search_plan)
    strict_core = bool(anchors.get("strict_core"))
    bounds = _year_bounds(brief, search_plan)

    paper_count = len(candidates)
    off_topic_titles = [cand.get("title", "") for cand in candidates if isinstance(cand, dict) and _off_topic(cand, strict_core=strict_core)]
    year_in_scope = [
        cand for cand in candidates
        if isinstance(cand, dict) and _year_in_range(_candidate_year(cand), bounds)
    ]
    fulltext_count = sum(1 for cand in candidates if isinstance(cand, dict) and cand.get("fulltext_available"))
    headings = [line for line in report.splitlines() if line.startswith("#")]
    section_headings = [line[3:].strip().lower() for line in report.splitlines() if line.startswith("## ")]
    synthesis_markers = [
        marker for marker in ("agreement", "disagreement", "trade-off", "evidence gap")
        if marker in report.lower()
    ]
    placeholder_hits = sum(
        len(re.findall(pattern, report, flags=re.IGNORECASE))
        for pattern in PLACEHOLDER_PATTERNS
    )
    grounded_ratio, supported_ratio = _grounding_ratios(review_feedback, claim_verification)
    duplicate_ratio = _duplicate_paragraph_ratio(report)
    required_sections = {
        "abstract",
        "introduction",
        "background",
        "taxonomy",
        "methods",
        "datasets",
        "evaluation",
        "discussion",
        "future work",
        "conclusion",
    }
    normalized_sections = {heading.replace("&", "and") for heading in section_headings}
    structure_score = len(required_sections.intersection(normalized_sections)) / len(required_sections)

    relevance_purity = 1.0 - (len(off_topic_titles) / paper_count) if paper_count else 0.0
    year_scope_ratio = len(year_in_scope) / paper_count if paper_count else 0.0
    fulltext_ratio = fulltext_count / paper_count if paper_count else 0.0
    rag_depth = min(paper_count / 12.0, 1.0)
    rag_score = round(
        100
        * (
            0.35 * relevance_purity
            + 0.2 * year_scope_ratio
            + 0.2 * fulltext_ratio
            + 0.25 * rag_depth
        ),
        1,
    )

    report_length_score = min(len(report) / 12000.0, 1.0)
    synthesis_score = min(len(synthesis_markers) / 3.0, 1.0)
    language_score = 0.0 if _has_chinese(report) else 1.0
    duplication_score = max(0.0, 1.0 - min(duplicate_ratio / 0.12, 1.0))
    evidence_score = min(1.0, grounded_ratio + max(0.0, supported_ratio - grounded_ratio) * 0.25)
    report_score = (
        100
        * (
            0.2 * structure_score
            + 0.15 * report_length_score
            + 0.35 * evidence_score
            + 0.15 * synthesis_score
            + 0.1 * language_score
            + 0.05 * duplication_score
        )
    )
    if review_feedback.get("passed") is False:
        report_score *= 0.85
    report_score = round(report_score, 1)

    title_line = next((line[2:].strip() for line in report.splitlines() if line.startswith("# ")), "")
    return {
        "task_dir": str(task_dir),
        "paper_count": paper_count,
        "fulltext_count": fulltext_count,
        "fulltext_ratio": round(fulltext_ratio, 3),
        "year_bounds": bounds,
        "year_scope_ratio": round(year_scope_ratio, 3),
        "off_topic_count": len(off_topic_titles),
        "off_topic_ratio": round((len(off_topic_titles) / paper_count), 3) if paper_count else 0.0,
        "off_topic_titles": off_topic_titles[:8],
        "rag_score": rag_score,
        "report_score": report_score,
        "grounded_ratio": round(grounded_ratio, 3),
        "supported_ratio": round(supported_ratio, 3),
        "review_passed": review_feedback.get("passed"),
        "report_chars": len(report),
        "heading_count": len(headings),
        "structure_score": round(structure_score, 3),
        "placeholder_hits": placeholder_hits,
        "duplicate_paragraph_ratio": round(duplicate_ratio, 4),
        "synthesis_markers": synthesis_markers,
        "title": title_line,
        "has_chinese": _has_chinese(report),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a research workspace task output.")
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", default="", help="Optional JSON output path")
    args = parser.parse_args()

    task_dir = (
        Path("output")
        / "workspaces"
        / args.workspace_id
        / "tasks"
        / args.task_id
    )
    result = score_task_dir(task_dir)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
