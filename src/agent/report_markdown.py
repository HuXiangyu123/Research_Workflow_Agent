from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.models.report import Citation, GroundingStats


DEFAULT_DISPLAY_NAMES: dict[str, str] = {
    "paper_information": "Paper Information",
    "abstract_and_motivation": "Abstract and Motivation",
    "background_and_related_work": "Background and Related Work",
    "methods_review": "Methods Review",
    "datasets_and_benchmarks": "Datasets and Benchmarks",
    "discussion_and_future_directions": "Discussion and Future Directions",
    "challenges_and_limitations": "Challenges and Limitations",
    "conclusion_and_outlook": "Conclusion and Outlook",
    "survey_outline": "Survey Outline",
    "suggested_followups": "Suggested Follow-Ups",
    "introduction_summary": "Introduction Summary",
}


def _display_name(section_key: str, display_names: Mapping[str, str] | None) -> str:
    if display_names and section_key in display_names:
        return display_names[section_key]
    if section_key in DEFAULT_DISPLAY_NAMES:
        return DEFAULT_DISPLAY_NAMES[section_key]
    if any(ch in section_key for ch in (" ", ".", "-")) and section_key != section_key.lower():
        return section_key
    return section_key.replace("_", " ").title()


def render_report_markdown(
    *,
    sections: Mapping[str, str],
    citations: Sequence[Citation] | None = None,
    grounding_stats: GroundingStats | None = None,
    report_confidence: str | None = None,
    title: str | None = None,
    section_order: Sequence[str] | None = None,
    display_names: Mapping[str, str] | None = None,
) -> str:
    lines: list[str] = []
    resolved_title = (title or sections.get("title") or "").strip()
    if resolved_title:
        lines.extend([f"# {resolved_title}", ""])

    seen: set[str] = set()
    ordered_keys = [key for key in (section_order or sections.keys()) if key in sections]
    trailing_keys = [key for key in sections.keys() if key not in ordered_keys]

    for section_key in [*ordered_keys, *trailing_keys]:
        if section_key in seen or section_key == "title":
            continue
        seen.add(section_key)
        content = str(sections.get(section_key, "") or "").strip()
        if not content:
            continue
        lines.extend([f"## {_display_name(section_key, display_names)}", "", content, ""])

    if citations:
        lines.extend(["## References", ""])
        for citation in citations:
            reason = f" - {citation.reason}" if citation.reason else ""
            lines.append(f"- {citation.label} {citation.url}{reason}")
        lines.append("")

    if grounding_stats:
        lines.extend(["## Grounding Summary", ""])
        if grounding_stats.total_claims > 0:
            lines.append(
                f"- Grounded: {grounding_stats.grounded}/{grounding_stats.total_claims}"
            )
            lines.append(
                f"- Partial: {grounding_stats.partial}/{grounding_stats.total_claims}"
            )
            lines.append(
                f"- Ungrounded: {grounding_stats.ungrounded}/{grounding_stats.total_claims}"
            )
            lines.append(
                f"- Abstained: {grounding_stats.abstained}/{grounding_stats.total_claims}"
            )
        lines.append(f"- Tier A source ratio: {round(grounding_stats.tier_a_ratio * 100)}%")
        lines.append(f"- Report confidence: {report_confidence or 'unknown'}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
