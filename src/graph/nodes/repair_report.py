from __future__ import annotations

SECTION_REQUIREMENTS = {
    "title": {"title", "Title", "标题"},
    "overview": {
        "paper_information",
        "Paper Information",
        "core_contributions",
        "Core Contributions",
        "abstract",
        "Abstract",
        "abstract_and_motivation",
        "Abstract And Motivation",
        "核心贡献",
        "论文信息",
    },
    "methods": {"methods", "Methods", "methods_review", "Methods Review", "方法概述"},
    "evidence": {
        "experiments",
        "Experiments",
        "experiments_and_results",
        "Experiments And Results",
        "evaluation",
        "Evaluation",
        "results",
        "Results",
        "关键实验",
    },
    "limitations": {
        "limitations",
        "Limitations",
        "discussion",
        "Discussion",
        "discussion_and_future_directions",
        "Discussion And Future Directions",
        "challenges",
        "Challenges",
        "challenges_and_limitations",
        "Challenges And Limitations",
        "局限性",
    },
}


def _missing_requirements(existing: set[str]) -> set[str]:
    missing: set[str] = set()
    for requirement, aliases in SECTION_REQUIREMENTS.items():
        if not existing.intersection(aliases):
            missing.add(requirement)
    return missing


def repair_report(state: dict) -> dict:
    report = state.get("draft_report")
    if not report:
        return {"warnings": ["repair_report: no draft_report, skipping"]}

    existing = set(report.sections.keys())
    missing = _missing_requirements(existing)

    if not missing and len(report.citations) > 0:
        return {}

    try:
        import json

        from langchain_core.messages import HumanMessage, SystemMessage

        from src.agent.settings import Settings
        from src.agent.llm import build_reason_llm

        settings = Settings.from_env()
        llm = build_reason_llm(settings)

        current_sections = json.dumps(report.sections, ensure_ascii=False, indent=2)
        repair_prompt = (
            f"The following report is missing these sections: {', '.join(missing) if missing else 'none'}.\n"
            f"It has {len(report.citations)} citations"
            f"{' (needs at least 1)' if not report.citations else ''}.\n\n"
            f"Current sections:\n{current_sections}\n\n"
            "Add the missing sections and/or citations. "
            "Output ONLY a JSON object with keys 'sections' and 'citations'."
        )

        from src.agent.report_frame import extract_llm_text, parse_json_object

        resp = llm.invoke([
            SystemMessage(content="You repair incomplete literature reports. Output valid JSON only."),
            HumanMessage(content=repair_prompt),
        ])
        text = extract_llm_text(resp)
        data = parse_json_object(text)
        if "sections" in data:
            merged = {**report.sections, **data["sections"]}
            report = report.model_copy(update={"sections": merged})
        if "citations" in data and not report.citations:
            from src.models.report import Citation

            new_cites = [
                Citation(label=c["label"], url=c["url"], reason=c.get("reason", ""))
                for c in data["citations"]
            ]
            report = report.model_copy(update={"citations": new_cites})

        return {"draft_report": report, "warnings": ["repair_report: repair pass triggered"]}

    except Exception as e:
        return {"warnings": [f"repair_report: repair failed ({e}), continuing with incomplete report"]}
