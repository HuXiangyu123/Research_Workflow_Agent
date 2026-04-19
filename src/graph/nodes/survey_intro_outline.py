from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.report_frame import SURVEY_INTRO_OUTLINE_SYSTEM_PROMPT, extract_llm_text, parse_json_object
from src.models.report import Citation, DraftReport, ReportFrame


def survey_intro_outline(state: dict) -> dict:
    doc = state.get("normalized_doc")
    if not doc:
        return {"errors": ["survey_intro_outline: no normalized_doc"]}

    meta = doc.metadata
    user_prompt = (
        f"Title: {meta.title}\n"
        f"Authors: {', '.join(meta.authors)}\n"
        f"Published: {meta.published}\n"
        f"Abstract: {meta.abstract}\n\n"
        f"Full document text:\n{doc.document_text}\n\n"
        "Generate an English introduction summary and survey outline as JSON."
    )

    try:
        from src.agent.settings import Settings
        from src.agent.llm import build_chat_llm

        settings = Settings.from_env()
        llm = build_chat_llm(settings, max_tokens=16384)
        resp = llm.invoke([
            SystemMessage(content=SURVEY_INTRO_OUTLINE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        reasoning = (getattr(resp, "additional_kwargs", None) or {}).get("reasoning_content", "")
        text = extract_llm_text(resp)
        data = parse_json_object(text)

        sections = {k: v for k, v in data.get("sections", {}).items() if isinstance(v, str)}
        followup_hints = [x for x in data.get("followup_hints", []) if isinstance(x, str)]
        citations = [
            Citation(label=c["label"], url=c["url"], reason=c.get("reason", ""))
            for c in data.get("citations", [])
        ]

        frame = ReportFrame(
            title=meta.title,
            paper_type="survey",
            mode="full",
            sections=sections,
            outline=data.get("outline"),
            claims=[],
            citations=citations,
        )
        draft = DraftReport(sections=sections, claims=[], citations=citations)
        result: dict = {
            "paper_type": "survey",
            "report_frame": frame,
            "draft_report": draft,
            "followup_hints": followup_hints,
        }
        if reasoning:
            result["_reasoning_content"] = reasoning
        return result
    except Exception as e:
        return {"errors": [f"survey_intro_outline: {e}"]}
