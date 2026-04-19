from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from src.models.report import DraftReport, Claim, Citation

DRAFT_SYSTEM_PROMPT = """You are a literature report generator. Given paper metadata and evidence, produce a structured JSON report.

Output format (JSON only, no markdown):
{
  "sections": {
    "title": "Specific English report title",
    "paper_information": "...",
    "core_contributions": "...",
    "methods": "...",
    "experiments_and_results": "...",
    "limitations": "...",
    "related_work": "..."
  },
  "claims": [
    {"id": "c1", "text": "claim text", "citation_labels": ["[1]"]}
  ],
  "citations": [
    {"label": "[1]", "url": "https://...", "reason": "..."}
  ]
}
"""


def draft_report(state: dict) -> dict:
    doc = state.get("normalized_doc")
    evidence = state.get("evidence")

    if not doc:
        return {"errors": ["draft_report: no normalized_doc"]}

    meta = doc.metadata
    evidence_text = ""
    if evidence:
        for r in evidence.rag_results[:5]:
            evidence_text += f"[RAG] {r.text[:500]}\n\n"
        for w in evidence.web_results[:3]:
            evidence_text += f"[WEB {w.url}] {w.text[:500]}\n\n"

    user_prompt = (
        f"Paper: {meta.title}\nAuthors: {', '.join(meta.authors)}\n"
        f"Abstract: {meta.abstract}\n\n"
        f"Document text (first 5000 chars):\n{doc.document_text[:5000]}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Generate the structured JSON report."
    )

    try:
        from src.agent.settings import Settings
        from src.agent.llm import build_reason_llm

        settings = Settings.from_env()
        llm = build_reason_llm(settings)

        messages = [
            SystemMessage(content=DRAFT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        from src.agent.report_frame import extract_llm_text, parse_json_object

        response = llm.invoke(messages)
        reasoning = (getattr(response, "additional_kwargs", None) or {}).get("reasoning_content", "")
        text = extract_llm_text(response)
        data = parse_json_object(text)
        claims = [
            Claim(
                id=c["id"],
                text=c["text"],
                citation_labels=c.get("citation_labels", []),
            )
            for c in data.get("claims", [])
        ]
        citations = [
            Citation(label=c["label"], url=c["url"], reason=c.get("reason", ""))
            for c in data.get("citations", [])
        ]
        report = DraftReport(
            sections=data.get("sections", {}),
            claims=claims,
            citations=citations,
        )
        result: dict = {"draft_report": report}
        if reasoning:
            result["_reasoning_content"] = reasoning
        return result

    except json.JSONDecodeError:
        fallback = DraftReport(
            sections={"report": text if "text" in dir() else "Generation failed"},
            claims=[],
            citations=[],
        )
        return {
            "draft_report": fallback,
            "warnings": ["draft_report: LLM output was not valid JSON, used fallback"],
        }
    except Exception as e:
        return {"errors": [f"draft_report: {e}"]}
