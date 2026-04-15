from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from src.agent.prompts import LITERATURE_REPORT_SYSTEM_PROMPT


def _last_ai_text(state: dict[str, Any]) -> str:
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            return msg.content
    if messages:
        last = messages[-1]
        content = getattr(last, "content", "")
        return content if isinstance(content, str) else ""
    return ""


def _final_report_to_markdown(final_report) -> str:
    """Convert a FinalReport to markdown string."""
    lines: list[str] = []
    for section_name, content in final_report.sections.items():
        lines.append(f"## {section_name}\n\n{content}\n")

    if final_report.citations:
        lines.append("## 引用\n")
        for c in final_report.citations:
            lines.append(f"- {c.label} {c.url} — {c.reason}")

    if final_report.grounding_stats:
        gs = final_report.grounding_stats
        lines.append("\n## 引用可信度\n")
        if gs.total_claims > 0:
            lines.append(f"- Grounded: {gs.grounded}/{gs.total_claims}")
            lines.append(f"- Partial: {gs.partial}/{gs.total_claims}")
            lines.append(f"- Ungrounded: {gs.ungrounded}/{gs.total_claims}")
            lines.append(f"- Abstained: {gs.abstained}/{gs.total_claims}")
        lines.append(f"- Tier A 来源占比: {round(gs.tier_a_ratio * 100)}%")
        lines.append(f"- 报告置信度: {final_report.report_confidence}")

    return "\n".join(lines)


def _build_initial_state(
    raw_input: str = "",
    pdf_text: str | None = None,
    report_mode: str = "draft",
) -> dict:
    """Build the initial AgentState dict for the new graph."""
    return {
        "raw_input": raw_input,
        "source_type": "arxiv",
        "report_mode": report_mode,
        "paper_type": None,
        "arxiv_id": None,
        "pdf_text": pdf_text,
        "source_manifest": None,
        "normalized_doc": None,
        "evidence": None,
        "report_frame": None,
        "draft_report": None,
        "resolved_report": None,
        "verified_report": None,
        "final_report": None,
        "draft_markdown": None,
        "full_markdown": None,
        "followup_hints": [],
        "tokens_used": 0,
        "warnings": [],
        "errors": [],
        "degradation_mode": "normal",
        "node_statuses": {},
    }


def generate_literature_report(
    agent: Runnable | None = None,
    arxiv_url_or_id: str | None = None,
    raw_text_content: str | None = None,
    callbacks: list[BaseCallbackHandler] | None = None,
    extra_system_context: str | None = None,
    chat_history: list[Any] | None = None,
) -> str:
    """
    Generate a literature report using the new StateGraph pipeline.

    Backward compatible: *agent* is accepted but ignored when the new graph
    is available.  Falls back to the old ReAct path only for multi-turn chat
    (when *chat_history* is provided together with an *agent*).
    """
    # Multi-turn chat: keep using the old agent (the graph is single-shot)
    if chat_history and agent:
        user_input = raw_text_content or arxiv_url_or_id
        if not user_input:
            return "Error: No input provided."
        messages = list(chat_history)
        messages.append(HumanMessage(content=user_input))
        config = {"callbacks": callbacks} if callbacks else None
        state = agent.invoke({"messages": messages}, config=config)
        return _last_ai_text(state)

    # New graph path
    from src.graph.builder import build_report_graph

    graph = build_report_graph()
    initial_state = _build_initial_state(
        raw_input=arxiv_url_or_id or "",
        pdf_text=raw_text_content if raw_text_content else None,
        report_mode="draft",
    )

    result = graph.invoke(initial_state)

    errors = result.get("errors", [])
    if errors and not result.get("final_report"):
        return "Error generating report:\n" + "\n".join(errors)

    final = result.get("final_report")
    if final:
        return _final_report_to_markdown(final)

    return "Error: No report generated."
