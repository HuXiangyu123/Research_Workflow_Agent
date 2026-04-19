from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from src.agent.report_markdown import render_report_markdown
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
    return render_report_markdown(
        sections=final_report.sections,
        citations=final_report.citations,
        grounding_stats=final_report.grounding_stats,
        report_confidence=final_report.report_confidence,
    )


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
    task_id: str | None = None,
) -> str:
    """
    Generate a literature report using the new StateGraph pipeline.

    Backward compatible: *agent* is accepted but ignored when the new graph
    is available.  Falls back to the old ReAct path only for multi-turn chat
    (when *chat_history* is provided together with an *agent*).

    If *task_id* is provided, the report is persisted to output/<task_id>/report.md.
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
        report_md = _final_report_to_markdown(final)
        # Persist report if task_id provided
        if task_id:
            from src.agent.output_workspace import write_report
            write_report(task_id, report_md)
        return report_md

    return "Error: No report generated."
