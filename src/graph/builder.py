from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agent.checkpointing import get_langgraph_checkpointer
from src.graph.callbacks import NodeEventEmitter
from src.graph.instrumentation import instrument_node
from src.graph.nodes.apply_policy import apply_policy
from src.graph.nodes.classify_paper_type import classify_paper_type
from src.graph.nodes.draft_report import draft_report
from src.graph.nodes.extract_document_text import extract_document_text
from src.graph.nodes.format_output import format_output
from src.graph.nodes.ingest_source import ingest_source
from src.graph.nodes.input_parse import input_parse
from src.graph.nodes.normalize_metadata import normalize_metadata
from src.graph.nodes.repair_report import repair_report
from src.graph.nodes.report_frame import report_frame
from src.graph.nodes.resolve_citations import resolve_citations
from src.graph.nodes.retrieve_evidence import retrieve_evidence
from src.graph.nodes.survey_intro_outline import survey_intro_outline
from src.graph.nodes.verify_claims import verify_claims
from src.graph.state import AgentState


def _should_abort(state: dict) -> str:
    if state.get("degradation_mode") == "safe_abort":
        return "format_output"
    return "continue"


def _select_generation_path(state: dict) -> str:
    report_mode = state.get("report_mode", "draft")
    paper_type = state.get("paper_type", "regular")
    if report_mode == "draft":
        return "draft_report"
    if paper_type == "survey":
        return "survey_intro_outline"
    return "report_frame"


def _has_draft(state: dict) -> str:
    """Guard after generation nodes: skip post-processing if no draft produced."""
    if state.get("draft_report") is not None:
        return "continue"
    return "format_output"


def build_report_graph(
    event_emitter: NodeEventEmitter | None = None,
    *,
    use_checkpointer: bool = False,
):
    g = StateGraph(AgentState)

    g.add_node("input_parse", instrument_node("input_parse", input_parse, event_emitter))
    g.add_node("ingest_source", instrument_node("ingest_source", ingest_source, event_emitter))
    g.add_node(
        "extract_document_text",
        instrument_node("extract_document_text", extract_document_text, event_emitter),
    )
    g.add_node(
        "normalize_metadata",
        instrument_node("normalize_metadata", normalize_metadata, event_emitter),
    )
    g.add_node(
        "classify_paper_type",
        instrument_node("classify_paper_type", classify_paper_type, event_emitter),
    )
    g.add_node(
        "retrieve_evidence",
        instrument_node("retrieve_evidence", retrieve_evidence, event_emitter),
    )
    g.add_node("draft_report", instrument_node("draft_report", draft_report, event_emitter))
    g.add_node("report_frame", instrument_node("report_frame", report_frame, event_emitter))
    g.add_node(
        "survey_intro_outline",
        instrument_node("survey_intro_outline", survey_intro_outline, event_emitter),
    )
    g.add_node("repair_report", instrument_node("repair_report", repair_report, event_emitter))
    g.add_node(
        "resolve_citations",
        instrument_node("resolve_citations", resolve_citations, event_emitter),
    )
    g.add_node("verify_claims", instrument_node("verify_claims", verify_claims, event_emitter))
    g.add_node("apply_policy", instrument_node("apply_policy", apply_policy, event_emitter))
    g.add_node("format_output", instrument_node("format_output", format_output, event_emitter))

    g.set_entry_point("input_parse")
    g.add_edge("input_parse", "ingest_source")
    g.add_edge("ingest_source", "extract_document_text")
    g.add_edge("extract_document_text", "normalize_metadata")

    g.add_conditional_edges(
        "normalize_metadata",
        _should_abort,
        {"continue": "retrieve_evidence", "format_output": "format_output"},
    )
    g.add_edge("retrieve_evidence", "classify_paper_type")
    g.add_conditional_edges(
        "classify_paper_type",
        _select_generation_path,
        {
            "draft_report": "draft_report",
            "report_frame": "report_frame",
            "survey_intro_outline": "survey_intro_outline",
        },
    )

    # Draft path: draft → repair → guard
    g.add_edge("draft_report", "repair_report")
    g.add_conditional_edges(
        "repair_report",
        _has_draft,
        {"continue": "resolve_citations", "format_output": "format_output"},
    )

    # Full-report path: report_frame → guard
    g.add_conditional_edges(
        "report_frame",
        _has_draft,
        {"continue": "resolve_citations", "format_output": "format_output"},
    )

    # Survey path: survey_intro_outline → guard
    g.add_conditional_edges(
        "survey_intro_outline",
        _has_draft,
        {"continue": "resolve_citations", "format_output": "format_output"},
    )

    g.add_edge("resolve_citations", "verify_claims")
    g.add_edge("verify_claims", "apply_policy")
    g.add_edge("apply_policy", "format_output")
    g.add_edge("format_output", END)

    if use_checkpointer:
        return g.compile(checkpointer=get_langgraph_checkpointer("report_graph"))
    return g.compile()
