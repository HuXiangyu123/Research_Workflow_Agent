"""Research workflow StateGraph builder.

Canonical workflow:
    clarify -> search_plan -> search -> extract -> draft -> review -> persist_artifacts
"""

from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from src.agent.checkpointing import get_langgraph_checkpointer
from src.graph.callbacks import NodeEventEmitter
from src.graph.instrumentation import instrument_node
from src.research.graph.nodes.clarify import run_clarify_node
from src.research.graph.nodes.draft import draft_node
from src.research.graph.nodes.extract import extract_node
from src.research.graph.nodes.extract_compression import extract_compression_node
from src.research.graph.nodes.persist_artifacts import persist_artifacts_node
from src.research.graph.nodes.review import review_node
from src.research.graph.nodes.search import search_node
from src.research.graph.nodes.search_plan import run_search_plan_node


def _route_after_clarify(state: dict) -> str:
    """After clarify: route to search_plan if brief exists, otherwise end."""
    brief = state.get("brief")
    if not brief:
        return END
    if state.get("awaiting_followup"):
        return END
    return "search_plan"


def _route_after_search_plan(state: dict) -> str:
    """After search_plan: stop for plan-only mode, otherwise continue to retrieval."""
    if not state.get("search_plan"):
        return END
    if state.get("research_depth", "plan") != "full":
        return END
    return "search"


def _route_after_review(state: dict) -> str:
    """Persist only reviewed/passed reports; failed reviews remain inspectable in task state."""
    return "persist_artifacts" if state.get("review_passed") else END


def _with_current_stage(node_name: str, fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
    def wrapped(state: dict) -> dict:
        output = fn(state)
        if isinstance(output, dict):
            output.setdefault("current_stage", node_name)
        return output

    return wrapped


def build_research_graph(
    event_emitter: NodeEventEmitter | None = None,
    *,
    use_checkpointer: bool = False,
):
    """Build the research workflow graph.

    Nodes:
      clarify -> search_plan -> search -> extract -> draft -> review -> persist_artifacts

    ``research_depth=plan`` preserves the lightweight clarify/search_plan flow
    for tests and explicit planning-only runs. Task execution sets
    ``research_depth=full``.
    """
    from src.graph.state import AgentState

    g = StateGraph(AgentState)

    g.add_node("clarify", instrument_node("clarify", run_clarify_node, event_emitter))
    g.add_node("search_plan", instrument_node("search_plan", run_search_plan_node, event_emitter))
    g.add_node("search", instrument_node("search", _with_current_stage("search", search_node), event_emitter))
    g.add_node("extract", instrument_node("extract", _with_current_stage("extract", extract_node), event_emitter))
    g.add_node(
        "extract_compression",
        instrument_node(
            "extract_compression",
            _with_current_stage("extract_compression", extract_compression_node),
            event_emitter,
        ),
    )
    g.add_node("draft", instrument_node("draft", _with_current_stage("draft", draft_node), event_emitter))
    g.add_node("review", instrument_node("review", _with_current_stage("review", review_node), event_emitter))
    g.add_node(
        "persist_artifacts",
        instrument_node(
            "persist_artifacts",
            _with_current_stage("persist_artifacts", persist_artifacts_node),
            event_emitter,
        ),
    )

    g.add_edge(START, "clarify")

    g.add_conditional_edges("clarify", _route_after_clarify, {
        "search_plan": "search_plan",
        END: END,
    })

    g.add_conditional_edges("search_plan", _route_after_search_plan, {
        "search": "search",
        END: END,
    })
    g.add_edge("search", "extract")
    g.add_edge("extract", "extract_compression")
    g.add_edge("extract_compression", "draft")
    g.add_edge("draft", "review")
    g.add_conditional_edges("review", _route_after_review, {
        "persist_artifacts": "persist_artifacts",
        END: END,
    })
    g.add_edge("persist_artifacts", END)

    if use_checkpointer:
        return g.compile(checkpointer=get_langgraph_checkpointer("research_graph"))
    return g.compile()
