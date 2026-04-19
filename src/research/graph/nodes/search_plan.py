"""SearchPlan node — wraps SearchPlanAgent for LangGraph."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


class SearchPlanNodeOutput(TypedDict):
    search_plan: dict[str, Any] | None
    search_plan_warnings: list[str]
    current_stage: str


def run_search_plan_node(state: dict) -> SearchPlanNodeOutput:
    """Execute SearchPlanAgent given the current state.

    Expects ``state["brief"]`` to be a dict (JSON-serializable ResearchBrief).
    Outputs a dict-serializable SearchPlan.
    """
    from src.research.agents.search_plan_agent import run

    brief = state.get("brief")
    emitter = state.get("_event_emitter")

    def emit_progress(message: str) -> None:
        if emitter:
            emitter.on_thinking("search_plan", message)

    if not brief:
        logger.warning("search_plan node called without brief in state")
        return SearchPlanNodeOutput(
            search_plan=None,
            search_plan_warnings=["No brief available in state"],
            current_stage="search_plan_failed",
        )

    if state.get("awaiting_followup"):
        return SearchPlanNodeOutput(
            search_plan=None,
            search_plan_warnings=[
                "Clarify requested follow-up in interactive mode; search planning paused"
            ],
            current_stage="clarify_followup_required",
        )

    # Use fallback plan when:
    # 1. needs_followup=True (clarify flagged ambiguity)
    # 2. confidence is low (<0.5)
    # 3. explicit use_heuristic flag
    needs_followup = brief.get("needs_followup", False)
    confidence = brief.get("confidence", 1.0)
    use_heuristic = state.get("use_heuristic", False) or needs_followup or confidence < 0.5

    if use_heuristic:
        from src.research.policies.search_plan_policy import to_fallback_plan

        plan = to_fallback_plan(brief)
        warnings = ["SearchPlanAgent fast path used heuristic fallback plan"]
        if needs_followup:
            warnings.append(f"Brief flagged needs_followup=True (confidence={confidence:.2f})")
        return SearchPlanNodeOutput(
            search_plan=plan.model_dump(mode="json"),
            search_plan_warnings=warnings,
            current_stage="search_plan",
        )

    try:
        result = run(brief, emit_progress=emit_progress)
        logger.info(
            "SearchPlanAgent completed in %d iterations, budget left: %d",
            result.memory.iteration_count,
            result.memory.remaining_budget,
        )

        return SearchPlanNodeOutput(
            search_plan=result.plan.model_dump(mode="json"),
            search_plan_warnings=result.warnings,
            current_stage="search_plan",
        )
    except Exception:
        logger.exception("SearchPlanAgent raised unhandled exception, falling back to heuristic")
        from src.research.policies.search_plan_policy import to_fallback_plan

        plan = to_fallback_plan(brief)
        return SearchPlanNodeOutput(
            search_plan=plan.model_dump(mode="json"),
            search_plan_warnings=["SearchPlanAgent crashed, used fallback plan"],
            current_stage="search_plan",
        )
