"""ClarifyAgent service — schema-bound research brief generation."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from enum import Enum
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src.agent.checkpointing import build_graph_config, get_langgraph_checkpointer
from src.agent.llm import build_reason_llm
from src.agent.report_frame import extract_json_block, extract_llm_text
from src.agent.settings import Settings
from src.research.policies.clarify_policy import is_brief_valid, to_limited_brief
from src.research.prompts.clarify_prompt import (
    CLARIFY_REPAIR_PROMPT,
    CLARIFY_SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    build_clarify_user_prompt,
)
from src.research.research_brief import ClarifyInput, ClarifyResult, ResearchBrief

logger = logging.getLogger(__name__)


class ParseStrategy(str, Enum):
    """Hierarchy of parsing strategies tried in order."""

    STRUCTURED_OUTPUT = "structured_output"
    JSON_PARSE = "json_parse"
    REPAIR = "repair"
    LIMITED = "limited"


class ClarifyGraphState(TypedDict, total=False):
    input: ClarifyInput
    settings: Settings
    user_prompt: str
    brief: ResearchBrief | None
    warnings: list[str]
    raw_text: str | None
    strategy_used: ParseStrategy
    emit_progress: Callable[[str], None] | None


def _emit_progress(emit_progress: Callable[[str], None] | None, message: str) -> None:
    """Emit progress message if callback is provided."""
    if emit_progress:
        emit_progress(message)


def _invoke_with_few_shot(
    settings: Settings, user_prompt: str, max_tokens: int = 8192
) -> str:
    """Send system + few-shot + user to LLM, return raw text."""
    llm = build_reason_llm(settings, max_tokens=max_tokens)
    messages = [
        SystemMessage(content=CLARIFY_SYSTEM_PROMPT),
        SystemMessage(content=FEW_SHOT_EXAMPLES),
        HumanMessage(content=user_prompt),
    ]
    resp = llm.invoke(messages)
    return extract_llm_text(resp)


def _try_structured_output(settings: Settings, user_prompt: str) -> ResearchBrief | None:
    """Try provider-native structured output. Returns None if unsupported."""
    try:
        llm = build_reason_llm(settings, max_tokens=8192)
        structured = llm.with_structured_output(ResearchBrief, method="json_mode")
        brief = structured.invoke([HumanMessage(content=user_prompt)])
        return brief
    except Exception as exc:
        logger.debug("Structured output not available (%s): %s", type(exc).__name__, exc)
        return None


def _try_json_parse(raw_text: str) -> ResearchBrief | None:
    """Parse raw LLM text as JSON → Pydantic model. Returns None on failure."""
    try:
        data = json.loads(extract_json_block(raw_text))
        return ResearchBrief.model_validate(data)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.debug("JSON parse failed: %s", exc)
        return None


def _try_repair(settings: Settings, bad_output: str) -> ResearchBrief | None:
    """Attempt to repair malformed output with a second LLM call."""
    try:
        prompt = CLARIFY_REPAIR_PROMPT.format(bad_output=bad_output)
        repaired_text = _invoke_with_few_shot(settings, prompt, max_tokens=4096)
        return _try_json_parse(repaired_text)
    except Exception as exc:
        logger.debug("Repair call failed: %s", exc)
        return None


def _fast_path_brief(input: ClarifyInput) -> ResearchBrief | None:
    """Optional deterministic shortcut for callers/tests that can avoid an LLM call.

    The production default is intentionally conservative to preserve the previous
    behavior: the LLM strategy chain still owns normal brief generation.
    """
    return None


def build_clarify_agent_graph(use_checkpointer: bool = False):
    """Build the official LangGraph strategy graph for ClarifyAgent.

    Args:
        use_checkpointer: If True, enables checkpointing. If False (default), the graph
            runs without checkpointing which avoids serialization issues with callable
            values like emit_progress.
    """
    workflow = StateGraph(ClarifyGraphState)
    workflow.add_node("prepare", _prepare_node)
    workflow.add_node("fast_path", _fast_path_node)
    workflow.add_node("structured_output", _structured_output_node)
    workflow.add_node("json_parse", _json_parse_node)
    workflow.add_node("repair", _repair_node)
    workflow.add_node("limited", _limited_node)
    workflow.add_node("post_validate", _post_validate_node)

    workflow.add_edge(START, "prepare")
    workflow.add_edge("prepare", "fast_path")
    workflow.add_conditional_edges(
        "fast_path",
        _route_after_fast_path,
        {"post_validate": "post_validate", "structured_output": "structured_output"},
    )
    workflow.add_conditional_edges(
        "structured_output",
        _route_after_structured_output,
        {"post_validate": "post_validate", "json_parse": "json_parse"},
    )
    workflow.add_conditional_edges(
        "json_parse",
        _route_after_json_parse,
        {"post_validate": "post_validate", "repair": "repair", "limited": "limited"},
    )
    workflow.add_conditional_edges(
        "repair",
        _route_after_repair,
        {"post_validate": "post_validate", "limited": "limited"},
    )
    workflow.add_edge("limited", "post_validate")
    workflow.add_edge("post_validate", END)
    if use_checkpointer:
        return workflow.compile(checkpointer=get_langgraph_checkpointer("clarify_agent"))
    return workflow.compile()


def _prepare_node(state: ClarifyGraphState) -> dict[str, Any]:
    input_obj = state["input"]
    emit_progress = state.get("emit_progress")
    _emit_progress(emit_progress, "Starting clarify pass from raw research query.")
    return {
        "settings": Settings.from_env(),
        "user_prompt": build_clarify_user_prompt(input_obj),
        "brief": None,
        "warnings": list(state.get("warnings", [])),
        "raw_text": None,
        "strategy_used": ParseStrategy.LIMITED,
    }


def _fast_path_node(state: ClarifyGraphState) -> dict[str, Any]:
    input_obj = state["input"]
    warnings = list(state.get("warnings", []))
    try:
        brief = _fast_path_brief(input_obj)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Fast-path clarify failed: {type(exc).__name__}: {exc}")
        return {"warnings": warnings}
    if brief is None:
        return {}
    return {
        "brief": brief,
        "strategy_used": ParseStrategy.LIMITED,
        "warnings": warnings,
    }


def _structured_output_node(state: ClarifyGraphState) -> dict[str, Any]:
    emit_progress = state.get("emit_progress")
    _emit_progress(emit_progress, "Trying provider-native structured output for ResearchBrief.")
    brief = _try_structured_output(state["settings"], state["user_prompt"])
    if brief is None:
        _emit_progress(emit_progress, "Structured output unavailable; falling back to JSON generation.")
        return {}
    _emit_progress(emit_progress, "Structured output succeeded.")
    return {"brief": brief, "strategy_used": ParseStrategy.STRUCTURED_OUTPUT}


def _json_parse_node(state: ClarifyGraphState) -> dict[str, Any]:
    emit_progress = state.get("emit_progress")
    warnings = list(state.get("warnings", []))
    try:
        _emit_progress(emit_progress, "Calling LLM for JSON-format brief.")
        raw_text = _invoke_with_few_shot(state["settings"], state["user_prompt"])
        brief = _try_json_parse(raw_text)
        if brief is not None:
            _emit_progress(emit_progress, "JSON parse succeeded.")
            return {
                "brief": brief,
                "raw_text": raw_text,
                "strategy_used": ParseStrategy.JSON_PARSE,
                "warnings": warnings,
            }
        _emit_progress(emit_progress, "JSON parse failed; attempting repair pass.")
        return {"raw_text": raw_text, "warnings": warnings}
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"JSON generation failed: {type(exc).__name__}: {exc}")
        _emit_progress(
            emit_progress,
            f"JSON generation failed with {type(exc).__name__}; continuing to fallback strategy.",
        )
        return {"warnings": warnings}


def _repair_node(state: ClarifyGraphState) -> dict[str, Any]:
    emit_progress = state.get("emit_progress")
    raw_text = state.get("raw_text")
    if raw_text is None:
        return {}
    repaired = _try_repair(state["settings"], raw_text)
    if repaired is None:
        _emit_progress(emit_progress, "Repair pass failed; using conservative fallback brief.")
        return {}
    warnings = list(state.get("warnings", []))
    warnings.append(
        "LLM output was malformed and required repair; "
        "some fields may be approximated."
    )
    _emit_progress(emit_progress, "Repair pass produced a valid brief.")
    return {
        "brief": repaired,
        "strategy_used": ParseStrategy.REPAIR,
        "warnings": warnings,
    }


def _limited_node(state: ClarifyGraphState) -> dict[str, Any]:
    input_obj = state["input"]
    emit_progress = state.get("emit_progress")
    warnings = list(state.get("warnings", []))
    _emit_progress(emit_progress, "Falling back to limited brief with needs_followup=True.")
    warnings.append(
        "All parsing strategies failed; returning a conservative limited brief. "
        "needs_followup is set to True."
    )
    return {
        "brief": to_limited_brief(input_obj.raw_query),
        "strategy_used": ParseStrategy.LIMITED,
        "warnings": warnings,
    }


def _post_validate_node(state: ClarifyGraphState) -> dict[str, Any]:
    brief = state.get("brief") or to_limited_brief(state["input"].raw_query)
    strategy_used = state.get("strategy_used", ParseStrategy.LIMITED)
    warnings = list(state.get("warnings", []))

    if not is_brief_valid(brief):
        warnings.append(
            f"Brief produced by strategy '{strategy_used.value}' failed post-validation; "
            "falling back to limited brief."
        )
        brief = to_limited_brief(state["input"].raw_query)

    if brief.confidence < 0.5:
        warnings.append(
            f"Low confidence score ({brief.confidence:.2f}); "
            "ambiguities may need human resolution."
        )
    if brief.needs_followup:
        warnings.append(
            "Brief has needs_followup=True; downstream planning should wait for "
            "human clarification or explicit disambiguation."
        )

    _emit_progress(
        state.get("emit_progress"),
        f"Clarify finished with strategy={strategy_used.value}, confidence={brief.confidence:.2f}.",
    )
    return {"brief": brief, "warnings": warnings, "strategy_used": strategy_used}


def _route_after_fast_path(state: ClarifyGraphState) -> str:
    return "post_validate" if state.get("brief") is not None else "structured_output"


def _route_after_structured_output(state: ClarifyGraphState) -> str:
    return "post_validate" if state.get("brief") is not None else "json_parse"


def _route_after_json_parse(state: ClarifyGraphState) -> str:
    if state.get("brief") is not None:
        return "post_validate"
    if state.get("raw_text") is not None:
        return "repair"
    return "limited"


def _route_after_repair(state: ClarifyGraphState) -> str:
    return "post_validate" if state.get("brief") is not None else "limited"


def run(
    input: ClarifyInput,
    emit_progress: Callable[[str], None] | None = None,
) -> ClarifyResult:
    """Main entry point — convert a raw research request to a ResearchBrief.

    Execution order:
      1. structured_output  (provider-native, preferred)
      2. json_parse          (JSON + Pydantic validation)
      3. repair              (second LLM call to fix malformed output)
      4. limited_brief       (conservative fallback, never crashes)

    Parameters
    ----------
    input : ClarifyInput
        Raw user query plus optional hints / context.

    Returns
    -------
    ClarifyResult
        brief   — valid ResearchBrief
        warnings — non-fatal notices (low confidence, significant ambiguity, etc.)
        raw_model_output — raw LLM text for debugging / thinking panel
    """
    graph = build_clarify_agent_graph(use_checkpointer=False)
    state = graph.invoke(
        {"input": input, "emit_progress": emit_progress},
        config=build_graph_config("clarify_agent", recursion_limit=16),
    )

    return ClarifyResult(
        brief=state["brief"],
        warnings=list(state.get("warnings", [])),
        raw_model_output=state.get("raw_text"),
    )
