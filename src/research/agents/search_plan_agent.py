"""SearchPlanAgent — 真实 Agent 循环：工具调用 + 工作记忆 + 反思 + 有界迭代。"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src.agent.checkpointing import build_graph_config, get_langgraph_checkpointer
from src.agent.llm import build_quick_llm
from src.agent.report_frame import extract_json_block, extract_llm_text
from src.agent.settings import Settings
from src.models.research import (
    SearchPlan,
    SearchPlannerMemory,
    SearchPlanResult,
)
from src.research.policies.search_plan_policy import should_stop, to_fallback_plan
from src.research.prompts.search_plan_prompt import (
    FEW_SHOT_EXAMPLES,
    SEARCHPLAN_SYSTEM_PROMPT,
    build_reflection_prompt,
)

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10


class SearchPlanGraphState(TypedDict, total=False):
    brief: dict[str, Any]
    emit_progress: Callable[[str], None] | None
    settings: Settings
    memory: SearchPlannerMemory
    raw_output: str | None
    warnings: list[str]
    plan: SearchPlan | None
    current_results: str
    brief_str: str
    topic: str
    expanded_kw: list[str]
    queries_to_try: list[str]
    iteration_results: list[str]
    stop_reason: str | None


def _emit_progress(emit_progress: Callable[[str], None] | None, message: str) -> None:
    if emit_progress:
        emit_progress(message)


def _parse_json(text: str) -> dict | None:
    try:
        raw = extract_json_block(text)
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def _validate_plan(data: dict) -> SearchPlan | None:
    try:
        return SearchPlan.model_validate(data)
    except Exception:
        return None


def _invoke_llm(
    settings: Settings,
    system_content: str,
    few_shot: str,
    user_content: str,
    max_tokens: int = 8192,
) -> str:
    llm = build_quick_llm(settings, max_tokens=max_tokens)
    messages = [
        SystemMessage(content=system_content),
        SystemMessage(content=few_shot),
        HumanMessage(content=user_content),
    ]
    resp = llm.invoke(messages)
    return extract_llm_text(resp)


# ─── 工具调用（按类型分组）────────────────────────────────────────────────────


def _call_search_arxiv(query: str, top_k: int = 10) -> dict[str, Any]:
    """调用 search_arxiv 工具，返回原始结果 dict。"""
    from src.tools.search_tools import search_arxiv as _fn

    try:
        return {"ok": True, "result": _fn.invoke({"query": query, "top_k": top_k})}
    except Exception as exc:
        logger.warning("search_arxiv failed for '%s': %s", query, exc)
        return {"ok": False, "error": str(exc)}


def _call_expand_keywords(topic: str, focus: str = "methods") -> dict[str, Any]:
    from src.tools.search_tools import expand_keywords as _fn

    try:
        return {"ok": True, "result": _fn.invoke({"topic": topic, "focus_dimension": focus})}
    except Exception as exc:
        logger.warning("expand_keywords failed for '%s': %s", topic, exc)
        return {"ok": False, "error": str(exc)}


def _call_summarize_hits(results: str) -> dict[str, Any]:
    from src.tools.search_tools import summarize_hits as _fn

    try:
        return {"ok": True, "result": _fn.invoke({"results": results})}
    except Exception as exc:
        logger.warning("summarize_hits failed: %s", exc)
        return {"ok": False, "error": str(exc)}


def _call_detect_noise(results: str) -> dict[str, Any]:
    from src.tools.search_tools import detect_sparse_or_noisy_queries as _fn

    try:
        return {"ok": True, "result": _fn.invoke({"results": results})}
    except Exception as exc:
        logger.warning("detect_sparse_or_noisy_queries failed: %s", exc)
        return {"ok": False, "error": str(exc)}


def _count_results(result_str: str) -> int:
    """从工具输出字符串中估算命中数量。"""
    lines = result_str.strip().split("\n")
    count = sum(1 for l in lines if l.startswith("[") and "] " in l)
    return count


# ─── LangGraph Agent Loop ─────────────────────────────────────────────────────


def build_search_plan_agent_graph():
    """Build the SearchPlanAgent bounded-iteration graph."""
    workflow = StateGraph(SearchPlanGraphState)
    workflow.add_node("initialize", _initialize_node)
    workflow.add_node("expand_keywords", _expand_keywords_node)
    workflow.add_node("iteration_start", _iteration_start_node)
    workflow.add_node("propose_queries", _propose_queries_node)
    workflow.add_node("run_searches", _run_searches_node)
    workflow.add_node("reflect", _reflect_node)
    workflow.add_node("generate_plan", _generate_plan_node)
    workflow.add_node("finalize", _finalize_node)

    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "expand_keywords")
    workflow.add_edge("expand_keywords", "iteration_start")
    workflow.add_conditional_edges(
        "iteration_start",
        _route_after_iteration_start,
        {"propose_queries": "propose_queries", "finalize": "finalize"},
    )
    workflow.add_edge("propose_queries", "run_searches")
    workflow.add_conditional_edges(
        "run_searches",
        _route_after_searches,
        {
            "reflect": "reflect",
            "iteration_start": "iteration_start",
            "finalize": "finalize",
        },
    )
    workflow.add_edge("reflect", "generate_plan")
    workflow.add_conditional_edges(
        "generate_plan",
        _route_after_generate_plan,
        {"iteration_start": "iteration_start", "finalize": "finalize"},
    )
    workflow.add_edge("finalize", END)
    return workflow.compile(checkpointer=get_langgraph_checkpointer("search_plan_agent"))


def run(
    brief: dict[str, Any],
    emit_progress: Callable[[str], None] | None = None,
) -> SearchPlanResult:
    """SearchPlanAgent 主入口：LangGraph 有界 Agent 循环。

    流程：初始化 → 工具观察 → 记忆更新 → 反思 → 修订/停止
    """
    state = build_search_plan_agent_graph().invoke(
        {"brief": brief, "emit_progress": emit_progress},
        config=build_graph_config(
            "search_plan_agent",
            recursion_limit=MAX_ITERATIONS * 8 + 20,
        ),
    )
    return SearchPlanResult(
        plan=state["plan"],
        memory=state["memory"],
        warnings=list(state.get("warnings", [])),
        raw_model_output=state.get("raw_output"),
    )


def _copy_memory(state: SearchPlanGraphState) -> SearchPlannerMemory:
    memory = state.get("memory") or SearchPlannerMemory()
    return memory.model_copy(deep=True)


def _initialize_node(state: SearchPlanGraphState) -> dict[str, Any]:
    brief = state.get("brief") or {}
    settings = Settings.from_env()
    memory = SearchPlannerMemory()
    brief_str = json.dumps(brief, ensure_ascii=False, indent=2)
    topic = brief.get("research_topic") or brief.get("topic", "")

    memory.iteration_count = 0
    memory.remaining_budget = MAX_ITERATIONS
    _emit_progress(state.get("emit_progress"), "Initializing search planning from research brief.")
    return {
        "settings": settings,
        "memory": memory,
        "raw_output": None,
        "warnings": list(state.get("warnings", [])),
        "plan": None,
        "current_results": "",
        "brief_str": brief_str,
        "topic": topic,
        "expanded_kw": [],
        "queries_to_try": [],
        "iteration_results": [],
        "stop_reason": None,
    }


def _expand_keywords_node(state: SearchPlanGraphState) -> dict[str, Any]:
    memory = _copy_memory(state)
    warnings = list(state.get("warnings", []))
    expanded_kw: list[str] = []
    emit_progress = state.get("emit_progress")
    try:
        _emit_progress(emit_progress, "Expanding topic keywords for first-pass retrieval.")
        kw_result = _call_expand_keywords(state.get("topic", ""), "methods")
        if kw_result["ok"]:
            expanded_kw = [l.strip("- ").strip() for l in kw_result["result"].split("\n") if l.strip()]
            memory.planner_reflections.append(f"关键词扩展得到 {len(expanded_kw)} 个候选词")
            _emit_progress(emit_progress, f"Keyword expansion produced {len(expanded_kw)} candidate phrases.")
    except Exception as exc:
        warnings.append(f"关键词扩展失败：{exc}，使用原始关键词继续")
        _emit_progress(emit_progress, f"Keyword expansion failed: {exc}. Continuing with raw topic.")

    return {"expanded_kw": expanded_kw, "memory": memory, "warnings": warnings}


def _iteration_start_node(state: SearchPlanGraphState) -> dict[str, Any]:
    memory = _copy_memory(state)
    emit_progress = state.get("emit_progress")
    plan = state.get("plan")
    stop_reason: str | None = None

    if memory.iteration_count >= MAX_ITERATIONS:
        return {"memory": memory, "stop_reason": "已达最大迭代次数"}

    memory.iteration_count += 1
    memory.remaining_budget = max(memory.remaining_budget - 1, 0)
    _emit_progress(
        emit_progress,
        f"Iteration {memory.iteration_count}: planning with {len(memory.attempted_queries)} attempted queries so far.",
    )

    if plan is not None:
        stop, reason = should_stop(memory, plan)
        if stop:
            stop_reason = reason
            memory.last_action = f"stop: {reason}"
            logger.info("SearchPlanAgent stopping: %s", reason)

    return {"memory": memory, "stop_reason": stop_reason, "iteration_results": []}


def _propose_queries_node(state: SearchPlanGraphState) -> dict[str, Any]:
    memory = _copy_memory(state)
    queries_to_try = _propose_queries(
        memory.iteration_count,
        state.get("topic", ""),
        list(state.get("expanded_kw", [])),
        memory,
    )
    _emit_progress(
        state.get("emit_progress"),
        f"Iteration {memory.iteration_count}: searching {len(queries_to_try)} queries in arXiv/SearXNG toolchain.",
    )
    return {"queries_to_try": queries_to_try}


def _run_searches_node(state: SearchPlanGraphState) -> dict[str, Any]:
    memory = _copy_memory(state)
    current_results = state.get("current_results", "")
    iteration_results: list[str] = []

    for q in state.get("queries_to_try", []):
        if q in memory.attempted_queries:
            continue
        memory.attempted_queries.append(q)

        res = _call_search_arxiv(q, top_k=10)
        if res["ok"]:
            raw_result = res["result"]
            iteration_results.append(raw_result)
            hit_count = _count_results(raw_result)
            memory.query_to_hits[q] = hit_count
            if hit_count == 0:
                memory.empty_queries.append(q)
            current_results += raw_result + "\n\n"
        else:
            memory.planner_reflections.append(f"查询 '{q}' 失败：{res.get('error')}")

    if not iteration_results:
        memory.last_action = "no_results"
        _emit_progress(
            state.get("emit_progress"),
            f"Iteration {memory.iteration_count}: no usable search results, continuing.",
        )

    return {
        "memory": memory,
        "iteration_results": iteration_results,
        "current_results": current_results,
    }


def _reflect_node(state: SearchPlanGraphState) -> dict[str, Any]:
    memory = _copy_memory(state)
    iteration = memory.iteration_count
    current_results = state.get("current_results", "")
    emit_progress = state.get("emit_progress")
    try:
        _emit_progress(emit_progress, f"Iteration {iteration}: summarizing current hits and checking noise.")
        sum_result = _call_summarize_hits(current_results)
        if sum_result["ok"]:
            memory.planner_reflections.append(
                f"Iteration {iteration} summary: {sum_result['result'][:200]}"
            )

        noise_result = _call_detect_noise(current_results)
        if noise_result["ok"]:
            text = noise_result["result"]
            for line in text.split("\n"):
                if "noisy" in line.lower() or "sparse" in line.lower():
                    memory.high_noise_queries.append(line.strip())
    except Exception as exc:
        memory.planner_reflections.append(f"Reflection-stage exception: {exc}")
        _emit_progress(emit_progress, f"Iteration {iteration}: reflection tools raised {exc}.")

    return {"memory": memory}


def _generate_plan_node(state: SearchPlanGraphState) -> dict[str, Any]:
    memory = _copy_memory(state)
    warnings = list(state.get("warnings", []))
    plan = state.get("plan")
    raw_output = state.get("raw_output")
    iteration = memory.iteration_count
    memory_dict = memory.model_dump()
    reflection_prompt = build_reflection_prompt(memory_dict)

    user_prompt = f"""\
## ResearchBrief

{state.get("brief_str", "{}")}

## Current Search Memory

{reflection_prompt}

## Current Search Result Summary

{state.get("current_results", "")[:3000] if state.get("current_results") else "(no results yet)"}

Generate the best current SearchPlan JSON from the context above.
If coverage is already sufficient, choose STOP.
If more search is needed, call tools first and then output JSON.
"""

    try:
        _emit_progress(
            state.get("emit_progress"),
            f"Iteration {iteration}: asking LLM to synthesize the current SearchPlan.",
        )
        raw_output = _invoke_llm(
            state["settings"],
            SEARCHPLAN_SYSTEM_PROMPT,
            FEW_SHOT_EXAMPLES,
            user_prompt,
        )
    except Exception as exc:
        logger.warning("Iteration %d LLM call failed: %s", iteration, exc)
        warnings.append(f"Iteration {iteration} LLM call failed: {exc}")
        memory.last_action = f"llm_failed: {exc}"
        _emit_progress(state.get("emit_progress"), f"Iteration {iteration}: LLM call failed with {exc}.")
        if plan is None:
            plan = to_fallback_plan(state.get("brief") or {})
            warnings.append("LLM failed; downgraded to policy fallback plan")
            _emit_progress(state.get("emit_progress"), "Using fallback plan after repeated LLM failure.")
        return {
            "memory": memory,
            "warnings": warnings,
            "plan": plan,
            "raw_output": raw_output,
        }

    data = _parse_json(raw_output)
    if data:
        new_plan = _validate_plan(data)
        if new_plan is not None:
            plan = new_plan
            _emit_progress(
                state.get("emit_progress"),
                f"Iteration {iteration}: generated a valid SearchPlan with {len(plan.query_groups)} query groups.",
            )

    stop_reason: str | None = None
    if plan is not None:
        stop, reason = should_stop(memory, plan)
        if stop:
            stop_reason = reason
            memory.last_action = f"stop: {reason}"
            _emit_progress(state.get("emit_progress"), f"Stopping planner: {reason}.")

    return {
        "memory": memory,
        "warnings": warnings,
        "plan": plan,
        "raw_output": raw_output,
        "stop_reason": stop_reason,
    }


def _finalize_node(state: SearchPlanGraphState) -> dict[str, Any]:
    memory = _copy_memory(state)
    warnings = list(state.get("warnings", []))
    plan = state.get("plan")
    if plan is None:
        warnings.append("Agent loop produced no valid plan; using fallback")
        plan = to_fallback_plan(state.get("brief") or {})
        memory.last_action = "fallback"
        memory.planner_reflections.append("Fallback plan used")
        _emit_progress(
            state.get("emit_progress"),
            "Planner produced no valid output; falling back to policy-generated plan.",
        )

    _emit_progress(
        state.get("emit_progress"),
        f"Search planning finished after {memory.iteration_count} iterations; remaining budget={memory.remaining_budget}.",
    )
    return {"plan": plan, "memory": memory, "warnings": warnings}


def _route_after_iteration_start(state: SearchPlanGraphState) -> str:
    if state.get("stop_reason"):
        return "finalize"
    memory = state.get("memory") or SearchPlannerMemory()
    if memory.iteration_count > MAX_ITERATIONS:
        return "finalize"
    return "propose_queries"


def _route_after_searches(state: SearchPlanGraphState) -> str:
    if state.get("iteration_results"):
        return "reflect"
    memory = state.get("memory") or SearchPlannerMemory()
    if memory.iteration_count >= MAX_ITERATIONS:
        return "finalize"
    return "iteration_start"


def _route_after_generate_plan(state: SearchPlanGraphState) -> str:
    if state.get("stop_reason"):
        return "finalize"
    memory = state.get("memory") or SearchPlannerMemory()
    if memory.iteration_count >= MAX_ITERATIONS:
        return "finalize"
    return "iteration_start"


# ─── 查询提议策略 ───────────────────────────────────────────────────────────────


def _propose_queries(
    iteration: int,
    topic: str,
    expanded_kw: list[str],
    memory: SearchPlannerMemory,
) -> list[str]:
    """根据迭代次数和当前状态提议候选查询列表。"""
    queries: list[str] = []

    if iteration == 1:
        # 第一轮：直接搜索主题 + 最核心的关键词
        queries.append(topic)
        if expanded_kw:
            queries.append(expanded_kw[0])

    elif iteration == 2:
        # 第二轮：扩展方向（如果有关键词）
        if len(expanded_kw) > 1:
            queries.append(expanded_kw[1])
        queries.append(f"{topic} survey review")
        queries.append(f"{topic} benchmark dataset")

    elif iteration == 3:
        # 第三轮：方法细化和应用方向
        if len(expanded_kw) > 2:
            queries.append(expanded_kw[2])
        queries.append(f"{topic} application")
        queries.append(f"{topic} limitation")

    else:
        # 后续迭代：针对 empty queries 和 coverage gap
        for eq in memory.empty_queries[-3:]:
            if eq not in memory.attempted_queries:
                queries.append(f"{eq} tutorial")
        # 针对 subquestion gaps（如果有）
        for kw in expanded_kw[3:6]:
            if kw not in memory.attempted_queries:
                queries.append(kw)

    # 兜底：确保至少有一个查询
    if not queries:
        queries = [topic]

    # 去重（已尝试过的跳过）
    return [q for q in queries if q not in memory.attempted_queries]
