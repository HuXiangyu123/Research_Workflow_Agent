"""Validate one research agent with a deterministic local smoke run.

Usage:
    python -m src.cmd.validate_agent clarify
    python -m src.cmd.validate_agent search_plan
    python -m src.cmd.validate_agent all
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

from dotenv import load_dotenv


AgentValidator = Callable[[], dict[str, Any]]


def _sample_brief() -> dict[str, Any]:
    return {
        "topic": "RAG",
        "goal": "survey_drafting",
        "desired_output": "survey_outline",
        "sub_questions": ["What retrieval strategies work best?"],
        "time_range": "2023-2026",
        "domain_scope": None,
        "source_constraints": [],
        "focus_dimensions": ["methods", "benchmarks"],
        "ambiguities": [],
        "needs_followup": False,
        "confidence": 0.9,
        "schema_version": "v1",
    }


def _ok(agent: str, result: dict[str, Any]) -> dict[str, Any]:
    return {"agent": agent, "ok": True, "result": result}


def validate_clarify() -> dict[str, Any]:
    from src.research.agents import clarify_agent
    from src.research.policies.clarify_policy import to_limited_brief
    from src.research.research_brief import ClarifyInput

    input_obj = ClarifyInput(raw_query="RAG survey_outline methods benchmarks")
    with patch.object(
        clarify_agent,
        "_fast_path_brief",
        lambda inp: to_limited_brief(inp.raw_query),
    ):
        result = clarify_agent.run(input_obj)

    graph_nodes = sorted(clarify_agent.build_clarify_agent_graph().get_graph().nodes)
    return _ok(
        "clarify",
        {
            "topic": result.brief.topic,
            "needs_followup": result.brief.needs_followup,
            "graph_nodes": graph_nodes,
        },
    )


def validate_search_plan() -> dict[str, Any]:
    from src.research.agents.search_plan_agent import build_search_plan_agent_graph
    from src.research.graph.nodes.search_plan import run_search_plan_node

    result = run_search_plan_node({"brief": _sample_brief(), "use_heuristic": True})
    graph_nodes = sorted(build_search_plan_agent_graph().get_graph().nodes)
    return _ok(
        "search_plan",
        {
            "current_stage": result["current_stage"],
            "query_groups": len((result["search_plan"] or {}).get("query_groups", [])),
            "graph_nodes": graph_nodes,
        },
    )


def validate_planner() -> dict[str, Any]:
    from src.models.research import SearchPlan
    from src.research.agents.planner_agent import PlannerAgent

    agent = PlannerAgent()
    plan = SearchPlan.model_validate(
        {
            "plan_goal": "Collect recent RAG papers",
            "query_groups": [
                {
                    "group_id": "g1",
                    "queries": ["retrieval augmented generation"],
                    "intent": "exploration",
                    "priority": 1,
                    "expected_hits": 5,
                }
            ],
        }
    )

    agent.plan_phase = lambda brief: {"plan": plan, "phases": ["phase-1"]}  # type: ignore[method-assign]
    agent.execute_phase = lambda phases: {  # type: ignore[method-assign]
        "candidates": [{"title": "Paper 1"}],
        "execution_log": [{"phase": "phase-1"}],
    }
    agent.validate_phase = lambda brief, plan_obj, execution_results: {  # type: ignore[method-assign]
        "validation": {"status": "complete"},
        "candidates": execution_results["candidates"],
        "execution_log": execution_results["execution_log"],
    }

    result = agent.run(_sample_brief())
    return _ok(
        "planner",
        {
            "candidate_count": len(result["candidates"]),
            "validation": result["validation"],
            "graph_nodes": sorted(agent.build_graph().get_graph().nodes),
        },
    )


def validate_retriever() -> dict[str, Any]:
    from src.research.agents.retriever_agent import RetrieverAgent

    agent = RetrieverAgent(workspace_id="cmd-smoke")
    agent._augmented_query_gen = lambda brief, search_plan=None: {  # type: ignore[method-assign]
        "queries": [{"query": "rag", "sources": ["arxiv"], "expected_hits": 5}]
    }
    agent._parallel_retrieval = lambda queries: {  # type: ignore[method-assign]
        "raw_results": [
            {
                "query": "rag",
                "hits": [{"title": "RAG Paper", "url": "https://example.com/p1", "content": "abstract"}],
            }
        ]
    }
    agent._context_assembly = lambda brief, raw_results: [  # type: ignore[method-assign]
        {
            "rank": 1,
            "title": "RAG Paper",
            "url": "https://example.com/p1",
            "abstract": "abstract",
            "source": "arxiv",
        }
    ]

    with patch("src.research.graph.nodes.search._ingest_paper_candidates", lambda *a, **kw: None):
        result = agent.run(_sample_brief(), {"plan_goal": "Collect RAG"})

    return _ok(
        "retriever",
        {
            "queries_generated": result["queries_generated"],
            "candidate_count": len(result["rag_result"]["paper_candidates"]),
            "graph_nodes": sorted(agent.build_graph().get_graph().nodes),
        },
    )


def validate_analyst() -> dict[str, Any]:
    from src.research.agents.analyst_agent import AnalystAgent
    from src.models.report import DraftReport

    agent = AnalystAgent()
    agent._build_structured_cards = lambda paper_cards: {  # type: ignore[method-assign]
        "cards": [{"title": "Paper 1", "methods": ["Dense retrieval"], "datasets": ["MS MARCO"]}],
        "confidence": 0.7,
    }
    agent._build_comparison_matrix = lambda artifacts: {  # type: ignore[method-assign]
        "matrix": {"rows": [{"paper": "Paper 1", "methods": "Dense retrieval"}]},
        "confidence": 0.8,
    }
    agent._build_outline = lambda state, brief=None: {  # type: ignore[method-assign]
        "outline": {"introduction": ["Background"], "methods": ["Dense retrieval"]},
        "confidence": 0.75,
    }
    agent._build_report_draft = lambda state: {  # type: ignore[method-assign]
        "draft": {
            "sections": {
                "title": "RAG Survey",
                "abstract": "Summary",
                "introduction": "Intro",
                "methods": "Methods",
                "conclusion": "Done",
            },
            "claims": [],
            "citations": [],
        },
        "confidence": 0.85,
    }
    agent._needs_grounded_redraft = lambda draft_report: False  # type: ignore[method-assign]
    agent._store_artifacts_memory = lambda state: None  # type: ignore[method-assign]

    with patch(
        "src.research.graph.nodes.draft._build_draft_report",
        lambda cards, brief: DraftReport(
            sections={
                "title": "RAG Survey",
                "abstract": "Summary",
                "introduction": "Intro",
                "background": "Background",
                "taxonomy": "Taxonomy",
                "methods": "Methods",
                "datasets": "Datasets",
                "evaluation": "Evaluation",
                "discussion": "Discussion",
                "future_work": "Future work",
                "conclusion": "Done",
            },
            claims=[],
            citations=[],
        ),
    ):
        result = agent.run(_sample_brief(), [{"title": "Paper 1"}])
    return _ok(
        "analyst",
        {
            "title": result["draft_report"].sections["title"],
            "overall_confidence": result["overall_confidence"],
            "graph_nodes": sorted(agent.build_graph().get_graph().nodes),
        },
    )


def validate_reviewer() -> dict[str, Any]:
    from src.research.agents.reviewer_agent import ReviewerAgent, SelfReflection

    agent = ReviewerAgent()
    attempts: list[int] = []
    agent._retrieve_reflections = lambda brief, draft_report: []  # type: ignore[method-assign]

    def fake_actor_review(**kwargs):
        attempt = kwargs["attempt"]
        attempts.append(attempt)
        return {"confidence": 0.3 if attempt == 1 else 0.9}

    agent._actor_review = fake_actor_review  # type: ignore[method-assign]
    agent._evaluate = lambda actor_result, attempt: {  # type: ignore[method-assign]
        "passed": attempt == 2,
        "confidence": actor_result["confidence"],
        "reason": "retry" if attempt == 1 else "ok",
        "task_type": "review_confidence",
        "issues": ["low confidence"] if attempt == 1 else [],
    }
    agent._self_reflect = lambda **kwargs: SelfReflection(  # type: ignore[method-assign]
        reflection_id="refl_1",
        task_type="review_confidence",
        failure_context="low confidence",
        root_cause="insufficient coverage",
        lessons=["improve retrieval"],
        improved_strategy="retrieve more evidence",
        confidence_gain=0.1,
        created_at=0.0,
    )
    agent._store_reflection = lambda reflection: None  # type: ignore[method-assign]

    result = agent.run(_sample_brief(), [{"title": "Paper 1"}], {"sections": {"introduction": "Intro"}})
    return _ok(
        "reviewer",
        {
            "attempts": attempts,
            "review_passed": result["review_passed"],
            "graph_nodes": sorted(agent.build_graph().get_graph().nodes),
        },
    )


async def _validate_supervisor_async() -> dict[str, Any]:
    from src.models.config import Phase4Config
    from src.research.agents.supervisor import AgentSupervisor

    supervisor = AgentSupervisor(config=Phase4Config())

    async def fake_run_node(node_name: str, state: dict, inputs: dict | None = None) -> dict:
        if node_name == "search_plan":
            return {"search_plan": {"plan_goal": "search"}, "current_stage": node_name}
        if node_name == "search":
            return {"rag_result": {"paper_candidates": [{"title": "p1"}]}, "current_stage": node_name}
        if node_name == "extract":
            return {"paper_cards": [{"title": "p1"}], "current_stage": node_name}
        if node_name == "draft":
            return {"draft_report": {"sections": {"introduction": "ok"}}, "current_stage": node_name}
        if node_name == "review":
            return {"review_feedback": {"passed": True}, "review_passed": True, "current_stage": node_name}
        if node_name == "persist_artifacts":
            return {"result_markdown": "# ok", "current_stage": node_name}
        return {"brief": _sample_brief(), "current_stage": node_name}

    supervisor.run_node = fake_run_node  # type: ignore[method-assign]
    result = await supervisor.collaborate({"brief": _sample_brief()}, start_node="search_plan")
    return _ok(
        "supervisor",
        {
            "trace_refs": result["trace_refs"],
            "current_stage": result["current_stage"],
            "graph_nodes": sorted(supervisor.build_graph().get_graph().nodes),
        },
    )


def validate_supervisor() -> dict[str, Any]:
    return asyncio.run(_validate_supervisor_async())


async def _validate_handoff_supervisor_async() -> dict[str, Any]:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from src.models.config import Phase4Config, SupervisorMode
    from src.research.agents.supervisor import AgentSupervisor

    class ToolCallingFakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            return self

    supervisor = AgentSupervisor(config=Phase4Config(supervisor_mode=SupervisorMode.LLM_HANDOFF))

    async def fake_run_node(node_name: str, state: dict, inputs: dict | None = None) -> dict:
        return {
            "search_plan": {"plan_goal": "Collect recent RAG papers", "query_groups": []},
            "current_stage": node_name,
            "_backend_mode": "v2",
            "_agent_paradigm": "plan_and_execute",
        }

    supervisor.run_node = fake_run_node  # type: ignore[method-assign]
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "transfer_to_search_plan", "args": {}, "id": "call_search_plan"}],
            ),
            AIMessage(content="Search plan prepared."),
        ]
    )
    result = await supervisor.collaborate_with_handoff(
        {"brief": _sample_brief()},
        start_node="search_plan",
        stop_after="search_plan",
        user_request="Prepare a search plan.",
        model=model,
    )
    graph_nodes = sorted(
        supervisor.build_official_supervisor_graph(
            model=model,
            node_names=["search_plan"],
        )
        .get_graph()
        .nodes
    )
    return _ok(
        "handoff_supervisor",
        {
            "trace_refs": result["trace_refs"],
            "current_stage": result["current_stage"],
            "supervisor_mode": result["supervisor_mode"],
            "graph_nodes": graph_nodes,
        },
    )


def validate_handoff_supervisor() -> dict[str, Any]:
    return asyncio.run(_validate_handoff_supervisor_async())


VALIDATORS: dict[str, AgentValidator] = {
    "clarify": validate_clarify,
    "search_plan": validate_search_plan,
    "planner": validate_planner,
    "retriever": validate_retriever,
    "analyst": validate_analyst,
    "reviewer": validate_reviewer,
    "supervisor": validate_supervisor,
}


def main() -> None:
    load_dotenv(".env")
    parser = argparse.ArgumentParser(description="Run a deterministic smoke validation for one agent.")
    parser.add_argument("agent", choices=[*VALIDATORS.keys(), "all"])
    args = parser.parse_args()

    if args.agent == "all":
        payload = [validator() for validator in VALIDATORS.values()]
    else:
        payload = VALIDATORS[args.agent]()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
