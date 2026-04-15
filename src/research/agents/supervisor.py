"""AgentSupervisor — Phase 4: multi-agent orchestration on top of the research graph.

The current research workflow truth lives in the canonical 7-node graph:

    clarify -> search_plan -> search -> extract -> draft -> review -> persist_artifacts

This supervisor provides:
- normalizes old Phase 4 node aliases to the current graph node names
- resumes from the right stage based on the current shared task state
- records lightweight collaboration traces so agent execution is inspectable
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent.checkpointing import build_graph_config, get_langgraph_checkpointer
from src.models.config import (
    AgentParadigm,
    ExecutionMode,
    NodeBackendMode,
    Phase4Config,
    SupervisorMode,
)

logger = logging.getLogger(__name__)


CANONICAL_NODE_ORDER: tuple[str, ...] = (
    "clarify",
    "search_plan",
    "search",
    "extract",
    "draft",
    "review",
    "persist_artifacts",
)

LEGACY_NODE_ALIASES: dict[str, str] = {
    "plan_search": "search_plan",
    "search_corpus": "search",
    "extract_cards": "extract",
    "synthesize": "draft",
    "write_report": "draft",
    "revise": "review",
}

LEGACY_NODE_TARGETS: dict[str, tuple[str, str]] = {
    "clarify": ("src.research.graph.nodes.clarify", "run_clarify_node"),
    "search_plan": ("src.research.graph.nodes.search_plan", "run_search_plan_node"),
    "search": ("src.research.graph.nodes.search", "search_node"),
    "extract": ("src.research.graph.nodes.extract", "extract_node"),
    "draft": ("src.research.graph.nodes.draft", "draft_node"),
    "review": ("src.research.graph.nodes.review", "review_node"),
    "persist_artifacts": ("src.research.graph.nodes.persist_artifacts", "persist_artifacts_node"),
}

V2_AGENT_TARGETS: dict[str, dict[str, str]] = {
    "search_plan": {
        "module": "src.research.agents.planner_agent",
        "fn": "run_planner_agent",
        "paradigm": AgentParadigm.PLAN_AND_EXECUTE.value,
    },
    "search": {
        "module": "src.research.agents.retriever_agent",
        "fn": "run_retriever_agent",
        "paradigm": AgentParadigm.TAG.value,
    },
    "draft": {
        "module": "src.research.agents.analyst_agent",
        "fn": "run_analyst_agent",
        "paradigm": AgentParadigm.REASONING_VIA_ARTIFACTS.value,
    },
    "review": {
        "module": "src.research.agents.reviewer_agent",
        "fn": "run_reviewer_agent",
        "paradigm": AgentParadigm.REFLEXION.value,
    },
}

RESULT_STATE_KEYS: tuple[str, ...] = (
    "brief",
    "search_plan",
    "rag_result",
    "paper_cards",
    "draft_report",
    "draft_markdown",
    "review_feedback",
    "review_passed",
    "result_markdown",
    "resolved_report",
    "verified_report",
    "final_report",
)


class NodeBackend:
    """节点后端接口（Protocol），legacy 和 v2 都实现此接口。"""

    async def run(self, state: dict, inputs: dict) -> dict:
        raise NotImplementedError


class SupervisorGraphState(TypedDict, total=False):
    workflow_state: dict[str, Any]
    canonical_start: str
    stop_at: str | None
    collaboration_trace: list[dict[str, Any]]
    last_node: str | None
    supervisor_result: dict[str, Any]


class AgentSupervisor:
    """Research multi-agent supervisor with aliasing, resume, and handoff tracing."""

    def __init__(self, config: Phase4Config | None = None):
        self.config = config or Phase4Config()
        self._node_backends: dict[str, NodeBackend] = {}

    def set_config(self, config: Phase4Config) -> None:
        self.config = config

    def register_backend(self, node_name: str, backend: NodeBackend) -> None:
        self._node_backends[self.normalize_node_name(node_name)] = backend

    def normalize_node_name(self, node_name: str | None) -> str:
        if not node_name:
            return "search_plan"
        return LEGACY_NODE_ALIASES.get(node_name, node_name)

    def _has_v2_backend(self, node_name: str) -> bool:
        return node_name in V2_AGENT_TARGETS

    def _get_backend_mode(self, node_name: str) -> NodeBackendMode:
        canonical = self.normalize_node_name(node_name)
        node_mode = self.config.node_backends.mode_for(canonical)
        if node_mode != NodeBackendMode.AUTO:
            return node_mode

        execution_mode = self.config.execution_mode
        if execution_mode == ExecutionMode.LEGACY:
            return NodeBackendMode.LEGACY
        if execution_mode == ExecutionMode.V2 and self._has_v2_backend(canonical):
            return NodeBackendMode.V2
        if execution_mode == ExecutionMode.V2:
            return NodeBackendMode.LEGACY
        return NodeBackendMode.AUTO if self._has_v2_backend(canonical) else NodeBackendMode.LEGACY

    async def run_node(
        self,
        node_name: str,
        state: dict,
        inputs: dict | None = None,
    ) -> dict:
        """Run a canonical node with legacy/v2 backend selection and trace metadata."""
        canonical = self.normalize_node_name(node_name)
        payload = inputs or {}
        backend_mode = self._get_backend_mode(canonical)

        if backend_mode == NodeBackendMode.LEGACY:
            result = await self._run_legacy(canonical, state, payload)
        elif backend_mode == NodeBackendMode.V2:
            if not self._has_v2_backend(canonical):
                logger.warning(
                    "[AgentSupervisor] node %s requested v2 backend, but no v2 implementation exists; using legacy",
                    canonical,
                )
                result = await self._run_legacy(canonical, state, payload)
                backend_mode = NodeBackendMode.LEGACY
            else:
                result = await self._run_v2(canonical, state, payload)
        else:
            try:
                result = await self._run_v2(canonical, state, payload)
                backend_mode = NodeBackendMode.V2
            except Exception as exc:
                logger.warning(
                    "[AgentSupervisor] v2 backend failed for %s: %s; falling back to legacy",
                    canonical,
                    exc,
                )
                result = await self._run_legacy(canonical, state, payload)
                backend_mode = NodeBackendMode.LEGACY

        if not isinstance(result, dict):
            result = {"result": result}

        result.setdefault("current_stage", canonical)
        result["_normalized_node"] = canonical
        result["_backend_mode"] = backend_mode.value
        result.setdefault("_agent_paradigm", "legacy" if backend_mode == NodeBackendMode.LEGACY else "unknown")
        return result

    async def _run_legacy(self, node_name: str, state: dict, inputs: dict | None) -> dict:
        backend = self._node_backends.get(node_name)
        if backend:
            return await backend.run(state, inputs)

        logger.info("[AgentSupervisor] legacy backend for %s", node_name)
        return await self._call_legacy_node(node_name, state)

    async def _run_v2(self, node_name: str, state: dict, inputs: dict | None) -> dict:
        backend = self._node_backends.get(node_name)
        if backend:
            return await backend.run(state, inputs)

        if not self._has_v2_backend(node_name):
            raise NotImplementedError(f"No v2 agent for: {node_name}")

        logger.info("[AgentSupervisor] v2 backend for %s", node_name)
        return await self._call_v2_agent(node_name, state, inputs)

    async def _call_legacy_node(self, node_name: str, state: dict) -> dict:
        target = LEGACY_NODE_TARGETS.get(node_name)
        if not target:
            raise NotImplementedError(f"No legacy node for: {node_name}")

        module = importlib.import_module(target[0])
        fn = getattr(module, target[1], None)
        if not fn:
            raise NotImplementedError(f"No function {target[1]} in {target[0]}")
        result = fn(state)
        return result if isinstance(result, dict) else {"result": result}

    async def _call_v2_agent(self, node_name: str, state: dict, inputs: dict | None) -> dict:
        info = V2_AGENT_TARGETS.get(node_name)
        if not info:
            raise NotImplementedError(f"No v2 agent for: {node_name}")

        module = importlib.import_module(info["module"])
        fn = getattr(module, info["fn"], None)
        if not fn:
            raise NotImplementedError(f"No function {info['fn']} in {info['module']}")

        result = fn(state, inputs or {})
        if isinstance(result, dict):
            result["_agent_paradigm"] = info["paradigm"]
        else:
            result = {"result": result, "_agent_paradigm": info["paradigm"]}
        logger.info("[AgentSupervisor] v2 agent %s paradigm=%s", node_name, info["paradigm"])
        return result

    def _infer_start_node(self, state: dict) -> str:
        if not state.get("brief"):
            return "clarify"
        brief = state.get("brief") or {}
        if isinstance(brief, dict) and brief.get("needs_followup"):
            return "clarify"
        if not state.get("search_plan"):
            return "search_plan"
        if not state.get("rag_result"):
            return "search"
        if not state.get("paper_cards"):
            return "extract"
        if not state.get("draft_report") and not state.get("draft_markdown"):
            return "draft"
        if state.get("review_feedback") is None:
            return "review"
        if state.get("review_passed") and not state.get("result_markdown"):
            return "persist_artifacts"
        return "review"

    def _merge_state(self, state: dict, result: dict) -> None:
        for key, value in result.items():
            if key.startswith("_"):
                continue
            state[key] = value

    def _collect_produced_fields(self, result: dict) -> list[str]:
        produced: list[str] = []
        for key, value in result.items():
            if key.startswith("_") or key in {"summary", "warnings", "current_stage"}:
                continue
            if value is None:
                continue
            if isinstance(value, (list, dict)) and not value:
                continue
            produced.append(key)
        return produced

    def _build_trace_entry(self, node_name: str, result: dict) -> dict[str, Any]:
        return {
            "node": node_name,
            "backend": result.get("_backend_mode", "legacy"),
            "paradigm": result.get("_agent_paradigm", "legacy"),
            "produced": self._collect_produced_fields(result),
            "warnings": list(result.get("warnings", []) or []),
        }

    def _should_stop_after(self, node_name: str, state: dict) -> bool:
        if node_name == "clarify":
            brief = state.get("brief") or {}
            return not brief or (isinstance(brief, dict) and brief.get("needs_followup"))
        if node_name == "review":
            return not bool(state.get("review_passed"))
        return False

    def _summarize_trace(self, start_node: str, trace: list[dict[str, Any]]) -> str:
        if not trace:
            return "Supervisor did not execute any agent nodes."
        node_path = " -> ".join(entry["node"] for entry in trace)
        backend_path = ", ".join(
            f"{entry['node']}[{entry['backend']}/{entry['paradigm']}]"
            for entry in trace
        )
        return (
            f"Supervisor resumed from {start_node} and coordinated {node_path}. "
            f"Backends: {backend_path}."
        )

    def build_graph(self):
        """Build the LangGraph supervisor workflow around canonical research nodes."""
        workflow = StateGraph(SupervisorGraphState)
        workflow.add_node("prepare", self._prepare_collaboration_node)
        for node_name in CANONICAL_NODE_ORDER:
            workflow.add_node(node_name, self._make_collaboration_node(node_name))
        workflow.add_node("finalize", self._finalize_collaboration_node)

        workflow.add_edge(START, "prepare")
        workflow.add_conditional_edges(
            "prepare",
            self._route_from_prepare,
            {**{name: name for name in CANONICAL_NODE_ORDER}, "finalize": "finalize"},
        )
        for node_name in CANONICAL_NODE_ORDER:
            workflow.add_conditional_edges(
                node_name,
                self._make_route_after_node(node_name),
                {**{name: name for name in CANONICAL_NODE_ORDER}, "finalize": "finalize"},
            )
        workflow.add_edge("finalize", END)
        return workflow.compile(checkpointer=get_langgraph_checkpointer("agent_supervisor"))

    async def collaborate(
        self,
        state: dict,
        *,
        start_node: str | None = None,
        stop_after: str | None = None,
        inputs: dict | None = None,
    ) -> dict:
        """Run the shared workflow sequentially, resuming from the right point."""
        canonical_start = self.normalize_node_name(start_node) if start_node else self._infer_start_node(state)
        if canonical_start not in CANONICAL_NODE_ORDER:
            canonical_start = self._infer_start_node(state)
        workflow = CANONICAL_NODE_ORDER[CANONICAL_NODE_ORDER.index(canonical_start):]
        stop_at = self.normalize_node_name(stop_after) if stop_after else None
        if stop_at and stop_at not in CANONICAL_NODE_ORDER:
            stop_at = None

        result = await self.build_graph().ainvoke(
            {
                "workflow_state": dict(state),
                "canonical_start": canonical_start,
                "stop_at": stop_at,
                "collaboration_trace": [],
                "last_node": None,
            },
            config=build_graph_config(
                "agent_supervisor",
                recursion_limit=len(workflow) * 3 + 8,
            ),
        )
        return result.get("supervisor_result") or {}

    def _prepare_collaboration_node(self, state: SupervisorGraphState) -> dict[str, Any]:
        workflow_state = dict(state.get("workflow_state") or {})
        return {
            "workflow_state": workflow_state,
            "collaboration_trace": list(state.get("collaboration_trace", [])),
        }

    def _route_from_prepare(self, state: SupervisorGraphState) -> str:
        canonical_start = state.get("canonical_start") or "search_plan"
        return canonical_start if canonical_start in CANONICAL_NODE_ORDER else "finalize"

    def _make_collaboration_node(self, node_name: str):
        async def _node(state: SupervisorGraphState) -> dict[str, Any]:
            workflow_state = dict(state.get("workflow_state") or {})
            result = await self.run_node(node_name, workflow_state, None)
            self._merge_state(workflow_state, result)
            trace = list(state.get("collaboration_trace", []))
            trace.append(self._build_trace_entry(node_name, result))
            return {
                "workflow_state": workflow_state,
                "collaboration_trace": trace,
                "last_node": node_name,
            }

        return _node

    def _make_route_after_node(self, node_name: str):
        def _route(state: SupervisorGraphState) -> str:
            stop_at = state.get("stop_at")
            workflow_state = state.get("workflow_state") or {}
            if stop_at and node_name == stop_at:
                return "finalize"
            if self._should_stop_after(node_name, workflow_state):
                return "finalize"
            current_idx = CANONICAL_NODE_ORDER.index(node_name)
            next_idx = current_idx + 1
            if next_idx >= len(CANONICAL_NODE_ORDER):
                return "finalize"
            return CANONICAL_NODE_ORDER[next_idx]

        return _route

    def _finalize_collaboration_node(self, state: SupervisorGraphState) -> dict[str, Any]:
        workflow_state = dict(state.get("workflow_state") or {})
        collaboration_trace = list(state.get("collaboration_trace", []))
        canonical_start = state.get("canonical_start") or "search_plan"
        summary = self._summarize_trace(canonical_start, collaboration_trace)
        supervisor_result = {
            "current_stage": workflow_state.get(
                "current_stage",
                collaboration_trace[-1]["node"] if collaboration_trace else canonical_start,
            ),
            "trace_refs": [entry["node"] for entry in collaboration_trace],
            "collaboration_trace": collaboration_trace,
            "summary": summary,
            "supervisor_mode": SupervisorMode.GRAPH.value,
            **{key: workflow_state.get(key) for key in RESULT_STATE_KEYS if key in workflow_state},
        }
        return {"supervisor_result": supervisor_result}


    def _prune_state_for_stage(self, state: dict, target_stage: str) -> dict:
        canonical = self.normalize_node_name(target_stage)
        pruned = dict(state)
        downstream_keys: dict[str, tuple[str, ...]] = {
            "clarify": (
                "brief",
                "search_plan",
                "rag_result",
                "paper_cards",
                "draft_report",
                "draft_markdown",
                "review_feedback",
                "review_passed",
                "result_markdown",
                "resolved_report",
                "verified_report",
                "final_report",
            ),
            "search_plan": (
                "search_plan",
                "rag_result",
                "paper_cards",
                "draft_report",
                "draft_markdown",
                "review_feedback",
                "review_passed",
                "result_markdown",
                "resolved_report",
                "verified_report",
                "final_report",
            ),
            "search": (
                "rag_result",
                "paper_cards",
                "draft_report",
                "draft_markdown",
                "review_feedback",
                "review_passed",
                "result_markdown",
                "resolved_report",
                "verified_report",
                "final_report",
            ),
            "extract": (
                "paper_cards",
                "draft_report",
                "draft_markdown",
                "review_feedback",
                "review_passed",
                "result_markdown",
                "resolved_report",
                "verified_report",
                "final_report",
            ),
            "draft": (
                "draft_report",
                "draft_markdown",
                "review_feedback",
                "review_passed",
                "result_markdown",
                "resolved_report",
                "verified_report",
                "final_report",
            ),
            "review": (
                "review_feedback",
                "review_passed",
                "result_markdown",
                "resolved_report",
                "verified_report",
                "final_report",
            ),
            "persist_artifacts": tuple(),
        }
        for key in downstream_keys.get(canonical, ()):
            pruned.pop(key, None)
        return pruned

    async def replan(
        self,
        state: dict,
        trigger_reason: str,
        target_stage: str = "search_plan",
    ) -> dict:
        """Re-run the shared workflow from a canonical or legacy stage alias."""
        canonical = self.normalize_node_name(target_stage)
        if canonical not in CANONICAL_NODE_ORDER:
            canonical = "search_plan"
        logger.info(
            "[AgentSupervisor] re-plan triggered: stage=%s reason=%s",
            canonical,
            trigger_reason,
        )
        new_state = self._prune_state_for_stage(state, canonical)
        payload = {"replan": True, "reason": trigger_reason}
        return await self.collaborate(new_state, start_node=canonical, inputs=payload)


_supervisor: AgentSupervisor | None = None


def get_supervisor() -> AgentSupervisor:
    global _supervisor
    if _supervisor is None:
        _supervisor = AgentSupervisor()
    return _supervisor
