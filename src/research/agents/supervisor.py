"""Official LangGraph supervisor wrapper for the research workflow.

This module keeps the public ``AgentSupervisor`` facade used by the API layer,
but its collaboration path now delegates orchestration to
``langgraph_supervisor.create_supervisor`` instead of a hand-written dispatch
graph. The canonical research execution order remains:

    clarify -> search_plan -> search -> extract -> draft -> review -> persist_artifacts
"""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Sequence
from typing import Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from src.agent.checkpointing import build_graph_config, get_langgraph_checkpointer
from src.agent.llm import build_reason_llm
from src.agent.output_workspace import write_node_output
from src.agent.settings import get_settings
from src.models.config import (
    AgentParadigm,
    ExecutionMode,
    NodeBackendMode,
    Phase4Config,
    SupervisorMode,
)

logger = logging.getLogger(__name__)


def _sync_node_to_workspace(
    task_id: str,
    workspace_id: str | None,
    node_name: str,
    result: dict[str, Any],
) -> None:
    """Write a node's output to the task workspace directory."""
    try:
        path = write_node_output(task_id, node_name, result, workspace_id=workspace_id)
        if path:
            logger.debug("[supervisor] synced %s -> %s", node_name, path.name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[supervisor] failed to sync node output to workspace: %s", exc)


def _append_review_revision(
    task_id: str,
    workflow_state: dict[str, Any],
    *,
    workspace_id: str | None = None,
) -> None:
    """Append the current draft as a revision when review fails."""
    try:
        from src.agent.output_workspace import append_revision

        draft_md = workflow_state.get("draft_markdown")
        draft_report = workflow_state.get("draft_report")
        review_passed = workflow_state.get("review_passed")

        revision_text: str | None = None
        if isinstance(draft_md, str) and draft_md:
            revision_text = draft_md
        elif isinstance(draft_report, dict):
            import json as _json

            revision_text = _json.dumps(draft_report, ensure_ascii=False, indent=2)

        if revision_text:
            label = "after_review" if review_passed is False else "revision"
            append_revision(task_id, revision_text, label=label, workspace_id=workspace_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[supervisor] failed to append review revision: %s", exc)


CANONICAL_NODE_ORDER: tuple[str, ...] = (
    "clarify",
    "search_plan",
    "search",
    "extract",
    "extract_compression",
    "draft",
    "review",
    "persist_artifacts",
)

LEGACY_NODE_TARGETS: dict[str, tuple[str, str]] = {
    "clarify": ("src.research.graph.nodes.clarify", "run_clarify_node"),
    "search_plan": ("src.research.graph.nodes.search_plan", "run_search_plan_node"),
    "search": ("src.research.graph.nodes.search", "search_node"),
    "extract": ("src.research.graph.nodes.extract", "extract_node"),
    "extract_compression": ("src.research.graph.nodes.extract_compression", "extract_compression_node"),
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
    "compression_result",
    "taxonomy",
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
    """Optional direct backend override used by direct single-node execution."""

    async def run(self, state: dict, inputs: dict) -> dict:
        raise NotImplementedError


class SupervisorGraphState(TypedDict, total=False):
    messages: list[BaseMessage]


def _tool_name(tool_spec: BaseTool | dict[str, Any] | Any) -> str | None:
    if isinstance(tool_spec, dict):
        function_spec = tool_spec.get("function")
        if isinstance(function_spec, dict) and function_spec.get("name"):
            return str(function_spec["name"])
        if tool_spec.get("name"):
            return str(tool_spec["name"])
        return None
    return getattr(tool_spec, "name", None)


class _BoundToolCallingModel(BaseChatModel):
    """Minimal deterministic model that calls exactly one bound tool once."""

    tool_name: str | None = None
    final_response: str = "Stage completed."

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # type: ignore[override]
        selected = self.tool_name
        if not selected:
            for tool_spec in tools:
                selected = _tool_name(tool_spec)
                if selected:
                    break
        return self.model_copy(update={"tool_name": selected})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        if any(
            isinstance(message, ToolMessage) and getattr(message, "name", None) == self.tool_name
            for message in messages
        ):
            response = AIMessage(content=self.final_response)
        else:
            if not self.tool_name:
                raise RuntimeError("No bound tool name configured for deterministic worker model.")
            response = AIMessage(
                content="",
                tool_calls=[{"name": self.tool_name, "args": {}, "id": f"call_{self.tool_name}"}],
            )
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "deterministic-single-tool"


class _PlannedHandoffModel(FakeMessagesListChatModel):
    """Deterministic supervisor model that emits a fixed handoff sequence."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # type: ignore[override]
        del tools, tool_choice, kwargs
        return self


class AgentSupervisor:
    """Research multi-agent supervisor backed by the official supervisor API."""

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
        return node_name

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
            except Exception as exc:  # noqa: BLE001
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
            return await backend.run(state, inputs or {})

        logger.info("[AgentSupervisor] legacy backend for %s", node_name)
        return await self._call_legacy_node(node_name, state, inputs)

    async def _run_v2(self, node_name: str, state: dict, inputs: dict | None) -> dict:
        backend = self._node_backends.get(node_name)
        if backend:
            return await backend.run(state, inputs or {})

        if not self._has_v2_backend(node_name):
            raise NotImplementedError(f"No v2 agent for: {node_name}")

        logger.info("[AgentSupervisor] v2 backend for %s", node_name)
        return await self._call_v2_agent(node_name, state, inputs)

    async def _call_legacy_node(self, node_name: str, state: dict, inputs: dict | None) -> dict:
        target = LEGACY_NODE_TARGETS.get(node_name)
        if not target:
            raise NotImplementedError(f"No legacy node for: {node_name}")

        module = importlib.import_module(target[0])
        fn = getattr(module, target[1], None)
        if not fn:
            raise NotImplementedError(f"No function {target[1]} in {target[0]}")
        execution_state = dict(state)
        event_emitter = (inputs or {}).get("_event_emitter")
        if event_emitter is not None:
            execution_state["_event_emitter"] = event_emitter
        result = fn(execution_state)
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
        if not state.get("compression_result"):
            return "extract_compression"
        if not state.get("draft_report") and not state.get("draft_markdown"):
            return "draft"
        if state.get("review_feedback") is None:
            return "review"
        if state.get("review_passed") and not state.get("result_markdown"):
            return "persist_artifacts"
        return "review"

    def _skip_reason(self, node_name: str, workflow_state: dict[str, Any]) -> str | None:
        if node_name == "clarify":
            return None
        if node_name == "search_plan":
            if not workflow_state.get("brief"):
                return "brief is missing"
            if workflow_state.get("awaiting_followup"):
                return "clarify requested interactive follow-up"
            return None
        if node_name == "search":
            if workflow_state.get("awaiting_followup"):
                return "clarify requested interactive follow-up"
            if not workflow_state.get("search_plan"):
                return "search_plan is missing"
            if workflow_state.get("research_depth", "full") != "full":
                return "research_depth is plan-only"
            return None
        if node_name == "extract":
            if not workflow_state.get("rag_result"):
                return "rag_result is missing"
            return None
        if node_name == "extract_compression":
            if not workflow_state.get("paper_cards"):
                return "paper_cards are missing"
            return None
        if node_name == "draft":
            if not workflow_state.get("paper_cards"):
                return "paper_cards are missing"
            return None
        if node_name == "review":
            if not workflow_state.get("draft_report") and not workflow_state.get("draft_markdown"):
                return "draft output is missing"
            return None
        if node_name == "persist_artifacts":
            if workflow_state.get("review_passed") is not True:
                return "review did not pass"
            return None
        return None

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

    def _planned_nodes(self, start_node: str, stop_after: str | None = None) -> tuple[str, ...]:
        canonical_start = self.normalize_node_name(start_node)
        if canonical_start not in CANONICAL_NODE_ORDER:
            canonical_start = self._infer_start_node({})

        planned = list(CANONICAL_NODE_ORDER[CANONICAL_NODE_ORDER.index(canonical_start):])
        if stop_after:
            canonical_stop = self.normalize_node_name(stop_after)
            if canonical_stop in planned:
                planned = planned[: planned.index(canonical_stop) + 1]
        return tuple(planned)

    def _build_collaboration_request(
        self,
        workflow_state: dict[str, Any],
        *,
        node_names: Sequence[str],
        user_request: str | None = None,
    ) -> str:
        topic = ""
        brief = workflow_state.get("brief")
        if isinstance(brief, dict):
            topic = str(brief.get("topic") or brief.get("research_topic") or "").strip()
        if not topic:
            topic = str(workflow_state.get("raw_input") or "").strip()
        if not topic:
            topic = "research workflow task"

        request = user_request or f"Execute the research workflow for: {topic}"
        stages = " -> ".join(node_names) if node_names else "(none)"
        return (
            f"{request}\n\n"
            f"Allowed stages for this run: {stages}.\n"
            "Run the next unfinished stage only once, preserve prior work in state, "
            "and stop after the last allowed stage completes."
        )

    def _build_supervisor_result(
        self,
        workflow_state: dict[str, Any],
        collaboration_trace: list[dict[str, Any]],
        *,
        canonical_start: str,
        supervisor_mode: SupervisorMode,
    ) -> dict[str, Any]:
        summary = self._summarize_trace(canonical_start, collaboration_trace)
        return {
            "current_stage": workflow_state.get(
                "current_stage",
                collaboration_trace[-1]["node"] if collaboration_trace else canonical_start,
            ),
            "trace_refs": [entry["node"] for entry in collaboration_trace],
            "collaboration_trace": collaboration_trace,
            "summary": summary,
            "supervisor_mode": supervisor_mode.value,
            **{key: workflow_state.get(key) for key in RESULT_STATE_KEYS if key in workflow_state},
        }

    def _build_node_tool(
        self,
        node_name: str,
        workflow_state: dict[str, Any],
        collaboration_trace: list[dict[str, Any]],
        executed_nodes: list[str],
        inputs: dict | None,
    ) -> BaseTool:
        tool_name = f"run_{node_name}_stage"

        @tool(tool_name, description=f"Execute the {node_name} stage of the research workflow.")
        async def _run_stage() -> str:
            event_emitter = (inputs or {}).get("_event_emitter")
            started = time.monotonic()
            if node_name in executed_nodes:
                return f"{node_name} stage already completed in this supervisor run."

            skip_reason = self._skip_reason(node_name, workflow_state)
            if skip_reason:
                executed_nodes.append(node_name)
                if event_emitter:
                    event_emitter.on_node_start(node_name)
                    event_emitter.on_thinking(node_name, f"Skipping stage: {skip_reason}.")
                    event_emitter.on_node_end(node_name, status="skipped", duration_ms=0)
                return f"Skipping {node_name}: {skip_reason}."

            if event_emitter:
                event_emitter.on_node_start(node_name)

            try:
                result = await self.run_node(node_name, workflow_state, inputs)
            except Exception as exc:
                if event_emitter:
                    event_emitter.on_node_end(
                        node_name,
                        status="failed",
                        duration_ms=int((time.monotonic() - started) * 1000),
                        error=str(exc),
                    )
                raise

            self._merge_state(workflow_state, result)

            trace_entry = self._build_trace_entry(node_name, result)
            collaboration_trace.append(trace_entry)
            executed_nodes.append(node_name)

            task_id = workflow_state.get("task_id")
            workspace_id = workflow_state.get("workspace_id")
            if task_id:
                _sync_node_to_workspace(str(task_id), str(workspace_id) if workspace_id else None, node_name, result)

            if node_name == "review" and task_id and not workflow_state.get("review_passed"):
                _append_review_revision(
                    str(task_id),
                    workflow_state,
                    workspace_id=str(workspace_id) if workspace_id else None,
                )

            if event_emitter:
                warning_fields = (
                    "warnings",
                    "search_plan_warnings",
                    "retriever_warnings",
                    "analyst_warnings",
                    "reviewer_warnings",
                )
                warnings: list[str] = []
                for field in warning_fields:
                    value = result.get(field)
                    if isinstance(value, list):
                        warnings.extend(str(item) for item in value if item)
                event_emitter.on_node_end(
                    node_name,
                    duration_ms=int((time.monotonic() - started) * 1000),
                    warnings=warnings,
                )

            produced = ", ".join(trace_entry["produced"]) or "no new fields"
            return f"{node_name} completed with backend={trace_entry['backend']}; produced {produced}."

        return _run_stage

    def _build_node_agent(
        self,
        node_name: str,
        workflow_state: dict[str, Any],
        collaboration_trace: list[dict[str, Any]],
        executed_nodes: list[str],
        inputs: dict | None,
    ):
        stage_tool = self._build_node_tool(node_name, workflow_state, collaboration_trace, executed_nodes, inputs)
        return create_react_agent(
            model=_BoundToolCallingModel(final_response=f"__completed_stage__:{node_name}"),
            tools=[stage_tool],
            name=node_name,
            prompt=(
                f"You own only the {node_name} stage of the research workflow. "
                "Always call your single tool exactly once, then stop."
            ),
        )

    def build_official_supervisor_graph(
        self,
        *,
        model: BaseChatModel | None = None,
        node_names: Sequence[str] | None = None,
        workflow_state: dict[str, Any] | None = None,
        collaboration_trace: list[dict[str, Any]] | None = None,
        executed_nodes: list[str] | None = None,
        inputs: dict | None = None,
    ):
        planned_nodes = tuple(node_names or CANONICAL_NODE_ORDER)
        active_state = workflow_state if workflow_state is not None else {}
        trace = collaboration_trace if collaboration_trace is not None else []
        completed = executed_nodes if executed_nodes is not None else []

        worker_agents = [
            self._build_node_agent(node_name, active_state, trace, completed, inputs)
            for node_name in planned_nodes
        ]

        if model is None:
            responses = [
                AIMessage(
                    content="",
                    tool_calls=[{"name": f"transfer_to_{node_name}", "args": {}, "id": f"call_transfer_to_{node_name}"}],
                )
                for node_name in planned_nodes
            ]
            responses.append(AIMessage(content="Research workflow completed."))
            supervisor_model = _PlannedHandoffModel(responses=responses)
        else:
            supervisor_model = model

        workflow = create_supervisor(
            worker_agents,
            model=supervisor_model,
            prompt=(
                "You supervise the research workflow. Hand off only to the next allowed "
                "stage, wait for the stage to finish, and stop when all allowed stages are done."
            ),
            output_mode="last_message",
            parallel_tool_calls=False,
        )
        return workflow.compile(checkpointer=get_langgraph_checkpointer("agent_supervisor"))

    def build_graph(self):
        """Compatibility helper returning the official compiled supervisor graph."""
        return self.build_official_supervisor_graph()

    async def collaborate(
        self,
        state: dict,
        *,
        start_node: str | None = None,
        stop_after: str | None = None,
        inputs: dict | None = None,
    ) -> dict:
        """Run the canonical workflow through the official supervisor API."""
        canonical_start = self.normalize_node_name(start_node) if start_node else self._infer_start_node(state)
        if canonical_start not in CANONICAL_NODE_ORDER:
            canonical_start = self._infer_start_node(state)

        workflow_state = dict(state)
        collaboration_trace: list[dict[str, Any]] = []
        executed_nodes: list[str] = []
        planned_nodes = self._planned_nodes(canonical_start, stop_after)

        graph = self.build_official_supervisor_graph(
            node_names=planned_nodes,
            workflow_state=workflow_state,
            collaboration_trace=collaboration_trace,
            executed_nodes=executed_nodes,
            inputs=inputs,
        )
        await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=self._build_collaboration_request(
                            workflow_state,
                            node_names=planned_nodes,
                        )
                    )
                ]
            },
            config=build_graph_config(
                "agent_supervisor",
                recursion_limit=max(len(planned_nodes) * 6 + 12, 24),
            ),
        )
        return self._build_supervisor_result(
            workflow_state,
            collaboration_trace,
            canonical_start=canonical_start,
            supervisor_mode=SupervisorMode.GRAPH,
        )

    async def collaborate_with_handoff(
        self,
        state: dict,
        *,
        start_node: str | None = None,
        stop_after: str | None = None,
        inputs: dict | None = None,
        user_request: str | None = None,
        model: BaseChatModel | None = None,
    ) -> dict:
        """Run the official LLM-routed supervisor graph."""
        canonical_start = self.normalize_node_name(start_node) if start_node else self._infer_start_node(state)
        if canonical_start not in CANONICAL_NODE_ORDER:
            canonical_start = self._infer_start_node(state)

        workflow_state = dict(state)
        collaboration_trace: list[dict[str, Any]] = []
        executed_nodes: list[str] = []
        planned_nodes = self._planned_nodes(canonical_start, stop_after)

        supervisor_model = model
        if supervisor_model is None:
            supervisor_model = build_reason_llm(get_settings(), max_tokens=4096)

        graph = self.build_official_supervisor_graph(
            model=supervisor_model,
            node_names=planned_nodes,
            workflow_state=workflow_state,
            collaboration_trace=collaboration_trace,
            executed_nodes=executed_nodes,
            inputs=inputs,
        )
        await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=self._build_collaboration_request(
                            workflow_state,
                            node_names=planned_nodes,
                            user_request=user_request,
                        )
                    )
                ]
            },
            config=build_graph_config(
                "agent_supervisor",
                recursion_limit=max(len(planned_nodes) * 8 + 16, 32),
            ),
        )
        return self._build_supervisor_result(
            workflow_state,
            collaboration_trace,
            canonical_start=canonical_start,
            supervisor_mode=SupervisorMode.LLM_HANDOFF,
        )

    def _prune_state_for_stage(self, state: dict, target_stage: str) -> dict:
        canonical = self.normalize_node_name(target_stage)
        pruned = dict(state)
        downstream_keys: dict[str, tuple[str, ...]] = {
            "clarify": (
                "brief",
                "search_plan",
                "rag_result",
                "paper_cards",
                "compression_result",
                "taxonomy",
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
                "compression_result",
                "taxonomy",
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
                "compression_result",
                "taxonomy",
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
                "compression_result",
                "taxonomy",
                "draft_report",
                "draft_markdown",
                "review_feedback",
                "review_passed",
                "result_markdown",
                "resolved_report",
                "verified_report",
                "final_report",
            ),
            "extract_compression": (
                "compression_result",
                "taxonomy",
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
        """Re-run the workflow from a chosen stage using the official supervisor."""
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
