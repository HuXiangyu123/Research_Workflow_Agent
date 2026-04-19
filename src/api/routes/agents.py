"""Agents API — Phase 4: list agents, run a single agent, or re-plan collaboratively."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from src.models.agent import (
    AgentDescriptor,
    AgentRole,
    AgentRunRequest,
    AgentRunResponse,
    ReplanRequest,
    ReplanResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])


ROLE_DEFAULT_NODES: dict[AgentRole, str] = {
    AgentRole.PLANNER: "search_plan",
    AgentRole.RETRIEVER: "search",
    AgentRole.ANALYST: "draft",
    AgentRole.REVIEWER: "review",
}


def _builtin_agents() -> list[AgentDescriptor]:
    return [
        AgentDescriptor(
            agent_id="supervisor",
            role=AgentRole.SUPERVISOR,
            title="Supervisor",
            description="Resumes the shared workflow and coordinates planner, retriever, analyst, and reviewer backends.",
            supported_nodes=["clarify", "search_plan", "search", "extract", "draft", "review", "persist_artifacts"],
        ),
        AgentDescriptor(
            agent_id="planner",
            role=AgentRole.PLANNER,
            title="Planner Agent",
            description="Builds or refreshes the SearchPlan from the current ResearchBrief.",
            supported_nodes=["search_plan"],
            supported_skills=["creative_reframe"],
        ),
        AgentDescriptor(
            agent_id="retriever",
            role=AgentRole.RETRIEVER,
            title="Retriever Agent",
            description="Executes paper search over external sources and local corpus when available.",
            supported_nodes=["search"],
            supported_skills=["research_lit_scan"],
        ),
        AgentDescriptor(
            agent_id="analyst",
            role=AgentRole.ANALYST,
            title="Analyst Agent",
            description="Turns extracted paper cards into structured artifacts and a usable survey draft.",
            supported_nodes=["draft"],
            supported_skills=["paper_plan_builder"],
        ),
        AgentDescriptor(
            agent_id="reviewer",
            role=AgentRole.REVIEWER,
            title="Reviewer Agent",
            description="Checks coverage, support, and citation quality before persistence.",
            supported_nodes=["review"],
        ),
    ]


class ListAgentsResponse(BaseModel):
    items: list[AgentDescriptor]


def _build_agent_state(
    *,
    task_id: str | None,
    requested_workspace_id: str,
    inputs: dict[str, Any] | None,
) -> dict[str, Any]:
    from src.api.routes.tasks import _get_task_record

    task = _get_task_record(task_id) if task_id else None
    workspace_id = (task.workspace_id if task and task.workspace_id else requested_workspace_id) or requested_workspace_id

    state = {
        "workspace_id": workspace_id,
        "task_id": task_id or "",
        "source_type": getattr(task, "source_type", "research") if task else "research",
        "report_mode": getattr(task, "report_mode", "draft") if task else "draft",
        "interaction_mode": "interactive",
    }

    if task:
        for field_name in (
            "brief",
            "search_plan",
            "rag_result",
            "paper_cards",
            "draft_report",
            "draft_markdown",
            "review_feedback",
            "review_passed",
            "current_stage",
            "result_markdown",
            "full_markdown",
        ):
            value = getattr(task, field_name, None)
            if value is not None:
                state[field_name] = value

    if inputs:
        state.update(inputs)

    return state


@router.get("", response_model=ListAgentsResponse)
async def list_agents() -> ListAgentsResponse:
    """返回所有注册的 agent 角色描述。"""
    return ListAgentsResponse(items=_builtin_agents())


@router.post("/run", response_model=AgentRunResponse)
async def run_agent(req: AgentRunRequest) -> AgentRunResponse:
    """Run a specialized agent or let the supervisor resume the shared workflow."""
    from src.research.agents.supervisor import get_supervisor
    from src.skills.registry import get_skills_registry

    resolved_role = req.role or AgentRole.SUPERVISOR
    state = _build_agent_state(
        task_id=req.task_id,
        requested_workspace_id=req.workspace_id,
        inputs=req.inputs,
    )
    resolved_workspace_id = str(state.get("workspace_id", req.workspace_id))

    if req.preferred_skill_id:
        registry = get_skills_registry()
        manifest = registry.get(req.preferred_skill_id)
        if manifest:
            from src.models.skills import SkillRunRequest

            skill_req = SkillRunRequest(
                workspace_id=resolved_workspace_id,
                task_id=req.task_id,
                skill_id=req.preferred_skill_id,
                inputs=req.inputs,
                preferred_agent=req.role,
            )
            try:
                skill_resp = await registry.run(skill_req, {"workspace_id": resolved_workspace_id})
                return AgentRunResponse(
                    workspace_id=resolved_workspace_id,
                    task_id=req.task_id,
                    role=resolved_role,
                    selected_skill_id=req.preferred_skill_id,
                    output_artifact_ids=skill_resp.output_artifact_ids,
                    summary=skill_resp.summary,
                )
            except Exception as exc:
                logger.warning("Skill run failed: %s", exc)

    supervisor = get_supervisor()
    node_name = req.node_name or ROLE_DEFAULT_NODES.get(resolved_role)

    try:
        if resolved_role == AgentRole.SUPERVISOR and not req.node_name:
            from src.models.config import SupervisorMode

            requested_mode = req.inputs.get("supervisor_mode") if isinstance(req.inputs, dict) else None
            configured_mode = getattr(getattr(supervisor, "config", None), "supervisor_mode", None)
            if requested_mode == SupervisorMode.LLM_HANDOFF.value or configured_mode == SupervisorMode.LLM_HANDOFF:
                result = await supervisor.collaborate_with_handoff(
                    state=state,
                    start_node=req.inputs.get("start_node") if isinstance(req.inputs, dict) else None,
                    stop_after=req.inputs.get("stop_after") if isinstance(req.inputs, dict) else None,
                    inputs=req.inputs,
                    user_request=(
                        req.inputs.get("user_request")
                        or req.inputs.get("message")
                        or req.inputs.get("query")
                    )
                    if isinstance(req.inputs, dict)
                    else None,
                )
            else:
                result = await supervisor.collaborate(
                    state=state,
                    start_node=req.inputs.get("start_node") if isinstance(req.inputs, dict) else None,
                    stop_after=req.inputs.get("stop_after") if isinstance(req.inputs, dict) else None,
                    inputs=req.inputs,
                )
        else:
            result = await supervisor.run_node(node_name or "search_plan", state, req.inputs)

        paradigm = result.get("_agent_paradigm", "collaboration")
        trace_refs = result.get("trace_refs", [])
        collaboration_trace = result.get("collaboration_trace", [])
        if not trace_refs and collaboration_trace:
            trace_refs = [entry.get("node", "") for entry in collaboration_trace if entry.get("node")]

        return AgentRunResponse(
            workspace_id=resolved_workspace_id,
            task_id=req.task_id or "",
            role=resolved_role,
            output_artifact_ids=result.get("output_artifact_ids", []),
            trace_refs=trace_refs,
            collaboration_trace=collaboration_trace,
            summary=result.get(
                "summary",
                f"{(node_name or 'supervisor')} ({paradigm}) completed",
            ),
        )
    except Exception as exc:
        logger.exception("Agent run failed: %s", exc)
        return AgentRunResponse(
            workspace_id=resolved_workspace_id,
            task_id=req.task_id or "",
            role=resolved_role,
            summary=f"Agent run failed: {exc}",
        )


@router.post("/replan", response_model=ReplanResponse)
async def replan(req: ReplanRequest) -> ReplanResponse:
    """Re-run the shared workflow from a chosen stage using the persisted task state."""
    from src.research.agents.supervisor import get_supervisor

    supervisor = get_supervisor()
    state = _build_agent_state(
        task_id=req.task_id,
        requested_workspace_id=req.workspace_id,
        inputs=req.inputs,
    )
    resolved_workspace_id = str(state.get("workspace_id", req.workspace_id))

    try:
        result = await supervisor.replan(
            state=state,
            trigger_reason=req.reason,
            target_stage=req.target_stage,
        )
        return ReplanResponse(
            workspace_id=resolved_workspace_id,
            task_id=req.task_id,
            trigger=req.trigger,
            target_stage=req.target_stage,
            output_artifact_ids=result.get("output_artifact_ids", []),
            trace_refs=result.get("trace_refs", []),
            collaboration_trace=result.get("collaboration_trace", []),
            summary=result.get("summary"),
        )
    except Exception as exc:
        logger.exception("Replan failed: %s", exc)
        return ReplanResponse(
            workspace_id=resolved_workspace_id,
            task_id=req.task_id,
            trigger=req.trigger,
            target_stage=req.target_stage,
            summary=f"Replan failed: {exc}",
        )


# ─── Circuit Breaker 状态 ────────────────────────────────────────────────────


class CircuitBreakerStatus(BaseModel):
    state: str
    failure_count: int
    success_count: int
    config: dict


class CircuitBreakerListResponse(BaseModel):
    breakers: dict[str, CircuitBreakerStatus]


@router.get("/circuit-breakers", response_model=CircuitBreakerListResponse)
async def get_circuit_breaker_status() -> CircuitBreakerListResponse:
    """
    返回所有熔断器当前状态（供前端监控面板使用）。

    状态含义：
    - closed: 正常，请求通过
    - open: 熔断，请求被拒绝或返回降级值
    - half_open: 半开，放行测试请求
    """
    from src.agent.circuit_breaker import get_all_breaker_status

    status = get_all_breaker_status()
    breakers = {
        key: CircuitBreakerStatus(
            state=info["state"],
            failure_count=info["failure_count"],
            success_count=info["success_count"],
            config=info["config"],
        )
        for key, info in status.items()
    }
    return CircuitBreakerListResponse(breakers=breakers)


class CircuitBreakerResetRequest(BaseModel):
    key: str


class CircuitBreakerResetResponse(BaseModel):
    ok: bool
    key: str
    message: str


@router.post("/circuit-breakers/reset", response_model=CircuitBreakerResetResponse)
async def reset_circuit_breaker(req: CircuitBreakerResetRequest) -> CircuitBreakerResetResponse:
    """
    手动重置指定熔断器（用于测试或故障恢复后人工干预）。

    注意：通常不需要手动重置，熔断器会自动在 timeout 后进入半开状态。
    """
    from src.agent.circuit_breaker import get_breaker, reset_breaker

    key = req.key
    try:
        # 先检查是否存在
        breakers = get_all_breaker_status()
        if key not in breakers:
            return CircuitBreakerResetResponse(
                ok=False,
                key=key,
                message=f"Circuit breaker '{key}' not found. Available keys: {list(breakers.keys())}",
            )
        reset_breaker(key)
        return CircuitBreakerResetResponse(
            ok=True,
            key=key,
            message=f"Circuit breaker '{key}' has been reset to CLOSED state.",
        )
    except Exception as exc:
        logger.exception("Failed to reset circuit breaker %s: %s", key, exc)
        return CircuitBreakerResetResponse(
            ok=False,
            key=key,
            message=f"Failed to reset: {exc}",
        )
