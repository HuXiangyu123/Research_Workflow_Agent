from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.routes.tasks import clear_tasks_store, get_tasks_store
from src.models.config import ExecutionMode, Phase4Config
from src.models.task import TaskRecord
from src.research.agents.supervisor import get_supervisor


@pytest.fixture(autouse=True)
def clean_state():
    clear_tasks_store()
    supervisor = get_supervisor()
    supervisor.set_config(Phase4Config())
    yield
    clear_tasks_store()
    supervisor.set_config(Phase4Config())


client = TestClient(app)


def test_phase4_config_updates_supervisor_runtime():
    payload = {
        "config": {
            "execution_mode": "v2",
            "agent_mode": "auto",
            "enable_mcp": True,
            "enable_skills": True,
            "enable_replan": True,
            "node_backends": {
                "clarify": "legacy",
                "search_plan": "auto",
                "search": "v2",
                "extract": "legacy",
                "draft": "v2",
                "review": "auto",
                "persist_artifacts": "legacy",
            },
        }
    }

    resp = client.post("/api/v1/config/phase4", json=payload)
    assert resp.status_code == 200
    assert resp.json()["config"]["execution_mode"] == "v2"
    assert get_supervisor().config.execution_mode == ExecutionMode.V2
    assert get_supervisor().config.node_backends.search.value == "v2"


def test_run_agent_hydrates_existing_task_state():
    task = TaskRecord(
        input_type="research",
        input_value="调研医疗AI agent",
        report_mode="draft",
        source_type="research",
        workspace_id="ws_real_123",
    )
    task.brief = {"topic": "医疗 AI agent", "needs_followup": False}
    task.search_plan = {"plan_goal": "搜索", "query_groups": [{"group_id": "g1", "queries": ["medical ai agent"]}]}
    task.paper_cards = [{"title": "Paper 1"}]
    task.draft_report = {"sections": {"introduction": "hello"}}
    get_tasks_store()[task.task_id] = task

    fake_supervisor = MagicMock()
    fake_supervisor.run_node = AsyncMock(return_value={"summary": "review complete", "_agent_paradigm": "reflexion"})

    with patch("src.research.agents.supervisor.get_supervisor", return_value=fake_supervisor):
        resp = client.post(
            "/api/v1/agents/run",
            json={
                "workspace_id": task.task_id,
                "task_id": task.task_id,
                "role": "reviewer",
                "inputs": {},
            },
        )

    assert resp.status_code == 200
    assert resp.json()["workspace_id"] == "ws_real_123"
    fake_supervisor.run_node.assert_awaited_once()
    node_name, state, inputs = fake_supervisor.run_node.await_args.args
    assert node_name == "review"
    assert state["workspace_id"] == "ws_real_123"
    assert state["draft_report"] == task.draft_report
    assert state["interaction_mode"] == "interactive"
    assert inputs == {}


def test_supervisor_run_returns_collaboration_trace():
    task = TaskRecord(
        input_type="research",
        input_value="调研RAG",
        report_mode="draft",
        source_type="research",
        workspace_id="ws_supervisor",
    )
    task.brief = {"topic": "RAG", "needs_followup": False}
    task.search_plan = {"plan_goal": "plan", "query_groups": [{"group_id": "g1", "queries": ["rag"]}]}
    task.rag_result = {"paper_candidates": [{"title": "p1"}]}
    get_tasks_store()[task.task_id] = task

    fake_supervisor = MagicMock()
    fake_supervisor.collaborate = AsyncMock(
        return_value={
            "summary": "Supervisor resumed from extract and coordinated extract -> draft -> review.",
            "trace_refs": ["extract", "draft", "review"],
            "collaboration_trace": [
                {"node": "extract", "backend": "legacy", "paradigm": "legacy", "produced": ["paper_cards"]},
                {"node": "draft", "backend": "v2", "paradigm": "reasoning_via_artifacts", "produced": ["draft_report"]},
            ],
        }
    )

    with patch("src.research.agents.supervisor.get_supervisor", return_value=fake_supervisor):
        resp = client.post(
            "/api/v1/agents/run",
            json={
                "workspace_id": "wrong_workspace",
                "task_id": task.task_id,
                "role": "supervisor",
                "inputs": {},
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["workspace_id"] == "ws_supervisor"
    assert data["trace_refs"] == ["extract", "draft", "review"]
    assert data["collaboration_trace"][0]["node"] == "extract"
    fake_supervisor.collaborate.assert_awaited_once()
    assert fake_supervisor.collaborate.await_args.kwargs["state"]["workspace_id"] == "ws_supervisor"


def test_replan_uses_hydrated_state():
    task = TaskRecord(
        input_type="research",
        input_value="调研Agent评测",
        report_mode="draft",
        source_type="research",
        workspace_id="ws_replan",
    )
    task.brief = {"topic": "Agent eval", "needs_followup": False}
    task.search_plan = {"plan_goal": "eval", "query_groups": [{"group_id": "g1", "queries": ["agent eval"]}]}
    task.paper_cards = [{"title": "Paper"}]
    get_tasks_store()[task.task_id] = task

    fake_supervisor = MagicMock()
    fake_supervisor.replan = AsyncMock(
        return_value={
            "summary": "Supervisor resumed from search_plan and coordinated search_plan -> search.",
            "trace_refs": ["search_plan", "search"],
            "collaboration_trace": [{"node": "search_plan", "backend": "v2", "paradigm": "plan_and_execute", "produced": ["search_plan"]}],
        }
    )

    with patch("src.research.agents.supervisor.get_supervisor", return_value=fake_supervisor):
        resp = client.post(
            "/api/v1/agents/replan",
            json={
                "workspace_id": "wrong_workspace",
                "task_id": task.task_id,
                "trigger": "user",
                "reason": "Need broader coverage",
                "target_stage": "search_plan",
            },
        )

    assert resp.status_code == 200
    assert resp.json()["workspace_id"] == "ws_replan"
    fake_supervisor.replan.assert_awaited_once()
    assert fake_supervisor.replan.await_args.kwargs["state"]["brief"] == task.brief
    assert fake_supervisor.replan.await_args.kwargs["target_stage"] == "search_plan"
