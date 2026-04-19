from __future__ import annotations

from unittest.mock import AsyncMock

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
import pytest

from src.agent import output_workspace as ow
from src.models.config import Phase4Config, SupervisorMode
from src.research.agents.supervisor import AgentSupervisor


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


@pytest.fixture(autouse=True)
def _isolate_workspace_output(tmp_path, monkeypatch):
    """Keep supervisor smoke/unit tests from polluting the real output/workspaces tree."""
    monkeypatch.setattr(ow, "OUTPUT_ROOT", tmp_path)


def test_supervisor_normalizes_legacy_aliases():
    """Test that legacy aliases are no longer normalized.

    NOTE: LEGACY_NODE_ALIASES was removed in task416.
    The current implementation does NOT normalize aliases anymore.
    Node names must be passed in canonical form.
    """
    supervisor = AgentSupervisor(config=Phase4Config())

    # Legacy aliases removed - normalize_node_name now returns as-is
    # These assertions reflect the new behavior
    assert supervisor.normalize_node_name("plan_search") == "plan_search"
    assert supervisor.normalize_node_name("search_corpus") == "search_corpus"
    assert supervisor.normalize_node_name("extract_cards") == "extract_cards"

    # Canonical names are returned unchanged
    assert supervisor.normalize_node_name("search_plan") == "search_plan"
    assert supervisor.normalize_node_name("search") == "search"
    assert supervisor.normalize_node_name("extract") == "extract"
    assert supervisor.normalize_node_name("draft") == "draft"
    assert supervisor.normalize_node_name("review") == "review"


@pytest.mark.asyncio
async def test_collaborate_resumes_from_missing_stage():
    supervisor = AgentSupervisor(config=Phase4Config())
    calls: list[str] = []

    async def fake_run_node(node_name: str, state: dict, inputs: dict | None = None) -> dict:
        calls.append(node_name)
        if node_name == "search":
            return {
                "rag_result": {"paper_candidates": [{"title": "p1"}]},
                "current_stage": "search",
                "_backend_mode": "v2",
                "_agent_paradigm": "tag",
            }
        if node_name == "extract":
            return {
                "paper_cards": [{"title": "p1"}],
                "current_stage": "extract",
                "_backend_mode": "legacy",
                "_agent_paradigm": "legacy",
            }
        if node_name == "extract_compression":
            return {
                "compression_result": {"taxonomy": {"categories": []}, "compressed_cards": [], "evidence_pools": {}},
                "current_stage": "extract_compression",
                "_backend_mode": "legacy",
                "_agent_paradigm": "legacy",
            }
        if node_name == "draft":
            return {
                "draft_report": {"sections": {"introduction": "ok"}},
                "current_stage": "draft",
                "_backend_mode": "v2",
                "_agent_paradigm": "reasoning_via_artifacts",
            }
        if node_name == "review":
            return {
                "review_feedback": {"passed": True},
                "review_passed": True,
                "current_stage": "review",
                "_backend_mode": "v2",
                "_agent_paradigm": "reflexion",
            }
        if node_name == "persist_artifacts":
            return {
                "artifact_count": 4,
                "current_stage": "persist_artifacts",
                "_backend_mode": "legacy",
                "_agent_paradigm": "legacy",
            }
        raise AssertionError(f"unexpected node {node_name}")

    supervisor.run_node = AsyncMock(side_effect=fake_run_node)

    state = {
        "workspace_id": "ws1",
        "task_id": "t1",
        "brief": {"topic": "RAG", "needs_followup": False},
        "search_plan": {"plan_goal": "search", "query_groups": [{"group_id": "g1", "queries": ["rag"]}]},
    }

    result = await supervisor.collaborate(state)

    assert calls == ["search", "extract", "extract_compression", "draft", "review", "persist_artifacts"]
    assert result["trace_refs"] == calls
    assert result["collaboration_trace"][0]["node"] == "search"
    assert result["collaboration_trace"][1]["node"] == "extract"
    assert result["collaboration_trace"][2]["node"] == "extract_compression"
    assert "Supervisor resumed from search" in result["summary"]


@pytest.mark.asyncio
async def test_replan_prunes_downstream_state_before_resuming():
    supervisor = AgentSupervisor(config=Phase4Config())
    captured_state: dict | None = None

    async def fake_collaborate(state: dict, **kwargs) -> dict:
        nonlocal captured_state
        captured_state = dict(state)
        return {"summary": "ok", "trace_refs": ["search_plan"], "collaboration_trace": []}

    supervisor.collaborate = AsyncMock(side_effect=fake_collaborate)

    await supervisor.replan(
        {
            "workspace_id": "ws1",
            "task_id": "t1",
            "brief": {"topic": "RAG"},
            "search_plan": {"plan_goal": "old"},
            "rag_result": {"paper_candidates": [{"title": "p1"}]},
            "paper_cards": [{"title": "p1"}],
            "draft_report": {"sections": {"intro": "x"}},
            "review_feedback": {"passed": False},
        },
        trigger_reason="coverage gap",
        target_stage="search_plan",
    )

    assert captured_state is not None
    assert captured_state["brief"] == {"topic": "RAG"}
    assert "search_plan" not in captured_state
    assert "rag_result" not in captured_state
    assert "paper_cards" not in captured_state


@pytest.mark.asyncio
async def test_collaborate_stops_real_execution_after_interactive_followup():
    supervisor = AgentSupervisor(config=Phase4Config())
    calls: list[str] = []

    async def fake_run_node(node_name: str, state: dict, inputs: dict | None = None) -> dict:
        calls.append(node_name)
        if node_name == "clarify":
            return {
                "brief": {"topic": "RAG", "needs_followup": True},
                "awaiting_followup": True,
                "current_stage": "clarify_followup_required",
                "_backend_mode": "legacy",
                "_agent_paradigm": "legacy",
            }
        raise AssertionError(f"unexpected node execution: {node_name}")

    supervisor.run_node = AsyncMock(side_effect=fake_run_node)

    result = await supervisor.collaborate(
        {
            "workspace_id": "ws1",
            "task_id": "t1",
            "raw_input": "help me explore recent RAG work",
        }
    )

    assert calls == ["clarify"]
    assert result["trace_refs"] == ["clarify"]
    assert result["current_stage"] == "clarify_followup_required"


@pytest.mark.asyncio
async def test_collaborate_skips_persist_when_review_fails():
    supervisor = AgentSupervisor(config=Phase4Config())
    calls: list[str] = []

    async def fake_run_node(node_name: str, state: dict, inputs: dict | None = None) -> dict:
        calls.append(node_name)
        if node_name == "review":
            return {
                "review_feedback": {"passed": False},
                "review_passed": False,
                "current_stage": "review",
                "_backend_mode": "v2",
                "_agent_paradigm": "reflexion",
            }
        raise AssertionError(f"unexpected node execution: {node_name}")

    supervisor.run_node = AsyncMock(side_effect=fake_run_node)

    result = await supervisor.collaborate(
        {
            "workspace_id": "ws1",
            "task_id": "t1",
            "brief": {"topic": "RAG", "needs_followup": False},
            "search_plan": {"plan_goal": "search", "query_groups": []},
            "rag_result": {"paper_candidates": [{"title": "p1"}]},
            "paper_cards": [{"title": "p1"}],
            "compression_result": {"taxonomy": {}, "compressed_cards": [], "evidence_pools": {}},
            "draft_report": {"sections": {"introduction": "ok"}},
        }
    )

    assert calls == ["review"]
    assert result["trace_refs"] == ["review"]
