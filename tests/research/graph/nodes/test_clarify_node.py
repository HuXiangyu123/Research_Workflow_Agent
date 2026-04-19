"""Tests for run_clarify_node graph node."""

from unittest.mock import MagicMock, patch

import pytest

from src.research.graph.nodes.clarify import run_clarify_node


# ---------------------------------------------------------------------------
# run_clarify_node — basic guard checks
# ---------------------------------------------------------------------------

def test_node_requires_research_source_type():
    result = run_clarify_node({"source_type": "arxiv", "raw_input": "调研多模态"})
    assert "errors" in result
    assert "source_type must be 'research'" in result["errors"][0]


def test_node_requires_non_empty_raw_input():
    result = run_clarify_node({"source_type": "research", "raw_input": "   "})
    assert "errors" in result
    assert "raw_input is empty" in result["errors"][0]


def test_node_requires_raw_input_key():
    result = run_clarify_node({"source_type": "research"})
    assert "errors" in result


# ---------------------------------------------------------------------------
# run_clarify_node — normal input
# ---------------------------------------------------------------------------

def test_node_normal_input_writes_brief(monkeypatch):
    """Normal research query should produce brief and done node_status."""
    state = {"source_type": "research", "raw_input": "调研多模态学习方法。"}

    mock_result = MagicMock()
    mock_result.warnings = []
    mock_result.brief.model_dump.return_value = {
        "topic": "多模态学习",
        "goal": "调研",
        "desired_output": "survey_outline",
        "sub_questions": ["有哪些方法？"],
        "time_range": "近三年",
        "domain_scope": "多模态",
        "source_constraints": [],
        "focus_dimensions": ["方法"],
        "ambiguities": [],
        "needs_followup": False,
        "confidence": 0.9,
        "schema_version": "v1",
    }
    mock_result.brief.needs_followup = False
    mock_result.brief.confidence = 0.9

    with patch(
        "src.research.graph.nodes.clarify.run_clarify_agent",
        return_value=mock_result,
    ):
        result = run_clarify_node(state)

    assert "brief" in result
    assert result["current_stage"] == "clarify"
    assert result["node_statuses"]["clarify"]["status"] == "done"


# ---------------------------------------------------------------------------
# run_clarify_node — needs_followup handling
# ---------------------------------------------------------------------------

def _mock_followup_result(*, confidence: float = 0.28):
    mock_result = MagicMock()
    mock_result.warnings = []
    mock_result.brief.model_dump.return_value = {
        "topic": "未明确",
        "goal": "初步探索",
        "desired_output": "research_brief",
        "sub_questions": ["用户想调研什么领域？"],
        "time_range": "最近",
        "domain_scope": None,
        "source_constraints": [],
        "focus_dimensions": [],
        "ambiguities": [
            {"field": "topic", "reason": "没有说明", "suggested_options": []}
        ],
        "needs_followup": True,
        "confidence": confidence,
        "schema_version": "v1",
    }
    mock_result.brief.needs_followup = True
    mock_result.brief.confidence = confidence
    return mock_result


def test_node_needs_followup_pauses_in_interactive_mode():
    state = {
        "source_type": "research",
        "raw_input": "帮我看看有什么好方法。",
        "interaction_mode": "interactive",
    }

    with patch(
        "src.research.graph.nodes.clarify.run_clarify_agent",
        return_value=_mock_followup_result(),
    ):
        result = run_clarify_node(state)

    assert result["node_statuses"]["clarify"]["status"] == "limited"
    assert result["current_stage"] == "clarify_followup_required"
    assert result["awaiting_followup"] is True
    assert any("Workflow paused for user follow-up" in w for w in result["warnings"])


def test_node_needs_followup_autofills_in_non_interactive_mode():
    state = {
        "source_type": "research",
        "raw_input": "帮我看看有什么好方法。",
        "interaction_mode": "non_interactive",
        "auto_fill": False,
    }

    filled = MagicMock()
    filled.warnings = []
    filled.brief.model_dump.return_value = {
        "topic": "代码智能体评测",
        "goal": "调研",
        "desired_output": "survey_outline",
        "sub_questions": ["关注哪些评测基准？"],
        "time_range": "近三年",
        "domain_scope": "软件工程",
        "source_constraints": [],
        "focus_dimensions": ["benchmarks"],
        "ambiguities": [],
        "needs_followup": False,
        "confidence": 0.72,
        "schema_version": "v1",
    }
    filled.brief.needs_followup = False
    filled.brief.confidence = 0.72

    with patch(
        "src.research.graph.nodes.clarify.run_clarify_agent",
        side_effect=[_mock_followup_result(), filled],
    ):
        result = run_clarify_node(state)

    assert result["node_statuses"]["clarify"]["status"] == "done"
    assert result["current_stage"] == "clarify"
    assert result["awaiting_followup"] is False
    assert result["followup_resolution"]["auto_fill_triggered"] is True
    assert any("Auto-fill fallback triggered" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# run_clarify_node — exception handling
# ---------------------------------------------------------------------------

def test_node_exception_returns_failed_status():
    state = {"source_type": "research", "raw_input": "调研多模态。"}

    with patch(
        "src.research.graph.nodes.clarify.run_clarify_agent",
        side_effect=RuntimeError("LLM service unavailable"),
    ):
        result = run_clarify_node(state)

    assert "errors" in result
    assert any("RuntimeError" in e for e in result["errors"])
    assert result["node_statuses"]["clarify"]["status"] == "failed"
    assert result["current_stage"] == "clarify"
