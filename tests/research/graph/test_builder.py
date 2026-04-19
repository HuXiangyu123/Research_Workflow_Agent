from __future__ import annotations

from langgraph.graph import END

from src.research.graph.builder import _route_after_clarify


def test_route_after_clarify_stops_when_waiting_for_followup():
    assert _route_after_clarify({"brief": {"topic": "RAG"}, "awaiting_followup": True}) == END


def test_route_after_clarify_continues_when_brief_ready():
    assert _route_after_clarify({"brief": {"topic": "RAG"}, "awaiting_followup": False}) == "search_plan"
