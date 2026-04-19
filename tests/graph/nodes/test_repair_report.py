from __future__ import annotations

from src.graph.nodes.repair_report import repair_report
from src.models.report import DraftReport, Claim, Citation


def _make_complete_report():
    return DraftReport(
        sections={
            "title": "T",
            "paper_information": "P",
            "methods": "M",
            "experiments": "E",
            "limitations": "L",
        },
        claims=[Claim(id="c1", text="claim", citation_labels=["[1]"])],
        citations=[Citation(label="[1]", url="https://example.com", reason="r")],
    )


def _make_incomplete_report():
    return DraftReport(
        sections={"title": "T"},
        claims=[],
        citations=[],
    )


def test_repair_passthrough():
    result = repair_report({"draft_report": _make_complete_report()})
    assert result == {}


def test_repair_passthrough_for_legacy_chinese_sections():
    legacy = DraftReport(
        sections={
            "标题": "T",
            "核心贡献": "C",
            "方法概述": "M",
            "关键实验": "E",
            "局限性": "L",
        },
        claims=[],
        citations=[Citation(label="[1]", url="https://example.com", reason="r")],
    )

    result = repair_report({"draft_report": legacy})
    assert result == {}


def test_repair_no_report():
    result = repair_report({})
    assert "warnings" in result


def test_repair_triggered():
    """When incomplete, repair should be attempted. With no LLM available, it should warn but not crash."""
    result = repair_report({"draft_report": _make_incomplete_report()})
    assert "warnings" in result
