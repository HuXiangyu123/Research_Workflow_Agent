from src.agent.report import _final_report_to_markdown
from src.graph.nodes.format_output import format_output
from src.models.report import (
    Claim,
    Citation,
    ClaimSupport,
    DraftReport,
    FinalReport,
    GroundingStats,
    VerifiedReport,
)


def test_format_output_no_report():
    result = format_output({})
    assert "errors" in result


def test_format_output_early_pipeline_failure():
    """safe_abort + accumulated errors → readable FinalReport, not bare format_output error."""
    result = format_output({
        "degradation_mode": "safe_abort",
        "errors": [
            "ingest_source: no arXiv entry for 2510.24036",
            "normalize_metadata: no document text and no abstract available",
        ],
    })
    assert "final_report" in result
    final = result["final_report"]
    assert final.report_confidence == "low"
    assert "无法生成报告" in final.sections
    assert "2510.24036" in final.sections["系统信息"]


def test_format_output_basic():
    report = DraftReport(
        sections={"标题": "Test", "核心贡献": "Something new"},
        claims=[Claim(id="c1", text="a claim", citation_labels=["[1]"])],
        citations=[Citation(label="[1]", url="https://example.com", reason="src")],
    )
    result = format_output({"draft_report": report})
    assert "final_report" in result
    final = result["final_report"]
    assert final.sections["标题"] == "Test"
    assert final.report_confidence == "low"  # 1 ungrounded claim → 0% → low
    assert final.grounding_stats.total_claims == 1


def test_format_output_limited():
    report = DraftReport(sections={"标题": "T"}, claims=[], citations=[])
    result = format_output({"draft_report": report, "degradation_mode": "limited"})
    assert result["final_report"].report_confidence == "limited"


def test_format_output_safe_abort():
    report = DraftReport(sections={"标题": "T"}, claims=[], citations=[])
    result = format_output({"draft_report": report, "degradation_mode": "safe_abort"})
    assert result["final_report"].report_confidence == "low"


def test_format_from_verified_report():
    claims = [
        Claim(
            id="c1",
            text="claim 1",
            citation_labels=["[1]"],
            overall_status="grounded",
            supports=[
                ClaimSupport(
                    claim_id="c1",
                    citation_label="[1]",
                    support_status="supported",
                )
            ],
        ),
        Claim(
            id="c2",
            text="claim 2",
            citation_labels=["[2]"],
            overall_status="ungrounded",
        ),
    ]
    citations = [
        Citation(
            label="[1]",
            url="https://arxiv.org/abs/1",
            reason="r",
            source_tier="A",
            reachable=True,
        ),
        Citation(
            label="[2]",
            url="https://github.com/x",
            reason="r",
            source_tier="B",
            reachable=True,
        ),
    ]
    verified = VerifiedReport(
        sections={"标题": "Test"}, claims=claims, citations=citations
    )

    result = format_output({"verified_report": verified})

    final = result["final_report"]
    assert final.grounding_stats.total_claims == 2
    assert final.grounding_stats.grounded == 1
    assert final.grounding_stats.ungrounded == 1
    assert final.grounding_stats.tier_a_ratio == 0.5
    assert final.grounding_stats.tier_b_ratio == 0.5
    assert final.report_confidence == "low"  # ungrounded claims block higher confidence


def test_format_confidence_from_grounding_low():
    """0% grounded → low confidence even if degradation_mode is normal."""
    claims = [
        Claim(id="c1", text="bad claim", citation_labels=[], overall_status="ungrounded"),
        Claim(id="c2", text="bad claim 2", citation_labels=[], overall_status="ungrounded"),
    ]
    verified = VerifiedReport(sections={"标题": "T"}, claims=claims, citations=[])
    result = format_output({"verified_report": verified, "degradation_mode": "normal"})
    assert result["final_report"].report_confidence == "low"


def test_format_confidence_worse_of_grounding_and_degradation():
    """Degradation=limited + grounding=high → limited (worse wins)."""
    claims = [
        Claim(id="c1", text="g", citation_labels=[], overall_status="grounded"),
    ]
    verified = VerifiedReport(sections={"标题": "T"}, claims=claims, citations=[])
    result = format_output({"verified_report": verified, "degradation_mode": "limited"})
    assert result["final_report"].report_confidence == "limited"


def test_format_confidence_partial_heavy_reports_do_not_score_high():
    claims = [
        Claim(id="c1", text="g", citation_labels=["[1]"], overall_status="grounded"),
        Claim(id="c2", text="p", citation_labels=["[1]"], overall_status="partial"),
        Claim(id="c3", text="p2", citation_labels=["[1]"], overall_status="partial"),
    ]
    verified = VerifiedReport(
        sections={"标题": "T"},
        claims=claims,
        citations=[Citation(label="[1]", url="https://example.com", reason="source")],
    )

    result = format_output({"verified_report": verified, "degradation_mode": "normal"})
    assert result["final_report"].report_confidence == "low"


def test_format_prefers_verified_over_draft():
    draft = DraftReport(sections={"标题": "Draft"}, claims=[], citations=[])
    verified = VerifiedReport(sections={"标题": "Verified"}, claims=[], citations=[])

    result = format_output({"draft_report": draft, "verified_report": verified})
    assert result["final_report"].sections["标题"] == "Verified"


def test_final_report_markdown_uses_english_appendix_labels():
    report = FinalReport(
        sections={
            "title": "Attention Report",
            "paper_information": "Paper info",
            "methods": "Method details",
        },
        claims=[],
        citations=[Citation(label="[1]", url="https://example.com", reason="primary source")],
        grounding_stats=GroundingStats(
            total_claims=1,
            grounded=1,
            partial=0,
            ungrounded=0,
            abstained=0,
            tier_a_ratio=1.0,
            tier_b_ratio=0.0,
        ),
        report_confidence="high",
    )

    markdown = _final_report_to_markdown(report)

    assert markdown.startswith("# Attention Report")
    assert "## Paper Information" in markdown
    assert "## References" in markdown
    assert "## Grounding Summary" in markdown
    assert "## 引用" not in markdown
    assert "## 引用可信度" not in markdown
