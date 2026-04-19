from unittest.mock import patch

from src.models.report import Citation, Claim, DraftReport, FinalReport, GroundingStats
from src.models.review import ReviewFeedback
from src.research.graph.nodes.review import review_node


def _make_draft() -> DraftReport:
    return DraftReport(
        sections={"introduction": "test"},
        claims=[Claim(id="c1", text="claim", citation_labels=["[1]"])],
        citations=[Citation(label="[1]", url="https://arxiv.org/abs/1706.03762", reason="paper")],
    )


def test_review_node_returns_grounding_outputs():
    draft = _make_draft()
    grounded_final = FinalReport(
        sections={"introduction": "grounded"},
        claims=[],
        citations=[],
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
    feedback = ReviewFeedback(task_id="t1", workspace_id="ws1", passed=True, summary="ok")

    with (
        patch(
            "src.research.graph.nodes.review.ground_draft_report",
            return_value={
                "resolved_report": {"ok": True},
                "verified_report": {"ok": True},
                "final_report": grounded_final,
                "draft_markdown": "# grounded",
                "warnings": ["citation warning"],
            },
        ),
        patch(
            "src.research.graph.nodes.review._run_reviewer_sync",
            return_value=feedback,
        ),
    ):
        result = review_node(
            {
                "task_id": "t1",
                "workspace_id": "ws1",
                "draft_report": draft,
                "paper_cards": [],
            }
        )

    assert result["review_feedback"].summary == "ok"
    assert result["review_passed"] is True
    assert result["final_report"].report_confidence == "high"
    assert result["draft_markdown"] == "# grounded"
    assert result["warnings"] == ["citation warning"]


def test_review_node_runs_claim_verification_on_verified_report():
    draft = _make_draft()
    feedback = ReviewFeedback(task_id="t1", workspace_id="ws1", passed=True, summary="ok")
    captured: dict[str, object] = {}

    def _capture_skill(*, workspace_id, task_id, draft_report):
        captured["workspace_id"] = workspace_id
        captured["task_id"] = task_id
        captured["draft_report"] = draft_report
        return ({"grounding_stats": {"supported_ratio": 1.0}}, [])

    with (
        patch(
            "src.research.graph.nodes.review.ground_draft_report",
            return_value={
                "verified_report": {"claims": [{"id": "c1", "overall_status": "grounded"}], "citations": [], "sections": {"introduction": "ok"}},
                "final_report": {"claims": [{"id": "c1", "overall_status": "partial"}], "citations": [], "sections": {"introduction": "ok"}},
            },
        ),
        patch(
            "src.research.graph.nodes.review._run_reviewer_sync",
            return_value=feedback,
        ),
        patch(
            "src.research.graph.nodes.review._run_claim_verification_skill",
            side_effect=_capture_skill,
        ),
    ):
        review_node(
            {
                "task_id": "t1",
                "workspace_id": "ws1",
                "draft_report": draft,
                "paper_cards": [],
            }
        )

    assert captured["workspace_id"] == "ws1"
    assert captured["task_id"] == "t1"
    assert captured["draft_report"]["claims"][0]["overall_status"] == "grounded"
