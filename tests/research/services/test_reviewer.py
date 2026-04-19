from __future__ import annotations

import asyncio

from src.models.report import Citation, Claim, DraftReport
from src.research.services.reviewer import ReviewerService


def test_reviewer_fails_when_ungrounded_claim_ratio_is_high():
    report = DraftReport(
        sections={"introduction": "test"},
        claims=[
            Claim(id="c1", text="claim one", citation_labels=["[1]"], overall_status="ungrounded"),
            Claim(id="c2", text="claim two", citation_labels=["[1]"], overall_status="ungrounded"),
            Claim(id="c3", text="claim three", citation_labels=["[1]"], overall_status="ungrounded"),
            Claim(id="c4", text="claim four", citation_labels=["[1]"], overall_status="partial"),
        ],
        citations=[Citation(label="[1]", url="https://arxiv.org/abs/1234.5678", reason="paper")],
    )

    feedback = asyncio.run(
        ReviewerService().review(
            task_id="t1",
            workspace_id="ws1",
            rag_result=None,
            paper_cards=[{"title": "Paper A", "authors": ["A"], "abstract": "Abstract text"}],
            report_draft=report,
        )
    )

    assert feedback.passed is False
    assert any(
        "claims are currently ungrounded" in issue.summary and issue.severity in {"error", "blocker"}
        for issue in feedback.issues
    )


def test_reviewer_fails_when_partial_claim_ratio_is_high():
    report = DraftReport(
        sections={"introduction": "test"},
        claims=[
            Claim(id="c1", text="claim one", citation_labels=["[1]"], overall_status="grounded"),
            Claim(id="c2", text="claim two", citation_labels=["[1]"], overall_status="partial"),
            Claim(id="c3", text="claim three", citation_labels=["[1]"], overall_status="partial"),
            Claim(id="c4", text="claim four", citation_labels=["[1]"], overall_status="partial"),
        ],
        citations=[Citation(label="[1]", url="https://arxiv.org/abs/1234.5678", reason="paper")],
    )

    feedback = asyncio.run(
        ReviewerService().review(
            task_id="t1",
            workspace_id="ws1",
            rag_result=None,
            paper_cards=[{"title": "Paper A", "authors": ["A"], "abstract": "Abstract text"}],
            report_draft=report,
        )
    )

    assert feedback.passed is False
    assert any(
        "only partially supported" in issue.summary and issue.severity == "error"
        for issue in feedback.issues
    )


def test_reviewer_flags_narrow_citation_pool_with_heavy_reuse():
    report = DraftReport(
        sections={
            "introduction": " ".join(["[1] [2] [3] [4] [5]"] * 6),
            "discussion": " ".join(["[1] [2] [3] [4] [5]"] * 6),
        },
        claims=[
            Claim(id="c1", text="claim one", citation_labels=["[1]", "[2]"], overall_status="grounded"),
            Claim(id="c2", text="claim two", citation_labels=["[3]", "[4]"], overall_status="grounded"),
        ],
        citations=[
            Citation(label=f"[{idx}]", url=f"https://arxiv.org/abs/1234.567{idx}", reason="paper")
            for idx in range(1, 6)
        ],
    )

    feedback = asyncio.run(
        ReviewerService().review(
            task_id="t1",
            workspace_id="ws1",
            rag_result=None,
            paper_cards=[{"title": f"Paper {idx}", "authors": ["A"], "abstract": "Abstract text"} for idx in range(1, 6)],
            report_draft=report,
        )
    )

    assert feedback.passed is False
    assert any(
        "narrow evidence base" in issue.summary and issue.severity == "error"
        for issue in feedback.issues
    )
