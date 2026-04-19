from unittest.mock import patch

from src.graph.nodes.resolve_citations import resolve_citations
from src.models.report import Citation, Claim, DraftReport


def _make_draft(citations=None):
    if citations is None:
        citations = [
            Citation(
                label="[1]",
                url="https://arxiv.org/abs/1706.03762",
                reason="original",
            ),
            Citation(label="[2]", url="https://github.com/user/repo", reason="code"),
            Citation(label="[3]", url="https://random-blog.xyz/post", reason="blog"),
        ]
    return DraftReport(
        sections={"标题": "Test"},
        claims=[Claim(id="c1", text="a claim", citation_labels=["[1]"])],
        citations=citations,
    )


def test_resolve_no_draft():
    result = resolve_citations({})
    assert "warnings" in result


def test_resolve_classifies_tiers():
    draft = _make_draft()
    with (
        patch(
            "src.graph.nodes.resolve_citations.check_url_reachable_sync",
            return_value=True,
        ),
        patch(
            "src.graph.nodes.resolve_citations._fetch_content_snippet",
            return_value="some content",
        ),
    ):
        result = resolve_citations({"draft_report": draft})

    resolved = result["resolved_report"]
    assert resolved.citations[0].source_tier == "A"  # arxiv
    assert resolved.citations[1].source_tier == "B"  # github
    assert resolved.citations[2].source_tier == "D"  # random blog


def test_resolve_marks_reachability():
    draft = _make_draft()

    def mock_reachable(url):
        return "arxiv" in url

    with (
        patch(
            "src.graph.nodes.resolve_citations.check_url_reachable_sync",
            side_effect=mock_reachable,
        ),
        patch(
            "src.graph.nodes.resolve_citations._fetch_content_snippet",
            return_value="content",
        ),
    ):
        result = resolve_citations({"draft_report": draft})

    resolved = result["resolved_report"]
    assert resolved.citations[0].reachable is True
    assert resolved.citations[1].reachable is False
    assert "warnings" in result


def test_resolve_fetches_content_when_reachable():
    """Reachable citations fetch content for every tier (claim judge needs it)."""
    draft = _make_draft(
        citations=[
            Citation(
                label="[1]",
                url="https://arxiv.org/abs/1706.03762",
                reason="paper",
            ),
            Citation(label="[2]", url="https://random-blog.xyz", reason="blog"),
        ]
    )

    with (
        patch(
            "src.graph.nodes.resolve_citations.check_url_reachable_sync",
            return_value=True,
        ),
        patch(
            "src.graph.nodes.resolve_citations._fetch_content_snippet",
            return_value="fetched text",
        ),
    ):
        result = resolve_citations({"draft_report": draft})

    resolved = result["resolved_report"]
    assert resolved.citations[0].fetched_content == "fetched text"  # Tier A
    assert resolved.citations[1].fetched_content == "fetched text"  # Tier D, reachable


def test_resolve_preserves_sections_and_claims():
    draft = _make_draft()
    with (
        patch(
            "src.graph.nodes.resolve_citations.check_url_reachable_sync",
            return_value=True,
        ),
        patch(
            "src.graph.nodes.resolve_citations._fetch_content_snippet",
            return_value=None,
        ),
    ):
        result = resolve_citations({"draft_report": draft})

    resolved = result["resolved_report"]
    assert resolved.sections == draft.sections
    assert len(resolved.claims) == len(draft.claims)


def test_resolve_preserves_existing_grounding_evidence():
    draft = _make_draft(
        citations=[
            Citation(
                label="[1]",
                url="https://arxiv.org/abs/1706.03762",
                reason="paper",
                fetched_content="This citation already contains abstract evidence about the method and results.",
            )
        ]
    )

    with (
        patch(
            "src.graph.nodes.resolve_citations.check_url_reachable_sync",
            return_value=True,
        ),
        patch(
            "src.graph.nodes.resolve_citations._fetch_content_snippet",
            return_value="remote html should not override existing evidence",
        ) as fetch_mock,
    ):
        result = resolve_citations({"draft_report": draft})

    resolved = result["resolved_report"]
    assert resolved.citations[0].fetched_content.startswith("This citation already contains")
    fetch_mock.assert_not_called()


def test_resolve_extracts_arxiv_abstract_from_html():
    draft = _make_draft(
        citations=[
            Citation(
                label="[1]",
                url="https://arxiv.org/abs/1706.03762",
                reason="paper",
            )
        ]
    )
    arxiv_html = """
    <!DOCTYPE html>
    <html>
      <body>
        <blockquote class="abstract mathjax">
          <span class="descriptor">Abstract:</span>
          This paper studies a grounded retrieval pipeline and reports strong evidence on benchmark tasks.
        </blockquote>
      </body>
    </html>
    """

    with (
        patch(
            "src.graph.nodes.resolve_citations.check_url_reachable_sync",
            return_value=True,
        ),
        patch(
            "src.graph.nodes.resolve_citations._fetch_content_snippet",
            return_value=arxiv_html,
        ),
    ):
        result = resolve_citations({"draft_report": draft})

    resolved = result["resolved_report"]
    assert "grounded retrieval pipeline" in resolved.citations[0].fetched_content
