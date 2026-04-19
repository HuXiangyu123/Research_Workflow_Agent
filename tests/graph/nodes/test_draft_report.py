from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

from src.graph.nodes.draft_report import draft_report
from src.models.paper import PaperMetadata, NormalizedDocument, EvidenceBundle


def _make_state():
    meta = PaperMetadata(title="Test", authors=["A"], abstract="Abstract")
    doc = NormalizedDocument(
        metadata=meta,
        document_text="Full text",
        document_sections={},
        source_manifest={},
    )
    evidence = EvidenceBundle(rag_results=[], web_results=[])
    return {"normalized_doc": doc, "evidence": evidence}


def _mock_llm_response(content: str):
    resp = MagicMock()
    resp.content = content
    return resp


def test_draft_no_doc():
    result = draft_report({})
    assert "errors" in result


def test_draft_valid_json():
    state = _make_state()
    json_output = json.dumps({
        "sections": {"title": "Test Paper", "core_contributions": "contribution"},
        "claims": [{"id": "c1", "text": "A claim", "citation_labels": ["[1]"]}],
        "citations": [{"label": "[1]", "url": "https://example.com", "reason": "source"}],
    })

    with patch("src.agent.settings.Settings.from_env") as mock_from_env, \
         patch("src.agent.llm.build_reason_llm") as mock_build:
        mock_from_env.return_value = MagicMock()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response(json_output)
        mock_build.return_value = mock_llm

        result = draft_report(state)

    assert "draft_report" in result
    assert result["draft_report"].sections["title"] == "Test Paper"
    assert len(result["draft_report"].claims) == 1
    assert len(result["draft_report"].citations) == 1


def test_draft_invalid_json_fallback():
    state = _make_state()
    with patch("src.agent.settings.Settings.from_env") as mock_from_env, \
         patch("src.agent.llm.build_reason_llm") as mock_build:
        mock_from_env.return_value = MagicMock()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response(
            "This is not JSON, just a text report."
        )
        mock_build.return_value = mock_llm

        result = draft_report(state)

    assert "draft_report" in result
    assert "warnings" in result
