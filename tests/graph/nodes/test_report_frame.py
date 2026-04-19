from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from src.graph.nodes.report_frame import report_frame
from src.models.paper import EvidenceBundle, NormalizedDocument, PaperMetadata


def _state():
    doc = NormalizedDocument(
        metadata=PaperMetadata(title="Attention", authors=["Vaswani"], abstract="Transformer paper"),
        document_text="Full paper text here",
        document_sections={},
        source_manifest={},
    )
    return {"normalized_doc": doc, "evidence": EvidenceBundle(rag_results=[], web_results=[])}


def test_report_frame_generates_draft_report():
    response = MagicMock()
    response.content = json.dumps({
        "sections": {
            "title": "Attention Is All You Need Report",
            "paper_information": "paper info",
            "abstract_and_motivation": "motivation",
            "background_and_related_work": "related work",
            "methods": "method",
            "experiments": "experiments",
            "discussion_and_future_directions": "discussion",
            "conclusion_and_outlook": "summary",
        },
        "claims": [{"id": "c1", "text": "claim", "citation_labels": ["[1]"]}],
        "citations": [{"label": "[1]", "url": "https://example.com", "reason": "source"}],
    })
    with patch("src.agent.settings.Settings.from_env") as mock_env, patch("src.agent.llm.build_chat_llm") as mock_build:
        mock_env.return_value = MagicMock()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = response
        mock_build.return_value = mock_llm
        result = report_frame(_state())

    assert "report_frame" in result
    assert "draft_report" in result
    assert result["draft_report"].sections["methods"] == "method"
    assert result["draft_report"].sections["title"] == "Attention Is All You Need Report"
