from __future__ import annotations

from unittest.mock import patch

from src.models.compression import CompressedCard
from src.research.services.compression import _card_to_text
from src.research.services.compression import compress_paper_cards


def test_compress_paper_cards_accepts_dict_brief():
    paper_cards = [
        {
            "title": "Medical Agent Paper",
            "summary": "A medical agent paper.",
            "abstract": "A medical agent paper.",
            "methods": ["agent"],
            "datasets": ["MIMIC-CXR"],
            "fulltext_available": True,
        }
    ]

    with patch("src.agent.llm.build_reason_llm", side_effect=RuntimeError("forced fallback")):
        result = compress_paper_cards(paper_cards, {"topic": "医疗 agent"})

    assert result.taxonomy is not None
    assert result.compression_stats["processed_cards"] == 1
    assert result.compression_stats["original_chars"] > 0
    assert result.compression_stats["compressed_chars"] > 0


def test_card_to_text_accepts_compressed_card_and_nested_lists():
    card = CompressedCard(
        title="Medical Agent Paper",
        arxiv_id="1234.5678",
        core_claim="Tool-augmented clinical agents improve triage throughput.",
        method_type="agent",
        key_result="AUROC 0.91",
        role_in_taxonomy="medical agent systems",
        connections=["connects radiology triage with VLM tooling"],
    )

    text = _card_to_text(card)

    assert "Medical Agent Paper" in text
    assert "AUROC 0.91" in text
    assert "VLM tooling" in text
