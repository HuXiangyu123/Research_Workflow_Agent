from __future__ import annotations

from types import SimpleNamespace

from src.research.graph.nodes.extract import (
    _extract_cards_batch,
    _enrich_card,
    _select_candidates_for_extract,
    _simple_card,
)


def _candidate(*, idx: int, fulltext: bool, score: float = 1.0) -> dict:
    candidate = {
        "title": f"Paper {idx}",
        "abstract": "abstract " * 80,
        "score": score,
        "fulltext_available": fulltext,
        "fulltext_snippets": [],
    }
    if fulltext:
        candidate["fulltext_snippets"] = [
            {"section": "method", "text": "fulltext evidence " * 120}
        ]
    return candidate


def test_select_candidates_for_extract_preserves_fulltext_ratio():
    candidates = [
        _candidate(idx=0, fulltext=True, score=3.0),
        _candidate(idx=1, fulltext=True, score=2.9),
        _candidate(idx=2, fulltext=True, score=2.8),
        _candidate(idx=3, fulltext=True, score=2.7),
        _candidate(idx=4, fulltext=False, score=2.6),
        _candidate(idx=5, fulltext=False, score=2.5),
    ]

    selected = _select_candidates_for_extract(
        candidates,
        max_candidates_hard=10,
        evidence_char_budget=20000,
        min_fulltext_ratio=0.7,
    )

    fulltext_count = sum(1 for cand in selected if cand.get("fulltext_available"))
    assert len(selected) == 5
    assert fulltext_count / len(selected) >= 0.7


def test_simple_card_prefers_fulltext_evidence_over_abstract():
    card = _simple_card(
        {
            "title": "Paper",
            "abstract": "short abstract",
            "fulltext_snippets": [{"section": "evaluation", "text": "reported evaluation evidence"}],
        },
        0,
    )

    assert card["fulltext_available"] is True
    assert "reported evaluation evidence" in card["abstract"]
    assert card["summary"].startswith("[evaluation]")


def test_enrich_card_replaces_placeholder_metadata_with_original_candidate():
    card = {
        "title": "Unknown",
        "authors": ["Unknown"],
        "arxiv_id": "N/A",
        "url": "N/A",
        "summary": "short summary",
    }
    original = {
        "title": "RadioRAG",
        "authors": ["Alice", "Bob"],
        "arxiv_id": "2407.15621",
        "url": "https://arxiv.org/abs/2407.15621",
        "fulltext_snippets": [{"section": "method", "text": "retrieval evidence " * 40}],
    }

    _enrich_card(card, original)

    assert card["title"] == "RadioRAG"
    assert card["authors"] == ["Alice", "Bob"]
    assert card["arxiv_id"] == "2407.15621"
    assert card["url"] == "https://arxiv.org/abs/2407.15621"
    assert card["fulltext_available"] is True


def test_extract_cards_batch_fills_missing_cards_from_fallback(monkeypatch):
    class DummyLLM:
        def invoke(self, _messages):
            return SimpleNamespace(
                content=(
                    '{"title":"Only One","authors":["Unknown"],"summary":"single card",'
                    '"arxiv_id":"N/A","url":"N/A","methods":[],"datasets":[],"limitations":[]}'
                )
            )

    monkeypatch.setattr("src.agent.settings.get_settings", lambda: object())
    monkeypatch.setattr("src.agent.llm.build_reason_llm", lambda *_args, **_kwargs: DummyLLM())
    monkeypatch.setattr("src.tools.deepxiv_client.batch_get_briefs", lambda *_args, **_kwargs: {})

    cards = _extract_cards_batch(
        [
            {"title": "Paper A", "authors": ["Alice"], "arxiv_id": "2407.15621", "url": "https://arxiv.org/abs/2407.15621", "abstract": "abstract a"},
            {"title": "Paper B", "authors": ["Bob"], "arxiv_id": "2406.17608", "url": "https://arxiv.org/abs/2406.17608", "abstract": "abstract b"},
            {"title": "Paper C", "authors": ["Carol"], "arxiv_id": "2509.26351", "url": "https://arxiv.org/abs/2509.26351", "abstract": "abstract c"},
        ],
        brief={"topic": "medical imaging agents"},
    )

    assert len(cards) == 3
    assert cards[0]["title"] == "Paper A"
    assert cards[0]["authors"] == ["Alice"]
    assert cards[1]["title"] == "Paper B"
    assert cards[2]["title"] == "Paper C"
