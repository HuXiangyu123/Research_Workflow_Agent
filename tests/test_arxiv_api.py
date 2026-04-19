from __future__ import annotations

from src.tools.arxiv_api import _extract_arxiv_id_from_url, enrich_search_results_with_arxiv


def test_extract_arxiv_id_from_url_strips_version():
    assert _extract_arxiv_id_from_url("https://arxiv.org/abs/2407.15621v3") == "2407.15621"
    assert _extract_arxiv_id_from_url("https://arxiv.org/pdf/2407.15621v3.pdf") == "2407.15621"


def test_enrich_search_results_preserves_existing_arxiv_id_when_metadata_lookup_misses(monkeypatch):
    monkeypatch.setattr("src.tools.arxiv_api.fetch_arxiv_papers_by_ids", lambda _ids: {})

    enriched = enrich_search_results_with_arxiv(
        [
            {
                "title": "RadioRAG",
                "arxiv_id": "2407.15621",
                "url": "http://arxiv.org/abs/2407.15621v3",
                "abstract": "radiology retrieval paper",
            }
        ]
    )

    assert len(enriched) == 1
    assert enriched[0]["arxiv_id"] == "2407.15621"
    assert enriched[0]["pdf_url"] == "https://arxiv.org/pdf/2407.15621.pdf"
