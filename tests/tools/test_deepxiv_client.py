from __future__ import annotations

from src.tools import deepxiv_client


class _FakeReader:
    def __init__(self) -> None:
        self.trending_days: list[int] = []

    def search(self, *_args, **_kwargs):
        return {
            "results": [
                {
                    "arxiv_id": "2401.00001",
                    "title": "Example Paper",
                    "abstract": "Example abstract",
                    "author_names": "Alice, Bob",
                    "publish_at": "2024-01-03T00:00:00",
                    "categories": ["cs.AI"],
                    "src_url": "https://arxiv.org/pdf/2401.00001.pdf",
                    "citation": 12,
                    "score": 0.9,
                }
            ]
        }

    def brief(self, _arxiv_id: str):
        return {
            "title": "Example Paper",
            "tldr": "Short summary",
            "keywords": ["agents"],
            "github_url": "https://github.com/example/repo",
            "citations": 9,
            "publish_at": "2024-01-03T00:00:00",
            "src_url": "https://arxiv.org/pdf/2401.00001.pdf",
        }

    def head(self, _arxiv_id: str):
        return {
            "title": "Example Paper",
            "abstract": "Full abstract",
            "authors": ["Alice", "Bob"],
            "sections": [{"name": "1. Introduction", "token_count": 100}],
            "token_count": 1234,
            "categories": ["cs.AI"],
            "publish_at": "2024-01-03T00:00:00",
            "src_url": "https://arxiv.org/pdf/2401.00001.pdf",
        }

    def section(self, _arxiv_id: str, _section: str):
        return "section text"

    def trending(self, *, days: int, limit: int):
        self.trending_days.append(days)
        return {
            "papers": [
                {
                    "arxiv_id": "2401.00002",
                    "title": "Trending Paper",
                    "abstract": "Trending abstract",
                    "authors": ["Carol"],
                    "publish_at": "2024-02-01T00:00:00",
                    "categories": ["cs.LG"],
                }
            ]
        }

    def social_impact(self, _arxiv_id: str):
        return {"total_tweets": 3}


def test_search_papers_maps_live_response_shape(monkeypatch):
    monkeypatch.setattr(deepxiv_client, "_init_reader", lambda: _FakeReader())

    papers = deepxiv_client.search_papers("agent memory", size=5)

    assert papers[0]["arxiv_id"] == "2401.00001"
    assert papers[0]["authors"] == ["Alice", "Bob"]
    assert papers[0]["published_date"] == "2024-01-03"
    assert papers[0]["citation_count"] == 12
    assert papers[0]["pdf_url"].endswith("2401.00001.pdf")


def test_get_paper_brief_maps_citations_and_pdf_url(monkeypatch):
    monkeypatch.setattr(deepxiv_client, "_init_reader", lambda: _FakeReader())

    brief = deepxiv_client.get_paper_brief("2401.00001")

    assert brief is not None
    assert brief["num_citations"] == 9
    assert brief["published_date"] == "2024-01-03"
    assert brief["pdf_url"].endswith("2401.00001.pdf")


def test_get_trending_papers_normalizes_unsupported_days(monkeypatch):
    reader = _FakeReader()
    monkeypatch.setattr(deepxiv_client, "_init_reader", lambda: reader)

    papers = deepxiv_client.get_trending_papers(days=52, size=5)

    assert reader.trending_days == [30]
    assert papers[0]["arxiv_id"] == "2401.00002"


def test_get_paper_popularity_prefers_social_impact(monkeypatch):
    monkeypatch.setattr(deepxiv_client, "_init_reader", lambda: _FakeReader())

    popularity = deepxiv_client.get_paper_popularity("2401.00001")

    assert popularity == {"total_tweets": 3}


def test_is_available_requires_configured_token(monkeypatch):
    monkeypatch.setattr(deepxiv_client, "_reader", None)
    monkeypatch.setattr(deepxiv_client, "_init_reader", lambda: _FakeReader())
    monkeypatch.setattr(
        deepxiv_client,
        "_current_reader_config",
        lambda: ("token-present", "https://data.rag.ac.cn", 60, 3),
    )
    monkeypatch.setattr(deepxiv_client, "_reader_init_ok", True)

    assert deepxiv_client.is_available() is True
