"""RagEvalRunner tests."""

from __future__ import annotations

import json

from src.corpus.search.models import EvidenceChunk
from src.eval.rag.models import GoldClaim, RagEvalCase
from src.eval.rag.runner import RagEvalRunner


def test_load_cases_full_uses_scifact_full_filename(tmp_path):
    case = {
        "case_id": "scifact-1",
        "query": "test query",
        "sub_questions": [],
        "gold_papers": [
            {
                "title": "Paper A",
                "canonical_id": "doc-1",
                "arxiv_id": "",
                "expected_rank": 0,
            }
        ],
        "gold_evidence": [],
        "gold_claims": [],
        "source": "scifact",
    }
    (tmp_path / "scifact_full.jsonl").write_text(
        json.dumps(case, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    runner = RagEvalRunner(cases_dir=tmp_path)
    cases = runner.load_cases("full")

    assert len(cases) == 1
    assert cases[0].case_id == "scifact-1"


def test_auto_init_retrievers_initializes_chunk_retriever(monkeypatch):
    import src.corpus.store as store_mod
    import src.corpus.search.deduper as deduper_mod
    import src.corpus.search.reranker as reranker_mod
    import src.corpus.search.retrievers.chunk_retriever as chunk_retriever_mod

    class FakeRepo:
        def __init__(self):
            self._chunk_store = None

        def connect(self):
            self._chunk_store = "fake-chunk-store"

    class FakeDeduper:
        pass

    class FakeReranker:
        def __init__(self, model=None, **kwargs):
            self.model = model
            self.kwargs = kwargs

    class FakeChunkRetriever:
        def __init__(self, chunk_store=None, milvus_client=None, embedding_model=None):
            self.chunk_store = chunk_store
            self.milvus_client = milvus_client
            self.embedding_model = embedding_model

    monkeypatch.setattr(store_mod, "CorpusRepository", FakeRepo)
    monkeypatch.setattr(deduper_mod, "PaperDeduper", FakeDeduper)
    monkeypatch.setattr(reranker_mod, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(chunk_retriever_mod, "ChunkRetriever", FakeChunkRetriever)

    runner = RagEvalRunner()
    runner._auto_init_retrievers()

    assert isinstance(runner._retriever, FakeRepo)
    assert isinstance(runner._deduper, FakeDeduper)
    assert isinstance(runner._reranker, FakeReranker)
    assert runner._reranker.model is None
    assert isinstance(runner._chunk_retriever, FakeChunkRetriever)
    assert runner._chunk_retriever.chunk_store == "fake-chunk-store"


def test_dict_to_case_infers_gold_evidence_paper_id():
    runner = RagEvalRunner()

    case = runner._dict_to_case(
        {
            "case_id": "scifact-1",
            "query": "test query",
            "gold_papers": [
                {
                    "title": "Paper A",
                    "canonical_id": "doc-1",
                    "arxiv_id": "",
                    "expected_rank": 0,
                }
            ],
            "gold_evidence": [
                {
                    "paper_title": "Paper A",
                    "expected_section": "Results",
                    "text_hint": "important finding",
                    "sub_question_id": "sq-1",
                    "expected_support_type": "claim_support",
                }
            ],
            "gold_claims": [],
        }
    )

    assert case.gold_evidence[0].paper_id == "doc-1"
    assert case.gold_evidence[0].text == "important finding"


def test_project_grounding_citations_uses_retrieved_papers_and_chunks():
    class Paper:
        canonical_id = "doc-1"
        primary_doc_id = "doc-1"
        title = "Paper A"
        dedup_info = type("DedupInfo", (), {"source_refs": ["https://example.com/paper-a"]})()

    case = RagEvalCase(
        case_id="grounding-1",
        query="test",
        gold_claims=[
            GoldClaim(
                claim_text="claim one",
                supported_by_paper="doc-1",
                supported_by_evidence_section="results",
            )
        ],
    )
    chunk = EvidenceChunk(
        chunk_id="chunk-1",
        paper_id="doc-1",
        canonical_id="doc-1",
        text="result text",
        section="Results",
    )

    runner = RagEvalRunner()
    records = runner._project_grounding_citations(case, [Paper()], [chunk])

    assert len(records) == 1
    assert records[0]["reachable"] is True
    assert records[0]["support_status"] == "supported"
    assert records[0]["url"] == "https://example.com/paper-a"


def test_project_grounding_citations_matches_chunk_support_type():
    class Paper:
        canonical_id = "doc-1"
        primary_doc_id = "doc-1"
        title = "Paper A"
        dedup_info = type("DedupInfo", (), {"source_refs": []})()

    case = RagEvalCase(
        case_id="grounding-2",
        query="test",
        gold_claims=[
            GoldClaim(
                claim_text="claim two",
                supported_by_paper="doc-1",
                supported_by_evidence_section="claim_support",
            )
        ],
    )
    chunk = EvidenceChunk(
        chunk_id="chunk-2",
        paper_id="doc-1",
        canonical_id="doc-1",
        text="support text",
        section="Abstract",
        support_type="claim_support",
    )

    runner = RagEvalRunner()
    records = runner._project_grounding_citations(case, [Paper()], [chunk])

    assert records[0]["support_status"] == "supported"
