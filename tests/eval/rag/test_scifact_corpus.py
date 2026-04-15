"""Tests for SciFact benchmark corpus ingestion helpers."""

from __future__ import annotations

import json

from src.eval.rag.scifact_corpus import (
    build_scifact_chunks,
    build_scifact_document,
    load_case_gold_doc_ids,
)


def test_load_case_gold_doc_ids_dedupes(tmp_path):
    case_a = {
        "case_id": "a",
        "query": "q1",
        "gold_papers": [
            {"title": "Paper A", "canonical_id": "1001", "arxiv_id": "", "expected_rank": 0},
            {"title": "Paper B", "canonical_id": "1002", "arxiv_id": "", "expected_rank": 0},
        ],
        "gold_evidence": [],
        "gold_claims": [],
    }
    case_b = {
        "case_id": "b",
        "query": "q2",
        "gold_papers": [
            {"title": "Paper A", "canonical_id": "1001", "arxiv_id": "", "expected_rank": 0},
        ],
        "gold_evidence": [],
        "gold_claims": [],
    }
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(case_a, ensure_ascii=False),
                json.dumps(case_b, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_case_gold_doc_ids(path) == ["1001", "1002"]


def test_build_scifact_document_and_chunks():
    corpus_entry = {
        "title": "Paper A",
        "abstract": [
            "Sentence one.",
            "Sentence two.",
            "Sentence three.",
            "Sentence four.",
        ],
    }

    doc = build_scifact_document("1001", corpus_entry)
    coarse_chunks, fine_chunks = build_scifact_chunks(doc, corpus_entry)

    assert doc.doc_id == "1001"
    assert doc.canonical_id == "1001"
    assert doc.abstract == "Sentence one. Sentence two. Sentence three. Sentence four."
    assert len(coarse_chunks) == 1
    assert coarse_chunks[0].section == "abstract"
    assert len(fine_chunks) == 4
    assert fine_chunks[0].doc_id == "1001"
    assert fine_chunks[0].canonical_id == "1001"
    assert fine_chunks[0].section == "abstract"
    assert fine_chunks[1].section == "introduction"
    assert "Sentence one." in fine_chunks[0].text
