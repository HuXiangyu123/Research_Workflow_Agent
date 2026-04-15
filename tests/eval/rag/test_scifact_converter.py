"""SciFact converter 单元测试。"""
from __future__ import annotations

import pytest
import json

from scripts.scifact.convert_scifact import (
    extract_sentence_text,
    scifact_claim_to_rag_case,
    LABEL_TO_SUPPORT_TYPE,
    MAX_SENTENCES_PER_DOC,
)


class TestLabelMapping:
    """extract_sentence_text 测试（使用 dict）。"""

    def test_extract_single_sentence(self):
        entry = {
            "title": "Test Paper",
            "abstract": ["Background.", "Methods.", "Results."]
        }
        text = extract_sentence_text(entry, [1])
        assert "Methods" in text

    def test_extract_multiple_sentences(self):
        entry = {
            "title": "Test Paper",
            "abstract": ["Background.", "Methods.", "Results."]
        }
        text = extract_sentence_text(entry, [0, 2])
        assert "Background" in text
        assert "Results" in text

    def test_out_of_bounds_ignored(self):
        entry = {
            "title": "Test Paper",
            "abstract": ["Background.", "Methods."]
        }
        text = extract_sentence_text(entry, [0, 5])
        assert "Background" in text

    def test_empty_abstract(self):
        entry = {"title": "Test", "abstract": []}
        text = extract_sentence_text(entry, [0])
        assert text == ""


class TestScifactClaimToRagCase:
    """scifact_claim_to_rag_case 测试。"""

    def test_basic_conversion(self):
        claim_entry = {
            "id": 123,
            "claim": "Test scientific claim.",
            "cited_doc_ids": ["999"],
            "evidence": {
                "999": [{"sentences": [0, 1], "label": "SUPPORT"}]
            },
        }
        corpus = {
            "999": {
                "title": "Test Paper",
                "abstract": ["Background.", "Main results."],
            }
        }

        case = scifact_claim_to_rag_case(claim_entry, corpus, "scifact")
        assert case is not None
        assert case["case_id"] == "scifact-123"
        assert case["query"] == "Test scientific claim."
        assert case["source"] == "scifact"
        assert len(case["gold_papers"]) == 1
        assert case["gold_papers"][0]["canonical_id"] == "999"
        assert len(case["gold_evidence"]) == 1
        assert case["gold_evidence"][0]["expected_support_type"] == "claim_support"

    def test_contradict_label(self):
        claim_entry = {
            "id": 456,
            "claim": "Another claim.",
            "cited_doc_ids": ["888"],
            "evidence": {
                "888": [{"sentences": [2], "label": "CONTRADICT"}]
            },
        }
        corpus = {
            "888": {
                "title": "Paper",
                "abstract": ["A.", "B.", "C."],
            }
        }

        case = scifact_claim_to_rag_case(claim_entry, corpus, "scifact")
        assert case["gold_evidence"][0]["expected_support_type"] == "limitation"

    def test_empty_claim_skipped(self):
        claim_entry = {"id": 1, "claim": "", "cited_doc_ids": [], "evidence": {}}
        case = scifact_claim_to_rag_case(claim_entry, {}, "scifact")
        assert case is None

    def test_no_evidence_returns_none(self):
        claim_entry = {
            "id": 2,
            "claim": "Claim without evidence.",
            "cited_doc_ids": [],
            "evidence": {},
        }
        case = scifact_claim_to_rag_case(claim_entry, {}, "scifact")
        assert case is None

    def test_missing_corpus_doc_skipped(self):
        claim_entry = {
            "id": 3,
            "claim": "Test.",
            "cited_doc_ids": ["999"],
            "evidence": {"999": [{"sentences": [0], "label": "SUPPORT"}]},
        }
        corpus = {}  # 缺少 doc
        case = scifact_claim_to_rag_case(claim_entry, corpus, "scifact")
        assert case is None  # gold_evidence 为空会返回 None
