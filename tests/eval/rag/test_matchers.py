"""matchers.py 单元测试。"""
from __future__ import annotations

import pytest

from src.eval.rag.matchers import (
    token_overlap_ratio,
    text_similarity,
    section_overlap,
    normalize_section,
    loose_match,
    strict_match,
    paper_match,
    get_matcher,
)


class MockChunk:
    def __init__(self, paper_id: str = "", section: str = "", text: str = ""):
        self.paper_id = paper_id
        self.canonical_id = paper_id
        self.doc_id = paper_id
        self.section = section
        self.text = text


class MockGoldEvidence:
    def __init__(self, paper_id: str = "", expected_section: str = "",
                 text: str = ""):
        self.paper_id = paper_id
        self.expected_section = expected_section
        self.text = text


class MockGoldPaper:
    def __init__(self, paper_id: str = "", canonical_id: str = "",
                 arxiv_id: str = "", title: str = ""):
        self.paper_id = paper_id
        self.canonical_id = canonical_id
        self.arxiv_id = arxiv_id
        self.title = title


class TestTextSimilarity:
    def test_identical(self):
        assert text_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert text_similarity("hello", "world") == 0.0

    def test_partial_overlap(self):
        score = text_similarity("the quick brown fox", "quick brown")
        assert 0.0 < score < 1.0


class TestSectionOverlap:
    def test_exact_match(self):
        assert section_overlap("Method", "Method") is True

    def test_substring(self):
        assert section_overlap("Experimental Results", "Results") is True

    def test_no_overlap(self):
        assert section_overlap("Method", "Abstract") is False

    def test_empty(self):
        assert section_overlap("", "Method") is False


class TestNormalizeSection:
    def test_method(self):
        assert normalize_section("Methodology") == "method"

    def test_result(self):
        assert normalize_section("Experimental Results") == "result"

    def test_unknown(self):
        assert normalize_section("Custom Section") == "custom section"


class TestLooseMatch:
    def test_paper_match(self):
        chunk = MockChunk(paper_id="paper-1", section="Method", text="proposed method")
        gold = MockGoldEvidence(paper_id="paper-1", expected_section="Method", text="proposed method")
        assert loose_match(chunk, gold) is True

    def test_paper_mismatch(self):
        chunk = MockChunk(paper_id="paper-2", section="Method")
        gold = MockGoldEvidence(paper_id="paper-1", expected_section="Method")
        assert loose_match(chunk, gold) is False

    def test_section_overlap(self):
        chunk = MockChunk(paper_id="paper-1", section="Experimental Results")
        gold = MockGoldEvidence(paper_id="paper-1", expected_section="Results")
        assert loose_match(chunk, gold) is True


class TestStrictMatch:
    def test_loose_fail(self):
        """宽松匹配失败时，严格匹配也失败。"""
        chunk = MockChunk(paper_id="paper-2", section="Method")
        gold = MockGoldEvidence(paper_id="paper-1", expected_section="Method")
        assert strict_match(chunk, gold) is False

    def test_loose_pass_strict_pass(self):
        """宽松匹配通过 + text 重叠率高。"""
        chunk = MockChunk(
            paper_id="paper-1",
            section="Method",
            text="We propose a new architecture for question answering using attention",
        )
        gold = MockGoldEvidence(
            paper_id="paper-1",
            expected_section="Method",
            text="propose architecture attention mechanism",
        )
        assert strict_match(chunk, gold) is True


class TestPaperMatch:
    def test_paper_id_match(self):
        p = MockGoldPaper(paper_id="p1")
        assert paper_match(p, MockGoldPaper(paper_id="p1")) is True

    def test_no_match(self):
        p = MockGoldPaper(paper_id="p1")
        assert paper_match(p, MockGoldPaper(paper_id="p2")) is False


class TestMatcherFactory:
    def test_loose_mode(self):
        fn = get_matcher("loose")
        assert fn is loose_match

    def test_strict_mode(self):
        fn = get_matcher("strict")
        assert fn is strict_match

    def test_unknown_defaults_to_loose(self):
        fn = get_matcher("unknown")
        assert fn is loose_match
