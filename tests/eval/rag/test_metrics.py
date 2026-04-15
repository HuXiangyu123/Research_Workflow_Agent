"""metrics.py 单元测试。"""
from __future__ import annotations

import pytest

from src.eval.rag.metrics import (
    ndcg,
    mrr,
    ap,
    map_k,
    compute_paper_retrieval_metrics,
    compute_evidence_retrieval_metrics,
)


class MockPaper:
    def __init__(self, pid: str, title: str = ""):
        self.canonical_id = pid
        self.primary_doc_id = pid
        self.title = title


class MockCase:
    def __init__(self, case_id: str, gold_ids: set[str], gold_titles: set[str] = None):
        self.case_id = case_id
        self.gold_ids = gold_ids
        self.gold_titles = gold_titles or set()

    def gold_paper_ids(self):
        return self.gold_ids

    def gold_paper_titles(self):
        return self.gold_titles

    @property
    def gold_evidence(self):
        return []


class TestNDCG:
    def test_ndcg_perfect_ranking(self):
        """完美排序（所有 relevant 都在前面）。"""
        relevance = [1, 1, 1, 0, 0]
        assert ndcg(relevance, k=5) == 1.0

    def test_ndcg_no_relevant(self):
        """无相关项。"""
        relevance = [0, 0, 0]
        assert ndcg(relevance, k=3) == 0.0

    def test_ndcg_k_truncate(self):
        """k 截断正确。"""
        relevance = [1, 0, 1, 0, 0]
        ndcg_full = ndcg(relevance)
        ndcg_3 = ndcg(relevance, k=3)
        assert ndcg_3 <= ndcg_full

    def test_ndcg_partial(self):
        """部分相关。"""
        relevance = [1, 0, 0, 1]
        score = ndcg(relevance, k=4)
        assert 0.0 < score < 1.0


class TestMRR:
    def test_mrr_first_relevant(self):
        """首个相关项在第 1 位。"""
        assert mrr([1, 0, 0]) == 1.0

    def test_mrr_third_relevant(self):
        """首个相关项在第 3 位。"""
        assert mrr([0, 0, 1]) == 1.0 / 3.0

    def test_mrr_no_relevant(self):
        """无相关项。"""
        assert mrr([0, 0, 0]) == 0.0


class TestAP:
    def test_ap_perfect(self):
        assert ap([1, 1, 1]) == 1.0

    def test_ap_none(self):
        assert ap([0, 0, 0]) == 0.0

    def test_ap_partial(self):
        score = ap([1, 0, 1])
        assert 0.0 < score < 1.0


class TestPaperRetrievalMetrics:
    def test_recall_all_found(self):
        """所有 gold papers 都在 top-10 中。"""
        case = MockCase("test-1", gold_ids={"p1", "p2"})
        papers = [MockPaper("p1"), MockPaper("p2")]
        metrics = compute_paper_retrieval_metrics(papers, case)
        assert metrics.paper_recall_10 == 1.0

    def test_recall_partial(self):
        """只有一半的 gold papers 被召回。"""
        case = MockCase("test-2", gold_ids={"p1", "p2"})
        papers = [MockPaper("p1"), MockPaper("p3")]
        metrics = compute_paper_retrieval_metrics(papers, case)
        assert metrics.paper_recall_10 == 0.5
        assert metrics.retrieved_gold_papers == 1

    def test_recall_none(self):
        """没有召回任何 gold paper。"""
        case = MockCase("test-3", gold_ids={"p1"})
        papers = [MockPaper("p2"), MockPaper("p3")]
        metrics = compute_paper_retrieval_metrics(papers, case)
        assert metrics.paper_recall_10 == 0.0

    def test_mrr_calculation(self):
        """MRR 计算正确。"""
        case = MockCase("test-4", gold_ids={"p2"})
        papers = [MockPaper("p1"), MockPaper("p2"), MockPaper("p3")]
        metrics = compute_paper_retrieval_metrics(papers, case)
        assert metrics.paper_mrr == 0.5  # p2 在第 2 位

    def test_ndcg_10(self):
        case = MockCase("test-5", gold_ids={"p1"})
        papers = [MockPaper("p1")]
        metrics = compute_paper_retrieval_metrics(papers, case)
        assert metrics.paper_ndcg_10 == 1.0
