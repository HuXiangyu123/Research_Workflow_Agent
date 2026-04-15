"""RAG Eval Models — 评测数据模型定义（模块 7）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Gold 标准 ────────────────────────────────────────────────────────────────


@dataclass
class GoldPaper:
    """单条 gold paper 标准。"""
    title: str = ""
    canonical_id: str = ""
    arxiv_id: str = ""
    expected_rank: int = 0
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None


@dataclass
class GoldEvidence:
    """单条 gold evidence 标准。"""
    paper_title: str = ""
    expected_section: str = ""
    text_hint: str = ""
    sub_question_id: str = ""
    expected_support_type: str = ""


@dataclass
class GoldClaim:
    """单条 gold claim 标准。"""
    claim_text: str = ""
    supported_by_paper: str = ""
    supported_by_evidence_section: str = ""


class RetrievalStrategy:
    """检索策略配置。"""
    def __init__(
        self,
        name: str,
        recall_top_k: int = 100,
        keyword_weight: float = 0.5,
        dense_weight: float = 0.5,
        rerank_enabled: bool = False,
        fusion_weights_rrf: tuple = (0.5, 0.5),
        fusion_weights_rerank: tuple = (0.5, 0.5),
        evidence_recall_enabled: bool = False,
        evidence_top_k: int = 50,
    ):
        self.name = name
        self.recall_top_k = recall_top_k
        self.keyword_weight = keyword_weight
        self.dense_weight = dense_weight
        self.rerank_enabled = rerank_enabled
        self.fusion_weights_rrf = fusion_weights_rrf
        self.fusion_weights_rerank = fusion_weights_rerank
        self.evidence_recall_enabled = evidence_recall_enabled
        self.evidence_top_k = evidence_top_k


# ── 预定义策略 ────────────────────────────────────────────────────────────────

STRATEGIES: dict[str, RetrievalStrategy] = {
    "keyword_only": RetrievalStrategy(
        name="keyword_only",
        recall_top_k=100,
        keyword_weight=1.0,
        dense_weight=0.0,
        rerank_enabled=False,
    ),
    "dense_only": RetrievalStrategy(
        name="dense_only",
        recall_top_k=100,
        keyword_weight=0.0,
        dense_weight=1.0,
        rerank_enabled=False,
    ),
    "hybrid_basic": RetrievalStrategy(
        name="hybrid_basic",
        recall_top_k=100,
        keyword_weight=0.5,
        dense_weight=0.5,
        rerank_enabled=False,
    ),
    "hybrid_rerank": RetrievalStrategy(
        name="hybrid_rerank",
        recall_top_k=200,
        keyword_weight=0.5,
        dense_weight=0.5,
        rerank_enabled=True,
        fusion_weights_rrf=(0.6, 0.4),
        fusion_weights_rerank=(0.4, 0.6),
    ),
    "hybrid_max_rrf": RetrievalStrategy(
        name="hybrid_max_rrf",
        recall_top_k=200,
        keyword_weight=0.5,
        dense_weight=0.5,
        rerank_enabled=False,
        fusion_weights_rrf=(0.7, 0.3),
    ),
    "evidence_hybrid": RetrievalStrategy(
        name="evidence_hybrid",
        recall_top_k=100,
        keyword_weight=0.5,
        dense_weight=0.5,
        rerank_enabled=True,
        evidence_recall_enabled=True,
        evidence_top_k=50,
    ),
}


# ── 兼容枚举（保留用于类型检查） ────────────────────────────────────────────────

class RetrievalStrategyEnum(str, Enum):
    """检索策略枚举（兼容旧代码）。"""
    KEYWORD_ONLY = "keyword_only"
    DENSE_ONLY = "dense_only"
    HYBRID_BASIC = "hybrid_basic"
    HYBRID_RERANK = "hybrid_rerank"
    HYBRID_MAX_RRF = "hybrid_max_rrf"


@dataclass
class RagEvalCase:
    """
    单条 RAG 评测 case。

    对应一个子问题（sub_question）的 gold 标准。
    """
    case_id: str = ""
    # 问题
    query: str = ""
    sub_questions: list[str] = field(default_factory=list)

    # Gold 标准
    gold_papers: list[GoldPaper] = field(default_factory=list)
    gold_evidence: list[GoldEvidence] = field(default_factory=list)
    gold_claims: list[GoldClaim] = field(default_factory=list)

    # 配置参数
    recall_top_k: int = 100
    rerank_top_m: int = 50
    evidence_top_k: int = 50

    # 元数据
    source: str = "manual"
    notes: str = ""

    def gold_paper_ids(self) -> set[str]:
        """返回 gold paper id 集合。"""
        return {p.canonical_id or p.arxiv_id for p in self.gold_papers
                if p.canonical_id or p.arxiv_id}

    def gold_paper_titles(self) -> set[str]:
        """返回 gold paper title 集合（小写）。"""
        return {p.title.lower() for p in self.gold_papers if p.title}


# ── Layer 1: Paper Retrieval Metrics ────────────────────────────────────────


@dataclass
class PaperRetrievalMetrics:
    """Layer 1：论文召回指标。"""
    # 召回率
    paper_recall_10: float = 0.0  # Top-10 recall
    paper_recall_50: float = 0.0  # Top-50 recall
    paper_recall_100: float = 0.0  # Top-100 recall

    # 排序质量
    paper_mrr: float = 0.0  # Mean Reciprocal Rank
    paper_ndcg_10: float = 0.0
    paper_ndcg_50: float = 0.0
    paper_map_10: float = 0.0  # MAP@10

    # 统计
    total_gold_papers: int = 0
    retrieved_gold_papers: int = 0


# ── Layer 2: Paper Ranking Metrics ───────────────────────────────────────────


@dataclass
class PaperRankingMetrics:
    """Layer 2：论文排序指标（dedup + rerank 后的排序质量）。"""
    # 排序质量
    ranking_ndcg_10: float = 0.0
    ranking_ndcg_50: float = 0.0
    ranking_mrr: float = 0.0

    # Dedup 效果
    dedup_precision: float = 1.0  # 去重后精确率（理想情况为 1.0）
    deduped_count: int = 0  # 去重后数量
    original_count: int = 0  # 原始候选数量

    # Rerank 改进
    rerank_improvement: float = 0.0  # 相对于 RRF baseline 的提升


# ── Layer 3: Evidence Retrieval Metrics ──────────────────────────────────────


@dataclass
class EvidenceRetrievalMetrics:
    """Layer 3：证据块召回指标。"""
    # 召回率
    evidence_recall_10: float = 0.0
    evidence_recall_25: float = 0.0
    evidence_recall_50: float = 0.0

    # 精确率
    evidence_precision_10: float = 0.0
    evidence_precision_25: float = 0.0

    # Section 命中率
    support_span_hit_rate: float = 0.0  # evidence section 命中 gold expected_section 的比率

    # Type 准确率
    evidence_type_accuracy: float = 0.0  # 命中的 evidence 中，type 正确的比率

    # Coverage 分析
    method_coverage: float = 0.0  # gold 包含 method 时，是否有 predicted 命中
    result_coverage: float = 0.0  # gold 包含 result 时，是否有 predicted 命中

    # 统计
    total_gold_evidence: int = 0
    retrieved_gold_evidence: int = 0


# ── Layer 4: Citation / Grounding Metrics ───────────────────────────────────


@dataclass
class CitationGroundingMetrics:
    """Layer 4：引用可达性和声明支撑指标。"""
    # Citation 可达性
    citation_reachability: float = 0.0  # 可达引用 / 总引用

    # Claim 支撑
    supported_claim_rate: float = 0.0  # 有支撑的 claim / 总 claim
    unsupported_claim_rate: float = 0.0  # 无支撑的 claim / 总 claim

    # Coverage gap
    coverage_gap_count: int = 0  # 未回答的子问题数

    # 综合分数
    grounding_score: float = 0.0  # (supported - unsupported) / total

    # 统计
    total_citations: int = 0
    reachable_citations: int = 0
    total_claims: int = 0
    supported_claims: int = 0


# ── 评测结果 ────────────────────────────────────────────────────────────────


@dataclass
class EvalCaseResult:
    """单条评测 case 的完整结果。"""
    case_id: str = ""
    strategy: str = "hybrid_basic"

    # 四层指标
    paper_retrieval: Optional[PaperRetrievalMetrics] = None
    paper_ranking: Optional[PaperRankingMetrics] = None
    evidence_retrieval: Optional[EvidenceRetrievalMetrics] = None
    citation_grounding: Optional[CitationGroundingMetrics] = None

    # 预测结果（用于后续分析）
    predicted_paper_ids: list[str] = field(default_factory=list)
    predicted_evidence_sections: list[str] = field(default_factory=list)

    # 元数据
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def is_success(self) -> bool:
        """评测是否成功（无错误）。"""
        return len(self.errors) == 0

    def overall_recall(self) -> float:
        """整体召回率（论文 + evidence 的平均召回）。"""
        paper_recall = (
            self.paper_retrieval.paper_recall_10
            if self.paper_retrieval
            else 0.0
        )
        evidence_recall = (
            self.evidence_retrieval.evidence_recall_10
            if self.evidence_retrieval
            else 0.0
        )
        return (paper_recall + evidence_recall) / 2


@dataclass
class StrategyMetrics:
    """单个策略的聚合指标。"""
    strategy_name: str = ""

    # 统计
    total_cases: int = 0
    success_cases: int = 0
    failure_cases: int = 0

    # Layer 1 聚合
    avg_paper_recall_10: float = 0.0
    avg_paper_recall_50: float = 0.0
    avg_paper_mrr: float = 0.0
    avg_paper_ndcg_10: float = 0.0

    # Layer 2 聚合
    avg_ranking_ndcg_10: float = 0.0
    avg_ranking_mrr: float = 0.0

    # Layer 3 聚合
    avg_evidence_recall_10: float = 0.0
    avg_evidence_recall_25: float = 0.0
    avg_evidence_precision_25: float = 0.0
    avg_support_span_hit_rate: float = 0.0
    avg_evidence_type_accuracy: float = 0.0

    # Layer 4 聚合
    avg_citation_reachability: float = 0.0
    avg_grounding_score: float = 0.0

    def aggregate(self, results: list[EvalCaseResult]) -> None:
        """从 case 结果列表聚合指标。"""
        if not results:
            return
        self.total_cases = len(results)
        self.success_cases = sum(1 for r in results if r.is_success())
        self.failure_cases = self.total_cases - self.success_cases

        success_results = [r for r in results if r.is_success()]
        if not success_results:
            return

        pr = [r.paper_retrieval for r in success_results if r.paper_retrieval]
        er = [r.evidence_retrieval for r in success_results if r.evidence_retrieval]
        rr = [r.paper_ranking for r in success_results if r.paper_ranking]
        cg = [r.citation_grounding for r in success_results if r.citation_grounding]

        if pr:
            self.avg_paper_recall_10 = _mean([p.paper_recall_10 for p in pr])
            self.avg_paper_recall_50 = _mean([p.paper_recall_50 for p in pr])
            self.avg_paper_mrr = _mean([p.paper_mrr for p in pr])
            self.avg_paper_ndcg_10 = _mean([p.paper_ndcg_10 for p in pr])
        if er:
            self.avg_evidence_recall_10 = _mean([e.evidence_recall_10 for e in er])
            self.avg_evidence_recall_25 = _mean([e.evidence_recall_25 for e in er])
            self.avg_evidence_precision_25 = _mean([e.evidence_precision_25 for e in er])
            self.avg_support_span_hit_rate = _mean([e.support_span_hit_rate for e in er])
            self.avg_evidence_type_accuracy = _mean([e.evidence_type_accuracy for e in er])
        if rr:
            self.avg_ranking_ndcg_10 = _mean([r.ranking_ndcg_10 for r in rr])
            self.avg_ranking_mrr = _mean([r.ranking_mrr for r in rr])
        if cg:
            self.avg_citation_reachability = _mean([c.citation_reachability for c in cg])
            self.avg_grounding_score = _mean([c.grounding_score for c in cg])


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    """计算平均值。"""
    return sum(values) / len(values) if values else 0.0


@dataclass
class StrategyComparisonReport:
    """多策略对比报告。"""
    # 元数据
    strategies: list[str] = field(default_factory=list)
    winner: str = ""
    generated_at: str = ""

    # 各策略指标
    per_strategy_metrics: dict[str, StrategyMetrics] = field(default_factory=dict)

    # 各指标最优策略
    best_per_metric: dict[str, str] = field(default_factory=dict)


@dataclass
class RagEvalReport:
    """完整 RAG 评测报告。"""
    # 基础信息
    report_id: str = ""
    description: str = ""
    generated_at: str = ""

    # 评测统计
    total_cases: int = 0
    total_errors: int = 0
    success_rate: float = 0.0
    total_duration_ms: float = 0.0

    # 整体指标（跨策略平均）
    overall_paper_recall_50: float = 0.0
    overall_paper_mrr: float = 0.0
    overall_evidence_recall_25: float = 0.0
    overall_grounding_score: float = 0.0

    # Case 级结果
    case_results: list[EvalCaseResult] = field(default_factory=list)
    failed_cases: list[str] = field(default_factory=list)

    # 策略比较
    strategy_comparison: Optional[StrategyComparisonReport] = None

    def best_strategy(self) -> Optional[str]:
        """按 paper_recall_50 返回最佳策略名称。"""
        if not self.strategy_comparison:
            return None
        return self.strategy_comparison.winner
