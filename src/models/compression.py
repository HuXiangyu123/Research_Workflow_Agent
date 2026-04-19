"""Compression 相关的数据模型 — 用于报告生成的上下文压缩。

设计文档：docs/features_oncoming/context-compression-for-report-generation.md

压缩管道：
    extract_node → extract_compression_node → draft_node

Layer 1: Taxonomy + CompressedCards（结构化摘要）
Layer 2: Per-Section Evidence Pool（分 section 的 evidence 分配）
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TaxonomyCategory(BaseModel):
    """论文分类的单个类别。"""

    name: str = Field(description="类别名称，如 'Benchmark驱动' / 'Agent架构'")
    description: str = Field(description="类别的描述")
    papers: list[str] = Field(
        default_factory=list,
        description="属于该类别的论文标题列表",
    )
    key_characteristics: list[str] = Field(
        default_factory=list,
        description="该类别的核心特征",
    )
    shared_insights: list[str] = Field(
        default_factory=list,
        description="该类别内跨论文的共同发现",
    )
    conflicts: list[str] = Field(
        default_factory=list,
        description="类别内论文间的冲突结论",
    )


class Taxonomy(BaseModel):
    """论文分类结构 — 将论文按技术路线/子领域组织。"""

    categories: list[TaxonomyCategory] = Field(
        default_factory=list,
        description="所有分类",
    )
    cross_category_themes: list[str] = Field(
        default_factory=list,
        description="跨分类主题（最重要）",
    )
    timeline: list[str] = Field(
        default_factory=list,
        description="技术发展时间线",
    )
    key_papers: list[str] = Field(
        default_factory=list,
        description="必引用的关键论文标题",
    )


class CompressedCard(BaseModel):
    """压缩后的论文卡片 — 从 ~1500 chars abstract 压缩到 ~300 chars。"""

    title: str = Field(description="论文标题")
    arxiv_id: str = Field(default="", description="arXiv ID")
    core_claim: str = Field(description="核心发现（一句话）")
    method_type: str = Field(description="方法类型")
    key_result: str = Field(default="", description="关键数值结果")
    role_in_taxonomy: str = Field(default="", description="在分类中的角色")
    connections: list[str] = Field(
        default_factory=list,
        description="与其他论文的关系/对比",
    )


class PoolEntry(BaseModel):
    """分配给某个 section 的单篇论文 evidence。"""

    card: CompressedCard
    allocated_chars: int = Field(
        default=300,
        description="分配给该论文的字符数",
    )
    focus_aspect: str = Field(
        default="",
        description="该 section 关注该论文的哪个方面",
    )


class EvidencePool(BaseModel):
    """某个 section 的 evidence pool。"""

    section: str = Field(description="section 名称")
    token_budget: int = Field(
        default=5000,
        description="该 section 的 token 预算",
    )
    papers: list[PoolEntry] = Field(
        default_factory=list,
        description="分配给该 section 的论文",
    )


class CompressionResult(BaseModel):
    """extract_compression_node 的输出结果。"""

    taxonomy: Taxonomy = Field(
        default_factory=Taxonomy,
        description="论文分类结构",
    )
    compressed_cards: list[CompressedCard] = Field(
        default_factory=list,
        description="压缩后的论文卡片列表",
    )
    evidence_pools: dict[str, EvidencePool] = Field(
        default_factory=dict,
        description="每个 section 的 evidence pool，key 为 section 名称",
    )
    compression_stats: dict = Field(
        default_factory=dict,
        description="压缩统计信息",
    )
