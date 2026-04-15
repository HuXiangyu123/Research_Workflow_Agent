"""Corpus 数据模型 — 文档对象定义。"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Source Type
# ---------------------------------------------------------------------------


class SourceType(str, Enum):
    """文档来源类型。"""

    ARXIV = "arxiv"
    LOCAL_PDF = "local_pdf"
    ONLINE_URL = "online_url"
    LOCAL_FOLDER = "local_folder"


# ---------------------------------------------------------------------------
# SourceRef — 来源记录
# ---------------------------------------------------------------------------


@dataclass
class SourceRef:
    """表示一个具体来源，不是论文本体。"""

    source_id: str
    source_type: SourceType
    uri_or_path: str
    external_id: Optional[str] = None  # arXiv ID / DOI
    version: Optional[str] = None  # arXiv v1/v2, conference/journal
    parse_quality: float = 0.0  # 0.0~1.0
    ingest_status: str = "pending"  # pending / processed / failed
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.source_id:
            # 自动生成：hash(source_type + uri)
            raw = f"{self.source_type.value}::{self.uri_or_path}"
            self.source_id = hashlib.md5(raw.encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# ParsedDocument — Parser 层输出
# ---------------------------------------------------------------------------


@dataclass
class PageText:
    """按页分段的文本，保留结构信息。"""

    page_num: int  # 1-indexed
    text: str
    char_start: int  # 在整篇文档中的字符偏移
    char_end: int


@dataclass
class ParsedDocument:
    """Parser 层的输出：原始解析结果 + 质量评分。"""

    source_ref: SourceRef
    extracted_text: str  # 拼接后的全文
    page_texts: list[PageText]  # 按页分段
    raw_metadata: dict  # 原始元数据（可能不干净）
    parse_quality_score: float = 0.0  # 0.0~1.0
    warnings: list[str] = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        return self.source_ref.source_id


# ---------------------------------------------------------------------------
# NormalizedMetadata — Normalizer 层输出
# ---------------------------------------------------------------------------


@dataclass
class NormalizedMetadata:
    """标准化后的元数据。"""

    title: str
    authors: list[str]  # 拆分后的作者列表
    year: Optional[int]
    venue: Optional[str]
    abstract: Optional[str]
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    raw_warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CanonicalKey — 论文身份归并 key
# ---------------------------------------------------------------------------


@dataclass
class CanonicalKey:
    """用于判断"是否是同一篇论文"的标准 key。"""

    normalized_title: str
    first_author_surname: str
    year: int
    # 加分信号（用于置信度判断）
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    venue: Optional[str] = None

    def to_key_string(self) -> str:
        """生成归并用的字符串 key。"""
        return (
            f"{self.normalized_title.lower().strip()}::"
            f"{self.first_author_surname.lower().strip()}::"
            f"{self.year}"
        )

    def to_hash(self) -> str:
        return hashlib.sha256(self.to_key_string().encode()).hexdigest()[:16]

    def confidence_bonus(self) -> float:
        """基于外部 ID 返回置信度加分。"""
        bonus = 0.0
        if self.doi:
            bonus += 0.5
        if self.arxiv_id:
            bonus += 0.5
        if self.venue:
            bonus += 0.1
        return bonus


# ---------------------------------------------------------------------------
# StandardizedDocument — Module 1 最终输出
# ---------------------------------------------------------------------------


@dataclass
class StandardizedDocument:
    """
    Module 1 的最终输出：标准化后的内部文档对象。

    包含 canonical_id（表示"它属于哪篇论文"）和 source_ref（保留来源）。
    一个 canonical paper 可以挂多个 source_ref。
    """

    doc_id: str  # hash(source_ref.source_id)
    workspace_id: Optional[str]
    canonical_id: Optional[str] = None  # 论文身份（归并后生成）
    source_ref: Optional[SourceRef] = None
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    abstract: Optional[str] = None
    normalized_text: str = ""  # 清洗后的正文
    ingest_status: str = "pending"  # pending / normalized / canonicalized / ready
    parse_quality_score: float = 0.0
    canonical_key: Optional[CanonicalKey] = None
    warnings: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.doc_id and self.source_ref:
            self.doc_id = self.source_ref.source_id


# ---------------------------------------------------------------------------
# Chunk Models (Module 2)
# ---------------------------------------------------------------------------

# Token 估算：简单按 chars * 0.25 近似
_CHUNK_TOKEN_RATIO = 0.25


@dataclass
class ChunkSpan:
    """块的位置信息。"""

    page_start: int = 1          # 1-indexed
    page_end: int = 1
    char_start: int = 0
    char_end: int = 0


@dataclass
class CoarseChunk:
    """
    粗粒度块，服务于 paper-level retrieval。

    特征：
    - 按 section 边界切分（保留主题完整性）
    - 目标 token 数 500~1500
    - 无 overlap
    """

    chunk_id: str
    doc_id: str
    canonical_id: Optional[str]
    section: str                           # e.g. "abstract", "introduction", "methods"
    section_level: int = 1                 # 章节层级（1=一级, 2=二级...）
    page_start: int = 1
    page_end: int = 1
    char_start: int = 0
    char_end: int = 0
    text: str = ""
    token_count: int = 0
    order: int = 0                         # 在文档中的顺序
    meta_info: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.sha256(
                f"{self.doc_id}:{self.section}:{self.order}".encode()
            ).hexdigest()[:24]
        if not self.token_count and self.text:
            self.token_count = int(len(self.text) * _CHUNK_TOKEN_RATIO)


@dataclass
class FineChunk:
    """
    细粒度块，服务于 evidence retrieval。

    特征：
    - 在 coarse chunk 内按 paragraph 切分
    - 目标 token 数 100~300
    - 可带轻量 overlap（10~15%）
    """

    chunk_id: str
    doc_id: str
    canonical_id: Optional[str]
    parent_coarse_chunk_id: str           # 父 coarse chunk
    section: str
    page_start: int = 1
    page_end: int = 1
    char_start: int = 0
    char_end: int = 0
    text: str = ""
    token_count: int = 0
    order: int = 0                         # 在 coarse chunk 内的顺序
    meta_info: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.sha256(
                f"{self.parent_coarse_chunk_id}:{self.section}:{self.order}:{self.char_start}".encode()
            ).hexdigest()[:24]
        if not self.token_count and self.text:
            self.token_count = int(len(self.text) * _CHUNK_TOKEN_RATIO)


# ---------------------------------------------------------------------------
# Legacy models (backward compatibility)
# ---------------------------------------------------------------------------

# 保留旧的 DocumentMeta / Chunk / Span 供现有代码引用
# 新代码应使用 StandardizedDocument / SourceRef
from pydantic import BaseModel, Field


class Span(BaseModel):
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None


class DocumentMeta(BaseModel):
    doc_id: str = Field(..., description="文档唯一ID，通常是 hash(source_id)")
    source_id: str = Field(..., description="原始ID，如 arxiv:1706.03762")
    source_uri: str = Field(..., description="来源URI，URL或文件路径")
    title: str = Field(..., description="文档标题")
    authors: Optional[str] = Field(None, description="作者列表")
    published_date: Optional[str] = Field(None, description="发布日期")
    summary: Optional[str] = Field(None, description="摘要")
    added_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    status: str = Field("pending", description="状态: pending/processed/failed")
    error: Optional[str] = Field(None, description="错误信息")

    @staticmethod
    def generate_id(source_id: str) -> str:
        return hashlib.md5(source_id.encode()).hexdigest()


class Chunk(BaseModel):
    chunk_id: str = Field(..., description="Chunk唯一ID，hash(doc_id + text)")
    doc_id: str = Field(..., description="所属文档ID")
    source_id: str = Field(..., description="冗余字段，方便检索")
    source_uri: str = Field(..., description="冗余字段，方便检索")
    title: str = Field(..., description="冗余字段，方便检索")
    section_path: list[str] = Field(default_factory=list)
    span: Span = Field(default_factory=Span)
    text: str = Field(..., description="文本内容")
    text_hash: str = Field(..., description="文本hash")
    len_chars: int = Field(..., description="字符长度")

    @staticmethod
    def generate_id(doc_id: str, text: str) -> str:
        return hashlib.md5((doc_id + text).encode()).hexdigest()
