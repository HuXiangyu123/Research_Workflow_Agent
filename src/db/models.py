"""SQLAlchemy ORM 模型（PostgreSQL）。"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.engine import Base

# pgvector 是可选的；如果未安装则 VectorChunk 无法使用
try:
    from pgvector.sqlalchemy import Vector as _V
    _HAS_PGVECTOR = True
except ImportError:
    _V = None  # type: ignore[assignment]
    _HAS_PGVECTOR = False

if TYPE_CHECKING:
    pass


class Document(Base):
    """文档元信息表。"""

    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    canonical_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("canonical_papers.canonical_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    source_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    source_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    authors: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    published_date: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    added_at: Mapped[float] = mapped_column(Float, default=time.time)
    updated_at: Mapped[float] = mapped_column(Float, default=time.time)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parse_quality: Mapped[float] = mapped_column(Float, default=0.0)

    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )
    canonical_paper: Mapped[Optional["CanonicalPaper"]] = relationship(
        "CanonicalPaper", back_populates="documents"
    )

    __table_args__ = (
        Index("ix_documents_source_status", "source_id", "status"),
    )

    @staticmethod
    def generate_id(source_id: str) -> str:
        return hashlib.md5(source_id.encode()).hexdigest()


class Chunk(Base):
    """文档切分块表。"""

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False
    )
    section_path: Mapped[Optional[list[str]]] = mapped_column(
        PG_ARRAY(String), nullable=True
    )
    page_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    page_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    char_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    char_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    len_chars: Mapped[int] = mapped_column(Integer, nullable=False)

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_doc_id", "doc_id"),
        Index("ix_chunks_text_hash", "text_hash"),
    )

    @staticmethod
    def generate_id(doc_id: str, text: str) -> str:
        return hashlib.md5((doc_id + text).encode()).hexdigest()


class VectorChunk(Base):
    """带向量表示的 Chunk 表（使用 pgvector）。用于语义检索。

    仅在安装 pgvector 后可用。
    创建表后需要额外执行：
        ALTER TABLE vector_chunks ADD COLUMN embedding vector(384);
    （维度需与 SentenceTransformer 模型一致，如 all-MiniLM-L6-v2 为 384）
    """

    __tablename__ = "vector_chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True
    )
    doc_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.doc_id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    source_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    source_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)

    # embedding 字段在 pgvector 安装后通过 migration 添加（见 docstring）
    # 此处仅声明接口，不作为 ORM 列以避免维度硬编码


# ---------------------------------------------------------------------------
# Module 1: Canonical Papers & Source Refs（论文身份归并表）
# ---------------------------------------------------------------------------


class CanonicalPaper(Base):
    """论文身份归并表。每行代表一篇唯一的论文（canonical paper）。"""

    __tablename__ = "canonical_papers"

    canonical_id: Mapped[str] = mapped_column(
        String(64), primary_key=True
    )
    # 归并 key（normalized title + author + year）
    canonical_key: Mapped[str] = mapped_column(
        String(512), unique=True, nullable=False, index=True
    )
    # 主要来源元数据
    primary_title: Mapped[str] = mapped_column(String(512), nullable=False)
    primary_authors: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    primary_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    primary_venue: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    # 外部 ID（用于高置信度归并）
    doi: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, index=True)
    arxiv_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    # 版本组（同论文不同版本）
    version_group: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, index=True
    )
    # 统计
    source_count: Mapped[int] = mapped_column(Integer, default=1)
    # 时间戳
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    updated_at: Mapped[float] = mapped_column(Float, default=time.time, onupdate=time.time)

    # 关系
    source_refs: Mapped[list["SourceRef"]] = relationship(
        "SourceRef", back_populates="canonical_paper", cascade="all, delete-orphan"
    )
    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="canonical_paper"
    )

    __table_args__ = (
        # 注意：doi/arxiv_id/version_group 已在 mapped_column(index=True) 创建同名索引
        # 此处仅保留额外复合索引
        Index("ix_canonical_papers_version_group", "version_group"),
    )


class SourceRef(Base):
    """来源记录表。表示一个具体来源（arXiv / PDF / URL），不是论文本体。"""

    __tablename__ = "source_refs"

    source_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    canonical_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("canonical_papers.canonical_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # 来源类型
    source_type: Mapped[str] = mapped_column(
        String(32), nullable=False, index=True
    )  # arxiv / local_pdf / online_url / local_folder
    # 来源路径
    uri_or_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    # 外部 ID
    external_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    # 版本信息
    version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    # 解析质量
    parse_quality: Mapped[float] = mapped_column(Float, default=0.0)
    # 处理状态
    ingest_status: Mapped[str] = mapped_column(
        String(32), default="pending"
    )  # pending / processed / failed
    # 错误信息
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # 时间戳
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    updated_at: Mapped[float] = mapped_column(Float, default=time.time)

    # 关系
    canonical_paper: Mapped[Optional["CanonicalPaper"]] = relationship(
        "CanonicalPaper", back_populates="source_refs"
    )

    __table_args__ = (
        Index("ix_source_refs_canonical_status", "canonical_id", "ingest_status"),
        Index("ix_source_refs_external_id", "external_id"),
    )


# ---------------------------------------------------------------------------
# Module 2: Chunk Tables（coarse_chunks / fine_chunks）
# ---------------------------------------------------------------------------


class CoarseChunk(Base):
    """粗粒度块表。服务于 paper-level retrieval。"""

    __tablename__ = "coarse_chunks"

    coarse_chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.doc_id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    canonical_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, index=True,
    )
    section: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    section_level: Mapped[int] = mapped_column(Integer, default=1)
    page_start: Mapped[int] = mapped_column(Integer, default=1)
    page_end: Mapped[int] = mapped_column(Integer, default=1)
    char_start: Mapped[int] = mapped_column(Integer, default=0)
    char_end: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    order_idx: Mapped[int] = mapped_column(Integer, default=0)
    meta_info: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)

    __table_args__ = (
        Index("ix_coarse_doc_section", "doc_id", "section"),
        Index("ix_coarse_canonical", "canonical_id"),
    )


class FineChunk(Base):
    """细粒度块表。服务于 evidence retrieval。"""

    __tablename__ = "fine_chunks"

    fine_chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.doc_id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    canonical_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, index=True,
    )
    coarse_chunk_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("coarse_chunks.coarse_chunk_id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    section: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    page_start: Mapped[int] = mapped_column(Integer, default=1)
    page_end: Mapped[int] = mapped_column(Integer, default=1)
    char_start: Mapped[int] = mapped_column(Integer, default=0)
    char_end: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    order_idx: Mapped[int] = mapped_column(Integer, default=0)
    meta_info: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)

    __table_args__ = (
        Index("ix_fine_coarse", "coarse_chunk_id"),
        Index("ix_fine_doc_section", "doc_id", "section"),
        Index("ix_fine_canonical", "canonical_id"),
    )
