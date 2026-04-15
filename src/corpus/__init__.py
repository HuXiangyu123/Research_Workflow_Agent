"""Corpus 模块 — 文档入库、标准化、层级切块。"""

from src.corpus.models import (
    CanonicalKey,
    ChunkSpan,
    CoarseChunk,
    FineChunk,
    PageText,
    SourceRef,
    SourceType,
    StandardizedDocument,
)
from src.corpus.ingest.pipeline import ingest
from src.corpus.ingest.chunking_pipeline import chunk_document

__all__ = [
    # Data models
    "SourceRef",
    "SourceType",
    "StandardizedDocument",
    "CanonicalKey",
    "CoarseChunk",
    "FineChunk",
    "ChunkSpan",
    "PageText",
    # Pipeline
    "ingest",
    "chunk_document",
]
