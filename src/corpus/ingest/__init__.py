"""Corpus Ingest 子模块 — 统一文档入库流水线。"""

from src.corpus.ingest.canonicalize import Canonicalizer
from src.corpus.ingest.chunkers import (
    DetectedSection,
    PaperStructure,
    Paragraph,
    StructureDetector,
)
from src.corpus.ingest.chunking_pipeline import (
    ChunkingPipeline,
    ChunkingResult,
    chunk_document,
)
from src.corpus.ingest.chunk_linker import ChunkLinker
from src.corpus.ingest.coarse_chunker import CoarseChunker
from src.corpus.ingest.fine_chunker import FineChunker
from src.corpus.ingest.loaders import (
    ArxivLoader,
    ArxivSourceInput,
    LocalPdfLoader,
    LocalPdfSourceInput,
    OnlineUrlLoader,
    OnlineUrlSourceInput,
    LoaderDispatcher,
    SourceInput,
)
from src.corpus.ingest.normalizers import MetadataNormalizer, TextNormalizer
from src.corpus.ingest.parsers import HTMLParser, MetadataExtractor, PDFParser
from src.corpus.ingest.pipeline import ingest, IngestPipeline, IngestResult

__all__ = [
    # Pipeline 入口
    "IngestPipeline",
    "IngestResult",
    "ingest",
    "ChunkingPipeline",
    "ChunkingResult",
    "chunk_document",
    # Loaders
    "ArxivLoader",
    "ArxivSourceInput",
    "LocalPdfLoader",
    "LocalPdfSourceInput",
    "OnlineUrlLoader",
    "OnlineUrlSourceInput",
    "LoaderDispatcher",
    "SourceInput",
    # Parsers
    "PDFParser",
    "HTMLParser",
    "MetadataExtractor",
    # Normalizers
    "TextNormalizer",
    "MetadataNormalizer",
    # Canonicalizer
    "Canonicalizer",
    # Chunking
    "StructureDetector",
    "CoarseChunker",
    "FineChunker",
    "ChunkLinker",
    "DetectedSection",
    "PaperStructure",
    "Paragraph",
]
