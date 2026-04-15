"""
Module 2 Chunking Pipeline — 串联 Structure Detection + Coarse + Fine + Linker。

统一入口：chunk_document(doc) → ChunkingResult
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from src.corpus.ingest.chunkers import StructureDetector
from src.corpus.ingest.chunk_linker import ChunkLinker
from src.corpus.ingest.coarse_chunker import CoarseChunker
from src.corpus.ingest.fine_chunker import FineChunker
from src.corpus.models import CoarseChunk, FineChunk, PageText, StandardizedDocument

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


class ChunkingResult:
    """
    chunk_document() 返回结果。

    使用 @property 确保 coarse_count/fine_count 始终从列表派生。
    """

    def __init__(
        self,
        coarse_chunks: list[CoarseChunk] | None = None,
        fine_chunks: list[FineChunk] | None = None,
        structure_warnings: list[str] | None = None,
        elapsed_ms: float = 0.0,
        errors: list[str] | None = None,
    ):
        self.coarse_chunks = coarse_chunks or []
        self.fine_chunks = fine_chunks or []
        self.structure_warnings = structure_warnings or []
        self.elapsed_ms = elapsed_ms
        self.errors = errors or []

    @property
    def coarse_count(self) -> int:
        return len(self.coarse_chunks)

    @property
    def fine_count(self) -> int:
        return len(self.fine_chunks)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ChunkingPipeline:
    """
    Module 2 统一流水线。

    将 StandardizedDocument 转换为 CoarseChunk[] + FineChunk[]。
    """

    def __init__(
        self,
        min_section_paragraphs: int = 1,
        db_session: "Session | None" = None,
    ):
        """
        Args:
            min_section_paragraphs: section 最少段落数
            db_session: 可选 DB session
        """
        self._detector = StructureDetector(min_section_paragraphs=min_section_paragraphs)
        self._coarse_chunker = CoarseChunker()
        self._fine_chunker = FineChunker()
        self._linker = ChunkLinker()
        self._db_session = db_session

    def chunk(
        self,
        doc: StandardizedDocument,
        page_texts: list[PageText] | None = None,
    ) -> ChunkingResult:
        """
        执行 chunking pipeline。

        Args:
            doc: StandardizedDocument（Module 1 输出）
            page_texts: 可选，保留页结构的 PageText 列表

        Returns:
            ChunkingResult
        """
        start = time.time()
        result = ChunkingResult()
        errors: list[str] = []

        try:
            # Step 1: Structure Detection
            structure = self._detector.detect(doc, page_texts or [])
            result.structure_warnings.extend(structure.warnings)

            # Step 2: Coarse Chunking
            coarse_chunks = self._coarse_chunker.chunk(
                structure, doc_id=doc.doc_id, canonical_id=doc.canonical_id
            )
            if not coarse_chunks:
                errors.append("Coarse chunking 生成 0 个块，可能是文本过短")

            result.coarse_chunks = coarse_chunks

            # Step 3: Fine Chunking
            fine_chunks = self._fine_chunker.chunk(coarse_chunks)
            if not fine_chunks:
                errors.append("Fine chunking 生成 0 个块")

            result.fine_chunks = fine_chunks

            # Step 4: Link & Persist
            if coarse_chunks and fine_chunks:
                self._linker.link_and_persist(
                    coarse_chunks, fine_chunks, self._db_session
                )
            elif coarse_chunks:
                self._linker.link_and_persist(
                    coarse_chunks, [], self._db_session
                )

            result.errors = errors

        except Exception as e:
            logger.error(f"[ChunkingPipeline] 处理失败：{e}")
            result.errors.append(str(e))

        result.elapsed_ms = (time.time() - start) * 1000

        logger.info(
            f"[ChunkingPipeline] doc={doc.doc_id} "
            f"→ coarse={result.coarse_count}, fine={result.fine_count} "
            f"({result.elapsed_ms:.0f}ms)"
        )

        return result


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

_pipeline: ChunkingPipeline | None = None


def chunk_document(
    doc: StandardizedDocument,
    page_texts: list[PageText] | None = None,
    db_session=None,
) -> ChunkingResult:
    """
    便捷入口：对单个 StandardizedDocument 执行完整 chunking。

    Examples:
        >>> from src.corpus.ingest import ingest, ArxivSourceInput
        >>> from src.corpus.chunking import chunk_document
        >>> result = ingest([ArxivSourceInput(arxiv_id="1706.03762")])
        >>> doc = result.successful[0]
        >>> chunking_result = chunk_document(doc)
        >>> print(f"Coarse: {chunking_result.coarse_count}, Fine: {chunking_result.fine_count}")
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = ChunkingPipeline(db_session=db_session)
    return _pipeline.chunk(doc, page_texts)
