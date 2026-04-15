"""
Corpus Ingest Pipeline — Module 1 统一流水线。

统一入口：ingest(sources) → list[StandardizedDocument]

流程：
  SourceInput[]
    ↓
LoaderDispatcher → ParsedDocument[]
    ↓
Parser (metadata extraction)
    ↓
Normalizer (text + metadata)
    ↓
Canonicalizer (canonical_id 归并)
    ↓
StandardizedDocument[]
    ↓
持久化（写入 PostgreSQL）
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from src.corpus.ingest.canonicalize import Canonicalizer
from src.corpus.ingest.loaders import (
    ArxivSourceInput,
    LoaderDispatcher,
    LocalPdfSourceInput,
    OnlineUrlSourceInput,
    SourceInput,
)
from src.corpus.ingest.normalizers import MetadataNormalizer, TextNormalizer
from src.corpus.ingest.parsers import MetadataExtractor
from src.corpus.models import (
    StandardizedDocument,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ingest Result
# ---------------------------------------------------------------------------


class IngestResult:
    """ingest() 返回结果，包含成功文档和错误信息。"""

    def __init__(
        self,
        documents: list[StandardizedDocument],
        errors: list[dict],
    ):
        self.documents = documents
        self.errors = errors

    @property
    def successful(self) -> list[StandardizedDocument]:
        return [d for d in self.documents if d.ingest_status == "ready"]

    @property
    def failed(self) -> list[StandardizedDocument]:
        return [d for d in self.documents if d.ingest_status == "failed"]

    def __repr__(self) -> str:
        return (
            f"IngestResult(successful={len(self.successful)}, "
            f"failed={len(self.failed)}, errors={len(self.errors)})"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class IngestPipeline:
    """
    Module 1 统一流水线。

    将 SourceInput[] 转换为 StandardizedDocument[]。
    """

    def __init__(self, db_session=None):
        self._canonicalizer = Canonicalizer(db_session=db_session)
        self._text_normalizer = TextNormalizer()
        self._metadata_normalizer = MetadataNormalizer()
        self._metadata_extractor = MetadataExtractor()

    def run(self, sources: list[SourceInput]) -> IngestResult:
        """
        执行 ingest pipeline。

        Args:
            sources: SourceInput 列表（ArxivSourceInput / LocalPdfSourceInput / OnlineUrlSourceInput）

        Returns:
            IngestResult：包含成功文档列表和错误详情
        """
        documents: list[StandardizedDocument] = []
        errors: list[dict] = []

        for source in sources:
            doc, error = self._process_one(source)
            if doc:
                documents.append(doc)
            if error:
                errors.append(error)

        return IngestResult(documents=documents, errors=errors)

    def _process_one(
        self, source: SourceInput
    ) -> tuple[Optional[StandardizedDocument], Optional[dict]]:
        """
        处理单个 SourceInput。
        """
        start = time.time()
        warnings: list[str] = []

        try:
            # Step 1: Loader → ParsedDocument
            parsed = LoaderDispatcher.load(source)
            warnings.extend(parsed.warnings)

            # Step 2: Metadata Extraction
            meta = self._metadata_extractor.extract(
                parsed.raw_metadata, parsed.page_texts
            )
            if meta.get("arxiv_id"):
                parsed.source_ref.external_id = meta["arxiv_id"]

            # Step 3: Normalize
            norm_meta = self._metadata_normalizer.normalize(
                title=meta.get("title", ""),
                authors=meta.get("authors", []),
                year=meta.get("year"),
                venue=meta.get("venue"),
            )
            if not norm_meta.title:
                warnings.append("标题提取失败，使用默认值")
                norm_meta.title = "Unknown Title"

            norm_text, text_warnings = self._text_normalizer.normalize(
                parsed.extracted_text, parsed.page_texts
            )
            warnings.extend(text_warnings)

            # Step 4: Build CanonicalKey
            canonical_key = self._canonicalizer.build_key(
                title=norm_meta.title,
                authors=norm_meta.authors,
                year=norm_meta.year,
                doi=norm_meta.doi,
                arxiv_id=norm_meta.arxiv_id,
                venue=norm_meta.venue,
            )

            # Step 5: Canonicalize（查 DB，决定归并）
            doc = StandardizedDocument(
                doc_id=parsed.doc_id,
                workspace_id=None,  # 后续阶段补充
                source_ref=parsed.source_ref,
                title=norm_meta.title,
                authors=norm_meta.authors,
                year=norm_meta.year,
                venue=norm_meta.venue,
                doi=norm_meta.doi,
                arxiv_id=norm_meta.arxiv_id,
                abstract=norm_meta.abstract,
                normalized_text=norm_text,
                ingest_status="normalized",
                parse_quality_score=parsed.parse_quality_score,
                canonical_key=canonical_key,
                warnings=warnings,
            )

            doc = self._canonicalizer.canonicalize_document(doc)
            doc.updated_at = time.time()
            doc.ingest_status = "ready"

            elapsed_ms = (time.time() - start) * 1000
            logger.info(
                f"[Ingest] {parsed.source_ref.source_type.value} "
                f"'{doc.title[:40]}' → canonical_id={doc.canonical_id} "
                f"({elapsed_ms:.0f}ms)"
            )

            return doc, None

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            logger.error(f"[Ingest] 处理 source 失败：{e}（{elapsed_ms:.0f}ms）")
            return None, {
                "source": str(source),
                "error": str(e),
                "elapsed_ms": elapsed_ms,
            }


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


_pipeline: IngestPipeline | None = None


def ingest(
    sources: list[SourceInput],
    db_session=None,
) -> IngestResult:
    """
    便捷入口：给定 SourceInput 列表，返回 StandardizedDocument 列表。

    Examples:
        >>> from src.corpus.ingest import ArxivSourceInput, ingest
        >>> result = ingest([ArxivSourceInput(arxiv_id="1706.03762")])
        >>> for doc in result.successful:
        ...     print(doc.title, doc.canonical_id)
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestPipeline(db_session=db_session)
    return _pipeline.run(sources)
