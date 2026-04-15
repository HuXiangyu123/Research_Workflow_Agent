"""SciFact benchmark corpus ingestion helpers for RAG eval."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scripts.scifact.convert_scifact import load_scifact_data
from src.corpus.models import CanonicalKey, CoarseChunk, FineChunk, SourceRef, SourceType, StandardizedDocument
from src.corpus.store import CorpusRepository
from src.corpus.store.vector_index import VectorRecord
from src.db.engine import init_db
from src.eval.rag.runner import RagEvalRunner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class SciFactIngestStats:
    """Benchmark corpus ingestion summary."""

    requested_docs: int = 0
    ingested_docs: int = 0
    skipped_existing: int = 0
    missing_docs: int = 0
    coarse_chunks: int = 0
    fine_chunks: int = 0
    vectors_written: int = 0
    missing_doc_ids: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.missing_doc_ids is None:
            self.missing_doc_ids = []


def _normalize_sentences(sentences: Iterable[str]) -> list[str]:
    return [" ".join(str(s).split()) for s in sentences if str(s).strip()]


def _sentence_char_starts(sentences: list[str]) -> list[int]:
    starts: list[int] = []
    cursor = 0
    for idx, sentence in enumerate(sentences):
        starts.append(cursor)
        cursor += len(sentence)
        if idx < len(sentences) - 1:
            cursor += 1
    return starts


def _infer_scifact_section(sentence_index: int, sentence_count: int) -> str:
    """Mirror the heuristic used when converting SciFact gold evidence."""
    if sentence_index == 0:
        return "abstract"
    if sentence_index <= 2:
        return "introduction"
    if sentence_index <= sentence_count * 0.5:
        return "method"
    if sentence_index <= sentence_count * 0.8:
        return "results"
    return "discussion"


def _make_chunk_id(doc_id: str, chunk_type: str, idx: int) -> str:
    """生成一致的 chunk_id 格式。"""
    return f"scifact_{doc_id}_{chunk_type}_{idx}"


# ---------------------------------------------------------------------------
# Gold Paper ID 加载
# ---------------------------------------------------------------------------


def load_case_gold_doc_ids(case_source: str | Path = "regression") -> list[str]:
    """Load unique gold paper ids from a RagEval case source."""
    runner = RagEvalRunner()
    cases = runner.load_cases(case_source)
    doc_ids: set[str] = set()
    for case in cases:
        for paper in case.gold_papers:
            paper_id = paper.canonical_id or paper.arxiv_id
            if paper_id:
                doc_ids.add(str(paper_id))
    return sorted(doc_ids)


# ---------------------------------------------------------------------------
# Document / Chunk 构建
# ---------------------------------------------------------------------------


def build_scifact_document(doc_id: str, corpus_entry: dict) -> StandardizedDocument:
    """Convert a SciFact corpus entry into a StandardizedDocument."""
    title = (corpus_entry.get("title") or "").strip() or f"SciFact Paper {doc_id}"
    abstract_sentences = _normalize_sentences(corpus_entry.get("abstract") or [])
    abstract_text = " ".join(abstract_sentences).strip()
    source_ref = SourceRef(
        source_id=f"scifact:{doc_id}",
        source_type=SourceType.ONLINE_URL,
        uri_or_path=f"scifact://corpus/{doc_id}",
        external_id=str(doc_id),
        parse_quality=1.0,
        ingest_status="processed",
    )
    canonical_key = CanonicalKey(
        normalized_title=title.lower(),
        first_author_surname="scifact",
        year=0,
    )
    return StandardizedDocument(
        doc_id=str(doc_id),
        workspace_id=None,
        canonical_id=str(doc_id),
        source_ref=source_ref,
        title=title,
        authors=[],
        year=None,
        venue="SciFact",
        abstract=abstract_text,
        normalized_text=abstract_text,
        ingest_status="ready",
        parse_quality_score=1.0,
        canonical_key=canonical_key,
        warnings=["benchmark_ingest: scifact abstract-only corpus"],
    )


def build_scifact_chunks(
    doc: StandardizedDocument,
    corpus_entry: dict,
    max_window_size: int = 3,
) -> tuple[list[CoarseChunk], list[FineChunk]]:
    """Build one coarse abstract chunk and sliding-window fine chunks."""
    sentences = _normalize_sentences(corpus_entry.get("abstract") or [])
    if not sentences:
        fallback_text = doc.abstract or doc.title
        sentences = [fallback_text] if fallback_text else []

    abstract_text = " ".join(sentences).strip()

    # 预生成 chunk_id（用于 Milvus 向量写入的一致性）
    coarse_chunk_id = _make_chunk_id(doc.doc_id, "coarse", 0)
    coarse_chunk = CoarseChunk(
        chunk_id=coarse_chunk_id,
        doc_id=doc.doc_id,
        canonical_id=doc.canonical_id,
        section="abstract",
        section_level=1,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(abstract_text),
        text=abstract_text,
        order=0,
        meta_info={
            "benchmark": "scifact",
            "sentence_count": len(sentences),
        },
    )

    fine_chunks: list[FineChunk] = []
    char_starts = _sentence_char_starts(sentences)
    for idx in range(len(sentences)):
        window_sentences = sentences[idx : idx + max_window_size]
        if not window_sentences:
            continue
        start = char_starts[idx]
        end = char_starts[idx + len(window_sentences) - 1] + len(window_sentences[-1])
        fine_chunks.append(
            FineChunk(
                chunk_id=_make_chunk_id(doc.doc_id, "fine", idx),
                doc_id=doc.doc_id,
                canonical_id=doc.canonical_id,
                parent_coarse_chunk_id=coarse_chunk_id,
                section=_infer_scifact_section(idx, len(sentences)),
                page_start=1,
                page_end=1,
                char_start=start,
                char_end=end,
                text=" ".join(window_sentences).strip(),
                order=idx,
                meta_info={
                    "benchmark": "scifact",
                    "sentence_start": idx,
                    "sentence_end": idx + len(window_sentences) - 1,
                    "window_size": len(window_sentences),
                },
            )
        )

    return [coarse_chunk], fine_chunks


# ---------------------------------------------------------------------------
# SciFact Corpus 入库（含 Qwen Embedding 生成）
# ---------------------------------------------------------------------------


def ingest_scifact_corpus(
    case_source: str | Path = "regression",
    data_split: str = "train",
    repo: CorpusRepository | None = None,
    skip_existing: bool = True,
) -> SciFactIngestStats:
    """
    Ingest all gold papers referenced by a RagEval case source.

    包含 Qwen text-embedding-v4 向量生成，写入 Milvus（1024维）。

    Args:
        case_source: RagEval case 文件（regression / scifact_full）
        data_split: SciFact 数据集划分（train / test）
        repo: 可选，外部传入的 CorpusRepository
        skip_existing: 跳过已入库的文档
    """
    init_db()
    target_doc_ids = load_case_gold_doc_ids(case_source)
    _, corpus_dict = load_scifact_data(data_split)

    owns_repo = repo is None
    if repo is None:
        repo = CorpusRepository()
        repo.connect()

    stats = SciFactIngestStats(requested_docs=len(target_doc_ids))

    # 加载 embedding client（Qwen API / 本地 SentenceTransformer）
    embedding_client = None
    try:
        from src.embeddings.client import get_embedding_client
        embedding_client = get_embedding_client()
        logger.info(f"[SciFactIngest] Embedding client: {embedding_client}")
    except Exception as e:
        logger.warning(f"[SciFactIngest] 无法加载 embedding client，跳过向量生成：{e}")

    for doc_id in target_doc_ids:
        if skip_existing and repo.get_document(str(doc_id)):
            stats.skipped_existing += 1
            continue

        corpus_entry = corpus_dict.get(str(doc_id))
        if not corpus_entry:
            stats.missing_docs += 1
            stats.missing_doc_ids.append(str(doc_id))
            continue

        doc = build_scifact_document(str(doc_id), corpus_entry)
        coarse_chunks, fine_chunks = build_scifact_chunks(doc, corpus_entry)

        # Step 1: 入库 PG（chunk_id 已预生成）
        result = repo.index_document(
            doc,
            coarse_chunks=coarse_chunks,
            fine_chunks=fine_chunks,
            embeddings=None,  # Milvus 向量在 Step 2 单独写入
        )

        if result.get("errors"):
            logger.warning(
                "[SciFactIngest] doc_id=%s indexing errors=%s",
                doc_id,
                result["errors"],
            )
            continue

        stats.ingested_docs += 1
        stats.coarse_chunks += len(coarse_chunks)
        stats.fine_chunks += len(fine_chunks)

        # Step 2: 生成 Qwen embedding 并写入 Milvus
        if embedding_client is not None:
            written = _write_scifact_vectors(
                repo, doc.doc_id, coarse_chunks, fine_chunks, embedding_client,
            )
            stats.vectors_written += written

    if owns_repo and getattr(repo, "_owns_session", False) and getattr(repo, "_db", None):
        repo._db.close()

    return stats


def _write_scifact_vectors(
    repo: CorpusRepository,
    doc_id: str,
    coarse_chunks: list[CoarseChunk],
    fine_chunks: list[FineChunk],
    embedding_client,
    batch_size: int = 10,
) -> int:
    """
    为 SciFact chunks 生成 Qwen embedding 并写入 Milvus。

    使用预生成的 chunk_id（scifact_{doc_id}_{type}_{idx}）作为向量 ID，
    保证 PG 和 Milvus 的一致性。

    Returns:
        写入 Milvus 的向量总数
    """
    written = 0

    for chunk_type, chunks in (("coarse", coarse_chunks), ("fine", fine_chunks)):
        if not chunks or not repo._vector_index:
            continue

        collection = "coarse_chunks" if chunk_type == "coarse" else "fine_chunks"
        texts = [c.text for c in chunks]

        # 分批生成 embedding（Qwen API 每批最多 10）
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_texts = [c.text for c in batch_chunks]

            # 过滤空文本
            valid = [(c, t) for c, t in zip(batch_chunks, batch_texts) if t.strip()]
            if not valid:
                continue

            valid_chunks, valid_texts = zip(*valid)
            try:
                embeddings = embedding_client.encode(list(valid_texts), show_progress_bar=False)
            except Exception as e:
                logger.warning(f"[_write_scifact_vectors] embedding 失败：{e}")
                continue

            records = [
                VectorRecord(
                    id=c.chunk_id,
                    doc_id=doc_id,
                    canonical_id=doc_id,
                    vector=emb.tolist(),
                    text=c.text,
                    section=c.section,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    token_count=len(c.text.split()),
                )
                for c, emb in zip(valid_chunks, embeddings)
            ]

            try:
                repo._vector_index.upsert(collection, records)
                written += len(records)
                logger.debug(f"[_write_scifact_vectors] {collection}: +{len(records)}")
            except Exception as e:
                logger.warning(f"[_write_scifact_vectors] upsert 失败：{e}")

    return written
