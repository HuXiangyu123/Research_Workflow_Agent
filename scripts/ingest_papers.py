#!/usr/bin/env python3
"""
批量论文入库脚本。

从 arXiv 获取论文 → 标准化 → 分块 → 生成向量 → 索引到 PostgreSQL + Milvus。

Usage:
    # 单个或多个 arXiv ID
    python scripts/ingest_papers.py --arxiv 1706.03762 2005.14165 1810.04805

    # 从文件读取（每行一个 arXiv ID）
    python scripts/ingest_papers.py --from-file arxiv_ids.txt

    # Dry run（只打印，不入库）
    python scripts/ingest_papers.py --arxiv 1706.03762 --dry-run

    # 详细输出
    python scripts/ingest_papers.py --arxiv 1706.03762 -v
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Setup path — 使脚本可以独立运行
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_sentence_encoder():
    """
    延迟加载 sentence encoder。

    使用 all-MiniLM-L6-v2 模型生成文本向量。
    该模型输出 384 维向量，适合快速语义检索场景。
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence encoder: all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info(f"Encoder loaded, device={model.device}")
    return model


def _generate_embeddings(
    chunks: list,
    model,
    text_field: str = "text",
    id_field: str = "chunk_id",
) -> dict[str, list[float]]:
    """
    为 chunks 生成向量 embedding。

    Args:
        chunks: chunk 对象列表
        model: SentenceTransformer 模型
        text_field: 文本字段名（CoarseChunk/FineChunk 的 text 属性）
        id_field: chunk ID 字段名

    Returns:
        {chunk_id: embedding_vector} 字典
    """
    # 从 chunk 对象提取文本
    texts = []
    for c in chunks:
        raw = getattr(c, text_field, "") or ""
        # 兼容 text 可能是 dict（如 FineChunk）的情况
        texts.append(raw if isinstance(raw, str) else str(raw))

    # 批量生成向量
    embeddings = model.encode(texts, show_progress_bar=False)

    # 转换为 {id: list[float]} 格式
    result = {}
    for chunk, emb in zip(chunks, embeddings):
        chunk_id = getattr(chunk, id_field, None)
        if chunk_id:
            result[chunk_id] = emb.tolist()

    return result


# ---------------------------------------------------------------------------
# Core ingestion logic
# ---------------------------------------------------------------------------


def ingest_papers(
    arxiv_ids: list[str],
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """
    批量入库主函数。

    流程：
        1. 通过 arXiv API 拉取 PDF 并解析元数据
        2. 调用 chunk_document() 进行结构化分块
        3. 使用 SentenceTransformer 生成向量
        4. 调用 CorpusRepository.index_document() 写入 PostgreSQL + Milvus

    Args:
        arxiv_ids: arXiv ID 列表（如 "1706.03762"）
        dry_run: True 则只打印操作，不写入存储
        verbose: True 则启用 DEBUG 日志级别

    Returns:
        统计字典，包含 total / successful / failed / coarse_chunks /
        fine_chunks / elapsed_s
    """
    from src.corpus.ingest import ArxivSourceInput, chunk_document, ingest
    from src.corpus.models import CoarseChunk, FineChunk
    from src.corpus.store import CorpusRepository

    start_time = time.time()
    total = len(arxiv_ids)
    logger.info(f"开始入库 {total} 篇论文...")

    # ── Stage 1: Fetch & Normalize ─────────────────────────────────────────

    logger.info("Stage 1/4: Fetching papers from arXiv...")
    sources = [ArxivSourceInput(arxiv_id=aid.strip()) for aid in arxiv_ids]
    ingest_result = ingest(sources)

    logger.info(
        f"  成功: {len(ingest_result.successful)}, "
        f"失败: {len(ingest_result.failed)}, "
        f"错误: {len(ingest_result.errors)}"
    )

    if not ingest_result.successful:
        logger.error("没有成功获取任何论文，退出")
        return {"total": total, "successful": 0, "failed": total}

    if dry_run:
        for doc in ingest_result.successful:
            logger.info(f"  [DRY RUN] Would ingest: {doc.doc_id} - {doc.title}")
        return {
            "total": total,
            "successful": len(ingest_result.successful),
            "failed": total - len(ingest_result.successful),
        }

    # ── Stage 2: Chunking ───────────────────────────────────────────────────

    logger.info("Stage 2/4: Chunking documents...")
    coarse_chunks_all: list[CoarseChunk] = []
    fine_chunks_all: list[FineChunk] = []

    for doc in ingest_result.successful:
        chunking = chunk_document(doc)
        coarse_chunks_all.extend(chunking.coarse_chunks)
        fine_chunks_all.extend(chunking.fine_chunks)

    logger.info(
        f"  Coarse chunks: {len(coarse_chunks_all)}, "
        f"Fine chunks: {len(fine_chunks_all)}"
    )

    # ── Stage 3: Embeddings ─────────────────────────────────────────────────

    logger.info("Stage 3/4: Generating embeddings...")
    encoder = _load_sentence_encoder()

    coarse_embeddings = _generate_embeddings(
        coarse_chunks_all, encoder,
        text_field="text",
        id_field="coarse_chunk_id",
    )
    fine_embeddings = _generate_embeddings(
        fine_chunks_all, encoder,
        text_field="text",
        id_field="fine_chunk_id",
    )

    logger.info(
        f"  Coarse embeddings: {len(coarse_embeddings)}, "
        f"Fine embeddings: {len(fine_embeddings)}"
    )

    # ── Stage 4: Index ──────────────────────────────────────────────────────

    logger.info("Stage 4/4: Indexing to PostgreSQL + Milvus...")
    repo = CorpusRepository()
    repo.connect()
    logger.info("  Connected to CorpusRepository")

    indexed_docs = 0
    for doc in ingest_result.successful:
        # 按 doc_id 过滤属于当前文档的 chunks
        doc_coarse = [c for c in coarse_chunks_all if c.doc_id == doc.doc_id]
        doc_fine = [f for f in fine_chunks_all if f.doc_id == doc.doc_id]

        # 按 chunk_id 过滤 embeddings
        coarse_ids = {c.coarse_chunk_id for c in doc_coarse}
        fine_ids = {f.fine_chunk_id for f in doc_fine}
        doc_coarse_emb = {k: v for k, v in coarse_embeddings.items() if k in coarse_ids}
        doc_fine_emb = {k: v for k, v in fine_embeddings.items() if k in fine_ids}

        # 合并 embedding 字典（coarse + fine 一起传给 index_document）
        repo.index_document(
            doc,
            coarse_chunks=doc_coarse,
            fine_chunks=doc_fine,
            embeddings={**doc_coarse_emb, **doc_fine_emb},
        )

        indexed_docs += 1
        logger.info(f"  Indexed: {doc.doc_id} - {doc.title[:50]}")

    elapsed = time.time() - start_time
    logger.info(
        f"完成！耗时 {elapsed:.1f}s，"
        f"平均 {elapsed / total:.1f}s/篇"
    )

    return {
        "total": total,
        "successful": indexed_docs,
        "failed": total - indexed_docs,
        "coarse_chunks": len(coarse_chunks_all),
        "fine_chunks": len(fine_chunks_all),
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main():
    parser = argparse.ArgumentParser(
        description="批量论文入库工具 — arXiv → 标准化 → 分块 → 向量 → 索引",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 单个或多个 arXiv ID
  python scripts/ingest_papers.py --arxiv 1706.03762 2005.14165

  # 从文件读取（每行一个 ID）
  python scripts/ingest_papers.py --from-file arxiv_ids.txt

  # Dry run（只打印，不入库）
  python scripts/ingest_papers.py --arxiv 1706.03762 --dry-run

  # 详细输出
  python scripts/ingest_papers.py --arxiv 1706.03762 -v
        """,
    )
    parser.add_argument(
        "--arxiv", "-a", nargs="+",
        help="arXiv ID（如 1706.03762），可指定多个",
    )
    parser.add_argument(
        "--from-file", "-f", type=Path,
        help="arXiv ID 列表文件（每行一个）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印操作，不实际写入存储",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="启用 DEBUG 日志级别",
    )
    args = parser.parse_args()

    # 参数校验
    if not args.arxiv and not args.from_file:
        parser.error("请提供 --arxiv 或 --from-file")

    arxiv_ids: list[str] = list(args.arxiv) if args.arxiv else []

    if args.from_file:
        if not args.from_file.exists():
            logger.error(f"文件不存在：{args.from_file}")
            sys.exit(1)
        with args.from_file.open("r", encoding="utf-8") as f:
            arxiv_ids += [line.strip() for line in f if line.strip()]

    if not arxiv_ids:
        logger.error("没有有效的 arXiv ID")
        sys.exit(1)

    # 去重并保持顺序
    arxiv_ids = list(dict.fromkeys(arxiv_ids))
    logger.info(f"待入库 {len(arxiv_ids)} 篇论文")

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    result = ingest_papers(
        arxiv_ids=arxiv_ids,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    # 打印汇总
    print("\n=== 入库结果 ===")
    print(f"总数:       {result['total']}")
    print(f"成功:       {result['successful']}")
    print(f"失败:       {result['failed']}")
    if result.get("coarse_chunks") is not None:
        print(f"Coarse:     {result['coarse_chunks']}")
        print(f"Fine:       {result['fine_chunks']}")
    print(f"耗时:       {result.get('elapsed_s', 0):.1f}s")


if __name__ == "__main__":
    _main()
