#!/usr/bin/env python3
"""
Milvus Reindex Script — 将现有 corpus 的向量从旧维度切换到 Qwen text-embedding-v4（1024维）。

工作流程：
    1. 读取 PostgreSQL 中的所有 coarse/fine chunks
    2. 用 Qwen API 生成 1024 维 embedding
    3. 删除旧 Milvus collection
    4. 创建新 collection（dim=1024）
    5. 批量写入新 embedding

Usage:
    # 标准重索引（自动确认）
    python scripts/reindex_milvus.py

    # 跳过确认（CI / 自动化场景）
    python scripts/reindex_milvus.py --yes

    # 只处理 fine chunks（跳过 coarse）
    python scripts/reindex_milvus.py --coarse-only

    # dry-run（只打印数量）
    python scripts/reindex_milvus.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Iterator

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    """加载 .env 环境变量。"""
    from dotenv import load_dotenv
    for candidate in [
        Path(__file__).parent.parent / ".env",
        Path.cwd() / ".env",
    ]:
        if candidate.exists():
            load_dotenv(candidate, override=False)
            logger.info(f"[Setup] .env loaded from {candidate}")
            return
    logger.warning("[Setup] .env not found, using environment variables only")


def _iter_chunks(
    session,
    chunk_type: str,
    batch_size: int = 100,
) -> Iterator[list[dict]]:
    """
    分批迭代 PostgreSQL 中的 chunks。

    Args:
        session: SQLAlchemy session
        chunk_type: "coarse" 或 "fine"
        batch_size: 每批数量

    Yields:
        每批 chunk dict，包含 chunk_id, doc_id, text, embedding（如果有）
    """
    from sqlalchemy import text as sql_text

    # coarse_chunks: coarse_chunk_id / fine_chunks: fine_chunk_id
    table_map = {
        "coarse": ("coarse_chunks", "coarse_chunk_id", "doc_id"),
        "fine": ("fine_chunks", "fine_chunk_id", "doc_id"),
    }
    table, chunk_col, doc_col = table_map[chunk_type]

    # 先确认表存在
    result = session.execute(sql_text(f"SELECT COUNT(*) FROM {table}"))
    total = result.scalar()
    logger.info(f"[DB] {table}: {total} rows")

    offset = 0
    while offset < total:
        rows = session.execute(
            sql_text(f"""
                SELECT {chunk_col}, {doc_col}, text, canonical_id
                FROM {table}
                ORDER BY {chunk_col}
                LIMIT :limit OFFSET :offset
            """),
            {"limit": batch_size, "offset": offset},
        )
        batch = []
        for row in rows:
            chunk_id, doc_id, text_val, canonical_id = row
            batch.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "canonical_id": canonical_id,
                "text": text_val or "",
            })
        if not batch:
            break
        yield batch
        offset += batch_size


def reindex_milvus(
    coarse_only: bool = False,
    fine_only: bool = False,
    dry_run: bool = False,
    batch_size: int = 50,
    confirm: bool = True,
) -> None:
    """
    执行 Milvus 重索引。

    Args:
        coarse_only: 只重索引 coarse chunks
        fine_only: 只重索引 fine chunks
        dry_run: 只打印信息，不实际写入
        batch_size: embedding 批处理大小
        confirm: 是否需要用户确认
    """
    _load_dotenv()

    from src.corpus.store.vector_index import (
        MilvusVectorIndex,
        MILVUS_COLLECTION_COARSE,
        MILVUS_COLLECTION_FINE,
    )
    from src.embeddings.client import get_embedding_client
    from src.db.engine import init_db, get_session_factory

    init_db()
    session = get_session_factory()()

    # 确认当前状态
    idx = MilvusVectorIndex()
    idx.connect()

    coarse_count = idx.count(MILVUS_COLLECTION_COARSE)
    fine_count = idx.count(MILVUS_COLLECTION_FINE)

    logger.info(f"[Status] coarse_chunks in Milvus: {coarse_count}")
    logger.info(f"[Status] fine_chunks in Milvus: {fine_count}")

    if dry_run:
        logger.info("[DryRun] dry-run 模式，仅打印信息")
        logger.info(f"[DryRun] 将重新生成 {coarse_count + fine_count} 个 embedding（1024维）")
        session.close()
        return

    # 用户确认
    if confirm:
        do_reindex = input(
            f"\n⚠️  确认重索引？（将删除现有 Milvus collection 并重新插入）[y/N]: "
        ).strip().lower()
        if do_reindex != "y":
            logger.info("已取消")
            session.close()
            return

    embedding_client = get_embedding_client()
    new_dim = embedding_client._dimension
    logger.info(f"[Embedding] 使用模型维度: {new_dim}")

    # 统计
    total_embedded = 0
    total_time = 0.0

    # ── 重索引 coarse_chunks ──────────────────────────────────────────────────
    if not fine_only:
        logger.info(f"[Reindex] 开始重索引 coarse_chunks...")

        # 删除旧 collection
        try:
            idx.drop_collection(MILVUS_COLLECTION_COARSE)
            logger.info(f"[Reindex] 已删除 {MILVUS_COLLECTION_COARSE}")
        except Exception as e:
            logger.warning(f"[Reindex] 删除旧 collection 失败（可能不存在）：{e}")

        # 创建新 collection
        idx.create_collection(
            MILVUS_COLLECTION_COARSE,
            dim=new_dim,
            description="Coarse chunks (paper-level retrieval) — Qwen text-embedding-v4",
        )
        logger.info(f"[Reindex] 创建新 collection {MILVUS_COLLECTION_COARSE} (dim={new_dim})")

        # 分批处理
        for batch in _iter_chunks(session, "coarse", batch_size=batch_size):
            texts = [c["text"] for c in batch]
            if not any(text.strip() for text in texts):
                continue

            t0 = time.time()
            embeddings = embedding_client.encode(texts, show_progress_bar=False)
            elapsed = time.time() - t0

            records = []
            for c, emb in zip(batch, embeddings):
                if not c["text"].strip():
                    continue
                from src.corpus.store.vector_index import VectorRecord
                records.append(VectorRecord(
                    id=c["chunk_id"],
                    doc_id=c["doc_id"],
                    canonical_id=c["canonical_id"] or c["doc_id"],
                    vector=emb.tolist(),
                    text=c["text"],
                    section="",
                    page_start=1,
                    page_end=1,
                    token_count=len(c["text"].split()),
                ))

            if records:
                idx.upsert(MILVUS_COLLECTION_COARSE, records)
                total_embedded += len(records)

            total_time += elapsed
            logger.info(
                f"[Reindex] coarse: {len(records)} chunks, "
                f"embedding耗时 {elapsed:.2f}s, 总计 {total_embedded}"
            )

    # ── 重索引 fine_chunks ───────────────────────────────────────────────────
    if not coarse_only:
        logger.info(f"[Reindex] 开始重索引 fine_chunks...")

        try:
            idx.drop_collection(MILVUS_COLLECTION_FINE)
            logger.info(f"[Reindex] 已删除 {MILVUS_COLLECTION_FINE}")
        except Exception as e:
            logger.warning(f"[Reindex] 删除旧 collection 失败（可能不存在）：{e}")

        idx.create_collection(
            MILVUS_COLLECTION_FINE,
            dim=new_dim,
            description="Fine chunks (evidence-level retrieval) — Qwen text-embedding-v4",
        )

        for batch in _iter_chunks(session, "fine", batch_size=batch_size):
            texts = [c["text"] for c in batch]
            if not any(text.strip() for text in texts):
                continue

            t0 = time.time()
            embeddings = embedding_client.encode(texts, show_progress_bar=False)
            elapsed = time.time() - t0

            records = []
            for c, emb in zip(batch, embeddings):
                if not c["text"].strip():
                    continue
                from src.corpus.store.vector_index import VectorRecord
                records.append(VectorRecord(
                    id=c["chunk_id"],
                    doc_id=c["doc_id"],
                    canonical_id=c["canonical_id"] or c["doc_id"],
                    vector=emb.tolist(),
                    text=c["text"],
                    section="",
                    page_start=1,
                    page_end=1,
                    token_count=len(c["text"].split()),
                ))

            if records:
                idx.upsert(MILVUS_COLLECTION_FINE, records)
                total_embedded += len(records)

            total_time += elapsed
            logger.info(
                f"[Reindex] fine: {len(records)} chunks, "
                f"embedding耗时 {elapsed:.2f}s, 总计 {total_embedded}"
            )

    session.close()

    logger.info(
        f"[Done] 重索引完成！共处理 {total_embedded} 个 chunks，"
        f"总耗时 {total_time:.1f}s，平均 {total_time/max(total_embedded,1):.3f}s/chunk"
    )

    # 验证
    idx2 = MilvusVectorIndex()
    idx2.connect()
    logger.info(
        f"[Verify] coarse={idx2.count(MILVUS_COLLECTION_COARSE)}, "
        f"fine={idx2.count(MILVUS_COLLECTION_FINE)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milvus 重索引（Qwen text-embedding-v4）")
    parser.add_argument("--yes", "-y", action="store_true", help="跳过确认")
    parser.add_argument("--dry-run", action="store_true", help="仅打印信息")
    parser.add_argument("--coarse-only", action="store_true", help="只处理 coarse")
    parser.add_argument("--fine-only", action="store_true", help="只处理 fine")
    parser.add_argument("--batch-size", type=int, default=10, help="embedding 批处理大小（Qwen 上限 10）")
    args = parser.parse_args()

    reindex_milvus(
        coarse_only=args.coarse_only,
        fine_only=args.fine_only,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        confirm=not args.yes,
    )
