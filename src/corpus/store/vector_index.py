"""Vector Index — 向量检索层（抽象基类 + Milvus 实现）。"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from pymilvus import Collection, CollectionSchema, FieldSchema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MILVUS_COLLECTION_COARSE = "coarse_chunks"
MILVUS_COLLECTION_FINE = "fine_chunks"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 默认维度

# ---------------------------------------------------------------------------
# Vector Record
# ---------------------------------------------------------------------------


@dataclass
class VectorRecord:
    """向量记录（对应 Milvus 的一条 entity）。"""

    id: str                   # chunk_id / doc_id
    vector: list[float]        # embedding
    text: str                 # 原始文本（用于 display / BM25 rerank）
    doc_id: str
    canonical_id: str
    section: str
    page_start: int = 1
    page_end: int = 1
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Search Result
# ---------------------------------------------------------------------------


@dataclass
class VectorSearchResult:
    """向量检索结果。"""

    id: str
    score: float
    text: str
    doc_id: str
    canonical_id: str
    section: str
    page_start: int = 1
    page_end: int = 1
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base Vector Index
# ---------------------------------------------------------------------------


class VectorIndex(ABC):
    """
    向量索引抽象基类。

    定义统一的向量检索接口，具体实现由 MilvusVectorIndex 等子类完成。
    """

    @abstractmethod
    def connect(self) -> None:
        """连接到向量数据库。"""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """检查是否已连接。"""
        ...

    @abstractmethod
    def create_collection(
        self,
        name: str,
        dim: int = EMBEDDING_DIM,
        description: str = "",
    ) -> None:
        """创建 Collection（如不存在）。"""
        ...

    @abstractmethod
    def upsert(self, name: str, records: list[VectorRecord]) -> int:
        """
        批量写入向量记录。

        Returns:
            写入的记录数
        """
        ...

    @abstractmethod
    def search(
        self,
        name: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[VectorSearchResult]:
        """
        向量检索。

        Args:
            name: Collection 名称
            query_vector: 查询向量
            top_k: 返回数量
            filters: metadata 过滤条件（实现方决定语法）

        Returns:
            VectorSearchResult 列表
        """
        ...

    @abstractmethod
    def delete_by_id(self, name: str, ids: list[str]) -> None:
        """通过 ID 删除向量记录。"""
        ...

    @abstractmethod
    def count(self, name: str) -> int:
        """返回 Collection 中的记录数。"""
        ...

    @abstractmethod
    def drop_collection(self, name: str) -> None:
        """删除 Collection。"""
        ...


# ---------------------------------------------------------------------------
# Milvus Vector Index
# ---------------------------------------------------------------------------


class MilvusVectorIndex(VectorIndex):
    """
    Milvus 向量索引实现。

    能力：
    - 连接 / 创建 Collection（HNSW / IVF-FLAT）
    - upsert 向量 + metadata
    - ANN search（支持 top_k 和 metadata filtering）
    - delete / count / drop

    环境变量：
    - MILVUS_HOST（默认 127.0.0.1）
    - MILVUS_PORT（默认 19530）
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        user: str = "",
        password: str = "",
        alias: str = "default",
    ):
        import os as _os

        self._host = host or _os.getenv("MILVUS_HOST", "127.0.0.1")
        self._port = int(port or int(_os.getenv("MILVUS_PORT", "19530")))
        self._user = user
        self._password = password
        self._alias = alias
        self._connected = False

    def connect(self) -> None:
        """连接到 Milvus。"""
        try:
            from pymilvus import connections

            alias = self._alias

            # 幂等：已存在同名 alias 时先断开（pytest 多实例场景）
            if connections.has_connection(alias):
                try:
                    connections.disconnect(alias)
                except Exception:
                    pass

            connections.connect(
                alias=alias,
                host=self._host,
                port=str(self._port),
                user=self._user,
                password=self._password,
            )
            self._alias = alias
            self._connected = True
            logger.info(f"[MilvusVectorIndex] Connected to {self._host}:{self._port}")
        except ImportError:
            raise RuntimeError(
                "pymilvus 未安装。请运行：pip install pymilvus"
            )
        except Exception as e:
            raise RuntimeError(f"Milvus 连接失败：{e}")

    def is_connected(self) -> bool:
        return self._connected

    def _ensure_connected(self) -> None:
        if not self._connected:
            self.connect()

    # -------------------------------------------------------------------------
    # Schema helpers
    # -------------------------------------------------------------------------

    def _get_or_create_schema(
        self,
        name: str,
        dim: int,
        description: str = "",
    ) -> "CollectionSchema":
        """
        创建或获取 Collection schema。

        Schema 结构：
        - id: 主键（VARCHAR，chunk_id）
        - vector: 向量（ FLOAT_VECTOR(dim)）
        - text: 原始文本（VARCHAR）
        - doc_id: 文档 ID（VARCHAR，可过滤）
        - canonical_id: 论文身份（VARCHAR，可过滤）
        - section: 章节（VARCHAR，可过滤）
        - page_start / page_end: 页码（INT，可过滤）
        - token_count: token 数（INT）
        - metadata: JSON 字符串（VARCHAR）
        """
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema

        try:
            collection = Collection(name, using=self._alias)
            # 已存在，直接返回
            return collection.schema
        except Exception:
            pass  # 不存在，需要创建

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64, is_partition_key=True),
            FieldSchema(name="canonical_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="page_start", dtype=DataType.INT16),
            FieldSchema(name="page_end", dtype=DataType.INT16),
            FieldSchema(name="token_count", dtype=DataType.INT32),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=4096),
        ]

        schema = CollectionSchema(
            fields=fields,
            description=description or f"Milvus collection: {name}",
        )
        return schema

    def create_collection(
        self,
        name: str,
        dim: int = EMBEDDING_DIM,
        description: str = "",
        index_type: str = "AUTOINDEX",
        metric_type: str = "COSINE",
    ) -> None:
        """
        创建 Collection 并建立索引。

        Args:
            name: Collection 名
            dim: 向量维度（默认 384 = all-MiniLM-L6-v2）
            description: 描述
            index_type: 索引类型（AUTOINDEX / HNSW / IVF_FLAT）
            metric_type: 距离度量（COSINE / L2 / IP）
        """
        self._ensure_connected()

        from pymilvus import Collection

        schema = self._get_or_create_schema(name, dim, description)

        try:
            collection = Collection(name, schema=schema, using=self._alias)
        except Exception:
            # 已存在
            collection = Collection(name, using=self._alias)

        # 建立索引（AUTOINDEX 让 Milvus 自动选择最优索引）
        # pymilvus 2.6 API: create_index(field_name, flat_dict)
        index_params = {
            "index_type": index_type,
            "metric_type": metric_type,
            "params": {},
        }

        try:
            collection.create_index("vector", index_params)
            collection.load()
            logger.info(
                f"[MilvusVectorIndex] Created/loaded collection '{name}' "
                f"(dim={dim}, index={index_type}, metric={metric_type})"
            )
        except Exception as e:
            logger.warning(f"[MilvusVectorIndex] 索引创建失败：{e}")
            try:
                collection.load()
            except Exception:
                pass

    def upsert(self, name: str, records: list[VectorRecord]) -> int:
        """批量 upsert 向量记录。"""
        self._ensure_connected()

        from pymilvus import Collection

        collection = Collection(name, using=self._alias)
        collection.load()

        import json

        # pymilvus 2.6+ 要求 column-oriented 格式：每个字段一个 list
        entities = [
            [r.id for r in records],
            [r.vector for r in records],
            [r.text[:8192] for r in records],
            [r.doc_id for r in records],
            [r.canonical_id or "" for r in records],
            [r.section for r in records],
            [r.page_start for r in records],
            [r.page_end for r in records],
            [r.token_count for r in records],
            [json.dumps(r.metadata or {}, ensure_ascii=False) for r in records],
        ]

        try:
            insert_result = collection.insert(entities)
            collection.flush()
            logger.debug(
                f"[MilvusVectorIndex] Inserted {len(records)} records into '{name}'"
            )
            return len(insert_result.primary_keys)
        except Exception as e:
            logger.error(f"[MilvusVectorIndex] upsert 失败：{e}")
            raise

    def search(
        self,
        name: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict | None = None,
        output_fields: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        """
        ANN 向量检索。

        Args:
            name: Collection 名
            query_vector: 查询向量
            top_k: 返回数量
            filters: metadata 过滤条件，支持：
                - doc_id: str
                - canonical_id: str
                - section: str
                - page_start: int
                - page_end: int
                - min_token_count: int
                - max_token_count: int
        """
        self._ensure_connected()

        from pymilvus import Collection

        collection = Collection(name, using=self._alias)
        collection.load()

        # 构建 Milvus 过滤表达式
        filter_expr = self._build_filter_expr(filters)

        if output_fields is None:
            output_fields = [
                "id", "text", "doc_id", "canonical_id",
                "section", "page_start", "page_end", "token_count", "metadata",
            ]

        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }

        try:
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=filter_expr or None,
                output_fields=output_fields,
            )
        except Exception as e:
            logger.warning(f"[MilvusVectorIndex] search 失败，尝试不带 filter：{e}")
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=output_fields,
            )

        search_results: list[VectorSearchResult] = []
        import json

        for hits in results:
            for hit in hits:
                metadata = {}
                raw_meta = hit.entity.get("metadata", "{}")
                try:
                    metadata = json.loads(raw_meta)
                except Exception:
                    pass

                search_results.append(
                    VectorSearchResult(
                        id=hit.entity.get("id", ""),
                        score=float(hit.distance),
                        text=hit.entity.get("text", ""),
                        doc_id=hit.entity.get("doc_id", ""),
                        canonical_id=hit.entity.get("canonical_id", ""),
                        section=hit.entity.get("section", ""),
                        page_start=int(hit.entity.get("page_start", 1)),
                        page_end=int(hit.entity.get("page_end", 1)),
                        token_count=int(hit.entity.get("token_count", 0)),
                        metadata=metadata,
                    )
                )

        return search_results

    def _build_filter_expr(self, filters: dict | None) -> str:
        """
        将 Python dict 转换为 Milvus 过滤表达式。

        支持字段：doc_id / canonical_id / section / page_start / page_end / token_count
        """
        if not filters:
            return ""

        expr_parts: list[str] = []

        if "doc_id" in filters:
            expr_parts.append(f"doc_id == '{filters['doc_id']}'")

        if "canonical_id" in filters:
            expr_parts.append(f"canonical_id == '{filters['canonical_id']}'")

        if "section" in filters:
            expr_parts.append(f"section == '{filters['section']}'")

        if "page_start" in filters:
            expr_parts.append(f"page_start >= {filters['page_start']}")

        if "page_end" in filters:
            expr_parts.append(f"page_end <= {filters['page_end']}")

        if "min_token_count" in filters:
            expr_parts.append(f"token_count >= {filters['min_token_count']}")

        if "max_token_count" in filters:
            expr_parts.append(f"token_count <= {filters['max_token_count']}")

        return " and ".join(expr_parts)

    def delete_by_id(self, name: str, ids: list[str]) -> None:
        """通过主键 ID 删除向量记录。"""
        self._ensure_connected()

        from pymilvus import Collection

        collection = Collection(name, using=self._alias)
        collection.load()
        for id_val in ids:
            collection.delete(expr=f'id == "{id_val}"')
        collection.flush()
        logger.debug(f"[MilvusVectorIndex] Deleted {len(ids)} records from '{name}'")

    def count(self, name: str) -> int:
        """返回 Collection 中的记录数。"""
        self._ensure_connected()

        from pymilvus import Collection

        try:
            collection = Collection(name, using=self._alias)
            collection.flush()
            # Milvus num_entities 在 delete 后不更新，用 query 取实际计数
            results = collection.query(expr="id != ''", output_fields=["id"], limit=10000)
            return len(results)
        except Exception:
            return 0

    def drop_collection(self, name: str) -> None:
        """删除 Collection。"""
        self._ensure_connected()

        from pymilvus import Collection

        try:
            Collection(name, using=self._alias).drop()
            logger.info(f"[MilvusVectorIndex] Dropped collection '{name}'")
        except Exception as e:
            logger.warning(f"[MilvusVectorIndex] drop_collection 失败：{e}")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_milvus_index: MilvusVectorIndex | None = None


def get_vector_index() -> MilvusVectorIndex:
    """获取全局 MilvusVectorIndex 单例。"""
    global _milvus_index
    if _milvus_index is None:
        _milvus_index = MilvusVectorIndex()
    return _milvus_index
