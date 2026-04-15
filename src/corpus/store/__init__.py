"""Corpus Store 子模块 — 文档/Chunk 存储 + 向量/关键词/元数据索引。"""

from src.corpus.store.chunk_store import ChunkStore
from src.corpus.store.document_store import DocumentStore
from src.corpus.store.keyword_index import KeywordIndex, KeywordSearchResult, PGKeywordIndex
from src.corpus.store.metadata_index import (
    FilterCondition,
    FilterOp,
    MetadataFilter,
    MetadataIndex,
)
from src.corpus.store.repository import (
    CorpusRepository,
    EvidenceSearchResult,
    PaperSearchResult,
)
from src.corpus.store.vector_index import (
    EMBEDDING_DIM,
    MILVUS_COLLECTION_COARSE,
    MILVUS_COLLECTION_FINE,
    MilvusVectorIndex,
    VectorIndex,
    VectorRecord,
    VectorSearchResult,
    get_vector_index,
)

__all__ = [
    # Stores
    "DocumentStore",
    "ChunkStore",
    # Vector
    "VectorIndex",
    "MilvusVectorIndex",
    "VectorRecord",
    "VectorSearchResult",
    "get_vector_index",
    "EMBEDDING_DIM",
    "MILVUS_COLLECTION_COARSE",
    "MILVUS_COLLECTION_FINE",
    # Keyword
    "KeywordIndex",
    "PGKeywordIndex",
    "KeywordSearchResult",
    # Metadata
    "MetadataIndex",
    "MetadataFilter",
    "FilterCondition",
    "FilterOp",
    # Repository
    "CorpusRepository",
    "PaperSearchResult",
    "EvidenceSearchResult",
]
