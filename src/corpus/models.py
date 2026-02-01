from typing import List, Optional
from pydantic import BaseModel, Field
import time
import hashlib

class DocumentMeta(BaseModel):
    """文档级元信息"""
    doc_id: str = Field(..., description="文档唯一ID，通常是 hash(source_id)")
    source_id: str = Field(..., description="原始ID，如 arxiv:1706.03762")
    source_uri: str = Field(..., description="来源URI，URL或文件路径")
    title: str = Field(..., description="文档标题")
    authors: Optional[str] = Field(None, description="作者列表")
    published_date: Optional[str] = Field(None, description="发布日期")
    summary: Optional[str] = Field(None, description="摘要")
    added_at: float = Field(default_factory=time.time, description="入库时间戳")
    updated_at: float = Field(default_factory=time.time, description="更新时间戳")
    status: str = Field("pending", description="状态: pending/processed/failed")
    error: Optional[str] = Field(None, description="错误信息")

    @staticmethod
    def generate_id(source_id: str) -> str:
        return hashlib.md5(source_id.encode()).hexdigest()

class Span(BaseModel):
    """文本片段位置信息"""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None

class Chunk(BaseModel):
    """切分后的数据块"""
    chunk_id: str = Field(..., description="Chunk唯一ID，hash(doc_id + text)")
    doc_id: str = Field(..., description="所属文档ID")
    source_id: str = Field(..., description="冗余字段，方便检索")
    source_uri: str = Field(..., description="冗余字段，方便检索")
    title: str = Field(..., description="冗余字段，方便检索")
    section_path: List[str] = Field(default_factory=list, description="章节路径，如 ['Introduction', 'Motivation']")
    span: Span = Field(default_factory=Span, description="位置信息")
    text: str = Field(..., description="文本内容")
    text_hash: str = Field(..., description="文本hash")
    len_chars: int = Field(..., description="字符长度")

    @staticmethod
    def generate_id(doc_id: str, text: str) -> str:
        return hashlib.md5((doc_id + text).encode()).hexdigest()
