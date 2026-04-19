"""Loader 层 — 统一入口 + 三大 Loader。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from html import unescape
from typing import Union

import feedparser
import hashlib
import io
import logging
import os
import re
import urllib.request
import urllib.parse

from src.corpus.models import (
    PageText,
    ParsedDocument,
    SourceRef,
    SourceType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source Input（Loader 入口参数）
# ---------------------------------------------------------------------------


@dataclass
class ArxivSourceInput:
    """arXiv 来源输入。"""
    arxiv_id: str  # 如 "1706.03762" 或 "1706.03762v1"


@dataclass
class LocalPdfSourceInput:
    """本地 PDF 来源输入。"""
    file_path: str  # 文件绝对路径或相对路径


@dataclass
class OnlineUrlSourceInput:
    """在线 URL 来源输入。"""
    url: str  # 直接 URL（如 PDF 直链或 HTML）


SourceInput = Union[ArxivSourceInput, LocalPdfSourceInput, OnlineUrlSourceInput]


# ---------------------------------------------------------------------------
# Base Loader
# ---------------------------------------------------------------------------


class BaseLoader(ABC):
    """Loader 抽象基类。"""

    @abstractmethod
    def load(self, source: SourceInput) -> ParsedDocument:
        """加载来源，返回 ParsedDocument。"""
        ...


# ---------------------------------------------------------------------------
# Helper：生成 PDF bytes 的通用下载函数
# ---------------------------------------------------------------------------


def _download_bytes(url: str, timeout: int = 30) -> bytes:
    """通用 HTTP/HTTPS 下载。"""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PaperReaderAgent/1.0 (+https://github.com)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# Arxiv Loader
# ---------------------------------------------------------------------------


class ArxivLoader(BaseLoader):
    """从 arXiv 加载论文：API 获取元数据 + PDF 下载。"""

    def load(self, source: SourceInput) -> ParsedDocument:
        if not isinstance(source, ArxivSourceInput):
            raise TypeError(f"ArxivLoader 需要 ArxivSourceInput，得到 {type(source)}")

        arxiv_id = _normalize_arxiv_id(source.arxiv_id)
        source_ref = SourceRef(
            source_id="",
            source_type=SourceType.ARXIV,
            uri_or_path=f"https://arxiv.org/abs/{arxiv_id}",
            external_id=arxiv_id,
            ingest_status="pending",
        )
        source_ref.source_id = f"arxiv:{arxiv_id}"

        # 1. 获取元数据（feedparser）
        metadata = self._fetch_arxiv_metadata(arxiv_id, source_ref)

        # 2. 下载 PDF
        pdf_bytes = self._download_pdf(arxiv_id, source_ref)

        # 3. 解析 PDF → page_texts
        # 延迟导入避免循环依赖
        from src.corpus.ingest import parsers as _parsers

        pdf_parser = _parsers.PDFParser()
        page_texts, warnings = pdf_parser.parse_bytes(pdf_bytes)

        extracted_text = "\n\n".join(pt.text for pt in page_texts)

        return ParsedDocument(
            source_ref=source_ref,
            extracted_text=extracted_text,
            page_texts=page_texts,
            raw_metadata=metadata,
            parse_quality_score=pdf_parser.estimate_quality(extracted_text, metadata),
            warnings=warnings,
        )

    def _fetch_arxiv_metadata(self, arxiv_id: str, source_ref: SourceRef) -> dict:
        """通过 arXiv API 获取元数据。"""
        api_urls = [
            f"https://export.arxiv.org/api/query?id_list={arxiv_id}",
            f"http://export.arxiv.org/api/query?id_list={arxiv_id}",
            f"https://arxiv.org/api/query?id_list={arxiv_id}",
        ]

        last_error = ""
        for api_url in api_urls:
            try:
                feed = _parse_arxiv_feed(api_url)
                if feed.entries:
                    entry = feed.entries[0]
                    authors = [a.name for a in getattr(entry, "authors", [])]
                    published = getattr(entry, "published", None)
                    year = None
                    if published:
                        try:
                            year = int(published[:4])
                        except (ValueError, IndexError):
                            pass
                    pdf_url = None
                    for link in getattr(entry, "links", []):
                        if getattr(link, "type", "") == "application/pdf":
                            pdf_url = getattr(link, "href", None)

                    metadata = {
                        "origin": "arxiv",
                        "arxiv_id": arxiv_id,
                        "title": entry.title.replace("\n", " ").strip(),
                        "authors": authors,
                        "abstract": entry.summary.replace("\n", " ").strip(),
                        "published": published,
                        "published_year": year,
                        "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    }
                    source_ref.ingest_status = "processed"
                    return metadata
            except Exception as e:
                last_error = str(e)
                continue

        # fallback：尝试抓取 HTML 页面
        fallback = self._fetch_arxiv_html_fallback(arxiv_id)
        if fallback:
            source_ref.ingest_status = "processed"
            return fallback

        source_ref.ingest_status = "failed"
        raise RuntimeError(
            f"无法获取 arXiv {arxiv_id} 元数据（arXiv API 可能暂时不可用）：{last_error}"
        )

    def _fetch_arxiv_html_fallback(self, arxiv_id: str) -> dict | None:
        """当 API 不可用时，从 HTML 页面抓取元数据。"""
        try:
            html = _download_bytes(f"https://arxiv.org/abs/{arxiv_id}").decode(
                "utf-8", errors="ignore"
            )
            title_match = re.search(
                r'<meta\s+property="og:title"\s+content="([^"]+)"',
                html,
                flags=re.IGNORECASE,
            )
            desc_match = re.search(
                r'<meta\s+property="og:description"\s+content="([^"]+)"',
                html,
                flags=re.IGNORECASE,
            )
            if not title_match and not desc_match:
                return None
            title = " ".join(unescape(title_match.group(1)).split()) if title_match else "Unknown"
            abstract = re.sub(
                r"^\s*Abstract:\s*", "",
                " ".join(unescape(desc_match.group(1)).split())) if desc_match else ""
            return {
                "origin": "arxiv",
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": [],
                "abstract": abstract,
                "published": None,
                "published_year": None,
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                "fallback_source": "arxiv_abs_html",
            }
        except Exception:
            return None

    def _download_pdf(self, arxiv_id: str, source_ref: SourceRef) -> bytes:
        """下载 PDF。"""
        pdf_url = (
            source_ref.uri_or_path.replace("/abs/", "/pdf/")
            + ".pdf"
        )
        try:
            return _download_bytes(pdf_url)
        except Exception as e:
            logger.warning(f"PDF 下载失败（{pdf_url}），尝试备用地址：{e}")
            alt_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            try:
                return _download_bytes(alt_url)
            except Exception as e2:
                raise RuntimeError(f"PDF 下载失败（{alt_url}）：{e2}")


def _parse_arxiv_feed(api_url: str, timeout: int = 8) -> feedparser.FeedParserDict:
    request = urllib.request.Request(
        api_url,
        headers={"User-Agent": "PaperReaderAgent/1.0 (+https://arxiv.org)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return feedparser.parse(resp.read())


def _normalize_arxiv_id(raw: str) -> str:
    """提取并规范化 arXiv ID（去掉 URL 前缀 / .pdf 后缀）。"""
    s = raw.strip()
    # 去掉 URL 前缀（https://arxiv.org/abs/ 或 https://arxiv.org/pdf/）
    s = re.sub(r"https?://arxiv\.org/(abs|pdf)/", "", s, flags=re.IGNORECASE)
    # 去掉 .pdf 后缀
    s = re.sub(r"\.pdf$", "", s, flags=re.IGNORECASE)
    return s


# ---------------------------------------------------------------------------
# Local PDF Loader
# ---------------------------------------------------------------------------


class LocalPdfLoader(BaseLoader):
    """从本地文件加载 PDF。"""

    def load(self, source: SourceInput) -> ParsedDocument:
        if not isinstance(source, LocalPdfSourceInput):
            raise TypeError(
                f"LocalPdfLoader 需要 LocalPdfSourceInput，得到 {type(source)}"
            )

        file_path = os.path.abspath(os.path.expanduser(source.file_path))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"本地 PDF 文件不存在：{file_path}")

        source_ref = SourceRef(
            source_id=f"local:{hashlib_sha256(file_path)[:16]}",
            source_type=SourceType.LOCAL_PDF,
            uri_or_path=file_path,
            ingest_status="pending",
        )

        with open(file_path, "rb") as f:
            pdf_bytes = f.read()

        from src.corpus.ingest.parsers import PDFParser

        parser = PDFParser()
        page_texts, warnings = parser.parse_bytes(pdf_bytes)
        extracted_text = "\n\n".join(pt.text for pt in page_texts)

        # 尝试从文件名提取 arXiv ID
        external_id = _extract_arxiv_id_from_path(file_path)

        metadata = {
            "origin": "local_pdf",
            "file_name": os.path.basename(file_path),
            "file_size": len(pdf_bytes),
            "arxiv_id_candidate": external_id,
        }

        return ParsedDocument(
            source_ref=source_ref,
            extracted_text=extracted_text,
            page_texts=page_texts,
            raw_metadata=metadata,
            parse_quality_score=parser.estimate_quality(extracted_text, metadata),
            warnings=warnings,
        )


def hashlib_sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Online URL Loader
# ---------------------------------------------------------------------------


class OnlineUrlLoader(BaseLoader):
    """从在线 URL 加载内容（PDF 直链或 HTML）。"""

    def load(self, source: SourceInput) -> ParsedDocument:
        if not isinstance(source, OnlineUrlSourceInput):
            raise TypeError(
                f"OnlineUrlLoader 需要 OnlineUrlSourceInput，得到 {type(source)}"
            )

        url = source.url.strip()

        source_ref = SourceRef(
            source_id=f"online:{hashlib_sha256(url)[:16]}",
            source_type=SourceType.ONLINE_URL,
            uri_or_path=url,
            ingest_status="pending",
        )

        # 判断 content type
        content_bytes, content_type = _probe_content_type(url)

        if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
            from src.corpus.ingest.parsers import PDFParser

            parser = PDFParser()
            page_texts, warnings = parser.parse_bytes(content_bytes)
            extracted_text = "\n\n".join(pt.text for pt in page_texts)
            metadata = {
                "origin": "online_pdf",
                "content_type": content_type,
                "url": url,
            }
        else:
            # HTML → 提取正文
            from src.corpus.ingest.parsers import HTMLParser

            parser = HTMLParser()
            text, warnings = parser.parse_bytes(content_bytes, url)
            page_texts = [
                PageText(
                    page_num=1,
                    text=text,
                    char_start=0,
                    char_end=len(text),
                )
            ]
            extracted_text = text
            metadata = {
                "origin": "online_html",
                "content_type": content_type,
                "url": url,
            }

        return ParsedDocument(
            source_ref=source_ref,
            extracted_text=extracted_text,
            page_texts=page_texts,
            raw_metadata=metadata,
            parse_quality_score=0.5,  # 在线来源默认中等质量
            warnings=warnings,
        )


def _probe_content_type(url: str) -> tuple[bytes, str]:
    """HEAD 请求探测 content-type，失败则降级为 GET。"""
    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "PaperReaderAgent/1.0 (+https://github.com)",
                "Range": "bytes=0-1024",  # 只取前 1KB
            },
        )
        with urllib.request.urlopen(request, timeout=10) as resp:
            content_type = resp.headers.get("Content-Type", "application/octet-stream")
            # 已知类型则不再 GET
            if "pdf" in content_type.lower():
                # PDF 需要全文，重新 GET
                request2 = urllib.request.Request(
                    url,
                    headers={"User-Agent": "PaperReaderAgent/1.0"},
                )
                with urllib.request.urlopen(request2, timeout=30) as resp2:
                    return resp2.read(), content_type
            return b"", content_type
    except Exception:
        # 无法 HEAD，降级为 GET
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "PaperReaderAgent/1.0"},
        )
        with urllib.request.urlopen(request, timeout=30) as resp:
            return resp.read(), resp.headers.get("Content-Type", "application/octet-stream")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class LoaderDispatcher:
    """统一 Loader 调度器。"""

    _loaders: dict[type, BaseLoader] = {}

    @classmethod
    def register(cls, source_type: type, loader: BaseLoader) -> None:
        cls._loaders[source_type] = loader

    @classmethod
    def load(cls, source: SourceInput) -> ParsedDocument:
        loader = cls._loaders.get(type(source))
        if loader is None:
            raise ValueError(f"未注册的 SourceInput 类型：{type(source)}")
        return loader.load(source)

    @classmethod
    def load_many(cls, sources: list[SourceInput]) -> list[ParsedDocument]:
        return [cls.load(s) for s in sources]


# 注册默认 Loader
LoaderDispatcher.register(ArxivSourceInput, ArxivLoader())
LoaderDispatcher.register(LocalPdfSourceInput, LocalPdfLoader())
LoaderDispatcher.register(OnlineUrlSourceInput, OnlineUrlLoader())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_arxiv_id_from_path(file_path: str) -> str | None:
    """从文件路径/名称中提取 arXiv ID。"""
    basename = os.path.basename(file_path)
    match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", basename)
    return match.group(1) if match else None
