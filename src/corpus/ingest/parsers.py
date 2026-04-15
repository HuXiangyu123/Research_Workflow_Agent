"""Parser 层 — PDF / HTML 解析 + 元数据提取 + 质量评分。"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field

from pypdf import PdfReader

from src.corpus.models import PageText

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PDF Parser
# ---------------------------------------------------------------------------


class PDFParser:
    """将 PDF bytes 解析为 PageText 列表 + 质量评分。"""

    def parse_bytes(self, pdf_bytes: bytes) -> tuple[list[PageText], list[str]]:
        """
        解析 PDF，返回按页分段的 PageText 列表。
        同时记录解析警告。
        """
        warnings: list[str] = []
        page_texts: list[PageText] = []

        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
        except Exception as e:
            warnings.append(f"PDF 解析失败（pypdf）：{e}")
            return [], warnings

        num_pages = len(reader.pages)
        if num_pages == 0:
            warnings.append("PDF 页面数为 0")
            return [], warnings

        char_offset = 0
        for page_num, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            text = raw_text.rstrip()

            page_texts.append(
                PageText(
                    page_num=page_num,
                    text=text,
                    char_start=char_offset,
                    char_end=char_offset + len(text),
                )
            )
            char_offset += len(text) + 2  # 加上换行符

        # 质量警告
        total_chars = sum(len(pt.text) for pt in page_texts)
        avg_chars = total_chars / num_pages if num_pages else 0
        if avg_chars < 100:
            warnings.append(
                f"PDF 平均每页字符数过低（{avg_chars:.0f}），可能是扫描版或解析失败"
            )
        if num_pages > 500:
            warnings.append(f"PDF 页数过多（{num_pages}），可能为预印本合集")

        return page_texts, warnings

    def estimate_quality(self, text: str, metadata: dict) -> float:
        """
        根据文本和元数据综合评估解析质量。

        0.9+: 完整元数据 + 充足文本
        0.6~0.8: 有元数据但文本一般
        < 0.5: 元数据缺失或文本过短
        """
        score = 0.5

        title = metadata.get("title", "")
        abstract = metadata.get("abstract", "")

        if title and len(title) > 5:
            score += 0.2
        if abstract and len(abstract) > 50:
            score += 0.15
        if text and len(text) > 500:
            score += 0.1
        if metadata.get("arxiv_id"):
            score += 0.05

        return min(score, 1.0)


# ---------------------------------------------------------------------------
# HTML Parser
# ---------------------------------------------------------------------------


class HTMLParser:
    """将 HTML bytes 解析为纯文本（用于在线来源）。"""

    def parse_bytes(
        self, html_bytes: bytes, url: str = ""
    ) -> tuple[str, list[str]]:
        """
        解析 HTML，返回提取的纯文本。
        使用基础正则提取，复杂页面可用 newspaper3k / trafilatura 增强。
        """
        warnings: list[str] = []

        try:
            html = html_bytes.decode("utf-8", errors="ignore")
        except Exception as e:
            warnings.append(f"HTML 解码失败：{e}")
            return "", warnings

        text = self._extract_text(html)
        if not text or len(text) < 100:
            warnings.append("HTML 正文提取结果过短，可能解析不完整")

        return text, warnings

    def _extract_text(self, html: str) -> str:
        """基础 HTML → 文本提取（去除 script/style/注释 + 合并空白）。"""
        # 去除 <script> <style> <noscript>
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.IGNORECASE | re.DOTALL)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.IGNORECASE | re.DOTALL)
        html = re.sub(r"<noscript[^>]*>.*?</noscript>", "", html, flags=re.IGNORECASE | re.DOTALL)
        # 去除注释
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
        # 去除所有 HTML 标签
        text = re.sub(r"<[^>]+>", " ", html)
        # 还原实体
        text = self._unescape_html_entities(text)
        # 合并空白
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text.strip()

    def _unescape_html_entities(self, text: str) -> str:
        """还原常见 HTML 实体。"""
        entities = {
            "&nbsp;": " ",
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"',
            "&#39;": "'",
            "&apos;": "'",
            "&mdash;": "—",
            "&ndash;": "–",
            "&hellip;": "…",
        }
        for entity, char in entities.items():
            text = text.replace(entity, char)
        # 处理数字实体
        text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
        text = re.sub(r"&#x([0-9a-fA-F]+);", lambda m: chr(int(m.group(1), 16)), text)
        return text


# ---------------------------------------------------------------------------
# Metadata Extractor
# ---------------------------------------------------------------------------


class MetadataExtractor:
    """
    从 raw_metadata 和 extracted_text 中提取结构化元数据。
    优先使用 API/feed 返回的元数据，仅做必要清洗。
    """

    def extract(self, raw_metadata: dict, page_texts: list[PageText]) -> dict:
        """
        提取并规范化元数据字段。

        输出字段：
        - title, authors, year, venue, abstract
        - doi, arxiv_id
        """
        origin = raw_metadata.get("origin", "")

        if origin == "arxiv":
            return self._extract_arxiv_metadata(raw_metadata)
        elif origin == "local_pdf":
            return self._extract_local_pdf_metadata(raw_metadata, page_texts)
        elif origin == "online_pdf":
            return self._extract_online_pdf_metadata(raw_metadata, page_texts)
        else:
            return self._extract_generic_metadata(raw_metadata, page_texts)

    def _extract_arxiv_metadata(self, raw: dict) -> dict:
        """从 arXiv API 返回的原始数据中提取。"""
        authors = raw.get("authors", [])
        if isinstance(authors, str):
            # 可能返回逗号分隔的字符串，尝试拆分
            authors = [a.strip() for a in re.split(r"[,;]", authors) if a.strip()]
        return {
            "title": raw.get("title", ""),
            "authors": authors,
            "year": raw.get("published_year"),
            "venue": None,  # arXiv 预印本无 venue
            "abstract": raw.get("abstract", ""),
            "doi": raw.get("doi"),
            "arxiv_id": raw.get("arxiv_id"),
        }

    def _extract_local_pdf_metadata(self, raw: dict, page_texts: list[PageText]) -> dict:
        """从本地 PDF 中提取元数据（目前较简单）。"""
        # 优先从文件名尝试 arXiv ID
        arxiv_id = raw.get("arxiv_id_candidate")

        # 尝试从首页文本提取标题（前几行通常含标题）
        title = self._extract_title_from_text(page_texts[0].text if page_texts else "")
        authors = self._extract_authors_from_text(page_texts[0].text if page_texts else "")
        year = self._extract_year_from_text(page_texts[0].text if page_texts else "")

        return {
            "title": title or raw.get("file_name", "Unknown"),
            "authors": authors,
            "year": year,
            "venue": None,
            "abstract": self._extract_abstract_from_text(page_texts),
            "doi": None,
            "arxiv_id": arxiv_id,
        }

    def _extract_online_pdf_metadata(self, raw: dict, page_texts: list[PageText]) -> dict:
        """从在线 PDF 提取元数据（与本地 PDF 类似）。"""
        return self._extract_local_pdf_metadata({"origin": "online_pdf"}, page_texts)

    def _extract_generic_metadata(self, raw: dict, page_texts: list[PageText]) -> dict:
        """通用提取策略（兜底）。"""
        title = (
            raw.get("title")
            or raw.get("og_title")
            or self._extract_title_from_text(page_texts[0].text if page_texts else "")
        )
        return {
            "title": title or "Unknown",
            "authors": [],
            "year": None,
            "venue": None,
            "abstract": raw.get("abstract") or self._extract_abstract_from_text(page_texts),
            "doi": raw.get("doi"),
            "arxiv_id": raw.get("arxiv_id"),
        }

    # -------------------------------------------------------------------------
    # 启发式提取（用于无 API 信息的 PDF）
    # -------------------------------------------------------------------------

    def _extract_title_from_text(self, text: str) -> str:
        """
        从首页文本中提取标题。
        策略：取前 3 行非空文本，过滤掉作者/机构信息。
        """
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        # 取前 3 行，假设第一行是标题
        candidates = []
        for line in lines[:3]:
            # 过滤掉包含 "@"（邮箱）、"http"、数字年份 开头的行
            if re.search(r"[\@\#]", line) or re.match(r"^\d{4}", line):
                continue
            if len(line) < 200 and not line.isupper():  # 全大写可能是机构名
                candidates.append(line)
        if candidates:
            return candidates[0]
        return ""

    def _extract_authors_from_text(self, text: str) -> list[str]:
        """从首页文本中提取作者列表。"""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        authors = []
        for line in lines[:10]:
            # 匹配 "Name Name" 或 "Name, Name" 模式（无机构/邮箱）
            parts = re.split(r"[,;|]", line)
            for part in parts:
                part = re.sub(r"\s+", " ", part).strip()
                # 过滤含邮箱/机构/链接的行
                if len(part) > 3 and len(part) < 100 and not re.search(r"[\@\#]", part):
                    authors.append(part)
        # 去重，保持顺序
        seen = set()
        unique = []
        for a in authors:
            if a.lower() not in seen:
                seen.add(a.lower())
                unique.append(a)
        return unique[:10]

    def _extract_year_from_text(self, text: str) -> int | None:
        """从文本中提取年份。"""
        matches = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
        if matches:
            try:
                return int(matches[0])
            except ValueError:
                pass
        return None

    def _extract_abstract_from_text(self, page_texts: list[PageText]) -> str:
        """从文本中提取摘要（查找 'Abstract' 关键词附近文本）。"""
        if not page_texts:
            return ""
        first_page = page_texts[0].text

        # 查找 Abstract 标题后的内容
        match = re.search(
            r"(?:Abstract|摘要)[\s:：]*\n?(.*)",
            first_page,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            abstract = match.group(1).strip()
            # 截取前 2000 字符
            return abstract[:2000]

        # 兜底：取首页前 1000 字符
        return first_page[:1000]
