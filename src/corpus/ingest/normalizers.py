"""Normalizer 层 — 文本清洗 + 元数据标准化。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.corpus.models import NormalizedMetadata

# ---------------------------------------------------------------------------
# Venue 别名映射表（常见venue 标准化）
# ---------------------------------------------------------------------------

VENUE_ALIASES: dict[str, str] = {
    # NeurIPS
    "neurips": "NeurIPS",
    "nips": "NeurIPS",
    "advances in neural information processing systems": "NeurIPS",
    # ICLR
    "iclr": "ICLR",
    "international conference on learning representations": "ICLR",
    # ICML
    "icml": "ICML",
    "international conference on machine learning": "ICML",
    # CVPR
    "cvpr": "CVPR",
    "conference on computer vision and pattern recognition": "CVPR",
    # ICCV
    "iccv": "ICCV",
    "international conference on computer vision": "ICCV",
    # ECCV
    "eccv": "ECCV",
    "european conference on computer vision": "ECCV",
    # ACL
    "acl": "ACL",
    "association for computational linguistics": "ACL",
    # EMNLP
    "emnlp": "EMNLP",
    "empirical methods in natural language processing": "EMNLP",
    # NAACL
    "naacl": "NAACL",
    "north american chapter of the association for computational linguistics": "NAACL",
    # COLING
    "coling": "COLING",
    "international conference on computational linguistics": "COLING",
    # AAAI
    "aaai": "AAAI",
    "association for the advancement of artificial intelligence": "AAAI",
    # IJCAI
    "ijcai": "IJCAI",
    "international joint conference on artificial intelligence": "IJCAI",
    # KDD
    "kdd": "KDD",
    "knowledge discovery and data mining": "KDD",
    # SIGIR
    "sigir": "SIGIR",
    "special interest group on information retrieval": "SIGIR",
    # WWW
    "www": "WWW",
    "the web conference": "WWW",
    # SIGKDD
    "sigkdd": "KDD",
    # JMLR
    "jmlr": "JMLR",
    "journal of machine learning research": "JMLR",
    # TMLR
    "tmlr": "TMLR",
    "transactions on machine learning research": "TMLR",
    # Nature
    "nature": "Nature",
    "nature machine intelligence": "Nature MI",
    # Science
    "science": "Science",
}


def normalize_venue(venue: str | None) -> str | None:
    """标准化 venue 名称。"""
    if not venue:
        return None
    v = venue.lower().strip()
    for alias, canonical in VENUE_ALIASES.items():
        if alias in v or v in alias:
            return canonical
    return venue.strip()


# ---------------------------------------------------------------------------
# Text Normalizer
# ---------------------------------------------------------------------------


class TextNormalizer:
    """
    文本标准化。

    处理：
    - 连续空格 / 换行清理
    - PDF 断句修复（句号后无空格）
    - 常见 header/footer 噪声去除
    - section 边界标记保留
    """

    # 常见 header/footer 噪声模式（正则）
    HEADER_FOOTER_PATTERNS = [
        r"^arXiv:\d{4}\.\d{4,5}(v\d+)?",  # arXiv ID
        r"^doi:\s*10\.\d+",               # DOI
        r"^\d+\s*/\s*\d+$",               # 页码（如 1/10）
        r"^\s*\d{4}\s*$",                  # 年份行
        r"Submitted to",                    # 提交信息
        r"Copyright \d{4}",                 # 版权声明
        r"^Page \d+ of",                    # Page X of Y
        r"\. Next\s+Prev\s+Page",          # PDF 阅读器导航文字
        r"^[A-Z]{10,}$",                   # 全大写字母行（通常是噪声）
    ]

    def normalize(self, text: str, page_texts: list | None = None) -> tuple[str, list[str]]:
        """
        清洗文本，返回 (cleaned_text, warnings)。

        策略：
        1. 基础清理（whitespace、换行）
        2. 断句修复（句号后无空格）
        3. 去除 header/footer 噪声
        4. 保留 section 边界
        """
        warnings: list[str] = []
        text = self._normalize_whitespace(text)
        text = self._fix_sentence_breaks(text)
        text, hw = self._remove_header_footer_noise(text)
        warnings.extend(hw)
        # section 边界已由 page_texts 的 char_start/char_end 保留，此处不再额外处理

        if len(text) < 100:
            warnings.append("标准化后文本过短，可能解析失败")

        return text, warnings

    def _normalize_whitespace(self, text: str) -> str:
        """合并连续空格、多余换行。"""
        # 合并多个空格为单个
        text = re.sub(r"[ \t]+", " ", text)
        # 合并连续换行（超过2个换行 → 2个）
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 行首尾空格
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(line for line in lines if line)

    def _fix_sentence_breaks(self, text: str) -> str:
        """修复 PDF 强行切断的句子（句号后无空格，直接跟大写字母）。"""
        # "Word.Word" → "Word. Word"  （句号紧跟下一个单词开头）
        text = re.sub(r"(\.)([A-Z])", r". \2", text)
        # 修复 "word!Word" / "word?Word"
        text = re.sub(r"(!|\?)([A-Z])", r"\1 \2", text)
        return text

    def _remove_header_footer_noise(self, text: str) -> tuple[str, list[str]]:
        """去除常见 header/footer 噪声。"""
        warnings: list[str] = []
        lines = text.split("\n")
        cleaned: list[str] = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            is_noise = False

            for pattern in self.HEADER_FOOTER_PATTERNS:
                if re.search(pattern, stripped, flags=re.IGNORECASE):
                    is_noise = True
                    break

            if not is_noise:
                # 过滤单字符/无意义行
                if len(stripped) > 2:
                    cleaned.append(line)

        result = "\n".join(cleaned)
        if len(result) < len(text) * 0.5:
            warnings.append(
                "文本经 header/footer 清洗后损失超过 50%，需确认是否过激"
            )

        return result, warnings


# ---------------------------------------------------------------------------
# Metadata Normalizer
# ---------------------------------------------------------------------------


class MetadataNormalizer:
    """
    元数据标准化。

    处理：
    - title：去空白、统一大小写、去掉版本后缀（v1 / [v2]）
    - authors：统一分隔符、去除脚注标记
    - year：强制 int
    - venue：别名规范化
    """

    def normalize(
        self,
        title: str,
        authors: list[str],
        year: int | None,
        venue: str | None,
    ) -> NormalizedMetadata:
        """
        标准化元数据，返回 NormalizedMetadata。
        """
        norm_title = self.normalize_title(title)
        norm_authors = self.normalize_authors(authors)
        norm_year = self.normalize_year(year)
        norm_venue = normalize_venue(venue)
        first_author_surname = self._extract_first_author_surname(norm_authors)

        return NormalizedMetadata(
            title=norm_title,
            authors=norm_authors,
            year=norm_year,
            venue=norm_venue,
            abstract=None,  # abstract 不在此处处理，由 parser 提取
        )

    def normalize_title(self, title: str) -> str:
        """标准化标题。"""
        if not title:
            return ""
        t = title.strip()
        # 合并多余空格
        t = re.sub(r"\s+", " ", t)
        # 去掉常见版本后缀
        t = re.sub(r"\s*v\.?\s*\d+\s*$", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*\[v\.?\s*\d+\]\s*$", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*\(revised.*?\)\s*$", "", t, flags=re.IGNORECASE)
        # 去掉首尾标点和空白
        t = t.strip(". \t\n")
        return t

    def normalize_authors(self, authors: list[str]) -> list[str]:
        """
        标准化作者列表。

        - 如果是字符串，尝试拆分
        - 去除脚注标记（[1], *1 等）
        - 去除多余空白
        """
        if isinstance(authors, str):
            authors = [a.strip() for a in re.split(r"[,;]", authors) if a.strip()]

        normalized: list[str] = []
        for author in authors:
            author = author.strip()
            # 去除脚注/机构标记（* [1] 等）
            author = re.sub(r"\s*[\[\(]?\**\d+[\]\)]?\s*$", "", author)
            author = re.sub(r"\s*\*+\s*$", "", author)
            author = re.sub(r"\s+", " ", author)
            if author and len(author) > 1:
                normalized.append(author)

        return normalized

    def normalize_year(self, year: int | str | None) -> int | None:
        """标准化年份（强制 int，过滤异常值）。"""
        if year is None:
            return None
        try:
            y = int(year) if isinstance(year, str) else int(year)
            if 1900 <= y <= 2030:
                return y
        except (ValueError, TypeError):
            pass
        return None

    def _extract_first_author_surname(self, authors: list[str]) -> str:
        """
        提取第一作者的 surname（用于 canonical key）。
        策略：取作者字符串的最后一个词作为 surname。
        """
        if not authors:
            return ""
        first = authors[0].strip()
        parts = re.split(r"[\s,]+", first)
        parts = [p for p in parts if p]
        return parts[-1] if parts else first
