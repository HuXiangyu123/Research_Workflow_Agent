"""Chunkers — Module 2：论文层级切块（Structure Detection + Coarse + Fine + Linker）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.corpus.models import (
    ChunkSpan,
    CoarseChunk,
    FineChunk,
    PageText,
    StandardizedDocument,
)

# ---------------------------------------------------------------------------
# Section Patterns
# ---------------------------------------------------------------------------

# 常见英文论文 section 标题正则（不区分大小写）
# 支持：Abstract, Introduction, Related Work, Methods, Methodology,
#       Experiments, Results, Discussion, Conclusion, References, Supplementary
SECTION_PATTERNS_COMMON = [
    r"(?i)^(?:abstract|acknowledgements|acknowledgments)\s*$",
    r"(?i)^(?:introduction|1\.?\s*Introduction)\s*$",
    r"(?i)^(?:related\s*work|background|2\.?\s*Related\s*Work)\s*$",
    r"(?i)^(?:preliminaries|3\.?\s*Preliminaries)\s*$",
    r"(?i)^(?:method(?:ology)?s?|model|4\.?\s*Method(?:ology)?s?)\s*$",
    r"(?i)^(?:experiments?(?:\s*and\s*results?)?|evaluation|5\.?\s*Experiments?)\s*$",
    r"(?i)^(?:results?(?:\s*and\s*discussion?)?|analysis|6\.?\s*Results?)\s*$",
    r"(?i)^(?:discussion|7\.?\s*Discussion)\s*$",
    r"(?i)^(?:conclusion(?:\s*and\s*future\s*work)?|conclusions?|8\.?\s*Conclusion)\s*$",
    r"(?i)^(?:references?|bibliography)\s*$",
    r"(?i)^(?:appendix|supplementary\s*material)\s*$",
]

# 用于检测"标题行"的启发式特征：
# 1. 全部或大部分单词首字母大写
# 2. 不以句号结尾
# 3. 比正文字符短
# 4. 前后有空行
TITLE_LINE_INDICATORS = [
    lambda line: _is_title_case(line),
    lambda line: len(line) < 120 and not line.endswith("."),
    lambda line: not re.search(r"\b(is|are|was|were|has|have|will|can|could|would)\b", line.lower()),
]


# ---------------------------------------------------------------------------
# Dataclasses: Detected Structure
# ---------------------------------------------------------------------------


@dataclass
class Paragraph:
    """检测到的段落。"""

    text: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    is_title: bool = False
    section: str = ""  # 所属章节


@dataclass
class DetectedSection:
    """检测到的章节。"""

    heading: str
    level: int           # 1=一级 2=二级...
    paragraphs: list[Paragraph]
    page_start: int
    page_end: int
    char_start: int
    char_end: int

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs if not p.is_title)


@dataclass
class PaperStructure:
    """
    检测到的论文结构。
    """

    title: str
    abstract: str | None
    sections: list[DetectedSection]
    page_boundaries: list[int]          # 每个 PageText 的 char_start 列表
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_title_case(text: str) -> bool:
    """判断是否为 Title Case（每个实词首字母大写，但不全大写）。"""
    words = text.split()
    if len(words) < 2:
        return False
    # 全大写不是 Title Case
    if text.isupper():
        return False
    upper_words = sum(1 for w in words if w[0].isupper())
    return upper_words / len(words) > 0.6


def _is_heading_line(line: str) -> bool:
    """判断某行是否为章节标题。"""
    stripped = line.strip()
    if not stripped or len(stripped) > 150:
        return False
    # 不能是纯数字
    if re.match(r"^\d+\.?\s*$", stripped):
        return False
    # 不能有句号结尾（正文句子通常以句号/问号/感叹号结尾）
    if re.search(r"[.!?]\s*$", stripped) and len(stripped) > 40:
        return False
    # 匹配已知 section 关键词
    for pat in SECTION_PATTERNS_COMMON:
        if re.search(pat, stripped):
            return True
    # 通用标题判断：Title Case + 短行
    if _is_title_case(stripped) and len(stripped) < 100:
        return True
    # 数字编号开头（1. 2. 1.1 2.3 等）
    if re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", stripped):
        return True
    return False


def _normalize_section_name(heading: str) -> str:
    """将章节标题标准化为小写 slug。"""
    slug = heading.lower().strip()
    # 去掉前导编号（"1. Section" 或 "1 Section" 格式）
    slug = re.sub(r"^\d+\.\s*", "", slug)
    slug = re.sub(r"^\d+\s+", "", slug)
    # 多余的点号（如 "1. " 后可能残留）
    slug = re.sub(r"^\.\s*", "", slug)
    # 空格变下划线
    slug = re.sub(r"\s+", "_", slug)
    # 映射常见名称（覆盖各种写法变体）
    section_map = {
        # Abstract / Acknowledgements
        "abstract": "abstract",
        "acknowledgements": "acknowledgements",
        # Introduction
        "introduction": "introduction",
        # Background / Related
        "related_work": "related_work",
        "related_work_and_background": "related_work",
        "background_and_related_work": "related_work",
        "background": "background",
        # Preliminaries
        "preliminaries": "preliminaries",
        # Methods
        "methodology": "methods",
        "methodology_and_model": "methods",
        "methods": "methods",
        "method": "methods",
        "model": "methods",
        "model_and_methodology": "methods",
        # Experiments
        "experiments": "experiments",
        "experiments_and_results": "experiments",
        "experiments_and_evaluation": "experiments",
        "evaluation": "experiments",
        "experimental_results": "experiments",
        # Results
        "results": "results",
        "results_and_analysis": "results",
        "analysis": "results",
        # Discussion
        "discussion": "discussion",
        # Conclusion
        "conclusion": "conclusion",
        "conclusions": "conclusion",
        "conclusion_and_future_work": "conclusion",
        "conclusions_and_future_work": "conclusion",
        "conclusion_and_limitations": "conclusion",
        "conclusions_and_limitations": "conclusion",
        # References
        "references": "references",
        "bibliography": "references",
        # Appendix
        "appendix": "appendix",
        "appendices": "appendix",
        "supplementary": "appendix",
        "supplementary_materials": "appendix",
    }
    return section_map.get(slug, slug)


def _split_into_paragraphs(
    text: str, page_start: int, page_end: int, char_offset: int
) -> list[Paragraph]:
    """
    将一段文本切分成段落。
    策略：按双换行分段，过长的段落再按句子边界拆。
    """
    paragraphs: list[Paragraph] = []
    # 双换行或大段空白分割
    raw_paras = re.split(r"\n{2,}|\n(?=\s{4,})", text)
    cur_char = char_offset

    for raw in raw_paras:
        # 清理单换行（PDF 有时不换段）
        para_text = re.sub(r"(?<=[a-z])\n(?=[a-zA-Z])", " ", raw).strip()
        para_text = re.sub(r"\s+", " ", para_text)
        if not para_text or len(para_text) < 20:
            cur_char += len(raw) + 2
            continue

        char_start = cur_char
        char_end = char_start + len(para_text)
        paragraphs.append(
            Paragraph(
                text=para_text,
                page_start=page_start,
                page_end=page_end,
                char_start=char_start,
                char_end=char_end,
            )
        )
        cur_char = char_end + 2

    return paragraphs


# ---------------------------------------------------------------------------
# Structure Detector
# ---------------------------------------------------------------------------


class StructureDetector:
    """
    从 StandardizedDocument 检测论文结构。

    输出：
    - PaperStructure（包含 title / abstract / sections / page_boundaries）
    """

    def __init__(self, min_section_paragraphs: int = 1):
        """
        Args:
            min_section_paragraphs: 章节最少段落数（用于过滤噪声）
        """
        self._min_section_paragraphs = min_section_paragraphs

    def detect(self, doc: StandardizedDocument, page_texts: list[PageText]) -> PaperStructure:
        """
        检测论文结构。
        """
        warnings: list[str] = []

        # 1. 提取标题（优先用 doc.title）
        title = self._extract_title(doc, page_texts)

        # 2. 构建 page boundaries（char offset 映射）
        page_boundaries = [pt.char_start for pt in page_texts]

        # 3. 检测章节 + 段落
        sections, abstract, section_warnings = self._detect_sections(
            doc.normalized_text, page_texts
        )
        warnings.extend(section_warnings)

        return PaperStructure(
            title=title,
            abstract=abstract,
            sections=sections,
            page_boundaries=page_boundaries,
            warnings=warnings,
        )

    def _extract_title(self, doc: StandardizedDocument, page_texts: list[PageText]) -> str:
        """提取论文标题（优先用 doc.title）。"""
        if doc.title and doc.title not in ("Unknown", "Unknown Title"):
            return doc.title

        # 从首页文本提取
        if page_texts:
            first_page = page_texts[0].text
            lines = [l.strip() for l in first_page.split("\n") if l.strip()]
            for line in lines[:5]:
                if _is_title_case(line) and len(line) < 150:
                    return line
        return "Unknown Title"

    def _detect_sections(
        self, text: str, page_texts: list[PageText]
    ) -> tuple[list[DetectedSection], str | None, list[str]]:
        """
        检测所有章节。
        返回：(sections, abstract, warnings)
        """
        warnings: list[str] = []
        lines = text.split("\n")
        sections: list[DetectedSection] = []
        current_heading = "introduction"
        current_level = 1
        current_paragraphs: list[Paragraph] = []
        char_offset = 0

        # 合并 char offset 信息
        page_offsets: list[tuple[int, int]] = []
        for pt in page_texts:
            page_offsets.append((pt.char_start, pt.page_num))

        def _get_page(char_pos: int) -> int:
            """根据字符位置找到页码。"""
            for i in range(len(page_offsets) - 1, -1, -1):
                if char_pos >= page_offsets[i][0]:
                    return page_offsets[i][1]
            return 1

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if _is_heading_line(stripped):
                # 保存前一个章节
                if current_paragraphs:
                    self._flush_section(
                        sections, current_heading, current_level,
                        current_paragraphs, warnings
                    )
                    current_paragraphs = []

                current_heading = _normalize_section_name(stripped)
                current_level = self._detect_section_level(stripped)
            elif stripped:
                para_text = stripped
                page = _get_page(char_offset)
                current_paragraphs.append(
                    Paragraph(
                        text=para_text,
                        page_start=page,
                        page_end=page,
                        char_start=char_offset,
                        char_end=char_offset + len(stripped),
                    )
                )

            char_offset += len(line) + 1
            i += 1

        # 最后一个章节
        if current_paragraphs:
            self._flush_section(
                sections, current_heading, current_level,
                current_paragraphs, warnings
            )

        # 提取 abstract
        abstract_section = next(
            (s for s in sections if s.heading == "abstract"), None
        )
        abstract = abstract_section.full_text if abstract_section else None

        # 过滤空 section
        sections = [s for s in sections if len(s.paragraphs) >= self._min_section_paragraphs]

        if not sections:
            warnings.append("未能检测到任何章节，文本可能格式异常")

        return sections, abstract, warnings

    def _flush_section(
        self,
        sections: list[DetectedSection],
        heading: str,
        level: int,
        paragraphs: list[Paragraph],
        warnings: list[str],
    ) -> None:
        """将当前累积的段落 flush 为一个 DetectedSection。"""
        if not paragraphs:
            return
        pages = [p.page_start for p in paragraphs]
        chars = [p.char_start for p in paragraphs]
        char_ends = [p.char_end for p in paragraphs if p.text]
        section = DetectedSection(
            heading=heading,
            level=level,
            paragraphs=paragraphs,
            page_start=min(pages),
            page_end=max(pages),
            char_start=min(chars),
            char_end=max(char_ends) if char_ends else 0,
        )
        sections.append(section)

    def _detect_section_level(self, heading: str) -> int:
        """检测章节层级（1=一级，2=二级等）。"""
        # 数字编号 "1." / "1.1" 等
        if re.match(r"^\d+\.\d+\s+", heading):
            return 2
        if re.match(r"^\d+\.\s+", heading):
            return 1
        # 全大写通常是重要章节（一级）
        if heading.isupper():
            return 1
        return 1  # 默认一级
