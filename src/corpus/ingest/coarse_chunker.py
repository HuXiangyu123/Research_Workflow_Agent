"""Coarse Chunker — 按 section 生成粗粒度块，服务于 paper-level retrieval。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.corpus.ingest.chunkers import (
    DetectedSection,
    PaperStructure,
    Paragraph,
    _is_heading_line,
    _normalize_section_name,
    _split_into_paragraphs,
)
from src.corpus.models import CoarseChunk, PageText


# ---------------------------------------------------------------------------
# Coarse Chunk Config
# ---------------------------------------------------------------------------

# 目标 token 数（估算：chars * 0.25 ≈ tokens）
_COARSE_MIN_CHARS = 500
_COARSE_MAX_CHARS = 4000
_COARSE_TARGET_CHARS = 1500


# ---------------------------------------------------------------------------
# Coarse Chunker
# ---------------------------------------------------------------------------


class CoarseChunker:
    """
    将 PaperStructure 转换为 CoarseChunk[]。

    切法策略：
    - abstract / conclusion 各自单独成块
    - introduction / related_work 按 paragraph group 分组（2~4 段一块）
    - methods / experiments 按 subsection 分块
    - 超出上限的长 section 在子段落边界二次切分
    - 禁止跨 section 切分
    """

    def chunk(self, structure: PaperStructure, doc_id: str, canonical_id: str | None) -> list[CoarseChunk]:
        """
        生成 coarse chunks。

        Args:
            structure: StructureDetector 输出的论文结构
            doc_id: 文档 ID
            canonical_id: 论文身份 ID

        Returns:
            list[CoarseChunk]
        """
        chunks: list[CoarseChunk] = []
        order = 0

        for section in structure.sections:
            section_chunks = self._chunk_section(section, doc_id, canonical_id, order)
            chunks.extend(section_chunks)
            order += len(section_chunks)

        # 确保 order 正确
        for i, chunk in enumerate(chunks):
            chunk.order = i

        return chunks

    def _chunk_section(
        self,
        section: DetectedSection,
        doc_id: str,
        canonical_id: str | None,
        base_order: int,
    ) -> list[CoarseChunk]:
        """将单个 section 转换为 1+ 个 coarse chunk。"""
        section_name = _normalize_section_name(section.heading)
        paragraphs = section.paragraphs

        # 特殊处理：abstract / conclusion 单独成一块
        if section_name in ("abstract", "conclusion", "references", "appendix"):
            return [self._build_single_chunk(
                section, doc_id, canonical_id, section_name, base_order
            )]

        # 过短 section：直接成一块
        full_text = section.full_text
        if len(full_text) < _COARSE_MIN_CHARS:
            return [self._build_single_chunk(
                section, doc_id, canonical_id, section_name, base_order
            )]

        # 正常 section：按段落分组
        return self._chunk_by_paragraph_groups(
            section, doc_id, canonical_id, section_name, base_order
        )

    def _build_single_chunk(
        self,
        section: DetectedSection,
        doc_id: str,
        canonical_id: str | None,
        section_name: str,
        order: int,
    ) -> CoarseChunk:
        """将整个 section 作为单个 chunk。"""
        section_name = _normalize_section_name(section.heading)
        paragraphs = section.paragraphs
        text = section.full_text

        return CoarseChunk(
            chunk_id="",  # auto-generate in __post_init__
            doc_id=doc_id,
            canonical_id=canonical_id,
            section=section_name,
            section_level=section.level,
            page_start=section.page_start,
            page_end=section.page_end,
            char_start=section.char_start,
            char_end=section.char_end,
            text=text,
            order=order,
    meta_info={
        "heading": section.heading,
        "num_paragraphs": len(paragraphs),
    },
        )

    def _chunk_by_paragraph_groups(
        self,
        section: DetectedSection,
        doc_id: str,
        canonical_id: str | None,
        section_name: str,
        base_order: int,
    ) -> list[CoarseChunk]:
        """
        将 section 的段落分组，每组作为一个 coarse chunk。

        分组策略：
        - 每组包含 2~5 个 paragraph
        - 累积字符数达到目标范围时停止，开启新 chunk
        - 段落组尽量保持语义连贯（不在段落中间切断）
        """
        chunks: list[CoarseChunk] = []
        paragraphs: list[Paragraph] = section.paragraphs
        if not paragraphs:
            return []

        group: list[Paragraph] = []
        group_chars = 0
        group_order = base_order

        for para in paragraphs:
            para_chars = len(para.text)

            # 判断是否开启新 chunk
            if (group_chars + para_chars > _COARSE_MAX_CHARS and group) or (
                group_chars >= _COARSE_TARGET_CHARS
                and group_chars + para_chars > _COARSE_TARGET_CHARS * 1.2
            ):
                # 生成当前 chunk
                chunk = self._build_chunk_from_group(
                    section, group, doc_id, canonical_id, section_name, group_order
                )
                chunks.append(chunk)
                group = []
                group_chars = 0
                group_order += 1

            group.append(para)
            group_chars += para_chars

            # 强制单段落超大时的兜底处理
            if para_chars > _COARSE_MAX_CHARS:
                if len(group) > 1:
                    group.pop()
                    group_chars -= para_chars
                    chunk = self._build_chunk_from_group(
                        section, group, doc_id, canonical_id, section_name, group_order
                    )
                    chunks.append(chunk)
                    group = []
                    group_chars = 0
                    group_order += 1
                # 超长段落单独成 chunk
                chunks.append(self._build_single_para_chunk(
                    section, para, doc_id, canonical_id, section_name, group_order
                ))
                group = []
                group_chars = 0
                group_order += 1

        # 最后一个 group
        if group:
            chunk = self._build_chunk_from_group(
                section, group, doc_id, canonical_id, section_name, group_order
            )
            chunks.append(chunk)

        return chunks

    def _build_chunk_from_group(
        self,
        section: DetectedSection,
        group: list,
        doc_id: str,
        canonical_id: str | None,
        section_name: str,
        order: int,
    ) -> CoarseChunk:
        """从段落组构建单个 chunk。"""
        texts = [p.text for p in group]
        text = "\n\n".join(texts)
        pages = [p.page_start for p in group]
        chars = [p.char_start for p in group]
        char_start = min(chars)
        char_end = max(p.char_end for p in group)

        return CoarseChunk(
            chunk_id="",
            doc_id=doc_id,
            canonical_id=canonical_id,
            section=section_name,
            section_level=section.level,
            page_start=min(pages),
            page_end=max(pages),
            char_start=char_start,
            char_end=char_end,
            text=text,
            order=order,
            meta_info={
                "heading": section.heading,
                "num_paragraphs": len(group),
            },
        )

    def _build_single_para_chunk(
        self,
        section: DetectedSection,
        para,
        doc_id: str,
        canonical_id: str | None,
        section_name: str,
        order: int,
    ) -> CoarseChunk:
        """从单个段落构建 chunk（用于超长段落兜底）。"""
        return CoarseChunk(
            chunk_id="",
            doc_id=doc_id,
            canonical_id=canonical_id,
            section=section_name,
            section_level=section.level,
            page_start=para.page_start,
            page_end=para.page_end,
            char_start=para.char_start,
            char_end=para.char_end,
            text=para.text,
            order=order,
            meta_info={"heading": section.heading, "num_paragraphs": 1},
        )
