"""Fine Chunker — 在 coarse chunk 内细切，服务于 evidence retrieval。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.corpus.ingest.chunkers import Paragraph
from src.corpus.models import CoarseChunk, FineChunk

# ---------------------------------------------------------------------------
# Fine Chunk Config
# ---------------------------------------------------------------------------

# 目标字符数（估算 tokens = chars * 0.25）
_FINE_MIN_CHARS = 80
_FINE_MAX_CHARS = 800
_FINE_TARGET_CHARS = 350
_FINE_OVERLAP_CHARS = 60       # 前后 overlap 字符数


# ---------------------------------------------------------------------------
# Sentence Splitter
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """
    将文本按句子分割。

    策略：
    - 按 [.!?] + 空格/换行 分割
    - 保留分割符
    - 过滤空片段
    """
    # 匹配句子结束：句号/问号/感叹号 + 空格/换行/文档末尾
    pattern = r"(?<=[.!?])\s+"
    raw_sentences = re.split(pattern, text)
    sentences: list[str] = []
    for s in raw_sentences:
        s = s.strip()
        if s:
            sentences.append(s)
    return sentences


# ---------------------------------------------------------------------------
# Fine Chunker
# ---------------------------------------------------------------------------


class FineChunker:
    """
    将 CoarseChunk[] 转换为 FineChunk[]。

    切法策略：
    1. 优先按 paragraph 切（最稳定）
    2. 长 paragraph 用 sliding window（2~5 句一组）
    3. 带轻量 overlap（前后各 60 chars ≈ 1~2 句）
    4. 过短的 chunk 合并到前一个
    """

    def chunk(
        self,
        coarse_chunks: list[CoarseChunk],
    ) -> list[FineChunk]:
        """
        在 coarse chunks 内生成 fine chunks。

        Args:
            coarse_chunks: CoarseChunk 列表

        Returns:
            list[FineChunk]
        """
        all_fine: list[FineChunk] = []

        for coarse in coarse_chunks:
            fine_chunks = self._chunk_coarse(coarse)
            all_fine.extend(fine_chunks)

        # 重排 order
        for i, chunk in enumerate(all_fine):
            chunk.order = i

        return all_fine

    def _chunk_coarse(self, coarse: CoarseChunk) -> list[FineChunk]:
        """将单个 coarse chunk 细分为 fine chunks。"""
        text = coarse.text
        if not text or len(text) < _FINE_MIN_CHARS:
            # 过短文本：直接作为单个 fine chunk
            return [self._make_fine(coarse, text, 0)]

        # 按段落分割（双换行）
        paragraphs = self._split_by_paragraphs(text)
        if len(paragraphs) == 1:
            # 单段落：按 sentence sliding window
            return self._chunk_by_sentence_window(coarse, text)
        else:
            # 多段落：每个段落一个 fine chunk（长段落再切）
            return self._chunk_paragraphs(coarse, paragraphs)

    def _split_by_paragraphs(self, text: str) -> list[str]:
        """按双换行或单换行 + 缩进切分段落。"""
        # 双换行优先
        parts = re.split(r"\n{2,}", text)
        if len(parts) > 1:
            return [p.strip() for p in parts if p.strip()]
        # 单换行 + 短行（可能是段落内换行）
        parts = re.split(r"(?<=[a-z])\n(?=[A-Z])", text)
        result: list[str] = []
        buf = ""
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if len(p) < 200 and not re.search(r"[.!?]$", p):
                # 可能是段落内换行，合并
                buf += " " + p
            else:
                if buf:
                    result.append(buf.strip())
                    buf = ""
                result.append(p)
        if buf:
            result.append(buf.strip())
        return [r for r in result if r]

    def _chunk_paragraphs(
        self, coarse: CoarseChunk, paragraphs: list[str]
    ) -> list[FineChunk]:
        """多段落时的细切策略。"""
        fine_chunks: list[FineChunk] = []
        order = 0

        for para_text in paragraphs:
            if len(para_text) > _FINE_MAX_CHARS:
                # 长段落：sentence window
                sub_chunks = self._chunk_by_sentence_window(coarse, para_text)
                for fc in sub_chunks:
                    fc.order = order
                    order += 1
                fine_chunks.extend(sub_chunks)
            else:
                # 正常段落：直接成 fine chunk
                fine_chunks.append(self._make_fine(coarse, para_text, order))
                order += 1

        return fine_chunks

    def _chunk_by_sentence_window(
        self, coarse: CoarseChunk, text: str
    ) -> list[FineChunk]:
        """
        按 sentence sliding window 细切。

        策略：
        - 2~5 句为一组
        - 达到目标字符数时切
        - 前后带轻量 overlap
        """
        fine_chunks: list[FineChunk] = []
        sentences = _split_sentences(text)
        if not sentences:
            return [self._make_fine(coarse, text, 0)]

        if len(sentences) == 1:
            return [self._make_fine(coarse, text, 0)]

        # Sliding window
        group: list[str] = []
        group_chars = 0
        order = 0
        i = 0

        while i < len(sentences):
            s = sentences[i]
            s_chars = len(s)

            # 强制开启新 chunk：单句超长
            if s_chars > _FINE_MAX_CHARS:
                if group:
                    # 先输出当前 group
                    chunk_text = " ".join(group)
                    fine_chunks.append(self._make_fine(coarse, chunk_text, order))
                    order += 1
                    group = []
                    group_chars = 0
                # 超长句单独成 chunk
                fine_chunks.append(self._make_fine(coarse, s, order))
                order += 1
                i += 1
                continue

            # 达到目标大小，开启新 chunk
            if group_chars + s_chars > _FINE_TARGET_CHARS and group:
                chunk_text = " ".join(group)
                fine_chunks.append(self._make_fine(coarse, chunk_text, order))
                order += 1
                # overlap：保留最后 1 句
                overlap_sentence = group[-1]
                group = [overlap_sentence]
                group_chars = len(overlap_sentence)
                continue

            group.append(s)
            group_chars += s_chars

            # 强制 5 句上限
            if len(group) >= 5 and group:
                chunk_text = " ".join(group)
                fine_chunks.append(self._make_fine(coarse, chunk_text, order))
                order += 1
                # overlap
                overlap_sentence = group[-1]
                group = [overlap_sentence]
                group_chars = len(overlap_sentence)

            i += 1

        # 最后一个 group
        if group:
            chunk_text = " ".join(group)
            fine_chunks.append(self._make_fine(coarse, chunk_text, order))

        return fine_chunks

    def _make_fine(
        self,
        coarse: CoarseChunk,
        text: str,
        order: int,
        char_start: int | None = None,
        char_end: int | None = None,
    ) -> FineChunk:
        """构建 FineChunk。"""
        # 估算 char offset（基于文本位置比例）
        if char_start is None:
            ratio = order / max(coarse.token_count, 1) if coarse.token_count else 0
            char_start = int(coarse.char_start + (coarse.char_end - coarse.char_start) * ratio)
        if char_end is None:
            char_end = char_start + len(text)

        return FineChunk(
            chunk_id="",
            doc_id=coarse.doc_id,
            canonical_id=coarse.canonical_id,
            parent_coarse_chunk_id=coarse.chunk_id,
            section=coarse.section,
            page_start=coarse.page_start,
            page_end=coarse.page_end,
            char_start=char_start,
            char_end=char_end,
            text=text,
            order=order,
            meta_info={
                "coarse_order": coarse.order,
                "coarse_section": coarse.section,
            },
        )
