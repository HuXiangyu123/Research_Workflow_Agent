"""Cross-Encoder Reranker — 对候选论文做本地语义重排。"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.corpus.search.deduper import DedupedCandidate

logger = logging.getLogger(__name__)

# ── 默认 Cross-Encoder 模型 ────────────────────────────────────────────────────

# ms-marco-MiniLM-L-6-v2：专为 MS MARCO  passage ranking 训练，
# 在学术论文检索场景表现优异，速度快（6 层 MiniLM）。
# 另有 cross-encoder/ms-marco-MiniLM-L-12-v2（更深）按需切换。
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── RerankResult ───────────────────────────────────────────────────────────────


@dataclass
class RerankResult:
    """单条 rerank 结果。"""
    doc_id: str
    canonical_id: str
    rerank_score: float      # Cross-Encoder 打分（越高越相关）
    rerank_index: int       # 在原始列表中的位置


# ── CrossEncoderReranker ──────────────────────────────────────────────────────


class CrossEncoderReranker:
    """
    本地 Cross-Encoder Reranker。

    基于 sentence-transformers 的 Cross-Encoder 模型，对候选论文做语义相关性重排。
    比 Cohere Rerank API 更适合本地部署、无 API 成本。

    推荐模型（来自 SBERT.net 官方 benchmark）：
      - cross-encoder/ms-marco-MiniLM-L-6-v2  （速度快，英文优）
      - cross-encoder/ms-marco-MiniLM-L-12-v2 （精度更高）
      - cross-encoder/ms-marco-MultiBERT-L-12 （多语言支持）

    使用方式：
        reranker = CrossEncoderReranker()          # 默认模型
        reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-12-v2")
        results = reranker.rerank(query="...", candidates=[...])
    """

    def __init__(
        self,
        model: str | None = None,
        max_length: int = 512,
        batch_size: int = 32,
        device: str | None = None,
    ):
        """
        Args:
            model: HuggingFace 模型名（默认 cross-encoder/ms-marco-MiniLM-L-6-v2）
            max_length: 最大 token 长度（超过截断）
            batch_size: 批处理大小（控制显存占用）
            device: 推理设备（None=自动选择 cuda/cpu）
        """
        self._model_name = model or os.getenv("RERANK_MODEL", DEFAULT_MODEL)
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = device
        self._model = None
        self._available = None  # None=未检测，True=可用，False=不可用

    @property
    def is_available(self) -> bool:
        """检测 reranker 是否可用（sentence-transformers 已安装且模型可加载）。"""
        if self._available is not None:
            return self._available

        try:
            from sentence_transformers import CrossEncoder
            CrossEncoder(self._model_name)
            self._available = True
        except ImportError:
            logger.warning(
                "[CrossEncoderReranker] sentence-transformers 未安装，rerank 不可用"
            )
            self._available = False
        except Exception as e:
            logger.warning(
                f"[CrossEncoderReranker] 模型 {self._model_name} 加载失败：{e}"
            )
            self._available = False

        return self._available

    def _get_model(self):
        """延迟加载模型。"""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self._model_name,
                max_length=self._max_length,
                device=self._device,
            )
            logger.info(
                f"[CrossEncoderReranker] 模型已加载：{self._model_name}，"
                f"设备={self._model.device}"
            )

        return self._model

    # ── 内部工具 ────────────────────────────────────────────────────────────────

    def _candidates_to_pairs(
        self,
        query: str,
        candidates: list["DedupedCandidate"],
    ) -> list[tuple[str, str]]:
        """
        将 DedupedCandidate 列表转换为 Cross-Encoder 输入对 (query, doc)。

        Cross-Encoder 输入格式：[query, document]，模型输出相关度分数。
        文档构造策略：
          - 标题 + 摘要（如果有），用 [SEP] 分隔
          - 截断到 max_length 限制
        """
        pairs = []
        for c in candidates:
            # 构造文档文本
            parts = []
            if c.title:
                parts.append(c.title.strip())
            if c.abstract:
                # 截断 abstract 避免过长
                abstract_text = c.abstract.strip()[:2000]
                if parts:
                    parts.append(abstract_text)
                else:
                    parts = [abstract_text]
            doc_text = " [SEP] ".join(parts) if parts else c.title or ""

            pairs.append((query, doc_text))

        return pairs

    # ── 公开 API ──────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: list["DedupedCandidate"],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """
        对候选论文做 Cross-Encoder Rerank。

        Args:
            query: 检索 query
            candidates: 待重排的 DedupedCandidate 列表
            top_n: 返回多少条（None = 返回全部，按 score 降序）

        Returns:
            RerankResult 列表（按 rerank_score 降序）
        """
        if not candidates:
            return []

        if not self.is_available:
            logger.warning(
                f"[CrossEncoderReranker] reranker 不可用（模型={self._model_name}）"
            )
            return []

        try:
            model = self._get_model()
            pairs = self._candidates_to_pairs(query, candidates)

            # Cross-Encoder.predict() 接受 list of pairs，返回 list of scores
            scores = model.predict(pairs, show_progress_bar=False, batch_size=self._batch_size)

            # 如果是单个值（标量），转成列表
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            elif not isinstance(scores, list):
                scores = [float(scores)]

            # 构建 RerankResult 并按分数降序排列
            results = [
                RerankResult(
                    doc_id=candidates[i].primary_doc_id,
                    canonical_id=candidates[i].canonical_id,
                    rerank_score=float(scores[i]),
                    rerank_index=i,
                )
                for i in range(len(candidates))
            ]

            results.sort(key=lambda r: r.rerank_score, reverse=True)

            if top_n is not None:
                results = results[:top_n]

            logger.info(
                f"[CrossEncoderReranker] query='{query[:30]}' "
                f"candidates={len(candidates)} returned={len(results)} "
                f"model={self._model_name}"
            )

            return results

        except Exception as e:
            logger.error(f"[CrossEncoderReranker] rerank 失败：{e}")
            return []

    def rerank_with_fusion(
        self,
        query: str,
        candidates: list["DedupedCandidate"],
        fusion_weights: tuple[float, float] = (0.4, 0.6),
        top_n: int | None = None,
    ) -> list["DedupedCandidate"]:
        """
        Rerank 并与 RRF 分数做加权融合。

        Args:
            query: 检索 query
            candidates: 待重排的 DedupedCandidate 列表
            fusion_weights: (rrf_weight, rerank_weight)，默认 (0.4, 0.6)
                            即 60% 依赖 rerank 分数，40% 保留 recall 信号
            top_n: 进入 rerank 的候选数量上限（None = 全部）

        Returns:
            融合后的 DedupedCandidate 列表（按 final_score 降序）
        """
        if not candidates:
            return []

        rrf_weight, rerank_weight = fusion_weights

        # RRF 归一化（以最高分为基准）
        max_rrf = max((c.rrf_score for c in candidates), default=1.0)
        rrf_normalized = {
            c.canonical_id: c.rrf_score / max_rrf for c in candidates
        }

        # 如果 top_n 限制了进入 rerank 的候选数，先截断
        pool = candidates
        if top_n is not None and len(candidates) > top_n:
            pool = sorted(candidates, key=lambda c: c.rrf_score, reverse=True)[:top_n]
            logger.info(
                f"[CrossEncoderReranker] RRF 截断 top_n={top_n}，"
                f"原始候选={len(candidates)}"
            )

        # Cross-Encoder rerank
        rerank_results = self.rerank(query, pool)
        rerank_map = {r.canonical_id: r for r in rerank_results}

        if not rerank_results:
            # Rerank 失败时退化为纯 RRF 排序
            logger.warning(
                "[CrossEncoderReranker] rerank 返回空，退化为 RRF 排序"
            )
            for c in candidates:
                c.final_score = c.rrf_score
                c.rerank_score = None
            return sorted(candidates, key=lambda x: x.rrf_score, reverse=True)

        # 归一化 rerank 分数（Cross-Encoder 分数范围不固定，以 max 为基准）
        max_rerank = max((r.rerank_score for r in rerank_results), default=1.0)
        rerank_normalized = {
            cid: result.rerank_score / max_rerank
            for cid, result in rerank_map.items()
        }

        # 加权融合，写入候选对象
        for c in candidates:
            rrf_n = rrf_normalized.get(c.canonical_id, 0.0)
            rerank_n = rerank_normalized.get(c.canonical_id, 0.0)
            c.rerank_score = rerank_n
            c.final_score = rrf_weight * rrf_n + rerank_weight * rerank_n

        return sorted(candidates, key=lambda c: c.final_score, reverse=True)
