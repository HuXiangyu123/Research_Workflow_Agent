"""Extract node — Phase 2: 从 RagResult.paper_candidates 抽取 PaperCards。

Progressive Reading 策略（参考 DeepXiv）：
1. DeepXiv brief（优先）：每篇论文先尝试 get_paper_brief，获取 TLDR + keywords + GitHub URL
2. LLM 抽取（次级）：brief 失败时，用 LLM 批量抽取 structured 信息
3. Fallback（兜底）：都失败时，用 _simple_card 保留原始 abstract

每张 PaperCard 包含：
- title, authors, arxiv_id, url
- summary（DeepXiv brief.tldr 或 LLM 摘要）
- keywords（DeepXiv brief.keywords 或规则提取）
- methods, datasets, limitations
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from src.tasking.trace_wrapper import get_trace_store, trace_node

logger = logging.getLogger(__name__)

_PLACEHOLDER_STRINGS = {
    "",
    "unknown",
    "unknown author",
    "unknown authors",
    "n/a",
    "na",
    "none",
    "null",
    "untitled",
}

# extract_node 处理论文数量软上限。实际选择改为预算驱动，而不是固定前 N 篇。
MAX_EXTRACT_CANDIDATES_HARD = 42
MAX_EXTRACT_EVIDENCE_CHARS = 60000
TARGET_FULLTEXT_RATIO = 0.70


@trace_node(node_name="extract", stage="search", store=get_trace_store())
def extract_node(state: dict) -> dict:
    """
    Phase 2 抽取节点。

    输入：state.rag_result.paper_candidates, state.brief
    输出：state.paper_cards（list[dict]）
    """
    rag_result = state.get("rag_result")
    brief = state.get("brief")

    # 兼容 dict 和 RagResult 对象
    if isinstance(rag_result, dict):
        candidates = rag_result.get("paper_candidates", [])
    elif hasattr(rag_result, "paper_candidates"):
        candidates = rag_result.paper_candidates
    else:
        candidates = []

    if not candidates:
        return {"paper_cards": []}

    total = len(candidates)
    candidates = _select_candidates_for_extract(
        candidates,
        max_candidates_hard=MAX_EXTRACT_CANDIDATES_HARD,
        evidence_char_budget=MAX_EXTRACT_EVIDENCE_CHARS,
        min_fulltext_ratio=TARGET_FULLTEXT_RATIO,
    )
    selected_fulltext = sum(1 for cand in candidates if _cand_has_fulltext(cand))
    logger.info(
        "[extract_node] selected %d/%d candidates for extraction (fulltext=%d, ratio=%.0f%%)",
        len(candidates),
        total,
        selected_fulltext,
        (selected_fulltext / len(candidates) * 100.0) if candidates else 0.0,
    )

    # 批量 LLM 抽取（并行）
    cards = _extract_cards_batch(candidates, brief)
    logger.info("[extract_node] extracted %d paper cards", len(cards))
    return {"paper_cards": cards}


def _select_candidates_for_extract(
    candidates: list[Any],
    *,
    max_candidates_hard: int,
    evidence_char_budget: int,
    min_fulltext_ratio: float,
) -> list[Any]:
    """优先选择有正文证据的候选，并按证据预算动态保留更多论文。"""
    if not candidates:
        return []

    ranked = sorted(
        candidates,
        key=lambda cand: (
            1 if _cand_has_fulltext(cand) else 0,
            _candidate_score(cand),
        ),
        reverse=True,
    )

    desired_total = min(max_candidates_hard, len(ranked))
    fulltext_total = sum(1 for cand in ranked if _cand_has_fulltext(cand))

    if fulltext_total == 0:
        return _select_under_budget(ranked, evidence_char_budget, desired_total)

    budget_selected = _select_under_budget(ranked, evidence_char_budget, desired_total)
    if not budget_selected:
        budget_selected = ranked[: min(8, desired_total)]

    required_fulltext = math.ceil(len(budget_selected) * min_fulltext_ratio)
    if sum(1 for cand in budget_selected if _cand_has_fulltext(cand)) >= required_fulltext:
        return budget_selected

    # 正文不足时，删掉最弱的非正文项，而不是简单固定截断前 N。
    selected = list(budget_selected)
    while selected:
        fulltext_selected = sum(1 for cand in selected if _cand_has_fulltext(cand))
        required = math.ceil(len(selected) * min_fulltext_ratio)
        if fulltext_selected >= required:
            break
        drop_idx = next(
            (idx for idx in range(len(selected) - 1, -1, -1) if not _cand_has_fulltext(selected[idx])),
            None,
        )
        if drop_idx is None:
            break
        selected.pop(drop_idx)

    return selected or budget_selected


def _select_under_budget(
    ranked: list[Any],
    evidence_char_budget: int,
    desired_total: int,
) -> list[Any]:
    selected: list[Any] = []
    used_chars = 0
    min_keep = min(12, desired_total)

    for cand in ranked:
        evidence_chars = _candidate_evidence_chars(cand)
        would_exceed = used_chars + evidence_chars > evidence_char_budget
        if selected and would_exceed and len(selected) >= min_keep:
            continue
        selected.append(cand)
        used_chars += evidence_chars
        if len(selected) >= desired_total:
            break

    return selected


def _candidate_score(cand: Any) -> float:
    """读取候选分数（用于稳定排序）。"""
    score_keys = ("combined_score", "coarse_score", "score")
    for key in score_keys:
        value = _cand_get(cand, key, None)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return 0.0


def _cand_get(cand: Any, key: str, default: Any = None) -> Any:
    if isinstance(cand, dict):
        return cand.get(key, default)
    return getattr(cand, key, default)


def _cand_has_fulltext(cand: Any) -> bool:
    if _cand_get(cand, "fulltext_available", False):
        return True
    snippets = _cand_get(cand, "fulltext_snippets", None)
    return isinstance(snippets, list) and len(snippets) > 0


def _candidate_evidence_chars(cand: Any) -> int:
    snippets = _extract_fulltext_snippets(cand)
    if snippets:
        return sum(len(item.get("text", "")) for item in snippets[:4]) or 1200
    abstract = str(_cand_get(cand, "abstract", "") or _cand_get(cand, "summary", "") or "")
    return max(400, min(len(abstract), 1800))


def _extract_cards_batch(candidates: list[Any], brief: Any | None) -> list[dict]:
    """
    Progressive Reading 策略（参考 DeepXiv）：

    1. DeepXiv brief（优先）：所有有 arxiv_id 的论文并行获取 TLDR + keywords + github
    2. LLM 抽取（次级）：用 LLM 批量抽取 structured 信息
    3. Fallback（兜底）：都失败时用 _simple_card

    每批 3 篇并行 LLM 抽取，避免 token 超限。
    """
    from src.agent.llm import build_reason_llm
    from src.agent.settings import get_settings
    from langchain_core.messages import HumanMessage, SystemMessage

    settings = get_settings()
    brief_ctx = _build_brief_context(brief)

    # ── Step 1：DeepXiv brief（Progressive reading 第一步）─────────────────
    brief_map: dict[str, dict] = {}
    arxiv_ids_for_brief: list[str] = []
    for cand in candidates:
        aid = None
        if isinstance(cand, dict):
            aid = cand.get("arxiv_id") or ""
        else:
            aid = getattr(cand, "arxiv_id", "") or ""
        if aid and len(aid) >= 6:
            arxiv_ids_for_brief.append(aid)

    if arxiv_ids_for_brief:
        logger.info("[extract_node] fetching DeepXiv briefs for %d papers", len(arxiv_ids_for_brief))
        try:
            from src.tools.deepxiv_client import batch_get_briefs
            brief_map = batch_get_briefs(arxiv_ids_for_brief, max_workers=4, delay_per_request=0.3)
        except Exception as e:
            logger.warning("[extract_node] DeepXiv brief batch failed: %s", e)

    # ── Step 2：LLM 批量抽取（每批 3 篇）─────────────────────────────────
    SYSTEM_PROMPT = (
        "You are a structured academic paper extraction expert.\n"
        "You will receive a batch of papers. Return a JSON array with EXACTLY one PaperCard object per paper, "
        "in the SAME order as the input papers.\n"
        "IMPORTANT: preserve original metadata exactly whenever title/authors/arxiv_id/url are provided; "
        "do not leave them empty or replace them with null or placeholders.\n"
        "Return strictly valid JSON with no markdown code fences. PaperCard field guide:\n"
        "{\n"
        '  "title": "Paper title (required, non-empty)",\n'
        '  "authors": ["Author 1", "Author 2"] (required; keep the provided authors whenever available),\n'
        '  "published_year": 2024,\n'
        '  "venue": "Conference or journal name",\n'
        '  "arxiv_id": "2412.00001",\n'
        '  "url": "https://...",\n'
        '  "summary": "Evidence-grounded summary within roughly 300 words",\n'
        '  "keywords": ["keyword 1", "keyword 2"],\n'
        '  "methods": ["method 1", "method 2"],\n'
        '  "datasets": ["dataset 1"],\n'
        '  "limitations": ["limitation 1"]\n'
        "}\n"
        "REQUIRED: return a JSON array only. Array length must equal the number of input papers."
    )

    USER_PROMPT = (
        "{brief_ctx}"
        "\n\n## Papers to Extract\n"
        "{paper_text}"
    )

    # 批量打包：每批 3 篇，减少 token 数量（避免超时）
    BATCH_SIZE = 3
    all_cards: list[dict] = []

    for i in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[i : i + BATCH_SIZE]
        paper_texts = []
        for idx, cand in enumerate(batch):
            if isinstance(cand, dict):
                title = cand.get("title", "Unknown")
                abstract = cand.get("abstract", "")
                authors = cand.get("authors", [])
                arxiv_id = cand.get("arxiv_id", "")
                url = cand.get("url", "")
            else:
                title = getattr(cand, "title", "Unknown")
                abstract = getattr(cand, "abstract", "")
                authors = getattr(cand, "authors", [])
                arxiv_id = getattr(cand, "arxiv_id", "")
                url = getattr(cand, "url", "")

            fulltext_snippets = _extract_fulltext_snippets(cand)
            fulltext_evidence = _snippets_to_text(
                fulltext_snippets,
                max_items=4,
                max_chars=2800,
            )

            # 正文优先；无正文时用 DeepXiv TLDR，再回退 abstract
            deepxiv_brief = brief_map.get(arxiv_id)
            display_abstract = abstract
            evidence_source = "abstract"
            if fulltext_evidence:
                display_abstract = fulltext_evidence
                evidence_source = "fulltext_chunks"
            elif deepxiv_brief:
                tldr = deepxiv_brief.get("tldr", "") or deepxiv_brief.get("abstract", "")
                if tldr:
                    display_abstract = tldr
                    evidence_source = "deepxiv_tldr"

            paper_texts.append(
                f"=== Paper {i + idx + 1} ===\n"
                f"Title: {title}\n"
                f"Authors: {', '.join(authors) if authors else 'Unknown'}\n"
                f"arXiv ID: {arxiv_id or 'N/A'}\n"
                f"URL: {url or 'N/A'}\n"
                f"Evidence source: {evidence_source}\n"
                f"Evidence text: {display_abstract[:3000]}"
            )

        user_content = USER_PROMPT.format(
            brief_ctx=brief_ctx,
            paper_text="\n\n".join(paper_texts),
        )

        try:
            llm = build_reason_llm(settings, max_tokens=2048, timeout_s=180)
            resp = llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ])
            text = getattr(resp, "content", "") or ""
            import json as _json
            raw = _extract_json(text)
            if raw:
                data = _json.loads(raw)
                if isinstance(data, list):
                    cards = data
                elif isinstance(data, dict) and isinstance(data.get("cards"), list):
                    cards = data.get("cards", [])
                elif isinstance(data, dict) and isinstance(data.get("papers"), list):
                    cards = data.get("papers", [])
                elif isinstance(data, dict):
                    cards = [data]
                else:
                    cards = []

                if len(cards) != len(batch):
                    logger.warning(
                        "[extract_node] parsed %d cards for batch size %d, filling missing cards with fallback",
                        len(cards),
                        len(batch),
                    )

                for batch_idx2, cand_item2 in enumerate(batch):
                    card = cards[batch_idx2] if batch_idx2 < len(cards) and isinstance(cards[batch_idx2], dict) else None
                    if card is None:
                        all_cards.append(_fallback_card(cand_item2, len(all_cards), brief_map))
                        continue

                    card["card_id"] = f"card_{len(all_cards)}"
                    _enrich_card(card, cand_item2)
                    aid = _candidate_arxiv_id(card) or _candidate_arxiv_id(cand_item2)
                    if aid and aid in brief_map:
                        _merge_deepxiv_brief(card, brief_map[aid])
                    all_cards.append(card)
            else:
                # LLM 失败：直接用 DeepXiv brief 或 raw candidate
                for cand_item2 in batch:
                    all_cards.append(_fallback_card(cand_item2, len(all_cards), brief_map))
        except Exception as exc:
            logger.warning("LLM extract failed for batch %d: %s", i // BATCH_SIZE, exc)
            # Fallback: DeepXiv brief 优先
            for cand_item2 in batch:
                all_cards.append(_fallback_card(cand_item2, len(all_cards), brief_map))

    return all_cards


def _candidate_arxiv_id(item: Any) -> str:
    value = _cand_get(item, "arxiv_id", "") or ""
    text = str(value).strip()
    return "" if _is_placeholder_scalar(text) else text


def _fallback_card(cand: Any, idx: int, brief_map: dict[str, dict]) -> dict:
    aid = _candidate_arxiv_id(cand)
    if aid and aid in brief_map:
        card = _card_from_deepxiv_brief(brief_map[aid], f"card_{idx}")
        _enrich_card(card, cand)
        return card
    return _simple_card(cand, idx)


def _extract_fulltext_snippets(cand: Any) -> list[dict[str, str]]:
    """从 candidate 中提取标准化的 fulltext_snippets。"""
    raw = _cand_get(cand, "fulltext_snippets", None)
    if not isinstance(raw, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        section = str(item.get("section") or "unknown").strip() or "unknown"
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        normalized.append({"section": section, "text": text})

    return normalized


def _snippets_to_text(
    snippets: list[dict[str, str]],
    *,
    max_items: int = 4,
    max_chars: int = 3200,
) -> str:
    """将正文 snippets 转为可喂给 LLM 的证据文本。"""
    if not snippets:
        return ""

    parts: list[str] = []
    for item in snippets[:max_items]:
        section = item.get("section", "unknown")
        text = item.get("text", "")
        if not text:
            continue
        parts.append(f"[{section}] {text}")

    return "\n\n".join(parts)[:max_chars]


def _merge_deepxiv_brief(card: dict, brief: dict) -> None:
    """将 DeepXiv brief 信息合并到 card 中（不覆盖已有字段）。"""
    if not brief:
        return
    # keywords：补充到已有的 keywords 列表
    if brief.get("keywords"):
        existing = card.get("keywords", [])
        if isinstance(existing, list) and existing:
            combined = list(dict.fromkeys([*existing, *[str(k) for k in brief["keywords"]]]))
            card["keywords"] = combined[:10]
        else:
            card["keywords"] = [str(k) for k in brief["keywords"][:10]]
    # github_url
    if brief.get("github_url") and not card.get("github_url"):
        card["github_url"] = brief["github_url"]
    # tldr：如果 summary 为空，用 tldr 填充
    tldr = brief.get("tldr", "")
    if tldr and not card.get("summary"):
        card["summary"] = tldr[:300]
        if not card.get("abstract"):
            card["abstract"] = tldr


def _card_from_deepxiv_brief(brief: dict, card_id: str) -> dict:
    """从 DeepXiv brief 构造 PaperCard（DeepXiv 失败时用）。"""
    return {
        "card_id": card_id,
        "title": brief.get("title", "Unknown"),
        "arxiv_id": brief.get("arxiv_id", ""),
        "url": brief.get("url", ""),
        "summary": brief.get("tldr", "") or "",
        "keywords": [str(k) for k in (brief.get("keywords") or [])],
        "github_url": brief.get("github_url") or "",
        "authors": brief.get("authors", []),
        "abstract": brief.get("tldr", "") or "",
        "methods": [],
        "datasets": [],
        "limitations": [],
        "citations": [],
    }


def _extract_json(text: str) -> str | None:
    """
    从 LLM 输出中提取 JSON（支持数组 [ ] 和对象 { }）。
    
    Robust 策略：
    1. 去掉 markdown 代码块
    2. 找第一个 [ 或 { 到最后一个 ] 或 } 
    3. 用 json.loads 验证合法性
    """
    import json
    text = text.strip()
    # 去掉 markdown 代码块
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # 尝试找数组格式 [ ... ]
    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        try:
            json.loads(text[arr_start:arr_end + 1])
            return text[arr_start:arr_end + 1]
        except json.JSONDecodeError:
            pass  # 继续尝试对象格式

    # 尝试找对象格式 { ... }
    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        try:
            json.loads(text[obj_start:obj_end + 1])
            return text[obj_start:obj_end + 1]
        except json.JSONDecodeError:
            pass

    # 最后：直接从开头尝试完整 JSON 解析
    for start_char in ["{", "["]:
        if text.startswith(start_char):
            try:
                json.loads(text)
                return text
            except json.JSONDecodeError:
                pass
    return None


def _build_brief_context(brief: Any | None) -> str:
    if not brief:
        return ""
    topic = (brief.get("topic") if isinstance(brief, dict) else getattr(brief, "topic", "")) or ""
    sub_questions = (brief.get("sub_questions") if isinstance(brief, dict) else getattr(brief, "sub_questions", [])) or []
    desired = (brief.get("desired_output") if isinstance(brief, dict) else getattr(brief, "desired_output", "")) or ""
    ctx = f"## Research Topic\n{topic}\n"
    if sub_questions:
        ctx += "\n## Sub-Questions\n" + "\n".join(f"- {q}" for q in sub_questions) + "\n"
    if desired:
        ctx += f"\n## Desired Output\n{desired}\n"
    return ctx


def _enrich_card(card: dict, original_cand: Any | None = None) -> None:
    """
    从 abstract 中补充 methods / datasets，并回填原始 candidate 的 authors/title。

    若 LLM 丢失了 authors，回填原始 metadata。
    """
    abstract = card.get("summary") or card.get("abstract", "")

    # 规则补充 methods / datasets（仅当未设置时）—— 扩大关键词范围
    if not card.get("methods") and abstract:
        methods = _extract_entities(abstract, [
            # Agent & SWE
            "swe-bench", "code generation", "agent", "multi-agent", "tool use",
            "llm-based agent", "autonomous agent", "software engineering agent",
            "repl", "command execution", "test-driven", "bug fix", "pr review",
            # Core methods
            "transformer", "attention", "reinforcement learning", "graph neural network",
            "diffusion", "language model", "retrieval", "neural network",
            "contrastive", "fine-tuning", "pre-training", "distillation",
            "chain-of-thought", "planning", "reasoning", "self-consistency", "reflection",
            "self-adaptive", "context pruning", "retrieval-augmented generation",
            "rag", "debate", "multi-agent debate", "ensemble", "verifier",
            "Monte Carlo Tree Search", "MCTS", "PBT", "program synthesis",
            "semantic search", "hybrid retrieval", "cross-encoder", "BM25",
            "context window", "long context", "memory mechanism",
            "experience replay", "curriculum learning", "self-supervised",
            "adversarial training", "data augmentation", "few-shot", "zero-shot",
            "coarse-to-fine", "iterative refinement", "beam search",
            "greedy decoding", "nucleus sampling", "temperature sampling",
        ])
        if methods:
            card["methods"] = methods

    if not card.get("datasets") and abstract:
        datasets = _extract_entities(abstract, [
            "SWE-bench", "SWE-bench Verified", "SWE-bench Lite",
            "HumanEval", "MBPP", "APPS", "DS-1000", "MMLU", "BEIR",
            "BigCodeBench", "EvalPlus", "CRUXEval", "NaturalCodeBench",
            "CodeAgentBench", "AgentBench", "BFCL", "API-Bank",
            "ImageNet", "COCO", "SQuAD", "GLUE", "MNLI", "PubMed", "Wiki",
        ])
        if datasets:
            card["datasets"] = datasets

    # 确保 abstract 字段被保留（供 grounding 使用）
    if not card.get("abstract") and abstract:
        card["abstract"] = abstract

    # 回填 authors（LLM 最容易丢失此字段）
    if original_cand:
        orig_title = (
            original_cand.get("title", "") if isinstance(original_cand, dict)
            else getattr(original_cand, "title", "")
        )
        orig_authors = None
        if isinstance(original_cand, dict):
            orig_authors = original_cand.get("authors", [])
        else:
            orig_authors = getattr(original_cand, "authors", [])

        if orig_title:
            card["title"] = orig_title

        card_authors = card.get("authors", [])
        if orig_authors:
            card["authors"] = orig_authors

        # 同时回填 url / arxiv_id
        if (
            original_cand.get("url") if isinstance(original_cand, dict)
            else getattr(original_cand, "url", "")
        ):
            card["url"] = (
                original_cand.get("url") if isinstance(original_cand, dict)
                else getattr(original_cand, "url", "")
            )
        if (
            original_cand.get("arxiv_id") if isinstance(original_cand, dict)
            else getattr(original_cand, "arxiv_id", "")
        ):
            card["arxiv_id"] = (
                original_cand.get("arxiv_id") if isinstance(original_cand, dict)
                else getattr(original_cand, "arxiv_id", "")
            )

        # 回填正文证据（若 search_node 已完成 PDF chunking）
        fulltext_snippets = _extract_fulltext_snippets(original_cand)
        if fulltext_snippets:
            evidence_text = _snippets_to_text(
                fulltext_snippets,
                max_items=5,
                max_chars=4200,
            )
            card["fulltext_available"] = True
            card["fulltext_snippets"] = fulltext_snippets
            if evidence_text:
                if not card.get("content"):
                    card["content"] = evidence_text

                summary_now = str(card.get("summary") or "").strip()
                abstract_now = str(card.get("abstract") or "").strip()
                if not summary_now or len(summary_now) < 120:
                    card["summary"] = evidence_text[:1200]
                if not abstract_now or len(abstract_now) < 240:
                    card["abstract"] = evidence_text[:2000]


def _is_placeholder_scalar(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    return value.strip().lower() in _PLACEHOLDER_STRINGS


def _is_placeholder_list(values: Any) -> bool:
    if not isinstance(values, list) or not values:
        return True
    return all(_is_placeholder_scalar(str(item)) for item in values)


def _extract_entities(text: str, keywords: list[str]) -> list[str]:
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lower]


def _simple_card(cand: Any, idx: int) -> dict:
    """从 raw candidate 构造一个最简 card（LLM 失败时 fallback）。

    关键：必须保留原始 abstract/summary 作为 summary 字段，
    因为它是后续 ground_draft_report 和 verify_claims 的主要 evidence 来源。
    """
    if isinstance(cand, dict):
        title = cand.get("title", "Untitled")
        authors = cand.get("authors", [])
        abstract = cand.get("abstract", "")
        arxiv_id = cand.get("arxiv_id")
        url = cand.get("url", "")
    else:
        title = getattr(cand, "title", "Untitled")
        authors = getattr(cand, "authors", [])
        abstract = getattr(cand, "abstract", "")
        arxiv_id = getattr(cand, "arxiv_id", None)
        url = getattr(cand, "url", "")

    fulltext_snippets = _extract_fulltext_snippets(cand)
    fulltext_text = _snippets_to_text(
        fulltext_snippets,
        max_items=5,
        max_chars=4200,
    )
    primary_text = fulltext_text or abstract or ""

    # 提取 summary（保留较长摘要供 drafting 使用，1500 字符足够 LLM 推断内容）
    summary_text = primary_text[:1500] + ("..." if len(primary_text) > 1500 else "")

    # 规则化提取 methods / datasets（扩大关键词范围）
    methods = _extract_entities(primary_text, [
        # Agent & SWE
        "swe-bench", "code generation", "agent", "multi-agent", "tool use",
        "llm-based agent", "autonomous agent", "software engineering agent",
        "repl", "command execution", "test-driven", "bug fix", "pr review",
        # Core methods
        "transformer", "attention", "reinforcement learning", "graph neural network",
        "diffusion", "language model", "retrieval", "neural network",
        "contrastive", "fine-tuning", "pre-training", "distillation",
        "chain-of-thought", "planning", "reasoning", "self-consistency", "reflection",
        "self-adaptive", "context pruning", "retrieval-augmented generation",
        "rag", "debate", "multi-agent debate", "ensemble", "verifier",
        "Monte Carlo Tree Search", "MCTS", "PBT", "program synthesis",
        "semantic search", "hybrid retrieval", "cross-encoder", "BM25",
        "context window", "long context", "memory mechanism",
        "experience replay", "curriculum learning", "self-supervised",
        "adversarial training", "data augmentation", "few-shot", "zero-shot",
    ])
    datasets = _extract_entities(primary_text, [
        "SWE-bench", "SWE-bench Verified", "SWE-bench Lite",
        "HumanEval", "MBPP", "APPS", "DS-1000", "MMLU", "BEIR",
        "BigCodeBench", "EvalPlus", "CRUXEval", "NaturalCodeBench",
        "CodeAgentBench", "AgentBench", "BFCL", "API-Bank",
        "ImageNet", "COCO", "SQuAD", "GLUE", "MNLI", "PubMed", "Wiki",
    ])

    return {
        "card_id": f"card_{idx}",
        "title": title,
        "authors": authors if isinstance(authors, list) else [],
        "abstract": primary_text,  # 正文优先；无正文时回退 abstract
        "arxiv_id": arxiv_id,
        "url": url,
        "published_year": None,
        "venue": None,
        "summary": summary_text,
        "methods": methods,
        "datasets": datasets,
        "limitations": [],
        "citations": [],
        "content": primary_text,
        "fulltext_available": bool(fulltext_snippets),
        "fulltext_snippets": fulltext_snippets,
    }
