"""Search node — Phase 2: 执行 SearchPlan 查询，产出 RagResult。

策略（3 路并行）：
1. SearXNG（广度召回）：关键词匹配，快速广覆盖
2. arXiv API（精度）：直接 API 查询，metadata 完整
3. DeepXiv（补充 + 热度）：TLDR 摘要、社交热度发现

三者并行执行，统一去重后输出 paper_candidates。
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

from src.models.paper import RagResult
from src.tasking.trace_wrapper import get_trace_store, trace_node

logger = logging.getLogger(__name__)

STRICT_CORE_GROUPS = {"agent", "medical", "multimodal_or_imaging", "diagnosis_or_triage"}
STRICT_CORE_FATAL_PENALTIES = {
    "outside_requested_time_range",
    "governance_without_clinical_scope",
    "off_topic_core_intent",
    "component_method_without_agent_scope",
    "component_or_overview_without_agentic_scope",
    "overview_paper_without_agentic_scope",
    "missing_agent_for_strict_scope",
    "missing_diagnosis_or_triage_for_strict_scope",
    "missing_agentic_workflow_signal",
}


def _run_searxng_queries(
    all_queries: list[tuple[str, str, int]],
) -> tuple[list[dict], list[dict]]:
    """
    并行执行所有 SearXNG 查询，返回 (search_results, query_traces)。
    """
    from src.tools.search_tools import _searxng_search

    MAX_WORKERS = min(8, len(all_queries))
    search_results: list[dict] = []
    query_traces: list[dict] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_searxng_search, q, engines="arxiv", max_results=h): q
            for q, _, h in all_queries
        }
        for future in as_completed(futures):
            q = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                query_traces.append({"query": q, "status": "error", "error": str(exc)})
                continue

            if not result.get("ok"):
                query_traces.append({"query": q, "status": "error", "error": result.get("error")})
                continue

            hits = result.get("hits", [])
            query_traces.append({
                "query": q,
                "status": "success",
                "hits_count": len(hits),
            })
            for hit in hits:
                hit["_search_query"] = q
            search_results.extend(hits)

    return search_results, query_traces


def _run_arxiv_direct_search(
    all_queries: list[tuple[str, str, int]],
    year_filter: str | None = None,
) -> list[dict]:
    """
    直接通过 arXiv API 执行每个查询，返回 paper metadata 列表。

    每个查询最多取 10 条结果。并行执行所有查询。
    """
    from src.tools.arxiv_api import search_arxiv_direct

    results: list[dict] = []

    def _search_one(q: str) -> list[dict]:
        try:
            papers = search_arxiv_direct(q, max_results=10, year_filter=year_filter)
            for paper in papers:
                paper["_source"] = "arxiv_direct"
                paper["_search_query"] = q
            return papers
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=min(8, len(all_queries))) as pool:
        futures = {pool.submit(_search_one, q): q for q, _, _ in all_queries}
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception:
                pass

    return results


# ─── node ─────────────────────────────────────────────────────────────────────


@trace_node(node_name="search", stage="search", store=get_trace_store())
def search_node(state: dict) -> dict:
    """
    Phase 2 搜索节点。

    输入：state.search_plan, state.brief
    输出：state.rag_result（含 paper_candidates）

    并行策略：
    - SearXNG 查询（广度）：并行执行所有查询
    - arXiv API 查询（精度）：并行执行，直接获取 metadata
    - 两者并行，合并去重，确保不漏不重
    """
    from src.tools.arxiv_api import enrich_search_results_with_arxiv

    search_plan = state.get("search_plan")
    brief = state.get("brief")

    if not search_plan:
        logger.warning("[search_node] no search_plan, skipping")
        return {"rag_result": None}

    query_groups = search_plan.get("query_groups", [])
    plan_goal = search_plan.get("plan_goal", "")
    time_range = search_plan.get("time_range") if search_plan else None

    # ── Step 1：收集所有查询 ──────────────────────────────────────────
    all_queries: list[tuple[str, str, int]] = []
    for group in query_groups:
        gid = group.get("group_id", "unknown")
        hits = group.get("expected_hits", 10)
        for q in group.get("queries", []):
            if q:
                all_queries.append((q, gid, hits))

    if not all_queries:
        logger.warning("[search_node] no queries in search_plan")
        return {"rag_result": None}

    # ── Step 2：SearXNG + arXiv API 并行执行 ──────────────────────────
    # 当 time_range 为空时，强制默认 2020 年过滤（避免搜到 2016 年旧论文）
    parsed_year_filter = _time_filter_from_range(time_range)
    effective_year_filter = parsed_year_filter
    if not effective_year_filter:
        from src.tools.arxiv_api import DEFAULT_YEAR_FILTER
        effective_year_filter = DEFAULT_YEAR_FILTER
        logger.info("[search_node] no time_range, applying default year filter: %s", effective_year_filter)

    with ThreadPoolExecutor(max_workers=3) as pool:
        searxng_future = pool.submit(_run_searxng_queries, all_queries)
        arxiv_future = pool.submit(_run_arxiv_direct_search, all_queries, effective_year_filter)
        deepxiv_future = pool.submit(_run_deepxiv_queries, all_queries, effective_year_filter)

    searxng_results, query_traces = searxng_future.result()
    arxiv_direct_results = arxiv_future.result()
    deepxiv_results = deepxiv_future.result()

    logger.info(
        "[search_node] searxng hits=%d, arxiv_direct hits=%d, deepxiv hits=%d",
        len(searxng_results), len(arxiv_direct_results), len(deepxiv_results),
    )

    # ── Step 3：合并候选并去重 ────────────────────────────────────────
    # 优先级：arXiv API > DeepXiv > SearXNG（metadata 完整性依次递减）
    combined: list[dict] = []
    seen_urls: set[str] = set()
    seen_arxiv_ids: set[str] = set()

    for paper in arxiv_direct_results:
        aid = paper.get("arxiv_id") or ""
        url = paper.get("url", "")
        if aid and aid in seen_arxiv_ids:
            continue
        if url and url in seen_urls:
            continue
        if aid:
            seen_arxiv_ids.add(aid)
        if url:
            seen_urls.add(url)
        combined.append(paper)

    for paper in deepxiv_results:
        aid = paper.get("arxiv_id") or ""
        url = paper.get("url", "")
        if aid and aid in seen_arxiv_ids:
            continue
        if url and url in seen_urls:
            continue
        if aid:
            seen_arxiv_ids.add(aid)
        if url:
            seen_urls.add(url)
        combined.append(paper)

    for hit in searxng_results:
        url = hit.get("url", "")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        combined.append(hit)

    # ── Step 4：用 arXiv API 批量补充 metadata ───────────────────────
    enriched = enrich_search_results_with_arxiv(combined)

    # ── Step 5：最终去重 ──────────────────────────────────────────────
    final_candidates: list[dict] = []
    final_seen: set[str] = set()
    for cand in enriched:
        aid = cand.get("arxiv_id") or ""
        url = cand.get("url", "")
        key = aid or url or cand.get("title", "")
        if key and key not in final_seen:
            final_seen.add(key)
            final_candidates.append(cand)

    # ── Step 5b：主题相关性强约束二次筛选 ────────────────────────────
    final_candidates, rerank_log = _rerank_and_filter_candidates(
        final_candidates,
        brief=brief or {},
        search_plan=search_plan or {},
    )

    # ── Step 6：将候选论文写入本地语料库（优先正文，abstract 兜底）──────────────
    ingest_stats = _ingest_paper_candidates(
        final_candidates,
        workspace_id=state.get("workspace_id"),
    )

    # ── Step 7：构建 RagResult ────────────────────────────────────────
    total = len(searxng_results) + len(arxiv_direct_results) + len(deepxiv_results)
    unique = len(final_candidates)

    fulltext_attempted = int(ingest_stats.get("fulltext_attempted", 0))
    fulltext_success = int(ingest_stats.get("fulltext_success", 0))
    fulltext_ratio = (
        (fulltext_success / fulltext_attempted) if fulltext_attempted > 0 else 0.0
    )

    rag_result = RagResult(
        query=plan_goal,
        sub_questions=(brief or {}).get("sub_questions", []) if brief else [],
        rag_strategy="searxng_broad + arxiv_direct_precision + deepxiv_trending + arxiv_api_enrich",
        paper_candidates=final_candidates,
        evidence_chunks=[],
        retrieval_trace=query_traces,
        dedup_log=[{"strategy": "arxiv_id+url dedup", "total": total, "unique": unique}],
        rerank_log=rerank_log,
        coverage_notes=[
            f"执行 {len(all_queries)} 个查询，"
            f"SearXNG {len(searxng_results)} 条 + arXiv API {len(arxiv_direct_results)} 条 + DeepXiv {len(deepxiv_results)} 条，"
            f"去重后 {unique} 篇",
            (
                f"二次筛选：保留 {len(final_candidates)} 篇高相关候选；"
                f"主题锚点={', '.join(_extract_anchor_groups(brief or {}, search_plan or {}).get('active_groups', [])) or 'none'}"
            ),
            f"正文下载尝试 {fulltext_attempted} 篇，成功 {fulltext_success} 篇（{fulltext_ratio:.0%}），"
            f"coarse chunks={int(ingest_stats.get('coarse_chunks', 0))}，"
            f"fine chunks={int(ingest_stats.get('fine_chunks', 0))}",
        ],
        total_papers=unique,
        total_chunks=0,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        "[search_node] done: %d queries → %d hits → %d unique papers",
        len(all_queries), total, unique,
    )
    return {"rag_result": rag_result}


def _ingest_paper_candidates(
    candidates: list[dict],
    *,
    workspace_id: str | None = None,
    fulltext_top_n: int = 16,
    abstract_top_n: int = 50,
) -> dict[str, Any]:
    """将候选论文写入本地语料库：优先正文 chunk，abstract 兜底。

    - 正文路径：arXiv PDF -> ingest -> chunking -> index_document。
    - 兜底路径：title + abstract 写入 coarse chunk。
    - 同时将正文证据回填到 candidate，供 extract/draft 优先使用。
    """
    import hashlib

    stats: dict[str, Any] = {
        "total_candidates": len(candidates),
        "fulltext_attempted": 0,
        "fulltext_success": 0,
        "fulltext_failed": 0,
        "coarse_chunks": 0,
        "fine_chunks": 0,
        "abstract_indexed": 0,
        "errors": [],
    }

    if not candidates:
        return stats

    for cand in candidates:
        cand.setdefault("fulltext_available", False)

    # ── Stage A：正文下载与 chunking（优先）──────────────────────────────
    fulltext_targets: list[tuple[str, dict]] = []
    seen_ids: set[str] = set()
    for cand in candidates:
        aid = _normalize_arxiv_id(str(cand.get("arxiv_id") or ""))
        if not aid or aid in seen_ids:
            continue
        seen_ids.add(aid)
        fulltext_targets.append((aid, cand))
        if len(fulltext_targets) >= fulltext_top_n:
            break

    stats["fulltext_attempted"] = len(fulltext_targets)

    if fulltext_targets:
        session = None
        try:
            from src.corpus.ingest import ArxivSourceInput, ingest, chunk_document
            from src.corpus.store import CorpusRepository
            from src.db.engine import get_session_factory

            session_factory = get_session_factory()
            session = session_factory()
            repo = CorpusRepository(db_session=session)
            repo.connect()

            target_by_id = {aid: cand for aid, cand in fulltext_targets}
            sources = [ArxivSourceInput(arxiv_id=aid) for aid, _ in fulltext_targets]
            ingest_result = ingest(sources)

            for err in ingest_result.errors:
                msg = err.get("error") if isinstance(err, dict) else str(err)
                if msg:
                    stats["errors"].append(str(msg))

            for doc in ingest_result.successful:
                resolved_aid = _normalize_arxiv_id(
                    str(
                        doc.arxiv_id
                        or (doc.source_ref.external_id if doc.source_ref else "")
                        or ""
                    )
                )
                cand = target_by_id.get(resolved_aid)
                if cand is None:
                    continue

                try:
                    chunking = chunk_document(doc)
                    snippets = _pick_fulltext_snippets(chunking.coarse_chunks)
                    _annotate_candidate_with_fulltext(
                        cand,
                        snippets=snippets,
                        coarse_count=len(chunking.coarse_chunks),
                        fine_count=len(chunking.fine_chunks),
                    )

                    repo.index_document(
                        doc,
                        coarse_chunks=chunking.coarse_chunks,
                        fine_chunks=chunking.fine_chunks,
                        embeddings=None,
                    )

                    stats["coarse_chunks"] += len(chunking.coarse_chunks)
                    stats["fine_chunks"] += len(chunking.fine_chunks)
                    if snippets:
                        stats["fulltext_success"] += 1
                    else:
                        stats["fulltext_failed"] += 1

                    if chunking.errors:
                        stats["errors"].extend(
                            [f"{resolved_aid}: {msg}" for msg in chunking.errors[:2] if msg]
                        )
                except Exception as exc:
                    stats["fulltext_failed"] += 1
                    stats["errors"].append(f"{resolved_aid}: {exc}")

            accounted = int(stats["fulltext_success"]) + int(stats["fulltext_failed"])
            missing = int(stats["fulltext_attempted"]) - accounted
            if missing > 0:
                stats["fulltext_failed"] += missing

            session.commit()
        except Exception as exc:
            logger.warning("[search_node] fulltext ingest failed, fallback to abstract-only: %s", exc)
            stats["errors"].append(f"fulltext_ingest: {exc}")
            if session is not None:
                session.rollback()
        finally:
            if session is not None:
                session.close()

    # ── Stage B：abstract 兜底写入（补齐未正文化候选）────────────────────
    to_ingest = candidates[:abstract_top_n]

    try:
        from src.db.engine import get_session_factory
        from src.db.models import Document, CoarseChunk as ORMCoarseChunk

        session_factory = get_session_factory()
        session = session_factory()
        try:
            for cand in to_ingest:
                if cand.get("fulltext_available"):
                    continue

                title = str(cand.get("title") or "")
                abstract = str(cand.get("abstract") or cand.get("summary") or "")
                if not title and not abstract:
                    continue

                arxiv_id = _normalize_arxiv_id(str(cand.get("arxiv_id") or ""))
                url = str(cand.get("url") or "")
                if arxiv_id:
                    doc_id = f"arxiv:{arxiv_id}"
                elif url:
                    doc_id = hashlib.md5(url.encode()).hexdigest()[:24]
                else:
                    doc_id = hashlib.md5(title[:200].encode()).hexdigest()[:24]

                chunk_text = (title + "\n\n" + abstract).strip()
                chunk_id = hashlib.sha256(
                    f"{doc_id}:abstract:0".encode()
                ).hexdigest()[:24]

                # upsert Document
                orm_doc = session.query(Document).filter(Document.doc_id == doc_id).first()
                if orm_doc is None:
                    # Build source_uri and source_id
                    source_uri = url or (f"arxiv:{arxiv_id}" if arxiv_id else doc_id)
                    source_id = arxiv_id or doc_id
                    orm_doc = Document(
                        doc_id=doc_id,
                        source_id=source_id,
                        source_uri=source_uri,
                        source_type="unknown",
                        title=title[:500] if title else "Unknown",
                    )
                    session.merge(orm_doc)

                # upsert CoarseChunk
                orm_chunk = ORMCoarseChunk(
                    coarse_chunk_id=chunk_id,
                    doc_id=doc_id,
                    canonical_id=doc_id,
                    section="abstract",
                    section_level=1,
                    page_start=1,
                    page_end=1,
                    char_start=0,
                    char_end=len(chunk_text),
                    text=chunk_text,
                    text_hash=hashlib.md5(chunk_text.encode()).hexdigest(),
                    token_count=int(len(chunk_text) * 0.25),
                    order_idx=0,
                    meta_info={
                        "workspace_id": workspace_id,
                        "source": "search_node_ingest",
                        "arxiv_id": arxiv_id,
                        "year": str(cand.get("year") or ""),
                        "authors": str(cand.get("authors") or ""),
                    },
                )
                session.merge(orm_chunk)
                stats["abstract_indexed"] += 1

            session.commit()
        except Exception as e:
            session.rollback()
            logger.warning("[search_node] abstract ingest failed: %s", e)
            stats["errors"].append(f"abstract_ingest: {e}")
        finally:
            session.close()
    except Exception as e:
        logger.warning("[search_node] ingest skipped (DB not available): %s", e)
        stats["errors"].append(f"db_unavailable: {e}")

    logger.info(
        "[search_node] ingest stats: fulltext=%d/%d, coarse=%d, fine=%d, abstract_fallback=%d",
        int(stats.get("fulltext_success", 0)),
        int(stats.get("fulltext_attempted", 0)),
        int(stats.get("coarse_chunks", 0)),
        int(stats.get("fine_chunks", 0)),
        int(stats.get("abstract_indexed", 0)),
    )
    return stats


def _normalize_arxiv_id(raw: str) -> str:
    """标准化 arXiv ID（去 URL、去 .pdf、去版本号）。"""
    if not raw:
        return ""
    aid = raw.strip()
    aid = re.sub(r"^arXiv:\s*", "", aid, flags=re.IGNORECASE)
    aid = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", aid, flags=re.IGNORECASE)
    aid = re.sub(r"\.pdf$", "", aid, flags=re.IGNORECASE)
    aid = re.sub(r"v\d+$", "", aid)
    return aid


def _pick_fulltext_snippets(
    coarse_chunks: list[Any],
    *,
    max_snippets: int = 6,
    max_chars: int = 1400,
) -> list[dict[str, str]]:
    """从 coarse chunks 中提取优先用于方法/实验写作的正文片段。"""
    priority_sections = (
        "method",
        "approach",
        "architecture",
        "experiment",
        "evaluation",
        "result",
        "discussion",
        "analysis",
        "dataset",
        "implementation",
        "ablation",
    )

    primary: list[dict[str, str]] = []
    secondary: list[dict[str, str]] = []

    for chunk in coarse_chunks:
        text = str(getattr(chunk, "text", "") or "").strip()
        if len(text) < 120:
            continue
        section = str(getattr(chunk, "section", "unknown") or "unknown")
        snippet = {
            "section": section,
            "text": text[:max_chars],
        }
        if any(token in section.lower() for token in priority_sections):
            primary.append(snippet)
        else:
            secondary.append(snippet)

    merged = primary + secondary
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in merged:
        key = (item["section"].lower() + "|" + item["text"][:180]).strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= max_snippets:
            break

    return deduped


def _annotate_candidate_with_fulltext(
    candidate: dict[str, Any],
    *,
    snippets: list[dict[str, str]],
    coarse_count: int,
    fine_count: int,
) -> None:
    """将正文解析结果写回 candidate，供 extract/draft 优先使用。"""
    candidate["fulltext_chunk_count"] = int(coarse_count)
    candidate["fulltext_fine_chunk_count"] = int(fine_count)
    candidate["fulltext_available"] = bool(snippets)

    if not snippets:
        return

    candidate["fulltext_snippets"] = snippets
    candidate["fulltext_source"] = "arxiv_pdf"
    joined = "\n\n".join(
        f"[{item.get('section', 'unknown')}] {item.get('text', '')}"
        for item in snippets
    ).strip()
    candidate["fulltext_excerpt"] = joined[:8000]
    # 兼容现有下游逻辑：content 会被 extract/draft 直接消费。
    candidate["content"] = joined[:8000]


def _run_deepxiv_queries(
    all_queries: list[tuple[str, str, int]],
    year_filter: str | None = None,
) -> list[dict]:
    """
    通过 DeepXiv 执行关键词搜索，发现 SearXNG/arXiv API 可能遗漏的相关论文。

    策略（参考 DeepXiv 设计）：
    1. 取前 3 个核心查询词在 DeepXiv 搜索（每词最多 10 篇）
    2. 同时追加 trending 热门论文（7 天内最多 15 篇）
    3. DeepXiv 结果与现有结果去重后返回

    注意：DeepXiv 提供 TLDR + keywords，比 raw abstract 信息更丰富，
    可在 extract_node 中直接用 brief 信息。
    """
    from src.tools.deepxiv_client import search_papers, get_trending_papers

    results: list[dict] = []
    seen_ids: set[str] = set()

    # DeepXiv 有每日请求限额，最多搜 3 个查询词
    core_queries = [q for q, _, _ in all_queries[:3]]

    for q in core_queries:
        try:
            papers = search_papers(q, size=10)
            for paper in papers:
                aid = paper.get("arxiv_id") or ""
                if aid and aid not in seen_ids:
                    seen_ids.add(aid)
                    results.append(paper)
        except Exception as e:
            logger.warning("[search_node] DeepXiv search(%r) failed: %s", q, e)

    # 追加 trending 热门论文（按热度发现，不依赖关键词匹配）
    try:
        days_back = 90
        if year_filter:
            try:
                yf = int(year_filter)
                from datetime import datetime, timezone
                y_now = datetime.now(timezone.utc).year
                days_back = min((y_now - yf) * 365, 365)
            except Exception:
                pass

        trending = get_trending_papers(days=int(days_back // 7), size=15)
        for paper in trending:
            aid = paper.get("arxiv_id") or ""
            if aid and aid not in seen_ids:
                seen_ids.add(aid)
                results.append(paper)
    except Exception as e:
        logger.warning("[search_node] DeepXiv trending failed: %s", e)

    logger.info("[search_node] DeepXiv: %d papers from queries + trending", len(results))
    return results


def _time_filter_from_range(time_range: str | None) -> str | None:
    """将 time_range 字符串转换为 arXiv API 的年份格式。"""
    if not time_range:
        return None
    import re
    now = datetime.now(timezone.utc).year
    if "近2年" in time_range or "2年" in time_range:
        return str(now - 2)
    if "近1年" in time_range or "1年" in time_range:
        return str(now - 1)
    if "近3年" in time_range or "3年" in time_range:
        return str(now - 3)
    m = re.search(r"20\d{2}", time_range)
    if m:
        return m.group(0)
    return None


def _rerank_and_filter_candidates(
    candidates: list[dict[str, Any]],
    *,
    brief: dict[str, Any],
    search_plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply a domain-aware second pass before extract.

    The goal is not perfect relevance modelling. The goal is to avoid obviously
    off-topic papers from flowing into extract/draft when the brief has strong
    anchor terms such as domain + task intent.
    """
    if not candidates:
        return [], []

    anchor_context = _extract_anchor_groups(brief, search_plan)
    year_bounds = _extract_year_bounds(brief, search_plan)
    rescored: list[dict[str, Any]] = []
    rerank_log: list[dict[str, Any]] = []

    for cand in candidates:
        score, diagnostics = _score_candidate_relevance(cand, anchor_context)
        enriched = dict(cand)
        enriched["combined_score"] = round(score, 4)
        enriched["relevance_diagnostics"] = diagnostics
        rescored.append(enriched)

    rescored.sort(
        key=lambda cand: (
            float(cand.get("combined_score") or 0.0),
            1 if cand.get("fulltext_available") else 0,
            len(str(cand.get("abstract") or cand.get("content") or "")),
        ),
        reverse=True,
    )

    active_groups = anchor_context.get("active_groups", [])
    strict_core = bool(anchor_context.get("strict_core"))
    has_in_range_candidate = any(
        _year_in_range(_candidate_year(cand), year_bounds)
        for cand in rescored
    )
    min_keep = min(len(rescored), 3 if strict_core else 12)
    hard_floor = min(len(rescored), 2 if strict_core else 3)
    threshold = 2.2 if strict_core else 1.2 if active_groups else 0.0

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for cand in rescored:
        score = float(cand.get("combined_score") or 0.0)
        cand_year = _candidate_year(cand)
        in_year_range = _year_in_range(cand_year, year_bounds)
        penalties = list(cand.get("relevance_diagnostics", {}).get("penalties", []))
        if not in_year_range and cand_year is not None:
            penalties.append("outside_requested_time_range")
            cand.setdefault("relevance_diagnostics", {})["penalties"] = penalties
        if strict_core and has_in_range_candidate and not in_year_range and cand_year is not None:
            dropped.append(cand)
            continue
        if strict_core and "governance_without_clinical_scope" in penalties:
            dropped.append(cand)
            continue
        if active_groups and score < 0.0 and len(kept) >= hard_floor:
            dropped.append(cand)
        elif strict_core and "off_topic_core_intent" in penalties and len(kept) >= hard_floor:
            dropped.append(cand)
        elif strict_core and "component_or_overview_without_agentic_scope" in penalties and kept:
            dropped.append(cand)
        elif strict_core and "missing_agentic_workflow_signal" in penalties and kept:
            dropped.append(cand)
        elif strict_core and "missing_agent_for_strict_scope" in penalties and kept:
            dropped.append(cand)
        elif not in_year_range and cand_year is not None and len(kept) >= hard_floor:
            dropped.append(cand)
        elif len(kept) < min_keep or score >= threshold:
            kept.append(cand)
        else:
            dropped.append(cand)

    if strict_core:
        kept, dropped = _supplement_strict_core_survey_candidates(
            kept,
            dropped,
            rescored=rescored,
        )

    if active_groups and len(kept) < hard_floor:
        if strict_core:
            safe_floor = [
                cand
                for cand in rescored
                if not set(cand.get("relevance_diagnostics", {}).get("penalties", [])).intersection(
                    {
                        "outside_requested_time_range",
                        "off_topic_core_intent",
                        "governance_without_clinical_scope",
                        "missing_agentic_workflow_signal",
                        "component_method_without_agent_scope",
                        "component_or_overview_without_agentic_scope",
                        "overview_paper_without_agentic_scope",
                    }
                )
            ]
        else:
            safe_floor = rescored

        selected = (safe_floor or rescored)[:hard_floor]
        selected_titles = {str(item.get("title") or "") for item in selected}
        kept = selected
        dropped = [
            cand
            for cand in rescored
            if str(cand.get("title") or "") not in selected_titles
        ]

    strict_core_kept = sum(1 for cand in kept if _is_strict_core_candidate(cand))
    adjacent_support_kept = max(0, len(kept) - strict_core_kept)

    rerank_log.append(
        {
            "strategy": "domain_aware_anchor_rerank",
            "active_groups": active_groups,
            "strict_core": strict_core,
            "kept": len(kept),
            "dropped": len(dropped),
            "strict_core_kept": strict_core_kept,
            "adjacent_support_kept": adjacent_support_kept,
            "threshold": threshold,
            "year_bounds": year_bounds,
        }
    )
    rerank_log.extend(
        {
            "paper": cand.get("title", ""),
            "score": cand.get("combined_score"),
            "decision": "kept",
            "matched_groups": cand.get("relevance_diagnostics", {}).get("matched_groups", []),
            "penalties": cand.get("relevance_diagnostics", {}).get("penalties", []),
        }
        for cand in kept[:8]
    )
    rerank_log.extend(
        {
            "paper": cand.get("title", ""),
            "score": cand.get("combined_score"),
            "decision": "dropped",
            "matched_groups": cand.get("relevance_diagnostics", {}).get("matched_groups", []),
            "penalties": cand.get("relevance_diagnostics", {}).get("penalties", []),
        }
        for cand in dropped[:8]
    )
    return kept, rerank_log


def _supplement_strict_core_survey_candidates(
    kept: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
    *,
    rescored: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recover a small amount of adjacent support without letting it dominate.

    Under strict-core topics, the candidate pool should stay majority-core.
    Adjacent clinical support papers are allowed only in small quotas, mainly
    for benchmarks or report-generation context, and only after more obviously
    off-scope/component papers have already been filtered out.
    """
    strict_core_selected: list[dict[str, Any]] = []
    strict_core_titles: set[str] = set()
    for cand in rescored:
        if _is_strict_core_candidate(cand):
            title = str(cand.get("title") or "")
            if title and title not in strict_core_titles:
                strict_core_selected.append(cand)
                strict_core_titles.add(title)

    strict_core_count = len(strict_core_selected)
    if strict_core_count == 0:
        adjacent_quota = 2
    elif strict_core_count == 1:
        adjacent_quota = 1
    elif strict_core_count <= 3:
        adjacent_quota = 2
    else:
        adjacent_quota = 3

    selected = list(strict_core_selected[:8])
    selected_titles = {str(item.get("title") or "") for item in selected}

    for cand in rescored:
        title = str(cand.get("title") or "")
        if not title or title in selected_titles:
            continue
        if len(selected) - min(len(selected), strict_core_count) >= adjacent_quota:
            break
        if _is_strict_core_adjacent_support_candidate(cand, strict_core_count=strict_core_count):
            selected.append(cand)
            selected_titles.add(title)

    if not selected:
        fallback_pool = [
            cand
            for cand in rescored
            if not set(cand.get("relevance_diagnostics", {}).get("penalties", [])).intersection(
                {
                    "outside_requested_time_range",
                    "governance_without_clinical_scope",
                    "off_topic_core_intent",
                    "component_method_without_agent_scope",
                    "component_or_overview_without_agentic_scope",
                    "overview_paper_without_agentic_scope",
                }
            )
        ]
        for cand in (fallback_pool or kept):
            title = str(cand.get("title") or "")
            if title and title not in selected_titles:
                selected.append(cand)
                selected_titles.add(title)
            if len(selected) >= 2:
                break

    remaining_dropped = [
        cand
        for cand in rescored
        if str(cand.get("title") or "") not in selected_titles
    ]
    return selected, remaining_dropped


def _strict_core_match_count(candidate: dict[str, Any]) -> int:
    matched = set(candidate.get("relevance_diagnostics", {}).get("matched_groups", []))
    return len(matched.intersection(STRICT_CORE_GROUPS))


def _is_strict_core_candidate(candidate: dict[str, Any]) -> bool:
    matched = set(candidate.get("relevance_diagnostics", {}).get("matched_groups", []))
    penalties = set(candidate.get("relevance_diagnostics", {}).get("penalties", []))
    return STRICT_CORE_GROUPS.issubset(matched) and not penalties.intersection(STRICT_CORE_FATAL_PENALTIES)


def _is_strict_core_adjacent_support_candidate(
    candidate: dict[str, Any],
    *,
    strict_core_count: int,
) -> bool:
    matched = set(candidate.get("relevance_diagnostics", {}).get("matched_groups", []))
    penalties = set(candidate.get("relevance_diagnostics", {}).get("penalties", []))
    if penalties.intersection(
        {
            "outside_requested_time_range",
            "governance_without_clinical_scope",
            "off_topic_core_intent",
            "component_method_without_agent_scope",
            "component_or_overview_without_agentic_scope",
            "overview_paper_without_agentic_scope",
            "missing_agentic_workflow_signal",
        }
    ):
        return False

    title = str(candidate.get("title") or "").lower()
    abstract = str(candidate.get("abstract") or candidate.get("content") or "").lower()
    score = float(candidate.get("combined_score") or 0.0)
    if score < 2.0:
        return False

    support_tokens = (
        "benchmark",
        "triage",
        "report generation",
        "report composition",
        "question answering",
        "retrieval",
        "workflow",
        "assistant",
        "decision support",
    )
    has_support_signal = any(
        _token_occurs(title, token) or _token_occurs(abstract, token)
        for token in support_tokens
    )
    match_count = _strict_core_match_count(candidate)
    if match_count < 3 and not (
        {"medical", "multimodal_or_imaging"}.issubset(matched)
        and has_support_signal
    ):
        return False

    if "missing_agent_for_strict_scope" in penalties:
        return strict_core_count >= 1 and (
            (has_support_signal and score >= 2.2)
            or (match_count >= 3 and score >= 2.6)
        )

    return has_support_signal or "diagnosis_or_triage" in matched


def _extract_anchor_groups(brief: dict[str, Any], search_plan: dict[str, Any]) -> dict[str, Any]:
    topic = " ".join(
        str(part or "").strip()
        for part in (
            brief.get("topic"),
            brief.get("domain_scope"),
            " ".join(brief.get("sub_questions", [])[:3]) if isinstance(brief.get("sub_questions"), list) else "",
            search_plan.get("plan_goal"),
        )
        if str(part or "").strip()
    ).lower()

    groups: dict[str, tuple[str, ...]] = {}

    if any(token in topic for token in ("agent", "智能体", "代理", "autonomous", "tool use", "llm-based agent")):
        groups["agent"] = (
            "agent",
            "agents",
            "multi-agent",
            "autonomous",
            "tool use",
            "llm agent",
            "software engineering agent",
            "智能体",
            "代理",
        )

    if any(token in topic for token in ("medical", "clinical", "health", "medicine", "医疗", "医学", "诊断", "hospital")):
        groups["medical"] = (
            "medical",
            "clinical",
            "health",
            "healthcare",
            "medicine",
            "hospital",
            "radiology",
            "pathology",
            "histopathology",
            "clinician",
            "ehr",
            "electronic health record",
            "emergency department",
            "diagnosis",
            "diagnostic",
            "patient",
            "biomedical",
            "医疗",
            "医学",
            "诊断",
            "病人",
            "临床",
        )

    if any(token in topic for token in ("image", "imaging", "vision", "multimodal", "影像", "多模态", "radiology")):
        groups["multimodal_or_imaging"] = (
            "image",
            "imaging",
            "vision",
            "multimodal",
            "radiology",
            "pathology",
            "histopathology",
            "ultrasound",
            "mri",
            "ct",
            "x-ray",
            "影像",
            "多模态",
            "视觉",
        )

    if any(token in topic for token in ("rag", "retrieval", "检索")):
        groups["retrieval"] = (
            "rag",
            "retrieval",
            "retriever",
            "retrieval-augmented",
            "dense retrieval",
            "sparse retrieval",
            "检索",
        )

    if any(token in topic for token in ("diagnosis", "diagnostic", "triage", "clinical decision", "分诊", "诊断")):
        groups["diagnosis_or_triage"] = (
            "diagnosis",
            "diagnostic",
            "triage",
            "clinical decision",
            "report generation",
            "report composition",
            "reporting",
            "question answering",
            "decision support",
            "诊断",
            "分诊",
        )

    return {
        "topic": topic,
        "groups": groups,
        "active_groups": list(groups.keys()),
        "strict_core": all(
            key in groups
            for key in ("agent", "medical", "multimodal_or_imaging", "diagnosis_or_triage")
        ),
    }


def _score_candidate_relevance(
    candidate: dict[str, Any],
    anchor_context: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    title = str(candidate.get("title") or "")
    abstract = str(candidate.get("abstract") or candidate.get("content") or "")
    text = f"{title}\n{abstract}".lower()
    title_lower = title.lower()

    score = 0.0
    matched_groups: list[str] = []
    penalties: list[str] = []

    for group_name, tokens in anchor_context.get("groups", {}).items():
        title_hits = sum(1 for token in tokens if token and _token_occurs(title_lower, token))
        body_hits = sum(1 for token in tokens if token and _token_occurs(text, token))
        if title_hits or body_hits:
            matched_groups.append(group_name)
            score += min(2.5, title_hits * 1.4 + body_hits * 0.35)
        else:
            score -= 1.1
            penalties.append(f"missing:{group_name}")

    score += min(0.8, float(candidate.get("score") or 0.0) * 0.2)
    if candidate.get("fulltext_available"):
        score += 0.4
    if candidate.get("source") == "arxiv_direct" or candidate.get("_source") == "arxiv_direct":
        score += 0.2

    strict_core = bool(anchor_context.get("strict_core"))
    if strict_core and len(set(matched_groups).intersection({"agent", "medical", "multimodal_or_imaging", "diagnosis_or_triage"})) < 3:
        score -= 2.5
        penalties.append("off_topic_core_intent")

    if strict_core and "agent" not in matched_groups:
        score -= 1.2
        penalties.append("missing_agent_for_strict_scope")

    if strict_core and "diagnosis_or_triage" not in matched_groups:
        score -= 1.0
        penalties.append("missing_diagnosis_or_triage_for_strict_scope")

    agentic_signal = _has_agentic_workflow_signal(text)
    if strict_core and agentic_signal:
        score += 0.6
    elif strict_core:
        score -= 1.9
        penalties.append("missing_agentic_workflow_signal")

    if strict_core and any(_token_occurs(text, token) for token in ("benchmark", "evaluation")) and any(
        _token_occurs(text, token) for token in ("triage", "workflow", "assistant", "llm-assisted")
    ):
        score += 1.2

    if strict_core and not agentic_signal:
        component_markers = (
            "segmentation",
            "classification",
            "classifier",
            "detection",
            "detector",
            "augmentation",
            "curriculum learning",
            "survey",
            "introduction",
            "modality",
            "modalities",
        )
        if any(_token_occurs(text, token) for token in component_markers):
            score -= 2.5
            penalties.append("component_or_overview_without_agentic_scope")

    off_topic_markers = (
        ("federated", "federated_learning_without_agent"),
        ("recommender", "recommender_without_anchor"),
        ("machine translation", "mt_without_anchor"),
        ("object detection", "vision_task_without_anchor"),
        ("regulatory", "governance_without_clinical_scope"),
        ("privacy", "governance_without_clinical_scope"),
        ("security", "governance_without_clinical_scope"),
        ("governance", "governance_without_clinical_scope"),
        ("e-commerce", "off_topic_core_intent"),
        ("adobe", "off_topic_core_intent"),
        ("auction", "off_topic_core_intent"),
    )
    for token, label in off_topic_markers:
        if _token_occurs(text, token) and (not matched_groups or strict_core):
            score -= 1.5
            penalties.append(label)

    if strict_core and any(_token_occurs(text, token) for token in ("segmentation", "registration", "clustering", "annotation")):
        if "agent" not in matched_groups and "diagnosis_or_triage" not in matched_groups:
            score -= 1.8
            penalties.append("component_method_without_agent_scope")

    if strict_core and any(
        _token_occurs(title_lower, token)
        for token in ("survey", "review", "tutorial", "introduction", "overview", "modality", "modalities")
    ) and "agent" not in matched_groups:
        score -= 1.0
        penalties.append("overview_paper_without_agentic_scope")

    diagnostics = {
        "matched_groups": matched_groups,
        "penalties": penalties,
    }
    return score, diagnostics


def _has_agentic_workflow_signal(text: str) -> bool:
    markers = (
        "agent",
        "agentic",
        "assistant",
        "question answering",
        "retrieval",
        "rag",
        "workflow",
        "orchestration",
        "planner",
        "coordinator",
        "benchmark",
        "tool",
        "llm-assisted",
        "report generation",
        "report composition",
        "decision support",
    )
    return any(_token_occurs(text, token) for token in markers)


def _token_occurs(text: str, token: str) -> bool:
    token = str(token or "").strip().lower()
    if not token:
        return False
    if any(ord(char) > 127 for char in token):
        return token in text

    escaped = re.escape(token).replace(r"\ ", r"\s+")
    pattern = rf"(?<!\w){escaped}(?!\w)"
    return re.search(pattern, text) is not None


def _extract_year_bounds(brief: dict[str, Any], search_plan: dict[str, Any]) -> tuple[int | None, int | None]:
    text = " ".join(
        part
        for part in (
            str(brief.get("time_range") or "").strip(),
            str(search_plan.get("plan_goal") or "").strip(),
        )
        if part
    )
    years = [int(item) for item in re.findall(r"20\d{2}", text)]
    if not years:
        return (None, None)
    if len(years) == 1:
        return (years[0], None)
    return (min(years), max(years))


def _candidate_year(candidate: dict[str, Any]) -> int | None:
    for key in ("published_year", "year"):
        value = candidate.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.search(r"20\d{2}", value)
            if match:
                return int(match.group(0))
    published_date = str(candidate.get("published_date") or "")
    match = re.search(r"20\d{2}", published_date)
    if match:
        return int(match.group(0))
    return None


def _year_in_range(candidate_year: int | None, bounds: tuple[int | None, int | None]) -> bool:
    start_year, end_year = bounds
    if candidate_year is None:
        return True
    if start_year is not None and candidate_year < start_year:
        return False
    if end_year is not None and candidate_year > end_year:
        return False
    return True
