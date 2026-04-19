"""RetrieverAgent — Tool-Augmented Generation (TAG) 模式。

设计模式说明：
- TAG = Tool-Augmented Generation：在 LLM 推理时实时调用工具增强生成质量。
- 核心思想：LLM 生成每一步时，按需调用外部工具（检索、查表），
  将工具结果无缝注入推理链，而不是事后 post-hoc 拼接检索结果。
- 适用场景：需要实时、精确外部知识的检索任务。

与 ReAct 的区别：
- ReAct：显式循环（Observe → Think → Act → ...），工具调用是独立步骤
- TAG：工具是推理链中的内联增强（LLM 自驱动按需调用），无需显式循环
- TAG 更适合："我需要查一下 X" 这类即时知识需求

阶段划分：
  Phase 1 (Augmented Query Generation):
      LLM 生成查询，同时内联调用 expand_keywords / rewrite_query 增强查询质量
  Phase 2 (Parallel Retrieval):
      批量并行检索，工具结果作为 LLM 推理的 context
  Phase 3 (Context Assembly):
      LLM 基于检索结果内联生成（直接输出最终 candidate list）
"""

from __future__ import annotations

from dataclasses import asdict
import logging
import time
from typing import Any
from typing import TypedDict

from langgraph.graph import START, StateGraph
from src.agent.checkpointing import build_graph_config, get_langgraph_checkpointer
from src.memory.manager import get_memory_manager
from src.models.paper import RagResult

logger = logging.getLogger(__name__)


# ─── TAG Prompt Templates ────────────────────────────────────────────────────


AUGMENTED_QUERY_PROMPT = """You are a query generation expert. Given a research topic and sub-questions, use tools only when they improve query quality.

Working style:
- Generate the enhanced query list directly.
- If keyword expansion or query rewriting is needed, describe the tool usage inside the <tools> block.
- Return only a JSON object containing the query list.

<tools>
Available tools: expand_keywords(topic, dimension), rewrite_query(query, mode)
</tools>

Output (strict JSON):
```json
{{
  "queries": [
    {{
      "query": "enhanced query text",
      "sources": ["arxiv", "semantic_scholar"],
      "tools_used": ["expand_keywords"],
      "expected_hits": 20
    }}
  ]
}}
```
"""


CONTEXT_ASSEMBLY_PROMPT = """You are a retrieval context assembly expert.

Given retrieval results from multiple sources, produce the final high-quality candidate paper list.

Rules:
1. Deduplicate by arXiv ID or URL.
2. Rank by relevance, prioritizing title match over abstract match.
3. Each record must contain title, url, abstract (first 200 characters), and source.
4. Return only a JSON array with no extra explanation.

Output (strict JSON array):
```json
[
  {{"rank": 1, "title": "...", "url": "...", "abstract": "...", "source": "arxiv"}},
  ...
]
```
Limit: at most {max_candidates} items.
"""

MAX_PLANNED_QUERIES = 12


# ─── RetrieverAgent ──────────────────────────────────────────────────────────


class RetrieverAgent:
    """
    Tool-Augmented Generation 模式的 Retriever Agent。

    工作流程：
      ┌────────────────────────┐
      │ AUGMENTED QUERY GEN    │  LLM 生成 + 内联工具增强（expand/rewrite）
      └──────────┬─────────────┘
                 │ augmented queries
                 ▼
      ┌────────────────────────┐
      │ PARALLEL RETRIEVAL     │  并行执行多源检索（SearXNG + local corpus）
      └──────────┬─────────────┘
                 │ raw results
                 ▼
      ┌────────────────────────┐
      │ CONTEXT ASSEMBLY       │  LLM 内联推理：直接基于检索结果生成最终 candidate list
      │ (TAG 核心)             │  不再经过中间状态，工具结果 → LLM 直接输出
      └──────────┬─────────────┘
                 │ final candidates
    """

    def __init__(self, workspace_id: str | None = None, task_id: str | None = None):
        self.workspace_id = workspace_id
        self.task_id = task_id
        self.mm = get_memory_manager(workspace_id) if workspace_id else None

    # ── Phase 1: Augmented Query Generation ─────────────────────────────────

    def _queries_from_search_plan(self, search_plan: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not isinstance(search_plan, dict):
            return []
        groups = search_plan.get("query_groups", [])
        if not isinstance(groups, list):
            return []

        source_preferences = search_plan.get("source_preferences", [])
        if not isinstance(source_preferences, list) or not source_preferences:
            source_preferences = ["arxiv"]

        grouped_queries: list[list[dict[str, Any]]] = []
        for group in groups:
            if not isinstance(group, dict):
                continue
            intent = str(group.get("intent", "exploration") or "exploration")
            expected_hits = int(group.get("expected_hits", 10) or 10)
            bucket: list[dict[str, Any]] = []
            for query in group.get("queries", []) or []:
                text = str(query or "").strip()
                if not text:
                    continue
                bucket.append(
                    {
                        "query": text,
                        "sources": list(source_preferences),
                        "tools_used": ["search_plan"],
                        "expected_hits": expected_hits,
                        "intent": intent,
                    }
                )
            if bucket:
                grouped_queries.append(bucket)

        if not grouped_queries:
            return []

        # Keep broader survey coverage by round-robining across query groups
        # instead of truncating the plan to the first few exploration queries.
        queries: list[dict[str, Any]] = []
        max_queries = min(
            MAX_PLANNED_QUERIES,
            sum(len(bucket) for bucket in grouped_queries),
        )
        cursor = 0
        while len(queries) < max_queries:
            progressed = False
            for bucket in grouped_queries:
                if cursor >= len(bucket):
                    continue
                queries.append(bucket[cursor])
                progressed = True
                if len(queries) >= max_queries:
                    break
            if not progressed:
                break
            cursor += 1
        return queries

    def _augmented_query_gen(self, brief: dict, search_plan: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Phase 1: 增强查询生成。

        TAG 的关键：LLM 推理时自驱动调用工具，不是预先定义好的调用。
        这里模拟 LLM 自驱动：先问 LLM "需要哪些工具"，然后执行，再继续。
        """
        planned_queries = self._queries_from_search_plan(search_plan)
        if planned_queries:
            if self.mm:
                self.mm.add_sensory("augmented_queries", {"queries": planned_queries, "source": "search_plan"})
            return {"queries": planned_queries, "phase": "augmented_query_gen"}

        from src.agent.llm import build_quick_llm
        from src.agent.settings import get_settings
        from langchain_core.messages import HumanMessage, SystemMessage

        settings = get_settings()
        llm = build_quick_llm(settings, max_tokens=2048)

        topic = brief.get("research_topic") or brief.get("topic", "")
        sub_questions = brief.get("sub_questions", [])
        if isinstance(sub_questions, str):
            sub_questions = [sub_questions]
        sq_text = "\n".join(f"- {sq}" for sq in sub_questions) if sub_questions else "(none)"

        brief_text = f"Topic: {topic}\nSub-questions:\n{sq_text}"

        # LLM 生成初始查询 + 判断是否需要工具增强
        try:
            resp = llm.invoke([
                SystemMessage(content=AUGMENTED_QUERY_PROMPT),
                HumanMessage(content=brief_text),
            ])
            raw = resp.content if hasattr(resp, "content") else str(resp)
            query_data = self._parse_json(raw)
        except Exception as exc:
            logger.warning("[RetrieverAgent] augmented query gen failed: %s", exc)
            query_data = None

        if not query_data or "queries" not in query_data:
            # Fallback：直接生成基础查询
            queries = [{"query": topic, "sources": ["arxiv"], "expected_hits": 20}]
            query_data = {"queries": queries}

        # TAG 内联增强：按需调用工具
        augmented_queries = []
        for item in query_data.get("queries", []):
            q = item.get("query", "")
            if not q:
                continue

            # 检查是否需要 expand_keywords（模拟 LLM 自驱动判断）
            if len(q) > 5 and len(sub_questions) > 1:
                try:
                    from src.tools.search_tools import expand_keywords as _expand_fn

                    kw_result = _expand_fn.invoke({"topic": q, "focus_dimension": "methods"})
                    expanded = [l.strip("- ").strip() for l in kw_result.split("\n") if l.strip()]
                    if expanded:
                        q = expanded[0]
                        if self.mm:
                            self.mm.add_tool_output("expand_keywords", {"original": item["query"], "expanded": expanded})
                except Exception:
                    pass

            # 检查是否需要 rewrite_query
            if len(q) > 30:
                try:
                    from src.tools.search_tools import rewrite_query as _rewrite_fn

                    rewritten = _rewrite_fn.invoke({"query": q, "mode": "precise"})
                    if rewritten and len(rewritten) < len(q):
                        q = rewritten
                        if self.mm:
                            self.mm.add_tool_output("rewrite_query", {"original": item["query"], "rewritten": q})
                except Exception:
                    pass

            item["query"] = q
            augmented_queries.append(item)

        if self.mm:
            self.mm.add_sensory("augmented_queries", {"queries": augmented_queries})

        return {"queries": augmented_queries, "phase": "augmented_query_gen"}

    # ── Phase 2: Parallel Retrieval ─────────────────────────────────────────

    def _parallel_retrieval(self, queries: list[dict]) -> dict[str, Any]:
        """
        Phase 2: 并行多源检索。

        TAG 的特点：工具结果是推理链的 context，而不是最终输出。
        这里将所有工具调用结果打包，供 Phase 3 内联使用。
        """
        import concurrent.futures

        from src.tools.search_tools import _searxng_search

        all_results: list[dict] = []
        query_order = {
            str(item.get("query", "")): idx
            for idx, item in enumerate(queries)
        }

        def _retrieve_one(item: dict) -> dict:
            q = item.get("query", "")
            sources = item.get("sources", ["arxiv"])
            expected_hits = item.get("expected_hits", 20)
            engines = ",".join(sources)

            result = _searxng_search(q, engines=engines, max_results=expected_hits)
            result["query"] = q
            result["query_meta"] = item
            result["query_order"] = query_order.get(q, 0)
            return result

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(queries))) as pool:
                futures = {pool.submit(_retrieve_one, q): q for q in queries}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        all_results.append(result)
                        if self.mm:
                            self.mm.add_tool_output(
                                "searxng",
                                {"query": result.get("query"), "hit_count": len(result.get("hits", []))},
                            )
                    except Exception as exc:
                        logger.warning("[RetrieverAgent] retrieval failed for query: %s", exc)
        except Exception as exc:
            logger.warning("[RetrieverAgent] parallel retrieval failed: %s", exc)

        return {"raw_results": all_results, "phase": "parallel_retrieval"}

    # ── Phase 3: Context Assembly (TAG 核心) ─────────────────────────────────

    def _context_assembly(self, brief: dict, raw_results: list[dict]) -> list[dict]:
        """
        Phase 3: 上下文组装（TAG 核心）。

        TAG 与 ReAct 的本质区别：
        - ReAct: 工具调用 → 观察结果 → LLM 推理 → 再调用 → ...
        - TAG:   工具调用 → 直接作为 LLM context → LLM 内联输出最终结果

        这里 LLM 直接基于原始检索结果推理，输出去重排序后的 candidate list，
        不经过中间状态转换。
        """
        from src.agent.llm import build_quick_llm
        from src.agent.settings import get_settings
        from langchain_core.messages import HumanMessage, SystemMessage

        settings = get_settings()
        llm = build_quick_llm(settings, max_tokens=4096)

        # 将检索结果整理为 LLM 可读的上下文
        context_lines = []
        total_hits = 0
        for res in raw_results:
            q = res.get("query", "?")
            hits = res.get("hits", [])
            total_hits += len(hits)
            sources = res.get("query_meta", {}).get("sources", [])
            context_lines.append(f"## Query: {q} (sources: {sources})")
            for i, hit in enumerate(hits[:6], 1):
                context_lines.append(
                    f"  [{i}] {hit.get('title', 'Unknown')}\n"
                    f"      URL: {hit.get('url', '')}\n"
                    f"      Abstract: {hit.get('content', '')[:220]}"
                )
            context_lines.append("")

        context_text = "\n".join(context_lines)
        max_candidates = 30

        # Candidate assembly is a refinement step, not a correctness-critical one.
        # When retrieval fan-out is already large, prefer deterministic fallback
        # instead of risking provider throttling on a giant context prompt.
        if len(context_text) > 15000 or total_hits > 40 or len(raw_results) > 8:
            logger.info(
                "[RetrieverAgent] using fallback candidate assembly (context_chars=%d, total_hits=%d, queries=%d)",
                len(context_text),
                total_hits,
                len(raw_results),
            )
            return self._fallback_candidates(raw_results)

        topic = brief.get("research_topic") or brief.get("topic", "")

        user_prompt = f"""## Research Topic

{topic}

## Retrieval Context

{context_text}

{CONTEXT_ASSEMBLY_PROMPT.replace("{max_candidates}", str(max_candidates))}
"""

        try:
            resp = llm.invoke([
                SystemMessage(content="You are a retrieval result assembly expert. Return JSON only with no explanation."),
                HumanMessage(content=user_prompt),
            ])
            raw = resp.content if hasattr(resp, "content") else str(resp)
            candidates = self._parse_json_array(raw)
        except Exception as exc:
            logger.warning("[RetrieverAgent] context assembly failed: %s", exc)
            candidates = []

        if not candidates:
            # Fallback：直接从 raw results 提取
            candidates = self._fallback_candidates(raw_results)

        return candidates

    # ── 完整 Pipeline ─────────────────────────────────────────────────────

    def run(self, brief: dict, search_plan: dict | None = None) -> dict[str, Any]:
        """
        完整 TAG Pipeline。

        流程：augmented_query_gen → parallel_retrieval → context_assembly
        """
        logger.info("[RetrieverAgent] Starting TAG pipeline via LangGraph")
        result = self.build_graph().invoke(
            {
                "brief": brief,
                "search_plan": search_plan or {},
                "warnings": [],
            },
            config=build_graph_config("retriever_agent"),
        )
        rag_result = result.get("rag_result")
        candidates = []
        if isinstance(rag_result, dict):
            candidates = list(rag_result.get("paper_candidates", []))

        logger.info("[RetrieverAgent] TAG pipeline done: %d candidates", len(candidates))
        return {
            "rag_result": rag_result,
            "raw_results_count": int(result.get("raw_results_count", 0)),
            "queries_generated": int(result.get("queries_generated", 0)),
            "paradigm": "tag",
            "summary": (
                f"TAG pipeline 完成：{int(result.get('queries_generated', 0))} 个增强查询 → "
                f"{len(candidates)} 篇候选论文"
            ),
            "retriever_warnings": list(result.get("warnings", [])),
        }

    # ── Fallback ─────────────────────────────────────────────────────────

    def _fallback_candidates(self, raw_results: list[dict]) -> list[dict]:
        """从 raw results 直接提取 candidate（当 context_assembly 失败时）。"""
        from src.tools.arxiv_api import enrich_search_results_with_arxiv

        normalized_results = sorted(
            list(raw_results),
            key=lambda item: int(item.get("query_order", 0) or 0),
        )
        seen_urls: set[str] = set()
        candidates: list[dict] = []
        rank = 1
        hit_queues: list[list[dict[str, Any]]] = [
            list(result.get("hits", []))
            for result in normalized_results
        ]

        while hit_queues and rank <= 30:
            progressed = False
            for idx, queue in enumerate(hit_queues):
                while queue:
                    hit = queue.pop(0)
                    url = hit.get("url", "")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    source_result = normalized_results[idx]
                    candidates.append({
                        "rank": rank,
                        "title": hit.get("title", ""),
                        "url": url,
                        "abstract": hit.get("content", "")[:500],
                        "source": hit.get("engine", "arxiv"),
                        "published_date": hit.get("publishedDate"),
                        "query": source_result.get("query", ""),
                    })
                    rank += 1
                    progressed = True
                    break
                if rank > 30:
                    break
            if not progressed:
                break

        return enrich_search_results_with_arxiv(candidates)

    def _recover_candidates_from_direct_sources(
        self,
        *,
        brief: dict[str, Any],
        search_plan: dict[str, Any],
        raw_results: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Recover candidates with arXiv-direct / DeepXiv when TAG search is sparse.

        The real supervisor path currently depends on SearXNG for TAG retrieval.
        When that leg returns few or zero hits, reuse the broader retrieval
        helpers from the graph search node so agent mode and node mode stay
        aligned.
        """
        from src.research.graph.nodes.search import _time_filter_from_range
        from src.tools.arxiv_api import (
            DEFAULT_YEAR_FILTER,
            enrich_search_results_with_arxiv,
            search_arxiv_direct,
        )
        from src.tools.deepxiv_client import search_papers

        total_hits = sum(len(result.get("hits", [])) for result in raw_results if isinstance(result, dict))
        if len(candidates) >= 6 or (candidates and total_hits > 0):
            return candidates, None

        all_queries: list[tuple[str, str, int]] = []
        seen_queries: set[str] = set()
        for result in raw_results:
            query = str(result.get("query") or "").strip()
            meta = result.get("query_meta") if isinstance(result.get("query_meta"), dict) else {}
            expected_hits = int(meta.get("expected_hits", 10) or 10)
            intent = str(meta.get("intent", "exploration") or "exploration")
            if not query or query in seen_queries:
                continue
            seen_queries.add(query)
            all_queries.append((query, intent, expected_hits))

        if not all_queries:
            for item in self._queries_from_search_plan(search_plan):
                query = str(item.get("query") or "").strip()
                if not query or query in seen_queries:
                    continue
                seen_queries.add(query)
                all_queries.append(
                    (
                        query,
                        str(item.get("intent", "exploration") or "exploration"),
                        int(item.get("expected_hits", 10) or 10),
                    )
                )

        if not all_queries:
            return candidates, None

        effective_year_filter = _time_filter_from_range(
            str(search_plan.get("time_range") or brief.get("time_range") or "")
        ) or DEFAULT_YEAR_FILTER

        recovery_queries = all_queries[:4]
        direct_candidates: list[dict[str, Any]] = []
        for idx, (query, _, _) in enumerate(recovery_queries):
            direct_candidates.extend(
                search_arxiv_direct(
                    query,
                    max_results=6,
                    year_filter=effective_year_filter,
                )
            )
            if idx < len(recovery_queries) - 1:
                time.sleep(0.8)

        deepxiv_candidates: list[dict[str, Any]] = []
        deepxiv_date_from = (
            f"{effective_year_filter}-01-01"
            if str(effective_year_filter).isdigit()
            else None
        )
        for query, _, _ in recovery_queries[:3]:
            deepxiv_candidates.extend(
                search_papers(
                    query,
                    size=6,
                    date_from=deepxiv_date_from,
                )
            )

        merged: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        def _add_many(items: list[dict[str, Any]]) -> None:
            for item in items:
                arxiv_id = str(item.get("arxiv_id") or "").strip().lower()
                url = str(item.get("url") or "").strip().lower()
                title = str(item.get("title") or "").strip().lower()
                key = arxiv_id or url or title
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                merged.append(item)

        _add_many(direct_candidates)
        _add_many(deepxiv_candidates)
        _add_many(candidates)

        if len(merged) <= len(candidates):
            return candidates, None

        recovered = enrich_search_results_with_arxiv(merged)
        note = (
            "TAG sparse-retrieval fallback added "
            f"{len(recovered) - len(candidates)} candidate(s) via throttled arXiv direct / DeepXiv."
        )
        logger.info("[RetrieverAgent] %s", note)
        return recovered, note

    # ── Helpers ─────────────────────────────────────────────────────────

    def _parse_json(self, text: str) -> dict | None:
        import json

        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _parse_json_array(self, text: str) -> list:
        import json

        text = text.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return []

    def build_graph(self):
        workflow = StateGraph(RetrieverGraphState)
        workflow.add_node("augmented_query_gen", self._query_node)
        workflow.add_node("parallel_retrieval", self._retrieval_node)
        workflow.add_node("context_assembly", self._assembly_node)
        workflow.add_node("finalize_rag_result", self._finalize_node)
        workflow.add_edge(START, "augmented_query_gen")
        workflow.add_edge("augmented_query_gen", "parallel_retrieval")
        workflow.add_edge("parallel_retrieval", "context_assembly")
        workflow.add_edge("context_assembly", "finalize_rag_result")
        return workflow.compile(checkpointer=get_langgraph_checkpointer("retriever_agent"))

    def _query_node(self, state: "RetrieverGraphState") -> dict[str, Any]:
        result = self._augmented_query_gen(
            state.get("brief") or {},
            state.get("search_plan") or {},
        )
        queries = list(result.get("queries", []))
        if self.mm:
            self.mm.add_sensory(
                "phase_completed",
                {"phase": "augmented_query_gen", "query_count": len(queries)},
            )
        return {"queries": queries, "queries_generated": len(queries)}

    def _retrieval_node(self, state: "RetrieverGraphState") -> dict[str, Any]:
        result = self._parallel_retrieval(list(state.get("queries", [])))
        raw_results = list(result.get("raw_results", []))
        return {
            "raw_results": raw_results,
            "raw_results_count": len(raw_results),
        }

    def _assembly_node(self, state: "RetrieverGraphState") -> dict[str, Any]:
        candidates = self._context_assembly(
            state.get("brief") or {},
            list(state.get("raw_results", [])),
        )
        if self.mm and candidates:
            try:
                for cand in candidates[:5]:
                    self.mm.add_semantic(
                        f"检索到论文: {cand.get('title', '')[:100]}",
                        memory_type="research_fact",
                        metadata={"source": "retriever_agent", "workspace_id": self.workspace_id},
                    )
            except Exception as exc:
                logger.warning("[RetrieverAgent] Failed to store semantic memory: %s", exc)
        return {"candidates": candidates}

    def _finalize_node(self, state: "RetrieverGraphState") -> dict[str, Any]:
        candidates = list(state.get("candidates", []))
        rag_result = self._build_rag_result(
            brief=state.get("brief") or {},
            search_plan=state.get("search_plan") or {},
            raw_results=list(state.get("raw_results", [])),
            candidates=candidates,
        )
        return {"rag_result": rag_result}

    def _build_rag_result(
        self,
        *,
        brief: dict[str, Any],
        search_plan: dict[str, Any],
        raw_results: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        from src.research.graph.nodes.search import (
            _ingest_paper_candidates,
            _rerank_and_filter_candidates,
        )

        query = search_plan.get("plan_goal") or brief.get("research_topic") or brief.get("topic", "")
        candidates, recovery_note = self._recover_candidates_from_direct_sources(
            brief=brief,
            search_plan=search_plan,
            raw_results=raw_results,
            candidates=candidates,
        )
        candidates, rerank_log = _rerank_and_filter_candidates(
            candidates,
            brief=brief,
            search_plan=search_plan,
        )
        query_traces = []
        for result in raw_results:
            query_traces.append(
                {
                    "query": result.get("query", ""),
                    "status": "success" if result.get("ok", True) else "error",
                    "hits_count": len(result.get("hits", [])),
                }
            )

        coverage_notes = [
            f"TAG agent 执行 {len(raw_results)} 组检索，二次筛选后保留 {len(candidates)} 篇候选论文",
        ]
        if recovery_note:
            coverage_notes.append(recovery_note)

        rag_result = RagResult(
            query=query,
            sub_questions=list(brief.get("sub_questions", [])) if isinstance(brief.get("sub_questions"), list) else [],
            rag_strategy="tag_augmented_query + parallel_retrieval + context_assembly",
            paper_candidates=candidates,
            evidence_chunks=[],
            retrieval_trace=query_traces,
            dedup_log=[{"strategy": "url/title", "total": len(candidates), "unique": len(candidates)}],
            rerank_log=rerank_log,
            coverage_notes=coverage_notes,
            total_papers=len(candidates),
            total_chunks=0,
            retrieved_at="",
        )
        try:
            _ingest_paper_candidates(candidates, workspace_id=self.workspace_id)
        except Exception as exc:
            logger.warning("[RetrieverAgent] Failed to ingest candidates: %s", exc)
        return asdict(rag_result)


class RetrieverGraphState(TypedDict, total=False):
    brief: dict[str, Any]
    search_plan: dict[str, Any]
    warnings: list[str]
    queries: list[dict[str, Any]]
    queries_generated: int
    raw_results: list[dict[str, Any]]
    raw_results_count: int
    candidates: list[dict[str, Any]]
    rag_result: dict[str, Any]


# ─── 入口函数 ────────────────────────────────────────────────────────────────


def run_retriever_agent(state: dict, inputs: dict) -> dict:
    """RetrieverAgent 入口（兼容 supervisor 格式）。"""
    workspace_id = inputs.get("workspace_id") or state.get("workspace_id")
    task_id = inputs.get("task_id") or state.get("task_id")
    brief = state.get("brief") or inputs.get("brief", {})
    search_plan = state.get("search_plan")
    emitter = inputs.get("_event_emitter")

    agent = RetrieverAgent(workspace_id=workspace_id, task_id=task_id)
    try:
        if emitter:
            emitter.on_thinking("search", "Retriever agent is executing diversified retrieval over the planned queries.")
        result = agent.run(brief=brief, search_plan=search_plan)
        if emitter:
            rag_result = result.get("rag_result") if isinstance(result, dict) else None
            if isinstance(rag_result, dict):
                paper_count = len(rag_result.get("paper_candidates", []) or [])
            else:
                paper_count = len(getattr(rag_result, "paper_candidates", []) or [])
            emitter.on_thinking("search", f"Retriever kept {paper_count} paper candidates after reranking.")
        return result
    except Exception as exc:
        logger.exception("[RetrieverAgent] run failed: %s", exc)
        return {
            "candidates": [],
            "paradigm": "tag",
            "retriever_warnings": [f"RetrieverAgent failed: {exc}"],
        }
