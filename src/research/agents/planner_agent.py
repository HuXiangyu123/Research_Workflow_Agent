"""PlannerAgent — Plan-and-Execute 模式。

设计模式说明：
- Plan-and-Execute：先规划（生成完整执行计划），再执行。
- 核心思想：LLM 先全面规划，然后由执行器按序执行，降低单步决策错误累积。
- 适用场景：复杂多步骤任务，如本研究调研。

与 ReAct 的区别：
- ReAct：每步推理后立即执行，再观察，再推理（紧密交织）
- Plan-and-Execute：先一次性规划完整路径，再逐一执行（松耦合）

阶段划分：
  Phase 1 (Plan):   LLM 一次性生成完整 SearchPlan（含所有 query_groups）
  Phase 2 (Execute): 执行器按 plan 执行，每步结果记录到 memory
  Phase 3 (Validate): 验证 plan 执行结果是否满足初始目标
"""

from __future__ import annotations

import logging
from typing import Any
from typing import TypedDict

from langgraph.graph import START, StateGraph
from src.agent.checkpointing import build_graph_config, get_langgraph_checkpointer
from src.memory.manager import get_memory_manager

logger = logging.getLogger(__name__)


# ─── Prompt Templates ───────────────────────────────────────────────────────


PLAN_PHASE_PROMPT = """You are a research planning expert.

Given the ResearchBrief below, generate a complete multi-stage SearchPlan in one pass.

Output requirements (strict JSON):
```json
{{
  "plan_goal": "One-sentence goal for this search run",
  "query_groups": [
    {{
      "group_id": "exploration_stage",
      "queries": ["query 1", "query 2", "..."],
      "intent": "exploration",
      "priority": 1,
      "expected_hits": 20,
      "notes": "Why this stage exists"
    }},
    {{
      "group_id": "refinement_stage",
      "queries": ["query 1", "query 2", "..."],
      "intent": "refinement",
      "priority": 2,
      "expected_hits": 15,
      "notes": "How this stage narrows the search"
    }},
    {{
      "group_id": "validation_stage",
      "queries": ["query 1", "..."],
      "intent": "validation",
      "priority": 3,
      "expected_hits": 10,
      "notes": "How this stage checks coverage or gaps"
    }}
  ],
  "source_preferences": ["arxiv", "semantic_scholar"],
  "coverage_notes": "Expected coverage and known blind spots"
}}
```

Rules:
- Use 2-4 query groups, never more than 4.
- Keep each query between 5 and 15 words.
- Cover the research topic, methods, datasets, and applications.
- expected_hits is an estimate, not a hard requirement.
"""

EXECUTE_PHASE_SYSTEM = """You are an execution validator.

Given the ResearchBrief, the generated SearchPlan, and the current execution results, decide:
1. whether current progress satisfies the plan goal,
2. whether downstream queries should be adjusted,
3. whether execution should stop.

Output (strict JSON):
```json
{{
  "status": "on_track | deviation | complete | stuck",
  "current_coverage": "Coverage summary",
  "adjustments": ["Adjustment 1", "Adjustment 2"],
  "should_stop": true,
  "stop_reason": "Reason for stopping if applicable"
}}
```
"""


# ─── PlannerAgent ──────────────────────────────────────────────────────────


class PlanAndExecutePhase:
    """单个执行阶段。"""

    def __init__(self, group_id: str, queries: list[str], intent: str, priority: int, expected_hits: int):
        self.group_id = group_id
        self.queries = queries
        self.intent = intent
        self.priority = priority
        self.expected_hits = expected_hits
        self.executed_queries: list[str] = []
        self.query_results: list[dict] = []
        self.completed = False

    def mark_executed(self, query: str, result: dict) -> None:
        self.executed_queries.append(query)
        self.query_results.append(result)
        if len(self.executed_queries) >= len(self.queries):
            self.completed = True

    def to_payload(self) -> dict[str, Any]:
        """Convert runtime phase state into a checkpoint-safe dict."""
        return {
            "group_id": self.group_id,
            "queries": list(self.queries),
            "intent": self.intent,
            "priority": self.priority,
            "expected_hits": self.expected_hits,
            "executed_queries": list(self.executed_queries),
            "query_results": list(self.query_results),
            "completed": self.completed,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> PlanAndExecutePhase:
        """Restore a runtime phase object from checkpoint-safe state."""
        phase = cls(
            group_id=str(payload.get("group_id", "")),
            queries=[str(item) for item in payload.get("queries", [])],
            intent=str(payload.get("intent", "")),
            priority=int(payload.get("priority", 0) or 0),
            expected_hits=int(payload.get("expected_hits", 0) or 0),
        )
        phase.executed_queries = [str(item) for item in payload.get("executed_queries", [])]
        phase.query_results = list(payload.get("query_results", []))
        phase.completed = bool(payload.get("completed", False))
        return phase


class PlannerGraphState(TypedDict, total=False):
    brief: dict[str, Any]
    warnings: list[str]
    plan: dict[str, Any] | None
    phases: list[Any]
    search_plan: dict[str, Any] | None
    candidates: list[dict[str, Any]]
    execution_log: list[dict[str, Any]]
    execution_results: dict[str, Any]
    validation: dict[str, Any]


class PlannerAgent:
    """
    Plan-and-Execute 模式的 Planner Agent。

    工作流程：
      ┌─────────────┐
      │  PLAN PHASE │  LLM 一次性生成完整 SearchPlan
      └──────┬──────┘
             │  SearchPlan
             ▼
      ┌─────────────┐
      │ EXECUTE LOOP │  按 priority 顺序执行每个 query_group
      │  Phase 1..N   │  每步记录结果到 memory
      └──────┬──────┘
             │  execution_results
             ▼
      ┌─────────────┐
      │ VALIDATE     │  LLM 验证执行结果是否满足目标
      └──────┬──────┘
             │  final_plan + validation_report
    """

    def __init__(self, workspace_id: str | None = None, task_id: str | None = None):
        self.workspace_id = workspace_id
        self.task_id = task_id
        self.mm = get_memory_manager(workspace_id) if workspace_id else None

    # ── Plan Phase ──────────────────────────────────────────────────────────

    def plan_phase(self, brief: dict) -> dict[str, Any]:
        """
        PLAN PHASE：LLM 一次性生成完整多阶段执行计划。

        与 SearchPlanAgent 的区别：
        - SearchPlanAgent：边推理边执行（ReAct）
        - PlannerAgent.plan_phase()：一次性生成所有阶段，再执行
        """
        from src.agent.llm import build_reason_llm
        from src.agent.settings import get_settings
        from src.models.research import SearchPlan
        from langchain_core.messages import HumanMessage, SystemMessage

        settings = get_settings()
        llm = build_reason_llm(settings, max_tokens=8192)

        brief_text = self._serialize_brief(brief)
        memory_context = ""
        if self.mm:
            try:
                topic = brief.get("research_topic") or brief.get("topic", "")
                memory_context = self.mm.build_context(topic=topic, max_semantic=3, max_episodes=3)
            except Exception:
                pass

        user_prompt = f"""## ResearchBrief

{brief_text}

## Historical Memory (Reference Only)

{memory_context or '(none)'}

Generate the complete multi-stage search plan from the context above."""

        try:
            resp = llm.invoke([
                SystemMessage(content=PLAN_PHASE_PROMPT),
                HumanMessage(content=user_prompt),
            ])
            raw = resp.content if hasattr(resp, "content") else str(resp)
            plan_data = self._parse_json(raw)

            if plan_data:
                plan = SearchPlan.model_validate(self._coerce_plan_payload(plan_data))
                self._store_plan_memory(plan, brief)
                return {
                    "plan": plan,
                    "phases": self._build_phases(plan),
                    "phase": "plan",
                    "plan_summary": plan.plan_goal,
                }
        except Exception as exc:
            logger.exception("[PlannerAgent.plan_phase] failed: %s", exc)

        # Fallback
        from src.research.policies.search_plan_policy import to_fallback_plan

        plan = to_fallback_plan(brief)
        return {
            "plan": plan,
            "phases": self._build_phases(plan),
            "phase": "plan",
            "plan_summary": plan.plan_goal,
            "warnings": ["LLM 规划失败，使用 fallback plan"],
        }

    # ── Execute Phase ─────────────────────────────────────────────────────

    def execute_phase(self, phases: list[Any]) -> dict[str, Any]:
        """
        EXECUTE PHASE：按 priority 顺序执行每个阶段的 queries。

        Plan-and-Execute 的关键设计：
        - 按阶段批量执行，而非逐条执行（提高效率）
        - 每个阶段执行完毕后才进入下一阶段（保持阶段边界）
        - 阶段之间可记录 memory，供 validate 使用
        """
        from src.tools.search_tools import _searxng_search

        all_candidates: list[dict] = []
        seen_urls: set[str] = set()
        execution_log: list[dict] = []

        normalized_phases = self._restore_phases(phases)

        for phase in normalized_phases:
            if phase.completed:
                continue

            logger.info("[PlannerAgent.execute] phase=%s, queries=%d", phase.group_id, len(phase.queries))

            # Runtime event buffer: record phase start.
            if self.mm:
                self.mm.add_sensory(
                    "phase_start",
                    {"phase": phase.group_id, "intent": phase.intent, "query_count": len(phase.queries)},
                )

            for query in phase.queries:
                if query in phase.executed_queries:
                    continue

                try:
                    result = _searxng_search(query, engines="arxiv", max_results=20)
                    if result.get("ok"):
                        hits = result.get("hits", [])
                        phase.mark_executed(query, result)

                        for hit in hits:
                            url = hit.get("url", "")
                            if url and url not in seen_urls:
                                seen_urls.add(url)
                                all_candidates.append({
                                    "title": hit.get("title", ""),
                                    "url": url,
                                    "abstract": hit.get("content", "")[:500],
                                    "engine": hit.get("engine", "arxiv"),
                                    "published_date": hit.get("publishedDate"),
                                    "query": query,
                                    "phase": phase.group_id,
                                    "intent": phase.intent,
                                })
                    else:
                        phase.mark_executed(query, {"ok": False, "error": result.get("error", "")})

                    # Runtime event buffer: record each query result.
                    if self.mm:
                        self.mm.add_tool_output("searxng", {"query": query, "result": result})

                except Exception as exc:
                    logger.warning("[PlannerAgent.execute] query '%s' failed: %s", query, exc)
                    phase.mark_executed(query, {"ok": False, "error": str(exc)})

            # Runtime vector cache: record phase summary.
            if self.mm:
                self.mm.add_semantic(
                    f"完成搜索阶段: {phase.group_id}，intent={phase.intent}，queries={len(phase.executed_queries)}",
                    memory_type="research_fact",
                    metadata={"source": "planner_agent", "workspace_id": self.workspace_id},
                )

            execution_log.append({
                "phase": phase.group_id,
                "intent": phase.intent,
                "executed": len(phase.executed_queries),
                "total": len(phase.queries),
                "candidates_from_phase": len([c for c in all_candidates if c.get("phase") == phase.group_id]),
            })

        return {
            "candidates": all_candidates,
            "execution_log": execution_log,
            "phase": "execute",
        }

    # ── Validate Phase ─────────────────────────────────────────────────────

    def validate_phase(self, brief: dict, plan: Any, execution_results: dict) -> dict[str, Any]:
        """
        VALIDATE PHASE：LLM 验证执行结果是否满足计划目标。

        Plan-and-Execute 的最后一步：
        - ReAct 无此步骤（每步即验证）
        - 这里显式验证：给 LLM 完整上下文做全局判断
        """
        from src.agent.llm import build_reason_llm
        from src.agent.settings import get_settings
        from langchain_core.messages import HumanMessage, SystemMessage

        settings = get_settings()
        llm = build_reason_llm(settings, max_tokens=2048)

        brief_text = self._serialize_brief(brief)
        plan_goal = plan.plan_goal if hasattr(plan, "plan_goal") else str(plan)
        execution_summary = self._summarize_execution(execution_results)

        user_prompt = f"""## ResearchBrief

{brief_text}

        ## SearchPlan Goal

{plan_goal}

## Execution Summary

{execution_summary}

Validate whether the execution results satisfy the SearchPlan goal."""

        try:
            resp = llm.invoke([
                SystemMessage(content=EXECUTE_PHASE_SYSTEM),
                HumanMessage(content=user_prompt),
            ])
            raw = resp.content if hasattr(resp, "content") else str(resp)
            validation = self._parse_json(raw) or {}

            return {
                "phase": "validate",
                "validation": validation,
                "candidates": execution_results.get("candidates", []),
                "execution_log": execution_results.get("execution_log", []),
            }
        except Exception as exc:
            logger.warning("[PlannerAgent.validate_phase] failed: %s", exc)
            return {
                "phase": "validate",
                "validation": {"status": "unknown", "error": str(exc)},
                "candidates": execution_results.get("candidates", []),
                "execution_log": execution_results.get("execution_log", []),
            }

    # ── 完整 Pipeline ─────────────────────────────────────────────────────

    def run(self, brief: dict) -> dict[str, Any]:
        """
        完整 Plan-and-Execute Pipeline。

        流程：plan_phase → execute_phase → validate_phase
        """
        logger.info("[PlannerAgent] Starting Plan-and-Execute pipeline via LangGraph")
        result = self.build_graph().invoke(
            {"brief": brief, "warnings": []},
            config=build_graph_config("planner_agent"),
        )
        candidates = list(result.get("candidates", []))
        return {
            "search_plan": result.get("search_plan"),
            "candidates": candidates,
            "execution_log": list(result.get("execution_log", [])),
            "validation": result.get("validation", {}),
            "paradigm": "plan_and_execute",
            "summary": f"Plan-and-Execute 完成：{len(candidates)} 篇候选论文",
            "planner_warnings": list(result.get("warnings", [])),
        }

    # ── Helpers ────────────────────────────────────────────────────────────

    def _build_phases(self, plan: Any) -> list[PlanAndExecutePhase]:
        """从 SearchPlan 构建可执行的阶段列表。"""
        phases = []
        for g in plan.query_groups:
            phase = PlanAndExecutePhase(
                group_id=g.group_id,
                queries=list(g.queries),
                intent=g.intent,
                priority=g.priority,
                expected_hits=g.expected_hits,
            )
            phases.append(phase)
        # 按 priority 排序
        phases.sort(key=lambda p: p.priority)
        return phases

    def _coerce_plan_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Normalize minor schema drift from the planner LLM before validation."""
        normalized = dict(payload)
        raw_groups = payload.get("query_groups", [])
        groups: list[dict[str, Any]] = []

        for idx, raw_group in enumerate(raw_groups[:4]):
            if not isinstance(raw_group, dict):
                continue

            group = dict(raw_group)
            queries = group.get("queries", [])
            if not isinstance(queries, list):
                queries = [queries] if queries else []

            try:
                raw_priority = group.get("priority", idx + 1)
                priority = int(raw_priority if raw_priority is not None else idx + 1)
            except (TypeError, ValueError):
                priority = idx + 1

            try:
                raw_expected_hits = group.get("expected_hits", 10)
                expected_hits = int(raw_expected_hits if raw_expected_hits is not None else 10)
            except (TypeError, ValueError):
                expected_hits = 10

            groups.append(
                {
                    **group,
                    "group_id": str(group.get("group_id") or f"group_{idx + 1}"),
                    "queries": [str(query).strip() for query in queries if str(query).strip()][:8],
                    "intent": str(group.get("intent") or "exploration"),
                    "priority": min(3, max(1, priority)),
                    "expected_hits": min(50, max(1, expected_hits)),
                }
            )

        normalized["query_groups"] = groups
        return normalized

    def _serialize_plan(self, plan: Any) -> dict[str, Any] | None:
        if plan is None:
            return None
        if isinstance(plan, dict):
            return dict(plan)
        if hasattr(plan, "model_dump"):
            return plan.model_dump(mode="json")
        return {"plan_goal": str(plan)}

    def _serialize_phases(self, phases: list[Any]) -> list[Any]:
        payloads: list[Any] = []
        for phase in phases:
            if isinstance(phase, PlanAndExecutePhase):
                payloads.append(phase.to_payload())
            elif isinstance(phase, dict):
                payloads.append(dict(phase))
            else:
                payloads.append(phase)
        return payloads

    def _restore_phases(self, phases: list[Any]) -> list[PlanAndExecutePhase]:
        restored: list[PlanAndExecutePhase] = []
        for phase in phases:
            if isinstance(phase, PlanAndExecutePhase):
                restored.append(phase)
            elif isinstance(phase, dict):
                restored.append(PlanAndExecutePhase.from_payload(phase))
        return restored

    def _serialize_brief(self, brief: dict) -> str:
        import json

        topic = brief.get("research_topic") or brief.get("topic", "")
        goal = brief.get("goal", "")
        sub_qs = brief.get("sub_questions", [])
        output = brief.get("desired_output", "")
        return f"Topic: {topic}\nGoal: {goal}\nSub-questions: {json.dumps(sub_qs, ensure_ascii=False)}\nDesired output: {output}"

    def _store_plan_memory(self, plan: Any, brief: dict) -> None:
        if not self.mm:
            return
        try:
            topic = brief.get("research_topic") or brief.get("topic", "")
            self.mm.add_semantic(
                f"SearchPlan: {plan.plan_goal}",
                memory_type="research_fact",
                metadata={
                    "source": "planner_agent",
                    "group_count": len(plan.query_groups),
                    "workspace_id": self.workspace_id,
                },
            )
        except Exception as exc:
            logger.warning("[PlannerAgent] Failed to store plan memory: %s", exc)

    def _summarize_execution(self, results: dict) -> str:
        candidates = results.get("candidates", [])
        log = results.get("execution_log", [])
        total = len(candidates)
        lines = [f"总候选论文数：{total}", f"执行阶段数：{len(log)}"]
        for entry in log:
            lines.append(f"  - {entry['phase']} ({entry['intent']}): {entry['executed']}/{entry['total']} queries, {entry['candidates_from_phase']} candidates")
        return "\n".join(lines)

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

    def build_graph(self):
        workflow = StateGraph(PlannerGraphState)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("validate", self._validate_node)
        workflow.add_edge(START, "plan")
        workflow.add_edge("plan", "execute")
        workflow.add_edge("execute", "validate")
        return workflow.compile(checkpointer=get_langgraph_checkpointer("planner_agent"))

    def _plan_node(self, state: PlannerGraphState) -> dict[str, Any]:
        result = self.plan_phase(state.get("brief") or {})
        plan = result.get("plan")
        warnings = list(state.get("warnings", []))
        warnings.extend(result.get("warnings", []))
        serialized_plan = self._serialize_plan(plan)
        return {
            "plan": serialized_plan,
            "phases": self._serialize_phases(list(result.get("phases", []))),
            "search_plan": serialized_plan,
            "warnings": warnings,
        }

    def _execute_node(self, state: PlannerGraphState) -> dict[str, Any]:
        phases = list(state.get("phases", []))
        result = self.execute_phase(phases)
        return {
            "execution_results": result,
            "candidates": list(result.get("candidates", [])),
            "execution_log": list(result.get("execution_log", [])),
        }

    def _validate_node(self, state: PlannerGraphState) -> dict[str, Any]:
        plan = state.get("search_plan") or state.get("plan")
        execution_results = state.get("execution_results") or {
            "candidates": state.get("candidates", []),
            "execution_log": state.get("execution_log", []),
        }
        if plan is None:
            warnings = list(state.get("warnings", []))
            warnings.append("Planner graph reached validation without a plan.")
            return {"validation": {"status": "missing_plan"}, "warnings": warnings}

        runtime_plan = plan
        if isinstance(plan, dict):
            try:
                from src.models.research import SearchPlan

                runtime_plan = SearchPlan.model_validate(plan)
            except Exception:
                runtime_plan = plan

        result = self.validate_phase(state.get("brief") or {}, runtime_plan, execution_results)
        return {
            "validation": result.get("validation", {}),
            "candidates": list(result.get("candidates", [])),
            "execution_log": list(result.get("execution_log", [])),
        }


# ─── 入口函数 ───────────────────────────────────────────────────────────────


def run_planner_agent(state: dict, inputs: dict) -> dict:
    """PlannerAgent 入口（兼容 supervisor 格式）。"""
    workspace_id = inputs.get("workspace_id") or state.get("workspace_id")
    task_id = inputs.get("task_id") or state.get("task_id")
    brief = state.get("brief") or inputs.get("brief", {})
    emitter = inputs.get("_event_emitter")

    agent = PlannerAgent(workspace_id=workspace_id, task_id=task_id)
    try:
        if emitter:
            emitter.on_thinking("search_plan", "Planner agent is expanding the brief into executable search phases.")
        result = agent.run(brief=brief)
        if emitter and result.get("search_plan"):
            query_groups = result.get("search_plan", {}).get("query_groups", [])
            emitter.on_thinking("search_plan", f"Planner produced {len(query_groups)} query groups.")
        return result
    except Exception as exc:
        logger.exception("[PlannerAgent] run failed: %s", exc)
        return {
            "search_plan": None,
            "candidates": [],
            "paradigm": "plan_and_execute",
            "planner_warnings": [f"PlannerAgent failed: {exc}"],
        }


def plan_search(state: dict, inputs: dict) -> dict:
    """兼容别名。"""
    return run_planner_agent(state, inputs)
