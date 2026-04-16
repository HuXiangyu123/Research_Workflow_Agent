"""ReviewerAgent — Reflexion 模式。

设计模式说明：
- Reflexion = 口头强化（Verbal Reinforcement）：Agent 在每次失败后，
  生成自我反思（self-reflection），将反思结果存储到运行期 memory adapter，
  下次遇到相似任务时检索并利用这些反思。
- 核心思想：
  1. Agent 执行任务并获得结果（成功或失败）
  2. LLM 自我反思：分析失败原因，生成可操作的改进建议
  3. 反思结果存入 Long-Term Memory（Episodic + Semantic）
  4. 下次执行前，从 Memory 中检索相似失败经验，避免重复犯错

三组件：
  1. Actor：执行任务，生成结果（这里复用 ReviewerService）
  2. Evaluator：判断结果是否达标（passed/failed）
  3. Self-Reflector：失败时生成深度反思，存入 memory

与 ReAct 的区别：
- ReAct：每步都有观察和反思（in-loop），反思是即时的
- Reflexion：反思是跨 episode 的，失败触发深度反思，结果持久化到 memory
- 本质区别：ReAct 的反思是瞬时的，Reflexion 的反思是持久化的

与 Plan-and-Execute 的区别：
- Plan-and-Execute：规划先行，执行在后，验证是全局的
- Reflexion：执行先行，失败触发反思，记忆驱动改进

迭代流程：
  Attempt 1: Actor → Evaluator → Failed? → Self-Reflector → Memory
  Attempt 2: Memory Retrieval → Actor (with reflection) → Evaluator → ...
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from dataclasses import asdict, dataclass, field
from inspect import isawaitable
from typing import Any
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from src.agent.checkpointing import build_graph_config, get_langgraph_checkpointer
from src.memory.manager import get_memory_manager

logger = logging.getLogger(__name__)


# ─── Self-Reflection 定义 ──────────────────────────────────────────────────


@dataclass
class SelfReflection:
    """自我反思条目：某次失败的深度分析。"""

    reflection_id: str
    task_type: str           # "review_coverage" | "review_claims" | "review_citations"
    failure_context: str     # 失败的具体上下文
    root_cause: str         # 根本原因（LLM 生成）
    lessons: list[str]      # 可操作的改进建议（LLM 生成）
    improved_strategy: str   # 下次改进的策略
    confidence_gain: float  # 预期置信度提升（0-1）
    created_at: float       # 时间戳

    def to_prompt(self) -> str:
        lines = [
            f"## Failure Case: {self.task_type}",
            f"**Failure context**: {self.failure_context}",
            f"**Root cause**: {self.root_cause}",
            f"**Improved strategy**: {self.improved_strategy}",
        ]
        if self.lessons:
            lines.append("**Lessons**:")
            for lesson in self.lessons:
                lines.append(f"- {lesson}")
        return "\n".join(lines)


# ─── Reflexion Prompt Templates ───────────────────────────────────────────────


SELF_REFLECTION_PROMPT = """You are a self-reflection expert.

Given the following review failure case, generate a deep self-reflection:

Task type: {task_type}
Failure context: {failure_context}
Failure details: {failure_details}

Analyze:
1. Root cause: why did this fail, and at which step?
2. Lessons: how should the system avoid this next time? Provide 2-3 actionable suggestions.
3. Improved strategy: what concrete strategy should be used for similar tasks in the future?

Output (strict JSON):
{{
  "root_cause": "Root-cause analysis",
  "lessons": ["Lesson 1", "Lesson 2", "Lesson 3"],
  "improved_strategy": "Concrete strategy for the next attempt"
}}
"""


MEMORY_RETRIEVAL_PROMPT = """You are an experience retrieval expert.

Given the current review task, choose the 2-3 most relevant prior failure lessons.

Current task: {task_description}
Historical failure lessons: {reflections}

Select the most relevant lessons and explain why they apply.
"""


# ─── ReviewerAgent ──────────────────────────────────────────────────────────


class ReviewerAgent:
    """
    Reflexion 模式的 Reviewer Agent。

    工作流程：
      ┌──────────────────────────────────────┐
      │ ATTEMPT LOOP                          │
      │                                       │
      │  ┌────────┐   ┌──────────┐  ┌───────┐ │
      │  │ MEMORY │ → │  ACTOR   │→│EVAL   │ │
      │  │ RETRIEVE│  │(ReviewSvc)│  │(check)│ │
      │  └────────┘   └──────────┘  └───┬───┘ │
      │                                  │     │
      │                    ┌─────────────┴──┐   │
      │                    │ Failed?        │   │
      │                    └───┬─────────────┘   │
      │                        │ yes            │ no
      │                        ▼                ▼
      │                   ┌────────┐        DONE
      │                   │SELF-REF│
      │                   │LECTOR  │
      │                   └───┬────┘
      │                       │ self-reflection
      │                       ▼
      │                   ┌────────┐
      │                   │MEMORY  │
      │                   │STORE   │
      │                   └───┬────┘
      └───────────────────────┼────────────────┘
                              │ (store & retry)
                              └→ next attempt
    """

    MAX_ATTEMPTS = 3
    REFLECTION_THRESHOLD = 0.7  # confidence < 此值则触发反思

    def __init__(self, workspace_id: str | None = None, task_id: str | None = None):
        self.workspace_id = workspace_id
        self.task_id = task_id
        self.mm = get_memory_manager(workspace_id) if workspace_id else None
        self._reflections: list[SelfReflection] = []

    def run(
        self,
        brief: dict,
        paper_cards: list,
        draft_report: Any | None,
        rag_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        完整 Reflexion Pipeline。

        与其他模式的关键区别：
        - 失败不是终点，而是学习机会
        - 每次失败生成 self-reflection，存入 memory
        - 下次尝试时检索历史 reflection，调整策略
        """
        logger.info("[ReviewerAgent] Starting Reflexion pipeline via LangGraph")
        result = self.build_graph().invoke(
            {
                "brief": brief,
                "paper_cards": paper_cards,
                "draft_report": draft_report,
                "rag_result": rag_result or {},
                "attempt": 0,
                "attempt_history": [],
                "prior_reflections": [],
            },
            config=build_graph_config(
                "reviewer_agent",
                recursion_limit=self.MAX_ATTEMPTS * 4 + 10,
            ),
        )
        return {
            "review_passed": bool(result.get("review_passed")),
            "review_feedback": result.get("review_feedback"),
            "attempt_history": list(result.get("attempt_history", [])),
            "best_attempt": result.get("best_attempt", {}),
            "draft_markdown": (result.get("best_attempt") or {}).get("draft_markdown"),
            "final_report": (result.get("best_attempt") or {}).get("final_report"),
            "claim_verification": result.get("claim_verification", {}),
            "skill_trace": list(result.get("skill_trace", [])),
            "total_attempts": int(result.get("total_attempts", 0)),
            "reflections_stored": int(result.get("reflections_stored", 0)),
            "paradigm": "reflexion",
            "summary": result.get("summary")
            or (
                f"Reflexion completed after {int(result.get('total_attempts', 0))} attempts, "
                f"passed={bool(result.get('review_passed'))}, "
                f"best_confidence={float((result.get('best_attempt') or {}).get('confidence', 0.0)):.2f}"
            ),
            "reviewer_warnings": list(result.get("warnings", [])),
        }

    # ── Actor ─────────────────────────────────────────────────────────────

    def _actor_review(
        self,
        brief: dict,
        paper_cards: list,
        draft_report: Any | None,
        rag_result: dict[str, Any] | None,
        prior_reflections: list[SelfReflection],
        attempt: int,
    ) -> dict[str, Any]:
        """Actor：基于当前上下文（含历史反思）执行审查。"""
        from src.research.graph.nodes.review import review_node

        # 将 prior_reflections 注入上下文
        reflection_context = ""
        if prior_reflections:
            reflection_lines = ["## 历史失败经验（供参考）："]
            for r in prior_reflections[-3:]:
                reflection_lines.append(r.to_prompt())
            reflection_context = "\n".join(reflection_lines)

        # 构建带反思上下文的 report_draft
        enhanced_draft = draft_report
        if reflection_context and hasattr(draft_report, "sections"):
            # 模拟：给 draft 附加反思上下文（实际场景中会注入到审查 prompt）
            pass

        try:
            review_result = review_node(
                {
                    "task_id": self.task_id or "",
                    "workspace_id": self.workspace_id or "",
                    "brief": brief,
                    "rag_result": rag_result or {},
                    "paper_cards": paper_cards,
                    "draft_report": draft_report,
                }
            )
            feedback = self._serialize_feedback(review_result.get("review_feedback"))
            claim_verification = review_result.get("claim_verification", {})
            grounding_stats = claim_verification.get("grounding_stats", {})
            supported_ratio = float(grounding_stats.get("supported_ratio", 0.0) or 0.0)
            issue_count = len(feedback.get("issues", [])) if isinstance(feedback, dict) else 0
            confidence = max(0.1, min(0.95, 0.35 + supported_ratio * 0.5 - issue_count * 0.05))
            if isinstance(feedback, dict) and feedback.get("passed") is True:
                confidence = max(confidence, 0.85)
            return {
                "feedback": feedback,
                "confidence": confidence,
                "reflection_context_used": bool(reflection_context),
                "draft_markdown": review_result.get("draft_markdown"),
                "final_report": self._serialize_feedback(review_result.get("final_report")),
                "claim_verification": claim_verification,
                "skill_trace": list(review_result.get("skill_trace", [])),
            }
        except Exception as exc:
            logger.warning("[ReviewerAgent] Actor review failed: %s", exc)
            return {"confidence": 0.0, "error": str(exc)}

    # ── Evaluator ──────────────────────────────────────────────────────────

    def _evaluate(self, actor_result: dict, attempt: int) -> dict:
        """Evaluator：判断审查是否达标。"""
        confidence = actor_result.get("confidence", 0.0)
        feedback = actor_result.get("feedback")

        passed = False
        reason = ""
        task_type = "review_coverage"
        issues: list[str] = []

        if feedback:
            if isinstance(feedback, dict):
                passed = bool(feedback.get("passed", False))
                raw_issues = feedback.get("issues", [])
                if isinstance(raw_issues, list):
                    issues = [
                        item.get("summary", str(item))
                        if isinstance(item, dict)
                        else str(item)
                        for item in raw_issues
                    ]
                reason = str(feedback.get("summary", "") or "")
            elif hasattr(feedback, "passed"):
                passed = feedback.passed
            if hasattr(feedback, "issues"):
                issues = [str(i) for i in (feedback.issues or [])]
            if hasattr(feedback, "summary"):
                reason = feedback.summary

        if confidence < self.REFLECTION_THRESHOLD:
            passed = False
            reason = f"Confidence {confidence:.2f} below threshold {self.REFLECTION_THRESHOLD}"
            task_type = "review_confidence"

        return {
            "passed": passed,
            "confidence": confidence,
            "reason": reason,
            "task_type": task_type,
            "issues": issues,
        }

    # ── Self-Reflector ────────────────────────────────────────────────────

    def _self_reflect(
        self,
        task_type: str,
        failure_context: str,
        failure_details: str,
    ) -> SelfReflection | None:
        """
        Self-Reflector：失败时生成深度自我反思。

        这是 Reflexion 模式的核心：不是简单重试，
        而是 LLM 生成结构化的失败分析，存入运行期 memory adapter。
        """
        import time

        from src.agent.llm import build_reason_llm
        from src.agent.settings import get_settings
        from langchain_core.messages import HumanMessage, SystemMessage

        settings = get_settings()
        llm = build_reason_llm(settings, max_tokens=4096)

        prompt = SELF_REFLECTION_PROMPT.format(
            task_type=task_type,
            failure_context=failure_context[:500],
            failure_details=failure_details[:1000],
        )

        try:
            resp = llm.invoke([
                SystemMessage(content="You are a self-reflection expert. Analyze the failure case and return JSON only."),
                HumanMessage(content=prompt),
            ])
            raw = resp.content if hasattr(resp, "content") else str(resp)
            data = self._parse_json(raw)
        except Exception as exc:
            logger.warning("[ReviewerAgent] Self-reflection failed: %s", exc)
            return None

        if not data:
            return None

        reflection_id = f"refl_{int(time.time() * 1000)}"
        reflection = SelfReflection(
            reflection_id=reflection_id,
            task_type=task_type,
            failure_context=failure_context[:500],
            root_cause=data.get("root_cause", "Unknown"),
            lessons=data.get("lessons", []),
            improved_strategy=data.get("improved_strategy", ""),
            confidence_gain=0.1,
            created_at=time.time(),
        )
        self._reflections.append(reflection)
        return reflection

    # ── Memory Operations ─────────────────────────────────────────────────

    def _retrieve_reflections(self, brief: dict, draft_report: Any | None) -> list[SelfReflection]:
        """
        Runtime memory retrieval: fetch prior failure lessons from the runtime cache.

        Reflexion 的关键：不是盲目重试，而是从历史失败中学习。
        """
        if not self.mm:
            return []

        topic = brief.get("research_topic") or brief.get("topic", "") if brief else ""
        reflections: list[SelfReflection] = []

        try:
            # Runtime vector cache: retrieve related failure lessons.
            semantic_entries = self.mm.search_semantic(
                query=f"审查失败经验 {topic}" if topic else "审查失败经验",
                top_k=3,
                memory_type="reflection",
            )
            for entry in semantic_entries:
                text = entry.text
                # 从 text 中解析出 reflection_id（格式：refl_xxx）
                import re

                match = re.search(r"refl_\d+", text)
                refl_id = match.group(0) if match else ""
                reflections.append(SelfReflection(
                    reflection_id=refl_id,
                    task_type="retrieved",
                    failure_context=text,
                    root_cause="（从 memory 检索）",
                    lessons=[],
                    improved_strategy="",
                    confidence_gain=0.0,
                    created_at=entry.created_at,
                ))
        except Exception as exc:
            logger.warning("[ReviewerAgent] Memory retrieval failed: %s", exc)

        logger.info("[ReviewerAgent] Retrieved %d prior reflections", len(reflections))
        return reflections

    def _store_reflection(self, reflection: SelfReflection) -> None:
        """Store a self-reflection in the runtime memory adapter."""
        if not self.mm:
            return

        try:
            # Runtime vector cache: store reflection text for retrieval.
            self.mm.add_semantic(
                f"[Reflexion] {reflection.task_type}: {reflection.root_cause} "
                f"→ {reflection.improved_strategy}",
                memory_type="reflection",
                metadata={
                    "source": "reviewer_agent",
                    "reflection_id": reflection.reflection_id,
                    "confidence_gain": reflection.confidence_gain,
                    "workspace_id": self.workspace_id,
                },
            )

            # Runtime preference cache: store improved strategy hint.
            if reflection.improved_strategy:
                self.mm.set_preference(
                    f"review_strategy_{reflection.task_type}",
                    reflection.improved_strategy[:200],
                )

            logger.info(
                "[ReviewerAgent] Stored reflection %s: root_cause=%s",
                reflection.reflection_id,
                reflection.root_cause[:100],
            )
        except Exception as exc:
            logger.warning("[ReviewerAgent] Failed to store reflection: %s", exc)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _build_feedback(self, attempt_history: list[dict]) -> dict:
        """汇总所有尝试，生成最终 feedback。"""
        passed = any(h["passed"] for h in attempt_history)
        total_issues = sum(len(h.get("issues", [])) for h in attempt_history)
        best_confidence = max((h["confidence"] for h in attempt_history), default=0.0)

        return {
            "passed": passed,
            "total_attempts": len(attempt_history),
            "total_issues": total_issues,
            "best_confidence": best_confidence,
            "attempt_history": attempt_history,
            "best_attempt": max(attempt_history, key=lambda item: item.get("confidence", 0.0)) if attempt_history else {},
            "issues": (max(attempt_history, key=lambda item: item.get("confidence", 0.0)).get("issues_detail", []) if attempt_history else []),
            "grounding_stats": (max(attempt_history, key=lambda item: item.get("confidence", 0.0)).get("grounding_stats", {}) if attempt_history else {}),
            "claim_verification": (max(attempt_history, key=lambda item: item.get("confidence", 0.0)).get("claim_verification", {}) if attempt_history else {}),
            "skill_trace": (max(attempt_history, key=lambda item: item.get("confidence", 0.0)).get("skill_trace", []) if attempt_history else []),
            "reflexion_summary": (
                f"{len(attempt_history)} attempts, "
                f"passed={passed}, "
                f"best_confidence={best_confidence:.2f}"
            ),
        }

    def _serialize_feedback(self, feedback: Any) -> dict[str, Any] | None:
        if feedback is None:
            return None
        if isinstance(feedback, dict):
            return dict(feedback)
        if hasattr(feedback, "model_dump"):
            return feedback.model_dump(mode="json")
        return {"summary": str(feedback)}

    def _serialize_reflection(self, reflection: SelfReflection) -> dict[str, Any]:
        return asdict(reflection)

    def _resolve_feedback(self, feedback: Any) -> Any:
        if not isawaitable(feedback):
            return feedback

        def _runner() -> Any:
            return asyncio.run(feedback)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return _runner()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_runner).result()

    def _restore_reflections(self, reflections: list[Any]) -> list[SelfReflection]:
        restored: list[SelfReflection] = []
        for item in reflections:
            if isinstance(item, SelfReflection):
                restored.append(item)
            elif isinstance(item, dict):
                try:
                    restored.append(SelfReflection(**item))
                except TypeError:
                    logger.warning("[ReviewerAgent] skipped malformed reflection payload")
        return restored

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
        workflow = StateGraph(ReviewerGraphState)
        workflow.add_node("retrieve_memory", self._retrieve_memory_node)
        workflow.add_node("actor_review", self._actor_review_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("self_reflect", self._self_reflect_node)
        workflow.add_node("finalize", self._finalize_node)
        workflow.add_edge(START, "retrieve_memory")
        workflow.add_edge("retrieve_memory", "actor_review")
        workflow.add_edge("actor_review", "evaluate")
        workflow.add_conditional_edges(
            "evaluate",
            self._route_after_evaluate,
            {
                "self_reflect": "self_reflect",
                "finalize": "finalize",
            },
        )
        workflow.add_edge("self_reflect", "actor_review")
        workflow.add_edge("finalize", END)
        return workflow.compile(checkpointer=get_langgraph_checkpointer("reviewer_agent"))

    def _retrieve_memory_node(self, state: "ReviewerGraphState") -> dict[str, Any]:
        prior_reflections = self._retrieve_reflections(
            state.get("brief") or {},
            state.get("draft_report"),
        )
        return {"prior_reflections": [self._serialize_reflection(item) for item in prior_reflections]}

    def _actor_review_node(self, state: "ReviewerGraphState") -> dict[str, Any]:
        attempt = int(state.get("attempt", 0)) + 1
        logger.info("[ReviewerAgent] Attempt %d/%d", attempt, self.MAX_ATTEMPTS)
        actor_result = self._actor_review(
            brief=state.get("brief") or {},
            paper_cards=list(state.get("paper_cards", [])),
            draft_report=state.get("draft_report"),
            rag_result=state.get("rag_result") or {},
            prior_reflections=self._restore_reflections(list(state.get("prior_reflections", []))),
            attempt=attempt,
        )
        return {"attempt": attempt, "actor_result": actor_result}

    def _evaluate_node(self, state: "ReviewerGraphState") -> dict[str, Any]:
        attempt = int(state.get("attempt", 0))
        actor_result = state.get("actor_result") or {}
        eval_result = self._evaluate(actor_result, attempt)
        attempt_history = list(state.get("attempt_history", []))
        attempt_history.append(
            {
                "attempt": attempt,
                "passed": eval_result["passed"],
                "confidence": actor_result.get("confidence", 0.0),
                "issues": eval_result.get("issues", []),
                "issues_detail": (actor_result.get("feedback", {}) or {}).get("issues", []) if isinstance(actor_result.get("feedback"), dict) else [],
                "grounding_stats": (actor_result.get("claim_verification", {}) or {}).get("grounding_stats", {}),
                "claim_verification": actor_result.get("claim_verification", {}),
                "skill_trace": actor_result.get("skill_trace", []),
                "draft_markdown": actor_result.get("draft_markdown"),
                "final_report": actor_result.get("final_report"),
            }
        )
        if self.mm:
            self.mm.add_sensory(
                "review_attempt",
                {
                    "attempt": attempt,
                    "passed": eval_result["passed"],
                    "confidence": actor_result.get("confidence", 0.0),
                },
            )
        return {
            "eval_result": eval_result,
            "attempt_history": attempt_history,
            "review_passed": eval_result["passed"],
        }

    def _route_after_evaluate(self, state: "ReviewerGraphState") -> str:
        eval_result = state.get("eval_result") or {}
        if eval_result.get("passed"):
            logger.info("[ReviewerAgent] Attempt %d PASSED", int(state.get("attempt", 0)))
            return "finalize"
        if int(state.get("attempt", 0)) >= self.MAX_ATTEMPTS:
            return "finalize"
        logger.info(
            "[ReviewerAgent] Attempt %d FAILED: %s",
            int(state.get("attempt", 0)),
            eval_result.get("reason", ""),
        )
        return "self_reflect"

    def _self_reflect_node(self, state: "ReviewerGraphState") -> dict[str, Any]:
        eval_result = state.get("eval_result") or {}
        actor_result = state.get("actor_result") or {}
        prior_reflections = list(state.get("prior_reflections", []))
        reflection = self._self_reflect(
            task_type=eval_result.get("task_type", "review"),
            failure_context=str(eval_result.get("issues", [])),
            failure_details=str(actor_result),
        )
        if reflection:
            self._store_reflection(reflection)
            prior_reflections.append(self._serialize_reflection(reflection))
        return {"prior_reflections": prior_reflections}

    def _finalize_node(self, state: "ReviewerGraphState") -> dict[str, Any]:
        attempt_history = list(state.get("attempt_history", []))
        passed = any(item.get("passed") for item in attempt_history)
        best_attempt = max(attempt_history, key=lambda item: item.get("confidence", 0.0)) if attempt_history else {}
        return {
            "review_passed": passed,
            "review_feedback": self._build_feedback(attempt_history),
            "best_attempt": best_attempt,
            "claim_verification": best_attempt.get("claim_verification", {}),
            "skill_trace": best_attempt.get("skill_trace", []),
            "total_attempts": len(attempt_history),
            "reflections_stored": len([item for item in attempt_history if not item.get("passed")]),
            "summary": (
                f"Reflexion completed after {len(attempt_history)} attempts, "
                f"passed={passed}, best_confidence={best_attempt.get('confidence', 0.0):.2f}"
            ),
        }


class ReviewerGraphState(TypedDict, total=False):
    brief: dict[str, Any]
    paper_cards: list[Any]
    draft_report: Any
    rag_result: dict[str, Any]
    warnings: list[str]
    attempt: int
    actor_result: dict[str, Any]
    eval_result: dict[str, Any]
    attempt_history: list[dict[str, Any]]
    prior_reflections: list[dict[str, Any]]
    review_passed: bool
    review_feedback: dict[str, Any]
    best_attempt: dict[str, Any]
    total_attempts: int
    reflections_stored: int
    summary: str


# ─── 入口函数 ───────────────────────────────────────────────────────────────


def run_reviewer_agent(state: dict, inputs: dict) -> dict:
    """ReviewerAgent 入口（兼容 supervisor 格式）。"""
    workspace_id = inputs.get("workspace_id") or state.get("workspace_id")
    task_id = inputs.get("task_id") or state.get("task_id")
    brief = state.get("brief") or {}
    paper_cards = state.get("paper_cards") or []
    draft_report = state.get("draft_report")
    rag_result = state.get("rag_result")
    emitter = inputs.get("_event_emitter")

    agent = ReviewerAgent(workspace_id=workspace_id, task_id=task_id)
    try:
        if emitter:
            emitter.on_thinking("review", "Reviewer agent is checking grounding and revision requirements.")
        result = agent.run(
            brief=brief,
            paper_cards=paper_cards,
            draft_report=draft_report,
            rag_result=rag_result,
        )
        if emitter:
            passed = result.get("review_passed")
            emitter.on_thinking("review", f"Review stage completed with review_passed={passed}.")
        return result
    except Exception as exc:
        logger.exception("[ReviewerAgent] run failed: %s", exc)
        return {
            "review_passed": False,
            "review_feedback": None,
            "paradigm": "reflexion",
            "reviewer_warnings": [f"ReviewerAgent failed: {exc}"],
        }
