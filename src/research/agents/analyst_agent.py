"""AnalystAgent — Reason-via-Artifacts (RVA) 模式。

设计模式说明：
- Reasoning-via-Artifacts：LLM 不是直接在上下文中推理，
  而是先构建中间"工件"（artifact），再基于这些工件做推理。
- 核心思想：
  1. LLM 生成 artifact（comparison_matrix, outline, structured_cards）
  2. 这些 artifact 成为后续推理的"外部化认知"
  3. 最终报告基于 artifact 组合生成，而非端到端生成
- 适用场景：复杂分析报告（如科研综述），需要多维度结构化推理

与 ReAct 的区别：
- ReAct：工具是外部服务（搜索、计算），LLM 在工具结果上继续推理
- RVA：工具生成的是"内部工件"，LLM 基于工件组合做推理
- 本质区别：ReAct 是"推理+行动"交替，RVA 是"构建+组合"

Artifact 层级：
  L0: raw paper_cards（原始论文元数据）
  L1: structured_cards（结构化抽取后的论文信息）
  L2: comparison_matrix（对比矩阵：方法/数据集/基准）
  L3: outline（报告大纲：章节结构）
  L4: report_draft（最终报告草稿）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from typing import TypedDict

from langgraph.graph import START, StateGraph
from src.agent.checkpointing import build_graph_config, get_langgraph_checkpointer
from src.memory.manager import get_memory_manager
from src.models.report import DraftReport

logger = logging.getLogger(__name__)


# ─── Artifact 定义 ──────────────────────────────────────────────────────────


@dataclass
class Artifact:
    """中间推理工件。"""

    level: int          # L0-L4
    artifact_type: str  # structured_card | comparison_matrix | outline | report_draft
    content: Any        # 工件内容
    confidence: float   # 工件置信度 0-1
    dependencies: list[str]  # 依赖的其他 artifact ID
    created_by: str     # "llm" | "skill" | "user"

    def is_ready(self, artifacts: list[Artifact]) -> bool:
        """检查依赖是否都存在。"""
        for dep_id in self.dependencies:
            if not any(a.artifact_type == dep_id or f"artifact_{dep_id}" == dep_id for a in artifacts):
                return False
        return True


@dataclass
class ReasoningState:
    """RVA 的推理状态：当前工件栈 + 置信度追踪。"""

    artifacts: list[Artifact] = field(default_factory=list)
    current_level: int = 0
    overall_confidence: float = 0.0

    def add(self, artifact: Artifact) -> None:
        self.artifacts.append(artifact)
        self.current_level = max(self.current_level, artifact.level)
        self._recompute_confidence()

    def _recompute_confidence(self) -> None:
        if not self.artifacts:
            self.overall_confidence = 0.0
            return
        weights = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.2, 4: 0.2}
        total_weight = sum(weights.get(a.level, 0) for a in self.artifacts)
        weighted = sum(a.confidence * weights.get(a.level, 0) for a in self.artifacts)
        self.overall_confidence = weighted / total_weight if total_weight > 0 else 0.0

    def get_by_level(self, level: int) -> list[Artifact]:
        return [a for a in self.artifacts if a.level == level]

    def get_by_type(self, artifact_type: str) -> list[Artifact]:
        return [a for a in self.artifacts if a.artifact_type == artifact_type]


# ─── AnalystAgent ───────────────────────────────────────────────────────────


class AnalystAgent:
    """
    Reason-via-Artifacts 模式的 Analyst Agent。

    工作流程：
      ┌──────────────┐
      │ BUILD ARTIFACTS │  构建多层工件（L1 structured_cards → L2 matrix → L3 outline）
      └──────┬───────┘
             │ artifacts
             ▼
      ┌──────────────┐
      │ REASON OVER    │  基于工件组合做推理（组合多个低层 artifact 生成高层 artifact）
      │ ARTIFACTS      │  例如：structured_cards + comparison_matrix → outline
      └──────┬───────┘
             │ reasoning_result
             ▼
      ┌──────────────┐
      │ VERIFY & REFINE│  验证当前工件置信度，若不足则继续构建
      └───────┬───────┘
              │ artifact_stack (L0-L4)
    """

    def __init__(self, workspace_id: str | None = None, task_id: str | None = None):
        self.workspace_id = workspace_id
        self.task_id = task_id
        self.mm = get_memory_manager(workspace_id) if workspace_id else None

    def run(self, brief: dict, paper_cards: list) -> dict[str, Any]:
        """
        完整 RVA Pipeline。

        与 ReAct/Plan-and-Execute 的关键区别：
        - 不依赖迭代循环，而是依赖工件依赖图（DAG）驱动
        - 每次生成的是"artifact"而非"action"
        - 置信度是主要驱动力：confidence 不足则继续构建更低层的 artifact
        """
        logger.info("[AnalystAgent] Starting Reason-via-Artifacts pipeline via LangGraph")
        result = self.build_graph().invoke(
            {
                "brief": brief,
                "paper_cards": paper_cards,
                "warnings": [],
            },
            config=build_graph_config("analyst_agent"),
        )
        artifacts = list(result.get("artifacts", []))
        return {
            "structured_cards": result.get("structured_cards"),
            "comparison_matrix": result.get("comparison_matrix"),
            "outline": result.get("outline"),
            "report_draft": result.get("report_draft"),
            "draft_report": result.get("draft_report"),
            "draft_markdown": result.get("draft_markdown"),
            "artifacts": artifacts,
            "overall_confidence": result.get("overall_confidence", 0.0),
            "verification": result.get("verification", {}),
            "paradigm": "reasoning_via_artifacts",
            "summary": (
                f"RVA pipeline 完成：{len(artifacts)} 个工件，"
                f"overall_confidence={float(result.get('overall_confidence', 0.0)):.2f}"
            ),
            "analyst_warnings": list(result.get("warnings", [])),
        }

    # ── L1: Structured Cards ──────────────────────────────────────────────

    def _build_structured_cards(self, raw_cards: list) -> dict:
        from src.skills.research_skills import comparison_matrix_builder

        if not raw_cards:
            return {"error": "No raw cards", "cards": []}

        cards = []
        for card in raw_cards[:20]:  # 限制数量
            if isinstance(card, dict):
                cards.append({
                    "title": card.get("title", ""),
                    "summary": card.get("summary", card.get("abstract", "")),
                    "methods": card.get("methods", []),
                    "datasets": card.get("datasets", []),
                })

        result = comparison_matrix_builder(
            {"paper_cards": cards, "compare_dimensions": ["methods", "datasets"], "format": "json"},
            {"workspace_id": self.workspace_id} if self.workspace_id else {},
        )
        if "error" in result:
            return {"error": result["error"], "cards": cards, "confidence": 0.4}

        # 从 matrix 中提取 structured info
        matrix = result.get("matrix", {})
        structured = []
        for row in (matrix.get("rows", []) if isinstance(matrix, dict) else []):
            paper = row.get("paper", "")
            structured.append({
                "title": paper,
                "methods": row.get("methods", ""),
                "datasets": row.get("datasets", ""),
            })

        return {"cards": structured, "confidence": 0.7}

    # ── L2: Comparison Matrix ──────────────────────────────────────────────

    def _build_comparison_matrix(self, cards_artifacts: list[Artifact]) -> dict:
        from src.skills.research_skills import comparison_matrix_builder

        if not cards_artifacts:
            return {"error": "No structured cards", "matrix": {}, "confidence": 0.0}

        cards = cards_artifacts[0].content
        if not cards:
            return {"error": "Empty cards", "matrix": {}, "confidence": 0.0}

        result = comparison_matrix_builder(
            {
                "paper_cards": cards,
                "compare_dimensions": ["methods", "datasets", "benchmarks", "limitations"],
                "format": "json",
            },
            {"workspace_id": self.workspace_id} if self.workspace_id else {},
        )
        if "error" in result:
            return {"error": result["error"], "matrix": {}, "confidence": 0.3}

        matrix = result.get("matrix", {})
        row_count = len(matrix.get("rows", [])) if isinstance(matrix, dict) else 0
        confidence = min(0.9, 0.4 + 0.1 * row_count)

        return {"matrix": matrix, "confidence": confidence}

    # ── L3: Outline ───────────────────────────────────────────────────────

    def _build_outline(self, state: ReasoningState, brief: dict[str, Any] | None = None) -> dict:
        cards_artifact = state.get_by_type("structured_cards")
        matrix_artifact = state.get_by_type("comparison_matrix")

        topic = ""
        cards = []
        matrix = {}

        if brief:
            topic = str(brief.get("research_topic") or brief.get("topic") or "").strip()

        if cards_artifact:
            cards = cards_artifact[0].content or []
            if not topic and cards and isinstance(cards[0], dict):
                topic = str(cards[0].get("title", "") or "").strip()

        if matrix_artifact:
            matrix = matrix_artifact[0].content or {}

        if not topic:
            topic = "Research Topic"

        method_items: list[str] = []
        dataset_items: list[str] = []
        if isinstance(matrix, dict):
            for row in matrix.get("rows", [])[:8]:
                if not isinstance(row, dict):
                    continue
                paper = str(row.get("paper", "")).strip()
                methods = str(row.get("methods", "")).strip()
                datasets = str(row.get("datasets", "")).strip()
                if paper and methods:
                    method_items.append(f"{paper}: {methods}")
                if paper and datasets:
                    dataset_items.append(f"{paper}: {datasets}")

        if not method_items:
            for card in cards[:6]:
                if not isinstance(card, dict):
                    continue
                title = str(card.get("title", "")).strip()
                methods = card.get("methods", [])
                if title and methods:
                    method_items.append(f"{title}: {methods}")

        outline = {
            "title": f"{topic} Survey",
            "abstract": "Summarize the research problem, scope, method families, benchmarks, and main findings.",
            "introduction": [
                f"Define the problem background and motivation for {topic}",
                "Explain the survey scope, problem setting, and contribution",
            ],
            "background": [
                "Introduce the core concepts, task definition, and typical system setup",
                "Explain why the problem matters in real applications",
            ],
            "methods_review": method_items[:6] or ["Summarize the main method families and representative papers"],
            "datasets_and_benchmarks": dataset_items[:6] or ["Review the main datasets, evaluation tasks, and metrics"],
            "challenges_and_limitations": [
                "Analyze coverage, generalization, robustness, and engineering cost",
                "Discuss evidence gaps, reproducibility concerns, and deployment constraints",
            ],
            "future_directions": [
                "Summarize open problems and feasible next directions",
                "Highlight evaluation standardization and real-world deployment",
            ],
            "conclusion": "Summarize the main findings and practical takeaways.",
        }
        confidence = (cards_artifact[0].confidence + matrix_artifact[0].confidence) / 2 if (cards_artifact and matrix_artifact) else 0.5

        return {"outline": outline, "confidence": min(0.9, confidence)}

    # ── L4: Report Draft ──────────────────────────────────────────────────

    def _build_report_draft(self, state: ReasoningState) -> dict:
        from src.agent.llm import build_reason_llm
        from src.agent.settings import get_settings
        from langchain_core.messages import HumanMessage, SystemMessage

        outline_artifact = state.get_by_type("outline")
        matrix_artifact = state.get_by_type("comparison_matrix")

        if not outline_artifact:
            return {"error": "No outline", "draft": {}, "confidence": 0.0}

        outline = outline_artifact[0].content or {}
        matrix = matrix_artifact[0].content or {} if matrix_artifact else {}

        settings = get_settings()
        llm = build_reason_llm(settings, max_tokens=16384)

        outline_text = self._format_outline(outline)
        matrix_text = self._format_matrix(matrix)

        SYSTEM = """You are a scientific survey drafting expert. Given a report outline and a comparison matrix, generate the complete report in JSON.

Return strict JSON:
{
  "title": "English report title",
  "sections": {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "methods_review": "Methods review",
    "datasets_and_benchmarks": "Datasets and benchmarks",
    "challenges": "Challenges and limitations",
    "conclusion": "Conclusion"
  }
}"""

        user_prompt = f"""## Report Outline

{outline_text}

## Comparison Matrix

{matrix_text}

Generate the full report JSON from the information above.
"""

        try:
            resp = llm.invoke([
                SystemMessage(content=SYSTEM),
                HumanMessage(content=user_prompt),
            ])
            raw = resp.content if hasattr(resp, "content") else str(resp)
            draft = self._parse_json(raw)
        except Exception as exc:
            logger.warning("[AnalystAgent] L4 draft failed: %s", exc)
            return {"error": str(exc), "draft": {}, "confidence": 0.3}

        outline_conf = outline_artifact[0].confidence
        matrix_conf = matrix_artifact[0].confidence if matrix_artifact else 0.5
        confidence = (outline_conf + matrix_conf) / 2 * 0.8  # 乘 0.8 因为是端到端生成

        return {"draft": draft or {}, "confidence": confidence}

    # ── L5: Confidence Verification ─────────────────────────────────────

    def _verify_confidence(self, state: ReasoningState) -> dict:
        """验证工件栈的整体置信度。"""
        overall = state.overall_confidence
        warnings = []
        actions = []

        if overall < 0.3:
            warnings.append("Overall confidence too low")
            actions.append("需要更多高质量 paper_cards")
        if not state.get_by_type("comparison_matrix"):
            warnings.append("Missing comparison_matrix")
            actions.append("需要构建 comparison_matrix")
        if not state.get_by_type("outline"):
            warnings.append("Missing outline")
            actions.append("需要生成 outline")

        return {
            "overall_confidence": round(overall, 3),
            "warnings": warnings,
            "actions": actions,
            "needs_refinement": overall < 0.5,
        }

    # ── Helpers ─────────────────────────────────────────────────────────

    def _format_outline(self, outline: Any) -> str:
        if isinstance(outline, dict):
            lines = ["## Report Outline"]
            for key, val in outline.items():
                if isinstance(val, list):
                    lines.append(f"### {key}")
                    for item in val:
                        lines.append(f"- {item}")
                elif isinstance(val, str):
                    lines.append(f"### {key}: {val}")
            return "\n".join(lines)
        return str(outline)

    def _format_matrix(self, matrix: Any) -> str:
        if not matrix:
            return "(No comparison matrix)"
        if isinstance(matrix, dict) and "rows" in matrix:
            rows = matrix["rows"]
            lines = ["## Comparison Matrix"]
            for row in rows[:10]:
                lines.append(f"- **{row.get('paper', '')}**: {row.get('methods', '')}")
            return "\n".join(lines)
        return str(matrix)

    def _needs_grounded_redraft(self, draft_report: DraftReport | None) -> bool:
        if draft_report is None:
            return True
        sections = draft_report.sections or {}
        if len(sections) < 8:
            return True
        body = "\n".join(str(value) for value in sections.values())
        if len(body) < 4000:
            return True
        lowered = body.lower()
        if "research topic" in lowered or "[research topic]" in lowered:
            return True
        return False

    def _store_artifacts_memory(self, state: ReasoningState) -> None:
        if not self.mm:
            return
        try:
            for artifact in state.artifacts:
                if artifact.level >= 2:  # 只存 L2 及以上的 artifact
                    self.mm.add_semantic(
                        f"Artifact [{artifact.artifact_type}] confidence={artifact.confidence}",
                        memory_type="research_fact",
                        metadata={
                            "source": "analyst_agent",
                            "level": artifact.level,
                            "workspace_id": self.workspace_id,
                        },
                    )
        except Exception as exc:
            logger.warning("[AnalystAgent] Failed to store artifacts memory: %s", exc)

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
        workflow = StateGraph(AnalystGraphState)
        workflow.add_node("seed_reasoning_state", self._seed_reasoning_state)
        workflow.add_node("build_structured_cards", self._structured_cards_node)
        workflow.add_node("build_comparison_matrix", self._comparison_matrix_node)
        workflow.add_node("build_outline", self._outline_node)
        workflow.add_node("build_report_draft", self._report_draft_node)
        workflow.add_node("verify_and_finalize", self._verify_and_finalize_node)
        workflow.add_edge(START, "seed_reasoning_state")
        workflow.add_edge("seed_reasoning_state", "build_structured_cards")
        workflow.add_edge("build_structured_cards", "build_comparison_matrix")
        workflow.add_edge("build_comparison_matrix", "build_outline")
        workflow.add_edge("build_outline", "build_report_draft")
        workflow.add_edge("build_report_draft", "verify_and_finalize")
        return workflow.compile(checkpointer=get_langgraph_checkpointer("analyst_agent"))

    def _seed_reasoning_state(self, state: "AnalystGraphState") -> dict[str, Any]:
        return {"reasoning_state": ReasoningState()}

    def _structured_cards_node(self, state: "AnalystGraphState") -> dict[str, Any]:
        reasoning_state = state.get("reasoning_state") or ReasoningState()
        result = self._build_structured_cards(state.get("paper_cards", []))
        warnings = list(state.get("warnings", []))
        if "error" not in result:
            reasoning_state.add(
                Artifact(
                    level=1,
                    artifact_type="structured_cards",
                    content=result.get("cards", []),
                    confidence=result.get("confidence", 0.7),
                    dependencies=["raw_cards"],
                    created_by="skill",
                )
            )
        else:
            warnings.append(str(result["error"]))
        return {"reasoning_state": reasoning_state, "warnings": warnings}

    def _comparison_matrix_node(self, state: "AnalystGraphState") -> dict[str, Any]:
        reasoning_state = state.get("reasoning_state") or ReasoningState()
        result = self._build_comparison_matrix(reasoning_state.get_by_type("structured_cards"))
        warnings = list(state.get("warnings", []))
        if "error" not in result:
            reasoning_state.add(
                Artifact(
                    level=2,
                    artifact_type="comparison_matrix",
                    content=result.get("matrix", {}),
                    confidence=result.get("confidence", 0.6),
                    dependencies=["structured_cards"],
                    created_by="skill",
                )
            )
        else:
            warnings.append(str(result["error"]))
        return {"reasoning_state": reasoning_state, "warnings": warnings}

    def _outline_node(self, state: "AnalystGraphState") -> dict[str, Any]:
        reasoning_state = state.get("reasoning_state") or ReasoningState()
        result = self._build_outline(reasoning_state, state.get("brief") or {})
        warnings = list(state.get("warnings", []))
        if "error" not in result:
            reasoning_state.add(
                Artifact(
                    level=3,
                    artifact_type="outline",
                    content=result.get("outline", {}),
                    confidence=result.get("confidence", 0.6),
                    dependencies=["structured_cards", "comparison_matrix"],
                    created_by="llm",
                )
            )
        else:
            warnings.append(str(result["error"]))
        return {"reasoning_state": reasoning_state, "warnings": warnings}

    def _report_draft_node(self, state: "AnalystGraphState") -> dict[str, Any]:
        from src.research.graph.nodes.draft import (
            _build_draft_report as _grounded_build_draft_report,
            build_drafting_skill_artifacts,
        )

        reasoning_state = state.get("reasoning_state") or ReasoningState()
        warnings = list(state.get("warnings", []))
        skill_artifacts = build_drafting_skill_artifacts(
            list(state.get("paper_cards", [])),
            state.get("brief"),
            workspace_id=self.workspace_id,
            task_id=self.task_id,
        )
        try:
            grounded_report = _grounded_build_draft_report(
                list(state.get("paper_cards", [])),
                state.get("brief"),
                skill_artifacts=skill_artifacts,
            )
            reasoning_state.add(
                Artifact(
                    level=4,
                    artifact_type="report_draft",
                    content=grounded_report.model_dump(mode="json"),
                    confidence=0.85,
                    dependencies=["structured_cards", "comparison_matrix", "outline"],
                    created_by="llm",
                )
            )
        except Exception as exc:
            result = self._build_report_draft(reasoning_state)
            if "error" not in result:
                reasoning_state.add(
                    Artifact(
                        level=4,
                        artifact_type="report_draft",
                        content=result.get("draft", {}),
                        confidence=result.get("confidence", 0.5),
                        dependencies=["structured_cards", "comparison_matrix", "outline"],
                        created_by="llm",
                    )
                )
            else:
                warnings.append(str(exc))
                warnings.append(str(result["error"]))
        return {
            "reasoning_state": reasoning_state,
            "warnings": warnings,
            "drafting_skill_artifacts": skill_artifacts,
        }

    def _verify_and_finalize_node(self, state: "AnalystGraphState") -> dict[str, Any]:
        from src.research.graph.nodes.draft import _build_markdown
        from src.research.graph.nodes.draft import (
            _build_draft_report as _grounded_build_draft_report,
            build_drafting_skill_artifacts,
        )

        reasoning_state = state.get("reasoning_state") or ReasoningState()
        verification = self._verify_confidence(reasoning_state)
        if self.mm:
            self._store_artifacts_memory(reasoning_state)

        skill_artifacts = state.get("drafting_skill_artifacts") or build_drafting_skill_artifacts(
            list(state.get("paper_cards", [])),
            state.get("brief"),
            workspace_id=self.workspace_id,
            task_id=self.task_id,
        )

        structured_cards = None
        comparison_matrix = None
        outline = None
        report_draft = None
        if reasoning_state.get_by_type("structured_cards"):
            structured_cards = reasoning_state.get_by_type("structured_cards")[0].content
        if reasoning_state.get_by_type("comparison_matrix"):
            comparison_matrix = reasoning_state.get_by_type("comparison_matrix")[0].content
        if reasoning_state.get_by_type("outline"):
            outline = reasoning_state.get_by_type("outline")[0].content
        if reasoning_state.get_by_type("report_draft"):
            report_draft = reasoning_state.get_by_type("report_draft")[0].content

        if comparison_matrix is None:
            comparison_matrix = skill_artifacts.get("comparison_matrix")
        if outline is None:
            outline = skill_artifacts.get("writing_outline")

        draft_report = self._coerce_draft_report(report_draft)
        if self._needs_grounded_redraft(draft_report):
            regenerated = _grounded_build_draft_report(
                list(state.get("paper_cards", [])),
                state.get("brief"),
                skill_artifacts=skill_artifacts,
            )
            draft_report = regenerated
            report_draft = regenerated.model_dump(mode="json")
            verification.setdefault("warnings", []).append(
                "AnalystAgent draft lacked grounded detail; regenerated with research draft builder."
            )
        draft_markdown = _build_markdown(draft_report, state.get("brief")) if draft_report else None
        return {
            "structured_cards": structured_cards,
            "comparison_matrix": comparison_matrix,
            "outline": outline,
            "writing_scaffold": skill_artifacts.get("writing_scaffold"),
            "writing_outline": skill_artifacts.get("writing_outline"),
            "mcp_prompt_payload": skill_artifacts.get("mcp_prompt_payload"),
            "skill_trace": skill_artifacts.get("skill_trace", []),
            "report_draft": report_draft,
            "draft_report": draft_report,
            "draft_markdown": draft_markdown,
            "artifacts": [
                {
                    "level": artifact.level,
                    "type": artifact.artifact_type,
                    "confidence": artifact.confidence,
                    "created_by": artifact.created_by,
                }
                for artifact in reasoning_state.artifacts
            ],
            "overall_confidence": reasoning_state.overall_confidence,
            "verification": verification,
        }

    def _coerce_draft_report(self, payload: Any) -> DraftReport | None:
        if isinstance(payload, DraftReport):
            return payload
        if not isinstance(payload, dict):
            return None

        sections = payload.get("sections")
        if not isinstance(sections, dict):
            sections = {
                key: value
                for key, value in payload.items()
                if isinstance(key, str) and isinstance(value, str)
            }
        claims = payload.get("claims", [])
        citations = payload.get("citations", [])
        try:
            return DraftReport(
                sections=sections or {},
                claims=list(claims) if isinstance(claims, list) else [],
                citations=list(citations) if isinstance(citations, list) else [],
            )
        except Exception as exc:
            logger.warning("[AnalystAgent] Failed to coerce DraftReport: %s", exc)
            return DraftReport(sections=sections or {}, claims=[], citations=[])


class AnalystGraphState(TypedDict, total=False):
    brief: dict[str, Any]
    paper_cards: list[Any]
    warnings: list[str]
    reasoning_state: ReasoningState
    structured_cards: list[dict[str, Any]] | None
    comparison_matrix: dict[str, Any] | None
    outline: Any
    report_draft: Any
    draft_report: DraftReport | None
    draft_markdown: str | None
    artifacts: list[dict[str, Any]]
    overall_confidence: float
    verification: dict[str, Any]
    drafting_skill_artifacts: dict[str, Any]
    writing_scaffold: dict[str, Any] | None
    writing_outline: list[str] | None
    mcp_prompt_payload: dict[str, Any] | None
    skill_trace: list[dict[str, Any]]


# ─── 入口函数 ───────────────────────────────────────────────────────────────


def run_analyst_agent(state: dict, inputs: dict) -> dict:
    """AnalystAgent 入口（兼容 supervisor 格式）。"""
    workspace_id = inputs.get("workspace_id") or state.get("workspace_id")
    task_id = inputs.get("task_id") or state.get("task_id")
    brief = state.get("brief") or {}
    paper_cards = state.get("paper_cards") or inputs.get("paper_cards", [])
    emitter = inputs.get("_event_emitter")

    agent = AnalystAgent(workspace_id=workspace_id, task_id=task_id)
    try:
        if emitter:
            emitter.on_thinking("draft", "Analyst agent is assembling writing artifacts and drafting the survey.")
        result = agent.run(brief=brief, paper_cards=paper_cards)
        if emitter:
            draft_report = result.get("draft_report") if isinstance(result, dict) else None
            if isinstance(draft_report, dict):
                citations = len(draft_report.get("citations", []) or [])
            else:
                citations = len(getattr(draft_report, "citations", []) or [])
            emitter.on_thinking("draft", f"Draft stage produced a structured report with {citations} citations.")
        return result
    except Exception as exc:
        logger.exception("[AnalystAgent] run failed: %s", exc)
        return {
            "paradigm": "reasoning_via_artifacts",
            "analyst_warnings": [f"AnalystAgent failed: {exc}"],
        }
