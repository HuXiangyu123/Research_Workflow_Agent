"""Review node — Phase 3: 生成 ReviewFeedback。"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any

from src.models.agent import AgentRole
from src.models.report import DraftReport
from src.models.review import CoverageGap, ReviewFeedback, ReviewIssue, ReviewSeverity, ReviewCategory
from src.research.services.reviewer import ReviewerService
from src.research.services.grounding import ground_draft_report
from src.tasking.trace_wrapper import trace_node, trace_tool, get_trace_store

logger = logging.getLogger(__name__)

_reviewer = ReviewerService()


def _run_reviewer_sync(**kwargs) -> ReviewFeedback:
    """Execute the async reviewer from both plain sync code and loop-backed contexts."""

    def _runner() -> ReviewFeedback:
        return asyncio.run(_reviewer.review(**kwargs))

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _runner()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_runner).result()


def _run_claim_verification_skill(
    *,
    workspace_id: str,
    task_id: str,
    draft_report: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from src.models.skills import SkillRunRequest
    from src.skills.registry import get_skills_registry

    registry = get_skills_registry()
    req = SkillRunRequest(
        workspace_id=workspace_id,
        task_id=task_id,
        skill_id="claim_verification",
        inputs={"draft_report": _serialize_report_payload(draft_report)},
        preferred_agent=AgentRole.REVIEWER,
    )
    resp = registry.run_sync(req, {"workspace_id": workspace_id, "task_id": task_id})
    return resp.result, [{"skill_id": "claim_verification", "backend": resp.backend.value, "summary": resp.summary}]


def _serialize_report_payload(report: Any) -> dict[str, Any]:
    if report is None:
        return {}
    if isinstance(report, dict):
        return report
    if hasattr(report, "model_dump"):
        return report.model_dump(mode="json")
    return {}


def _coerce_draft_report(report: Any) -> Any:
    if isinstance(report, DraftReport):
        return report
    if not isinstance(report, dict):
        return report

    sections = report.get("sections")
    if not isinstance(sections, dict):
        sections = {
            key: value
            for key, value in report.items()
            if isinstance(key, str) and isinstance(value, str)
        }
    claims = report.get("claims", [])
    citations = report.get("citations", [])
    try:
        return DraftReport(
            sections=sections or {},
            claims=list(claims) if isinstance(claims, list) else [],
            citations=list(citations) if isinstance(citations, list) else [],
        )
    except Exception:
        return report


@trace_node(node_name="review", stage="review", store=get_trace_store())
def review_node(state: dict) -> dict:
    """
    Phase 3 review 节点。

    从 state 中读取：
    - rag_result
    - paper_cards
    - report_draft

    写入 state：
    - review_feedback: ReviewFeedback
    - review_passed: bool
    """
    task_id = str(state.get("task_id", ""))
    workspace_id = str(state.get("workspace_id", ""))
    rag_result = state.get("rag_result")
    paper_cards = state.get("paper_cards") or []
    draft_report = state.get("draft_report")
    draft_report = _coerce_draft_report(draft_report)
    report_draft = draft_report or state.get("draft_markdown")
    grounding_result: dict[str, Any] = {}
    grounding_warnings: list[str] = []
    skill_trace: list[dict[str, Any]] = []
    claim_verification: dict[str, Any] = {}

    if draft_report is not None:
        import logging as _logging
        _logger = _logging.getLogger(__name__)
        _logger.debug(
            "[review_node] before grounding: paper_cards=%d, draft_report type=%s",
            len(paper_cards),
            type(draft_report).__name__,
        )
        try:
            grounding_result = ground_draft_report(
                draft_report,
                paper_cards=paper_cards,
                report_mode=str(state.get("report_mode", "draft") or "draft"),
                degradation_mode=str(state.get("degradation_mode", "normal") or "normal"),
            )
            report_draft = (
                grounding_result.get("verified_report")
                or grounding_result.get("final_report")
                or draft_report
            )
            grounding_warnings = [
                str(item) for item in grounding_result.get("warnings", []) if item
            ]
        except Exception as exc:
            logger.exception("[review_node] grounding failed: %s", exc)
            grounding_warnings = [f"grounding failed: {exc}"]

    logger.info(
        f"[review_node] task={task_id} "
        f"rag_result={rag_result is not None} "
        f"paper_cards={len(paper_cards)}"
    )

    # 如果没有 paper_cards 和 draft，发出警告而非继续生成空报告
    if not paper_cards and not draft_report and not report_draft:
        logger.warning(
            "[review_node] no paper_cards and no draft_report — "
            "search/extract/draft pipeline likely failed. Returning early."
        )
        return {
            "review_feedback": ReviewFeedback(
                task_id=task_id,
                workspace_id=workspace_id,
                passed=False,
                summary="No paper_cards available: the search or extract pipeline failed. "
                        "The draft cannot be generated. Please retry with a more specific topic.",
                issues=[
                    ReviewIssue(
                        severity=ReviewSeverity.BLOCKER,
                        category=ReviewCategory.COVERAGE_GAP,
                        target="paper_cards",
                        summary="检索阶段未返回任何论文，可能是查询不相关或服务不可用",
                    ),
                ],
                coverage_gaps=[
                    CoverageGap(
                        missing_topics=["检索结果为空", "无有效 PaperCards"],
                        note="建议使用更具体的研究主题（包含英文技术术语），并确保 SearXNG/arXiv API 服务正常",
                    ),
                ],
            ),
            "review_passed": False,
        }

    # Run reviewer synchronously (LLM calls inside ReviewerService)
    try:
        feedback = _run_reviewer_sync(
            task_id=task_id,
            workspace_id=workspace_id,
            rag_result=rag_result,
            paper_cards=paper_cards,
            report_draft=report_draft,
        )
    except Exception as exc:
        logger.exception(f"[review_node] reviewer failed: {exc}")
        # Fallback: create a minimal failed feedback
        feedback = ReviewFeedback(
            task_id=task_id,
            workspace_id=workspace_id,
            passed=False,
            summary=f"Reviewer service error: {exc}",
        )

    grounded_report_for_skill = (
        grounding_result.get("verified_report")
        or grounding_result.get("final_report")
        or grounding_result.get("draft_report")
        or report_draft
    )
    if grounded_report_for_skill is not None and workspace_id and task_id:
        try:
            claim_verification, skill_entries = _run_claim_verification_skill(
                workspace_id=workspace_id,
                task_id=task_id,
                draft_report=grounded_report_for_skill,
            )
            skill_trace.extend(skill_entries)
        except Exception as exc:  # noqa: BLE001
            grounding_warnings.append(f"claim verification skill failed: {exc}")

    result = {
        **grounding_result,
        "review_feedback": feedback,
        "review_passed": feedback.passed,
        "claim_verification": claim_verification,
        "skill_trace": skill_trace,
    }
    if grounding_warnings:
        result["warnings"] = grounding_warnings
    return result
