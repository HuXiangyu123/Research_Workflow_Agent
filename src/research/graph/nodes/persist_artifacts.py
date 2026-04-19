"""Persist artifacts node — Phase 3: 扩展 artifact 类型到 review/trace/eval。"""

from __future__ import annotations

import logging
from typing import Any

from src.models.workspace import ArtifactType, WorkspaceArtifact
from src.tasking.trace_wrapper import trace_node, get_trace_store

logger = logging.getLogger(__name__)


# In-memory artifact store (replace with DB in production)
_artifact_store: dict[str, list[WorkspaceArtifact]] = {}


def get_artifact_store() -> dict[str, list[WorkspaceArtifact]]:
    return _artifact_store


def save_artifact(artifact: WorkspaceArtifact) -> WorkspaceArtifact:
    """持久化 artifact（默认内存存储）。"""
    ws_id = artifact.workspace_id
    if ws_id not in _artifact_store:
        _artifact_store[ws_id] = []
    _artifact_store[ws_id].append(artifact)
    logger.info(
        f"[persist_artifacts] saved artifact={artifact.artifact_id} "
        f"type={artifact.artifact_type.value} ws={ws_id}"
    )
    return artifact


def list_artifacts(
    workspace_id: str,
    artifact_type: ArtifactType | None = None,
) -> list[WorkspaceArtifact]:
    """列出 workspace 下的 artifacts。"""
    artifacts = _artifact_store.get(workspace_id, [])
    if artifact_type:
        artifacts = [a for a in artifacts if a.artifact_type == artifact_type]
    return artifacts


@trace_node(node_name="persist_artifacts", stage="persist", store=get_trace_store())
def persist_artifacts_node(state: dict) -> dict:
    """
    Phase 3 persist_artifacts 节点。

    将 review_feedback / trace / eval_report 等新 artifact 类型写入 workspace。
    扩展了 Phase 1-2 的 persist 能力（brief / search_plan / paper_card / upload）。
    """
    workspace_id = str(state.get("workspace_id", ""))
    task_id = str(state.get("task_id", ""))
    created_by_node = "persist_artifacts"

    artifacts_created: list[dict] = []

    def _save(
        artifact_type: ArtifactType,
        title: str,
        content_ref: str | None,
        summary: str | None,
        tags: list[str],
        metadata: dict,
    ) -> str:
        artifact = save_artifact(
            WorkspaceArtifact(
                workspace_id=workspace_id,
                task_id=task_id,
                artifact_type=artifact_type,
                title=title,
                content_ref=content_ref,
                summary=summary,
                tags=tags,
                metadata=metadata,
                created_by_node=created_by_node,
            )
        )
        artifacts_created.append({
            "artifact_id": artifact.artifact_id,
            "artifact_type": artifact.artifact_type.value,
            "title": artifact.title,
        })
        return artifact.artifact_id

    # ── 1. Review Feedback ──────────────────────────────────────────────
    review_feedback = state.get("review_feedback")
    if review_feedback is not None:
        if hasattr(review_feedback, "model_dump"):
            review_feedback_payload = review_feedback.model_dump(mode="json")
        elif isinstance(review_feedback, dict):
            review_feedback_payload = review_feedback
        else:
            review_feedback_payload = {}

        summary = review_feedback_payload.get("summary")
        review_id = review_feedback_payload.get("review_id", "")
        _save(
            artifact_type=ArtifactType.REVIEW_FEEDBACK,
            title=f"Review for task {task_id}",
            content_ref=None,
            summary=str(summary) if summary else None,
            tags=["review", "quality"],
            metadata={"review_id": str(review_id) if review_id else ""},
        )

    # ── 2. RAG Result ────────────────────────────────────────────────
    rag_result = state.get("rag_result")
    if rag_result is not None:
        _save(
            artifact_type=ArtifactType.RAG_RESULT,
            title=f"RAG result for task {task_id}",
            content_ref=None,
            summary=None,
            tags=["rag", "retrieval"],
            metadata={},
        )

    # ── 3. Report Draft ─────────────────────────────────────────────
    draft = state.get("draft_report") or state.get("draft_markdown")
    if draft:
        _save(
            artifact_type=ArtifactType.REPORT_DRAFT,
            title=f"Draft report for task {task_id}",
            content_ref=None,
            summary=str(draft)[:200] if draft else None,
            tags=["draft", "report"],
            metadata={},
        )

    # ── 4. Comparison Matrix ──────────────────────────────────────────
    matrix = state.get("comparison_matrix")
    if matrix:
        _save(
            artifact_type=ArtifactType.COMPARISON_MATRIX,
            title=f"Comparison matrix for task {task_id}",
            content_ref=None,
            summary=None,
            tags=["matrix", "comparison"],
            metadata={},
        )

    # ── 5. Eval Report ───────────────────────────────────────────────
    eval_report = state.get("eval_report")
    if eval_report:
        _save(
            artifact_type=ArtifactType.EVAL_REPORT,
            title=f"Eval report for task {task_id}",
            content_ref=None,
            summary=None,
            tags=["eval", "quality"],
            metadata={},
        )

    # ── 6. Node/ Tool Trace ──────────────────────────────────────────
    # Trace artifacts are persisted separately via InMemoryTraceStore
    # Here we record that the trace was saved
    trace_store = get_trace_store()
    node_runs = trace_store.get_node_runs(task_id)
    tool_runs = trace_store.get_tool_runs(task_id)
    if node_runs or tool_runs:
        _save(
            artifact_type=ArtifactType.NODE_TRACE,
            title=f"Node trace for task {task_id}",
            content_ref=None,
            summary=f"{len(node_runs)} node runs, {len(tool_runs)} tool runs",
            tags=["trace", "debugging"],
            metadata={
                "node_run_count": len(node_runs),
                "tool_run_count": len(tool_runs),
            },
        )

    logger.info(f"[persist_artifacts] created {len(artifacts_created)} artifacts")

    return {
        "artifacts_created": artifacts_created,
        "artifact_count": len(artifacts_created),
    }
