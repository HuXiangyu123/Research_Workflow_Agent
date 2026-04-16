"""Output workspace persistence.

Workspace-first layout:

        output/
            workspaces/
                <workspace_id>/
                    workspace.json
                    tasks/
                        <task_id>/
                            metadata.json
                            brief.json
                            search_plan.json
                            rag_result.json
                            paper_cards.json
                            draft.md
                            review_feedback.json
                            revisions/
                            report.md

Legacy fallback (when no workspace_id is provided):

        output/
            <task_id>/
                ...
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.models.workspace import ArtifactType, WorkspaceArtifact

logger = logging.getLogger(__name__)

# Root output directory — configurable via environment variable
OUTPUT_ROOT = Path(os.environ.get("PAPERREADER_OUTPUT_ROOT", "output"))
WORKSPACES_DIRNAME = "workspaces"
DEFAULT_WORKSPACE_USER = os.environ.get("PAPERREADER_DEFAULT_USER", "user")
CUSTOM_ARTIFACTS_DIRNAME = "artifacts"


def build_workspace_id(user_id: str = DEFAULT_WORKSPACE_USER, *, opened_at: datetime | None = None) -> str:
    """Build a readable workspace id from user + UTC timestamp."""
    current = (opened_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    safe_user = _sanitize_user_id(user_id)
    stamp = current.strftime("%Y%m%dT%H%M%SZ")
    return f"{safe_user}_{stamp}_{uuid4().hex[:6]}"


def get_workspace_root(workspace_id: str) -> Path:
    """Return the workspace root directory."""
    return OUTPUT_ROOT / WORKSPACES_DIRNAME / workspace_id


def ensure_workspace_root(
    *,
    workspace_id: str | None = None,
    user_id: str = DEFAULT_WORKSPACE_USER,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Create or refresh a workspace root manifest without creating a task yet."""
    now = datetime.now(timezone.utc)
    resolved_workspace_id = (workspace_id or "").strip() or build_workspace_id(user_id)
    payload = metadata or {}

    root = get_workspace_root(resolved_workspace_id)
    root.mkdir(parents=True, exist_ok=True)
    (root / "tasks").mkdir(parents=True, exist_ok=True)

    manifest_path = root / "workspace.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    opened_at = manifest.get("opened_at") or payload.get("workspace_opened_at") or now.isoformat()

    manifest_payload = {
        **manifest,
        "workspace_id": resolved_workspace_id,
        "user_id": user_id,
        "opened_at": opened_at,
        "updated_at": now.isoformat(),
        "task_ids": list(manifest.get("task_ids") or []),
    }
    if payload.get("source_type") is not None:
        manifest_payload["source_type"] = payload.get("source_type")
    elif manifest.get("source_type") is not None:
        manifest_payload["source_type"] = manifest.get("source_type")
    if payload.get("report_mode") is not None:
        manifest_payload["report_mode"] = payload.get("report_mode")
    elif manifest.get("report_mode") is not None:
        manifest_payload["report_mode"] = manifest.get("report_mode")

    _write_json_to_path(manifest_path, manifest_payload)
    return resolved_workspace_id


def workspace_exists(workspace_id: str) -> bool:
    """Return whether the workspace root exists on disk."""
    return get_workspace_root(workspace_id).is_dir()


def get_workspace_path(task_id: str, workspace_id: str | None = None) -> Path:
    """Return the task workspace directory.

    When ``workspace_id`` is provided, files are written under the workspace-first
    layout. Otherwise this falls back to the legacy ``output/<task_id>/`` layout.
    """
    if workspace_id:
        return get_workspace_root(workspace_id) / "tasks" / task_id
    return OUTPUT_ROOT / task_id


def create_workspace(
    task_id: str,
    metadata: dict[str, Any],
    *,
    workspace_id: str | None = None,
    user_id: str = DEFAULT_WORKSPACE_USER,
) -> Path:
    """
    Create the task workspace directory and write initial metadata.

    Idempotent: if the directory already exists, this is a no-op.
    """
    now = datetime.now(timezone.utc)
    workspace = get_workspace_path(task_id, workspace_id=workspace_id)
    workspace.mkdir(parents=True, exist_ok=True)

    # Ensure subdirectories exist
    (workspace / "revisions").mkdir(parents=True, exist_ok=True)

    if workspace_id:
        _upsert_workspace_manifest(
            workspace_id=workspace_id,
            task_id=task_id,
            user_id=user_id,
            metadata=metadata,
            timestamp=now,
        )

    metadata_path = workspace / "metadata.json"
    if metadata_path.exists():
        existing = _read_json(metadata_path)
    else:
        existing = {}

    _write_json_to_path(metadata_path, {
            **existing,
            "task_id": task_id,
            "workspace_id": workspace_id,
            "user_id": user_id,
            "created_at": existing.get("created_at") or now.isoformat(),
            **metadata,
        })
    logger.debug("[output_workspace] created %s", workspace)

    return workspace


def write_brief(task_id: str, brief: dict[str, Any], *, workspace_id: str | None = None) -> Path:
    """Write the clarify node output."""
    return _write_json(task_id, "brief.json", brief, workspace_id=workspace_id)


def write_search_plan(task_id: str, search_plan: dict[str, Any], *, workspace_id: str | None = None) -> Path:
    """Write the search_plan node output."""
    return _write_json(task_id, "search_plan.json", search_plan, workspace_id=workspace_id)


def write_rag_result(task_id: str, rag_result: dict[str, Any], *, workspace_id: str | None = None) -> Path:
    """Write the search/retrieval output."""
    return _write_json(task_id, "rag_result.json", rag_result, workspace_id=workspace_id)


def write_paper_cards(
    task_id: str,
    paper_cards: list[dict[str, Any]],
    *,
    workspace_id: str | None = None,
) -> Path:
    """Write the extract node output."""
    return _write_json(task_id, "paper_cards.json", paper_cards, workspace_id=workspace_id)


def write_draft(task_id: str, draft_markdown: str, *, workspace_id: str | None = None) -> Path:
    """Write the draft report markdown."""
    workspace = get_workspace_path(task_id, workspace_id=workspace_id)
    path = workspace / "draft.md"
    workspace.mkdir(parents=True, exist_ok=True)
    path.write_text(draft_markdown, encoding="utf-8")
    logger.debug("[output_workspace] wrote draft.md (%d chars)", len(draft_markdown))
    _touch_workspace_manifest(workspace_id, task_id)
    return path


def write_review_feedback(
    task_id: str,
    review_feedback: dict[str, Any],
    *,
    workspace_id: str | None = None,
) -> Path:
    """Write the review node output."""
    return _write_json(task_id, "review_feedback.json", review_feedback, workspace_id=workspace_id)


def write_draft_report(task_id: str, draft_report: dict[str, Any], *, workspace_id: str | None = None) -> Path:
    """Write the structured draft report (DraftReport JSON)."""
    return _write_json(task_id, "draft_report.json", draft_report, workspace_id=workspace_id)


def write_named_json(
    task_id: str,
    filename: str,
    payload: Any,
    *,
    workspace_id: str | None = None,
) -> Path:
    """Write an auxiliary JSON artifact under the task workspace."""
    return _write_json(task_id, filename, payload, workspace_id=workspace_id)


def append_revision(
    task_id: str,
    revision_markdown: str,
    label: str | None = None,
    *,
    workspace_id: str | None = None,
) -> Path:
    """
    Append a new revision to the revisions/ directory.

    Files are named as ``<3-digit index>_<label>.md``, e.g. ``001_initial.md``,
    ``002_after_review.md``. The label is inferred from the revision content
    when not provided.
    """
    workspace = get_workspace_path(task_id, workspace_id=workspace_id)
    revisions_dir = workspace / "revisions"
    revisions_dir.mkdir(parents=True, exist_ok=True)

    # Find next revision number
    existing = sorted(revisions_dir.glob("*.md"))
    next_num = len(existing) + 1

    # Infer label from first meaningful line
    if label is None:
        label = _infer_revision_label(revision_markdown)

    filename = f"{next_num:03d}_{label}.md"
    path = revisions_dir / filename
    path.write_text(revision_markdown, encoding="utf-8")
    logger.info("[output_workspace] appended revision %s", path.relative_to(OUTPUT_ROOT))
    _touch_workspace_manifest(workspace_id, task_id)
    return path


def write_report(task_id: str, report_markdown: str, *, workspace_id: str | None = None) -> Path:
    """
    Write the final report to ``report.md``.

    Also copies the last revision from revisions/ as ``report.md``.
    """
    workspace = get_workspace_path(task_id, workspace_id=workspace_id)
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / "report.md"
    path.write_text(report_markdown, encoding="utf-8")

    # Update metadata with completed_at
    metadata_path = workspace / "metadata.json"
    if metadata_path.exists():
        metadata = _read_json(metadata_path)
        metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_json_to_path(metadata_path, metadata)

    _touch_workspace_manifest(workspace_id, task_id, mark_completed=True)

    logger.info("[output_workspace] wrote report.md (%d chars)", len(report_markdown))
    return path


def write_node_output(
    task_id: str,
    node_name: str,
    node_result: Any,
    *,
    workspace_id: str | None = None,
) -> Path | None:
    """
    Write a node's output to the task workspace directory.

    Dispatches based on node_name, extracting the relevant field from the
    full node_result dict (which may contain metadata like _backend_mode, etc).
    """
    if node_name == "clarify":
        brief = node_result.get("brief") if isinstance(node_result, dict) else None
        return write_brief(task_id, brief, workspace_id=workspace_id) if brief else None
    elif node_name == "search_plan":
        plan = node_result.get("search_plan") if isinstance(node_result, dict) else None
        return write_search_plan(task_id, plan, workspace_id=workspace_id) if plan else None
    elif node_name == "search":
        rag = node_result.get("rag_result") if isinstance(node_result, dict) else None
        return write_rag_result(task_id, rag, workspace_id=workspace_id) if rag else None
    elif node_name == "extract":
        cards = node_result.get("paper_cards") if isinstance(node_result, dict) else None
        return write_paper_cards(task_id, cards, workspace_id=workspace_id) if cards else None
    elif node_name == "draft":
        if isinstance(node_result, dict):
            md = node_result.get("draft_markdown")
            dr = node_result.get("draft_report")
            if node_result.get("comparison_matrix"):
                write_named_json(task_id, "comparison_matrix.json", node_result.get("comparison_matrix"), workspace_id=workspace_id)
            if node_result.get("writing_scaffold"):
                write_named_json(task_id, "writing_scaffold.json", node_result.get("writing_scaffold"), workspace_id=workspace_id)
            if node_result.get("writing_outline"):
                write_named_json(task_id, "writing_outline.json", node_result.get("writing_outline"), workspace_id=workspace_id)
            if node_result.get("mcp_prompt_payload"):
                write_named_json(task_id, "mcp_prompt_payload.json", node_result.get("mcp_prompt_payload"), workspace_id=workspace_id)
            if node_result.get("skill_trace"):
                write_named_json(task_id, "draft_skill_trace.json", node_result.get("skill_trace"), workspace_id=workspace_id)
            if md:
                return write_draft(task_id, md, workspace_id=workspace_id)
            elif dr:
                return write_draft_report(task_id, dr, workspace_id=workspace_id)
        elif isinstance(node_result, str):
            return write_draft(task_id, node_result, workspace_id=workspace_id)
        return None
    elif node_name == "review":
        fb = node_result.get("review_feedback") if isinstance(node_result, dict) else None
        if isinstance(node_result, dict):
            if node_result.get("claim_verification"):
                write_named_json(task_id, "claim_verification.json", node_result.get("claim_verification"), workspace_id=workspace_id)
            if node_result.get("skill_trace"):
                write_named_json(task_id, "review_skill_trace.json", node_result.get("skill_trace"), workspace_id=workspace_id)
        return write_review_feedback(task_id, fb, workspace_id=workspace_id) if fb else None
    else:
        logger.debug("[output_workspace] no canonical file for node %s", node_name)
        return None


def list_revisions(task_id: str, *, workspace_id: str | None = None) -> list[Path]:
    """Return sorted list of revision file paths."""
    workspace = get_workspace_path(task_id, workspace_id=workspace_id)
    revisions_dir = workspace / "revisions"
    if not revisions_dir.is_dir():
        return []
    return sorted(revisions_dir.glob("*.md"))


def get_workspace_summary(task_id: str, *, workspace_id: str | None = None) -> dict[str, Any]:
    """Return a summary of what's in the workspace."""
    workspace = get_workspace_path(task_id, workspace_id=workspace_id)
    if not workspace.is_dir():
        return {
            "exists": False,
            "task_id": task_id,
            "workspace_id": workspace_id,
        }

    files = {p.name: p.stat().st_size for p in workspace.rglob("*") if p.is_file()}
    revisions = [
        {"name": p.name, "size": p.stat().st_size}
        for p in sorted((workspace / "revisions").glob("*.md"))
    ]

    return {
        "exists": True,
        "task_id": task_id,
        "workspace_id": workspace_id,
        "path": str(workspace),
        "files": files,
        "revision_count": len(revisions),
        "revisions": revisions,
    }


def load_workspace_manifest(workspace_id: str) -> dict[str, Any] | None:
    """Load ``workspace.json`` for a workspace if present."""
    manifest_path = get_workspace_root(workspace_id) / "workspace.json"
    if not manifest_path.is_file():
        return None
    return _read_json(manifest_path)


def list_workspaces(*, limit: int | None = None) -> list[dict[str, Any]]:
    """List workspace manifests from disk, newest first."""
    root = OUTPUT_ROOT / WORKSPACES_DIRNAME
    if not root.is_dir():
        return []

    manifests: list[dict[str, Any]] = []
    for workspace_dir in root.iterdir():
        if not workspace_dir.is_dir():
            continue
        manifest = load_workspace_manifest(workspace_dir.name)
        if not isinstance(manifest, dict):
            continue
        manifests.append(
            {
                **manifest,
                "workspace_id": manifest.get("workspace_id") or workspace_dir.name,
            }
        )

    manifests.sort(
        key=lambda item: str(item.get("updated_at") or item.get("opened_at") or ""),
        reverse=True,
    )
    return manifests[: max(limit, 0)] if limit is not None else manifests


def create_workspace_artifact(artifact: WorkspaceArtifact) -> Path:
    """Persist a user-created workspace artifact under the workspace root."""
    root = get_workspace_root(artifact.workspace_id)
    if not root.is_dir():
        raise FileNotFoundError(f"Workspace {artifact.workspace_id!r} does not exist")

    artifacts_dir = root / CUSTOM_ARTIFACTS_DIRNAME
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / f"{artifact.artifact_id}.json"
    _write_json_to_path(path, artifact.model_dump(mode="json"))
    _touch_workspace_manifest(artifact.workspace_id, artifact.task_id or "")
    return path


def list_workspace_artifacts(
    workspace_id: str,
    artifact_type: ArtifactType | str | None = None,
) -> list[WorkspaceArtifact]:
    """List artifacts backed by the on-disk workspace layout."""
    root = get_workspace_root(workspace_id)
    if not root.is_dir():
        return []

    requested_type: ArtifactType | None = None
    if artifact_type is not None:
        try:
            requested_type = artifact_type if isinstance(artifact_type, ArtifactType) else ArtifactType(artifact_type)
        except ValueError:
            return []

    artifacts: list[WorkspaceArtifact] = []
    tasks_root = root / "tasks"
    if tasks_root.is_dir():
        for task_dir in sorted(tasks_root.iterdir()):
            if not task_dir.is_dir():
                continue
            artifacts.extend(_list_task_artifacts(workspace_id, task_dir))

    custom_dir = root / CUSTOM_ARTIFACTS_DIRNAME
    if custom_dir.is_dir():
        for artifact_file in sorted(custom_dir.glob("*.json")):
            artifact = _load_custom_artifact(artifact_file)
            if artifact is not None:
                artifacts.append(artifact)

    if requested_type is not None:
        artifacts = [artifact for artifact in artifacts if artifact.artifact_type == requested_type]

    artifacts.sort(key=_artifact_sort_key, reverse=True)
    return artifacts


# ─── helpers ───────────────────────────────────────────────────────────────────


def _write_json(task_id: str, filename: str, data: Any, *, workspace_id: str | None = None) -> Path:
    workspace = get_workspace_path(task_id, workspace_id=workspace_id)
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / filename
    _write_json_to_path(path, data)
    _touch_workspace_manifest(workspace_id, task_id)
    return path


def _write_json_to_path(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return _read_json(path)
    except Exception:
        logger.warning("[output_workspace] failed to read json %s", path)
        return None


def _infer_revision_label(text: str) -> str:
    """Infer a short label from the revision content."""
    first_line = text.strip().split("\n")[0][:40] if text.strip() else "revision"
    # Replace spaces/special chars
    label = "".join(c if c.isalnum() else "_" for c in first_line)
    return label.lower()[:30] or "revision"


def _sanitize_user_id(user_id: str) -> str:
    cleaned = "".join(c.lower() if c.isalnum() else "_" for c in user_id.strip())
    compact = "_".join(part for part in cleaned.split("_") if part)
    return compact or "user"


def _list_task_artifacts(workspace_id: str, task_dir: Path) -> list[WorkspaceArtifact]:
    task_id = task_dir.name
    artifacts: list[WorkspaceArtifact] = []

    for filename in (
        "brief.json",
        "search_plan.json",
        "rag_result.json",
        "paper_cards.json",
        "comparison_matrix.json",
        "writing_scaffold.json",
        "writing_outline.json",
        "mcp_prompt_payload.json",
        "claim_verification.json",
        "draft_skill_trace.json",
        "review_skill_trace.json",
        "draft_report.json",
        "draft.md",
        "review_feedback.json",
        "report.md",
    ):
        path = task_dir / filename
        artifact = _artifact_from_task_file(workspace_id, task_id, path)
        if artifact is not None:
            artifacts.append(artifact)

    revisions_dir = task_dir / "revisions"
    if revisions_dir.is_dir():
        for revision_path in sorted(revisions_dir.glob("*.md")):
            artifact = _artifact_from_task_file(workspace_id, task_id, revision_path)
            if artifact is not None:
                artifacts.append(artifact)

    return artifacts


def _artifact_from_task_file(workspace_id: str, task_id: str, path: Path) -> WorkspaceArtifact | None:
    if not path.is_file():
        return None

    created_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    name = path.name
    artifact_id = f"art_{task_id}_{path.stem}".replace("-", "_")
    tags = [task_id]
    metadata: dict[str, Any] = {
        "task_id": task_id,
        "file_name": name,
    }
    summary: str | None = None
    title: str
    artifact_type: ArtifactType
    created_by_node: str | None

    if name == "brief.json":
        payload = _read_json_if_exists(path) or {}
        topic = str(payload.get("topic") or payload.get("research_topic") or "").strip()
        artifact_type = ArtifactType.BRIEF
        title = f"Brief for task {task_id}"
        created_by_node = "clarify"
        summary = topic or None
    elif name == "search_plan.json":
        payload = _read_json_if_exists(path) or {}
        artifact_type = ArtifactType.SEARCH_PLAN
        title = f"Search plan for task {task_id}"
        created_by_node = "search_plan"
        summary = str(payload.get("plan_goal") or "").strip() or None
    elif name == "rag_result.json":
        payload = _read_json_if_exists(path) or {}
        artifact_type = ArtifactType.RAG_RESULT
        title = f"RAG result for task {task_id}"
        created_by_node = "search"
        total_papers = int(payload.get("total_papers") or len(payload.get("paper_candidates") or []))
        query = str(payload.get("query") or "").strip()
        summary = f"{total_papers} papers" + (f" for {query}" if query else "")
    elif name == "paper_cards.json":
        payload = _read_json_if_exists(path) or []
        artifact_type = ArtifactType.PAPER_CARD
        title = f"Paper cards for task {task_id}"
        created_by_node = "extract"
        count = len(payload) if isinstance(payload, list) else 0
        summary = f"{count} paper cards"
    elif name == "comparison_matrix.json":
        payload = _read_json_if_exists(path) or {}
        artifact_type = ArtifactType.COMPARISON_MATRIX
        title = f"Comparison matrix for task {task_id}"
        created_by_node = "draft"
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
        summary = f"{len(rows)} comparison rows" if isinstance(rows, list) else None
    elif name == "writing_scaffold.json":
        payload = _read_json_if_exists(path) or {}
        artifact_type = ArtifactType.REPORT_OUTLINE
        title = f"Writing scaffold for task {task_id}"
        created_by_node = "draft"
        summary = f"{len(payload)} planned sections" if isinstance(payload, dict) else None
    elif name == "writing_outline.json":
        payload = _read_json_if_exists(path) or []
        artifact_type = ArtifactType.REPORT_OUTLINE
        title = f"Writing outline for task {task_id}"
        created_by_node = "draft"
        count = len(payload) if isinstance(payload, list) else 0
        summary = f"{count} outline entries" if count else None
    elif name == "mcp_prompt_payload.json":
        payload = _read_json_if_exists(path) or {}
        artifact_type = ArtifactType.TOOL_TRACE
        title = f"MCP writing prompt payload for task {task_id}"
        created_by_node = "draft"
        summary = str(payload.get("prompt", "")).strip()[:160] or None
    elif name == "claim_verification.json":
        payload = _read_json_if_exists(path) or {}
        artifact_type = ArtifactType.REVIEW_FEEDBACK
        title = f"Claim verification for task {task_id}"
        created_by_node = "review"
        stats = payload.get("grounding_stats", {}) if isinstance(payload, dict) else {}
        summary = (
            f"supported_ratio={stats.get('supported_ratio')}"
            if isinstance(stats, dict) and stats
            else None
        )
    elif name in {"draft_skill_trace.json", "review_skill_trace.json"}:
        payload = _read_json_if_exists(path) or []
        artifact_type = ArtifactType.NODE_TRACE
        title = f"{path.stem.replace('_', ' ').title()} for task {task_id}"
        created_by_node = "draft" if name.startswith("draft_") else "review"
        count = len(payload) if isinstance(payload, list) else 0
        summary = f"{count} skill invocations" if count else None
    elif name == "draft_report.json":
        payload = _read_json_if_exists(path) or {}
        artifact_type = ArtifactType.REPORT_DRAFT
        title = f"Structured draft for task {task_id}"
        created_by_node = "draft"
        sections = payload.get("sections", {}) if isinstance(payload, dict) else {}
        summary = f"{len(sections)} sections" if isinstance(sections, dict) else None
        tags.append("structured")
    elif name == "draft.md":
        artifact_type = ArtifactType.REPORT_DRAFT
        title = f"Draft markdown for task {task_id}"
        created_by_node = "draft"
        summary = _summarize_markdown(path)
    elif name == "report.md":
        artifact_type = ArtifactType.REPORT_DRAFT
        title = f"Final report for task {task_id}"
        created_by_node = "persist_artifacts"
        summary = _summarize_markdown(path)
        tags.append("final")
    elif name == "review_feedback.json":
        payload = _read_json_if_exists(path) or {}
        artifact_type = ArtifactType.REVIEW_FEEDBACK
        title = f"Review feedback for task {task_id}"
        created_by_node = "review"
        summary = str(payload.get("summary") or "").strip() or None
    elif path.parent.name == "revisions" and path.suffix == ".md":
        artifact_type = ArtifactType.REPORT_DRAFT
        title = f"Revision {path.stem} for task {task_id}"
        created_by_node = "review"
        summary = _summarize_markdown(path)
        tags.extend(["revision", path.stem])
    else:
        return None

    return WorkspaceArtifact(
        artifact_id=artifact_id,
        workspace_id=workspace_id,
        task_id=task_id,
        artifact_type=artifact_type,
        title=title,
        status="ready",
        created_at=created_at,
        created_by_node=created_by_node,
        content_ref=str(path.resolve()),
        summary=summary,
        tags=tags,
        metadata=metadata,
    )


def _load_custom_artifact(path: Path) -> WorkspaceArtifact | None:
    payload = _read_json_if_exists(path)
    if not isinstance(payload, dict):
        return None
    try:
        artifact = WorkspaceArtifact.model_validate(payload)
    except Exception:
        logger.warning("[output_workspace] failed to validate custom artifact %s", path)
        return None
    if not artifact.content_ref:
        artifact.content_ref = str(path.resolve())
    return artifact


def _summarize_markdown(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:200]
    return None


def _artifact_sort_key(artifact: WorkspaceArtifact) -> float:
    created_at = artifact.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return created_at.timestamp()


def _upsert_workspace_manifest(
    *,
    workspace_id: str,
    task_id: str,
    user_id: str,
    metadata: dict[str, Any],
    timestamp: datetime,
) -> None:
    root = get_workspace_root(workspace_id)
    root.mkdir(parents=True, exist_ok=True)
    (root / "tasks").mkdir(parents=True, exist_ok=True)

    manifest_path = root / "workspace.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    task_ids = list(manifest.get("task_ids") or [])
    if task_id not in task_ids:
        task_ids.append(task_id)

    opened_at = manifest.get("opened_at") or metadata.get("workspace_opened_at") or timestamp.isoformat()
    manifest_payload = {
        **manifest,
        "workspace_id": workspace_id,
        "user_id": user_id,
        "opened_at": opened_at,
        "updated_at": timestamp.isoformat(),
        "task_ids": task_ids,
        "source_type": metadata.get("source_type", manifest.get("source_type")),
        "report_mode": metadata.get("report_mode", manifest.get("report_mode")),
        "latest_task_id": task_id,
    }
    _write_json_to_path(manifest_path, manifest_payload)


def _touch_workspace_manifest(
    workspace_id: str | None,
    task_id: str,
    *,
    mark_completed: bool = False,
) -> None:
    if not workspace_id:
        return

    now = datetime.now(timezone.utc)
    root = get_workspace_root(workspace_id)
    manifest_path = root / "workspace.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    task_ids = list(manifest.get("task_ids") or [])
    if task_id not in task_ids:
        task_ids.append(task_id)

    payload = {
        **manifest,
        "workspace_id": workspace_id,
        "user_id": manifest.get("user_id") or DEFAULT_WORKSPACE_USER,
        "opened_at": manifest.get("opened_at") or now.isoformat(),
        "updated_at": now.isoformat(),
        "task_ids": task_ids,
        "latest_task_id": task_id,
    }
    if mark_completed:
        payload["completed_at"] = now.isoformat()

    _write_json_to_path(manifest_path, payload)
