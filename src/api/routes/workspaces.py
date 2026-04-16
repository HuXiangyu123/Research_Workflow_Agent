"""Workspaces API — Phase 3: artifact 面板数据接口。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agent.output_workspace import (
    DEFAULT_WORKSPACE_USER,
    create_workspace_artifact,
    ensure_workspace_root,
    get_workspace_root,
    list_workspaces,
    list_workspace_artifacts,
    load_workspace_manifest,
    workspace_exists,
)
from src.models.workspace import ArtifactType, WorkspaceArtifact

router = APIRouter(prefix="/api/v1/workspaces", tags=["workspaces"])


# ─── Request/Response Models ────────────────────────────────────────────────


class WorkspaceSummaryResponse(BaseModel):
    workspace_id: str
    status: str
    current_stage: str | None = None
    warnings: list[str] = Field(default_factory=list)
    artifact_count: int = 0
    latest_task_id: str | None = None
    opened_at: str | None = None
    updated_at: str | None = None
    source_type: str | None = None
    report_mode: str | None = None


class WorkspaceListResponse(BaseModel):
    items: list[WorkspaceSummaryResponse]
    total: int


class WorkspaceArtifactItem(BaseModel):
    artifact_id: str
    workspace_id: str
    task_id: str | None = None
    artifact_type: str
    title: str
    status: str
    created_at: str
    created_by_node: str | None = None
    content_ref: str | None = None
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class WorkspaceArtifactsResponse(BaseModel):
    workspace_id: str
    items: list[WorkspaceArtifactItem]
    total: int


class WorkspaceArtifactContentResponse(BaseModel):
    workspace_id: str
    artifact_id: str
    artifact_type: str
    title: str
    content_type: str
    content: object
    path: str


class CreateArtifactRequest(BaseModel):
    artifact_type: str
    title: str
    task_id: str | None = None
    content_ref: str | None = None
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class CreateWorkspaceRequest(BaseModel):
    workspace_id: str | None = None
    user_id: str = Field(default=DEFAULT_WORKSPACE_USER or "user")
    source_type: str | None = None
    report_mode: str | None = None


def _require_workspace(workspace_id: str) -> dict:
    manifest = load_workspace_manifest(workspace_id)
    if manifest is None or not workspace_exists(workspace_id):
        raise HTTPException(status_code=404, detail="Workspace not found")
    return manifest


def _workspace_status(workspace_id: str, manifest: dict) -> str:
    latest_task_id = str(manifest.get("latest_task_id") or "")
    if latest_task_id:
        task_dir = get_workspace_root(workspace_id) / "tasks" / latest_task_id
        if (task_dir / "report.md").is_file():
            return "completed"
    if manifest.get("completed_at"):
        return "completed"
    return "active"


def _workspace_stage(workspace_id: str, manifest: dict) -> str | None:
    latest_task_id = str(manifest.get("latest_task_id") or "")
    if not latest_task_id:
        return None

    task_dir = get_workspace_root(workspace_id) / "tasks" / latest_task_id
    if (task_dir / "report.md").is_file():
        return "completed"
    if (task_dir / "review_feedback.json").is_file():
        return "review"
    if (task_dir / "draft.md").is_file() or (task_dir / "draft_report.json").is_file():
        return "draft"
    if (task_dir / "paper_cards.json").is_file():
        return "extract"
    if (task_dir / "rag_result.json").is_file():
        return "search"
    if (task_dir / "search_plan.json").is_file():
        return "search_plan"
    if (task_dir / "brief.json").is_file():
        return "clarify"
    return None


def _to_item(artifact: WorkspaceArtifact) -> WorkspaceArtifactItem:
    return WorkspaceArtifactItem(
        artifact_id=artifact.artifact_id,
        workspace_id=artifact.workspace_id,
        task_id=artifact.task_id,
        artifact_type=artifact.artifact_type.value,
        title=artifact.title,
        status=artifact.status,
        created_at=artifact.created_at.isoformat(),
        created_by_node=artifact.created_by_node,
        content_ref=artifact.content_ref,
        summary=artifact.summary,
        tags=artifact.tags,
        metadata=artifact.metadata,
    )


def _to_summary_response(workspace_id: str, manifest: dict) -> WorkspaceSummaryResponse:
    artifacts = list_workspace_artifacts(workspace_id)
    return WorkspaceSummaryResponse(
        workspace_id=workspace_id,
        status=_workspace_status(workspace_id, manifest),
        current_stage=_workspace_stage(workspace_id, manifest),
        warnings=[],
        artifact_count=len(artifacts),
        latest_task_id=str(manifest.get("latest_task_id") or "") or None,
        opened_at=str(manifest.get("opened_at") or "") or None,
        updated_at=str(manifest.get("updated_at") or "") or None,
        source_type=str(manifest.get("source_type") or "") or None,
        report_mode=str(manifest.get("report_mode") or "") or None,
    )


def _resolve_artifact(workspace_id: str, artifact_id: str) -> WorkspaceArtifact:
    artifacts = list_workspace_artifacts(workspace_id)
    for artifact in artifacts:
        if artifact.artifact_id == artifact_id:
            return artifact
    raise HTTPException(status_code=404, detail="Artifact not found")


# ─── Routes ───────────────────────────────────────────────────────────────


@router.get("", response_model=WorkspaceListResponse)
async def list_workspaces_endpoint(limit: int = 50) -> WorkspaceListResponse:
    """List persisted workspaces from output/workspaces."""
    items: list[WorkspaceSummaryResponse] = []
    for manifest in list_workspaces(limit=limit):
        workspace_id = str(manifest.get("workspace_id") or "").strip()
        if not workspace_id:
            continue
        items.append(_to_summary_response(workspace_id, manifest))
    return WorkspaceListResponse(items=items, total=len(items))


@router.post("", response_model=WorkspaceSummaryResponse)
async def create_workspace(req: CreateWorkspaceRequest) -> WorkspaceSummaryResponse:
    """Create an empty workspace shell that later tasks can attach to."""
    user_id = (req.user_id or DEFAULT_WORKSPACE_USER or "user").strip() or "user"
    workspace_id = ensure_workspace_root(
        workspace_id=req.workspace_id,
        user_id=user_id,
        metadata={
            "workspace_opened_at": None,
            "source_type": req.source_type,
            "report_mode": req.report_mode,
        },
    )
    manifest = _require_workspace(workspace_id)
    return _to_summary_response(workspace_id, manifest)


@router.get("/{workspace_id}", response_model=WorkspaceSummaryResponse)
async def get_workspace(workspace_id: str) -> WorkspaceSummaryResponse:
    """返回 workspace 概览（状态、stage、warnings）。"""
    manifest = _require_workspace(workspace_id)
    return _to_summary_response(workspace_id, manifest)


@router.get("/{workspace_id}/artifacts", response_model=WorkspaceArtifactsResponse)
async def list_workspace_artifacts_endpoint(
    workspace_id: str,
    artifact_type: str | None = None,
) -> WorkspaceArtifactsResponse:
    """
    返回 workspace 下的 artifact 列表。

    Query 参数：
        artifact_type: 可选，按类型过滤（review_feedback / rag_result / report_draft / ...）
    """
    _require_workspace(workspace_id)
    artifacts = list_workspace_artifacts(workspace_id, artifact_type)
    items = [_to_item(a) for a in artifacts]
    return WorkspaceArtifactsResponse(
        workspace_id=workspace_id,
        items=items,
        total=len(items),
    )


@router.get("/{workspace_id}/artifacts/{artifact_id}", response_model=WorkspaceArtifactItem)
async def get_artifact(workspace_id: str, artifact_id: str) -> WorkspaceArtifactItem:
    """返回单个 artifact 的详细信息。"""
    _require_workspace(workspace_id)
    return _to_item(_resolve_artifact(workspace_id, artifact_id))


@router.get(
    "/{workspace_id}/artifacts/{artifact_id}/content",
    response_model=WorkspaceArtifactContentResponse,
)
async def get_artifact_content(
    workspace_id: str,
    artifact_id: str,
) -> WorkspaceArtifactContentResponse:
    """Read artifact content from the persisted workspace file."""
    _require_workspace(workspace_id)
    artifact = _resolve_artifact(workspace_id, artifact_id)
    content_ref = str(artifact.content_ref or "").strip()
    if not content_ref:
        raise HTTPException(status_code=404, detail="Artifact content is not available")

    path = Path(content_ref).expanduser().resolve()
    workspace_root = get_workspace_root(workspace_id).resolve()
    try:
        path.relative_to(workspace_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Artifact content is outside workspace root") from exc

    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found")

    if path.suffix == ".json":
        content_type = "json"
        content: object = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix == ".md":
        content_type = "markdown"
        content = path.read_text(encoding="utf-8")
    else:
        content_type = "text"
        content = path.read_text(encoding="utf-8")

    return WorkspaceArtifactContentResponse(
        workspace_id=workspace_id,
        artifact_id=artifact.artifact_id,
        artifact_type=artifact.artifact_type.value,
        title=artifact.title,
        content_type=content_type,
        content=content,
        path=str(path),
    )


@router.post("/{workspace_id}/artifacts", response_model=WorkspaceArtifactItem)
async def create_artifact(
    workspace_id: str,
    req: CreateArtifactRequest,
) -> WorkspaceArtifactItem:
    """创建新 artifact。"""
    _require_workspace(workspace_id)

    try:
        at = ArtifactType(req.artifact_type)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid artifact_type: {req.artifact_type}",
        ) from exc

    artifact = WorkspaceArtifact(
        workspace_id=workspace_id,
        task_id=req.task_id,
        artifact_type=at,
        title=req.title,
        content_ref=req.content_ref,
        summary=req.summary,
        tags=req.tags,
        metadata=req.metadata,
    )
    create_workspace_artifact(artifact)
    return _to_item(artifact)


@router.get("/{workspace_id}/reviews", response_model=list)
async def list_workspace_reviews(workspace_id: str) -> list:
    """返回 workspace 下所有 review_feedback artifacts。"""
    _require_workspace(workspace_id)
    artifacts = list_workspace_artifacts(workspace_id, "review_feedback")
    return [
        {
            "artifact_id": a.artifact_id,
            "workspace_id": a.workspace_id,
            "task_id": a.task_id,
            "title": a.title,
            "summary": a.summary,
            "created_at": a.created_at.isoformat(),
            "metadata": a.metadata,
        }
        for a in artifacts
    ]
