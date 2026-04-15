"""Workspaces API — Phase 3: artifact 面板数据接口。"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.models.workspace import ArtifactType, WorkspaceArtifact

router = APIRouter(prefix="/api/v1/workspaces", tags=["workspaces"])


# ─── In-memory store (replace with DB in production) ─────────────────────────


_workspaces: dict[str, dict] = {}
_artifact_store: dict[str, list[WorkspaceArtifact]] = {}


def get_or_create_workspace(workspace_id: str) -> dict:
    if workspace_id not in _workspaces:
        _workspaces[workspace_id] = {
            "workspace_id": workspace_id,
            "status": "active",
            "current_stage": None,
            "warnings": [],
        }
    return _workspaces[workspace_id]


def list_workspace_artifacts(
    workspace_id: str,
    artifact_type: str | None = None,
) -> list[WorkspaceArtifact]:
    artifacts = _artifact_store.get(workspace_id, [])
    if artifact_type:
        try:
            at = ArtifactType(artifact_type)
            artifacts = [a for a in artifacts if a.artifact_type == at]
        except ValueError:
            pass
    return artifacts


# ─── Request/Response Models ────────────────────────────────────────────────


class WorkspaceSummaryResponse(BaseModel):
    workspace_id: str
    status: str
    current_stage: str | None = None
    warnings: list[str] = Field(default_factory=list)
    artifact_count: int = 0


class WorkspaceArtifactItem(BaseModel):
    artifact_id: str
    artifact_type: str
    title: str
    status: str
    created_at: str
    created_by_node: str | None = None
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)


class WorkspaceArtifactsResponse(BaseModel):
    workspace_id: str
    items: list[WorkspaceArtifactItem]
    total: int


class CreateArtifactRequest(BaseModel):
    artifact_type: str
    title: str
    task_id: str | None = None
    content_ref: str | None = None
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


# ─── Routes ───────────────────────────────────────────────────────────────


@router.get("/{workspace_id}", response_model=WorkspaceSummaryResponse)
async def get_workspace(workspace_id: str) -> WorkspaceSummaryResponse:
    """返回 workspace 概览（状态、stage、warnings）。"""
    ws = get_or_create_workspace(workspace_id)
    artifacts = list_workspace_artifacts(workspace_id)
    return WorkspaceSummaryResponse(
        workspace_id=workspace_id,
        status=ws.get("status", "active"),
        current_stage=ws.get("current_stage"),
        warnings=ws.get("warnings", []),
        artifact_count=len(artifacts),
    )


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
    artifacts = list_workspace_artifacts(workspace_id, artifact_type)
    items = [
        WorkspaceArtifactItem(
            artifact_id=a.artifact_id,
            artifact_type=a.artifact_type.value,
            title=a.title,
            status=a.status,
            created_at=a.created_at.isoformat(),
            created_by_node=a.created_by_node,
            summary=a.summary,
            tags=a.tags,
        )
        for a in artifacts
    ]
    return WorkspaceArtifactsResponse(
        workspace_id=workspace_id,
        items=items,
        total=len(items),
    )


@router.get("/{workspace_id}/artifacts/{artifact_id}", response_model=WorkspaceArtifactItem)
async def get_artifact(workspace_id: str, artifact_id: str) -> WorkspaceArtifactItem:
    """返回单个 artifact 的详细信息。"""
    artifacts = list_workspace_artifacts(workspace_id)
    for a in artifacts:
        if a.artifact_id == artifact_id:
            return WorkspaceArtifactItem(
                artifact_id=a.artifact_id,
                artifact_type=a.artifact_type.value,
                title=a.title,
                status=a.status,
                created_at=a.created_at.isoformat(),
                created_by_node=a.created_by_node,
                summary=a.summary,
                tags=a.tags,
            )
    raise HTTPException(status_code=404, detail="Artifact not found")


@router.post("/{workspace_id}/artifacts", response_model=WorkspaceArtifactItem)
async def create_artifact(
    workspace_id: str,
    req: CreateArtifactRequest,
) -> WorkspaceArtifactItem:
    """创建新 artifact。"""
    get_or_create_workspace(workspace_id)

    try:
        at = ArtifactType(req.artifact_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid artifact_type: {req.artifact_type}",
        )

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

    if workspace_id not in _artifact_store:
        _artifact_store[workspace_id] = []
    _artifact_store[workspace_id].append(artifact)

    return WorkspaceArtifactItem(
        artifact_id=artifact.artifact_id,
        artifact_type=artifact.artifact_type.value,
        title=artifact.title,
        status=artifact.status,
        created_at=artifact.created_at.isoformat(),
        created_by_node=artifact.created_by_node,
        summary=artifact.summary,
        tags=artifact.tags,
    )


@router.get("/{workspace_id}/reviews", response_model=list)
async def list_workspace_reviews(workspace_id: str) -> list:
    """返回 workspace 下所有 review_feedback artifacts。"""
    artifacts = list_workspace_artifacts(workspace_id, "review_feedback")
    return [
        {
            "artifact_id": a.artifact_id,
            "title": a.title,
            "summary": a.summary,
            "created_at": a.created_at.isoformat(),
            "metadata": a.metadata,
        }
        for a in artifacts
    ]
