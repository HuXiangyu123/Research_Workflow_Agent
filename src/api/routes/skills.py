"""Skills API — Phase 4: list skills, get skill, run skill."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.models.agent import AgentRole
from src.models.skills import (
    SkillBackend,
    SkillManifest,
    SkillRunRequest,
    SkillRunResponse,
    SkillVisibility,
)

router = APIRouter(prefix="/api/v1/skills", tags=["skills"])


class ListSkillsResponse(BaseModel):
    items: list[dict[str, Any]]


class GetSkillResponse(BaseModel):
    manifest: SkillManifest


# Lazy-load registry to avoid circular imports at module load time
def _registry() -> Any:
    from src.skills.registry import get_skills_registry
    return get_skills_registry()


@router.get("", response_model=ListSkillsResponse)
async def list_skills() -> ListSkillsResponse:
    """返回所有注册的 skills（progressive disclosure 目录）。"""
    registry = _registry()
    return ListSkillsResponse(items=registry.list_meta())


@router.get("/{skill_id}", response_model=GetSkillResponse)
async def get_skill(skill_id: str) -> GetSkillResponse:
    """返回单个 skill 的完整 manifest。"""
    registry = _registry()
    manifest = registry.get(skill_id)
    if not manifest:
        raise HTTPException(status_code=404, detail=f"Skill not found: {skill_id}")
    return GetSkillResponse(manifest=manifest)


@router.post("/run", response_model=SkillRunResponse)
async def run_skill(req: SkillRunRequest) -> SkillRunResponse:
    """
    执行指定 skill。

    支持 backend: local_graph / local_function / mcp_toolchain / mcp_prompt。
    """
    registry = _registry()
    try:
        resp = await registry.run(req, {"workspace_id": req.workspace_id})
        return resp
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Skill execution failed: {e}")
