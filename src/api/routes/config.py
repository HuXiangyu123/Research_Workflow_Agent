"""Config API — Phase 4: GET/POST /api/v1/config/phase4。"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from src.models.config import Phase4Config
from src.research.agents.supervisor import get_supervisor

router = APIRouter(prefix="/api/v1/config", tags=["config"])


class GetConfigResponse(BaseModel):
    config: Phase4Config


class UpdateConfigRequest(BaseModel):
    config: Phase4Config

@router.get("/phase4", response_model=GetConfigResponse)
async def get_phase4_config() -> GetConfigResponse:
    """返回当前 Phase 4 配置。"""
    return GetConfigResponse(config=get_supervisor().config)


@router.post("/phase4", response_model=GetConfigResponse)
async def update_phase4_config(req: UpdateConfigRequest) -> GetConfigResponse:
    """更新 Phase 4 配置。"""
    supervisor = get_supervisor()
    supervisor.set_config(req.config)
    return GetConfigResponse(config=supervisor.config)
