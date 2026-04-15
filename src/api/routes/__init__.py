"""API routes — 统一路由注册。"""
from __future__ import annotations

from fastapi import APIRouter

from src.api.routes.tasks import router as tasks_router
from src.api.routes.corpus_search import router as corpus_search_router

api_router = APIRouter()

api_router.include_router(tasks_router)
api_router.include_router(corpus_search_router)
