"""MCP API — Phase 4: list servers, register, catalog, invoke."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.models.mcp import (
    MCPServerConfig,
    MCPServerTransport,
    MCPToolDescriptor,
    MCPPromptDescriptor,
    MCPResourceDescriptor,
    MCPInvocationRequest,
    MCPInvocationResponse,
)

router = APIRouter(prefix="/api/v1/mcp", tags=["mcp"])


class ListServersResponse(BaseModel):
    items: list[MCPServerConfig]


class MCPCatalogResponse(BaseModel):
    tools: list[MCPToolDescriptor] = Field(default_factory=list)
    prompts: list[MCPPromptDescriptor] = Field(default_factory=list)
    resources: list[MCPResourceDescriptor] = Field(default_factory=list)


def _adapter():
    from src.tools.mcp_adapter import get_mcp_adapter
    return get_mcp_adapter()


@router.get("/servers", response_model=ListServersResponse)
async def list_mcp_servers() -> ListServersResponse:
    """返回已注册的 MCP servers。"""
    adapter = _adapter()
    return ListServersResponse(items=adapter.list_servers())


@router.post("/servers", response_model=MCPServerConfig)
async def register_mcp_server(req: MCPServerConfig) -> MCPServerConfig:
    """
    注册 MCP server 配置。

    Note: server 不会立即启动，需要显式调用 /start 或 /invoke。
    """
    adapter = _adapter()
    adapter.register(req)
    return req


@router.post("/servers/{server_id}/start")
async def start_mcp_server(server_id: str) -> dict:
    """启动指定的 MCP server。"""
    adapter = _adapter()
    try:
        await adapter.start_server(server_id)
        return {"status": "started", "server_id": server_id}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Server not registered: {server_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start server: {e}")


@router.post("/servers/{server_id}/stop")
async def stop_mcp_server(server_id: str) -> dict:
    """停止指定的 MCP server。"""
    adapter = _adapter()
    await adapter.stop_server(server_id)
    return {"status": "stopped", "server_id": server_id}


@router.post("/servers/{server_id}/test")
async def test_mcp_server(server_id: str) -> dict:
    """测试 MCP server 连接（启动后调用 tools/list）。"""
    adapter = _adapter()
    server = adapter.get_server(server_id)
    if not server:
        raise HTTPException(status_code=404, detail=f"Server not registered: {server_id}")
    try:
        if not server.is_running:
            await adapter.start_server(server_id)
        tools = server.tools
        return {
            "status": "ok",
            "server_id": server_id,
            "tools_count": len(tools),
            "tools": [{"name": t.tool_name, "description": t.description} for t in tools[:5]],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server test failed: {e}")


@router.get("/catalog", response_model=MCPCatalogResponse)
async def get_mcp_catalog() -> MCPCatalogResponse:
    """返回全局 MCP catalog（所有已启动 server 的 tools/prompts/resources）。"""
    adapter = _adapter()
    tools, prompts, resources = adapter.get_catalog()
    return MCPCatalogResponse(tools=tools, prompts=prompts, resources=resources)


@router.post("/invoke", response_model=MCPInvocationResponse)
async def invoke_mcp(req: MCPInvocationRequest) -> MCPInvocationResponse:
    """
    调用 MCP tool / prompt / resource。

    Phase 4 最小实现：返回占位响应。
    完整实现：通过 MCPAdapter 实际调用远程 server。
    """
    adapter = _adapter()
    try:
        return await adapter.invoke(req)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"MCP server not found: {req.server_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCP invoke failed: {e}")
