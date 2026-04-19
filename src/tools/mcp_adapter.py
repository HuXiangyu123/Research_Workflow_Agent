"""MCP Adapter — Phase 4: MCP Client SDK wrapper.

支持 stdio 和 remote HTTP 两种 transport，
先接 tools，再接 resources/prompts。
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.models.mcp import (
    MCPServerConfig,
    MCPServerTransport,
    MCPToolDescriptor,
    MCPPromptDescriptor,
    MCPResourceDescriptor,
    MCPInvokeKind,
    MCPInvocationRequest,
    MCPInvocationResponse,
)

logger = logging.getLogger(__name__)


# ─── Transport 抽象 ─────────────────────────────────────────────────────────────


class MCPTransport(ABC):
    """MCP 传输层抽象。"""

    @abstractmethod
    async def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def send(self, method: str, params: dict | None = None) -> dict:
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        raise NotImplementedError


class StdioTransport(MCPTransport):
    """通过 stdio 启动并与 MCP server 通信。"""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._proc: subprocess.Popen | None = None
        self._request_id = 0
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        command = self.config.command or ""
        args = list(self.config.args) if self.config.args else []
        env = dict(self.config.env) if self.config.env else {}

        logger.info(f"[MCP Stdio] Starting server: {command} {' '.join(args)}")
        self._proc = subprocess.Popen(
            [command, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**subprocess.os.environ, **env},
            text=False,
        )
        await asyncio.sleep(0.5)  # give server time to start
        logger.info(f"[MCP Stdio] Server started (pid={self._proc.pid})")

    async def send(self, method: str, params: dict | None = None) -> dict:
        if not self._proc or self._proc.poll() is not None:
            raise RuntimeError("MCP server process is not running")

        async with self._lock:
            self._request_id += 1
            payload = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params or {},
            }
            request_str = json.dumps(payload) + "\n"
            self._proc.stdin.write(request_str.encode("utf-8"))
            self._proc.stdin.flush()

            # Read response
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("MCP server closed stdout")
            return json.loads(line.decode("utf-8"))

    async def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=5)
            self._proc = None


class RemoteHttpTransport(MCPTransport):
    """通过 HTTP 与 remote MCP server 通信。"""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._base_url = str(config.url or "")
        self._session: Any = None

    async def start(self) -> None:
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for remote MCP transport: pip install httpx")
        self._session = httpx.AsyncClient(base_url=self._base_url, timeout=30.0)
        logger.info(f"[MCP Remote] Connected to {self._base_url}")

    async def send(self, method: str, params: dict | None = None) -> dict:
        if not self._session:
            raise RuntimeError("MCP HTTP session not started")
        resp = await self._session.post(
            "/mcp",
            json={"method": method, "params": params or {}, "jsonrpc": "2.0", "id": 1},
        )
        resp.raise_for_status()
        return resp.json()

    async def stop(self) -> None:
        if self._session:
            await self._session.aclose()
            self._session = None


# ─── MCP Server 实例 ─────────────────────────────────────────────────────────────


class MCPServerInstance:
    """单个 MCP server 的运行时实例。"""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._transport: MCPTransport | None = None
        self._tools: list[MCPToolDescriptor] = []
        self._prompts: list[MCPPromptDescriptor] = []
        self._resources: list[MCPResourceDescriptor] = []

    @property
    def server_id(self) -> str:
        return self.config.server_id

    @property
    def is_running(self) -> bool:
        return self._transport is not None

    async def start(self) -> None:
        if self._transport:
            return

        if self.config.transport == MCPServerTransport.STDIO:
            self._transport = StdioTransport(self.config)
        else:
            self._transport = RemoteHttpTransport(self.config)

        await self._transport.start()
        await self._discover_capabilities()
        logger.info(
            f"[MCP Server] {self.server_id} started with "
            f"{len(self._tools)} tools, {len(self._prompts)} prompts, "
            f"{len(self._resources)} resources"
        )

    async def _discover_capabilities(self) -> None:
        """发现 server 提供的 tools / prompts / resources。"""
        if not self._transport:
            return

        try:
            resp = await self._transport.send("tools/list")
            self._tools = [
                MCPToolDescriptor(
                    server_id=self.server_id,
                    tool_name=t.get("name", ""),
                    title=t.get("title", t.get("name", "")),
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}),
                    tags=t.get("tags", []),
                )
                for t in resp.get("result", {}).get("tools", [])
            ]
        except Exception as e:
            logger.warning(f"[MCP] Failed to list tools for {self.server_id}: {e}")

        try:
            resp = await self._transport.send("prompts/list")
            self._prompts = [
                MCPPromptDescriptor(
                    server_id=self.server_id,
                    prompt_name=p.get("name", ""),
                    title=p.get("title", p.get("name", "")),
                    description=p.get("description", ""),
                    input_schema=p.get("inputSchema", {}),
                    tags=p.get("tags", []),
                )
                for p in resp.get("result", {}).get("prompts", [])
            ]
        except Exception as e:
            logger.warning(f"[MCP] Failed to list prompts for {self.server_id}: {e}")

        try:
            resp = await self._transport.send("resources/list")
            self._resources = [
                MCPResourceDescriptor(
                    server_id=self.server_id,
                    resource_uri=r.get("uri", ""),
                    title=r.get("title", r.get("uri", "")),
                    description=r.get("description", ""),
                    mime_type=r.get("mimeType"),
                    tags=r.get("tags", []),
                )
                for r in resp.get("result", {}).get("resources", [])
            ]
        except Exception as e:
            logger.warning(f"[MCP] Failed to list resources for {self.server_id}: {e}")

    async def invoke(self, req: MCPInvocationRequest) -> MCPInvocationResponse:
        """调用 MCP server 的 tool / prompt / resource。"""
        if not self._transport:
            await self.start()

        if req.kind == MCPInvokeKind.TOOL:
            result = await self._invoke_tool(req.name_or_uri, req.arguments)
        elif req.kind == MCPInvokeKind.PROMPT:
            result = await self._invoke_prompt(req.name_or_uri, req.arguments)
        else:
            result = await self._invoke_resource(req.name_or_uri, req.arguments)

        return MCPInvocationResponse(
            workspace_id=req.workspace_id,
            task_id=req.task_id,
            server_id=self.server_id,
            kind=req.kind,
            name_or_uri=req.name_or_uri,
            result_summary={"result": result},
        )

    async def _invoke_tool(self, name: str, args: dict) -> Any:
        resp = await self._transport.send("tools/call", {"name": name, "arguments": args})
        return resp.get("result", {})

    async def _invoke_prompt(self, name: str, args: dict) -> Any:
        resp = await self._transport.send("prompts/get", {"name": name, "arguments": args})
        return resp.get("result", {})

    async def _invoke_resource(self, uri: str, args: dict) -> Any:
        resp = await self._transport.send("resources/read", {"uri": uri})
        return resp.get("result", {})

    async def stop(self) -> None:
        if self._transport:
            await self._transport.stop()
            self._transport = None

    @property
    def tools(self) -> list[MCPToolDescriptor]:
        return list(self._tools)

    @property
    def prompts(self) -> list[MCPPromptDescriptor]:
        return list(self._prompts)

    @property
    def resources(self) -> list[MCPResourceDescriptor]:
        return list(self._resources)


# ─── MCP Adapter（全局管理器）──────────────────────────────────────────────


class MCPAdapter:
    """
    全局 MCP 适配器。

    管理多个 MCPServerInstance，支持启动/停止/discover/invoke。
    """

    def __init__(self):
        self._servers: dict[str, MCPServerInstance] = {}
        self._configs: dict[str, MCPServerConfig] = {}

    def register(self, config: MCPServerConfig) -> None:
        """注册 MCP server 配置。"""
        self._configs[config.server_id] = config
        self._servers[config.server_id] = MCPServerInstance(config)
        logger.info(f"[MCP Adapter] Registered server: {config.server_id} ({config.transport.value})")

    def unregister(self, server_id: str) -> None:
        """注销 MCP server。"""
        if server_id in self._servers:
            server = self._servers.pop(server_id)
            asyncio.create_task(server.stop())
        self._configs.pop(server_id, None)

    async def start_server(self, server_id: str) -> None:
        """启动指定 server。"""
        if server_id not in self._servers:
            raise KeyError(f"MCP server not registered: {server_id}")
        await self._servers[server_id].start()

    async def stop_server(self, server_id: str) -> None:
        """停止指定 server。"""
        if server_id in self._servers:
            await self._servers[server_id].stop()

    def get_server(self, server_id: str) -> MCPServerInstance | None:
        return self._servers.get(server_id)

    def list_servers(self) -> list[MCPServerConfig]:
        return list(self._configs.values())

    def get_catalog(
        self,
    ) -> tuple[list[MCPToolDescriptor], list[MCPPromptDescriptor], list[MCPResourceDescriptor]]:
        """返回全局 catalog（所有 server 的 tools/prompts/resources）。"""
        tools: list[MCPToolDescriptor] = []
        prompts: list[MCPPromptDescriptor] = []
        resources: list[MCPResourceDescriptor] = []
        for server in self._servers.values():
            if server.is_running:
                tools.extend(server.tools)
                prompts.extend(server.prompts)
                resources.extend(server.resources)
        return tools, prompts, resources

    async def invoke(self, req: MCPInvocationRequest) -> MCPInvocationResponse:
        """通过 server_id 找到 server 并调用。"""
        server = self._servers.get(req.server_id)
        if not server:
            raise KeyError(f"MCP server not found: {req.server_id}")
        return await server.invoke(req)


# Global singleton
_mcp_adapter: MCPAdapter | None = None


def get_mcp_adapter() -> MCPAdapter:
    global _mcp_adapter
    if _mcp_adapter is None:
        _mcp_adapter = MCPAdapter()
        server_script = (
            Path(__file__).resolve().parent.parent
            / "mcp_servers"
            / "academic_writing_server.py"
        )
        if server_script.is_file():
            _mcp_adapter.register(
                MCPServerConfig(
                    server_id="academic_writing",
                    name="Academic Writing Support",
                    transport=MCPServerTransport.STDIO,
                    command=sys.executable,
                    args=[str(server_script)],
                    env={},
                    enabled=True,
                    workspace_scoped=False,
                )
            )
    return _mcp_adapter
