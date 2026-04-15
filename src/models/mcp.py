"""MCP models — Phase 4: MCPServer, MCPInvocation."""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class MCPServerTransport(str, Enum):
    STDIO = "stdio"
    REMOTE = "remote"


class MCPServerConfig(BaseModel):
    """MCP Server 配置（用于注册和管理）。"""

    model_config = ConfigDict(extra="forbid")

    server_id: str
    name: str
    transport: MCPServerTransport
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: HttpUrl | None = None
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    workspace_scoped: bool = False
    auth_ref: str | None = None


class MCPCapability(str, Enum):
    TOOLS = "tools"
    RESOURCES = "resources"
    PROMPTS = "prompts"
    APPS = "apps"


class MCPToolDescriptor(BaseModel):
    """MCP Tool 的描述（用于 catalog 展示）。"""

    model_config = ConfigDict(extra="forbid")

    server_id: str
    tool_name: str
    title: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    requires_approval: bool = True
    tags: list[str] = Field(default_factory=list)


class MCPPromptDescriptor(BaseModel):
    """MCP Prompt 的描述。"""

    model_config = ConfigDict(extra="forbid")

    server_id: str
    prompt_name: str
    title: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class MCPResourceDescriptor(BaseModel):
    """MCP Resource 的描述。"""

    model_config = ConfigDict(extra="forbid")

    server_id: str
    resource_uri: str
    title: str
    description: str
    mime_type: str | None = None
    tags: list[str] = Field(default_factory=list)


class MCPInvokeKind(str, Enum):
    TOOL = "tool"
    PROMPT = "prompt"
    RESOURCE = "resource"


class MCPInvocationRequest(BaseModel):
    """MCP 调用请求。"""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    task_id: str | None = None
    server_id: str
    kind: MCPInvokeKind
    name_or_uri: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    require_user_approval: bool = True


class MCPInvocationResponse(BaseModel):
    """MCP 调用响应。"""

    model_config = ConfigDict(extra="forbid")

    invocation_id: str = Field(default_factory=lambda: f"mcp_{uuid4().hex[:12]}")
    workspace_id: str
    task_id: str | None = None
    server_id: str
    kind: MCPInvokeKind
    name_or_uri: str
    result_summary: dict[str, Any] = Field(default_factory=dict)
    trace_refs: list[str] = Field(default_factory=list)
