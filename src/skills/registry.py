"""Skills Registry — Phase 4: 技能注册、发现、懒加载与执行。"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

from src.models.agent import AgentRole
from src.models.skills import (
    SkillBackend,
    SkillManifest,
    SkillRunRequest,
    SkillRunResponse,
    SkillVisibility,
)

logger = logging.getLogger(__name__)


# ─── Backend handler ─────────────────────────────────────────────────────────────


class SkillBackendHandler(ABC):
    """Skill backend 的执行处理器。"""

    @abstractmethod
    async def run(
        self,
        manifest: SkillManifest,
        inputs: dict[str, Any],
        context: dict,
    ) -> dict:
        """执行 skill，返回 dict 形式的结果（用于构建 artifact）。"""
        raise NotImplementedError


class LocalGraphHandler(SkillBackendHandler):
    """LOCAL_GRAPH: 通过 local LangGraph 节点执行（同步或异步均支持）。"""

    def __init__(self, node_registry: dict[str, Callable]):
        self._nodes = node_registry

    async def run(
        self,
        manifest: SkillManifest,
        inputs: dict[str, Any],
        context: dict,
    ) -> dict:
        node_fn = self._nodes.get(manifest.backend_ref.replace("graph:", ""))
        if not node_fn:
            raise KeyError(f"Graph node not found: {manifest.backend_ref}")

        import asyncio
        if asyncio.iscoroutinefunction(node_fn):
            result = await node_fn(inputs)
        else:
            result = node_fn(inputs)
        return result if isinstance(result, dict) else {"result": result}


class LocalFunctionHandler(SkillBackendHandler):
    """LOCAL_FUNCTION: 直接调用 Python 函数（同步或异步均支持）。"""

    def __init__(self, fn_registry: dict[str, Callable]):
        self._fns = fn_registry

    async def run(
        self,
        manifest: SkillManifest,
        inputs: dict[str, Any],
        context: dict,
    ) -> dict:
        fn = self._fns.get(manifest.backend_ref.replace("fn:", ""))
        if not fn:
            raise KeyError(f"Function not found: {manifest.backend_ref}")

        import asyncio
        if asyncio.iscoroutinefunction(fn):
            result = await fn(inputs, context=context)
        else:
            result = fn(inputs, context=context)
        return result if isinstance(result, dict) else {"result": result}


class MCPToolchainHandler(SkillBackendHandler):
    """MCP_TOOLCHAIN: 通过 MCP adapter 执行。"""

    def __init__(self, mcp_adapter: Any):
        self._adapter = mcp_adapter

    async def run(
        self,
        manifest: SkillManifest,
        inputs: dict[str, Any],
        context: dict,
    ) -> dict:
        # backend_ref 格式: "mcp:server_id.tool_name"
        ref = manifest.backend_ref
        if not ref.startswith("mcp:"):
            raise ValueError(f"Invalid MCP ref: {ref}")
        parts = ref[4:].split(".", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid MCP ref format: {ref} (expected 'mcp:server_id.tool_name')")
        server_id, tool_name = parts

        from src.models.mcp import MCPInvokeKind, MCPInvocationRequest
        req = MCPInvocationRequest(
            workspace_id=context.get("workspace_id", ""),
            task_id=context.get("task_id"),
            server_id=server_id,
            kind=MCPInvokeKind.TOOL,
            name_or_uri=tool_name,
            arguments=inputs,
        )
        resp = await self._adapter.invoke(req)
        return resp.result_summary.get("result", resp.result_summary)


class MCPPromptHandler(SkillBackendHandler):
    """MCP_PROMPT: 通过 MCP prompt 获取生成的 prompt 字符串。"""

    def __init__(self, mcp_adapter: Any):
        self._adapter = mcp_adapter

    async def run(
        self,
        manifest: SkillManifest,
        inputs: dict[str, Any],
        context: dict,
    ) -> dict:
        ref = manifest.backend_ref
        if not ref.startswith("prompt:"):
            raise ValueError(f"Invalid prompt ref: {ref}")
        prompt_name = ref[7:]

        from src.models.mcp import MCPInvokeKind, MCPInvocationRequest
        req = MCPInvocationRequest(
            workspace_id=context.get("workspace_id", ""),
            task_id=context.get("task_id"),
            server_id=context.get("_mcp_server_id", ""),
            kind=MCPInvokeKind.PROMPT,
            name_or_uri=prompt_name,
            arguments=inputs,
        )
        resp = await self._adapter.invoke(req)
        return resp.result_summary.get("result", resp.result_summary)


# ─── Skills Registry ─────────────────────────────────────────────────────────────


class SkillsRegistry:
    """
    Skills 注册与执行中心。

    支持懒加载：平时只给模型看 name + description，
    命中后加载完整 SKILL.md，需要时再读 scripts/。
    """

    def __init__(self):
        # 注册表：skill_id -> SkillManifest
        self._manifests: dict[str, SkillManifest] = {}
        # Backend handlers
        self._graph_handler: LocalGraphHandler | None = None
        self._fn_handler: LocalFunctionHandler | None = None
        self._mcp_handler: MCPToolchainHandler | None = None
        self._mcp_prompt_handler: MCPPromptHandler | None = None
        # Skill metadata cache (progressive disclosure)
        self._meta_cache: dict[str, dict] = {}

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, manifest: SkillManifest) -> None:
        """注册 skill manifest。"""
        self._manifests[manifest.skill_id] = manifest
        logger.info(
            f"[SkillsRegistry] Registered: {manifest.skill_id} "
            f"(backend={manifest.backend.value}, agent={manifest.default_agent.value})"
        )

    def register_many(self, manifests: list[SkillManifest]) -> None:
        for m in manifests:
            self.register(m)

    def unregister(self, skill_id: str) -> None:
        self._manifests.pop(skill_id, None)
        self._meta_cache.pop(skill_id, None)

    def discover_from_filesystem(
        self,
        base: str = ".",
        roots: list[str] | None = None,
    ) -> list[SkillManifest]:
        """
        从文件系统发现 skills（.agents/skills / .claude/skills）。

        对应"发现 → 解析 → 目录注入"三环。
        """
        from src.skills.discovery import SkillsDiscovery

        roots = roots or [
            ".agents/skills",
            ".claude/skills",
        ]
        discovery = SkillsDiscovery(roots=roots)
        manifests = discovery.discover(base)
        self.register_many(manifests)
        logger.info(f"[SkillsRegistry] Discovered {len(manifests)} skills from filesystem")
        return manifests

    # ── Discovery ────────────────────────────────────────────────────────

    def get(self, skill_id: str) -> SkillManifest | None:
        return self._manifests.get(skill_id)

    def list_all(self) -> list[SkillManifest]:
        return list(self._manifests.values())

    def list_visible(self, visibility: SkillVisibility | None = None) -> list[SkillManifest]:
        """按可见性过滤。"""
        if visibility is None:
            return self.list_all()
        return [m for m in self._manifests.values() if m.visibility == visibility]

    def list_for_agent(self, role: AgentRole) -> list[SkillManifest]:
        """返回指定 agent 角色的可用 skills。"""
        return [
            m for m in self._manifests.values()
            if m.default_agent == role
            or role.value in m.tags
        ]

    def list_meta(self) -> list[dict]:
        """
        Progressive disclosure: 从缓存的 raw metadata 生成目录（不含完整内容）。

        Returns:
            list of minimal skill metadata (name + description only, no SKILL.md content).
        """
        return [
            {
                "skill_id": m.skill_id,
                "name": m.name,
                "description": m.description,
                "backend": m.backend.value,
                "default_agent": m.default_agent.value,
                "tags": m.tags,
                "visibility": m.visibility.value,
            }
            for m in self._manifests.values()
        ]

    # ── Execution ──────────────────────────────────────────────────────

    def set_handlers(
        self,
        graph_nodes: dict[str, Callable] | None = None,
        functions: dict[str, Callable] | None = None,
        mcp_adapter: Any = None,
    ) -> None:
        """设置 backend handlers（需在运行前配置）。"""
        if graph_nodes:
            self._graph_handler = LocalGraphHandler(graph_nodes)
        if functions:
            self._fn_handler = LocalFunctionHandler(functions)
        if mcp_adapter:
            self._mcp_handler = MCPToolchainHandler(mcp_adapter)
            self._mcp_prompt_handler = MCPPromptHandler(mcp_adapter)

    async def run(self, req: SkillRunRequest, context: dict) -> SkillRunResponse:
        """执行指定 skill。"""
        manifest = self._manifests.get(req.skill_id)
        if not manifest:
            raise KeyError(f"Skill not found: {req.skill_id}")

        handler = self._resolve_handler(manifest)
        result = await handler.run(manifest, req.inputs, context)

        return SkillRunResponse(
            workspace_id=req.workspace_id,
            task_id=req.task_id,
            skill_id=req.skill_id,
            backend=manifest.backend,
            output_artifact_ids=[],
            summary=result.get("summary") or f"Skill {req.skill_id} executed",
            result=result,
        )

    def run_sync(self, req: SkillRunRequest, context: dict) -> SkillRunResponse:
        """同步上下文中执行 skill。"""

        async def _runner() -> SkillRunResponse:
            return await self.run(req, context)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_runner())

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(_runner())).result()

    def _resolve_handler(self, manifest: SkillManifest) -> SkillBackendHandler:
        if manifest.backend == SkillBackend.LOCAL_GRAPH:
            if not self._graph_handler:
                raise RuntimeError("Graph handler not configured")
            return self._graph_handler
        if manifest.backend == SkillBackend.LOCAL_FUNCTION:
            if not self._fn_handler:
                raise RuntimeError("Function handler not configured")
            return self._fn_handler
        if manifest.backend == SkillBackend.MCP_TOOLCHAIN:
            if not self._mcp_handler:
                raise RuntimeError("MCP handler not configured")
            return self._mcp_handler
        if manifest.backend == SkillBackend.MCP_PROMPT:
            if not self._mcp_prompt_handler:
                raise RuntimeError("MCP prompt handler not configured")
            return self._mcp_prompt_handler
        raise ValueError(f"Unknown skill backend: {manifest.backend}")


# Global singleton
_skills_registry: SkillsRegistry | None = None


def get_skills_registry() -> SkillsRegistry:
    global _skills_registry
    if _skills_registry is None:
        _skills_registry = SkillsRegistry()
        # 1. 注册内置 skills（包含 ARIS 风格科研 skills）
        _register_builtin_skills(_skills_registry)
        # 2. 从文件系统发现并注册（支持 .agents/skills / .claude/skills）
        import os as _os
        base = _os.environ.get("SKILLS_SCAN_BASE", ".")
        _skills_registry.discover_from_filesystem(base)
        # 3. 注册 research_skills / MCP handlers
        _register_research_skills_handlers(_skills_registry)
    return _skills_registry


def _workspace_policy_loader_stub(inputs: dict, context: dict) -> dict:
    """Workspace policy loader stub — 加载工作区约束和约定作为上下文。"""
    workspace_id = context.get("workspace_id", "")
    return {
        "summary": f"Loaded workspace policy for {workspace_id}",
        "policy": {
            "naming_convention": "snake_case",
            "citation_format": "apa",
            "report_language": "en",
        },
    }


def _register_research_skills_handlers(registry: SkillsRegistry) -> None:
    """注册 research_skills.py 中的 skill 函数 handler。"""
    from src.tools.mcp_adapter import get_mcp_adapter
    from src.skills.research_skills import (
        lit_review_scanner,
        claim_verification,
        comparison_matrix_builder,
        experiment_replicator,
        writing_scaffold_generator,
    )
    from src.skills.registry import LocalFunctionHandler

    fn_map = {
        "lit_review_scanner": lit_review_scanner,
        "claim_verification": claim_verification,
        "comparison_matrix_builder": comparison_matrix_builder,
        "experiment_replicator": experiment_replicator,
        "writing_scaffold_generator": writing_scaffold_generator,
        # 兼容旧名
        "workspace_policy_loader": _workspace_policy_loader_stub,
    }
    registry.set_handlers(functions=fn_map, mcp_adapter=get_mcp_adapter())


def _register_builtin_skills(registry: SkillsRegistry) -> None:
    """注册内置 skills（包括 ARIS 风格科研 skills）。"""
    from src.models.agent import AgentRole

    # ── 原有 4 个 skills ───────────────────────────────────────────────────────
    registry.register(SkillManifest(
        skill_id="research_lit_scan",
        name="Research Literature Scan",
        description="Multi-source candidate paper scan: arXiv, Semantic Scholar, Google Scholar",
        backend=SkillBackend.LOCAL_FUNCTION,
        visibility=SkillVisibility.BOTH,
        default_agent=AgentRole.RETRIEVER,
        output_artifact_type="rag_result",
        backend_ref="fn:lit_review_scanner",
        tags=["retrieval", "multi-source", "arxiv"],
        input_schema={"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]},
    ))

    registry.register(SkillManifest(
        skill_id="paper_plan_builder",
        name="Paper Plan Builder",
        description="Generate section outline from PaperCards and ComparisonMatrix",
        backend=SkillBackend.LOCAL_FUNCTION,
        visibility=SkillVisibility.BOTH,
        default_agent=AgentRole.ANALYST,
        output_artifact_type="report_outline",
        backend_ref="fn:comparison_matrix_builder",
        tags=["writing", "outline", "comparison"],
        input_schema={"type": "object", "properties": {"topic": {"type": "string"}, "paper_cards": {"type": "array"}}, "required": ["topic"]},
    ))

    registry.register(SkillManifest(
        skill_id="creative_reframe",
        name="Creative Reframe",
        description="Refine topic framing and sub-questions for re-plan",
        backend=SkillBackend.LOCAL_FUNCTION,
        visibility=SkillVisibility.BOTH,
        default_agent=AgentRole.PLANNER,
        output_artifact_type="search_plan",
        backend_ref="fn:creative_reframe",
        tags=["planner", "replan", "creativity"],
        input_schema={"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]},
    ))

    registry.register(SkillManifest(
        skill_id="workspace_policy_skill",
        name="Workspace Policy Skill",
        description="Inject workspace-specific constraints and conventions as context",
        backend=SkillBackend.LOCAL_FUNCTION,
        visibility=SkillVisibility.BOTH,
        default_agent=AgentRole.SUPERVISOR,
        output_artifact_type=None,
        backend_ref="fn:workspace_policy_loader",
        tags=["policy", "workspace", "constraints"],
        input_schema={"type": "object", "properties": {}},
    ))

    # ── ARIS 风格科研 skills ───────────────────────────────────────────────────
    registry.register(SkillManifest(
        skill_id="lit_review_scanner",
        name="Literature Review Scanner",
        description="Multi-source academic literature scan and candidate ranking. Fetches papers from arXiv/Semantic Scholar, deduplicates, and ranks by relevance.",
        backend=SkillBackend.LOCAL_FUNCTION,
        visibility=SkillVisibility.BOTH,
        default_agent=AgentRole.RETRIEVER,
        output_artifact_type="rag_result",
        backend_ref="fn:lit_review_scanner",
        tags=["retrieval", "multi-source", "arxiv", "academic", "research"],
        input_schema={"type": "object", "properties": {"query": {"type": "string", "description": "Literature search query"}, "max_results": {"type": "integer", "description": "Maximum number of results", "default": 20}}, "required": ["query"]},
    ))

    registry.register(SkillManifest(
        skill_id="claim_verification",
        name="Claim Verification",
        description="Verify scientific claims against retrieved evidence. Categorizes claims as grounded / partial / ungrounded / abstained.",
        backend=SkillBackend.LOCAL_FUNCTION,
        visibility=SkillVisibility.BOTH,
        default_agent=AgentRole.REVIEWER,
        output_artifact_type="verified_report",
        backend_ref="fn:claim_verification",
        tags=["verification", "claims", "evidence", "grounding", "review"],
        input_schema={"type": "object", "properties": {"claims": {"type": "array", "description": "Claims to verify"}}, "required": ["claims"]},
    ))

    registry.register(SkillManifest(
        skill_id="comparison_matrix_builder",
        name="Comparison Matrix Builder",
        description="Build a structured comparison matrix from paper cards. Extracts methods, datasets, benchmarks, and limitations for survey writing.",
        backend=SkillBackend.LOCAL_FUNCTION,
        visibility=SkillVisibility.BOTH,
        default_agent=AgentRole.ANALYST,
        output_artifact_type="comparison_matrix",
        backend_ref="fn:comparison_matrix_builder",
        tags=["analysis", "comparison", "papers", "survey", "matrix"],
        input_schema={"type": "object", "properties": {"paper_cards": {"type": "array"}, "compare_dimensions": {"type": "array", "items": {"type": "string"}, "description": "Dimensions to compare"}}, "required": ["paper_cards"]},
    ))

    registry.register(SkillManifest(
        skill_id="experiment_replicator",
        name="Experiment Replicator",
        description="Analyze experimental settings and results from academic papers. Extracts dataset splits, hyperparameters, and evaluation metrics to assess reproducibility.",
        backend=SkillBackend.LOCAL_FUNCTION,
        visibility=SkillVisibility.BOTH,
        default_agent=AgentRole.ANALYST,
        output_artifact_type="experiment_analysis",
        backend_ref="fn:experiment_replicator",
        tags=["experiment", "replication", "analysis", "reproducibility", "datasets"],
        input_schema={"type": "object", "properties": {"paper_id": {"type": "string", "description": "Paper arXiv ID or URL"}}, "required": ["paper_id"]},
    ))

    registry.register(SkillManifest(
        skill_id="writing_scaffold_generator",
        name="Writing Scaffold Generator",
        description="Generate structured writing scaffold for academic survey papers. Produces Title, Abstract, Introduction, and section outlines.",
        backend=SkillBackend.LOCAL_FUNCTION,
        visibility=SkillVisibility.BOTH,
        default_agent=AgentRole.ANALYST,
        output_artifact_type="report_outline",
        backend_ref="fn:writing_scaffold_generator",
        tags=["writing", "scaffold", "outline", "survey", "generation"],
        input_schema={"type": "object", "properties": {"topic": {"type": "string", "description": "Survey topic"}, "desired_length": {"type": "string", "enum": ["short", "medium", "long"], "default": "medium"}}, "required": ["topic"]},
    ))

    registry.register(SkillManifest(
        skill_id="academic_review_writer_prompt",
        name="Academic Review Writer Prompt",
        description="Fetch an academic review-writing rubric and prompt scaffold from the MCP writing support server.",
        backend=SkillBackend.MCP_PROMPT,
        visibility=SkillVisibility.BOTH,
        default_agent=AgentRole.ANALYST,
        output_artifact_type="report_outline",
        backend_ref="prompt:academic_review_writer",
        tags=["writing", "mcp", "prompt", "survey", "review"],
        input_schema={
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "time_range": {"type": "string"},
                "focus_dimensions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["topic"],
        },
    ))
