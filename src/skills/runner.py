"""Skills Runner — Phase 4: 懒加载 + scripts/references/assets 执行。"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Any

from src.models.skills import (
    SkillBackend,
    SkillManifest,
    SkillRunRequest,
    SkillRunResponse,
)
from src.skills.discovery import SkillsDiscovery, build_skill_manifest

logger = logging.getLogger(__name__)


# ─── Script runner ─────────────────────────────────────────────────────────────


class ScriptRunner:
    """执行 skill scripts/ 下的脚本文件。"""

    def __init__(self, base_dir: str | Path | None = None):
        self._base_dir = Path(base_dir) if base_dir else Path(".")

    def run(
        self,
        script_path: str,
        args: dict[str, Any],
        context: dict,
    ) -> dict:
        """
        执行单个脚本文件。

        支持类型：
        - .py  → python3 解释执行
        - .sh  → bash 执行
        - .js  → node 执行
        - 其他  → 当作命令参数传递

        Args:
            script_path: 相对路径（如 "search.py" 或 "search.py --arg val"）
            args: 传递给脚本的参数（JSON 序列化传入）
            context: 执行上下文（包含 workspace_id, task_id 等）
        """
        import json, tempfile, os, shlex

        # Resolve script path
        script_file = (self._base_dir / script_path).resolve()
        if not script_file.exists():
            raise FileNotFoundError(f"Script not found: {script_file}")

        suffix = script_file.suffix.lower()
        ext_map = {
            ".py": ["python3"],
            ".sh": ["bash"],
            ".js": ["node"],
        }
        cmd_parts = ext_map.get(suffix, [])
        cmd_parts.append(str(script_file))

        # Serialize args to JSON and pass via stdin or temp file
        args_json = json.dumps({**args, **context}, ensure_ascii=False)

        try:
            if suffix == ".py":
                # python3 -c "import json,sys; args=json.load(sys.stdin); exec(open('script.py').read())"
                encoded = args_json.replace("'", "\\'")
                full_cmd = cmd_parts + ["-c", f"import json,sys; data=json.load(sys.stdin); exec(open('{script_file}').read())"]
                proc = subprocess.run(
                    full_cmd,
                    input=args_json.encode("utf-8"),
                    capture_output=True,
                    timeout=60,
                )
            else:
                # Write args to temp JSON and pass as first arg
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                    json.dump({**args, **context}, f)
                    tmp_path = f.name
                try:
                    proc = subprocess.run(
                        cmd_parts + [tmp_path],
                        capture_output=True,
                        timeout=60,
                    )
                finally:
                    os.unlink(tmp_path)

            if proc.returncode != 0:
                logger.warning(f"[ScriptRunner] script {script_path} failed: {proc.stderr.decode()}")
                return {"error": proc.stderr.decode(), "stdout": proc.stdout.decode()}
            return {"stdout": proc.stdout.decode()}

        except subprocess.TimeoutExpired:
            logger.error(f"[ScriptRunner] script {script_path} timed out")
            return {"error": "script timed out after 60s"}
        except Exception as exc:
            logger.exception(f"[ScriptRunner] script {script_path} failed: {exc}")
            return {"error": str(exc)}


# ─── Skills Runner ─────────────────────────────────────────────────────────────


class SkillsRunner:
    """
    Skills 执行引擎。

    实现懒加载：
    - list_meta() 时不加载内容
    - load_content(skill_id) 时才读 SKILL.md
    - run(skill_id) 时才执行 scripts/ 或 backend
    """

    def __init__(
        self,
        discovery: SkillsDiscovery | None = None,
        script_base: str | Path | None = None,
    ):
        self._discovery = discovery or SkillsDiscovery()
        self._script_runner = ScriptRunner(base_dir=script_base)
        self._handlers: dict[SkillBackend, Any] = {}

    def register_handler(self, backend: SkillBackend, handler: Any) -> None:
        """注册 backend handler（如 MCP、local function）。"""
        self._handlers[backend] = handler

    async def run(
        self,
        req: SkillRunRequest,
        context: dict,
    ) -> SkillRunResponse:
        """
        执行 skill（懒加载完整内容）。

        执行顺序：
        1. 解析 manifest（从缓存或 disk）
        2. 按 backend 选择 handler
        3. 若 backend=local_script，加载 scripts/ 并执行
        4. 返回 SkillRunResponse
        """
        manifest = self._discovery._cache.get(req.skill_id)
        if not manifest:
            raise KeyError(f"Skill not discovered: {req.skill_id}")

        # Build manifest from raw
        skill_dir = Path(manifest.get("_root", "."))
        skill_manifest = build_skill_manifest(req.skill_id, manifest, skill_dir)

        # Lazy load content for logging/debugging
        content = self._discovery.load_content(req.skill_id)
        logger.info(
            f"[SkillsRunner] Running skill={req.skill_id} "
            f"backend={skill_manifest.backend.value} "
            f"(SKILL.md loaded: {len(content)} chars)"
        )

        # Log available assets
        assets = self._discovery.load_assets(req.skill_id)
        for asset_type, files in assets.items():
            if files:
                logger.debug(f"[SkillsRunner] {req.skill_id}/{asset_type}: {files}")

        # Execute via backend
        result = await self._execute(skill_manifest, req.inputs, context)

        return SkillRunResponse(
            workspace_id=req.workspace_id,
            task_id=req.task_id,
            skill_id=req.skill_id,
            backend=skill_manifest.backend,
            output_artifact_ids=[],
            trace_refs=[],
            summary=result.get("summary") or result.get("stdout") or f"Skill {req.skill_id} executed",
            result=result,
        )

    async def _execute(
        self,
        manifest: SkillManifest,
        inputs: dict,
        context: dict,
    ) -> dict:
        """按 backend 选择执行路径。"""
        backend = manifest.backend
        backend_ref = manifest.backend_ref

        # ── SCRIPT backend ─────────────────────────────────────────
        if backend_ref.startswith("script:"):
            script_path = backend_ref[7:]  # strip "script:"
            result = self._script_runner.run(script_path, inputs, context)
            return result

        # ── MCP_TOOLCHAIN backend ─────────────────────────────────────
        if backend == SkillBackend.MCP_TOOLCHAIN:
            handler = self._handlers.get(SkillBackend.MCP_TOOLCHAIN)
            if not handler:
                raise RuntimeError("MCP handler not configured")
            return await handler.run(manifest, inputs, context)

        # ── MCP_PROMPT backend ───────────────────────────────────────
        if backend == SkillBackend.MCP_PROMPT:
            handler = self._handlers.get(SkillBackend.MCP_PROMPT)
            if not handler:
                raise RuntimeError("MCP prompt handler not configured")
            return await handler.run(manifest, inputs, context)

        # ── LOCAL_FUNCTION backend ───────────────────────────────────
        if backend == SkillBackend.LOCAL_FUNCTION:
            handler = self._handlers.get(SkillBackend.LOCAL_FUNCTION)
            if not handler:
                raise RuntimeError("Local function handler not configured")
            return await handler.run(manifest, inputs, context)

        # ── LOCAL_GRAPH backend ──────────────────────────────────────
        if backend == SkillBackend.LOCAL_GRAPH:
            handler = self._handlers.get(SkillBackend.LOCAL_GRAPH)
            if not handler:
                raise RuntimeError("Local graph handler not configured")
            return await handler.run(manifest, inputs, context)

        raise ValueError(f"Unknown backend: {backend}")

    # ─── Progressive disclosure helpers ────────────────────────────────────

    def list_meta(self) -> list[dict]:
        """目录注入：只返回 name + description（不加载内容）。"""
        return self._discovery._cache_to_meta()

    def load_content(self, skill_id: str) -> str:
        """懒加载 SKILL.md 完整内容。"""
        return self._discovery.load_content(skill_id)

    def load_assets(self, skill_id: str) -> dict[str, list[str]]:
        """懒加载 skill assets（scripts / references / assets）。"""
        return self._discovery.load_assets(skill_id)
