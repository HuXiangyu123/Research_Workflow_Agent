"""Skills Discovery — Phase 4: 扫描 .agents/skills 和 .claude/skills 目录。

实现 6 环节中的第 1 环：发现 → 解析 → 目录注入。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterator

from src.models.agent import AgentRole
from src.models.skills import (
    SkillBackend,
    SkillManifest,
    SkillVisibility,
)

logger = logging.getLogger(__name__)

# ─── Skill directory conventions ─────────────────────────────────────────────


class SkillPaths:
    """
    支持的 skill 根目录规范。

    支持：
    - .agents/skills/      (Codex / Agent Skills 格式)
    - .claude/skills/      (Claude Code 格式)
    - .claude/plugins/*/skills/  (官方插件格式)
    """

    DEFAULT_ROOTS = [
        ".agents/skills",
        ".claude/skills",
    ]

    PLUGIN_ROOT = ".claude/plugins"

    SKILL_FILE = "SKILL.md"
    MANIFEST_FILE = "skill.json"

    @classmethod
    def get_all_skill_dirs(cls, base: str | Path = ".") -> Iterator[Path]:
        """递归扫描所有 skill 目录。"""
        base = Path(base)
        for root in cls.DEFAULT_ROOTS:
            root_path = base / root
            if root_path.is_dir():
                yield from cls._scan_skill_dir(root_path)

        # Scan plugin skills
        plugin_root = base / cls.PLUGIN_ROOT
        if plugin_root.is_dir():
            for plugin_dir in plugin_root.iterdir():
                if not plugin_dir.is_dir():
                    continue
                plugin_skills = plugin_dir / "skills"
                if plugin_skills.is_dir():
                    yield from cls._scan_skill_dir(plugin_skills)

    @classmethod
    def _scan_skill_dir(cls, root: Path) -> Iterator[Path]:
        """扫描单个 skill 根目录。"""
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            skill_md = entry / cls.SKILL_FILE
            skill_json = entry / cls.MANIFEST_FILE
            if skill_md.exists() or skill_json.exists():
                yield entry


# ─── SKILL.md parser ────────────────────────────────────────────────────────


class SkillMetadataParser:
    """
    懒加载解析 SKILL.md 的 frontmatter / YAML 头。

    只在需要时读取文件，不会一次性把所有 skill 内容都加载进内存。
    """

    @classmethod
    def parse_skill_dir(cls, skill_dir: Path | str) -> dict[str, Any]:
        """
        解析 skill 目录，返回元数据。

        返回值用于构建 SkillManifest。
        """
        skill_dir = Path(skill_dir)
        skill_md = skill_dir / SkillPaths.SKILL_FILE
        skill_json = skill_dir / SkillPaths.MANIFEST_FILE

        # Priority: JSON manifest > SKILL.md frontmatter
        if skill_json.exists():
            return cls._parse_json(skill_json)
        elif skill_md.exists():
            return cls._parse_skill_md(skill_md, skill_dir)
        else:
            logger.warning(f"[Discovery] No SKILL.md or skill.json in {skill_dir}")
            return {}

    @classmethod
    def _parse_json(cls, path: Path) -> dict[str, Any]:
        """解析 skill.json manifest。"""
        import json
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def _parse_skill_md(cls, path: Path, skill_dir: Path) -> dict[str, Any]:
        """
        解析 SKILL.md frontmatter。

        提取 YAML frontmatter 作为 skill metadata。
        支持格式：
            ---
            name: skill_name
            description: ...
            backend: local_graph
            default_agent: planner
            ---
            # SKILL.md content...
        """
        content = path.read_text(encoding="utf-8")
        frontmatter = cls._extract_frontmatter(content)

        # Infer from directory name if name not in frontmatter
        if "name" not in frontmatter:
            frontmatter["name"] = skill_dir.name.replace("-", "_").replace(" ", "_")

        # Resolve relative paths
        skill_id = frontmatter.get("name", skill_dir.name)
        frontmatter["skill_id"] = skill_id
        frontmatter["_root"] = str(skill_dir)

        return frontmatter

    @classmethod
    def _extract_frontmatter(cls, content: str) -> dict[str, Any]:
        """提取 YAML frontmatter。"""
        stripped = content.lstrip()
        if not stripped.startswith("---"):
            return {}

        end = stripped.find("\n---", 4)
        if end == -1:
            return {}

        yaml_block = stripped[3:end].strip()
        try:
            import yaml
            return yaml.safe_load(yaml_block) or {}
        except ImportError:
            return cls._parse_frontmatter_manual(yaml_block)
        except Exception:
            logger.warning(f"[Discovery] Failed to parse YAML frontmatter")
            return cls._parse_frontmatter_manual(yaml_block)

    @classmethod
    def _parse_frontmatter_manual(cls, yaml_block: str) -> dict[str, Any]:
        """手动解析简单 YAML（无需 yaml 库）。"""
        result: dict[str, Any] = {}
        current_key = ""
        current_list: list[str] = []
        in_list = False

        for line in yaml_block.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # List item
            if stripped.startswith("- "):
                if not in_list:
                    in_list = True
                    current_list = []
                current_list.append(stripped[1:].strip().strip('"').strip("'"))
                result[current_key] = current_list
                continue

            # Key: value
            in_list = False
            if ": " in stripped:
                key, val = stripped.split(": ", 1)
                current_key = key.strip()
                val = val.strip().strip('"').strip("'")
                if val.lower() in ("true", "false"):
                    result[current_key] = val.lower() == "true"
                elif val.isdigit():
                    result[current_key] = int(val)
                else:
                    result[current_key] = val
            elif stripped.endswith(":") and not stripped.startswith("-"):
                current_key = stripped[:-1].strip()
                in_list = False

        return result

    @classmethod
    def load_skill_content(cls, skill_dir: Path | str) -> str:
        """按需加载完整 SKILL.md 内容（懒加载）。"""
        skill_dir = Path(skill_dir)
        skill_md = skill_dir / SkillPaths.SKILL_FILE
        if skill_md.exists():
            return skill_md.read_text(encoding="utf-8")
        return ""

    @classmethod
    def list_skill_assets(cls, skill_dir: Path | str) -> dict[str, list[str]]:
        """列出 skill 目录下的所有 assets（scripts/references/assets）。"""
        skill_dir = Path(skill_dir)
        assets: dict[str, list[str]] = {
            "scripts": [],
            "references": [],
            "assets": [],
        }
        for subdir in ["scripts", "references", "assets"]:
            subdir_path = skill_dir / subdir
            if subdir_path.is_dir():
                for f in subdir_path.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(subdir_path).as_posix()
                        assets[subdir].append(rel)
        return assets


# ─── Skill manifest builder ────────────────────────────────────────────────────


def _parse_visibility(v: str | None) -> SkillVisibility:
    if not v:
        return SkillVisibility.BOTH
    try:
        return SkillVisibility(v)
    except ValueError:
        return SkillVisibility.BOTH


def _parse_backend(v: str | None) -> SkillBackend:
    if not v:
        return SkillBackend.LOCAL_FUNCTION
    try:
        return SkillBackend(v)
    except ValueError:
        return SkillBackend.LOCAL_FUNCTION


def _parse_agent_role(v: str | None) -> AgentRole:
    if not v:
        return AgentRole.PLANNER
    try:
        return AgentRole(v)
    except ValueError:
        return AgentRole.PLANNER


def build_skill_manifest(
    skill_id: str,
    raw: dict[str, Any],
    root_dir: Path,
) -> SkillManifest:
    """
    将解析出的 raw metadata 构建为 SkillManifest。

    支持从 SKILL.md frontmatter 或 skill.json 构造。
    """
    # Resolve backend_ref: use explicit backend_ref or infer from structure
    backend_ref = raw.get("backend_ref") or raw.get("backend_ref")
    if not backend_ref:
        # Infer: check if scripts/ directory exists
        scripts_dir = root_dir / "scripts"
        if scripts_dir.is_dir():
            scripts_files = list(scripts_dir.iterdir())
            if scripts_files:
                # Use first script as backend_ref
                backend_ref = f"script:{skill_id}/{scripts_files[0].name}"
            else:
                backend_ref = f"fn:{skill_id}"
        else:
            backend_ref = f"fn:{skill_id}"

    return SkillManifest(
        skill_id=skill_id,
        name=raw.get("name", skill_id),
        description=raw.get("description", ""),
        backend=_parse_backend(raw.get("backend")),
        visibility=_parse_visibility(raw.get("visibility")),
        default_agent=_parse_agent_role(raw.get("default_agent")),
        tags=raw.get("tags", []) or [],
        input_schema=raw.get("input_schema", {}),
        output_artifact_type=raw.get("output_artifact_type"),
        backend_ref=backend_ref,
    )


# ─── Discovery service ────────────────────────────────────────────────────────


class SkillsDiscovery:
    """
    Skills 发现服务。

    扫描指定根目录（支持多个 convention），
    按需懒加载每个 skill 的 SKILL.md 内容。
    """

    def __init__(self, roots: list[str] | None = None):
        self._roots = [Path(r) for r in (roots or SkillPaths.DEFAULT_ROOTS)]
        self._cache: dict[str, dict[str, Any]] = {}  # skill_id -> raw metadata
        self._content_cache: dict[str, str] = {}  # skill_id -> full SKILL.md content

    def discover(self, base: str | Path = ".") -> list[SkillManifest]:
        """
        扫描所有 skill 根目录，返回 SkillManifest 列表。
        """
        manifests: list[SkillManifest] = []
        base = Path(base)

        for skill_dir in SkillPaths.get_all_skill_dirs(base):
            try:
                raw = SkillMetadataParser.parse_skill_dir(skill_dir)
                skill_id = raw.get("skill_id") or skill_dir.name
                if not skill_id:
                    continue

                manifest = build_skill_manifest(skill_id, raw, skill_dir)
                self._cache[skill_id] = raw
                manifests.append(manifest)
                logger.debug(f"[Discovery] Found skill: {skill_id} at {skill_dir}")

            except Exception as exc:
                logger.warning(f"[Discovery] Failed to parse {skill_dir}: {exc}")

        return manifests

    def load_content(self, skill_id: str) -> str:
        """
        懒加载完整 SKILL.md 内容。

        只在调用时读取文件，不会一次性全部加载。
        """
        if skill_id in self._content_cache:
            return self._content_cache[skill_id]

        raw = self._cache.get(skill_id, {})
        root_str = raw.get("_root", "")
        if not root_str:
            return ""

        content = SkillMetadataParser.load_skill_content(Path(root_str))
        self._content_cache[skill_id] = content
        return content

    def load_assets(self, skill_id: str) -> dict[str, list[str]]:
        """列出 skill 的 scripts / references / assets 文件。"""
        raw = self._cache.get(skill_id, {})
        root_str = raw.get("_root", "")
        if not root_str:
            return {"scripts": [], "references": [], "assets": []}
        return SkillMetadataParser.list_skill_assets(Path(root_str))

    def discover_from_home(self) -> list[SkillManifest]:
        """从用户 home 目录扫描 skill 目录。"""
        home = Path.home()
        all_manifests: list[SkillManifest] = []

        for root in self._roots:
            root_abs = home / root
            if root_abs.exists():
                disc = SkillsDiscovery()
                all_manifests.extend(disc.discover(str(root_abs)))

        return all_manifests

    def _cache_to_meta(self) -> list[dict]:
        """
        Progressive disclosure: 从缓存的 raw metadata 生成目录（不含完整内容）。
        """
        return [
            {
                "skill_id": raw.get("skill_id", ""),
                "name": raw.get("name", ""),
                "description": raw.get("description", ""),
                "backend": raw.get("backend", "local_function"),
                "default_agent": raw.get("default_agent", "planner"),
                "tags": raw.get("tags", []),
                "visibility": raw.get("visibility", "both"),
            }
            for raw in self._cache.values()
        ]
