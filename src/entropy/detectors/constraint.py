"""Entropy 检测器集合 — 检测文档漂移、死代码、约束违反等。

设计文档：docs/features_oncoming/entropy-management.md
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.entropy.scanner import DriftReport, DriftType, Severity

logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent


class ConstraintViolationDetector:
    """检测对项目硬约束的违反。"""

    # 硬约束定义
    HARD_CONSTRAINTS: list[dict[str, Any]] = [
        {
            "id": "no_sqlite",
            "description": "禁止引入 SQLite 数据库或 sqlite:/// URL",
            "check_patterns": [r"sqlite:///"],
            "file_patterns": ["*.py", "*.yaml", "*.yml", "*.json"],
            "severity": Severity.CRITICAL,
        },
        {
            "id": "no_sqlite_file",
            "description": "禁止创建 .sqlite 文件",
            "check_patterns": [r"\.sqlite\b"],
            "file_patterns": ["*.py", "*.sh", "*.md"],
            "severity": Severity.CRITICAL,
        },
        {
            "id": "explicit_dotenv",
            "description": "脚本和测试必须显式 load_dotenv('.env')",
            "check_patterns": [r"open.*\.env", r"load_dotenv.*\.env"],
            "file_patterns": ["tests/**/*.py"],
            "severity": Severity.WARNING,
        },
    ]

    def scan(self, files: list[str] | None = None) -> list[DriftReport]:
        """扫描约束违反。"""
        reports = []
        root = PROJECT_ROOT

        if files:
            paths = [root / f for f in files if not f.startswith("/")]
        else:
            paths = list(root.rglob("src/**/*.py"))

        for path in paths:
            if not path.is_file():
                continue

            for constraint in self.HARD_CONSTRAINTS:
                violations = self._check_constraint(path, constraint)
                for violation in violations:
                    reports.append(violation)

        return reports

    def _check_constraint(
        self, path: Path, constraint: dict[str, Any]
    ) -> list[DriftReport]:
        """检查单个约束。"""
        reports = []
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return reports

        for pattern in constraint.get("check_patterns", []):
            if re.search(pattern, content):
                reports.append(
                    DriftReport(
                        drift_type=DriftType.UNENFORCED_CONSTRAINT,
                        source_file=str(path.relative_to(PROJECT_ROOT)),
                        expected_state=f"不包含 '{pattern}'",
                        actual_state=f"包含违反约束的代码: {pattern}",
                        severity=constraint["severity"],
                        fix_suggestion=f"约束 {constraint['id']}: {constraint['description']}",
                        auto_fixable=False,
                    )
                )
                break  # 一个约束只报告一次

        return reports


class DeadCodeDetector:
    """检测死代码和孤立文件。"""

    def scan(self, files: list[str] | None = None) -> list[DriftReport]:
        """扫描死代码问题。"""
        reports = []
        root = PROJECT_ROOT

        # 1. 检查 supervisor 中的节点引用是否都存在
        supervisor_path = root / "src/research/agents/supervisor.py"
        if supervisor_path.exists():
            reports.extend(self._check_supervisor_node_refs(supervisor_path))

        # 2. 检查孤立文件（没有被任何地方引用的 Python 文件）
        reports.extend(self._check_orphaned_files(root))

        return reports

    def _check_supervisor_node_refs(self, supervisor_path: Path, files: list[str] | None = None) -> list[DriftReport]:
        """检查 supervisor 中引用的节点是否都存在。"""
        reports = []
        root = PROJECT_ROOT

        try:
            content = supervisor_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return reports

        # 提取 LEGACY_NODE_TARGETS 和 V2_AGENT_TARGETS
        legacy_nodes = self._extract_dict_values(content, "LEGACY_NODE_TARGETS")
        v2_nodes = self._extract_dict_values(content, "V2_AGENT_TARGETS")

        # 检查 legacy 节点文件
        nodes_dir = root / "src/research/graph/nodes"
        if nodes_dir.exists():
            for node in legacy_nodes:
                node_file = nodes_dir / f"{node}.py"
                if not node_file.exists():
                    reports.append(
                        DriftReport(
                            drift_type=DriftType.MISSING_NODE_FILE,
                            source_file=str(supervisor_path.relative_to(root)),
                            expected_state=f"节点文件 {node}.py 存在",
                            actual_state=f"节点文件不存在",
                            severity=Severity.CRITICAL,
                            fix_suggestion=f"移除 LEGACY_NODE_TARGETS 中对 {node} 的引用，或创建节点文件",
                            auto_fixable=False,
                        )
                    )

        # 检查 v2 agent 文件
        agents_dir = root / "src/research/agents"
        if agents_dir.exists():
            for node in v2_nodes:
                agent_file = agents_dir / f"{node}_agent.py"
                if not agent_file.exists():
                    reports.append(
                        DriftReport(
                            drift_type=DriftType.MISSING_NODE_FILE,
                            source_file=str(supervisor_path.relative_to(root)),
                            expected_state=f"Agent 文件 {node}_agent.py 存在",
                            actual_state=f"Agent 文件不存在",
                            severity=Severity.WARNING,
                            fix_suggestion=f"移除 V2_AGENT_TARGETS 中对 {node} 的引用，或创建 agent 文件",
                            auto_fixable=False,
                        )
                    )

        return reports

    def _check_orphaned_files(self, root: Path, files: list[str] | None = None) -> list[DriftReport]:
        """检查孤立文件。"""
        reports = []

        # 收集所有导入
        all_imports: set[str] = set()
        src_dir = root / "src"
        if not src_dir.exists():
            return reports

        for py_file in src_dir.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue
            imports = self._extract_imports(py_file)
            all_imports.update(imports)

        # 检查每个 Python 文件是否被引用
        for py_file in src_dir.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue
            if self._is_entry_point(py_file):
                continue

            # 检查是否被其他文件导入
            module_name = self._file_to_module(py_file.relative_to(src_dir.parent))
            if module_name not in all_imports and not self._has_any_reference(py_file):
                # 检查是否是测试文件（测试文件可能不被显式导入）
                if "test_" not in py_file.name and py_file.name != "__init__.py":
                    reports.append(
                        DriftReport(
                            drift_type=DriftType.ORPHANED_FILE,
                            source_file=str(py_file.relative_to(root)),
                            expected_state="文件被其他代码引用",
                            actual_state="文件未被任何地方引用",
                            severity=Severity.INFO,
                            fix_suggestion=f"考虑删除孤立文件 {py_file.name} 或将其添加到相关模块",
                            auto_fixable=False,
                        )
                    )

        return reports

    def _extract_dict_values(self, content: str, dict_name: str) -> list[str]:
        """从 Python 代码中提取字典的值列表。"""
        values = []
        # 匹配字典定义
        pattern = rf"{dict_name}\s*=\s*\{{([^}}]+)\}}"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            dict_content = match.group(1)
            # 提取引号中的值
            for m in re.finditer(r'"([^"]+)":|([^:]+):', dict_content):
                value = m.group(1) or m.group(2)
                value = value.strip().strip("'\"")
                if value and value not in values:
                    values.append(value)
        return values

    def _extract_imports(self, py_file: Path) -> set[str]:
        """提取 Python 文件中的所有导入。"""
        imports: set[str] = set()
        try:
            content = py_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return imports

        # 匹配 import 语句
        for match in re.finditer(
            r"^import\s+(\S+)|^from\s+(\S+)\s+import", content, re.MULTILINE
        ):
            module = match.group(1) or match.group(2)
            if module:
                imports.add(module.split(".")[0])

        return imports

    def _is_entry_point(self, path: Path) -> bool:
        """检查是否是入口文件。"""
        entry_patterns = [
            "__main__.py",
            "cli.py",
            "__init__.py",
        ]
        return path.name in entry_patterns

    def _has_any_reference(self, path: Path) -> bool:
        """检查文件是否被任何地方引用。"""
        root = PROJECT_ROOT
        content_needle = path.stem  # 文件名（不含扩展名）

        for py_file in (root / "src").rglob("*.py"):
            if py_file == path:
                continue
            try:
                content = py_file.read_text(encoding="utf-8")
                if content_needle in content:
                    return True
            except (OSError, UnicodeDecodeError):
                pass

        return False

    def _file_to_module(self, rel_path: Path) -> str:
        """将相对路径转换为模块名。"""
        parts = list(rel_path.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1].replace(".py", "")
        return ".".join(parts)


class DocDriftDetector:
    """检测文档漂移（代码改了，文档没改）。"""

    # 需要对应文档的模块模式
    MODULE_DOC_PAIRS = [
        (r"src/research/agents/(\w+)_agent\.py", r"docs/research/agents/\1.md"),
        (r"src/research/graph/nodes/(\w+)\.py", r"docs/research/graph/nodes/\1.md"),
        (r"src/tools/(\w+)\.py", r"docs/tools/\1.md"),
    ]

    def scan(self, files: list[str] | None = None) -> list[DriftReport]:
        """扫描文档漂移。"""
        reports = []
        root = PROJECT_ROOT

        for module_pattern, doc_pattern in self.MODULE_DOC_PAIRS:
            for match in root.rglob("*.py"):
                module_match = re.match(module_pattern, str(match.relative_to(root)))
                if module_match:
                    module_name = module_match.group(1)
                    doc_path = root / doc_pattern.replace(module_name, module_name)
                    if not doc_path.exists():
                        reports.append(
                            DriftReport(
                                drift_type=DriftType.MISSING_DOC,
                                source_file=str(match.relative_to(root)),
                                expected_state=f"文档 {doc_path.name} 存在",
                                actual_state="文档不存在",
                                severity=Severity.WARNING,
                                fix_suggestion=f"为 {module_name} 创建对应文档",
                                auto_fixable=False,
                            )
                        )

        return reports
