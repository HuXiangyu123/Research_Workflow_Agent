"""Entropy Management System — 检测、清理、预防 AI Agent 系统的代码腐化。

设计文档：docs/features_oncoming/entropy-management.md

核心功能：
- EntropyScanner: 检测文档漂移、代码风格不一致、死代码、约束违反
- EntropyCleaner: 清理孤立文件、修复约束违反
- EntropyScheduler: 调度扫描任务（on-commit, daily, on-pr, manual）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


class DriftType(str, Enum):
    """熵类型枚举。"""

    MISSING_DOC = "missing_doc"
    UNENFORCED_CONSTRAINT = "unenforced_constraint"
    MISSING_NODE_FILE = "missing_node_file"
    ORPHANED_FILE = "orphaned_file"
    STYLE_DRIFT = "style_drift"
    ARTIFACT_QUALITY = "artifact_quality"


class Severity(str, Enum):
    """严重程度枚举。"""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class DriftReport:
    """单个漂移报告。"""

    drift_type: DriftType
    source_file: str
    expected_state: str
    actual_state: str
    severity: Severity
    fix_suggestion: str
    auto_fixable: bool = False


@dataclass
class EntropySummary:
    """熵扫描摘要。"""

    total_issues: int = 0
    critical: int = 0
    warning: int = 0
    info: int = 0
    entropy_delta: float = 0.0


@dataclass
class EntropyReport:
    """完整的熵管理报告。"""

    timestamp: str = ""
    trigger: str = "manual"
    summary: EntropySummary = field(default_factory=EntropySummary)
    drift_reports: list[DriftReport] = field(default_factory=list)
    auto_fix_changes: list[dict] = field(default_factory=list)
    pending_changes: list[dict] = field(default_factory=list)
    entropy_score: float = 100.0

    def add_drift(self, drift: DriftReport) -> None:
        """添加漂移报告并更新摘要。"""
        self.drift_reports.append(drift)
        self.summary.total_issues += 1
        if drift.severity == Severity.CRITICAL:
            self.summary.critical += 1
        elif drift.severity == Severity.WARNING:
            self.summary.warning += 1
        else:
            self.summary.info += 1

        # 更新熵评分（越低越差）
        if drift.severity == Severity.CRITICAL:
            self.entropy_score -= 10
        elif drift.severity == Severity.WARNING:
            self.entropy_score -= 3
        else:
            self.entropy_score -= 1

        self.entropy_score = max(0, self.entropy_score)
