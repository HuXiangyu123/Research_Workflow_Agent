"""Entropy Management CLI — 命令行入口。

用法：
    python -m src.entropy.cli scan
    python -m src.entropy.cli scan --files src/research/agents/supervisor.py
    python -m src.entropy.cli scan --format json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import click


@click.group()
def cli():
    """Entropy Management CLI — 检测和管理 AI Agent 系统的代码腐化。"""
    pass


@cli.command()
@click.option("--files", multiple=True, help="只扫描指定文件")
@click.option("--format", type=click.Choice(["text", "json"]), default="text", help="输出格式")
@click.option("--output", type=click.Path(), help="输出到文件")
def scan(files: tuple[str, ...], format: str, output: str | None):
    """扫描仓库中的熵问题。"""
    from src.entropy.scanner import EntropyReport
    from src.entropy.detectors.constraint import ConstraintViolationDetector, DeadCodeDetector, DocDriftDetector

    # 收集所有漂移报告
    all_drifts = []
    detectors = [
        ConstraintViolationDetector(),
        DeadCodeDetector(),
        DocDriftDetector(),
    ]

    file_list = list(files) if files else None
    for detector in detectors:
        try:
            drifts = detector.scan(file_list)
            all_drifts.extend(drifts)
        except Exception as e:
            click.echo(f"Warning: {detector.__class__.__name__} failed: {e}", err=True)

    # 生成报告
    report = EntropyReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        trigger="cli",
    )
    for drift in all_drifts:
        report.add_drift(drift)

    # 输出
    if format == "json":
        output_data = {
            "timestamp": report.timestamp,
            "trigger": report.trigger,
            "summary": {
                "total": report.summary.total_issues,
                "critical": report.summary.critical,
                "warning": report.summary.warning,
                "info": report.summary.info,
                "entropy_score": report.entropy_score,
            },
            "drifts": [
                {
                    "type": d.drift_type.value,
                    "file": d.source_file,
                    "severity": d.severity.value,
                    "expected": d.expected_state,
                    "actual": d.actual_state,
                    "fix": d.fix_suggestion,
                }
                for d in report.drift_reports
            ],
        }
        output_str = json.dumps(output_data, indent=2, ensure_ascii=False)
    else:
        lines = [
            f"# Entropy Scan Report — {report.timestamp}",
            f"# Entropy Score: {report.entropy_score}/100",
            f"# Total Issues: {report.summary.total_issues} "
            f"(Critical: {report.summary.critical}, "
            f"Warning: {report.summary.warning}, "
            f"Info: {report.summary.info})",
            "",
            "## Drift Reports",
            "",
        ]
        if not report.drift_reports:
            lines.append("No issues found.")
        else:
            for d in report.drift_reports:
                severity_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(
                    d.severity.value, "⚪"
                )
                lines.append(f"{severity_icon} [{d.severity.value.upper()}] {d.source_file}")
                lines.append(f"   Type: {d.drift_type.value}")
                lines.append(f"   Expected: {d.expected_state}")
                lines.append(f"   Actual: {d.actual_state}")
                lines.append(f"   Fix: {d.fix_suggestion}")
                lines.append("")

        output_str = "\n".join(lines)

    # 写入输出
    if output:
        Path(output).write_text(output_str, encoding="utf-8")
        click.echo(f"Report written to {output}")
    else:
        click.echo(output_str)

    # 返回退出码
    sys.exit(0 if report.summary.critical == 0 else 1)


@cli.command()
@click.option("--files", multiple=True, help="只扫描指定文件")
def check(files: tuple[str, ...]):
    """快速检查关键约束违反（如 SQLite）。"""
    from src.entropy.detectors.constraint import ConstraintViolationDetector

    detector = ConstraintViolationDetector()
    file_list = list(files) if files else None

    try:
        drifts = detector.scan(file_list)
        critical_drifts = [d for d in drifts if d.severity.value == "critical"]

        if critical_drifts:
            click.echo(f"❌ Found {len(critical_drifts)} critical violations:", err=True)
            for d in critical_drifts:
                click.echo(f"  {d.source_file}: {d.drift_type.value}", err=True)
            sys.exit(1)
        else:
            click.echo("✅ No critical violations found.")
            sys.exit(0)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
