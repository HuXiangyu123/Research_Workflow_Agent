from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from typing import Any

from dotenv import load_dotenv
from fastapi.testclient import TestClient

from src.agent.output_workspace import get_workspace_summary
from src.api.app import app


PLACEHOLDER_PATTERNS = [
    r"\[Research Topic\]",
    r"\[Dataset [^\]]+\]",
    r"\[Example [^\]]+\]",
    r"\[Adjacent Field [^\]]+\]",
    r"\bTBD\b",
    r"\bTODO\b",
    r"待补",
    r"待完善",
    r"占位",
]


def _duplicate_paragraph_ratio(markdown: str) -> float:
    paragraphs = [
        re.sub(r"\s+", " ", p).strip()
        for p in re.split(r"\n\s*\n", markdown)
        if len(re.sub(r"\s+", " ", p).strip()) >= 80
    ]
    if not paragraphs:
        return 0.0
    counts = Counter(paragraphs)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    return duplicates / len(paragraphs)


def _quantify(task: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    report = (result.get("result_markdown") or task.get("result_markdown") or "").strip()
    headings = [line for line in report.splitlines() if line.startswith("#")]
    placeholder_hits = sum(
        len(re.findall(pattern, report, flags=re.IGNORECASE))
        for pattern in PLACEHOLDER_PATTERNS
    )
    rag_result = task.get("rag_result") or {}
    paper_count = 0
    if isinstance(rag_result, dict):
        paper_count = int(rag_result.get("total_papers") or len(rag_result.get("paper_candidates") or []))

    metrics = {
        "status": task.get("status"),
        "current_stage": task.get("current_stage"),
        "supervisor_mode": task.get("supervisor_mode"),
        "trace_refs": [entry.get("node") for entry in task.get("collaboration_trace", []) if isinstance(entry, dict)],
        "paper_count": paper_count,
        "report_chars": len(report),
        "heading_count": len(headings),
        "placeholder_hits": placeholder_hits,
        "duplicate_paragraph_ratio": round(_duplicate_paragraph_ratio(report), 4),
        "review_passed": task.get("review_passed"),
    }
    metrics["quality_gate_passed"] = (
        metrics["status"] == "completed"
        and metrics["paper_count"] >= 6
        and metrics["report_chars"] >= 4000
        and metrics["heading_count"] >= 8
        and metrics["placeholder_hits"] == 0
        and metrics["duplicate_paragraph_ratio"] <= 0.12
    )
    return metrics


def main() -> int:
    load_dotenv(".env")
    parser = argparse.ArgumentParser(description="Run a real research task and score the result.")
    parser.add_argument("query", help="Research query")
    parser.add_argument("--timeout", type=int, default=900, help="Polling timeout in seconds")
    parser.add_argument("--auto-fill", action="store_true", default=True, help="Enable clarify auto fill")
    args = parser.parse_args()

    with TestClient(app) as client:
        create_resp = client.post(
            "/tasks",
            json={
                "input_type": "research",
                "input_value": args.query,
                "source_type": "research",
                "report_mode": "draft",
                "auto_fill": args.auto_fill,
            },
        )
        create_resp.raise_for_status()
        created = create_resp.json()
        task_id = created["task_id"]
        deadline = time.time() + args.timeout

        task_payload: dict[str, Any] | None = None
        while time.time() < deadline:
            resp = client.get(f"/tasks/{task_id}")
            resp.raise_for_status()
            task_payload = resp.json()
            if task_payload["status"] in {"completed", "failed"}:
                break
            time.sleep(2)

        if task_payload is None:
            raise RuntimeError("task polling returned no payload")

        result_resp = client.get(f"/tasks/{task_id}/result")
        result_resp.raise_for_status()
        result_payload = result_resp.json()
        workspace_summary = get_workspace_summary(task_id, workspace_id=created.get("workspace_id"))

    output = {
        "task": {
            "task_id": task_id,
            "workspace_id": created.get("workspace_id"),
            "status": task_payload.get("status"),
        },
        "metrics": _quantify(task_payload, result_payload),
        "workspace": workspace_summary,
        "report_preview": (result_payload.get("result_markdown") or "")[:2000],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
