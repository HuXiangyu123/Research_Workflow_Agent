"""Evaluation runner for PaperReader agent.

Supports testing both paper_read mode and research mode with full task lifecycle.
Reports are saved via output_workspace.py to output/<task_id>/report.md
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

# Load .env before any other imports that might use LLM config
from dotenv import load_dotenv

load_dotenv(".env")

from eval.layers.hard_rules import run_layer1


def load_cases(path: str = "eval/cases.jsonl") -> list[dict]:
    """Load test cases from JSONL file."""
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_paper_read_via_api(arxiv_id: str) -> tuple[str, str]:
    """Run paper_read task via API, return (task_id, report_md)."""
    import requests

    base_url = os.getenv("PAPERREADER_API_URL", "http://localhost:8000")
    payload = {
        "input_type": "arxiv",
        "input_value": arxiv_id,
        "report_mode": "draft",
        "source_type": "arxiv",
    }
    resp = requests.post(f"{base_url}/tasks", json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    task_id = data["task_id"]
    print(f"  [{task_id[:12]}] ", end="", flush=True)

    # Poll until done
    for _ in range(120):
        time.sleep(5)
        result_resp = requests.get(f"{base_url}/tasks/{task_id}", timeout=10)
        result_resp.raise_for_status()
        result = result_resp.json()
        status = result.get("status", "unknown")
        if status in ("completed", "failed"):
            if status == "completed":
                report = result.get("result_markdown") or result.get("draft_markdown") or result.get("full_markdown") or ""
                return task_id, report
            else:
                raise RuntimeError(f"Task failed: {result.get('error', 'unknown error')}")
        print(".", end="", flush=True)

    raise TimeoutError(f"Task {task_id} did not complete within 600s")


def run_paper_read_direct(arxiv_id: str, pdf_text: str | None = None) -> tuple[str, str]:
    """Run paper_read task directly, return (task_id, report_md)."""
    from src.agent.report import generate_literature_report

    # Create a unique task_id for direct mode
    task_id = f"paper-{arxiv_id or 'pdf'}-{int(time.time())}"

    report_md = generate_literature_report(
        arxiv_url_or_id=arxiv_id if arxiv_id else None,
        raw_text_content=pdf_text,
        task_id=task_id,
    )
    return task_id, report_md


def run_research_via_api(topic: str) -> tuple[str, str]:
    """Run research task via API, return (task_id, report_md)."""
    import requests

    base_url = os.getenv("PAPERREADER_API_URL", "http://localhost:8000")
    payload = {
        "input_type": "arxiv",
        "input_value": topic,
        "report_mode": "draft",
        "source_type": "research",
    }
    resp = requests.post(f"{base_url}/tasks", json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    task_id = data["task_id"]
    print(f"  [{task_id[:12]}] ", end="", flush=True)

    for _ in range(120):
        time.sleep(5)
        result_resp = requests.get(f"{base_url}/tasks/{task_id}", timeout=10)
        result_resp.raise_for_status()
        result = result_resp.json()
        status = result.get("status", "unknown")
        if status in ("completed", "failed"):
            if status == "completed":
                report = result.get("result_markdown") or result.get("draft_markdown") or result.get("full_markdown") or ""
                return task_id, report
            else:
                raise RuntimeError(f"Task failed: {result.get('error', 'unknown error')}")
        print(".", end="", flush=True)

    raise TimeoutError(f"Task {task_id} did not complete within 600s")


def run_research_direct(topic: str) -> tuple[str, str]:
    """Run research task directly, return (task_id, report_md)."""
    from src.api.routes.tasks import _run_graph, _tasks, _build_state_template
    from src.models.task import TaskRecord, TaskStatus

    task = TaskRecord(
        input_type="arxiv",
        input_value=topic,
        report_mode="draft",
        source_type="research",
    )
    _tasks[task.task_id] = task

    asyncio.run(_run_graph(task.task_id))

    # Reload from store
    task = _tasks.get(task.task_id)
    if not task:
        raise RuntimeError(f"Task {task.task_id} not found after execution")

    if task.status != TaskStatus.COMPLETED:
        raise RuntimeError(f"Task failed: {task.error or task.status.value}")

    report = task.result_markdown or task.draft_markdown or ""
    return task.task_id, report


def run_case(case: dict, *, mode: str, use_api: bool, layer: int) -> dict:
    """Run a single test case and return the result."""
    case_id = case["id"]
    case_type = case.get("type", "arxiv")
    start = time.time()

    try:
        if mode == "research":
            task_input = case.get("input", "")
            if use_api:
                task_id, report_md = run_research_via_api(task_input)
            else:
                task_id, report_md = run_research_direct(task_input)
        elif mode == "paper_read":
            if case_type == "pdf":
                # PDF mode: use raw_text_content
                pdf_text = case.get("pdf_text", "")
                task_id, report_md = run_paper_read_direct(None, pdf_text)
            else:
                arxiv_id = case.get("input", "")
                if use_api:
                    task_id, report_md = run_paper_read_via_api(arxiv_id)
                else:
                    task_id, report_md = run_paper_read_direct(arxiv_id)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        duration = time.time() - start

        if layer >= 1:
            l1 = run_layer1(report_md, case)
            result = {
                "id": case_id,
                "type": case_type,
                "mode": mode,
                "task_id": task_id,
                "layer1": l1,
                "duration": round(duration, 2),
                "report_length": len(report_md),
            }
            passed = l1.get("pass", False)
            print(f"{'PASS' if passed else 'FAIL'} ({duration:.1f}s, {len(report_md)} chars)")
            return result
        else:
            print(f"OK ({duration:.1f}s, {len(report_md)} chars)")
            return {
                "id": case_id,
                "type": case_type,
                "mode": mode,
                "task_id": task_id,
                "duration": round(duration, 2),
                "report_length": len(report_md),
            }

    except Exception as e:
        duration = time.time() - start
        print(f"ERROR ({duration:.1f}s): {e}")
        return {
            "id": case_id,
            "type": case_type,
            "mode": mode,
            "error": str(e),
            "duration": round(duration, 2),
        }


def run_eval(
    cases_path: str = "eval/cases.jsonl",
    layer: int = 1,
    out_dir: str = "eval/runs",
    *,
    mode: str = "paper_read",
    use_api: bool = False,
    research_topic: str | None = None,
    case_filter: str | None = None,
) -> dict:
    """Run evaluation with configurable mode.

    Modes:
      paper_read  - arxiv papers via generate_literature_report (default)
      research    - research workflow via supervisor
      api         - call /tasks API endpoint (requires server running)
      full        - run both paper_read and research modes

    Args:
      cases_path: path to cases.jsonl
      layer: evaluation layer (1 = hard rules)
      out_dir: output directory for results
      mode: paper_read | research | api | full
      use_api: use HTTP API instead of direct import
      research_topic: topic for research mode (single topic test)
      case_filter: only run cases matching this id prefix
    """
    cases = load_cases(cases_path)
    if case_filter:
        cases = [c for c in cases if c["id"].startswith(case_filter)]

    results = []

    if mode == "full":
        # Run both paper_read and research tests
        print("\n" + "=" * 60)
        print("FULL TEST: Running both paper_read and research modes")
        print("=" * 60)

        # Paper read tests
        print("\n### Paper Read Mode ###")
        paper_read_cases = [c for c in cases if c.get("type") in ("arxiv", "gold")][:3]  # Limit to 3 cases
        for case in paper_read_cases:
            print(f"\nRunning case: {case['id']}")
            result = run_case(case, mode="paper_read", use_api=use_api, layer=layer)
            results.append(result)

        # Research tests - use cases from cases.jsonl
        print("\n### Research Mode ###")
        research_cases = [c for c in cases if c.get("type") == "research"]
        if not research_cases:
            # Fallback if no research cases defined
            research_cases = [
                {"id": "research-01", "type": "research", "input": "AI Agent", "must_include": ["Agent"], "min_citations": 0, "sections": ["Abstract", "Introduction"]},
            ]
        for case in research_cases:
            print(f"\nRunning case: {case['id']}")
            result = run_case(case, mode="research", use_api=use_api, layer=layer)
            results.append(result)

    elif mode == "research" and research_topic:
        print(f"\n=== Research Mode: {research_topic} ===")
        case = {
            "id": "research-single",
            "type": "research",
            "input": research_topic,
            "must_include": [],
            "min_citations": 5,
        }
        result = run_case(case, mode="research", use_api=use_api, layer=layer)
        results.append(result)
    else:
        print(f"\n=== {mode.upper()} Mode ===")
        for case in cases:
            result = run_case(case, mode=mode, use_api=use_api, layer=layer)
            results.append(result)

    total = len(results)
    passed = sum(1 for r in results if r.get("layer1", {}).get("pass", False))
    errors = sum(1 for r in results if "error" in r)
    summary = {
        "total": total,
        "passed": passed,
        "failed": total - passed - errors,
        "errors": errors,
    }

    output = {
        "summary": summary,
        "results": results,
        "config": {
            "mode": mode,
            "use_api": use_api,
            "layer": layer,
            "cases_path": cases_path,
            "research_topic": research_topic,
        },
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(out_dir, f"run-{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {errors} errors")
    print(f"Reports saved to: output/<task_id>/report.md")
    print(f"JSON log: {out_path}")

    return output


def print_test_report(output: dict) -> str:
    """Generate a markdown test report from evaluation results."""
    summary = output.get("summary", {})
    results = output.get("results", [])
    config = output.get("config", {})

    lines = [
        "# PaperReader Evaluation Report",
        "",
        f"**Timestamp**: {output.get('timestamp', 'N/A')}",
        f"**Mode**: {config.get('mode', 'N/A')}",
        f"**Total**: {summary.get('total', 0)} | **Passed**: {summary.get('passed', 0)} | **Failed**: {summary.get('failed', 0)} | **Errors**: {summary.get('errors', 0)}",
        "",
        "---",
        "",
        "## Test Results",
        "",
        "| Case ID | Mode | Task ID | Duration | Report Length | Layer1 Pass |",
        "|---|---|---|---|---|---|",
    ]

    for r in results:
        task_id = r.get("task_id", "N/A")[:12] + "..." if r.get("task_id") else "N/A"
        layer1_pass = r.get("layer1", {}).get("pass", False) if "layer1" in r else "N/A"
        lines.append(
            f"| {r.get('id', 'N/A')} | {r.get('mode', 'N/A')} | {task_id} | {r.get('duration', 0):.1f}s | {r.get('report_length', 0)} chars | {layer1_pass} |"
        )

    lines.extend(["", "## Task Output Locations", ""])
    for r in results:
        if r.get("task_id"):
            lines.append(f"- `{r['task_id']}` → `output/{r['task_id']}/report.md`")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PaperReader evaluation")
    parser.add_argument("--cases", default="eval/cases.jsonl", help="Path to cases JSONL")
    parser.add_argument("--layer", type=int, default=1, help="Evaluation layer (1=hard rules)")
    parser.add_argument("--out-dir", default="eval/runs", help="Output directory for results")
    parser.add_argument(
        "--mode",
        default="paper_read",
        choices=["paper_read", "research", "api", "full"],
        help="Evaluation mode: paper_read (default), research, api, or full (both modes)",
    )
    parser.add_argument("--use-api", action="store_true", help="Use HTTP API instead of direct import")
    parser.add_argument("--research-topic", help="Topic for research mode test")
    parser.add_argument("--case-filter", help="Only run cases matching this ID prefix")
    parser.add_argument("--report", action="store_true", help="Print markdown test report after run")
    args = parser.parse_args()

    output = run_eval(
        cases_path=args.cases,
        layer=args.layer,
        out_dir=args.out_dir,
        mode=args.mode,
        use_api=args.use_api,
        research_topic=args.research_topic,
        case_filter=args.case_filter,
    )

    if args.report:
        print("\n" + print_test_report(output))