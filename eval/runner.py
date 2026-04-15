from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from eval.layers.hard_rules import run_layer1


def load_cases(path: str = "eval/cases.jsonl") -> list[dict]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_report_via_api(task_input: str, task_type: str, report_mode: str = "draft") -> str:
    """Call the /tasks API to run a report/research task and return the result markdown."""
    import requests

    base_url = os.getenv("PAPERREADER_API_URL", "http://localhost:8000")
    payload = {
        "input_type": "arxiv",
        "input_value": task_input,
        "report_mode": report_mode,
        "source_type": task_type,
    }
    resp = requests.post(f"{base_url}/tasks", json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    task_id = data["task_id"]
    print(f"  task_id={task_id[:8]}... ", end="", flush=True)

    # Poll until done
    for _ in range(120):
        time.sleep(5)
        result_resp = requests.get(f"{base_url}/tasks/{task_id}", timeout=10)
        result_resp.raise_for_status()
        result = result_resp.json()
        status = result.get("status", "unknown")
        if status in ("completed", "failed"):
            if status == "completed":
                return result.get("result_markdown") or result.get("draft_markdown") or result.get("full_markdown") or ""
            else:
                raise RuntimeError(f"Task failed: {result.get('error', 'unknown error')}")
        print(".", end="", flush=True)

    raise TimeoutError(f"Task {task_id} did not complete within 600s")


def run_report_direct(task_input: str, task_type: str, report_mode: str = "draft") -> str:
    """Run report directly (no API) using generate_literature_report."""
    from src.agent.report import generate_literature_report

    if task_type == "pdf":
        return generate_literature_report(raw_text_content=task_input)
    elif task_type == "sequential":
        parts = []
        for inp in task_input.split(","):
            inp = inp.strip()
            part = generate_literature_report(arxiv_url_or_id=inp)
            parts.append(part)
        return "\n\n---\n\n".join(parts)
    else:
        return generate_literature_report(arxiv_url_or_id=task_input)


def run_case(case: dict, *, mode: str, use_api: bool, layer: int) -> dict:
    """Run a single test case and return the result."""
    case_id = case["id"]
    case_type = case.get("type", "arxiv")
    start = time.time()

    try:
        if mode == "research":
            task_input = case.get("input", "")
            if use_api:
                report_md = run_report_via_api(task_input, "research", report_mode="draft")
            else:
                # research mode: use supervisor directly
                report_md = _run_research_direct(task_input, case)
        elif mode == "paper_read":
            # Single paper deep read
            if case_type == "pdf":
                if use_api:
                    report_md = run_report_via_api(case.get("pdf_text", ""), "pdf", report_mode="draft")
                else:
                    report_md = run_report_direct(case.get("pdf_text", ""), "pdf", report_mode="draft")
            else:
                arxiv_id = case.get("input", "")
                if use_api:
                    report_md = run_report_via_api(arxiv_id, "arxiv", report_mode="full")
                else:
                    report_md = run_report_direct(arxiv_id, "arxiv", report_mode="full")
        else:
            # Default: paper reading mode (arxiv/pdf/sequential via generate_literature_report)
            if use_api:
                if case_type == "pdf":
                    report_md = run_report_via_api(case.get("pdf_text", ""), "pdf", report_mode="draft")
                elif case_type == "sequential":
                    inputs = case.get("inputs", [])
                    parts = []
                    for inp in inputs:
                        part = run_report_via_api(inp, "arxiv", report_mode="draft")
                        parts.append(part)
                    report_md = "\n\n---\n\n".join(parts)
                else:
                    report_md = run_report_via_api(case.get("input", ""), "arxiv", report_mode="draft")
            else:
                report_md = run_report_direct(case.get("input", ""), case_type, report_mode="draft")

        duration = time.time() - start

        if layer >= 1:
            l1 = run_layer1(report_md, case)
            result = {
                "id": case_id,
                "type": case_type,
                "mode": mode,
                "layer1": l1,
                "duration": round(duration, 2),
                "report_snippet": report_md[:300] if report_md else "",
                "report_length": len(report_md) if report_md else 0,
            }
            passed = l1["pass"]
            print(f"{'PASS' if passed else 'FAIL'} ({duration:.1f}s, {len(report_md)} chars)")
            return result
        else:
            print(f"OK ({duration:.1f}s, {len(report_md)} chars)")
            return {
                "id": case_id,
                "type": case_type,
                "mode": mode,
                "duration": round(duration, 2),
                "report_snippet": report_md[:300] if report_md else "",
                "report_length": len(report_md) if report_md else 0,
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


def _run_research_direct(topic: str, case: dict) -> str:
    """Run research workflow directly (no HTTP API)."""
    from src.api.routes.tasks import _run_graph
    from src.models.task import TaskRecord, TaskStatus

    task = TaskRecord(
        input_type="arxiv",
        input_value=topic,
        report_mode="draft",
        source_type="research",
    )
    from src.api.routes.tasks import _tasks
    _tasks[task.task_id] = task

    asyncio.run(_run_graph(task.task_id))

    # Reload from store
    task = _tasks.get(task.task_id)
    if not task:
        raise RuntimeError(f"Task {task.task_id} not found after execution")

    if task.status != TaskStatus.COMPLETED:
        raise RuntimeError(f"Task failed: {task.error or task.status.value}")

    return task.result_markdown or task.draft_markdown or ""


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
      paper_read  - arxiv/pdf/sequential papers via generate_literature_report (default)
      research    - research workflow via supervisor
      api         - call /tasks API endpoint (requires server running)

    Args:
      cases_path: path to cases.jsonl
      layer: evaluation layer (1 = hard rules)
      out_dir: output directory for results
      mode: paper_read | research | api
      use_api: use HTTP API instead of direct import
      research_topic: topic for research mode (single topic test)
      case_filter: only run cases matching this id prefix
    """
    cases = load_cases(cases_path)
    if case_filter:
        cases = [c for c in cases if c["id"].startswith(case_filter)]

    results = []

    if mode == "research" and research_topic:
        # Single topic research test
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
    }

    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(out_dir, f"run-{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults: {passed}/{total} passed, {errors} errors → {out_path}")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run report generation evaluation")
    parser.add_argument("--cases", default="eval/cases.jsonl", help="Path to cases JSONL")
    parser.add_argument("--layer", type=int, default=1, help="Evaluation layer (1=hard rules)")
    parser.add_argument(
        "--out-dir", default="eval/runs", help="Output directory for results"
    )
    parser.add_argument(
        "--mode",
        default="paper_read",
        choices=["paper_read", "research", "api"],
        help="Evaluation mode: paper_read (default), research, api",
    )
    parser.add_argument(
        "--use-api", action="store_true", help="Use HTTP API instead of direct import"
    )
    parser.add_argument(
        "--research-topic", help="Topic for research mode test"
    )
    parser.add_argument(
        "--case-filter", help="Only run cases matching this ID prefix"
    )
    args = parser.parse_args()

    run_eval(
        cases_path=args.cases,
        layer=args.layer,
        out_dir=args.out_dir,
        mode=args.mode,
        use_api=args.use_api,
        research_topic=args.research_topic,
        case_filter=args.case_filter,
    )
