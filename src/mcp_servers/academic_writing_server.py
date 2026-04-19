"""Minimal MCP stdio server for academic survey writing support.

This server intentionally implements only the subset of the protocol that the
project adapter currently consumes: tools/list, prompts/list, resources/list,
prompts/get, and resources/read. It runs as a long-lived stdio process and can
be registered through ``src.tools.mcp_adapter``.
"""

from __future__ import annotations

import json
import sys
from typing import Any


WRITING_RULES: list[str] = [
    "Organize review sections by themes, method families, debates, or evidence gaps, not by paper order.",
    "Use the introduction to define scope, boundary conditions, and the organizing logic of the survey.",
    "Keep background short and only include enabling work that is necessary to frame the in-scope literature.",
    "The methods section should compare representative systems, design trade-offs, and evidence, rather than restating abstracts.",
    "The discussion should explicitly cover agreements, disagreements, trade-offs, and evidence gaps across papers.",
    "Future directions must be derived from unresolved gaps in evaluation, grounding, safety, or deployment evidence.",
]

WRITING_SOURCES: list[dict[str, str]] = [
    {
        "title": "Monash University - Structuring a literature review",
        "url": "https://www.monash.edu/student-academic-success/excel-at-writing/how-to-write/literature-review/structuring-a-literature-review",
    },
    {
        "title": "UMass Amherst Writing Center - Literature Reviews",
        "url": "https://www.umass.edu/writing-center/resources/literature-reviews",
    },
    {
        "title": "Elsevier Researcher Academy - An editor's guide to writing a review article",
        "url": "https://researcheracademy.elsevier.com/writing-research/technical-writing-skills/editor-guide-writing-review-article",
    },
]


def _jsonrpc_ok(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(request_id: Any, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32000, "message": message},
    }


def _prompt_payload(arguments: dict[str, Any]) -> dict[str, Any]:
    topic = str(arguments.get("topic") or "the requested research topic").strip()
    time_range = str(arguments.get("time_range") or "").strip()
    focus_dimensions = arguments.get("focus_dimensions") or []
    dimensions_text = ", ".join(str(item).strip() for item in focus_dimensions if str(item).strip())
    scope_note = f"Time range: {time_range}." if time_range else "Use the time range exactly as specified in the brief."
    focus_note = (
        f"Focus dimensions: {dimensions_text}."
        if dimensions_text
        else "Focus on architecture, evidence quality, datasets, evaluation, and limitations."
    )
    prompt = "\n".join(
        [
            f"Write an English academic survey about {topic}.",
            scope_note,
            focus_note,
            "Writing rules:",
            *[f"{idx}. {rule}" for idx, rule in enumerate(WRITING_RULES, start=1)],
            "Hard scope control:",
            "- Keep the title and the body aligned with the requested time range.",
            "- Use adjacent enabling work only when it helps explain in-scope agent systems.",
            "- Drop clearly off-topic governance or component-only papers from the main synthesis.",
            "- Prefer evidence from full-text snippets over abstract-only summaries when available.",
        ]
    )
    return {
        "prompt": prompt,
        "rules": WRITING_RULES,
        "sources": WRITING_SOURCES,
    }


def _handle(method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method == "tools/list":
        return {"tools": []}
    if method == "prompts/list":
        return {
            "prompts": [
                {
                    "name": "academic_review_writer",
                    "title": "Academic Review Writer",
                    "description": "Survey-writing rubric and scope guardrails for academic review generation.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string"},
                            "time_range": {"type": "string"},
                            "focus_dimensions": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["topic"],
                    },
                    "tags": ["writing", "survey", "review"],
                }
            ]
        }
    if method == "resources/list":
        return {
            "resources": [
                {
                    "uri": "resource://academic-writing/references",
                    "title": "Academic Review Writing References",
                    "description": "External references used by the writing support server.",
                    "mimeType": "application/json",
                    "tags": ["writing", "references"],
                }
            ]
        }
    if method == "prompts/get":
        name = str(params.get("name") or "")
        if name != "academic_review_writer":
            raise KeyError(f"Unknown prompt: {name}")
        return _prompt_payload(params.get("arguments") or {})
    if method == "resources/read":
        uri = str(params.get("uri") or "")
        if uri != "resource://academic-writing/references":
            raise KeyError(f"Unknown resource: {uri}")
        return {"contents": WRITING_SOURCES}
    raise KeyError(f"Unsupported method: {method}")


def main() -> int:
    for raw_line in sys.stdin.buffer:
        if not raw_line.strip():
            continue
        try:
            request = json.loads(raw_line.decode("utf-8"))
            response = _jsonrpc_ok(
                request.get("id"),
                _handle(
                    str(request.get("method") or ""),
                    request.get("params") or {},
                ),
            )
        except Exception as exc:  # noqa: BLE001
            request_id = None
            try:
                request_id = request.get("id")  # type: ignore[name-defined]
            except Exception:
                pass
            response = _jsonrpc_error(request_id, str(exc))
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
