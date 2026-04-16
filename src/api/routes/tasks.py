from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.graph.callbacks import NodeEventEmitter
from src.models.task import TaskRecord, TaskStatus
from src.agent.report_frame import REGULAR_CHAT_SYSTEM_PROMPT, SURVEY_CHAT_SYSTEM_PROMPT
from src.db.task_persistence import (
    list_task_snapshots,
    load_task_report,
    load_task_snapshot,
    save_task_report,
    upsert_task_snapshot,
)

router = APIRouter(prefix="/tasks", tags=["tasks"])

_tasks: dict[str, TaskRecord] = {}
_STREAMABLE_TASK_FILES = {
    "brief.json",
    "search_plan.json",
    "rag_result.json",
    "paper_cards.json",
    "comparison_matrix.json",
    "writing_scaffold.json",
    "writing_outline.json",
    "mcp_prompt_payload.json",
    "draft_skill_trace.json",
    "review_skill_trace.json",
    "claim_verification.json",
    "review_feedback.json",
    "draft.md",
    "report.md",
}


class CreateTaskRequest(BaseModel):
    input_type: str = Field(default="arxiv", description="'arxiv' or 'pdf'")
    input_value: str = Field(..., description="arXiv ID/URL or PDF text")
    report_mode: str = Field(default="draft", description="'draft' or 'full'")
    source_type: str = Field(
        default="arxiv",
        description="'arxiv', 'pdf' (paper-ingestion) or 'research' (clarify workflow)",
    )
    workspace_id: str | None = Field(
        default=None,
        description="Optional existing workspace_id to attach this task to",
    )
    auto_fill: bool = Field(
        default=False,
        description="If True, LLM auto-completes ambiguous fields in brief instead of requiring human followup",
    )


class CreateTaskResponse(BaseModel):
    task_id: str
    status: str
    workspace_id: str
    source_type: str
    report_mode: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    role: str
    content: str


def _json_safe(value):
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except TypeError:
        return str(value)


def _sync_task_snapshot(task: TaskRecord) -> None:
    try:
        task.persisted_to_db = bool(upsert_task_snapshot(task))
        task.persistence_error = None if task.persisted_to_db else task.persistence_error
    except Exception as exc:  # noqa: BLE001
        task.persistence_error = str(exc)
        task.persisted_to_db = False


def _get_task_record(task_id: str) -> TaskRecord | None:
    task = _tasks.get(task_id)
    if task:
        return task
    task = load_task_snapshot(task_id)
    if task:
        _tasks[task.task_id] = task
    return task


def _task_payload(task: TaskRecord) -> dict:
    return {
        "task_id": task.task_id,
        "status": task.status.value,
        "created_at": task.created_at,
        "completed_at": task.completed_at,
        "workspace_id": task.workspace_id,
        "report_mode": task.report_mode,
        "source_type": task.source_type,
        "paper_type": task.paper_type,
        "draft_markdown": task.draft_markdown,
        "full_markdown": task.full_markdown,
        "result_markdown": task.result_markdown,
        "brief": task.brief,
        "search_plan": task.search_plan,
        "rag_result": task.rag_result,
        "paper_cards": task.paper_cards,
        "compression_result": task.compression_result,
        "taxonomy": task.taxonomy,
        "draft_report": task.draft_report,
        "review_feedback": task.review_feedback,
        "review_passed": task.review_passed,
        "artifacts_created": task.artifacts_created,
        "artifact_count": task.artifact_count,
        "collaboration_trace": task.collaboration_trace,
        "supervisor_mode": task.supervisor_mode,
        "current_stage": task.current_stage,
        "followup_hints": task.followup_hints,
        "awaiting_followup": task.awaiting_followup,
        "followup_resolution": task.followup_resolution,
        "chat_history": task.chat_history,
        "error": task.error,
        "persisted_to_db": task.persisted_to_db,
        "persisted_report_id": task.persisted_report_id,
        "persistence_error": task.persistence_error,
    }


def _task_workspace_dir(task: TaskRecord) -> Path | None:
    if not task.workspace_id:
        return None
    from src.agent.output_workspace import get_workspace_path

    path = get_workspace_path(task.task_id, workspace_id=task.workspace_id)
    return path if path.is_dir() else None


def _workspace_stream_events(
    task: TaskRecord,
    seen_files: dict[str, int],
) -> list[dict]:
    from datetime import datetime, timezone

    task_dir = _task_workspace_dir(task)
    if task_dir is None:
        return []

    candidate_paths = [
        path
        for path in (task_dir / name for name in sorted(_STREAMABLE_TASK_FILES))
        if path.is_file()
    ]
    revisions_dir = task_dir / "revisions"
    if revisions_dir.is_dir():
        candidate_paths.extend(sorted(revisions_dir.glob("*.md")))

    events: list[dict] = []
    for path in candidate_paths:
        rel_path = str(path.relative_to(task_dir))
        mtime_ns = path.stat().st_mtime_ns
        if seen_files.get(rel_path) == mtime_ns:
            continue
        seen_files[rel_path] = mtime_ns

        timestamp = datetime.now(timezone.utc).isoformat()
        if path.suffix == ".md":
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                continue
            events.append(
                {
                    "type": "report_snapshot",
                    "artifact_name": rel_path,
                    "workspace_id": task.workspace_id,
                    "timestamp": timestamp,
                    "content": content,
                    "is_final": path.name == "report.md",
                }
            )
            continue

        summary = path.name
        if path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    summary = ", ".join(list(payload.keys())[:4]) or path.name
                elif isinstance(payload, list):
                    summary = f"{len(payload)} items"
            except Exception:
                summary = path.name

        events.append(
            {
                "type": "artifact_ready",
                "artifact_name": rel_path,
                "workspace_id": task.workspace_id,
                "timestamp": timestamp,
                "summary": summary,
            }
        )
    return events


@router.post("", response_model=CreateTaskResponse)
async def create_task(req: CreateTaskRequest) -> CreateTaskResponse:
    from src.agent.output_workspace import DEFAULT_WORKSPACE_USER, build_workspace_id

    source_type = req.source_type if req.source_type in {"arxiv", "pdf", "research"} else "arxiv"
    workspace_user = DEFAULT_WORKSPACE_USER or "user"
    workspace_id = (req.workspace_id or "").strip() or build_workspace_id(workspace_user)
    task = TaskRecord(
        input_type=req.input_type,
        input_value=req.input_value,
        report_mode=req.report_mode if req.report_mode in {"draft", "full"} else "draft",
        source_type=source_type,
        auto_fill=req.auto_fill,
        workspace_id=workspace_id,
    )
    _tasks[task.task_id] = task
    _sync_task_snapshot(task)

    asyncio.create_task(_run_graph(task.task_id))

    return CreateTaskResponse(
        task_id=task.task_id,
        status=task.status.value,
        workspace_id=task.workspace_id or "",
        source_type=task.source_type,
        report_mode=task.report_mode,
    )


@router.get("")
async def list_tasks():
    tasks_by_id = {t.task_id: t for t in list_task_snapshots()}
    tasks_by_id.update(_tasks)
    return [
        {
            "task_id": t.task_id,
            "status": t.status.value,
            "created_at": t.created_at,
            "workspace_id": t.workspace_id,
            "source_type": t.source_type,
        }
        for t in sorted(tasks_by_id.values(), key=lambda item: item.created_at, reverse=True)
    ]


@router.get("/{task_id}")
async def get_task(task_id: str):
    task = _get_task_record(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return _task_payload(task)


@router.get("/{task_id}/result")
async def get_task_result(task_id: str):
    task = _get_task_record(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    report_row = load_task_report(task_id)
    persisted_markdown = report_row.get("content_markdown") if report_row else None
    report_id = report_row.get("report_id") if report_row else task.persisted_report_id
    report_kind = report_row.get("report_kind") if report_row else (
        "research_report" if task.source_type == "research" else "final_report"
    )
    return {
        "task_id": task.task_id,
        "workspace_id": task.workspace_id,
        "status": task.status.value,
        "source_type": task.source_type,
        "report_kind": report_kind,
        "report_id": report_id,
        "persisted": bool(report_row or task.persisted_to_db),
        "result_markdown": persisted_markdown or task.result_markdown,
        "draft_markdown": task.draft_markdown,
        "full_markdown": task.full_markdown,
        "brief": task.brief,
        "search_plan": task.search_plan,
        "rag_result": task.rag_result,
        "paper_cards": task.paper_cards,
        "compression_result": task.compression_result,
        "taxonomy": task.taxonomy,
        "draft_report": task.draft_report,
        "review_feedback": task.review_feedback,
        "review_passed": task.review_passed,
        "artifacts_created": task.artifacts_created,
        "artifact_count": task.artifact_count,
        "collaboration_trace": task.collaboration_trace,
        "supervisor_mode": task.supervisor_mode,
        "awaiting_followup": task.awaiting_followup,
        "followup_resolution": task.followup_resolution,
    }


@router.post("/{task_id}/chat", response_model=ChatResponse)
async def task_chat(task_id: str, payload: ChatRequest) -> ChatResponse:
    task = _get_task_record(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Task is not completed yet")

    report_context = task.report_context_snapshot or task.result_markdown
    if not report_context:
        raise HTTPException(status_code=400, detail="No report context available")

    from src.agent.settings import Settings
    from src.agent.llm import build_chat_llm

    settings = Settings.from_env()
    llm = build_chat_llm(settings)

    system_prompt = SURVEY_CHAT_SYSTEM_PROMPT if task.paper_type == "survey" else REGULAR_CHAT_SYSTEM_PROMPT
    context_prompt = (
        f"original_input: {task.input_value}\n"
        f"paper_type: {task.paper_type or 'regular'}\n"
        f"report_mode: {task.report_mode}\n"
        f"report_context:\n{report_context}\n\n"
    )
    if task.followup_hints:
        context_prompt += "followup_hints:\n" + "\n".join(f"- {x}" for x in task.followup_hints) + "\n\n"

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=context_prompt)]
    for msg in task.chat_history[-8:]:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "user":
            messages.append(HumanMessage(content=content))
    messages.append(HumanMessage(content=payload.message))

    try:
        resp = llm.invoke(messages)
        content = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    task.chat_history.append({"role": "user", "content": payload.message})
    task.chat_history.append({"role": "assistant", "content": content})
    _sync_task_snapshot(task)
    return ChatResponse(role="assistant", content=content)


@router.get("/{task_id}/events")
async def task_events_sse(task_id: str):
    task = _get_task_record(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_stream():
        last_idx = 0
        seen_files: dict[str, int] = {}
        while True:
            while last_idx < len(task.node_events):
                event = task.node_events[last_idx]
                yield f"data: {json.dumps(event)}\n\n"
                last_idx += 1

            for event in _workspace_stream_events(task, seen_files):
                yield f"data: {json.dumps(event)}\n\n"

            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                yield f"data: {json.dumps({'type': 'done', 'status': task.status.value})}\n\n"
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _run_graph_sync(initial_state: dict, node_events: list, graph=None, graph_config: dict | None = None) -> dict | None:
    """Run graph in a thread, streaming node events in real time."""
    from src.graph.builder import build_report_graph

    if graph is None:
        graph = build_report_graph()
    final_report = None
    collected_errors: list[str] = []
    state_snapshot: dict = {}

    stream_kwargs = {"config": graph_config} if graph_config else {}
    for chunk in graph.stream(initial_state, **stream_kwargs):
        for node_name, output in chunk.items():
            if node_name.startswith("__"):
                continue
            if isinstance(output, dict):
                state_snapshot.update(output)
                if "final_report" in output:
                    final_report = output["final_report"]
                if "errors" in output:
                    collected_errors.extend(output["errors"])

    return {"final_report": final_report, "errors": collected_errors, "state": state_snapshot}


def _append_supervisor_trace_events(node_events: list, trace: list[dict]) -> None:
    from datetime import datetime, timezone

    for entry in trace:
        node_name = entry.get("node")
        if not node_name:
            continue
        timestamp = datetime.now(timezone.utc).isoformat()
        node_events.append({
            "type": "node_start",
            "node": node_name,
            "timestamp": timestamp,
        })
        node_events.append({
            "type": "node_end",
            "node": node_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "done",
            "warnings": list(entry.get("warnings") or []),
            "tokens_delta": int(entry.get("tokens_delta") or 0),
            "duration_ms": entry.get("duration_ms"),
            "error": entry.get("error"),
        })


def _run_supervisor_sync(
    supervisor,
    initial_state: dict,
    *,
    use_handoff: bool,
    inputs: dict | None = None,
) -> dict:
    """Run the async supervisor in a worker thread for task background jobs."""

    async def _runner() -> dict:
        if use_handoff:
            return await supervisor.collaborate_with_handoff(
                state=initial_state,
                inputs=inputs,
                user_request=initial_state.get("raw_input"),
            )
        return await supervisor.collaborate(state=initial_state, inputs=inputs)

    return asyncio.run(_runner())


def _build_state_template(report_mode: str) -> dict:
    """Shared state template for both report and research workflows."""
    return {
        "raw_input": "",
        "task_id": "",
        "workspace_id": "",
        "source_type": "arxiv",
        "report_mode": report_mode,
        "research_depth": "full",
        "interaction_mode": "non_interactive",
        "paper_type": None,
        "auto_fill": False,
        "brief": None,
        "search_plan": None,
        "search_plan_warnings": [],
        "rag_result": None,
        "paper_cards": [],
        "compression_result": None,
        "taxonomy": None,
        "review_feedback": None,
        "review_passed": None,
        "artifacts_created": [],
        "artifact_count": 0,
        "current_stage": None,
        "arxiv_id": None,
        "pdf_text": None,
        "source_manifest": None,
        "normalized_doc": None,
        "evidence": None,
        "report_frame": None,
        "draft_report": None,
        "resolved_report": None,
        "verified_report": None,
        "final_report": None,
        "draft_markdown": None,
        "full_markdown": None,
        "followup_hints": [],
        "awaiting_followup": False,
        "followup_resolution": None,
        "tokens_used": 0,
        "warnings": [],
        "errors": [],
        "degradation_mode": "normal",
        "node_statuses": {},
    }


async def _run_graph(task_id: str):
    """Background coroutine that runs the report or research graph and updates task state."""
    from datetime import datetime, timezone

    task = _tasks.get(task_id)
    if not task:
        return

    task.status = TaskStatus.RUNNING
    task.node_events.append({"type": "status_change", "status": "running"})
    _sync_task_snapshot(task)

    try:
        # Create the output workspace directory
        from src.agent.output_workspace import DEFAULT_WORKSPACE_USER, build_workspace_id, create_workspace

        workspace_user = DEFAULT_WORKSPACE_USER or "user"
        if not task.workspace_id:
            task.workspace_id = build_workspace_id(workspace_user)

        create_workspace(task.task_id, {
            "workspace_id": task.workspace_id or "",
            "workspace_opened_at": task.created_at,
            "task_created_at": task.created_at,
            "source_type": task.source_type,
            "report_mode": task.report_mode,
            "input_value": task.input_value,
        }, workspace_id=task.workspace_id, user_id=workspace_user)

        import functools
        from src.agent.checkpointing import build_graph_config

        source_type = getattr(task, "source_type", None) or "arxiv"

        if source_type == "research":
            from src.models.config import SupervisorMode
            from src.research.agents.supervisor import get_supervisor

            supervisor = get_supervisor()
            emitter = NodeEventEmitter()
            emitter.events = task.node_events
            initial_state: dict = {
                **_build_state_template(task.report_mode),
                "task_id": task.task_id,
                "workspace_id": task.workspace_id or "",
                "source_type": "research",
                "raw_input": task.input_value,
                "auto_fill": getattr(task, "auto_fill", False),
            }
            configured_mode = getattr(supervisor.config, "supervisor_mode", SupervisorMode.GRAPH)
            task.supervisor_mode = configured_mode.value if hasattr(configured_mode, "value") else str(configured_mode)
            use_handoff = configured_mode == SupervisorMode.LLM_HANDOFF

            loop = asyncio.get_running_loop()
            supervisor_result = await loop.run_in_executor(
                None,
                functools.partial(
                    _run_supervisor_sync,
                    supervisor,
                    initial_state,
                    use_handoff=use_handoff,
                    inputs={
                        "_event_emitter": emitter,
                        "task_id": task.task_id,
                        "workspace_id": task.workspace_id or "",
                    },
                ),
            )
            supervisor_state = supervisor_result or {}
            trace = list(supervisor_state.get("collaboration_trace") or [])
            task.collaboration_trace = _json_safe(trace) or []
            if trace and not any(event.get("type") == "node_start" for event in task.node_events):
                _append_supervisor_trace_events(task.node_events, trace)
            result = {
                "final_report": supervisor_state.get("final_report"),
                "errors": list(supervisor_state.get("errors") or []),
                "state": supervisor_state,
            }
        else:
            from src.graph.builder import build_report_graph
            
            emitter = NodeEventEmitter()
            emitter.events = task.node_events

            graph = build_report_graph(emitter, use_checkpointer=True)
            if task.input_type == "pdf":
                initial_state = {
                    **_build_state_template(task.report_mode),
                    "task_id": task.task_id,
                    "workspace_id": task.workspace_id or "",
                    "source_type": "pdf",
                    "pdf_text": task.input_value,
                }
            else:
                initial_state = {
                    **_build_state_template(task.report_mode),
                    "task_id": task.task_id,
                    "workspace_id": task.workspace_id or "",
                    "raw_input": task.input_value,
                }
            graph_config = build_graph_config(
                "report_graph",
                thread_id=task.task_id,
                recursion_limit=80,
            )

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                functools.partial(_run_graph_sync, initial_state, task.node_events, graph, graph_config),
            )

        final = result.get("final_report") if result else None
        state_result = result.get("state", {}) if result else {}
        task.paper_type = state_result.get("paper_type")
        task.current_stage = state_result.get("current_stage")
        task.followup_hints = list(state_result.get("followup_hints") or [])
        task.brief = _json_safe(state_result.get("brief"))
        task.search_plan = _json_safe(state_result.get("search_plan"))
        task.rag_result = _json_safe(state_result.get("rag_result"))
        task.paper_cards = _json_safe(state_result.get("paper_cards")) or []
        task.compression_result = _json_safe(state_result.get("compression_result"))
        task.taxonomy = _json_safe(state_result.get("taxonomy"))
        task.draft_report = _json_safe(state_result.get("draft_report"))
        task.review_feedback = _json_safe(state_result.get("review_feedback"))
        task.review_passed = state_result.get("review_passed")
        task.artifacts_created = _json_safe(state_result.get("artifacts_created")) or []
        task.artifact_count = int(state_result.get("artifact_count") or 0)
        task.awaiting_followup = bool(state_result.get("awaiting_followup", False))
        task.followup_resolution = _json_safe(state_result.get("followup_resolution"))

        if source_type == "research":
            brief = task.brief
            search_plan = task.search_plan
            markdown = (
                state_result.get("result_markdown")
                or state_result.get("draft_markdown")
                or state_result.get("full_markdown")
            )
            if not markdown and final:
                from src.agent.report import _final_report_to_markdown

                markdown = _final_report_to_markdown(final)

            if markdown:
                task.result_markdown = str(markdown)
                task.draft_markdown = str(markdown)
                task.report_context_snapshot = task.result_markdown
                task.status = TaskStatus.COMPLETED
            elif brief:
                # No markdown generated - likely due to retrieval failure
                # Generate a helpful message report instead of JSON dump
                import json as _json

                errors = state_result.get("errors", [])
                rag_result = task.rag_result or {}

                # Build a descriptive error/fallback report
                topic = brief.get("topic", task.input_value or "Unknown topic")
                sub_questions = brief.get("sub_questions", [])
                total_papers = rag_result.get("total_papers", 0) if isinstance(rag_result, dict) else 0

                error_report = f"""# Research Report: {topic}

## Status

**Status**: Partial completion - no papers retrieved

## Research Brief

**Topic**: {topic}

**Sub-questions**:
{chr(10).join(f"- {q}" for q in sub_questions) if sub_questions else "- (none specified)"}

**Confidence**: {brief.get("confidence", "unknown")}

## Issues

"""
                if errors:
                    error_report += f"""### Errors Encountered

{chr(10).join(f"- {e}" for e in errors if isinstance(e, str))}

"""
                if total_papers == 0:
                    error_report += f"""### Retrieval Result

No papers were retrieved for this topic. Possible reasons:
- The search service may be unavailable
- The topic may not have relevant papers indexed
- Search queries may need refinement

## Recommendations

1. Verify that the search service (SearXNG) is running
2. Try a more specific topic
3. Check the RAG pipeline logs for details

## Raw Data

For debugging, the following data was collected:

- **Brief**: Available in `brief.json`
- **Search Plan**: {'Available' if search_plan else 'Not generated'}
- **Papers Retrieved**: {total_papers}
"""
                else:
                    error_report += f"""### Retrieval Result

{total_papers} paper(s) were retrieved but could not be processed into a report.
Please check the system logs for details.

"""

                task.result_markdown = error_report
                task.draft_markdown = error_report
                task.report_context_snapshot = error_report

                if search_plan or brief.get("needs_followup"):
                    task.status = TaskStatus.COMPLETED
                else:
                    task.error = "; ".join(errors) if errors else "SearchPlanAgent produced no plan"
                    task.status = TaskStatus.FAILED
            else:
                errors = result.get("errors", []) if result else []
                task.error = "; ".join(errors) if errors else "ClarifyAgent produced no brief"
                task.status = TaskStatus.FAILED
        elif final:
            from src.agent.report import _final_report_to_markdown

            markdown = _final_report_to_markdown(final)
            task.result_markdown = markdown
            if task.report_mode == "full":
                task.full_markdown = state_result.get("full_markdown") or markdown
                task.report_context_snapshot = task.full_markdown
            else:
                task.draft_markdown = state_result.get("draft_markdown") or markdown
                task.report_context_snapshot = task.draft_markdown
            task.status = TaskStatus.COMPLETED
        else:
            errors = result.get("errors", []) if result else []
            task.error = "; ".join(errors) if errors else "No report generated"
            task.status = TaskStatus.FAILED

        if task.result_markdown:
            report_id = save_task_report(
                task=task,
                report_kind="research_report" if source_type == "research" else "final_report",
                content_markdown=task.result_markdown,
                content_json={
                    "brief": task.brief,
                    "search_plan": task.search_plan,
                    "rag_result": task.rag_result,
                    "paper_cards": task.paper_cards,
                    "compression_result": task.compression_result,
                    "taxonomy": task.taxonomy,
                    "draft_report": task.draft_report,
                    "review_feedback": task.review_feedback,
                    "review_passed": task.review_passed,
                    "artifacts_created": task.artifacts_created,
                    "artifact_count": task.artifact_count,
                    "collaboration_trace": task.collaboration_trace,
                    "supervisor_mode": task.supervisor_mode,
                    "awaiting_followup": task.awaiting_followup,
                    "followup_resolution": task.followup_resolution,
                },
            )
            if report_id:
                task.persisted_report_id = report_id

            # Write to output workspace
            from src.agent.output_workspace import write_report
            write_report(task.task_id, task.result_markdown, workspace_id=task.workspace_id)

    except Exception as e:
        task.error = str(e)
        task.status = TaskStatus.FAILED

    task.completed_at = datetime.now(timezone.utc).isoformat()
    task.node_events.append({"type": "status_change", "status": task.status.value})
    _sync_task_snapshot(task)


def get_tasks_store():
    """Expose store for testing."""
    return _tasks


def clear_tasks_store():
    """Clear store (for testing)."""
    _tasks.clear()
