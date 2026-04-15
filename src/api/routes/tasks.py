from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.graph.callbacks import NodeEventEmitter
from src.models.task import TaskRecord, TaskStatus
from src.agent.report_frame import REGULAR_CHAT_SYSTEM_PROMPT, SURVEY_CHAT_SYSTEM_PROMPT

router = APIRouter(prefix="/tasks", tags=["tasks"])

_tasks: dict[str, TaskRecord] = {}


class CreateTaskRequest(BaseModel):
    input_type: str = Field(default="arxiv", description="'arxiv' or 'pdf'")
    input_value: str = Field(..., description="arXiv ID/URL or PDF text")
    report_mode: str = Field(default="draft", description="'draft' or 'full'")
    source_type: str = Field(
        default="arxiv",
        description="'arxiv', 'pdf' (paper-ingestion) or 'research' (clarify workflow)",
    )


class CreateTaskResponse(BaseModel):
    task_id: str
    status: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    role: str
    content: str


@router.post("", response_model=CreateTaskResponse)
async def create_task(req: CreateTaskRequest) -> CreateTaskResponse:
    source_type = req.source_type if req.source_type in {"arxiv", "pdf", "research"} else "arxiv"
    task = TaskRecord(
        input_type=req.input_type,
        input_value=req.input_value,
        report_mode=req.report_mode if req.report_mode in {"draft", "full"} else "draft",
        source_type=source_type,
    )
    _tasks[task.task_id] = task

    asyncio.create_task(_run_graph(task.task_id))

    return CreateTaskResponse(task_id=task.task_id, status=task.status.value)


@router.get("")
async def list_tasks():
    return [
        {
            "task_id": t.task_id,
            "status": t.status.value,
            "created_at": t.created_at,
            "source_type": t.source_type,
        }
        for t in _tasks.values()
    ]


@router.get("/{task_id}")
async def get_task(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": task.task_id,
        "status": task.status.value,
        "created_at": task.created_at,
        "completed_at": task.completed_at,
        "report_mode": task.report_mode,
        "source_type": task.source_type,
        "paper_type": task.paper_type,
        "draft_markdown": task.draft_markdown,
        "full_markdown": task.full_markdown,
        "result_markdown": task.result_markdown,
        "brief": task.brief,
        "search_plan": task.search_plan,
        "current_stage": task.current_stage,
        "followup_hints": task.followup_hints,
        "chat_history": task.chat_history,
        "error": task.error,
    }


@router.post("/{task_id}/chat", response_model=ChatResponse)
async def task_chat(task_id: str, payload: ChatRequest) -> ChatResponse:
    task = _tasks.get(task_id)
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
    return ChatResponse(role="assistant", content=content)


@router.get("/{task_id}/events")
async def task_events_sse(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_stream():
        last_idx = 0
        while True:
            while last_idx < len(task.node_events):
                event = task.node_events[last_idx]
                yield f"data: {json.dumps(event)}\n\n"
                last_idx += 1

            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                yield f"data: {json.dumps({'type': 'done', 'status': task.status.value})}\n\n"
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _run_graph_sync(initial_state: dict, node_events: list, graph=None) -> dict | None:
    """Run graph in a thread, streaming node events in real time."""
    from src.graph.builder import build_report_graph

    if graph is None:
        graph = build_report_graph()
    final_report = None
    collected_errors: list[str] = []
    state_snapshot: dict = {}

    for chunk in graph.stream(initial_state):
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


def _build_state_template(report_mode: str) -> dict:
    """Shared state template for both report and research workflows."""
    return {
        "raw_input": "",
        "source_type": "arxiv",
        "report_mode": report_mode,
        "paper_type": None,
        "brief": None,
        "search_plan": None,
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

    try:
        import functools

        # Support research-mode tasks that pass source_type explicitly
        source_type = getattr(task, "source_type", None) or "arxiv"

        if source_type == "research":
            from src.research.graph.builder import build_research_graph
            
            emitter = NodeEventEmitter()
            emitter.events = task.node_events

            graph = build_research_graph(emitter)
            initial_state: dict = {
                **_build_state_template(task.report_mode),
                "source_type": "research",
                "raw_input": task.input_value,
            }
        else:
            from src.graph.builder import build_report_graph
            
            emitter = NodeEventEmitter()
            emitter.events = task.node_events

            graph = build_report_graph(emitter)
            if task.input_type == "pdf":
                initial_state = {
                    **_build_state_template(task.report_mode),
                    "source_type": "pdf",
                    "pdf_text": task.input_value,
                }
            else:
                initial_state = {
                    **_build_state_template(task.report_mode),
                    "raw_input": task.input_value,
                }

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            functools.partial(_run_graph_sync, initial_state, task.node_events, graph),
        )

        final = result.get("final_report") if result else None
        state_result = result.get("state", {}) if result else {}
        task.paper_type = state_result.get("paper_type")
        task.current_stage = state_result.get("current_stage")
        task.followup_hints = list(state_result.get("followup_hints") or [])

        if source_type == "research":
            brief = state_result.get("brief")
            search_plan = state_result.get("search_plan")
            task.brief = brief
            task.search_plan = search_plan
            if brief:
                import json as _json

                result_payload = {
                    "brief": brief,
                    "search_plan": search_plan,
                }
                task.result_markdown = _json.dumps(result_payload, ensure_ascii=False, indent=2)
                task.report_context_snapshot = task.result_markdown
                if search_plan or brief.get("needs_followup"):
                    task.status = TaskStatus.COMPLETED
                else:
                    errors = result.get("errors", []) if result else []
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

    except Exception as e:
        task.error = str(e)
        task.status = TaskStatus.FAILED

    task.completed_at = datetime.now(timezone.utc).isoformat()
    task.node_events.append({"type": "status_change", "status": task.status.value})


def get_tasks_store():
    """Expose store for testing."""
    return _tasks


def clear_tasks_store():
    """Clear store (for testing)."""
    _tasks.clear()
