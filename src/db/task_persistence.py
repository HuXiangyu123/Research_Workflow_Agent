"""Long-lived task/report persistence for the task API."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any
from uuid import uuid4

from sqlalchemy import Boolean, Float, Index, Integer, JSON, String, Text, delete, select, text
from sqlalchemy.orm import Mapped, mapped_column

from src.db.engine import Base, get_db_session, get_engine
from src.models.task import TaskRecord, TaskStatus

logger = logging.getLogger(__name__)


class PersistedTask(Base):
    __tablename__ = "persisted_tasks"

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    input_type: Mapped[str] = mapped_column(String(32), nullable=False)
    input_value: Mapped[str] = mapped_column(Text, nullable=False)
    report_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    auto_fill: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    workspace_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    paper_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    completed_at: Mapped[str | None] = mapped_column(String(64), nullable=True)
    draft_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    full_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    brief: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    search_plan: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    rag_result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    paper_cards: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    compression_result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    taxonomy: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    draft_report: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    current_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    report_context_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    followup_hints: Mapped[list[str]] = mapped_column(JSON, default=list)
    awaiting_followup: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    followup_resolution: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    chat_history: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    chat_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    node_events: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    review_feedback: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    review_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    artifacts_created: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    artifact_count: Mapped[int] = mapped_column(Integer, default=0)
    collaboration_trace: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    supervisor_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    persisted_report_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    persistence_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[float] = mapped_column(Float, default=time.time, onupdate=time.time)


class PersistedReport(Base):
    __tablename__ = "persisted_reports"

    report_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: f"rep_{uuid4().hex[:16]}",
    )
    task_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    workspace_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    report_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    content_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    updated_at: Mapped[float] = mapped_column(Float, default=time.time, onupdate=time.time)

    __table_args__ = (
        Index("ix_persisted_reports_task_kind", "task_id", "report_kind", unique=True),
    )


_TABLES_READY: bool | None = None

_OPTIONAL_TASK_COLUMNS: dict[str, str] = {
    "auto_fill": "BOOLEAN DEFAULT FALSE",
    "rag_result": "JSON",
    "paper_cards": "JSON",
    "compression_result": "JSON",
    "taxonomy": "JSON",
    "draft_report": "JSON",
    "awaiting_followup": "BOOLEAN DEFAULT FALSE",
    "followup_resolution": "JSON",
    "artifacts_created": "JSON",
    "artifact_count": "INTEGER",
    "collaboration_trace": "JSON",
    "supervisor_mode": "VARCHAR(32)",
}


def _database_configured() -> bool:
    return bool(os.getenv("DATABASE_URL", "").strip())


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def is_task_persistence_available() -> bool:
    return _ensure_tables()


def _ensure_tables() -> bool:
    global _TABLES_READY
    if not _database_configured():
        return False
    if _TABLES_READY is True:
        return True
    if _TABLES_READY is False:
        return False
    try:
        engine = get_engine()
        Base.metadata.create_all(
            engine,
            tables=[PersistedTask.__table__, PersistedReport.__table__],
            checkfirst=True,
        )
        # create_all(checkfirst=True) does not evolve existing installations.
        # Keep the task API schema aligned without requiring Alembic for tests/dev.
        with engine.begin() as conn:
            for column_name, column_type in _OPTIONAL_TASK_COLUMNS.items():
                conn.execute(
                    text(
                        f"ALTER TABLE persisted_tasks "
                        f"ADD COLUMN IF NOT EXISTS {column_name} {column_type}"
                    )
                )
        _TABLES_READY = True
        return True
    except Exception as exc:
        logger.warning("task persistence disabled: %s", exc)
        _TABLES_READY = False
        return False


def reset_task_persistence_state() -> None:
    global _TABLES_READY
    _TABLES_READY = None


def upsert_task_snapshot(task: TaskRecord) -> bool:
    if not _ensure_tables():
        return False

    payload = {
        "task_id": task.task_id,
        "status": task.status.value,
        "input_type": task.input_type,
        "input_value": task.input_value,
        "report_mode": task.report_mode,
        "source_type": task.source_type,
        "auto_fill": task.auto_fill,
        "workspace_id": task.workspace_id,
        "paper_type": task.paper_type,
        "created_at": task.created_at,
        "completed_at": task.completed_at,
        "draft_markdown": task.draft_markdown,
        "full_markdown": task.full_markdown,
        "result_markdown": task.result_markdown,
        "brief": _json_safe(task.brief),
        "search_plan": _json_safe(task.search_plan),
        "rag_result": _json_safe(task.rag_result),
        "paper_cards": _json_safe(task.paper_cards) or [],
        "compression_result": _json_safe(task.compression_result),
        "taxonomy": _json_safe(task.taxonomy),
        "draft_report": _json_safe(task.draft_report),
        "current_stage": task.current_stage,
        "report_context_snapshot": task.report_context_snapshot,
        "followup_hints": _json_safe(task.followup_hints) or [],
        "awaiting_followup": task.awaiting_followup,
        "followup_resolution": _json_safe(task.followup_resolution),
        "chat_history": _json_safe(task.chat_history) or [],
        "chat_summary": task.chat_summary,
        "error": task.error,
        "node_events": _json_safe(task.node_events) or [],
        "review_feedback": _json_safe(task.review_feedback),
        "review_passed": task.review_passed,
        "artifacts_created": _json_safe(task.artifacts_created) or [],
        "artifact_count": task.artifact_count,
        "collaboration_trace": _json_safe(task.collaboration_trace) or [],
        "supervisor_mode": task.supervisor_mode,
        "persisted_report_id": task.persisted_report_id,
        "persistence_error": task.persistence_error,
    }

    with get_db_session() as session:
        existing = session.get(PersistedTask, task.task_id)
        if existing is None:
            session.add(PersistedTask(**payload))
        else:
            for key, value in payload.items():
                setattr(existing, key, value)
    return True


def save_task_report(
    *,
    task: TaskRecord,
    report_kind: str,
    content_markdown: str | None,
    content_json: dict[str, Any] | None,
) -> str | None:
    if not _ensure_tables():
        return None

    with get_db_session() as session:
        existing = session.execute(
            select(PersistedReport).where(
                PersistedReport.task_id == task.task_id,
                PersistedReport.report_kind == report_kind,
            )
        ).scalar_one_or_none()

        if existing is None:
            report = PersistedReport(
                task_id=task.task_id,
                workspace_id=task.workspace_id,
                source_type=task.source_type,
                report_kind=report_kind,
                content_markdown=content_markdown,
                content_json=_json_safe(content_json),
            )
            session.add(report)
            session.flush()
            return report.report_id

        existing.workspace_id = task.workspace_id
        existing.source_type = task.source_type
        existing.content_markdown = content_markdown
        existing.content_json = _json_safe(content_json)
        session.flush()
        return existing.report_id


def load_task_snapshot(task_id: str) -> TaskRecord | None:
    if not _ensure_tables():
        return None

    with get_db_session() as session:
        row = session.get(PersistedTask, task_id)
        if row is None:
            return None
        return TaskRecord(
            task_id=row.task_id,
            status=TaskStatus(row.status),
            input_type=row.input_type,
            input_value=row.input_value,
            report_mode=row.report_mode,
            source_type=row.source_type,
            auto_fill=getattr(row, 'auto_fill', False),
            workspace_id=row.workspace_id,
            paper_type=row.paper_type,
            created_at=row.created_at,
            completed_at=row.completed_at,
            draft_markdown=row.draft_markdown,
            full_markdown=row.full_markdown,
            result_markdown=row.result_markdown,
            brief=row.brief,
            search_plan=row.search_plan,
            rag_result=row.rag_result,
            paper_cards=row.paper_cards or [],
            compression_result=row.compression_result,
            taxonomy=row.taxonomy,
            draft_report=row.draft_report,
            current_stage=row.current_stage,
            report_context_snapshot=row.report_context_snapshot,
            followup_hints=row.followup_hints or [],
            awaiting_followup=bool(getattr(row, "awaiting_followup", False)),
            followup_resolution=getattr(row, "followup_resolution", None),
            chat_history=row.chat_history or [],
            chat_summary=row.chat_summary,
            error=row.error,
            node_events=row.node_events or [],
            review_feedback=row.review_feedback,
            review_passed=row.review_passed,
            artifacts_created=row.artifacts_created or [],
            artifact_count=row.artifact_count or 0,
            collaboration_trace=row.collaboration_trace or [],
            supervisor_mode=row.supervisor_mode,
            persisted_to_db=True,
            persisted_report_id=row.persisted_report_id,
            persistence_error=row.persistence_error,
        )


def list_task_snapshots() -> list[TaskRecord]:
    if not _ensure_tables():
        return []

    with get_db_session() as session:
        rows = session.execute(
            select(PersistedTask).order_by(PersistedTask.created_at.desc())
        ).scalars()
        return [
            TaskRecord(
                task_id=row.task_id,
                status=TaskStatus(row.status),
                input_type=row.input_type,
                input_value=row.input_value,
                report_mode=row.report_mode,
                source_type=row.source_type,
                auto_fill=getattr(row, 'auto_fill', False),
                workspace_id=row.workspace_id,
                paper_type=row.paper_type,
                created_at=row.created_at,
                completed_at=row.completed_at,
                draft_markdown=row.draft_markdown,
                full_markdown=row.full_markdown,
                result_markdown=row.result_markdown,
                brief=row.brief,
                search_plan=row.search_plan,
                rag_result=row.rag_result,
                paper_cards=row.paper_cards or [],
                compression_result=row.compression_result,
                taxonomy=row.taxonomy,
                draft_report=row.draft_report,
                current_stage=row.current_stage,
                report_context_snapshot=row.report_context_snapshot,
                followup_hints=row.followup_hints or [],
                awaiting_followup=bool(getattr(row, "awaiting_followup", False)),
                followup_resolution=getattr(row, "followup_resolution", None),
                chat_history=row.chat_history or [],
                chat_summary=row.chat_summary,
                error=row.error,
                node_events=row.node_events or [],
                review_feedback=row.review_feedback,
                review_passed=row.review_passed,
                artifacts_created=row.artifacts_created or [],
                artifact_count=row.artifact_count or 0,
                collaboration_trace=row.collaboration_trace or [],
                supervisor_mode=row.supervisor_mode,
                persisted_to_db=True,
                persisted_report_id=row.persisted_report_id,
                persistence_error=row.persistence_error,
            )
            for row in rows
        ]


def load_task_report(task_id: str) -> dict[str, Any] | None:
    if not _ensure_tables():
        return None

    with get_db_session() as session:
        row = session.execute(
            select(PersistedReport)
            .where(PersistedReport.task_id == task_id)
            .order_by(PersistedReport.updated_at.desc())
        ).scalars().first()
        if row is None:
            return None
        return {
            "report_id": row.report_id,
            "task_id": row.task_id,
            "workspace_id": row.workspace_id,
            "source_type": row.source_type,
            "report_kind": row.report_kind,
            "content_markdown": row.content_markdown,
            "content_json": row.content_json,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }


def clear_task_persistence() -> None:
    if not _ensure_tables():
        return
    with get_db_session() as session:
        session.execute(delete(PersistedReport))
        session.execute(delete(PersistedTask))


def delete_task_persistence(task_id: str) -> None:
    if not _ensure_tables():
        return
    with get_db_session() as session:
        session.execute(
            delete(PersistedReport).where(PersistedReport.task_id == task_id)
        )
        session.execute(
            delete(PersistedTask).where(PersistedTask.task_id == task_id)
        )
