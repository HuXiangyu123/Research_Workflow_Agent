from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.agent import output_workspace as ow
from src.models.workspace import ArtifactType, WorkspaceArtifact


def _set_output_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ow, "OUTPUT_ROOT", tmp_path)


def test_build_workspace_id_includes_user_and_timestamp() -> None:
    opened_at = datetime(2026, 4, 16, 9, 30, 0, tzinfo=timezone.utc)
    workspace_id = ow.build_workspace_id("user", opened_at=opened_at)

    assert workspace_id.startswith("user_20260416T093000Z_")
    assert len(workspace_id.split("_")) == 3


def test_create_workspace_uses_workspace_first_layout(tmp_path: Path, monkeypatch) -> None:
    _set_output_root(tmp_path, monkeypatch)

    workspace_id = "user_20260416T093000Z_ab12cd"
    metadata = {
        "workspace_opened_at": "2026-04-16T09:30:00+00:00",
        "source_type": "research",
        "report_mode": "draft",
        "input_value": "test topic",
    }
    task_dir = ow.create_workspace(
        "task-001",
        metadata,
        workspace_id=workspace_id,
        user_id="user",
    )

    expected = tmp_path / "workspaces" / workspace_id / "tasks" / "task-001"
    assert task_dir == expected
    assert (task_dir / "revisions").is_dir()

    task_meta = json.loads((task_dir / "metadata.json").read_text(encoding="utf-8"))
    assert task_meta["task_id"] == "task-001"
    assert task_meta["workspace_id"] == workspace_id
    assert task_meta["user_id"] == "user"
    assert task_meta["source_type"] == "research"

    manifest = json.loads(
        (tmp_path / "workspaces" / workspace_id / "workspace.json").read_text(encoding="utf-8")
    )
    assert manifest["workspace_id"] == workspace_id
    assert manifest["user_id"] == "user"
    assert manifest["task_ids"] == ["task-001"]
    assert manifest["opened_at"] == "2026-04-16T09:30:00+00:00"


def test_ensure_workspace_root_creates_empty_manifest(tmp_path: Path, monkeypatch) -> None:
    _set_output_root(tmp_path, monkeypatch)

    workspace_id = ow.ensure_workspace_root(
        workspace_id="user_20260416T101500Z_ee1122",
        user_id="user",
        metadata={"source_type": "research", "report_mode": "draft"},
    )

    manifest = json.loads(
        (tmp_path / "workspaces" / workspace_id / "workspace.json").read_text(encoding="utf-8")
    )
    assert manifest["workspace_id"] == workspace_id
    assert manifest["user_id"] == "user"
    assert manifest["task_ids"] == []
    assert manifest["source_type"] == "research"
    assert manifest["report_mode"] == "draft"


def test_write_report_and_revision_under_workspace(tmp_path: Path, monkeypatch) -> None:
    _set_output_root(tmp_path, monkeypatch)

    workspace_id = "user_20260416T100000Z_fed321"
    ow.create_workspace(
        "task-002",
        {
            "workspace_opened_at": "2026-04-16T10:00:00+00:00",
            "source_type": "arxiv",
            "report_mode": "full",
            "input_value": "1706.03762",
        },
        workspace_id=workspace_id,
        user_id="user",
    )

    ow.write_draft("task-002", "# draft", workspace_id=workspace_id)
    rev_path = ow.append_revision("task-002", "# revision", label="initial", workspace_id=workspace_id)
    report_path = ow.write_report("task-002", "# final", workspace_id=workspace_id)

    task_dir = tmp_path / "workspaces" / workspace_id / "tasks" / "task-002"
    assert (task_dir / "draft.md").read_text(encoding="utf-8") == "# draft"
    assert rev_path.name == "001_initial.md"
    assert report_path == task_dir / "report.md"

    task_meta = json.loads((task_dir / "metadata.json").read_text(encoding="utf-8"))
    assert task_meta["completed_at"]

    manifest = json.loads(
        (tmp_path / "workspaces" / workspace_id / "workspace.json").read_text(encoding="utf-8")
    )
    assert manifest["latest_task_id"] == "task-002"
    assert manifest["completed_at"]


def test_legacy_task_layout_still_supported(tmp_path: Path, monkeypatch) -> None:
    _set_output_root(tmp_path, monkeypatch)

    path = ow.write_report("legacy-task", "legacy report")

    assert path == tmp_path / "legacy-task" / "report.md"
    assert path.read_text(encoding="utf-8") == "legacy report"


def test_list_workspace_artifacts_reads_disk_files_and_custom_artifacts(tmp_path: Path, monkeypatch) -> None:
    _set_output_root(tmp_path, monkeypatch)

    workspace_id = "user_20260416T120000Z_abc123"
    ow.create_workspace(
        "task-003",
        {
            "workspace_opened_at": "2026-04-16T12:00:00+00:00",
            "source_type": "research",
            "report_mode": "draft",
            "input_value": "rag evaluation",
        },
        workspace_id=workspace_id,
        user_id="user",
    )
    ow.write_brief("task-003", {"topic": "RAG evaluation"}, workspace_id=workspace_id)
    ow.write_search_plan("task-003", {"plan_goal": "collect papers"}, workspace_id=workspace_id)
    ow.write_rag_result(
        "task-003",
        {"query": "rag evaluation", "paper_candidates": [{"title": "p1"}], "total_papers": 1},
        workspace_id=workspace_id,
    )
    ow.write_draft("task-003", "# Draft", workspace_id=workspace_id)
    ow.write_review_feedback(
        "task-003",
        {"summary": "Needs more grounding"},
        workspace_id=workspace_id,
    )

    custom = WorkspaceArtifact(
        workspace_id=workspace_id,
        task_id="task-003",
        artifact_type=ArtifactType.TASK_LOG,
        title="Operator note",
        summary="manual note",
    )
    ow.create_workspace_artifact(custom)

    artifacts = ow.list_workspace_artifacts(workspace_id)
    artifact_types = {artifact.artifact_type for artifact in artifacts}
    titles = {artifact.title for artifact in artifacts}

    assert ArtifactType.BRIEF in artifact_types
    assert ArtifactType.SEARCH_PLAN in artifact_types
    assert ArtifactType.RAG_RESULT in artifact_types
    assert ArtifactType.REPORT_DRAFT in artifact_types
    assert ArtifactType.REVIEW_FEEDBACK in artifact_types
    assert ArtifactType.TASK_LOG in artifact_types
    assert "Operator note" in titles
