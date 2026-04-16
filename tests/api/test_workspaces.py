from __future__ import annotations

from fastapi.testclient import TestClient

from src.agent import output_workspace as ow
from src.api.app import app


client = TestClient(app)


def _set_output_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ow, "OUTPUT_ROOT", tmp_path)


def test_workspace_artifacts_api_reads_disk_backed_workspace(tmp_path, monkeypatch) -> None:
    _set_output_root(tmp_path, monkeypatch)

    workspace_id = "user_20260416T123000Z_aaa111"
    ow.create_workspace(
        "task-api-1",
        {
            "workspace_opened_at": "2026-04-16T12:30:00+00:00",
            "source_type": "research",
            "report_mode": "draft",
            "input_value": "rag benchmarking",
        },
        workspace_id=workspace_id,
        user_id="user",
    )
    ow.write_brief("task-api-1", {"topic": "RAG benchmarking"}, workspace_id=workspace_id)
    ow.write_rag_result(
        "task-api-1",
        {"query": "rag benchmarking", "paper_candidates": [{"title": "Paper 1"}], "total_papers": 1},
        workspace_id=workspace_id,
    )
    ow.write_review_feedback(
        "task-api-1",
        {"summary": "review summary"},
        workspace_id=workspace_id,
    )

    summary_resp = client.get(f"/api/v1/workspaces/{workspace_id}")
    assert summary_resp.status_code == 200
    assert summary_resp.json()["artifact_count"] >= 3

    artifacts_resp = client.get(f"/api/v1/workspaces/{workspace_id}/artifacts")
    assert artifacts_resp.status_code == 200
    items = artifacts_resp.json()["items"]
    artifact_types = {item["artifact_type"] for item in items}
    assert "brief" in artifact_types
    assert "rag_result" in artifact_types
    assert "review_feedback" in artifact_types
    assert all(item["content_ref"] for item in items)

    report_path = ow.write_draft("task-api-1", "# Draft\n\nhello", workspace_id=workspace_id)
    assert report_path.is_file()
    artifacts_resp = client.get(f"/api/v1/workspaces/{workspace_id}/artifacts")
    report_artifact = next(
        item for item in artifacts_resp.json()["items"] if item["title"].startswith("Draft markdown")
    )
    content_resp = client.get(
        f"/api/v1/workspaces/{workspace_id}/artifacts/{report_artifact['artifact_id']}/content"
    )
    assert content_resp.status_code == 200
    assert content_resp.json()["content_type"] == "markdown"
    assert "hello" in content_resp.json()["content"]


def test_create_workspace_endpoint_creates_empty_workspace(tmp_path, monkeypatch) -> None:
    _set_output_root(tmp_path, monkeypatch)

    resp = client.post(
        "/api/v1/workspaces",
        json={
            "workspace_id": "user_20260416T123500Z_ccc333",
            "user_id": "user",
            "source_type": "research",
            "report_mode": "draft",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["workspace_id"] == "user_20260416T123500Z_ccc333"
    assert data["status"] == "active"
    assert data["artifact_count"] == 0

    manifest = (tmp_path / "workspaces" / data["workspace_id"] / "workspace.json").read_text(encoding="utf-8")
    assert "\"task_ids\": []" in manifest


def test_list_workspaces_endpoint_reads_output_backed_history(tmp_path, monkeypatch) -> None:
    _set_output_root(tmp_path, monkeypatch)

    ws_old = ow.ensure_workspace_root(
        workspace_id="user_20260416T120000Z_old111",
        user_id="user",
        metadata={"source_type": "research", "report_mode": "draft"},
    )
    ws_new = ow.ensure_workspace_root(
        workspace_id="user_20260416T130000Z_new222",
        user_id="user",
        metadata={"source_type": "research", "report_mode": "draft"},
    )
    ow.create_workspace(
        "task-old",
        {"workspace_opened_at": "2026-04-16T12:00:00+00:00", "source_type": "research", "report_mode": "draft"},
        workspace_id=ws_old,
        user_id="user",
    )
    ow.create_workspace(
        "task-new",
        {"workspace_opened_at": "2026-04-16T13:00:00+00:00", "source_type": "research", "report_mode": "draft"},
        workspace_id=ws_new,
        user_id="user",
    )
    ow.write_brief("task-new", {"topic": "new workspace"}, workspace_id=ws_new)

    resp = client.get("/api/v1/workspaces")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert data["items"][0]["workspace_id"] == ws_new
    assert data["items"][0]["latest_task_id"] == "task-new"


def test_workspace_create_artifact_persists_to_disk(tmp_path, monkeypatch) -> None:
    _set_output_root(tmp_path, monkeypatch)

    workspace_id = "user_20260416T124500Z_bbb222"
    ow.create_workspace(
        "task-api-2",
        {
            "workspace_opened_at": "2026-04-16T12:45:00+00:00",
            "source_type": "research",
            "report_mode": "draft",
            "input_value": "agent evaluation",
        },
        workspace_id=workspace_id,
        user_id="user",
    )

    create_resp = client.post(
        f"/api/v1/workspaces/{workspace_id}/artifacts",
        json={
            "artifact_type": "task_log",
            "title": "Manual note",
            "task_id": "task-api-2",
            "summary": "operator summary",
            "metadata": {"source": "api-test"},
        },
    )
    assert create_resp.status_code == 200
    artifact_id = create_resp.json()["artifact_id"]

    artifacts_resp = client.get(f"/api/v1/workspaces/{workspace_id}/artifacts")
    assert artifacts_resp.status_code == 200
    items = artifacts_resp.json()["items"]
    created = next(item for item in items if item["artifact_id"] == artifact_id)
    assert created["artifact_type"] == "task_log"
    assert created["metadata"]["source"] == "api-test"
