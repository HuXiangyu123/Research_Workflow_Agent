from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.routes.tasks import clear_tasks_store, get_tasks_store
from src.models.task import TaskStatus


@pytest.fixture(autouse=True)
def clean_store():
    clear_tasks_store()
    yield
    clear_tasks_store()


client = TestClient(app)


def test_list_tasks_empty():
    resp = client.get("/tasks")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_task():
    with patch("src.api.routes.tasks._run_graph"):
        resp = client.post("/tasks", json={"input_type": "arxiv", "input_value": "1706.03762", "report_mode": "full"})
    assert resp.status_code == 200
    data = resp.json()
    assert "task_id" in data
    assert data["status"] == "pending"


def test_create_task_reuses_requested_workspace_id():
    with patch("src.api.routes.tasks._run_graph"):
        resp1 = client.post(
            "/tasks",
            json={"input_value": "1706.03762", "report_mode": "draft", "workspace_id": "ws_shared_api"},
        )
        resp2 = client.post(
            "/tasks",
            json={
                "input_type": "research",
                "input_value": "RAG survey",
                "source_type": "research",
                "workspace_id": "ws_shared_api",
            },
        )

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.json()["workspace_id"] == "ws_shared_api"
    assert resp2.json()["workspace_id"] == "ws_shared_api"


def test_get_task():
    with patch("src.api.routes.tasks._run_graph"):
        resp = client.post("/tasks", json={"input_value": "1706.03762", "report_mode": "draft"})
    task_id = resp.json()["task_id"]

    resp = client.get(f"/tasks/{task_id}")
    assert resp.status_code == 200
    assert resp.json()["task_id"] == task_id
    assert resp.json()["report_mode"] == "draft"


def test_get_task_not_found():
    resp = client.get("/tasks/nonexistent")
    assert resp.status_code == 404


def test_list_tasks_after_create():
    with patch("src.api.routes.tasks._run_graph"):
        client.post("/tasks", json={"input_value": "1706.03762"})
        client.post("/tasks", json={"input_value": "1810.04805"})

    resp = client.get("/tasks")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_existing_report_endpoint_still_works():
    """Verify backward compatibility of the old /report endpoint."""
    with patch("src.api.app._build_react_agent"), \
         patch("src.api.app.generate_literature_report", return_value="## Title\n\nTest report"):
        resp = client.post("/report", json={"arxiv_url_or_id": "1706.03762"})
    assert resp.status_code == 200
    assert "report" in resp.json()


def test_task_chat():
    with patch("src.api.routes.tasks._run_graph"):
        resp = client.post("/tasks", json={"input_value": "1706.03762", "report_mode": "full"})
    task_id = resp.json()["task_id"]
    store = get_tasks_store()
    task = store[task_id]
    task.status = TaskStatus.COMPLETED
    task.result_markdown = "## 标题\n\nTest report"
    task.report_context_snapshot = task.result_markdown
    task.paper_type = "regular"

    mock_resp = MagicMock()
    mock_resp.content = "Follow-up answer"
    with patch("src.agent.settings.Settings.from_env", return_value=MagicMock()), \
         patch("src.agent.llm.build_chat_llm") as mock_build:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_resp
        mock_build.return_value = mock_llm
        resp = client.post(f"/tasks/{task_id}/chat", json={"message": "Explain more"})

    assert resp.status_code == 200
    assert resp.json()["content"] == "Follow-up answer"
