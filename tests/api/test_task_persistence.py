from __future__ import annotations

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.routes.tasks import clear_tasks_store, get_tasks_store
from src.db.engine import reset_engine
from src.db.task_persistence import (
    delete_task_persistence,
    load_task_report,
    load_task_snapshot,
    reset_task_persistence_state,
    upsert_task_snapshot,
)
from src.models.task import TaskRecord, TaskStatus


def _setup_mocks():
    mock_entry = MagicMock()
    mock_entry.title = "Attention Is All You Need"
    mock_entry.summary = (
        "The dominant sequence transduction models are based on complex "
        "recurrent or convolutional neural networks."
    )
    mock_entry.published = "2017-06-12T17:57:34Z"
    author = MagicMock()
    author.name = "Ashish Vaswani"
    mock_entry.authors = [author]
    link = MagicMock()
    link.type = "application/pdf"
    link.href = "http://arxiv.org/pdf/1706.03762v7"
    link.title = "pdf"
    mock_entry.links = [link]
    mock_feed = MagicMock()
    mock_feed.entries = [mock_entry]

    draft_json = json.dumps(
        {
            "sections": {
                "标题": "Attention Is All You Need",
                "核心贡献": "Proposes the Transformer architecture based entirely on attention mechanisms.",
                "方法概述": "Self-attention replaces recurrence and convolutions.",
                "关键实验": "Achieves 28.4 BLEU on WMT 2014 EN-DE.",
                "局限性": "High computational cost for very long sequences.",
            },
            "claims": [
                {
                    "id": "c1",
                    "text": "Transformer achieves state-of-the-art BLEU score",
                    "citation_labels": ["[1]"],
                }
            ],
            "citations": [
                {
                    "label": "[1]",
                    "url": "https://arxiv.org/abs/1706.03762",
                    "reason": "Original Transformer paper",
                }
            ],
        },
        ensure_ascii=False,
    )
    mock_llm_resp = MagicMock()
    mock_llm_resp.content = draft_json
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_llm_resp

    mock_http_resp = MagicMock()
    mock_http_resp.content = b"%PDF-1.4 fake"
    mock_http_resp.raise_for_status = MagicMock()
    mock_http_resp.status_code = 200
    mock_http_client = MagicMock()
    mock_http_client.get.return_value = mock_http_resp
    mock_http_client.head.return_value = mock_http_resp
    mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
    mock_http_client.__exit__ = MagicMock(return_value=False)

    return {
        "feedparser": patch(
            "src.graph.nodes.ingest_source.feedparser.parse",
            return_value=mock_feed,
        ),
        "httpx_client": patch(
            "src.graph.nodes.extract_document_text.httpx.Client",
            return_value=mock_http_client,
        ),
        "pdf_extract": patch(
            "src.graph.nodes.extract_document_text.extract_text_from_pdf_bytes",
            return_value="Full paper text about Transformer architecture and self-attention mechanisms.",
        ),
        "settings": patch("src.agent.settings.Settings"),
        "llm_builder": patch(
            "src.agent.llm.build_reason_llm",
            return_value=mock_llm,
        ),
        "reachability": patch(
            "src.graph.nodes.resolve_citations.check_url_reachable_sync",
            return_value=True,
        ),
        "fetch_snippet": patch(
            "src.graph.nodes.resolve_citations._fetch_content_snippet",
            return_value="Transformer paper content",
        ),
    }


@pytest.fixture
def postgres_persistence():
    load_dotenv(".env")
    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL not configured")
    reset_engine()
    reset_task_persistence_state()
    clear_tasks_store()
    created_task_ids: list[str] = []
    yield created_task_ids
    for task_id in created_task_ids:
        delete_task_persistence(task_id)
    get_tasks_store().clear()
    clear_tasks_store()
    reset_engine()
    reset_task_persistence_state()


@pytest.mark.task_persistence
def test_task_report_is_persisted_and_reloadable(postgres_persistence):
    mocks = _setup_mocks()
    [m.start() for m in mocks.values()]

    try:
        with TestClient(app) as client:
            create_resp = client.post(
                "/tasks",
                json={
                    "input_type": "arxiv",
                    "input_value": "1706.03762",
                    "report_mode": "draft",
                },
            )
            assert create_resp.status_code == 200
            task_id = create_resp.json()["task_id"]
            postgres_persistence.append(task_id)
            workspace_id = create_resp.json()["workspace_id"]
            assert workspace_id

            data = None
            for _ in range(60):
                resp = client.get(f"/tasks/{task_id}")
                assert resp.status_code == 200
                data = resp.json()
                if data["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.2)

            assert data is not None
            assert data["status"] == "completed"
            assert data["persisted_to_db"] is True
            assert data["persisted_report_id"]

            result_resp = client.get(f"/tasks/{task_id}/result")
            assert result_resp.status_code == 200
            result = result_resp.json()
            assert result["persisted"] is True
            assert result["report_kind"] == "final_report"
            assert "Attention Is All You Need" in (result["result_markdown"] or "")

            report_row = load_task_report(task_id)
            assert report_row is not None
            assert report_row["report_id"] == result["report_id"]

            get_tasks_store().clear()

            reload_resp = client.get(f"/tasks/{task_id}")
            assert reload_resp.status_code == 200
            reloaded = reload_resp.json()
            assert reloaded["task_id"] == task_id
            assert reloaded["persisted_to_db"] is True
            assert reloaded["workspace_id"] == workspace_id

            snapshot = load_task_snapshot(task_id)
            assert snapshot is not None
            assert snapshot.result_markdown
            assert snapshot.persisted_report_id == result["report_id"]
    finally:
        for m in mocks.values():
            m.stop()


@pytest.mark.task_persistence
def test_task_snapshot_round_trips_extended_research_fields(postgres_persistence):
    task = TaskRecord(
        status=TaskStatus.COMPLETED,
        input_type="research",
        input_value="Survey biomedical LLM agents",
        report_mode="draft",
        source_type="research",
        auto_fill=True,
        workspace_id="ws_persist_roundtrip",
        result_markdown="# Report",
        brief={"topic": "biomedical LLM agents"},
        search_plan={"plan_goal": "collect evidence"},
        rag_result={"paper_candidates": [{"title": "Paper A"}]},
        paper_cards=[{"title": "Paper A"}],
        compression_result={
            "taxonomy": {"categories": [{"name": "Agentic RAG"}]},
            "compressed_cards": [{"title": "Paper A"}],
            "evidence_pools": {"methods": ["e1"]},
        },
        taxonomy={"categories": [{"name": "Agentic RAG"}]},
        awaiting_followup=False,
        followup_resolution={"auto_fill_triggered": True, "reason": "non_interactive"},
        review_passed=True,
    )
    postgres_persistence.append(task.task_id)

    assert upsert_task_snapshot(task) is True

    loaded = load_task_snapshot(task.task_id)
    assert loaded is not None
    assert loaded.auto_fill is True
    assert loaded.compression_result is not None
    assert loaded.compression_result["evidence_pools"]["methods"] == ["e1"]
    assert loaded.taxonomy == {"categories": [{"name": "Agentic RAG"}]}
    assert loaded.awaiting_followup is False
    assert loaded.followup_resolution == {
        "auto_fill_triggered": True,
        "reason": "non_interactive",
    }
