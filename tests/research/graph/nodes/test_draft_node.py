from __future__ import annotations

from src.models.report import DraftReport
from src.research.graph.nodes.draft import (
    _build_markdown,
    _compose_survey_system_prompt,
    _ensure_minimum_citation_coverage,
    _fallback_draft,
)


def test_compose_survey_system_prompt_returns_string_with_gap_rule():
    prompt = _compose_survey_system_prompt(compressed_context=False)

    assert isinstance(prompt, str)
    assert "SURVEY WRITING RULES:" in prompt
    assert "Future work MUST be derived from cross-paper gaps" in prompt
    assert "Maintain citation diversity" in prompt
    assert "title in English" in prompt
    assert "Chinese characters" not in prompt


def test_build_markdown_uses_english_sections_without_duplicate_chinese_summary_block():
    draft = DraftReport(
        sections={
            "title": "Medical Imaging Agent Survey",
            "abstract": "Abstract body",
            "introduction": "Introduction body",
            "methods": "Methods body",
            "evaluation": "Evaluation body",
            "discussion": "Discussion body",
            "conclusion": "Conclusion body",
        },
        claims=[],
        citations=[],
    )

    markdown = _build_markdown(draft, {"topic": "medical imaging agents"})

    assert markdown.startswith("# Medical Imaging Agent Survey")
    assert "## Abstract" in markdown
    assert "## Methods" in markdown
    assert "## Evaluation" in markdown
    assert "## 标题" not in markdown
    assert "## 核心贡献" not in markdown
    assert "## 方法概述" not in markdown
    assert "## 关键实验" not in markdown
    assert "## 局限性" not in markdown


def test_fallback_draft_future_work_is_gap_based_not_limitation_rewrite():
    draft = _fallback_draft(
        [
            {
                "title": "Clinical Agent A",
                "arxiv_id": "2401.00001",
                "abstract": "Clinical diagnosis agent with limited generalization across hospitals.",
                "summary": "Clinical diagnosis agent with limited generalization across hospitals.",
                "methods": ["multi-agent planning"],
                "datasets": ["HospitalQA"],
                "limitations": ["Limited generalization to a single hospital dataset."],
                "fulltext_available": True,
                "fulltext_snippets": [{"section": "discussion", "text": "Generalization remains limited to one hospital dataset."}],
            },
            {
                "title": "Clinical Agent B",
                "arxiv_id": "2401.00002",
                "abstract": "System has high latency and expensive tool calls in deployment.",
                "summary": "System has high latency and expensive tool calls in deployment.",
                "methods": ["tool use"],
                "datasets": ["HospitalQA"],
                "limitations": ["High latency during deployment and tool execution."],
            },
        ],
        {"topic": "医疗 AI agent"},
    )

    future_work = draft.sections["future_work"]
    assert "Limited generalization to a single hospital dataset." not in future_work
    assert "High latency during deployment and tool execution." not in future_work
    assert "need further research" not in future_work.lower()
    assert "cross-dataset" in future_work or "cross-institution" in future_work


def test_ensure_minimum_citation_coverage_backfills_references_and_inline_mentions():
    draft = DraftReport(
        sections={
            "title": "Medical Imaging Agent Survey",
            "introduction": "Current systems remain unevenly grounded.",
            "methods": "Tool use and retrieval remain the main design axes.",
            "discussion": "Clinical validation remains sparse.",
        },
        claims=[],
        citations=[
            {
                "label": "[1]",
                "url": "https://arxiv.org/abs/2401.00001",
                "reason": "Paper A",
            }
        ],
    )

    repaired = _ensure_minimum_citation_coverage(
        draft,
        [
            {"title": "Paper A", "arxiv_id": "2401.00001", "abstract": "Agentic medical imaging pipeline."},
            {"title": "Paper B", "arxiv_id": "2401.00002", "abstract": "Radiology retrieval workflow."},
            {"title": "Paper C", "arxiv_id": "2401.00003", "abstract": "Multimodal diagnosis benchmark."},
        ],
        brief={"desired_output": "survey_outline"},
    )

    assert len(repaired.citations) == 3
    assert "[2]" in repaired.sections["introduction"] or "[2]" in repaired.sections["methods"] or "[2]" in repaired.sections["discussion"]
    assert any(citation.reason == "Paper B" for citation in repaired.citations)


def test_ensure_minimum_citation_coverage_uses_section_evidence_map_for_distribution():
    draft = DraftReport(
        sections={
            "title": "Medical Imaging Agent Survey",
            "introduction": "The field is still emerging [1].",
            "methods": "Current systems combine retrieval and orchestration [1].",
            "datasets": "Benchmark detail remains sparse [1].",
            "discussion": "Deployment evidence is limited [1].",
        },
        claims=[],
        citations=[
            {
                "label": "[1]",
                "url": "https://arxiv.org/abs/2401.00001",
                "reason": "Paper A",
            }
        ],
    )

    repaired = _ensure_minimum_citation_coverage(
        draft,
        [
            {"title": "Paper A", "arxiv_id": "2401.00001", "abstract": "Agentic medical imaging pipeline."},
            {"title": "Paper B", "arxiv_id": "2401.00002", "abstract": "Radiology retrieval workflow with tool use."},
            {"title": "Paper C", "arxiv_id": "2401.00003", "abstract": "Benchmark paper with detailed dataset coverage."},
            {"title": "Paper D", "arxiv_id": "2401.00004", "abstract": "Discussion of deployment gaps and evidence limitations."},
        ],
        brief={"desired_output": "survey_outline"},
        skill_artifacts={
            "section_evidence_map": {
                "methods": ["Paper B"],
                "datasets": ["Paper C"],
                "discussion": ["Paper D"],
            }
        },
    )

    assert len(repaired.citations) >= 4
    assert "[2]" in repaired.sections["methods"]
    assert "[3]" in repaired.sections["datasets"]
    assert "[4]" in repaired.sections["discussion"]
