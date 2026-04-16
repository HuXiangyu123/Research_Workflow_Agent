from __future__ import annotations

from src.research.graph.nodes.search import (
    _extract_anchor_groups,
    _rerank_and_filter_candidates,
    _score_candidate_relevance,
    _supplement_strict_core_survey_candidates,
    _token_occurs,
)


def test_score_candidate_relevance_penalizes_missing_anchor_groups():
    anchors = _extract_anchor_groups(
        {
            "topic": "AI agent 在医学影像诊断中的应用",
            "sub_questions": ["多模态 agent 如何做临床诊断？"],
        },
        {"plan_goal": "medical agent survey"},
    )

    good_score, good_diag = _score_candidate_relevance(
        {
            "title": "Multimodal Medical Agent for Clinical Diagnosis",
            "abstract": "We study medical agents for radiology diagnosis with multimodal reasoning.",
            "fulltext_available": True,
        },
        anchors,
    )
    bad_score, bad_diag = _score_candidate_relevance(
        {
            "title": "A Survey of Federated Learning for Edge Caching",
            "abstract": "This paper reviews caching and communication strategies.",
            "fulltext_available": False,
        },
        anchors,
    )

    assert "agent" in good_diag["matched_groups"]
    assert "medical" in good_diag["matched_groups"]
    assert any(p.startswith("missing:") for p in bad_diag["penalties"])
    assert good_score > bad_score


def test_rerank_and_filter_candidates_drops_off_topic_tail_candidates():
    brief = {
        "topic": "AI agent 在医学影像诊断中的应用",
        "sub_questions": ["多模态 agent 如何做临床诊断？"],
    }
    search_plan = {"plan_goal": "medical agent survey"}

    candidates = [
        {
            "title": f"Medical Agent Paper {idx}",
            "abstract": "medical agent clinical diagnosis multimodal workflow",
            "fulltext_available": idx % 2 == 0,
        }
        for idx in range(12)
    ]
    candidates.append(
        {
            "title": "Federated Learning for Wireless Edge Caching",
            "abstract": "communication and caching survey without medical or agent context",
            "fulltext_available": False,
        }
    )

    kept, rerank_log = _rerank_and_filter_candidates(candidates, brief=brief, search_plan=search_plan)

    assert len(kept) == 12
    assert all("Federated Learning for Wireless Edge Caching" != item["title"] for item in kept)
    assert any(entry.get("decision") == "dropped" for entry in rerank_log)


def test_rerank_and_filter_candidates_applies_time_range_and_strict_core_scope():
    brief = {
        "topic": "AI agents for medical imaging diagnosis and triage",
        "sub_questions": ["How do multimodal agents support diagnosis and triage?"],
        "time_range": "2023 to 2026",
    }
    search_plan = {"plan_goal": "medical imaging diagnosis and triage agents from 2023 to 2026"}

    candidates = [
        {
            "title": "Medical Imaging Agent for Clinical Triage",
            "abstract": "medical imaging agent clinical diagnosis and triage workflow",
            "published_year": 2024,
            "fulltext_available": True,
        },
        {
            "title": "Security, Privacy, and Agentic AI in a Regulatory View",
            "abstract": "regulatory agentic AI security and privacy reflections",
            "published_year": 2026,
            "fulltext_available": False,
        },
        {
            "title": "Language Meets Vision Transformer in Medical Image Segmentation",
            "abstract": "medical image segmentation with language guidance",
            "published_year": 2022,
            "fulltext_available": False,
        },
        {
            "title": "Radiology Agent with Retrieval and Report Grounding",
            "abstract": "radiology medical imaging agent diagnosis triage retrieval workflow",
            "published_year": 2025,
            "fulltext_available": True,
        },
    ]

    kept, rerank_log = _rerank_and_filter_candidates(candidates, brief=brief, search_plan=search_plan)

    kept_titles = {item["title"] for item in kept}
    assert "Medical Imaging Agent for Clinical Triage" in kept_titles
    assert "Radiology Agent with Retrieval and Report Grounding" in kept_titles
    assert "Security, Privacy, and Agentic AI in a Regulatory View" not in kept_titles
    assert "Language Meets Vision Transformer in Medical Image Segmentation" not in kept_titles
    assert any(entry.get("strict_core") is True for entry in rerank_log if entry.get("strategy") == "domain_aware_anchor_rerank")


def test_rerank_and_filter_candidates_keeps_contextual_overviews_but_drops_component_only_papers_under_strict_scope():
    brief = {
        "topic": "AI agents for medical imaging diagnosis and triage",
        "sub_questions": ["How do multimodal agents support diagnosis and triage?"],
        "time_range": "2023 to 2026",
    }
    search_plan = {"plan_goal": "medical imaging diagnosis and triage agents from 2023 to 2026"}

    candidates = [
        {
            "title": "RadioRAG: Online Retrieval-augmented Generation for Radiology Question Answering",
            "abstract": "radiology retrieval-augmented generation workflow for diagnosis support",
            "published_year": 2024,
            "fulltext_available": False,
        },
        {
            "title": "Medical Knowledge-Guided Deep Curriculum Learning for Elbow Fracture Diagnosis from X-Ray Images",
            "abstract": "medical imaging diagnosis classifier for x-ray fracture detection",
            "published_year": 2023,
            "fulltext_available": False,
        },
        {
            "title": "LLM-Assisted Emergency Triage Benchmark: Bridging Hospital-Rich and MCI-Like Field Simulation",
            "abstract": "llm-assisted triage benchmark for clinical workflow evaluation",
            "published_year": 2025,
            "fulltext_available": False,
        },
        {
            "title": "Introduction of Medical Imaging Modalities",
            "abstract": "overview of medical imaging modalities for diagnostic workflows, triage, and report generation",
            "published_year": 2024,
            "fulltext_available": False,
        },
    ]

    kept, _ = _rerank_and_filter_candidates(candidates, brief=brief, search_plan=search_plan)

    kept_titles = {item["title"] for item in kept}
    assert "RadioRAG: Online Retrieval-augmented Generation for Radiology Question Answering" in kept_titles
    assert "LLM-Assisted Emergency Triage Benchmark: Bridging Hospital-Rich and MCI-Like Field Simulation" in kept_titles
    assert "Introduction of Medical Imaging Modalities" in kept_titles
    assert "Medical Knowledge-Guided Deep Curriculum Learning for Elbow Fracture Diagnosis from X-Ray Images" in kept_titles


def test_token_occurs_uses_word_boundaries_for_short_ascii_tokens():
    assert _token_occurs("brain mri diagnosis", "mri") is True
    assert _token_occurs("primary disorder diagnosis", "mri") is False
    assert _token_occurs("ct triage workflow", "ct") is True
    assert _token_occurs("predictive workflow", "ct") is False


def test_rerank_and_filter_candidates_strict_scope_drops_out_of_range_even_if_ranked_high():
    brief = {
        "topic": "AI agents for medical imaging diagnosis and triage",
        "sub_questions": ["How do multimodal agents support diagnosis and triage?"],
        "time_range": "2023 to 2026",
    }
    search_plan = {"plan_goal": "medical imaging diagnosis and triage agents from 2023 to 2026"}

    candidates = [
        {
            "title": "Psychotherapy Multi-Agent Workflow for Diagnosis",
            "abstract": "multi-agent workflow for mental health diagnosis",
            "published_year": 2025,
            "fulltext_available": True,
        },
        {
            "title": "Multi Agent Communication System for Online Auction with Decision Support System",
            "abstract": "multi-agent decision support system for online auctions",
            "published_year": 2011,
            "fulltext_available": True,
        },
        {
            "title": "RadioRAG for Radiology Question Answering",
            "abstract": "radiology retrieval workflow for diagnosis support",
            "published_year": 2024,
            "fulltext_available": True,
        },
    ]

    kept, _ = _rerank_and_filter_candidates(candidates, brief=brief, search_plan=search_plan)

    kept_titles = {item["title"] for item in kept}
    assert "Multi Agent Communication System for Online Auction with Decision Support System" not in kept_titles


def test_supplement_strict_core_survey_candidates_recovers_adjacent_in_scope_papers():
    kept = [
        {
            "title": "CT-Agent",
            "combined_score": 8.4,
            "relevance_diagnostics": {
                "matched_groups": ["agent", "medical", "multimodal_or_imaging", "diagnosis_or_triage"],
                "penalties": [],
            },
        },
        {
            "title": "RadioRAG",
            "combined_score": 2.85,
            "relevance_diagnostics": {
                "matched_groups": ["medical", "multimodal_or_imaging", "diagnosis_or_triage"],
                "penalties": ["missing:agent", "missing_agent_for_strict_scope"],
            },
        },
    ]
    dropped = [
        {
            "title": "Clinical Report Generation with Frozen LLMs",
            "combined_score": 1.9,
            "relevance_diagnostics": {
                "matched_groups": ["medical", "multimodal_or_imaging", "diagnosis_or_triage"],
                "penalties": ["missing:agent", "missing_agent_for_strict_scope"],
            },
        },
        {
            "title": "Introduction of Medical Imaging Modalities",
            "combined_score": 3.8,
            "relevance_diagnostics": {
                "matched_groups": ["medical", "multimodal_or_imaging", "diagnosis_or_triage"],
                "penalties": [
                    "missing:agent",
                    "missing_agent_for_strict_scope",
                    "component_or_overview_without_agentic_scope",
                    "overview_paper_without_agentic_scope",
                ],
            },
        },
        {
            "title": "Medical Image Segmentation Baseline",
            "combined_score": 1.7,
            "relevance_diagnostics": {
                "matched_groups": ["medical", "multimodal_or_imaging", "diagnosis_or_triage"],
                "penalties": ["component_or_overview_without_agentic_scope"],
            },
        },
        {
            "title": "Elbow Fracture Diagnosis from X-Ray Images",
            "combined_score": -0.3,
            "relevance_diagnostics": {
                "matched_groups": ["medical", "multimodal_or_imaging", "diagnosis_or_triage"],
                "penalties": [
                    "missing:agent",
                    "missing_agent_for_strict_scope",
                    "missing_agentic_workflow_signal",
                    "component_or_overview_without_agentic_scope",
                ],
            },
        },
        {
            "title": "Regulatory Agentic AI Survey",
            "combined_score": 1.6,
            "relevance_diagnostics": {
                "matched_groups": ["agent", "medical", "diagnosis_or_triage"],
                "penalties": ["governance_without_clinical_scope"],
            },
        },
    ]

    new_kept, new_dropped = _supplement_strict_core_survey_candidates(
        kept,
        dropped,
        rescored=kept + dropped,
    )

    kept_titles = {item["title"] for item in new_kept}
    dropped_titles = {item["title"] for item in new_dropped}
    assert "Clinical Report Generation with Frozen LLMs" in kept_titles
    assert "Introduction of Medical Imaging Modalities" in kept_titles
    assert "Medical Image Segmentation Baseline" in dropped_titles
    assert "Regulatory Agentic AI Survey" in dropped_titles
