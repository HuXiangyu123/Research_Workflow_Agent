from __future__ import annotations

from src.models.research import SearchPlan
from src.models.report import DraftReport
from src.models.review import ReviewFeedback
from src.research.agents.analyst_agent import AnalystAgent
from src.research.agents.clarify_agent import build_clarify_agent_graph, run as run_clarify_agent
from src.research.agents.planner_agent import PlannerAgent
from src.research.agents.retriever_agent import RetrieverAgent
from src.research.agents.reviewer_agent import ReviewerAgent, SelfReflection
from src.research.research_brief import ClarifyInput, ResearchBrief


def _make_brief(**overrides) -> ResearchBrief:
    data = {
        "topic": "RAG",
        "goal": "survey_drafting",
        "desired_output": "research_brief",
        "sub_questions": ["What retrieval strategies work best?"],
        "needs_followup": False,
        "confidence": 0.9,
    }
    data.update(overrides)
    return ResearchBrief.model_validate(data)


def test_clarify_agent_builds_langgraph_strategy_graph(monkeypatch):
    graph = build_clarify_agent_graph().get_graph()
    assert {
        "prepare",
        "fast_path",
        "structured_output",
        "json_parse",
        "repair",
        "limited",
        "post_validate",
    }.issubset(graph.nodes)

    monkeypatch.setattr(
        "src.research.agents.clarify_agent._fast_path_brief",
        lambda input_obj: _make_brief(topic=input_obj.raw_query.upper()),
    )

    result = run_clarify_agent(ClarifyInput(raw_query="rag"))

    assert result.brief.topic == "RAG"
    assert result.raw_model_output is None


def test_planner_agent_run_executes_langgraph_nodes_in_order(monkeypatch):
    agent = PlannerAgent()
    graph = agent.build_graph().get_graph()
    assert {"plan", "execute", "validate"}.issubset(graph.nodes)

    calls: list[str] = []
    plan = SearchPlan.model_validate(
        {
            "plan_goal": "Collect recent RAG papers",
            "query_groups": [
                {
                    "group_id": "g1",
                    "queries": ["retrieval augmented generation"],
                    "intent": "exploration",
                    "priority": 1,
                    "expected_hits": 5,
                }
            ],
        }
    )

    def fake_plan_phase(brief: dict) -> dict:
        calls.append("plan")
        return {"plan": plan, "phases": ["phase-1"]}

    def fake_execute_phase(phases: list[str]) -> dict:
        calls.append("execute")
        assert phases == ["phase-1"]
        return {"candidates": [{"title": "Paper 1"}], "execution_log": [{"phase": "phase-1"}]}

    def fake_validate_phase(brief: dict, plan_obj: SearchPlan, execution_results: dict) -> dict:
        calls.append("validate")
        assert plan_obj.plan_goal == "Collect recent RAG papers"
        return {
            "validation": {"status": "complete"},
            "candidates": execution_results["candidates"],
            "execution_log": execution_results["execution_log"],
        }

    monkeypatch.setattr(agent, "plan_phase", fake_plan_phase)
    monkeypatch.setattr(agent, "execute_phase", fake_execute_phase)
    monkeypatch.setattr(agent, "validate_phase", fake_validate_phase)

    result = agent.run({"topic": "RAG"})

    assert calls == ["plan", "execute", "validate"]
    assert result["search_plan"]["plan_goal"] == "Collect recent RAG papers"
    assert result["validation"]["status"] == "complete"


def test_planner_agent_coerces_out_of_range_priorities_before_validation():
    agent = PlannerAgent()

    normalized = agent._coerce_plan_payload(
        {
            "plan_goal": "Collect recent RAG papers",
            "query_groups": [
                {
                    "group_id": "g1",
                    "queries": ["rag retrieval"],
                    "intent": "exploration",
                    "priority": 4,
                    "expected_hits": 8,
                },
                {
                    "group_id": "g2",
                    "queries": ["dense retrieval"],
                    "intent": "refinement",
                    "priority": 0,
                    "expected_hits": 6,
                },
            ],
        }
    )

    plan = SearchPlan.model_validate(normalized)

    assert [group.priority for group in plan.query_groups] == [3, 1]


def test_retriever_agent_run_builds_rag_result_with_langgraph(monkeypatch):
    agent = RetrieverAgent(workspace_id="ws-1")
    graph = agent.build_graph().get_graph()
    assert {
        "augmented_query_gen",
        "parallel_retrieval",
        "context_assembly",
        "finalize_rag_result",
    }.issubset(graph.nodes)

    monkeypatch.setattr(
        agent,
        "_augmented_query_gen",
        lambda brief, search_plan=None: {"queries": [{"query": "rag", "sources": ["arxiv"], "expected_hits": 5}]},
    )
    monkeypatch.setattr(
        agent,
        "_parallel_retrieval",
        lambda queries: {
            "raw_results": [
                {
                    "query": "rag",
                    "hits": [{"title": "RAG Paper", "url": "https://example.com/p1", "content": "abstract"}],
                }
            ]
        },
    )
    monkeypatch.setattr(
        agent,
        "_context_assembly",
        lambda brief, raw_results: [
            {
                "rank": 1,
                "title": "RAG Paper",
                "url": "https://example.com/p1",
                "abstract": "abstract",
                "source": "arxiv",
            }
        ],
    )
    monkeypatch.setattr(
        "src.research.graph.nodes.search._ingest_paper_candidates",
        lambda candidates, workspace_id=None: None,
    )

    result = agent.run({"topic": "RAG"}, {"plan_goal": "Collect RAG"})

    assert result["queries_generated"] == 1
    assert result["rag_result"]["query"] == "Collect RAG"
    assert result["rag_result"]["paper_candidates"][0]["title"] == "RAG Paper"


def test_retriever_agent_fallback_candidates_enriches_arxiv_metadata(monkeypatch):
    agent = RetrieverAgent(workspace_id="ws-1")

    monkeypatch.setattr(
        "src.tools.arxiv_api.enrich_search_results_with_arxiv",
        lambda candidates: [{**candidates[0], "arxiv_id": "2407.15621", "pdf_url": "https://arxiv.org/pdf/2407.15621.pdf"}],
    )

    candidates = agent._fallback_candidates(
        [
            {
                "query": "radiology rag",
                "hits": [
                    {
                        "title": "RadioRAG",
                        "url": "https://arxiv.org/abs/2407.15621v3",
                        "content": "radiology retrieval paper",
                        "engine": "arxiv",
                        "publishedDate": "2024-07-22",
                    }
                ],
            }
        ]
    )

    assert candidates[0]["arxiv_id"] == "2407.15621"
    assert candidates[0]["pdf_url"] == "https://arxiv.org/pdf/2407.15621.pdf"


def test_retriever_agent_queries_from_search_plan_round_robins_groups():
    agent = RetrieverAgent(workspace_id="ws-1")

    queries = agent._queries_from_search_plan(
        {
            "source_preferences": ["arxiv", "semantic_scholar"],
            "query_groups": [
                {
                    "priority": 1,
                    "intent": "exploration",
                    "expected_hits": 10,
                    "queries": [f"explore {idx}" for idx in range(6)],
                },
                {
                    "priority": 2,
                    "intent": "refinement",
                    "expected_hits": 8,
                    "queries": [f"refine {idx}" for idx in range(5)],
                },
                {
                    "priority": 3,
                    "intent": "validation",
                    "expected_hits": 6,
                    "queries": [f"validate {idx}" for idx in range(4)],
                },
            ],
        }
    )

    assert len(queries) == 12
    assert [item["query"] for item in queries[:6]] == [
        "explore 0",
        "refine 0",
        "validate 0",
        "explore 1",
        "refine 1",
        "validate 1",
    ]


def test_retriever_agent_fallback_candidates_round_robins_query_hits(monkeypatch):
    agent = RetrieverAgent(workspace_id="ws-1")

    monkeypatch.setattr(
        "src.tools.arxiv_api.enrich_search_results_with_arxiv",
        lambda candidates: candidates,
    )

    candidates = agent._fallback_candidates(
        [
            {
                "query": "q1",
                "query_order": 0,
                "hits": [
                    {"title": "A1", "url": "https://example.com/a1", "content": "a1", "engine": "arxiv"},
                    {"title": "A2", "url": "https://example.com/a2", "content": "a2", "engine": "arxiv"},
                ],
            },
            {
                "query": "q2",
                "query_order": 1,
                "hits": [
                    {"title": "B1", "url": "https://example.com/b1", "content": "b1", "engine": "arxiv"},
                    {"title": "B2", "url": "https://example.com/b2", "content": "b2", "engine": "arxiv"},
                ],
            },
        ]
    )

    assert [item["title"] for item in candidates[:4]] == ["A1", "B1", "A2", "B2"]


def test_retriever_agent_build_rag_result_recovers_from_direct_sources_when_tag_hits_are_empty(monkeypatch):
    agent = RetrieverAgent(workspace_id="ws-1")

    monkeypatch.setattr(
        "src.tools.arxiv_api.search_arxiv_direct",
        lambda query, max_results=10, year_filter=None, filter_noise=True: [
            {
                "title": "Recovered Medical Imaging Agent",
                "url": "https://arxiv.org/abs/2401.00001",
                "abstract": "medical imaging agent workflow for diagnosis and triage",
                "arxiv_id": "2401.00001",
                "published_year": 2024,
            }
        ],
    )
    monkeypatch.setattr(
        "src.tools.deepxiv_client.search_papers",
        lambda query, size=10, date_from=None, categories=None: [],
    )
    monkeypatch.setattr(
        "src.tools.arxiv_api.enrich_search_results_with_arxiv",
        lambda candidates: candidates,
    )
    monkeypatch.setattr(
        "src.research.graph.nodes.search._rerank_and_filter_candidates",
        lambda candidates, brief, search_plan: (candidates, [{"strategy": "test"}]),
    )
    monkeypatch.setattr(
        "src.research.graph.nodes.search._ingest_paper_candidates",
        lambda candidates, workspace_id=None: None,
    )

    rag_result = agent._build_rag_result(
        brief={"topic": "medical imaging diagnosis and triage agents", "time_range": "2023 to 2026"},
        search_plan={
            "plan_goal": "Collect medical imaging diagnosis and triage agents",
            "query_groups": [
                {
                    "intent": "exploration",
                    "expected_hits": 10,
                    "queries": ["medical imaging diagnosis and triage agents"],
                }
            ],
        },
        raw_results=[
            {
                "query": "medical imaging diagnosis and triage agents",
                "query_meta": {"intent": "exploration", "expected_hits": 10},
                "hits": [],
            }
        ],
        candidates=[],
    )

    assert rag_result["paper_candidates"][0]["title"] == "Recovered Medical Imaging Agent"
    assert any("arXiv direct / DeepXiv" in note for note in rag_result["coverage_notes"])


def test_analyst_agent_run_emits_draft_report_and_markdown(monkeypatch):
    agent = AnalystAgent()
    graph = agent.build_graph().get_graph()
    assert {
        "seed_reasoning_state",
        "build_structured_cards",
        "build_comparison_matrix",
        "build_outline",
        "build_report_draft",
        "verify_and_finalize",
    }.issubset(graph.nodes)

    monkeypatch.setattr(
        agent,
        "_build_structured_cards",
        lambda paper_cards: {
            "cards": [{"title": "Paper 1", "methods": ["Dense retrieval"], "datasets": ["MS MARCO"]}],
            "confidence": 0.7,
        },
    )
    monkeypatch.setattr(
        agent,
        "_build_comparison_matrix",
        lambda artifacts: {"matrix": {"rows": [{"paper": "Paper 1", "methods": "Dense retrieval"}]}, "confidence": 0.8},
    )
    monkeypatch.setattr(
        agent,
        "_build_outline",
        lambda state, brief=None: {"outline": {"introduction": ["Background"], "methods": ["Dense retrieval"]}, "confidence": 0.75},
    )
    monkeypatch.setattr(
        agent,
        "_build_report_draft",
        lambda state: {
            "draft": {
                "sections": {
                    "title": "RAG Survey",
                    "abstract": "Summary",
                    "introduction": "Intro",
                    "methods": "Methods",
                    "conclusion": "Done",
                },
                "claims": [],
                "citations": [],
            },
            "confidence": 0.85,
        },
    )
    monkeypatch.setattr(agent, "_needs_grounded_redraft", lambda draft_report: False)
    monkeypatch.setattr(agent, "_store_artifacts_memory", lambda state: None)
    monkeypatch.setattr(
        "src.research.graph.nodes.draft._build_draft_report",
        lambda cards, brief: DraftReport(
            sections={
                "title": "RAG Survey",
                "abstract": "Summary",
                "introduction": "Intro",
                "background": "Background",
                "taxonomy": "Taxonomy",
                "methods": "Methods",
                "datasets": "Datasets",
                "evaluation": "Evaluation",
                "discussion": "Discussion",
                "future_work": "Future",
                "conclusion": "Done",
            },
            claims=[],
            citations=[],
        ),
    )

    result = agent.run({"topic": "RAG"}, [{"title": "Paper 1"}])

    assert result["draft_report"].sections["title"] == "RAG Survey"
    assert result["draft_markdown"].startswith("# RAG Survey")
    assert result["overall_confidence"] > 0


def test_reviewer_agent_langgraph_loops_until_review_passes(monkeypatch):
    agent = ReviewerAgent()
    graph = agent.build_graph().get_graph()
    assert {
        "retrieve_memory",
        "actor_review",
        "evaluate",
        "self_reflect",
        "finalize",
    }.issubset(graph.nodes)

    attempts: list[int] = []

    monkeypatch.setattr(agent, "_retrieve_reflections", lambda brief, draft_report: [])

    def fake_actor_review(**kwargs):
        attempt = kwargs["attempt"]
        attempts.append(attempt)
        return {"confidence": 0.3 if attempt == 1 else 0.9}

    def fake_evaluate(actor_result: dict, attempt: int) -> dict:
        return {
            "passed": attempt == 2,
            "confidence": actor_result["confidence"],
            "reason": "retry" if attempt == 1 else "ok",
            "task_type": "review_confidence",
            "issues": ["low confidence"] if attempt == 1 else [],
        }

    monkeypatch.setattr(agent, "_actor_review", fake_actor_review)
    monkeypatch.setattr(agent, "_evaluate", fake_evaluate)
    monkeypatch.setattr(
        agent,
        "_self_reflect",
        lambda **kwargs: SelfReflection(
            reflection_id="refl_1",
            task_type="review_confidence",
            failure_context="low confidence",
            root_cause="insufficient coverage",
            lessons=["improve retrieval"],
            improved_strategy="retrieve more evidence",
            confidence_gain=0.1,
            created_at=0.0,
        ),
    )
    monkeypatch.setattr(agent, "_store_reflection", lambda reflection: None)

    result = agent.run({"topic": "RAG"}, [{"title": "Paper 1"}], {"sections": {"introduction": "Intro"}})

    assert attempts == [1, 2]
    assert result["review_passed"] is True
    assert result["total_attempts"] == 2


def test_reviewer_agent_awaits_async_review_service_and_stores_serializable_feedback(monkeypatch):
    agent = ReviewerAgent(workspace_id="ws-1", task_id="task-1")

    monkeypatch.setattr(agent, "_retrieve_reflections", lambda brief, draft_report: [])

    async def fake_review(self, **kwargs):
        return ReviewFeedback(
            task_id=kwargs["task_id"],
            workspace_id=kwargs["workspace_id"],
            passed=True,
            summary="async reviewer ok",
        )

    monkeypatch.setattr("src.research.services.reviewer.ReviewerService.review", fake_review)

    result = agent.run(
        {"topic": "RAG"},
        [{"title": "Paper 1"}],
        {"sections": {"introduction": "Intro"}},
    )

    assert result["review_passed"] is True
    assert result["review_feedback"]["passed"] is True
    assert result["total_attempts"] == 1
