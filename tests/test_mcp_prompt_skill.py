from __future__ import annotations

from src.models.agent import AgentRole
from src.models.skills import SkillRunRequest
from src.skills.registry import get_skills_registry


def test_academic_review_writer_prompt_skill_returns_prompt():
    registry = get_skills_registry()
    response = registry.run_sync(
        SkillRunRequest(
            workspace_id="ws_test",
            task_id="task_test",
            skill_id="academic_review_writer_prompt",
            inputs={
                "topic": "medical imaging agents",
                "time_range": "2023 to 2026",
                "focus_dimensions": ["tool use", "evaluation"],
            },
            preferred_agent=AgentRole.ANALYST,
        ),
        {
            "workspace_id": "ws_test",
            "task_id": "task_test",
            "_mcp_server_id": "academic_writing",
        },
    )

    assert "Write an English academic survey" in response.result["prompt"]
    assert response.backend.value == "mcp_prompt"
