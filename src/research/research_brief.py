"""Research brief and clarify domain models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AmbiguityItem(BaseModel):
    """An unresolved ambiguity in the user's research request."""

    field: str = Field(description="Which field of the request is ambiguous")
    reason: str = Field(description="Why this field is ambiguous or underspecified")
    suggested_options: list[str] = Field(
        default_factory=list,
        description="Suggested concrete options to resolve the ambiguity",
    )


class ResearchBrief(BaseModel):
    """Structured output of ClarifyAgent — feeds SearchPlanAgent and downstream nodes."""

    topic: str = Field(description="Core research topic or problem area")
    goal: str = Field(
        description="Practical purpose: survey_drafting, baseline_exploration, "
        "related_work_support, idea_exploration, paper_reading, etc."
    )
    desired_output: str = Field(
        description="Expected artifact type: survey_outline, paper_cards, "
        "related_work_draft, reading_notes, research_brief"
    )
    sub_questions: list[str] = Field(
        default_factory=list,
        description="Concrete research questions guiding downstream search and extraction",
    )
    time_range: str | None = Field(
        default=None,
        description="Explicit or inferred time scope, e.g. '近三年', '2018-2024'; null if unspecified",
    )
    domain_scope: str | None = Field(
        default=None,
        description="Domain boundaries, e.g. '多模态学习', '医学影像'; null if broad",
    )
    source_constraints: list[str] = Field(
        default_factory=list,
        description="Restrictions on sources, venues, datasets, paper types, or language",
    )
    focus_dimensions: list[str] = Field(
        default_factory=list,
        description="Angles the user cares about: methods, benchmarks, datasets, "
        "grounding, reproducibility, limitations, trends, etc.",
    )
    ambiguities: list[AmbiguityItem] = Field(
        default_factory=list,
        description="Unresolved uncertainties made explicit by ClarifyAgent",
    )
    needs_followup: bool = Field(
        default=False,
        description="Whether human clarification is required before confident downstream planning",
    )
    confidence: float = Field(
        default=0.0,
        description="ClarifyAgent's confidence in this brief, float in [0.0, 1.0]",
    )
    schema_version: str = Field(default="v1", description="Always 'v1' for this schema")


class ClarifyInput(BaseModel):
    """Input passed to ClarifyAgentService.run()."""

    raw_query: str = Field(description="The raw user research request")
    preferred_output: str | None = Field(
        default=None,
        description="Optional hint about desired output type (survey_outline, paper_cards, etc.)",
    )
    workspace_context: str | None = Field(
        default=None,
        description="Optional background from a prior research workspace/session",
    )
    uploaded_source_summaries: list[str] = Field(
        default_factory=list,
        description="Short text summaries of uploaded sources (PDF metadata, etc.)",
    )
    auto_fill: bool = Field(
        default=False,
        description="If True, LLM will auto-complete ambiguous fields in the brief "
        "instead of setting needs_followup=True. User confirmation is skipped.",
    )


class ClarifyResult(BaseModel):
    """Result returned by ClarifyAgentService.run()."""

    brief: ResearchBrief
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings, e.g. 'low confidence', 'significant ambiguity'",
    )
    raw_model_output: str | None = Field(
        default=None,
        description="Raw LLM text response (for debugging / thinking panel)",
    )
