"""Policy helpers for ClarifyAgent."""

from __future__ import annotations

import re

from src.research.research_brief import AmbiguityItem, ResearchBrief

_DESIRED_OUTPUT_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("survey_outline", ("survey_outline", "综述大纲", "综述框架", "大纲", "outline")),
    ("paper_cards", ("paper_cards", "论文卡片", "paper card", "paper cards", "文献卡片")),
    ("reading_notes", ("reading_notes", "阅读笔记", "精读笔记", "reading notes", "笔记")),
    (
        "related_work_draft",
        ("related_work_draft", "related work", "related-work", "相关工作草稿", "related work draft"),
    ),
    ("research_brief", ("research_brief", "研究简报", "研究摘要", "brief")),
]

_DOMAIN_KEYWORDS: tuple[str, ...] = (
    "医学影像",
    "医疗问答",
    "临床决策",
    "电子病历",
    "医学报告生成",
    "病理",
    "药物发现",
    "医疗",
    "医学",
)

_FOCUS_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("方法", "方法"),
    ("数据集", "数据集"),
    ("评测指标", "评测指标"),
    ("benchmark", "评测指标"),
    ("局限", "局限性"),
    ("limitations", "局限性"),
    ("可复现", "可复现性"),
    ("工具调用", "工具调用"),
    ("多模态", "多模态"),
)

_TOPIC_CLAUSE_SPLIT_RE = re.compile(
    r"[，,。；;]\s*(输出|希望输出|产出|形式|重点关注|关注|聚焦|重点看|并关注)\b.*$",
    re.IGNORECASE,
)
_LEADING_RESEARCH_PREFIX_RE = re.compile(
    r"^(请|帮我|麻烦|想|需要)?\s*(调研|研究|分析|梳理|看看|了解)\s*",
)
_LEADING_TIME_RANGE_RE = re.compile(
    r"^(近[一二两三四五六七八九十\d]+年|最近|近几年|20\d{2}\s*[-~—–至到]\s*20\d{2}\s*年?)\s*",
)


def _infer_desired_output(raw_query: str) -> str | None:
    lowered = raw_query.lower()
    for desired_output, patterns in _DESIRED_OUTPUT_PATTERNS:
        if any(pattern.lower() in lowered for pattern in patterns):
            return desired_output
    return None


def _infer_time_range(raw_query: str) -> str | None:
    patterns = (
        r"近[一二两三四五六七八九十\d]+年",
        r"最近",
        r"近几年",
        r"20\d{2}\s*[-~—–至到]\s*20\d{2}",
        r"20\d{2}\s*-\s*20\d{2}",
    )
    for pattern in patterns:
        match = re.search(pattern, raw_query)
        if match:
            return match.group(0)
    return None


def _infer_domain_scope(raw_query: str) -> str | None:
    for keyword in _DOMAIN_KEYWORDS:
        if keyword in raw_query:
            return keyword
    return None


def _infer_focus_dimensions(raw_query: str) -> list[str]:
    dims: list[str] = []
    lowered = raw_query.lower()
    for needle, dim in _FOCUS_KEYWORDS:
        if needle.lower() in lowered and dim not in dims:
            dims.append(dim)
    return dims


def _extract_topic(raw_query: str) -> str | None:
    query = raw_query.strip()
    if not query:
        return None

    has_structured_constraints = (
        _infer_desired_output(query) is not None
        or _infer_time_range(query) is not None
        or any(marker in query for marker in ("输出", "希望输出", "重点关注", "聚焦", "关注"))
    )
    if not has_structured_constraints:
        return query

    topic = _TOPIC_CLAUSE_SPLIT_RE.sub("", query.strip("。！？；;,.， ")).strip("。！？；;,.， ")
    topic = _LEADING_RESEARCH_PREFIX_RE.sub("", topic).strip()
    topic = _LEADING_TIME_RANGE_RE.sub("", topic).strip()
    topic = re.sub(r"\s+", " ", topic).strip("。！？；;,.， ")
    return topic or query


def _is_topic_specific(raw_query: str) -> bool:
    if len(raw_query.strip()) >= 20:
        return True
    return _infer_domain_scope(raw_query) is not None


def _infer_goal(desired_output: str) -> str:
    mapping = {
        "survey_outline": "为后续综述写作和方法梳理做前期调研",
        "paper_cards": "收集代表论文并形成结构化论文卡片",
        "reading_notes": "围绕指定主题形成精读笔记",
        "related_work_draft": "为 related work 写作收集并组织证据",
        "research_brief": "初步探索可能的研究方向",
    }
    return mapping.get(desired_output, "初步探索可能的研究方向")


def _infer_sub_questions(
    raw_query: str,
    desired_output: str,
    focus_dimensions: list[str],
) -> list[str]:
    dims_text = "、".join(focus_dimensions[:3]) if focus_dimensions else "代表性方法与证据"
    if desired_output == "paper_cards":
        return [
            "这个主题下近年的代表论文有哪些？",
            f"这些论文在 {dims_text} 方面各自有什么特点？",
        ]
    if desired_output == "reading_notes":
        return [
            "这个主题下值得精读的代表论文有哪些？",
            f"这些论文在 {dims_text} 方面有哪些可记录的关键信息？",
        ]
    if desired_output == "related_work_draft":
        return [
            "这个方向近年的方法脉络如何演进？",
            f"哪些论文最适合作为 related work 中关于 {dims_text} 的证据？",
        ]
    return [
        "这个方向近年的代表性工作有哪些？",
        f"这些工作在 {dims_text} 方面呈现出哪些共性与差异？",
    ]


def is_brief_valid(brief: ResearchBrief) -> bool:
    """Check required fields are non-empty and confidence is in [0, 1].

    Uses conservative defaults rather than raising, so callers can decide
    whether to warn or fall back to limited brief.
    """
    if not brief.topic or not brief.topic.strip():
        return False
    if not brief.goal or not brief.goal.strip():
        return False
    if not brief.desired_output or not brief.desired_output.strip():
        return False
    if not brief.sub_questions:
        return False
    if not (0.0 <= brief.confidence <= 1.0):
        return False
    return True


def should_request_followup(brief: ResearchBrief) -> bool:
    """Return True when human clarification is strongly recommended.

    True when:
    - needs_followup is already True, OR
    - confidence is very low (< 0.4), OR
    - there are unresolved ambiguities on core fields (topic, desired_output)
    """
    if brief.needs_followup:
        return True
    if brief.confidence < 0.4:
        return True
    core_fields = {a.field for a in brief.ambiguities}
    if "topic" in core_fields or "desired_output" in core_fields:
        return True
    return False


def to_limited_brief(raw_query: str) -> ResearchBrief:
    """Return a conservative fallback ResearchBrief when all parsing fails.

    The fallback preserves high-signal hints from the raw query when possible
    (for example desired output, time range, domain keywords) so transient LLM
    failures do not erase user intent.

    Even with limited info, we aim to NOT set needs_followup=True unless truly necessary.
    A clear topic should result in a productive brief that can generate a report.
    """
    query = raw_query.strip()
    topic = _extract_topic(query) or "未提供研究主题"

    # Try to infer desired output from query patterns
    inferred_output = _infer_desired_output(query)
    desired_output = inferred_output or "survey"

    time_range = _infer_time_range(query)
    domain_scope = _infer_domain_scope(query)
    focus_dimensions = _infer_focus_dimensions(query)
    topic_specific = _is_topic_specific(query)

    # Only add ambiguities if truly critical info is missing
    # A clear topic is enough to proceed - don't block the pipeline
    ambiguities: list[AmbiguityItem] = []

    # Only set needs_followup if topic is genuinely unclear
    # For short but specific topics (like "Transformer architecture"),
    # we should NOT require followup - just proceed with what we have
    needs_followup = False  # Default to False to allow report generation
    if not topic_specific or topic == "未提供研究主题":
        needs_followup = True
        ambiguities.append(
            AmbiguityItem(
                field="topic",
                reason="原始查询信息不足，无法推断具体研究主题",
                suggested_options=["多模态学习", "RAG", "Agent", "报告生成"],
            )
        )
        # Also ask for desired output if topic is unclear
        if inferred_output is None:
            ambiguities.append(
                AmbiguityItem(
                    field="desired_output",
                    reason="原始查询没有明确说明期望的输出形式",
                    suggested_options=[
                        "survey_outline",
                        "paper_cards",
                        "reading_notes",
                        "related_work_draft",
                    ],
                )
            )

    # Confidence based on how much info we extracted
    if topic_specific and topic != "未提供研究主题":
        # We have a clear topic - moderate confidence
        confidence = 0.65
        if focus_dimensions:
            confidence = 0.75
    else:
        confidence = 0.35

    return ResearchBrief(
        topic=topic,
        goal=_infer_goal(desired_output),
        desired_output=desired_output,
        sub_questions=_infer_sub_questions(query, desired_output, focus_dimensions),
        time_range=time_range,
        domain_scope=domain_scope,
        source_constraints=[],
        focus_dimensions=focus_dimensions,
        ambiguities=ambiguities,
        needs_followup=needs_followup,
        confidence=confidence,
        schema_version="v1",
    )
