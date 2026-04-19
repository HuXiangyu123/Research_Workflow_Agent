"""Draft node — Phase 2: 综合 PaperCards 生成综述草稿。"""

from __future__ import annotations

from collections import Counter
import logging
from typing import Any

from src.agent.report_markdown import render_report_markdown
from src.models.report import Citation, Claim, DraftReport
from src.research.prompts.survey_writing import build_survey_writing_rules_block
from src.tasking.trace_wrapper import get_trace_store, trace_node

logger = logging.getLogger(__name__)

SURVEY_RULES_BLOCK = build_survey_writing_rules_block()
SURVEY_SECTION_ORDER = [
    "abstract",
    "introduction",
    "background",
    "taxonomy",
    "methods",
    "datasets",
    "evaluation",
    "discussion",
    "future_work",
    "conclusion",
]
SURVEY_SECTION_CITATION_FLOORS: dict[str, int] = {
    "introduction": 4,
    "background": 2,
    "taxonomy": 4,
    "methods": 5,
    "datasets": 3,
    "evaluation": 3,
    "discussion": 3,
    "future_work": 2,
}


def build_drafting_skill_artifacts(
    cards: list[Any],
    brief: Any | None,
    *,
    workspace_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Run drafting support skills and return structured artifacts for prompting."""
    from src.models.agent import AgentRole
    from src.models.skills import SkillRunRequest
    from src.skills.registry import get_skills_registry

    topic = (
        (brief.get("topic") if isinstance(brief, dict) else getattr(brief, "topic", ""))
        or ""
    ).strip()
    time_range = (
        (brief.get("time_range") if isinstance(brief, dict) else getattr(brief, "time_range", ""))
        or ""
    ).strip()
    focus_dimensions = (
        (brief.get("focus_dimensions") if isinstance(brief, dict) else getattr(brief, "focus_dimensions", []))
        or []
    )

    if not topic:
        return {"skill_trace": []}

    registry = get_skills_registry()
    context = {
        "workspace_id": workspace_id or "",
        "task_id": task_id,
        "_mcp_server_id": "academic_writing",
    }

    bundle: dict[str, Any] = {"skill_trace": []}

    def _run(skill_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
        req = SkillRunRequest(
            workspace_id=workspace_id or "",
            task_id=task_id,
            skill_id=skill_id,
            inputs=inputs,
            preferred_agent=AgentRole.ANALYST,
        )
        resp = registry.run_sync(req, context)
        bundle["skill_trace"].append(
            {
                "skill_id": skill_id,
                "summary": resp.summary,
                "backend": resp.backend.value,
            }
        )
        return resp.result

    try:
        comparison_result = _run(
            "comparison_matrix_builder",
            {
                "paper_cards": cards,
                "compare_dimensions": ["methods", "datasets", "benchmarks", "limitations"],
                "format": "table",
            },
        )
        bundle["comparison_matrix"] = comparison_result.get("matrix")
    except Exception as exc:  # noqa: BLE001
        bundle["skill_trace"].append({"skill_id": "comparison_matrix_builder", "error": str(exc)})

    try:
        scaffold_result = _run(
            "writing_scaffold_generator",
            {
                "topic": topic,
                "paper_cards": cards,
                "comparison_matrix": bundle.get("comparison_matrix") or {},
                "desired_length": "long",
            },
        )
        bundle["writing_scaffold"] = scaffold_result.get("scaffold")
        bundle["writing_outline"] = scaffold_result.get("outline")
        bundle["section_evidence_map"] = scaffold_result.get("section_evidence_map")
        bundle["writing_guidance"] = scaffold_result.get("writing_guidance")
    except Exception as exc:  # noqa: BLE001
        bundle["skill_trace"].append({"skill_id": "writing_scaffold_generator", "error": str(exc)})

    try:
        prompt_result = _run(
            "academic_review_writer_prompt",
            {
                "topic": topic,
                "time_range": time_range,
                "focus_dimensions": list(focus_dimensions) if isinstance(focus_dimensions, list) else [],
            },
        )
        bundle["mcp_prompt_payload"] = prompt_result
    except Exception as exc:  # noqa: BLE001
        bundle["skill_trace"].append({"skill_id": "academic_review_writer_prompt", "error": str(exc)})

    return bundle


@trace_node(node_name="draft", stage="draft", store=get_trace_store())
def draft_node(state: dict) -> dict:
    """
    Phase 2 草稿节点。

    输入：state.paper_cards, state.brief, 可选 state.compression_result
    输出：state.draft_report (DraftReport), state.draft_markdown (str)

    如果 state.compression_result 存在，使用压缩后的上下文（taxonomy + compressed_cards）
    否则使用原始 paper_cards。
    """
    paper_cards = state.get("paper_cards", [])
    brief = state.get("brief")
    compression_result = state.get("compression_result")
    workspace_id = str(state.get("workspace_id") or "")
    task_id = str(state.get("task_id") or "") or None
    emitter = state.get("_event_emitter")
    skill_artifacts = build_drafting_skill_artifacts(
        paper_cards,
        brief,
        workspace_id=workspace_id,
        task_id=task_id,
    )

    if not paper_cards:
        logger.warning("[draft_node] no paper_cards, skipping")
        return {"draft_report": None, "draft_markdown": None}

    _write_scaffold_preview(
        brief,
        skill_artifacts,
        task_id=task_id,
        workspace_id=workspace_id or None,
        emitter=emitter,
    )

    # ── 1. 构建结构化 DraftReport ────────────────────────────────
    # 根据是否有压缩结果选择上下文
    if compression_result:
        draft_report = _build_draft_with_compression(
            paper_cards, brief, compression_result, skill_artifacts=skill_artifacts
        )
    else:
        draft_report = _build_draft_report(paper_cards, brief, skill_artifacts=skill_artifacts)

    # ── 2. 生成可读 Markdown ─────────────────────────────────────
    markdown = _build_markdown(draft_report, brief)
    _stream_draft_markdown_snapshots(
        draft_report,
        brief,
        task_id=task_id,
        workspace_id=workspace_id or None,
        emitter=emitter,
    )

    logger.info(
        "[draft_node] drafted %d sections from %d papers",
        len(draft_report.sections),
        len(paper_cards),
    )
    return {
        "draft_report": draft_report,
        "draft_markdown": markdown,
        "comparison_matrix": skill_artifacts.get("comparison_matrix"),
        "writing_scaffold": skill_artifacts.get("writing_scaffold"),
        "writing_outline": skill_artifacts.get("writing_outline"),
        "section_evidence_map": skill_artifacts.get("section_evidence_map"),
        "mcp_prompt_payload": skill_artifacts.get("mcp_prompt_payload"),
        "skill_trace": skill_artifacts.get("skill_trace", []),
        # 关键：传递 paper_cards 到 state，供后续 verify_claims 使用
        "paper_cards": paper_cards,
    }


# ─── internal helpers ────────────────────────────────────────────────────────────


def _compose_survey_system_prompt(*, compressed_context: bool) -> str:
    context_note = (
        "IMPORTANT: You are working with STRUCTURED COMPRESSED CONTEXT that preserves taxonomy, "
        "compressed claims, and section-wise evidence allocation.\n"
        if compressed_context
        else "IMPORTANT: You are working from paper cards whose evidence may come from full-text snippets, "
        "DeepXiv TLDRs, or abstracts. Prefer full-text evidence whenever present.\n"
    )
    return "\n".join(
        [
            "You are an expert academic survey writer. Generate a comprehensive, evidence-grounded survey report.",
            "The output MUST be strictly valid JSON with no markdown code fences.",
            "",
            context_note,
            SURVEY_RULES_BLOCK,
            "",
            "ADDITIONAL EXECUTION RULES:",
            "1. Analyze before writing. Identify patterns, method families, benchmark clusters, and unresolved debates across papers.",
            "2. Synthesize instead of paper-by-paper paraphrase.",
            "3. NEVER repeat abstract wording when discussing methods; explain the technical approach, design trade-offs, and reported evidence in your own analytical voice.",
            "4. Future work MUST be derived from cross-paper gaps, missing evidence, evaluation blind spots, or deployment constraints; do not copy limitation sentences into future_work.",
            "5. Cite representative papers in every substantive section using citation labels like [1], [2].",
            "6. Maintain citation diversity. Do not let one or two papers dominate every major section when broader support exists in the corpus.",
            "7. Keep central claims close to direct evidence. Strong synthesis claims should usually cite multiple directly relevant papers instead of one weak proxy citation.",
            "",
            "JSON OUTPUT SCHEMA:",
            '{',
            '  "sections": {',
            '    "title": "Survey paper title in English, specific and publication-ready",',
            '    "abstract": "Executive summary (400-600 chars) — motivation, scope, key findings, contributions",',
            '    "introduction": "Comprehensive introduction (1500-2500 chars) — field evolution, motivation, scope, and organizing logic. Must reference at least 8 papers.",',
            '    "background": "Background & motivation (800-1200 chars) — core concepts, task setting, historical context, why the topic matters.",',
            '    "taxonomy": "Detailed taxonomy & categorization (1500-2000 chars) — organize methods/approaches hierarchically and name representative papers.",',
            '    "methods": "Core methods deep-dive (2000-3000 chars) — method families, technical design, innovations, strengths, weaknesses, and evidence. Must reference at least 10 papers.",',
            '    "datasets": "Datasets & experimental settings (800-1200 chars) — datasets, evaluation metrics, protocols; table format is welcome.",',
            '    "evaluation": "Performance comparison & analysis (1200-1800 chars) — compare methods on reported benchmarks and metrics.",',
            '    "discussion": "Discussion & insights (1000-1500 chars) — agreements, conflicts, trade-offs, reproducibility issues, and lessons learned.",',
            '    "future_work": "Open challenges & future directions (800-1200 chars) — gap-driven future directions, not a paraphrase of limitations.",',
            '    "conclusion": "Conclusion (400-600 chars) — synthesized takeaways and broader significance"',
            "  },",
            '  "claims": [{"id": "c1", "text": "verifiable synthesized claim", "citation_labels": ["[1]", "[2]"]}],',
            '  "citations": [{"label": "[1]", "url": "https://arxiv.org/abs/...", "reason": "why this paper is cited", "arxiv_id": "2301.01234"}]',
            '}',
        ]
    )


def _render_skill_context(skill_artifacts: dict[str, Any] | None) -> str:
    if not skill_artifacts:
        return ""

    lines: list[str] = []
    scaffold = skill_artifacts.get("writing_scaffold") or {}
    outline = skill_artifacts.get("writing_outline") or []
    matrix = skill_artifacts.get("comparison_matrix") or {}
    section_evidence_map = skill_artifacts.get("section_evidence_map") or {}
    prompt_payload = skill_artifacts.get("mcp_prompt_payload") or {}
    writing_guidance = str(skill_artifacts.get("writing_guidance") or "").strip()

    if isinstance(prompt_payload, dict) and prompt_payload.get("prompt"):
        lines.append("## MCP Writing Prompt")
        lines.append(str(prompt_payload.get("prompt")).strip())
        lines.append("")

    if writing_guidance:
        lines.append("## Writing Guidance")
        lines.append(writing_guidance)
        lines.append("")

    if isinstance(outline, list) and outline:
        lines.append("## Scaffold Outline")
        lines.extend(f"- {item}" for item in outline[:12])
        lines.append("")

    if isinstance(scaffold, dict) and scaffold:
        lines.append("## Section Planning Scaffold")
        for section, plan in scaffold.items():
            if section == "title":
                lines.append(f"- Title guidance: {plan}")
                continue
            if isinstance(plan, list):
                joined = "; ".join(str(item).strip() for item in plan[:3] if str(item).strip())
                if joined:
                    lines.append(f"- {section}: {joined}")
            elif isinstance(plan, str) and plan.strip():
                lines.append(f"- {section}: {plan.strip()}")
        lines.append("")

    if isinstance(section_evidence_map, dict) and section_evidence_map:
        lines.append("## Section Evidence Map")
        for section, papers in section_evidence_map.items():
            if not isinstance(papers, list):
                continue
            selected = [str(item).strip() for item in papers[:6] if str(item).strip()]
            if selected:
                lines.append(f"- {section}: {'; '.join(selected)}")
        lines.append("")

    if isinstance(matrix, dict):
        rows = matrix.get("rows", [])
        if isinstance(rows, list) and rows:
            lines.append("## Comparison Matrix Highlights")
            for row in rows[:8]:
                if not isinstance(row, dict):
                    continue
                paper = str(row.get("paper") or "").strip()
                methods = str(row.get("methods") or "").strip()
                datasets = str(row.get("datasets") or "").strip()
                limitations = str(row.get("limitations") or "").strip()
                summary = "; ".join(
                    part
                    for part in (
                        f"methods={methods}" if methods else "",
                        f"datasets={datasets}" if datasets else "",
                        f"limitations={limitations}" if limitations else "",
                    )
                    if part
                )
                if paper and summary:
                    lines.append(f"- {paper}: {summary}")
            lines.append("")

    return "\n".join(lines).strip()


def _build_draft_report(
    cards: list[Any],
    brief: Any | None,
    *,
    skill_artifacts: dict[str, Any] | None = None,
) -> DraftReport:
    """用 LLM 综合 PaperCards 生成结构化 DraftReport。"""
    from src.agent.llm import build_reason_llm
    from src.agent.settings import get_settings
    from langchain_core.messages import HumanMessage, SystemMessage

    brief_ctx = _build_brief_context(brief)
    skill_ctx = _render_skill_context(skill_artifacts)
    # 传给 LLM 最多 20 张卡片（每张 ~1500 chars abstract ≈ ~750 tokens，20 张 ≈ 15k tokens）
    # 加上 system prompt (~2k tokens) + brief ctx (~500 tokens) + output (~8k tokens) = ~26k tokens，在 context 窗口内
    cards_text = _render_cards(cards[:20])

    SYSTEM = _compose_survey_system_prompt(compressed_context=False)

    USER = (
        "{brief_ctx}"
        "{skill_ctx}"
        "\n\n## Paper Cards\n{cards_text}\n\n"
        "Generate a detailed survey draft from the paper cards above.\n"
        "Analysis requirements:\n"
        "1. Read all cards first and identify themes, method families, and cross-paper relationships.\n"
        "2. Synthesize across papers instead of summarizing one paper at a time.\n"
        "3. The introduction must cite at least 8 papers and explain the field trajectory.\n"
        "4. The methods section must cite at least 10 papers and analyze each method family.\n"
        "5. Do not leave any placeholder text; every section must contain substantive content.\n\n"
        "CRITICAL: All placeholder text MUST be replaced with actual content from the paper cards:\n"
        "- [Research Topic] -> use the brief topic or infer it from the paper cards\n"
        "- [Dataset X], [Dataset Y], [Synthetic Dataset Z] -> replace with actual datasets from paper_cards\n"
        "- [Example A1], [Example B1], [Example C1] -> replace with actual paper titles or method names from paper_cards\n"
        "- [Adjacent Field A], [Adjacent Field B] -> derive them from the provided evidence\n"
        "- Keep the title, introduction, and conclusion aligned with the requested time range; do not broaden the survey window\n"
        "- Use background-only or off-topic component papers only as brief supporting context, not as the center of the synthesis\n"
        "- Do not output any [...] placeholders; all content must come from the paper card data\n"
    ).format(brief_ctx=brief_ctx, skill_ctx=f"\n\n{skill_ctx}" if skill_ctx else "", cards_text=cards_text)

    try:
        settings = get_settings()
        llm = build_reason_llm(settings, max_tokens=16384, timeout_s=300)
        resp = llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=USER)])
        text = getattr(resp, "content", "") or ""

        import json as _json
        raw = _extract_json(text)
        if raw:
            data = _json.loads(raw)
            sections = data.get("sections", {})
            claims = [Claim(**c) for c in data.get("claims", []) if c.get("id") and c.get("text")]
            citations = [Citation(**c) for c in data.get("citations", []) if c.get("label") and c.get("url")]

            # ── 关键修复：用 paper_cards 内容填充 citation.fetched_content ──
            citations = _inject_citation_content(citations, cards)

            return _ensure_minimum_citation_coverage(
                DraftReport(
                    sections=sections,
                    claims=claims,
                    citations=citations,
                ),
                cards,
                brief=brief,
                skill_artifacts=skill_artifacts,
            )
    except Exception as exc:
        logger.warning("[draft_node] LLM failed: %s, using template fallback", exc)

    # Fallback：基于 cards 构造基础 sections
    draft = _fallback_draft(cards, brief, skill_artifacts=skill_artifacts)
    # Fallback 也要注入 content
    draft.citations = _inject_citation_content(draft.citations, cards)
    return _ensure_minimum_citation_coverage(
        draft,
        cards,
        brief=brief,
        skill_artifacts=skill_artifacts,
    )


def _build_draft_with_compression(
    cards: list[Any],
    brief: Any | None,
    compression_result: dict,
    *,
    skill_artifacts: dict[str, Any] | None = None,
) -> DraftReport:
    """
    使用压缩后的上下文生成 DraftReport。

    当 extract_compression_node 已生成 taxonomy + compressed_cards + evidence_pools 时，
    使用这些压缩后的上下文而非原始 paper_cards。

    压缩上下文的优势：
    - taxonomy 提供了论文分类和跨分类主题，帮助 LLM 理解整体结构
    - compressed_cards 将每张卡片从 ~1500 chars 压缩到 ~300 chars
    - evidence_pools 按 section 分配论文，避免重复引用
    """
    from src.agent.llm import build_reason_llm
    from src.agent.settings import get_settings
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.models.compression import CompressionResult, CompressedCard, EvidencePool, Taxonomy

    # 反序列化 compression_result
    if isinstance(compression_result, dict):
        result = CompressionResult(**compression_result)
    else:
        result = compression_result

    taxonomy = result.taxonomy
    compressed_cards = result.compressed_cards
    evidence_pools = result.evidence_pools

    brief_ctx = _build_brief_context(brief)
    skill_ctx = _render_skill_context(skill_artifacts)

    # 构建 taxonomy 描述文本
    taxonomy_text = _render_taxonomy(taxonomy)

    # 构建 compressed cards 文本
    compressed_text = _render_compressed_cards(compressed_cards)

    # 构建 evidence pools 摘要
    pools_text = _render_evidence_pools(evidence_pools)

    SYSTEM = _compose_survey_system_prompt(compressed_context=True)

    USER = (
        "{brief_ctx}\n\n"
        "{skill_ctx}\n\n"
        "## Taxonomy\n{taxonomy_text}\n\n"
        "## Compressed Paper Cards\n{compressed_text}\n\n"
        "## Evidence Allocation By Section\n{pools_text}\n\n"
        "Generate a detailed survey draft from the compressed context above.\n"
        "Analysis requirements:\n"
        "1. Understand the taxonomy first; it captures categories and cross-category themes.\n"
        "2. Use the core_claim and key_result fields from compressed_cards to drive the analysis.\n"
        "3. The introduction must cite at least 8 papers and explain the field trajectory.\n"
        "4. The methods section must cite at least 10 papers and analyze each method family.\n"
        "5. Do not leave any placeholder text; every section must contain substantive content.\n\n"
        "CRITICAL: All placeholder text MUST be replaced with actual content from the provided context:\n"
        "- Keep the time range and topic boundary exactly aligned with the brief.\n"
        "- Use the comparison matrix and writing scaffold to drive section planning and evidence allocation.\n"
        "- Do not output any [...] placeholders; all content must come from the compressed context\n"
    ).format(
        brief_ctx=brief_ctx,
        skill_ctx=skill_ctx or "(No additional skill context)",
        taxonomy_text=taxonomy_text,
        compressed_text=compressed_text,
        pools_text=pools_text,
    )

    try:
        settings = get_settings()
        llm = build_reason_llm(settings, max_tokens=16384, timeout_s=300)
        resp = llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=USER)])
        text = getattr(resp, "content", "") or ""

        import json as _json

        raw = _extract_json(text)
        if raw:
            data = _json.loads(raw)
            sections = data.get("sections", {})
            claims = [
                Claim(**c) for c in data.get("claims", []) if c.get("id") and c.get("text")
            ]
            citations = [
                Citation(**c) for c in data.get("citations", []) if c.get("label") and c.get("url")
            ]

            # 注入 citation content
            citations = _inject_citation_content(citations, cards)

            return _ensure_minimum_citation_coverage(
                DraftReport(
                    sections=sections,
                    claims=claims,
                    citations=citations,
                ),
                cards,
                brief=brief,
                skill_artifacts=skill_artifacts,
            )
    except Exception as exc:
        logger.warning(
            "[draft_node] compression-aware draft failed: %s, falling back to original",
            exc,
        )

    # Fallback：使用原始 cards
    return _build_draft_report(cards, brief, skill_artifacts=skill_artifacts)


def _render_taxonomy(taxonomy: Taxonomy) -> str:
    """渲染 taxonomy 为文本。"""
    if not taxonomy or not taxonomy.categories:
        return "(No taxonomy information)"

    lines = []
    lines.append("=== Taxonomy Structure ===\n")

    for cat in taxonomy.categories:
        lines.append(f"\n[{cat.name}] {cat.description}")
        if cat.papers:
            lines.append(f"  Papers: {', '.join(cat.papers[:5])}")
        if cat.shared_insights:
            lines.append(f"  Shared insights: {'; '.join(cat.shared_insights[:3])}")

    if taxonomy.cross_category_themes:
        lines.append("\n=== Cross-Category Themes ===")
        for theme in taxonomy.cross_category_themes[:5]:
            lines.append(f"- {theme}")

    if taxonomy.key_papers:
        lines.append("\n=== Key Papers (Must Cite) ===")
        for paper in taxonomy.key_papers[:5]:
            lines.append(f"- {paper}")

    return "\n".join(lines)


def _render_compressed_cards(compressed_cards: list[CompressedCard]) -> str:
    """渲染压缩后的卡片为文本。"""
    if not compressed_cards:
        return "(No paper cards)"

    lines = []
    lines.append(f"=== Compressed Paper Cards ({len(compressed_cards)} total) ===\n")

    for i, card in enumerate(compressed_cards, 1):
        lines.append(f"\n[{i}] {card.title}")
        lines.append(f"    Core claim: {card.core_claim}")
        if card.method_type:
            lines.append(f"    Method type: {card.method_type}")
        if card.key_result:
            lines.append(f"    Key result: {card.key_result}")
        if card.connections:
            lines.append(f"    Connections: {', '.join(card.connections[:2])}")

    return "\n".join(lines)


def _render_evidence_pools(evidence_pools: dict[str, EvidencePool]) -> str:
    """渲染 evidence pools 为文本。"""
    if not evidence_pools:
        return "(No section allocation information)"

    lines = []
    lines.append("=== Paper Allocation by Section ===\n")

    for section, pool in evidence_pools.items():
        if not pool.papers:
            continue
        titles = [p.card.title for p in pool.papers]
        lines.append(f"\n[{section}] ({len(pool.papers)} papers)")
        for title in titles[:3]:
            lines.append(f"  - {title}")
        if len(titles) > 3:
            lines.append(f"  ... and {len(titles)} papers total")

    return "\n".join(lines)


def _build_markdown(draft: DraftReport, brief: Any | None) -> str:
    """将 DraftReport 渲染为可读 Markdown。"""
    topic = ""
    if brief:
        topic = (brief.get("topic") if isinstance(brief, dict) else getattr(brief, "topic", "")) or ""
    return render_report_markdown(
        sections=draft.sections,
        citations=draft.citations,
        title=str(draft.sections.get("title") or topic or "").strip() or None,
        section_order=SURVEY_SECTION_ORDER,
    )


def _write_scaffold_preview(
    brief: Any | None,
    skill_artifacts: dict[str, Any] | None,
    *,
    task_id: str | None,
    workspace_id: str | None,
    emitter: Any | None,
) -> None:
    if not task_id or not workspace_id:
        return

    outline = (skill_artifacts or {}).get("writing_outline") or []
    if not isinstance(outline, list) or not outline:
        return

    topic = (brief.get("topic") if isinstance(brief, dict) else getattr(brief, "topic", "")) if brief else ""
    scaffold = (skill_artifacts or {}).get("writing_scaffold") or {}
    title = str((scaffold.get("title") if isinstance(scaffold, dict) else "") or topic or "Research Survey").strip()
    preview_lines = [
        f"# {title}",
        "",
        "## Draft Plan",
        "",
        "Generating the survey section by section. This scaffold preview will be replaced by drafted content as soon as each section is ready.",
        "",
        "## Planned Sections",
        "",
    ]
    preview_lines.extend(f"- {item}" for item in outline[:12])

    try:
        from src.agent.output_workspace import write_draft

        write_draft(task_id, "\n".join(preview_lines).strip() + "\n", workspace_id=workspace_id)
        if emitter:
            emitter.on_thinking("draft", "Published scaffold preview to the workspace.")
    except Exception as exc:  # noqa: BLE001
        logger.debug("[draft_node] failed to write scaffold preview: %s", exc)


def _stream_draft_markdown_snapshots(
    draft: DraftReport,
    brief: Any | None,
    *,
    task_id: str | None,
    workspace_id: str | None,
    emitter: Any | None,
) -> None:
    if not task_id or not workspace_id:
        return

    try:
        from src.agent.output_workspace import write_draft
    except Exception:  # pragma: no cover - import failure should silently skip live preview
        return

    title = str(
        draft.sections.get("title")
        or ((brief.get("topic") if isinstance(brief, dict) else getattr(brief, "topic", "")) if brief else "")
        or "Research Survey"
    ).strip()
    available_sections = [key for key in SURVEY_SECTION_ORDER if str(draft.sections.get(key, "") or "").strip()]
    partial_sections: dict[str, str] = {"title": title}

    for index, section_key in enumerate(available_sections, start=1):
        partial_sections[section_key] = str(draft.sections.get(section_key) or "").strip()
        snapshot = render_report_markdown(
            sections=partial_sections,
            citations=None,
            title=title,
            section_order=SURVEY_SECTION_ORDER,
        )
        try:
            write_draft(task_id, snapshot, workspace_id=workspace_id)
            if emitter:
                emitter.on_thinking(
                    "draft",
                    f"Live draft preview updated: section {index}/{len(available_sections)} ({section_key}).",
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[draft_node] failed to stream draft snapshot: %s", exc)
            break

    try:
        write_draft(task_id, _build_markdown(draft, brief), workspace_id=workspace_id)
        if emitter:
            emitter.on_thinking("draft", "Live draft preview updated with the complete markdown and references.")
    except Exception as exc:  # noqa: BLE001
        logger.debug("[draft_node] failed to write final streamed draft snapshot: %s", exc)


def _fallback_draft(
    cards: list[Any],
    brief: Any | None,
    *,
    skill_artifacts: dict[str, Any] | None = None,
) -> DraftReport:
    """
    当 LLM 综合失败时，基于 cards 构造 DraftReport。

    策略：不用模板占位，而是从每张 card 的 abstract 中提取并重组内容，
    生成与卡片内容严格对应的结构化报告。
    """
    sections: dict[str, str] = {}

    topic = ""
    sub_questions = []
    time_range = ""
    if brief:
        topic = (brief.get("topic") if isinstance(brief, dict) else getattr(brief, "topic", "")) or ""
        sq = brief.get("sub_questions") if isinstance(brief, dict) else getattr(brief, "sub_questions", [])
        sub_questions = sq if isinstance(sq, list) else []
        time_range = (brief.get("time_range") if isinstance(brief, dict) else getattr(brief, "time_range", "")) or ""

    scaffold = (skill_artifacts or {}).get("writing_scaffold") or {}
    matrix = (skill_artifacts or {}).get("comparison_matrix") or {}
    matrix_rows = matrix.get("rows", []) if isinstance(matrix, dict) else []
    scaffold_title = scaffold.get("title") if isinstance(scaffold, dict) else None

    sections["title"] = str(scaffold_title or (f"{topic} Survey" if topic else "Research Survey")).strip()

    # ── 1. 收集所有卡片数据 ──────────────────────────────────────────
    all_abstracts: list[dict] = []
    all_methods: list[dict] = []  # {paper_idx, method}
    all_datasets: list[dict] = []  # {paper_idx, dataset}
    all_limitations: list[dict] = []

    for i, card in enumerate(cards[:20]):
        abstract = _get_field(card, "abstract", "") or _get_field(card, "summary", "")
        title = _get_field(card, "title", f"Paper {i+1}")
        arxiv_id = _get_field(card, "arxiv_id", "")
        authors = _get_field(card, "authors", [])
        authors_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "") if authors else "Unknown"

        card_meta = {
            "idx": i + 1,
            "label": f"[{i+1}]",
            "title": title,
            "authors": authors_str,
            "arxiv_id": arxiv_id,
            "abstract": abstract,
        }

        if abstract and len(abstract) > 30:
            all_abstracts.append(card_meta)

        # 收集方法
        methods = _get_field(card, "methods", [])
        if isinstance(methods, list):
            for m in methods:
                all_methods.append({**card_meta, "method": str(m)})

        # 收集数据集
        datasets = _get_field(card, "datasets", [])
        if isinstance(datasets, list):
            for d in datasets:
                all_datasets.append({**card_meta, "dataset": str(d)})

        # 收集局限性
        limitations = _get_field(card, "limitations", [])
        if isinstance(limitations, list):
            for l in limitations:
                all_limitations.append({**card_meta, "limitation": str(l)})

    method_labels = [m["method"] for m in all_methods if m.get("method")]
    dataset_labels = [d["dataset"] for d in all_datasets if d.get("dataset")]
    limitation_labels = [l["limitation"] for l in all_limitations if l.get("limitation")]
    top_methods = _top_ranked_terms(method_labels, limit=4)
    top_datasets = _top_ranked_terms(dataset_labels, limit=4)
    top_limitations = _top_ranked_terms(limitation_labels, limit=4)
    scope_text = f" during {time_range}" if time_range else ""

    # ── 2. Abstract ─────────────────────────────────────────────────
    if all_abstracts:
        sections["abstract"] = (
            f"This survey examines {topic or 'the target research area'}{scope_text} through {len(all_abstracts)} representative papers. "
            f"The evidence clusters around {', '.join(top_methods) or 'agent architectures, multimodal grounding, and evaluation design'}. "
            f"Across the corpus, the strongest recurring gaps concern {', '.join(top_limitations) or 'grounding, evaluation coverage, and deployment evidence'}."
        )

    # ── 3. Introduction ────────────────────────────────────────────
    intro_parts = [
        f"This survey focuses on {topic or 'the target research area'}{scope_text} and analyzes {len(all_abstracts)} relevant papers."
    ]
    if sub_questions:
        intro_parts.append(
            "It is organized around the following research questions: "
            + "; ".join(sub_questions[:3])
            + "."
        )
    if top_methods:
        intro_parts.append(
            "The synthesis is structured by method families including "
            + ", ".join(top_methods)
            + "."
        )
    sections["introduction"] = "".join(intro_parts)

    # ── 4. Background ───────────────────────────────────────────────
    if all_abstracts:
        bg_parts = [
            f"The background for {topic or 'this topic'} combines multimodal perception, task-specific tooling, and workflow-level reasoning. ",
            (
                f"Representative enabling directions include {', '.join(top_methods[:3])}. "
                if top_methods else
                "Representative enabling directions include multimodal modelling, tool use, and task-specific evaluation. "
            ),
            (
                f"Common benchmark settings mention {', '.join(top_datasets[:3])}. "
                if top_datasets else
                "Benchmark reporting is often incomplete in the extracted evidence, which itself is a major survey finding. "
            ),
            "This section therefore frames the field at the level of capability classes rather than repeating each paper abstract.\n",
        ]
        sections["background"] = "".join(bg_parts)

    # ── 5. Taxonomy（从论文中归纳分类）─────────────────────────────────
    if all_abstracts:
        taxonomy_parts = [
            f"Based on {len(all_abstracts)} papers, the field can be organized into the following categories:\n\n"
        ]
        category_names = top_methods or [
            "end-to-end agent systems",
            "multimodal grounding modules",
            "evaluation and deployment studies",
        ]
        for idx, category in enumerate(category_names[:4], start=1):
            taxonomy_parts.append(f"**Category {idx}: {category.title()}**\n")
            matched = [
                c for c in all_abstracts
                if category.lower() in c["title"].lower()
                or category.lower() in c["abstract"].lower()
            ]
            if not matched and idx == 1:
                matched = all_abstracts[:3]
            for c in matched[:4]:
                taxonomy_parts.append(f"- {c['label']} {c['title']} — {c['authors']}\n")
            taxonomy_parts.append("\n")

        sections["taxonomy"] = "".join(taxonomy_parts)

    # ── 6. Methods ─────────────────────────────────────────────────
    if all_methods:
        method_parts = ["The main method families covered in this survey are:\n\n"]
        # 按方法名分组
        method_map: dict[str, list] = {}
        for m_info in all_methods:
            key = m_info["method"]
            if key not in method_map:
                method_map[key] = []
            method_map[key].append(m_info)

        for method, occurrences in method_map.items():
            papers = ", ".join(f"{occ['label']}" for occ in occurrences[:3])
            method_parts.append(
                f"- **{method}**: appears in {len(occurrences)} papers ({papers})\n"
            )
        sections["methods"] = "".join(method_parts)
    elif all_abstracts:
        # 从 abstract 推断方法
        method_parts = ["The main methods inferred from the abstracts are:\n\n"]
        for c in all_abstracts[:8]:
            method_parts.append(
                f"- **{c['label']} {c['title']}**: "
                f"{c['abstract'][:300]}...\n\n"
            )
        sections["methods"] = "".join(method_parts)

    # ── 7. Datasets ────────────────────────────────────────────────
    if all_datasets:
        dataset_parts = ["The main datasets and benchmarks referenced in the survey are:\n\n"]
        dataset_parts.append("| Dataset / Benchmark | Papers | Notes |\n")
        dataset_parts.append("|---|---|---|\n")
        dataset_map: dict[str, list] = {}
        for d_info in all_datasets:
            key = d_info["dataset"]
            if key not in dataset_map:
                dataset_map[key] = []
            dataset_map[key].append(d_info)
        for dataset, occurrences in dataset_map.items():
            papers = ", ".join(occ["label"] for occ in occurrences[:5])
            dataset_parts.append(f"| {dataset} | {papers} | |\n")
        sections["datasets"] = "".join(dataset_parts)
    elif all_abstracts:
        sections["datasets"] = (
            "Datasets mentioned in the paper set include "
            + ", ".join(c.get("title", "") for c in all_abstracts[:6])
            + ". See the individual papers for exact benchmark details."
        )

    # ── 8. Evaluation ───────────────────────────────────────────
    if all_abstracts:
        benchmark_rows = [
            row for row in matrix_rows
            if isinstance(row, dict) and str(row.get("benchmarks", "")).strip()
        ]
        missing_benchmark_count = sum(
            1 for row in matrix_rows
            if isinstance(row, dict) and str(row.get("benchmarks", "")).startswith("Not ")
        )
        eval_parts = [
            "Evaluation reporting remains heterogeneous across the retrieved corpus.\n\n",
            (
                "Named benchmark or metric signals include "
                + ", ".join(
                    str(row.get("benchmarks", "")).strip()
                    for row in benchmark_rows[:5]
                    if str(row.get("benchmarks", "")).strip()
                )
                + ".\n\n"
                if benchmark_rows else
                ""
            ),
            (
                f"{missing_benchmark_count} papers do not expose usable benchmark detail in the extracted evidence, "
                "which limits precise cross-paper comparison.\n\n"
                if missing_benchmark_count else
                ""
            ),
            "The survey should therefore interpret evaluation claims cautiously and distinguish component-level evidence from end-to-end clinical validation.\n",
        ]
        sections["evaluation"] = "".join(eval_parts)

    # ── 9. Discussion ──────────────────────────────────────────────
    if all_limitations:
        disc_parts = [
            "Cross-paper comparison reveals several recurring trade-offs.\n\n",
            (
                f"Agreement: the literature consistently treats {', '.join(top_methods[:2]) or 'multimodal grounding and tool use'} as central design levers.\n"
            ),
            (
                f"Trade-off: broader system flexibility often comes with weaker evidence on {', '.join(top_limitations[:2]) or 'grounding quality and deployment readiness'}.\n"
            ),
            (
                f"Evidence gap: dataset and benchmark disclosure remains uneven, with recurring mentions of {', '.join(top_datasets[:2]) or 'incomplete evaluation settings'}.\n"
            ),
        ]
        sections["discussion"] = "".join(disc_parts)
    elif all_abstracts:
        sections["discussion"] = (
            "The collected papers propose different innovations. "
            + ", ".join(c["title"] for c in all_abstracts[:3])
            + " are representative examples, with trade-offs discussed throughout the survey."
        )

    # ── 10. Future Work ─────────────────────────────────────────
    future_directions = _synthesize_future_work(
        topic=topic,
        cards=cards,
        limitations=all_limitations,
        methods=all_methods,
        datasets=all_datasets,
    )
    sections["future_work"] = "Future work should prioritize the following directions:\n" + "\n".join(
        f"- {direction}" for direction in future_directions
    )

    # ── 11. Conclusion ──────────────────────────────────────────
    sections["conclusion"] = (
        f"This survey synthesized {len(all_abstracts)} papers on {topic or 'the target research area'}. "
        "It outlined the main methodological directions, highlighted common datasets and evaluation practices, "
        "and distilled the major limitations and next-step opportunities."
    )

    # ── 12. Citations ─────────────────────────────────────────────
    citations: list[Citation] = []
    claims: list[Claim] = []
    for i, card in enumerate(cards[:20]):
        label = f"[{i+1}]"
        title = _get_field(card, "title", f"Paper {i+1}")
        arxiv_id = _get_field(card, "arxiv_id", "")
        url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else _get_field(card, "url", "")
        title_str = title if isinstance(title, str) and title else "Unknown"
        full_abstract = _get_field(card, "abstract", "") or _get_field(card, "summary", "")
        citations.append(Citation(
            label=label,
            url=url,
            reason=title_str,
            fetched_content=full_abstract[:1000] if full_abstract else "",
        ))
        claim_text = full_abstract[:300] if full_abstract else f"{title_str} is an important paper in this area"
        claims.append(Claim(
            id=f"c{i+1}",
            text=claim_text,
            citation_labels=[label],
        ))

    return DraftReport(sections=sections, claims=claims, citations=citations)


def _synthesize_future_work(
    *,
    topic: str,
    cards: list[Any],
    limitations: list[dict],
    methods: list[dict],
    datasets: list[dict],
) -> list[str]:
    """Derive future directions from gaps instead of copying limitations.

    The synthesis intentionally maps recurring limitation signals into
    researchable directions so fallback output does not simply rewrite the
    raw limitation sentence with a future-tense suffix.
    """
    signals: set[str] = set()
    for item in limitations:
        text = str(item.get("limitation", "")).lower()
        if not text:
            continue
        if any(token in text for token in ("general", "domain shift", "泛化", "cross-domain", "robust")):
            signals.add("generalization")
        if any(token in text for token in ("efficien", "latency", "cost", "deploy", "resource", "效率", "部署", "成本")):
            signals.add("efficiency")
        if any(token in text for token in ("interpret", "explain", "trust", "safety", "可解释", "安全")):
            signals.add("trustworthiness")
        if any(token in text for token in ("benchmark", "dataset bias", "metric", "evaluation", "数据集", "评测", "指标")):
            signals.add("evaluation")
        if any(token in text for token in ("multimodal", "tool", "workflow", "memory", "多模态", "工具", "工作流", "记忆")):
            signals.add("system_design")

    if len({str(d.get("dataset", "")).strip().lower() for d in datasets if d.get("dataset")}) <= 2:
        signals.add("evaluation")

    if len({str(m.get("method", "")).strip().lower() for m in methods if m.get("method")}) <= 2:
        signals.add("method_diversity")

    if any(_get_field(card, "fulltext_available", False) is False for card in cards[:8]):
        signals.add("evidence_coverage")

    topic_display = topic or "this research area"
    mapped: list[str] = []
    for signal in sorted(signals):
        if signal == "generalization":
            mapped.append(
                f"Build cross-dataset, cross-institution, or cross-environment validation protocols for {topic_display} to measure generalization under distribution shift."
            )
        elif signal == "efficiency":
            mapped.append(
                "Improve inference latency, tool-call cost, and deployment efficiency without sacrificing effectiveness."
            )
        elif signal == "trustworthiness":
            mapped.append(
                "Strengthen interpretability, safety constraints, and failure diagnosis so systems can signal when to answer, abstain, or fall back."
            )
        elif signal == "evaluation":
            mapped.append(
                "Establish more consistent benchmarks, metric definitions, and realistic task settings so results are comparable across papers."
            )
        elif signal == "system_design":
            mapped.append(
                "Study the joint design of multimodal perception, tool orchestration, memory management, and long-horizon planning instead of optimizing each module in isolation."
            )
        elif signal == "method_diversity":
            mapped.append(
                "Explore hybrid architectures and stronger baselines beyond the dominant paradigm to clarify where each method family works best."
            )
        elif signal == "evidence_coverage":
            mapped.append(
                "Collect fuller full-text evidence and reproducibility material, especially experimental settings, error analyses, and failure cases."
            )

    if not mapped:
        mapped = [
            f"Define more consistent tasks, data benchmarks, and evaluation criteria for {topic_display} so results remain comparable across studies.",
            "Expand real-world validation, error analysis, and failure reporting instead of focusing only on best-case performance numbers.",
            "Characterize the trade-offs among capability, engineering cost, and interpretability to guide deployment and scaling decisions.",
        ]

    return mapped[:5]


def _top_ranked_terms(values: list[str], *, limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned:
            continue
        counts[cleaned] = counts.get(cleaned, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))
    return [item[0] for item in ranked[:limit]]


def _get_field(obj: Any, key: str, default: str = "") -> Any:
    """
    安全获取字段。None 值被视为缺失，返回默认值。
    """
    if isinstance(obj, dict):
        val = obj.get(key, default)
        return val if val is not None else default
    val = getattr(obj, key, default)
    return val if val is not None else default


def _extract_arxiv_id_from_url(url: str) -> str | None:
    """从 URL 中提取 arXiv ID。"""
    import re
    patterns = [
        r"arxiv\.org/abs/(\d+\.\d+)",
        r"arxiv\.org/pdf/(\d+\.\d+)",
        r"arxiv\.org/(\d+\.\d+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def _inject_citation_content(citations: list[Citation], cards: list[Any]) -> list[Citation]:
    """
    关键修复：用 paper_cards 的内容填充 citation.fetched_content。

    问题：draft 生成的 citations 只有 url/reason，没有 paper 内容，
    导致 resolve_citations 无法获取足够 evidence 来验证 claim。

    解决：直接用对应 paper_card 的 abstract/summary 作为 fetched_content。
    """
    updated: list[Citation] = []

    for cit in citations:
        # 尝试匹配 paper_card
        matched_content: str | None = None
        cit_arxiv_id = _extract_arxiv_id_from_url(cit.url) if cit.url else None

        for card in cards:
            card_arxiv_id = _get_field(card, "arxiv_id", "")
            card_url = _get_field(card, "url", "")
            card_title = _get_field(card, "title", "")

            # 匹配逻辑：arxiv_id 或 url 匹配
            matched = False
            if cit_arxiv_id and card_arxiv_id and cit_arxiv_id == card_arxiv_id:
                matched = True
            elif card_url and cit.url and card_url == cit.url:
                matched = True
            elif card_title and cit.reason and card_title.lower() in cit.reason.lower():
                matched = True

            if matched:
                # 优先使用正文 snippets，其次 content，再回退 abstract/summary
                snippets = _get_field(card, "fulltext_snippets", [])
                snippet_parts: list[str] = []
                if isinstance(snippets, list):
                    for item in snippets[:5]:
                        if not isinstance(item, dict):
                            continue
                        section = str(item.get("section") or "unknown").strip() or "unknown"
                        text = str(item.get("text") or "").strip()
                        if text:
                            snippet_parts.append(f"[{section}] {text}")

                content = (
                    "\n\n".join(snippet_parts)
                    or _get_field(card, "content", "")
                    or _get_field(card, "abstract", "")
                    or _get_field(card, "summary", "")
                )
                if content and len(content) > 50:
                    matched_content = content
                    break

        if matched_content:
            updated.append(cit.model_copy(update={"fetched_content": matched_content}))
        else:
            updated.append(cit)

    logger.debug(
        "[draft_node] injected content into %d/%d citations",
        sum(1 for c in updated if getattr(c, "fetched_content", None)),
        len(updated),
    )
    return updated


def _ensure_minimum_citation_coverage(
    draft: DraftReport,
    cards: list[Any],
    *,
    brief: Any | None = None,
    skill_artifacts: dict[str, Any] | None = None,
) -> DraftReport:
    """Backfill citations and rebalance section coverage when the model under-cites.

    The survey prompt asks for broad citation coverage, but the model can still
    collapse to a small reference list. This repair keeps existing citations
    intact, then appends additional paper-card citations up to a survey-sized
    floor and redistributes underused citations across major sections.
    """
    if not cards:
        return draft

    target = _target_citation_count(cards, brief)
    existing_citations = list(draft.citations)
    seen_keys = {
        key
        for citation in existing_citations
        if (key := _citation_identity_key(citation))
    }
    next_label = _next_citation_label(existing_citations)
    added: list[Citation] = []

    for card in cards:
        candidate = _citation_from_card(card, label=f"[{next_label}]")
        key = _citation_identity_key(candidate)
        if not key or key in seen_keys:
            continue
        added.append(candidate)
        seen_keys.add(key)
        next_label += 1
        if len(existing_citations) + len(added) >= target:
            break

    citation_pool = existing_citations + added
    sections, citation_pool = _ensure_section_citation_distribution(
        dict(draft.sections),
        citation_pool,
        cards,
        skill_artifacts=skill_artifacts,
    )
    return DraftReport(
        sections=sections,
        claims=list(draft.claims),
        citations=citation_pool,
    )


def _target_citation_count(cards: list[Any], brief: Any | None) -> int:
    desired = ""
    if isinstance(brief, dict):
        desired = str(brief.get("desired_output") or "").strip().lower()
    elif brief is not None:
        desired = str(getattr(brief, "desired_output", "") or "").strip().lower()

    if desired in {"paper_cards", "reading_notes"}:
        return min(len(cards), 6)
    survey_floor = max(12, round(len(cards) * 0.8))
    return min(len(cards), survey_floor)


def _citation_identity_key(citation: Citation) -> str:
    arxiv_id = _extract_arxiv_id_from_url(citation.url or "")
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    url = str(citation.url or "").strip().lower()
    if url:
        return f"url:{url}"
    reason = str(citation.reason or "").strip().lower()
    if reason:
        return f"reason:{reason}"
    return ""


def _next_citation_label(citations: list[Citation]) -> int:
    import re

    max_seen = 0
    for citation in citations:
        match = re.search(r"\[(\d+)\]", str(citation.label or ""))
        if match:
            max_seen = max(max_seen, int(match.group(1)))
    return max_seen + 1 if max_seen else 1


def _citation_from_card(card: Any, *, label: str) -> Citation:
    title = str(_get_field(card, "title", "") or "Representative paper").strip()
    arxiv_id = str(_get_field(card, "arxiv_id", "") or "").strip()
    url = (
        f"https://arxiv.org/abs/{arxiv_id}"
        if arxiv_id
        else str(_get_field(card, "url", "") or "").strip()
    )
    if not url:
        url = "https://example.com/unknown-paper"

    citation = Citation(
        label=label,
        url=url,
        reason=title,
    )
    return _inject_citation_content([citation], [card])[0]


def _append_additional_citation_mentions(
    sections: dict[str, str],
    citations: list[Citation],
) -> dict[str, str]:
    section_order = [
        "introduction",
        "taxonomy",
        "methods",
        "datasets",
        "evaluation",
        "discussion",
        "future_work",
        "conclusion",
    ]
    usable_sections = [
        key
        for key in section_order
        if str(sections.get(key, "") or "").strip()
    ]
    if not usable_sections:
        usable_sections = [
            key
            for key, value in sections.items()
            if key != "title" and str(value or "").strip()
        ]
    if not usable_sections:
        return sections

    updated = dict(sections)
    for idx, citation in enumerate(citations):
        section_key = usable_sections[idx % len(usable_sections)]
        mention = (
            f" Additional representative evidence includes "
            f"{citation.reason} {citation.label}."
        )
        if mention.strip() not in updated[section_key]:
            updated[section_key] = updated[section_key].rstrip() + mention
    return updated


def _extract_inline_citation_labels(text: str) -> list[str]:
    import re

    seen: set[str] = set()
    labels: list[str] = []
    for label in re.findall(r"\[\d+\]", str(text or "")):
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def _normalize_title(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _find_existing_citation_for_card(card: Any, citations: list[Citation]) -> Citation | None:
    card_arxiv_id = str(_get_field(card, "arxiv_id", "") or "").strip()
    card_url = str(_get_field(card, "url", "") or "").strip().lower()
    card_title = _normalize_title(_get_field(card, "title", "") or "")

    for citation in citations:
        cit_arxiv_id = _extract_arxiv_id_from_url(citation.url or "")
        if card_arxiv_id and cit_arxiv_id == card_arxiv_id:
            return citation
        if card_url and str(citation.url or "").strip().lower() == card_url:
            return citation
        if card_title and _normalize_title(citation.reason or "") == card_title:
            return citation
    return None


def _section_support_sentence(section_key: str, citations: list[Citation]) -> str:
    labels = [citation.label for citation in citations if citation.label]
    if not labels:
        return ""

    joined = ", ".join(labels)
    prefix_map = {
        "introduction": "Representative studies framing the scope here include",
        "background": "Background evidence in this section also includes",
        "taxonomy": "Representative category-defining papers here include",
        "methods": "Additional method-specific evidence in this section comes from",
        "datasets": "Dataset and benchmark evidence in this section is also supported by",
        "evaluation": "Comparative evaluation evidence in this section is further grounded in",
        "discussion": "Cross-paper trade-offs in this section are additionally informed by",
        "future_work": "These open problems are also motivated by evidence from",
    }
    prefix = prefix_map.get(section_key, "Representative evidence in this section includes")
    return f"{prefix} {joined}."


def _ensure_section_citation_distribution(
    sections: dict[str, str],
    citations: list[Citation],
    cards: list[Any],
    *,
    skill_artifacts: dict[str, Any] | None = None,
) -> tuple[dict[str, str], list[Citation]]:
    if not sections or not citations:
        return sections, citations

    updated_sections = dict(sections)
    updated_citations = list(citations)
    evidence_map = (
        (skill_artifacts or {}).get("section_evidence_map")
        if isinstance(skill_artifacts, dict)
        else {}
    ) or {}
    cards_by_title = {
        _normalize_title(_get_field(card, "title", "") or ""): card
        for card in cards
        if _normalize_title(_get_field(card, "title", "") or "")
    }
    usage = Counter()
    for section_key, content in updated_sections.items():
        if section_key == "title":
            continue
        usage.update(_extract_inline_citation_labels(content))

    next_label = _next_citation_label(updated_citations)

    for section_key, base_floor in SURVEY_SECTION_CITATION_FLOORS.items():
        body = str(updated_sections.get(section_key, "") or "").strip()
        if not body:
            continue

        current_labels = set(_extract_inline_citation_labels(body))
        floor = min(max(len(updated_citations), 1), base_floor)
        if len(current_labels) >= floor:
            continue

        selected: list[Citation] = []
        desired_titles = evidence_map.get(section_key, []) if isinstance(evidence_map, dict) else []
        if isinstance(desired_titles, list):
            for title in desired_titles:
                card = cards_by_title.get(_normalize_title(title))
                if not card:
                    continue
                candidate = _find_existing_citation_for_card(card, updated_citations)
                if candidate is None:
                    candidate = _citation_from_card(card, label=f"[{next_label}]")
                    updated_citations.append(candidate)
                    next_label += 1
                if candidate.label in current_labels or any(item.label == candidate.label for item in selected):
                    continue
                selected.append(candidate)
                if len(current_labels) + len(selected) >= floor:
                    break

        if len(current_labels) + len(selected) < floor:
            for candidate in sorted(
                updated_citations,
                key=lambda item: (usage.get(item.label, 0), item.label),
            ):
                if candidate.label in current_labels or any(item.label == candidate.label for item in selected):
                    continue
                selected.append(candidate)
                if len(current_labels) + len(selected) >= floor:
                    break

        support_sentence = _section_support_sentence(section_key, selected)
        if support_sentence:
            updated_sections[section_key] = body.rstrip() + "\n\n" + support_sentence
            for citation in selected:
                usage[citation.label] += 1

    return updated_sections, updated_citations


def _build_brief_context(brief: Any | None) -> str:
    if not brief:
        return ""
    topic = (brief.get("topic") if isinstance(brief, dict) else getattr(brief, "topic", "")) or ""
    sub_questions = (brief.get("sub_questions") if isinstance(brief, dict) else getattr(brief, "sub_questions", [])) or []
    time_range = (brief.get("time_range") if isinstance(brief, dict) else getattr(brief, "time_range", "")) or ""
    focus_dimensions = (brief.get("focus_dimensions") if isinstance(brief, dict) else getattr(brief, "focus_dimensions", [])) or []
    desired = (brief.get("desired_output") if isinstance(brief, dict) else getattr(brief, "desired_output", "")) or "survey"
    ctx = f"## Research Topic\n{topic}\n\n"
    if time_range:
        ctx += f"## Time Range\n{time_range}\n\n"
    if focus_dimensions:
        ctx += "## Focus Dimensions\n" + "\n".join(f"- {item}" for item in focus_dimensions[:8]) + "\n\n"
    if sub_questions:
        ctx += "## Sub-Questions\n" + "\n".join(f"- {q}" for q in sub_questions) + "\n\n"
    ctx += f"## Desired Output\n{desired}\n\n"
    return ctx


def _render_cards(cards: list[Any]) -> str:
    """将 PaperCards 渲染为供 LLM 消费的文本（包含完整摘要 + 所有结构化字段）。"""
    parts = []
    for i, card in enumerate(cards):
        title = _get_field(card, "title", "Untitled")
        authors = _get_field(card, "authors", [])
        if isinstance(authors, list):
            authors_str = ", ".join(authors[:5]) + ("..." if len(authors) > 5 else "")
        else:
            authors_str = str(authors)
        # 优先用完整 abstract，fallback 到 summary
        full_abstract = _get_field(card, "abstract", "") or _get_field(card, "summary", "")
        methods = _get_field(card, "methods", [])
        datasets = _get_field(card, "datasets", [])
        limitations = _get_field(card, "limitations", [])
        keywords = _get_field(card, "keywords", [])
        arxiv_id = _get_field(card, "arxiv_id", "")
        url = _get_field(card, "url", "")
        year = _get_field(card, "published_year", "")
        venue = _get_field(card, "venue", "")

        part = f"=== Paper {i+1} ===\n"
        part += f"Title: {title}\n"
        part += f"Authors: {authors_str or 'Unknown'}\n"
        if year:
            part += f"Year: {year}\n"
        if venue:
            part += f"Venue: {venue}\n"
        part += f"arXiv ID: {arxiv_id or 'N/A'}\n"
        part += f"URL: {url or 'N/A'}\n"
        if methods:
            part += f"Methods: {', '.join(methods[:8])}\n"
        if datasets:
            part += f"Datasets / Benchmarks: {', '.join(datasets[:8])}\n"
        if limitations:
            part += f"Limitations: {', '.join(limitations[:5])}\n"
        if keywords:
            part += f"Keywords: {', '.join(keywords[:10])}\n"
        # 传给 LLM 的摘要不要截断，让 LLM 自己决定消费方式
        part += f"Abstract (full):\n{full_abstract}\n"
        parts.append(part)
    return "\n\n".join(parts)


def _extract_json(text: str) -> str | None:
    """
    从 LLM 输出中提取 JSON（支持数组 [ ] 和对象 { }）。
    """
    import json, re
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    for start_char, end_char in [("[", "]"), ("{", "}")]:
        idx_start = text.find(start_char)
        idx_end = text.rfind(end_char)
        if idx_start != -1 and idx_end != -1 and idx_end > idx_start:
            try:
                json.loads(text[idx_start:idx_end + 1])
                return text[idx_start:idx_end + 1]
            except json.JSONDecodeError:
                pass

    if text and text[0] in '{"[':
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
    return None
