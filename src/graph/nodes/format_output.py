from __future__ import annotations

from src.models.report import FinalReport, GroundingStats


def _compute_grounding_stats(claims, citations) -> GroundingStats:
    total = len(claims)
    if total == 0:
        return GroundingStats(
            total_claims=0,
            grounded=0,
            partial=0,
            ungrounded=0,
            abstained=0,
            tier_a_ratio=0.0,
            tier_b_ratio=0.0,
        )

    grounded = sum(1 for c in claims if c.overall_status == "grounded")
    partial = sum(1 for c in claims if c.overall_status == "partial")
    ungrounded = sum(1 for c in claims if c.overall_status == "ungrounded")
    abstained = sum(1 for c in claims if c.overall_status == "abstained")

    total_cits = len(citations) or 1
    tier_a = sum(1 for c in citations if getattr(c, "source_tier", None) == "A")
    tier_b = sum(1 for c in citations if getattr(c, "source_tier", None) == "B")

    return GroundingStats(
        total_claims=total,
        grounded=grounded,
        partial=partial,
        ungrounded=ungrounded,
        abstained=abstained,
        tier_a_ratio=tier_a / total_cits,
        tier_b_ratio=tier_b / total_cits,
    )


def _error_final_report(errors: list[str]) -> FinalReport:
    """User-visible report when the pipeline stops before a draft exists."""
    body = "\n".join(f"- {e}" for e in errors) if errors else "未返回具体错误信息。"
    return FinalReport(
        sections={
            "无法生成报告": (
                "文献获取或元数据规范化失败。常见原因：arXiv ID 在官方库中不存在或尚未公开、"
                "网络无法访问 export.arxiv.org、或 PDF 无法解析出正文。"
            ),
            "系统信息": body,
        },
        claims=[],
        citations=[],
        grounding_stats=GroundingStats(
            total_claims=0,
            grounded=0,
            partial=0,
            ungrounded=0,
            abstained=0,
            tier_a_ratio=0.0,
            tier_b_ratio=0.0,
        ),
        report_confidence="low",
    )


def format_output(state: dict) -> dict:
    verified = state.get("verified_report")
    draft = state.get("draft_report")

    source = verified or draft
    if not source:
        errors = list(state.get("errors") or [])
        degradation = state.get("degradation_mode", "normal")
        if errors or degradation == "safe_abort":
            return {"final_report": _error_final_report(errors)}
        return {"errors": ["format_output: no report to format"]}

    degradation = state.get("degradation_mode", "normal")
    grounding = _compute_grounding_stats(source.claims, source.citations)

    if grounding.total_claims > 0:
        grounded_ratio = grounding.grounded / grounding.total_claims
        partial_ratio = grounding.partial / grounding.total_claims
        ungrounded_ratio = grounding.ungrounded / grounding.total_claims
        if grounded_ratio >= 0.8 and partial_ratio <= 0.2 and ungrounded_ratio == 0.0:
            grounding_confidence = "high"
        elif grounded_ratio >= 0.5 and ungrounded_ratio <= 0.2:
            grounding_confidence = "limited"
        else:
            grounding_confidence = "low"
    else:
        grounding_confidence = "high"

    _rank = {"high": 0, "limited": 1, "low": 2}
    if degradation == "safe_abort":
        degradation_confidence = "low"
    elif degradation == "limited":
        degradation_confidence = "limited"
    else:
        degradation_confidence = "high"

    confidence = grounding_confidence if _rank[grounding_confidence] > _rank[degradation_confidence] else degradation_confidence

    final = FinalReport(
        sections=dict(source.sections),
        claims=list(source.claims),
        citations=list(source.citations),
        grounding_stats=grounding,
        report_confidence=confidence,
    )
    mode = state.get("report_mode", "draft")
    result: dict = {"final_report": final}
    try:
        from src.agent.report import _final_report_to_markdown

        markdown = _final_report_to_markdown(final)
        if mode == "full":
            result["full_markdown"] = markdown
        else:
            result["draft_markdown"] = markdown
    except Exception:
        pass
    return result
