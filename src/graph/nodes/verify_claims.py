from __future__ import annotations

from src.models.report import Claim, ResolvedReport, VerifiedReport
from src.verification.claim_judge import judge_claim_citation


def verify_claims(state: dict) -> dict:
    resolved: ResolvedReport | None = state.get("resolved_report")
    if not resolved:
        return {"warnings": ["verify_claims: no resolved_report, skipping"]}

    cit_map = {c.label: c for c in resolved.citations}

    llm = None
    try:
        from src.agent.llm import build_reason_llm
        from src.agent.settings import Settings

        settings = Settings.from_env()
        llm = build_reason_llm(settings)
    except Exception:
        pass

    verified_claims: list[Claim] = []
    warnings: list[str] = []

    for claim in resolved.claims:
        supports = []
        for label in claim.citation_labels:
            cit = cit_map.get(label)
            if not cit:
                warnings.append(
                    f"verify_claims: claim {claim.id} references unknown citation {label}"
                )
                continue

            support = judge_claim_citation(
                claim_id=claim.id,
                claim_text=claim.text,
                citation_label=label,
                citation_content=cit.fetched_content,
                llm=llm,
            )
            supports.append(support)

        if not supports:
            overall = "abstained"
        else:
            statuses = [s.support_status for s in supports]
            if any(s == "supported" for s in statuses):
                overall = "grounded"
            elif any(s == "partial" for s in statuses):
                overall = "partial"
            elif all(s == "unverifiable" for s in statuses):
                overall = "abstained"
            else:
                overall = "ungrounded"

        verified_claims.append(
            claim.model_copy(update={"supports": supports, "overall_status": overall})
        )

    verified = VerifiedReport(
        sections=dict(resolved.sections),
        claims=verified_claims,
        citations=list(resolved.citations),
    )

    result: dict = {"verified_report": verified}
    if warnings:
        result["warnings"] = warnings
    return result
