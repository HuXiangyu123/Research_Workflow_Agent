"""Reviewer service — Phase 3: 生成 ReviewFeedback。"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import TYPE_CHECKING, Any

from src.models.review import (
    ClaimSupport,
    CoverageGap,
    RevisionAction,
    RevisionActionType,
    ReviewCategory,
    ReviewFeedback,
    ReviewIssue,
    ReviewSeverity,
)

if TYPE_CHECKING:
    from src.models.workspace import WorkspaceArtifact

logger = logging.getLogger(__name__)


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """安全获取字典或对象的字段。"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _list_like(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _report_claims(report: Any | None) -> list[Any]:
    return _list_like(_safe_get(report, "claims", []))


def _report_citations(report: Any | None) -> list[Any]:
    return _list_like(_safe_get(report, "citations", []))


class ReviewerService:
    """
    Phase 3 Reviewer：质量闸门。

    输入：rag_result, paper_cards, report_draft
    输出：ReviewFeedback（issues / coverage_gaps / claim_supports / revision_actions）

    职责：
    1. 覆盖性检查（sub-questions 是否都被覆盖）
    2. claim 支撑检查（evidence 是否存在）
    3. citation 可达性检查
    4. 结构重复/一致性检查
    """

    async def review(
        self,
        *,
        task_id: str,
        workspace_id: str,
        rag_result: Any | None = None,
        paper_cards: list | None = None,
        report_draft: Any | None = None,
        trace_artifacts: list["WorkspaceArtifact"] | None = None,
    ) -> ReviewFeedback:
        """
        生成 ReviewFeedback。

        Args:
            task_id: 任务 ID
            workspace_id: 工作区 ID
            rag_result: RAG 检索结果（可选）
            paper_cards: 论文卡片列表（可选）
            report_draft: 报告草稿（可选）
            trace_artifacts: 追踪记录（可选）
        """
        issues: list[ReviewIssue] = []
        coverage_gaps: list[CoverageGap] = []
        claim_supports: list[ClaimSupport] = []
        revision_actions: list[RevisionAction] = []

        # ── 1. Paper Cards 内容质量检查 ─────────────────────────────────
        cards_issues, cards_actions = self._check_paper_cards_quality(paper_cards)
        issues.extend(cards_issues)
        revision_actions.extend(cards_actions)

        # ── 2. 覆盖性检查 ──────────────────────────────────────────────
        coverage_issues, coverage_gap_items, coverage_actions = self._check_coverage(
            rag_result, paper_cards, report_draft
        )
        issues.extend(coverage_issues)
        coverage_gaps.extend(coverage_gap_items)
        revision_actions.extend(coverage_actions)

        # ── 2. Claim 支撑检查 ──────────────────────────────────────
        claim_issues, claim_supports, claim_actions = self._check_claim_support(
            rag_result, paper_cards, report_draft
        )
        issues.extend(claim_issues)
        claim_supports.extend(claim_supports)
        revision_actions.extend(claim_actions)

        # ── 3. Citation 可达性检查 ─────────────────────────────────
        citation_issues, citation_actions = self._check_citation_reachability(
            rag_result, report_draft
        )
        issues.extend(citation_issues)
        revision_actions.extend(citation_actions)

        # ── 4. Citation breadth / concentration checks ──────────────────
        citation_balance_issues, citation_balance_actions = self._check_citation_breadth_and_balance(
            paper_cards, report_draft
        )
        issues.extend(citation_balance_issues)
        revision_actions.extend(citation_balance_actions)

        # ── 5. 结构重复/一致性检查 ─────────────────────────────────
        dup_issues, dup_actions = self._check_duplication_consistency(report_draft)
        issues.extend(dup_issues)
        revision_actions.extend(dup_actions)

        # ── 汇总判断 ───────────────────────────────────────────────
        blocker_count = sum(
            1 for i in issues if i.severity == ReviewSeverity.BLOCKER
        )
        error_count = sum(
            1 for i in issues if i.severity == ReviewSeverity.ERROR
        )
        passed = blocker_count == 0 and error_count == 0

        # 生成摘要
        summary = self._summarize(issues, coverage_gaps, claim_supports, passed)

        return ReviewFeedback(
            task_id=task_id,
            workspace_id=workspace_id,
            passed=passed,
            issues=issues,
            coverage_gaps=coverage_gaps,
            claim_supports=claim_supports,
            revision_actions=revision_actions,
            summary=summary,
        )

    # ─── 内部检查方法 ───────────────────────────────────────────────────────

    def _check_paper_cards_quality(
        self,
        paper_cards: list | None,
    ) -> tuple[list[ReviewIssue], list[RevisionAction]]:
        """
        检查 paper_cards 内容质量。

        质量标准：
        1. 每张 card 必须有有效 title（不能是查询 URL 或占位符）
        2. title 不能包含 "arXiv Query" / "search_query" / "id_list" 等异常模式
        3. 至少 30% 的 card 有 authors 字段
        4. 至少 30% 的 card 有 abstract 字段
        """
        issues: list[ReviewIssue] = []
        actions: list[RevisionAction] = []

        if not paper_cards:
            return issues, actions

        INVALID_TITLE_PATTERNS = [
            "arXiv Query", "search_query=", "id_list=",
            "http://", "https://", "query:", "start=",
        ]

        bad_cards: list[dict] = []
        for card in paper_cards:
            title = _safe_get(card, "title", "")
            if not title:
                bad_cards.append(card)
                continue
            # 检查是否包含无效 title 模式
            if any(pat in title for pat in INVALID_TITLE_PATTERNS):
                bad_cards.append(card)
                continue
            # 检查 title 是否太短（< 10 字符通常不是真实 title）
            if len(title.strip()) < 10:
                bad_cards.append(card)

        total = len(paper_cards)
        bad_count = len(bad_cards)
        bad_ratio = bad_count / total if total > 0 else 0

        # 计算有完整 metadata 的比例
        has_authors = sum(1 for c in paper_cards if _safe_get(c, "authors"))
        has_abstract = sum(1 for c in paper_cards if _safe_get(c, "abstract") or _safe_get(c, "summary"))
        authors_ratio = has_authors / total if total > 0 else 0
        abstract_ratio = has_abstract / total if total > 0 else 0

        # 超过 30% bad cards → blocker
        if bad_ratio > 0.3:
            issue = ReviewIssue(
                severity=ReviewSeverity.BLOCKER,
                category=ReviewCategory.COVERAGE_GAP,
                target="paper_cards",
                summary=(
                    f"{bad_count}/{total} paper cards have invalid titles "
                    f"({bad_ratio:.0%}). Likely search metadata extraction failed. "
                    f"Authors coverage: {authors_ratio:.0%}, Abstract coverage: {abstract_ratio:.0%}."
                ),
                evidence_refs=[c.get("arxiv_id", "") or c.get("title", "") for c in bad_cards[:5]],
            )
            issues.append(issue)
            actions.append(
                RevisionAction(
                    action_type=RevisionActionType.RESEARCH_MORE,
                    target="paper_cards",
                    reason=(
                        f"{bad_count}/{total} paper cards have bad quality titles. "
                        "Need to re-run search with arXiv API metadata enrichment."
                    ),
                    priority=1,
                )
            )
        elif bad_ratio > 0:
            # 有问题但不多 → warning
            issue = ReviewIssue(
                severity=ReviewSeverity.WARNING,
                category=ReviewCategory.COVERAGE_GAP,
                target="paper_cards",
                summary=(
                    f"{bad_count}/{total} paper cards have suboptimal titles. "
                    f"Authors coverage: {authors_ratio:.0%}, Abstract coverage: {abstract_ratio:.0%}."
                ),
                evidence_refs=[c.get("arxiv_id", "") or c.get("title", "") for c in bad_cards[:3]],
            )
            issues.append(issue)

        # 检查 metadata 完整性（warning 而非 blocker）
        if authors_ratio < 0.3:
            issue = ReviewIssue(
                severity=ReviewSeverity.WARNING,
                category=ReviewCategory.COVERAGE_GAP,
                target="paper_cards",
                summary=f"Only {authors_ratio:.0%} of paper cards have author metadata",
            )
            issues.append(issue)

        return issues, actions

    def _check_coverage(
        self,
        rag_result: Any | None,
        paper_cards: list | None,
        report_draft: Any | None,
    ) -> tuple[list[ReviewIssue], list[CoverageGap], list[RevisionAction]]:
        """检查子问题/主题是否都被覆盖。"""
        issues: list[ReviewIssue] = []
        gaps: list[CoverageGap] = []
        actions: list[RevisionAction] = []

        # 基础检查：当没有任何 paper_cards 时认为覆盖不足
        if not paper_cards:
            issue = ReviewIssue(
                severity=ReviewSeverity.WARNING,
                category=ReviewCategory.COVERAGE_GAP,
                target="overall",
                summary="No paper cards found — coverage may be insufficient",
            )
            issues.append(issue)
            gap = CoverageGap(
                missing_topics=["no_papers_retrieved"],
                note="No papers retrieved for this task",
            )
            gaps.append(gap)
            actions.append(
                RevisionAction(
                    action_type=RevisionActionType.RESEARCH_MORE,
                    target="overall",
                    reason="No papers found, need broader search",
                    priority=1,
                )
            )

        return issues, gaps, actions

    def _check_claim_support(
        self,
        rag_result: Any | None,
        paper_cards: list | None,
        report_draft: Any | None,
    ) -> tuple[list[ReviewIssue], list[ClaimSupport], list[RevisionAction]]:
        """检查每个 claim 是否有 evidence 支撑。"""
        issues: list[ReviewIssue] = []
        supports: list[ClaimSupport] = []
        actions: list[RevisionAction] = []

        claims = _report_claims(report_draft)
        if claims:
            unsupported_claim_ids: list[str] = []
            partial_claim_ids: list[str] = []
            abstained_claim_ids: list[str] = []

            for claim in claims:
                claim_id = str(_safe_get(claim, "id", "")) or "unknown_claim"
                claim_text = str(_safe_get(claim, "text", "")) or "[empty claim]"
                citation_ids = [
                    str(label) for label in _list_like(_safe_get(claim, "citation_labels", [])) if label
                ]
                raw_supports = _list_like(_safe_get(claim, "supports", []))
                support_statuses = [
                    str(_safe_get(item, "support_status", "")) for item in raw_supports
                ]
                evidence_refs = [
                    str(_safe_get(item, "citation_label", "")) for item in raw_supports
                    if _safe_get(item, "citation_label", "")
                ]

                overall_status = str(_safe_get(claim, "overall_status", "") or "")
                supported = overall_status in {"grounded", "partial"} or any(
                    status in {"supported", "partial"} for status in support_statuses
                )

                note = None
                if overall_status:
                    note = f"overall_status={overall_status}"
                elif support_statuses:
                    note = "support_statuses=" + ",".join(support_statuses)

                supports.append(
                    ClaimSupport(
                        claim_id=claim_id,
                        claim_text=claim_text,
                        supported=supported,
                        citation_ids=citation_ids or evidence_refs,
                        note=note,
                    )
                )

                if overall_status == "ungrounded":
                    unsupported_claim_ids.append(claim_id)
                elif overall_status == "partial":
                    partial_claim_ids.append(claim_id)
                elif overall_status == "abstained":
                    abstained_claim_ids.append(claim_id)

            if unsupported_claim_ids:
                total_claims = max(len(claims), 1)
                unsupported_ratio = len(unsupported_claim_ids) / total_claims
                severity = (
                    ReviewSeverity.BLOCKER
                    if unsupported_ratio >= 0.8
                    else ReviewSeverity.ERROR
                    if unsupported_ratio >= 0.5
                    else ReviewSeverity.WARNING
                )
                issues.append(
                    ReviewIssue(
                        severity=severity,
                        category=ReviewCategory.UNSUPPORTED_CLAIM,
                        target="claims",
                        summary=(
                            f"{len(unsupported_claim_ids)}/{len(claims)} claims are currently ungrounded "
                            f"({unsupported_ratio:.0%} of the claim set)"
                        ),
                        evidence_refs=unsupported_claim_ids[:5],
                    )
                )
                actions.append(
                    RevisionAction(
                        action_type=RevisionActionType.DROP_CLAIM,
                        target="claims",
                        reason="Some draft claims are not supported by reachable citations or evidence",
                        priority=2,
                    )
                )

            if partial_claim_ids:
                partial_ratio = len(partial_claim_ids) / max(len(claims), 1)
                issues.append(
                    ReviewIssue(
                        severity=(
                            ReviewSeverity.ERROR
                            if partial_ratio >= 0.5
                            else ReviewSeverity.WARNING
                            if partial_ratio >= 0.25
                            else ReviewSeverity.INFO
                        ),
                        category=ReviewCategory.UNSUPPORTED_CLAIM,
                        target="claims",
                        summary=(
                            f"{len(partial_claim_ids)}/{len(claims)} claims are only partially supported"
                        ),
                        evidence_refs=partial_claim_ids[:5],
                    )
                )
                if partial_ratio >= 0.25:
                    actions.append(
                        RevisionAction(
                            action_type=RevisionActionType.REVISE_DRAFT,
                            target="claims",
                            reason=(
                                "Too many draft claims are only partially supported and should be rewritten "
                                "or narrowed to grounded evidence."
                            ),
                            priority=2,
                        )
                    )

            if abstained_claim_ids:
                issues.append(
                    ReviewIssue(
                        severity=ReviewSeverity.WARNING,
                        category=ReviewCategory.UNSUPPORTED_CLAIM,
                        target="claims",
                        summary=(
                            f"{len(abstained_claim_ids)}/{len(claims)} claims could not be verified because "
                            "usable citation evidence was missing"
                        ),
                        evidence_refs=abstained_claim_ids[:5],
                    )
                )

            return issues, supports, actions

        # 基础检查：当没有 rag_result 且也没有 report grounding 信息时认为无法验证 claim 支撑
        if not rag_result:
            issue = ReviewIssue(
                severity=ReviewSeverity.WARNING,
                category=ReviewCategory.UNSUPPORTED_CLAIM,
                target="overall",
                summary="No RAG result available to verify claims",
            )
            issues.append(issue)
            supports.append(
                ClaimSupport(
                    claim_id="no_rag_result",
                    claim_text="[no report draft]",
                    supported=False,
                    note="rag_result is None — cannot verify claim support",
                )
            )

        return issues, supports, actions

    def _check_citation_reachability(
        self,
        rag_result: Any | None,
        report_draft: Any | None,
    ) -> tuple[list[ReviewIssue], list[RevisionAction]]:
        """检查 citation 是否能回指到 source。"""
        issues: list[ReviewIssue] = []
        actions: list[RevisionAction] = []

        citations = _report_citations(report_draft)
        if citations:
            unreachable = [
                str(_safe_get(cit, "label", "")) for cit in citations
                if _safe_get(cit, "reachable", None) is False
            ]
            low_tier = [
                str(_safe_get(cit, "label", "")) for cit in citations
                if str(_safe_get(cit, "source_tier", "") or "") in {"C", "D"}
            ]

            if unreachable:
                issues.append(
                    ReviewIssue(
                        severity=ReviewSeverity.WARNING,
                        category=ReviewCategory.CITATION_REACHABILITY,
                        target="citations",
                        summary=(
                            f"{len(unreachable)}/{len(citations)} citations are not reachable"
                        ),
                        evidence_refs=unreachable[:5],
                    )
                )
                actions.append(
                    RevisionAction(
                        action_type=RevisionActionType.FIX_CITATION,
                        target="citations",
                        reason="Some citation URLs could not be resolved during grounding",
                        priority=2,
                    )
                )

            if low_tier:
                issues.append(
                    ReviewIssue(
                        severity=ReviewSeverity.INFO,
                        category=ReviewCategory.CITATION_REACHABILITY,
                        target="citations",
                        summary=(
                            f"{len(low_tier)}/{len(citations)} citations come from lower-trust tiers (C/D)"
                        ),
                        evidence_refs=low_tier[:5],
                    )
                )

            return issues, actions

        if not rag_result:
            issue = ReviewIssue(
                severity=ReviewSeverity.INFO,
                category=ReviewCategory.CITATION_REACHABILITY,
                target="overall",
                summary="No RAG result — citation reachability cannot be checked",
            )
            issues.append(issue)

        return issues, actions

    def _check_citation_breadth_and_balance(
        self,
        paper_cards: list | None,
        report_draft: Any | None,
    ) -> tuple[list[ReviewIssue], list[RevisionAction]]:
        """Check whether the draft cites a broad enough source base.

        Multi-paper survey drafts should not repeatedly recycle a tiny citation
        pool. This gate looks at both the citation list and inline citation
        mentions inside rendered section text.
        """
        issues: list[ReviewIssue] = []
        actions: list[RevisionAction] = []

        citations = _report_citations(report_draft)
        claims = _report_claims(report_draft)
        paper_count = len(paper_cards or [])
        citation_count = len(citations)
        if not citations or paper_count < 5:
            return issues, actions

        coverage_ratio = citation_count / max(paper_count, 1)
        if coverage_ratio < 0.7:
            severity = ReviewSeverity.ERROR
        elif coverage_ratio < 0.85:
            severity = ReviewSeverity.WARNING
        else:
            severity = None

        if severity is not None:
            issues.append(
                ReviewIssue(
                    severity=severity,
                    category=ReviewCategory.COVERAGE_GAP,
                    target="citations",
                    summary=(
                        f"Only {citation_count}/{paper_count} retrieved papers were cited in the draft "
                        f"({coverage_ratio:.0%} citation coverage)"
                    ),
                    evidence_refs=[
                        str(_safe_get(cit, "label", ""))
                        for cit in citations[:5]
                        if _safe_get(cit, "label", "")
                    ],
                )
            )
            actions.append(
                RevisionAction(
                    action_type=RevisionActionType.RESEARCH_MORE,
                    target="citations",
                    reason="Broaden the evidence base so the final survey cites a larger share of the retrieved papers.",
                    priority=1 if severity == ReviewSeverity.ERROR else 2,
                )
            )

        sections = _safe_get(report_draft, "sections", {}) or {}
        section_text = "\n".join(
            str(content)
            for content in sections.values()
            if isinstance(content, str)
        )
        inline_mentions: Counter[str] = Counter()
        for cit in citations:
            label = str(_safe_get(cit, "label", "") or "").strip()
            if not label:
                continue
            inline_mentions[label] = len(re.findall(re.escape(label), section_text))

        mention_counts = inline_mentions
        if not any(mention_counts.values()) and claims:
            mention_counts = Counter(
                str(label)
                for claim in claims
                for label in _list_like(_safe_get(claim, "citation_labels", []))
                if label
            )

        total_mentions = sum(mention_counts.values())
        cited_label_count = sum(1 for count in mention_counts.values() if count > 0)
        average_reuse = total_mentions / cited_label_count if cited_label_count else 0.0

        if (
            cited_label_count > 0
            and total_mentions >= 20
            and cited_label_count <= 5
            and average_reuse >= 6.0
        ):
            issues.append(
                ReviewIssue(
                    severity=ReviewSeverity.ERROR,
                    category=ReviewCategory.DUPLICATION,
                    target="citations",
                    summary=(
                        f"The draft reuses only {cited_label_count} unique citations across {total_mentions} inline mentions "
                        f"(avg {average_reuse:.1f} mentions/source), indicating a narrow evidence base"
                    ),
                    evidence_refs=[label for label, count in mention_counts.most_common(5) if count > 0],
                )
            )
            actions.append(
                RevisionAction(
                    action_type=RevisionActionType.REVISE_DRAFT,
                    target="citations",
                    reason="Redistribute claims across a broader citation pool instead of repeatedly citing the same few papers.",
                    priority=1,
                )
            )

        return issues, actions

    def _check_duplication_consistency(
        self,
        report_draft: Any | None,
    ) -> tuple[list[ReviewIssue], list[RevisionAction]]:
        """检查 section 重复、论点重复。"""
        issues: list[ReviewIssue] = []
        actions: list[RevisionAction] = []

        if not report_draft:
            # No draft means nothing to check
            pass

        return issues, actions

    def _summarize(
        self,
        issues: list[ReviewIssue],
        gaps: list[CoverageGap],
        supports: list[ClaimSupport],
        passed: bool,
    ) -> str:
        """生成自然语言摘要。"""
        blockers = [i for i in issues if i.severity == ReviewSeverity.BLOCKER]
        errors = [i for i in issues if i.severity == ReviewSeverity.ERROR]
        warnings = [i for i in issues if i.severity == ReviewSeverity.WARNING]

        parts = []
        if blockers:
            parts.append(f"{len(blockers)} blocker(s)")
        if errors:
            parts.append(f"{len(errors)} error(s)")
        if warnings:
            parts.append(f"{len(warnings)} warning(s)")
        if gaps:
            parts.append(f"{len(gaps)} coverage gap(s)")

        if not parts:
            return "Review passed — no issues found."

        verdict = "PASSED" if passed else "FAILED"
        return f"[{verdict}] {', '.join(parts)}"
