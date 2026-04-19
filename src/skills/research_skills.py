"""Research Skills — 5 个 ARIS 风格科研 skill 函数实现。

每个函数对应一个 SKILL.md，backed by LOCAL_FUNCTION handler。
严格遵循 OpenCode 风格：
- 输入：dict（来自 agent 的 structured inputs）
- 输出：dict（带 summary + output + artifacts）
- 错误：返回带 error 字段的 dict
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ─── Skill 1: lit_review_scanner ───────────────────────────────────────────────


def lit_review_scanner(inputs: dict[str, Any], context: dict) -> dict:
    """
    Multi-source academic literature scan and candidate ranking.

    inputs:
        topic: str — research topic
        sub_questions: list[str] — sub-questions to cover
        max_results: int (default: 30)
        year_filter: str (optional)

    returns:
        summary: str
        paper_candidates: list[dict]
        search_queries: list[str]
    """
    topic = inputs.get("topic", "")
    sub_questions = inputs.get("sub_questions", [])
    max_results = inputs.get("max_results", 30)
    year_filter = inputs.get("year_filter", "")

    if not topic:
        return {"error": "lit_review_scanner: topic is required"}

    try:
        from src.tools.search_tools import _searxng_search

        all_hits: list[dict] = []
        queries = [topic]
        if sub_questions:
            queries.extend(sub_questions[:5])

        for query in queries:
            result = _searxng_search(
                query,
                engines="arxiv",
                max_results=max(5, max_results // len(queries)),
            )
            if result.get("ok"):
                all_hits.extend(result.get("hits", []))

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_hits: list[dict] = []
        for hit in all_hits:
            url = hit.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_hits.append(hit)

        candidates = []
        for i, hit in enumerate(unique_hits[:max_results], 1):
            candidates.append({
                "rank": i,
                "title": hit.get("title", ""),
                "url": hit.get("url", ""),
                "abstract": hit.get("content", "")[:500],
                "engine": hit.get("engine", "arxiv"),
                "published_date": hit.get("publishedDate"),
            })

        summary = (
            f"Scanned {len(queries)} queries and kept {len(candidates)} "
            f"deduplicated candidate papers."
        )
        return {
            "summary": summary,
            "paper_candidates": candidates,
            "search_queries": queries,
            "total_found": len(unique_hits),
            "dedup_strategy": "url_exact",
        }

    except Exception as exc:
        logger.exception("lit_review_scanner failed: %s", exc)
        return {"error": f"lit_review_scanner failed: {exc}"}


# ─── Skill 2: claim_verification ────────────────────────────────────────────────


def claim_verification(inputs: dict[str, Any], context: dict) -> dict:
    """
    Verify scientific claims against retrieved evidence.

    inputs:
        draft_report: dict — must contain claims and citations
        evidence_sources: list[dict] — optional external evidence
        claim_ids: list[str] (optional) — verify specific claims only

    returns:
        summary: str
        verified_claims: list[dict]
        grounding_stats: dict
    """
    draft_report = inputs.get("draft_report")
    if not draft_report:
        return {"error": "claim_verification: draft_report is required"}

    claims = draft_report.get("claims", [])
    citations = draft_report.get("citations", [])

    if not claims:
        return {
            "summary": "No claims to verify",
            "verified_claims": [],
            "grounding_stats": {"total": 0, "grounded": 0, "partial": 0, "ungrounded": 0},
        }

    cit_map: dict[str, dict] = {
        c.get("label", ""): c for c in citations
    }

    try:
        verified_claims: list[dict[str, Any]] = []
        grounded_count = 0
        partial_count = 0
        ungrounded_count = 0
        abstained_count = 0

        for claim in claims:
            claim_id = str(claim.get("id", "") or "")
            claim_text = str(claim.get("text", "") or "")
            citation_labels = _clean_list(claim.get("citation_labels", []))
            supports = claim.get("supports", []) if isinstance(claim, dict) else []
            overall_status = str(claim.get("overall_status", "") or "").strip().lower()

            if overall_status not in {"grounded", "partial", "ungrounded", "abstained"}:
                support_statuses = {
                    str(item.get("support_status", "") or "").strip().lower()
                    for item in supports
                    if isinstance(item, dict)
                }
                has_reachable_citation = any(
                    (cit_map.get(label, {}) or {}).get("fetched_content")
                    or (cit_map.get(label, {}) or {}).get("reachable") is True
                    for label in citation_labels
                )
                if "supported" in support_statuses:
                    overall_status = "grounded"
                elif "partial" in support_statuses:
                    overall_status = "partial"
                elif has_reachable_citation and citation_labels:
                    overall_status = "partial"
                elif citation_labels:
                    overall_status = "ungrounded"
                else:
                    overall_status = "abstained"

            if overall_status == "grounded":
                grounded_count += 1
            elif overall_status == "partial":
                partial_count += 1
            elif overall_status == "abstained":
                abstained_count += 1
            else:
                ungrounded_count += 1

            reason = {
                "grounded": "Existing grounding evidence supports the claim.",
                "partial": "The claim has incomplete but non-zero evidence support.",
                "ungrounded": "The cited evidence does not sufficiently support the claim.",
                "abstained": "No usable citation evidence was available for verification.",
            }[overall_status]
            verified_claims.append(
                {
                    "claim_id": claim_id,
                    "text": claim_text,
                    "status": overall_status,
                    "reason": reason,
                    "confidence": 0.9 if overall_status == "grounded" else 0.7 if overall_status == "partial" else 0.2,
                    "cited_labels": citation_labels,
                }
            )

        total = len(verified_claims)
        supported_total = grounded_count + partial_count
        return {
            "summary": (
                f"Verified {total} claims: {grounded_count} grounded, "
                f"{partial_count} partial, {ungrounded_count} ungrounded, "
                f"{abstained_count} abstained."
            ),
            "verified_claims": verified_claims,
            "grounding_stats": {
                "total": total,
                "grounded": grounded_count,
                "partial": partial_count,
                "ungrounded": ungrounded_count,
                "abstained": abstained_count,
                "grounded_ratio": round((grounded_count / total), 3) if total else 0.0,
                "supported_ratio": round((supported_total / total), 3) if total else 0.0,
            },
        }

    except Exception as exc:
        logger.exception("claim_verification failed: %s", exc)
        return {"error": f"claim_verification failed: {exc}"}


# ─── Skill 3: comparison_matrix_builder ───────────────────────────────────────


def comparison_matrix_builder(inputs: dict[str, Any], context: dict) -> dict:
    """
    Build a structured comparison matrix from paper cards.

    inputs:
        paper_cards: list[dict]
        compare_dimensions: list[str] (default: methods, datasets, benchmarks, limitations)
        format: "table" | "json" (default: "table")

    returns:
        summary: str
        matrix: list[dict]  — one row per paper, columns = dimensions
        missing_fields: list[str]  — papers with missing info
    """
    paper_cards = inputs.get("paper_cards", [])
    dimensions = inputs.get("compare_dimensions", [
        "methods", "datasets", "benchmarks", "limitations"
    ])
    output_format = inputs.get("format", "table")

    if not paper_cards:
        return {"error": "comparison_matrix_builder: paper_cards is required"}

    try:
        rows: list[dict[str, Any]] = []
        missing: list[str] = []

        for i, card in enumerate(paper_cards, 1):
            title = _first_text(card, "title", "paper_title") or f"Paper {i}"
            abstract = _first_text(card, "summary", "abstract", "content")
            methods = _stringify_field_list(card.get("methods"))
            datasets = _stringify_field_list(card.get("datasets"))
            limitations = _stringify_field_list(card.get("limitations"))
            metrics = _stringify_field_list(card.get("metrics"))
            benchmarks = metrics or _infer_benchmark_terms(abstract)
            evidence_source = "fulltext" if card.get("fulltext_available") or card.get("fulltext_snippets") else "abstract"

            row = {
                "paper": title,
                "methods": "; ".join(methods) or "Not specified in extracted evidence",
                "datasets": "; ".join(datasets) or "Not specified in extracted evidence",
                "benchmarks": "; ".join(benchmarks) or "Not specified in extracted evidence",
                "limitations": "; ".join(limitations) or "Not explicitly stated in extracted evidence",
                "evidence_source": evidence_source,
                "year": card.get("published_year") or card.get("year"),
                "arxiv_id": card.get("arxiv_id"),
            }
            rows.append(row)

            missing_dims = [
                dim
                for dim in ("methods", "datasets", "benchmarks", "limitations")
                if str(row.get(dim, "")).startswith("Not ")
            ]
            if missing_dims:
                missing.append(f"{title}: missing {', '.join(missing_dims)}")

        header = ["Paper"] + [dim.title().replace("_", " ") for dim in dimensions]
        table_lines = [" | ".join(header), " | ".join(["---"] * len(header))]
        for row in rows:
            cells = [row.get("paper", "")]
            for dim in dimensions:
                cells.append(str(row.get(dim, "")))
            table_lines.append(" | ".join(cells))

        matrix_payload = {
            "header": header,
            "rows": rows,
            "table_text": "\n".join(table_lines),
        }

        if output_format == "json":
            matrix_payload = {"rows": rows}

        return {
            "summary": (
                f"Built a comparison matrix for {len(rows)} papers across "
                f"{len(dimensions)} dimensions."
            ),
            "matrix": matrix_payload,
            "missing_fields": missing,
            "dimensions": dimensions,
        }

    except Exception as exc:
        logger.exception("comparison_matrix_builder failed: %s", exc)
        return {"error": f"comparison_matrix_builder failed: {exc}"}


# ─── Skill 4: experiment_replicator ─────────────────────────────────────────────


def experiment_replicator(inputs: dict[str, Any], context: dict) -> dict:
    """
    Analyze experimental settings and results from academic papers.

    inputs:
        paper_cards: list[dict]
        focus_papers: list[str] (optional) — arXiv IDs or titles to prioritize

    returns:
        summary: str
        experiments: list[dict]
        reproducibility_scores: dict
    """
    paper_cards = inputs.get("paper_cards", [])
    focus_papers = inputs.get("focus_papers", [])

    if not paper_cards:
        return {"error": "experiment_replicator: paper_cards is required"}

    try:
        from src.agent.llm import build_reason_llm
        from src.agent.settings import get_settings
        from langchain_core.messages import HumanMessage, SystemMessage

        settings = get_settings()
        llm = build_reason_llm(settings, max_tokens=16384)

        papers_text = []
        for i, card in enumerate(paper_cards, 1):
            title = card.get("title", card.get("paper_title", f"Paper {i}"))
            abstract = card.get("summary", card.get("abstract", ""))
            papers_text.append(f"Paper {i}: {title}\nAbstract: {abstract[:500]}")

        SYSTEM = """You are a reproducibility analyst. Extract experimental settings from academic paper abstracts.

Output strictly JSON:
{"experiments": [
  {
    "paper": "title",
    "datasets": ["dataset1", "dataset2"],
    "metrics": ["metric1", "metric2"],
    "baselines": ["baseline1", "baseline2"],
    "hyperparameters": {"key": "value"},
    "reproducibility_score": 0.0-1.0,
    "missing_info": ["what is missing for full reproducibility"]
  }
]}"""

        user_prompt = (
            "Analyze these papers for experimental reproducibility:\n\n"
            + "\n\n".join(papers_text)
        )

        resp = llm.invoke([
            SystemMessage(content=SYSTEM),
            HumanMessage(content=user_prompt),
        ])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        data = _extract_json(raw) or {}

        experiments = data.get("experiments", [])
        scores = {
            e.get("paper", ""): e.get("reproducibility_score", 0.0)
            for e in experiments
        }
        avg_score = round(sum(scores.values()) / len(scores), 3) if scores else 0.0

        return {
            "summary": (
                f"分析了 {len(experiments)} 篇论文的实验设置，"
                f"平均可复现性评分：{avg_score}"
            ),
            "experiments": experiments,
            "reproducibility_scores": scores,
            "average_score": avg_score,
        }

    except Exception as exc:
        logger.exception("experiment_replicator failed: %s", exc)
        return {"error": f"experiment_replicator failed: {exc}"}


# ─── Skill 5: writing_scaffold_generator ───────────────────────────────────────


def writing_scaffold_generator(inputs: dict[str, Any], context: dict) -> dict:
    """
    Generate structured writing scaffold for academic survey papers.

    inputs:
        topic: str
        paper_cards: list[dict]
        comparison_matrix: dict (optional)
        desired_length: "short" | "medium" | "long" (default: "medium")

    returns:
        summary: str
        scaffold: dict
        outline: list[str]
    """
    topic = inputs.get("topic", "")
    paper_cards = inputs.get("paper_cards", [])
    comparison_matrix = inputs.get("comparison_matrix", {})
    desired_length = inputs.get("desired_length", "medium")

    if not topic:
        return {"error": "writing_scaffold_generator: topic is required"}

    try:
        rows = list(comparison_matrix.get("rows", [])) if isinstance(comparison_matrix, dict) else []
        if not rows:
            matrix_result = comparison_matrix_builder(
                {
                    "paper_cards": paper_cards,
                    "compare_dimensions": ["methods", "datasets", "benchmarks", "limitations"],
                },
                context,
            )
            matrix_payload = matrix_result.get("matrix", {})
            if isinstance(matrix_payload, dict):
                rows = list(matrix_payload.get("rows", []))

        def _select_titles(*fields: str, limit: int) -> list[str]:
            selected: list[str] = []
            seen: set[str] = set()
            for row in rows:
                if not isinstance(row, dict):
                    continue
                paper = str(row.get("paper", "") or "").strip()
                if not paper or paper in seen:
                    continue
                if fields:
                    has_signal = False
                    for field in fields:
                        value = str(row.get(field, "") or "").strip()
                        if value and not value.startswith("Not "):
                            has_signal = True
                            break
                    if not has_signal:
                        continue
                selected.append(paper)
                seen.add(paper)
                if len(selected) >= limit:
                    break
            return selected

        method_families = _top_phrases([str(row.get("methods", "")) for row in rows], limit=4)
        dataset_terms = _top_phrases([str(row.get("datasets", "")) for row in rows], limit=4)
        limitation_terms = _top_phrases([str(row.get("limitations", "")) for row in rows], limit=4)
        evidence_titles = [str(row.get("paper", "")).strip() for row in rows if row.get("paper")]
        broad_titles = _select_titles(limit=10) or evidence_titles[:10]
        method_titles = _select_titles("methods", limit=10) or broad_titles[:10]
        dataset_titles = _select_titles("datasets", limit=8) or broad_titles[:8]
        benchmark_titles = _select_titles("benchmarks", limit=8) or dataset_titles[:8] or broad_titles[:8]
        limitation_titles = _select_titles("limitations", limit=8) or broad_titles[:8]

        length_targets = {
            "short": "Keep each major section concise and evidence-focused.",
            "medium": "Develop each major section with enough depth for cross-paper synthesis.",
            "long": "Expand each section with deeper comparisons, evidence gaps, and evaluation detail.",
        }

        scaffold = {
            "title": _build_survey_title(topic, evidence_titles),
            "abstract": [
                f"State the review scope and problem setting for {topic}.",
                "Name the corpus boundary, organizing logic, and central synthesis claim.",
                "Close with the main evidence gaps and practical implications.",
            ],
            "introduction": [
                "Define the topic, scope, and inclusion boundary.",
                "Explain why the review matters and preview the organizing logic.",
                "Avoid opening with a paper-by-paper inventory.",
            ],
            "background": [
                "Separate enabling background from fully in-scope agent systems.",
                "Use only the background needed to support the review question.",
            ],
            "taxonomy": [
                f"Organize the literature into method families such as {', '.join(method_families) or 'agent architecture, multimodal grounding, and evaluation design'}.",
                "Name representative papers in each category and justify the grouping rule.",
            ],
            "methods": [
                "Compare architecture, tool use, modality fusion, and orchestration choices across papers.",
                "Prefer cross-paper trade-offs over per-paper summaries.",
            ],
            "datasets": [
                f"Summarize datasets, settings, and metrics, including {', '.join(dataset_terms) or 'benchmark coverage and missing evaluation detail'}.",
                "Explicitly flag where dataset information is missing in the evidence.",
            ],
            "evaluation": [
                "Contrast metrics, validation settings, and evidence quality.",
                "Distinguish component-level gains from clinically meaningful outcomes.",
            ],
            "discussion": [
                "Synthesize agreements, disagreements, trade-offs, and evidence gaps.",
                "Explain what the current literature does and does not establish.",
            ],
            "future_work": [
                "Derive future directions from missing evaluations, weak grounding, and unresolved trade-offs.",
                f"Use recurring limitation clusters such as {', '.join(limitation_terms) or 'grounding, robustness, and deployment evidence'} as gap signals instead of rewriting limitation sentences.",
            ],
            "conclusion": [
                "Close with the main takeaways and what the field still needs before reliable deployment.",
            ],
        }

        outline = [
            "1. Introduction",
            "2. Background and Scope",
            "3. Taxonomy of In-Scope Systems",
            "4. Method Families and Tool Use",
            "5. Datasets, Metrics, and Evaluation Design",
            "6. Cross-Paper Discussion",
            "7. Open Challenges and Future Directions",
            "8. Conclusion",
        ]

        section_evidence_map = {
            "introduction": broad_titles[:6],
            "background": broad_titles[:4],
            "taxonomy": method_titles[:8],
            "methods": method_titles[:10],
            "datasets": dataset_titles[:8],
            "evaluation": benchmark_titles[:8],
            "discussion": limitation_titles[:8],
            "future_work": limitation_titles[:6] or broad_titles[:6],
        }

        return {
            "summary": (
                f"Generated a survey writing scaffold with {len(outline)} sections "
                f"for a corpus of {len(paper_cards)} papers."
            ),
            "scaffold": scaffold,
            "outline": outline,
            "section_evidence_map": section_evidence_map,
            "writing_guidance": "\n".join(
                [
                    "Organize the review by themes and method families, not by paper order.",
                    "Keep the discussion comparative: agreements, disagreements, trade-offs, and evidence gaps.",
                    "Derive future directions from gaps in evaluation, grounding, or deployment evidence.",
                    length_targets.get(desired_length, length_targets["medium"]),
                ]
            ),
            "desired_length": desired_length,
        }

    except Exception as exc:
        logger.exception("writing_scaffold_generator failed: %s", exc)
        return {"error": f"writing_scaffold_generator failed: {exc}"}


# ─── Helpers ───────────────────────────────────────────────────────────────────


def _extract_json(text: str) -> dict | None:
    """Extract first JSON object/dict from LLM output text."""
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try JSON block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


def _clean_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _first_text(card: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = card.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _stringify_field_list(value: Any) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in _clean_list(value):
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return cleaned


def _infer_benchmark_terms(text: str) -> list[str]:
    lowered = str(text or "").lower()
    markers = [
        "accuracy",
        "auc",
        "dice",
        "f1",
        "sensitivity",
        "specificity",
        "calibration",
        "latency",
        "throughput",
    ]
    inferred = []
    for marker in markers:
        if marker in lowered:
            inferred.append(marker.upper() if len(marker) <= 3 else marker)
    return inferred


def _top_phrases(values: list[str], *, limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for value in values:
        for part in re.split(r"[;,/]| and ", str(value or ""), flags=re.IGNORECASE):
            cleaned = part.strip()
            if not cleaned or cleaned.lower().startswith("not "):
                continue
            counts[cleaned] = counts.get(cleaned, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))
    return [item[0] for item in ranked[:limit]]


def _build_survey_title(topic: str, evidence_titles: list[str]) -> str:
    cleaned_topic = str(topic or "").strip().rstrip(".")
    if cleaned_topic:
        if "survey" in cleaned_topic.lower():
            return cleaned_topic
        return f"{cleaned_topic}: A Structured Survey"
    if evidence_titles:
        return f"{evidence_titles[0]} and Related Systems: A Structured Survey"
    return "Structured Research Survey"
