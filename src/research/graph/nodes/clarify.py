"""Graph node: run ClarifyAgent to produce a ResearchBrief from raw user input."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from src.research.agents.clarify_agent import run as run_clarify_agent
from src.research.research_brief import ClarifyInput, ClarifyResult


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return len(value) == 0
    return False


def _build_followup_hints(brief_payload: dict) -> list[str]:
    hints: list[str] = []
    ambiguities = brief_payload.get("ambiguities") or []
    if isinstance(ambiguities, list):
        for item in ambiguities:
            if not isinstance(item, dict):
                continue
            field = str(item.get("field") or "unknown")
            reason = str(item.get("reason") or "unspecified")
            options = item.get("suggested_options") or []
            if isinstance(options, list) and options:
                option_text = ", ".join(str(opt) for opt in options[:4] if opt)
                hints.append(f"{field}: {reason}. 建议选项: {option_text}")
            else:
                hints.append(f"{field}: {reason}")
    return hints


def _summarize_autofill(before: dict, after: dict, interaction_mode: str) -> str:
    tracked = [
        "topic",
        "goal",
        "desired_output",
        "time_range",
        "domain_scope",
        "sub_questions",
        "focus_dimensions",
        "source_constraints",
    ]
    filled: list[str] = []
    for field in tracked:
        if _is_missing(before.get(field)) and not _is_missing(after.get(field)):
            filled.append(field)

    before_conf = before.get("confidence")
    after_conf = after.get("confidence")
    filled_text = ", ".join(filled) if filled else "none"
    return (
        "Auto-fill fallback triggered "
        f"(mode={interaction_mode}) fields_filled={filled_text} "
        f"confidence={before_conf}->{after_conf}."
    )


def run_clarify_node(state: dict) -> dict:
    """Clarify a raw research request into a structured ResearchBrief.

    Input state fields consumed:
        raw_input : str
            The raw user query. May come from a research-mode task.
        source_type : str
            Must be "research" for this node to run; otherwise returns no-op.

    Output state patch:
        brief          : ResearchBrief as dict (compatible with ResearchBrief.model_validate)
        current_stage  : "clarify"
        warnings       : list[str]  (accumulated via Annotated)
        errors         : list[str]  (accumulated via Annotated)

    This node does NOT run for paper-ingestion mode tasks.
    """
    # Guard: only activate for research-mode tasks
    source_type = state.get("source_type", "")
    if source_type != "research":
        return {"errors": ["run_clarify_node: source_type must be 'research' for clarify"]}

    raw_query = state.get("raw_input", "")
    if not raw_query or not raw_query.strip():
        return {
            "errors": ["run_clarify_node: raw_input is empty"],
            "current_stage": "clarify",
        }

    started_at = datetime.now(UTC).isoformat()
    t0 = time.monotonic()

    emitter = state.get("_event_emitter")

    def emit_progress(message: str) -> None:
        if emitter:
            emitter.on_thinking("clarify", message)

    try:
        auto_fill = bool(state.get("auto_fill", False))
        interaction_mode = str(state.get("interaction_mode", "non_interactive") or "non_interactive").lower()
        inp = ClarifyInput(raw_query=raw_query.strip(), auto_fill=auto_fill)
        result: ClarifyResult = run_clarify_agent(inp, emit_progress=emit_progress)

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        # Token estimate: ~4 chars/token
        tokens_estimate = len(raw_query) // 4 + 1200
        warnings = list(result.warnings)
        resolved = result
        awaiting_followup = False
        auto_fill_triggered = False

        if result.brief.needs_followup:
            if auto_fill:
                warnings.append(
                    f"ClarifyAgent still returned needs_followup=True in auto_fill mode "
                    f"(confidence={result.brief.confidence:.2f})."
                )
            elif interaction_mode == "interactive":
                awaiting_followup = True
                warnings.append(
                    f"ClarifyAgent flagged needs_followup=True (confidence={result.brief.confidence:.2f}). "
                    "Workflow paused for user follow-up."
                )
            else:
                auto_fill_triggered = True
                before_payload = result.brief.model_dump(mode="json")
                fallback_input = ClarifyInput(raw_query=raw_query.strip(), auto_fill=True)
                fallback_result: ClarifyResult = run_clarify_agent(
                    fallback_input,
                    emit_progress=emit_progress,
                )
                resolved = fallback_result
                warnings.extend(fallback_result.warnings)
                after_payload = fallback_result.brief.model_dump(mode="json")
                warnings.append(_summarize_autofill(before_payload, after_payload, interaction_mode))
                if fallback_result.brief.needs_followup:
                    warnings.append(
                        "Auto-fill fallback still returned needs_followup=True; "
                        "non-interactive mode will continue with heuristic planning."
                    )

        brief_payload = resolved.brief.model_dump(mode="json")
        followup_hints = _build_followup_hints(brief_payload)
        status = "limited" if awaiting_followup else "done"
        current_stage = "clarify_followup_required" if awaiting_followup else "clarify"

        return {
            "brief": brief_payload,
            "current_stage": current_stage,
            "warnings": warnings,
            "followup_hints": followup_hints,
            "awaiting_followup": awaiting_followup,
            "followup_resolution": {
                "interaction_mode": interaction_mode,
                "auto_fill_triggered": auto_fill_triggered,
                "awaiting_followup": awaiting_followup,
            },
            "node_statuses": {
                "clarify": {
                    "node": "clarify",
                    "status": status,
                    "started_at": started_at,
                    "ended_at": datetime.now(UTC).isoformat(),
                    "duration_ms": elapsed_ms,
                    "warnings": warnings,
                    "error": None,
                    "tokens_delta": tokens_estimate,
                    "repair_triggered": any("repair" in w.lower() for w in warnings) or auto_fill_triggered,
                }
            },
        }

    except Exception as exc:  # noqa: BLE001
        import traceback
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        err_msg = f"run_clarify_node: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        return {
            "errors": [err_msg],
            "current_stage": "clarify",
            "node_statuses": {
                "clarify": {
                    "node": "clarify",
                    "status": "failed",
                    "started_at": started_at,
                    "ended_at": datetime.now(UTC).isoformat(),
                    "duration_ms": elapsed_ms,
                    "warnings": [],
                    "error": err_msg,
                    "tokens_delta": 0,
                    "repair_triggered": False,
                }
            },
        }
