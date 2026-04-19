"""Survey writing rules distilled from external academic survey guidance.

The canonical human-readable source lives under ``docs/template`` so the repo
has a single writing contract for both documentation and prompt injection.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

DEFAULT_SURVEY_WRITING_RULES: tuple[str, ...] = (
    "Abstract must state motivation, scope, key synthesis, and why the review matters.",
    "Introduction must define the topic, review boundary, and organizing logic.",
    "The body should organize evidence by themes, methods, datasets, benchmarks, or debates rather than by paper order.",
    "Discussion must compare findings, surface trade-offs, and identify evidence gaps.",
    "Future work must come from unresolved gaps, missing evaluations, or deployment constraints, not from paraphrased limitations.",
    "Maintain citation diversity and avoid letting a tiny subset of papers dominate the whole report when the corpus supports broader coverage.",
)

TEMPLATE_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "template"
    / "survey_writing_constraints.md"
)
PROMPT_RULES_START = "<!-- SURVEY_PROMPT_RULES_START -->"
PROMPT_RULES_END = "<!-- SURVEY_PROMPT_RULES_END -->"


def _fallback_rules_block() -> str:
    lines = ["SURVEY WRITING RULES:"]
    for idx, rule in enumerate(DEFAULT_SURVEY_WRITING_RULES, start=1):
        lines.append(f"{idx}. {rule}")
    lines.append("These rules are grounded in academic review-writing guidance, not generic blogging style.")
    return "\n".join(lines)


def _extract_prompt_rules_block(text: str) -> str | None:
    start = text.find(PROMPT_RULES_START)
    end = text.find(PROMPT_RULES_END)
    if start == -1 or end == -1 or end <= start:
        return None
    block = text[start + len(PROMPT_RULES_START):end].strip()
    return block or None


@lru_cache(maxsize=1)
def build_survey_writing_rules_block() -> str:
    """Render the survey writing rules as a compact prompt block."""
    try:
        content = TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError:
        return _fallback_rules_block()

    block = _extract_prompt_rules_block(content)
    return block or _fallback_rules_block()
