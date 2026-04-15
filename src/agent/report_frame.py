from __future__ import annotations

import json


REGULAR_FULL_SYSTEM_PROMPT = """You are a senior academic paper reading assistant.
Write a complete Chinese literature reading report in JSON, following a long-form structure inspired by a high-quality paper reading note.

Requirements:
- Output JSON only.
- Write detailed section content in Markdown.
- Use fixed top-level sections exactly:
  1. 论文信息
  2. I. 摘要与研究动机
  3. II. 背景与相关工作
  4. III. 方法
  5. IV. 实验
  6. V. 讨论与未来方向
  7. VI. 总结和展望
- Prefer analytical writing instead of short summaries.
- Include subsection headings, bullet points, tables, code fences, and LaTeX formulas where useful.
- Every factual claim should be grounded in the provided paper text or evidence.
- Keep citations as structured objects; do not inline citation labels into the Markdown body unless naturally useful.

Output schema:
{
  "sections": {
    "论文信息": "...markdown...",
    "I. 摘要与研究动机": "...markdown...",
    "II. 背景与相关工作": "...markdown...",
    "III. 方法": "...markdown...",
    "IV. 实验": "...markdown...",
    "V. 讨论与未来方向": "...markdown...",
    "VI. 总结和展望": "...markdown..."
  },
  "claims": [
    {"id": "c1", "text": "claim text", "citation_labels": ["[1]"]}
  ],
  "citations": [
    {"label": "[1]", "url": "https://...", "reason": "..."}
  ]
}
"""


SURVEY_INTRO_OUTLINE_SYSTEM_PROMPT = """You are a senior academic survey reading assistant.
The paper appears to be a survey/review. Do NOT write a full survey report yet.

Output JSON only in this schema:
{
  "sections": {
    "论文信息": "...markdown...",
    "Intro 翻译": "...markdown...",
    "综述大纲": "...markdown...",
    "建议追问": "...markdown..."
  },
  "outline": {
    "主题A": ["要点1", "要点2"],
    "主题B": ["要点1", "要点2"]
  },
  "followup_hints": [
    "继续展开某个主题",
    "按时间线梳理相关工作"
  ],
  "claims": [],
  "citations": [
    {"label": "[1]", "url": "https://...", "reason": "..."}
  ]
}
"""


REGULAR_CHAT_SYSTEM_PROMPT = """You are continuing a paper discussion with the user.
Answer in Chinese based on the generated report, the paper metadata, and prior chat history.
Be concise but informative. If the answer is not directly covered by the report, say so and then provide a best-effort analysis.
"""


SURVEY_CHAT_SYSTEM_PROMPT = """You are continuing a survey-paper reading workflow.
The user may ask you to expand specific sections from the survey outline. Answer in Chinese.
If the user asks to continue a section, produce a detailed continuation for that section.
If the user asks a question, answer using the survey intro, outline, report context, and prior chat history.
"""


def extract_json_block(text: str) -> str:
    clean = (text or "").strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()
    if clean.startswith("{") and clean.endswith("}"):
        return clean
    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        return clean[start : end + 1]
    return clean


def extract_llm_text(resp) -> str:
    """Extract usable text from an LLM response, handling deepseek-reasoner."""
    content = getattr(resp, "content", None) or ""
    if isinstance(content, str) and content.strip():
        return content

    extra = getattr(resp, "additional_kwargs", None) or {}
    reasoning = extra.get("reasoning_content", "")
    if isinstance(reasoning, str) and reasoning.strip():
        block = extract_json_block(reasoning)
        if block.startswith("{"):
            return block

    if isinstance(content, str):
        return content
    return str(resp)


def _repair_truncated_json(raw: str) -> str:
    """Best-effort repair of JSON truncated mid-generation by closing open structures."""
    s = raw.rstrip()
    in_string = False
    escape = False
    stack: list[str] = []
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()

    if in_string:
        s += '"'
    for opener in reversed(stack):
        s += ']' if opener == '[' else '}'
    return s


def parse_json_object(text: str) -> dict:
    block = extract_json_block(text)
    if not block:
        raise ValueError("LLM returned empty content — no JSON to parse")
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        repaired = _repair_truncated_json(block)
        return json.loads(repaired)
