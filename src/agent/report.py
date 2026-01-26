from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from src.agent.prompts import LITERATURE_REPORT_SYSTEM_PROMPT
from src.validators.citations_validator import has_citations_section


def _last_ai_text(state: dict[str, Any]) -> str:
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            return msg.content
    if messages:
        last = messages[-1]
        content = getattr(last, "content", "")
        return content if isinstance(content, str) else ""
    return ""


from langchain_core.callbacks import BaseCallbackHandler

def generate_literature_report(
    agent: Runnable, 
    arxiv_url_or_id: str | None = None, 
    raw_text_content: str | None = None,
    callbacks: list[BaseCallbackHandler] | None = None
) -> str:
    """
    生成文献报告。
    可以传入 arxiv 链接/ID，也可以直接传入论文文本内容。
    """
    if raw_text_content:
        # 直接基于文本内容生成
        # 注意：这里可能会受到 Context Window 限制，DeepSeek V3/R1 支持长上下文
        prompt = (
            "请基于以下提供的论文内容（全文文本）生成详细的文献报告。\n"
            "输出为 Markdown。\n"
            "必须包含：标题、核心贡献、方法概述、关键实验/结果、局限性、可复现要点、相关工作。\n"
            "最后必须包含“引用”小节，列出文中提到的参考文献或链接（如果文本中包含引用信息），格式为 label、url、reason。\n"
            "如果无法获取外部 URL，请根据文本内容推断或保留原始引用标记。\n\n"
            "=== 论文内容开始 ===\n"
            f"{raw_text_content[:100000]}..." # 简单截断防止过长，假设 DeepSeek 足够强
            "\n=== 论文内容结束 ==="
        )
    elif arxiv_url_or_id:
        prompt = (
            "请根据以下 arXiv 链接或 arXiv ID 生成文献报告：\n"
            f"{arxiv_url_or_id}\n\n"
            "输出为 Markdown。\n"
            "必须包含：标题、核心贡献、方法概述、关键实验/结果、局限性、可复现要点、相关工作。\n"
            "最后必须包含“引用”小节，列出每条引用的 label、url、reason。"
        )
    else:
        return "Error: No input provided (arxiv_url_or_id or raw_text_content required)."

    config = {"callbacks": callbacks} if callbacks else None

    state = agent.invoke(
        {"messages": [SystemMessage(content=LITERATURE_REPORT_SYSTEM_PROMPT), HumanMessage(content=prompt)]},
        config=config
    )
    text = _last_ai_text(state)

    if has_citations_section(text):
        return text

    repair_prompt = (
        "你上一轮输出缺少“引用”小节。请保持原结构不变，在末尾补充“引用”小节，"
        "列出至少 3 条可追溯引用，每条包含 label、url、reason。输出 Markdown。\n\n"
        f"原输出：\n{text}"
    )
    repaired_state = agent.invoke(
        {
            "messages": [
                SystemMessage(content=LITERATURE_REPORT_SYSTEM_PROMPT),
                HumanMessage(content=repair_prompt),
            ]
        },
        config=config
    )
    repaired_text = _last_ai_text(repaired_state)
    return repaired_text or text

