from __future__ import annotations

from typing import Any, Iterable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.prebuilt import create_react_agent

from src.agent.prompts import LITERATURE_REPORT_SYSTEM_PROMPT
from src.tools.arxiv_paper import get_arxiv_paper_info
from src.tools.web_fetch import fetch_webpage_text


def build_react_agent(llm: Runnable) -> Runnable:
    tools = [get_arxiv_paper_info, fetch_webpage_text]
    return create_react_agent(llm, tools)


def stream_literature_report(agent: Runnable, arxiv_url_or_id: str) -> Iterable[dict[str, Any]]:
    prompt = (
        "请根据以下 arXiv 链接或 arXiv ID 生成文献报告：\n"
        f"{arxiv_url_or_id}\n\n"
        "要求：先获取论文元信息，然后检索相关开源资料与相关工作，最后输出结构化报告与引用列表。"
    )
    inputs = {
        "messages": [
            SystemMessage(content=LITERATURE_REPORT_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    }
    return agent.stream(inputs, stream_mode="values")

