from __future__ import annotations

from typing import Any, Iterable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.prebuilt import create_react_agent

from src.agent.prompts import LITERATURE_REPORT_SYSTEM_PROMPT
from src.tools.arxiv_paper import get_arxiv_paper_info
from src.tools.local_fs import read_local_file, search_local_files
from src.tools.web_fetch import fetch_webpage_text
from src.tools.rag_search import rag_search


def build_react_agent(llm: Runnable) -> Runnable:
    tools = [get_arxiv_paper_info, fetch_webpage_text, search_local_files, read_local_file, rag_search]
    return create_react_agent(llm, tools)


def stream_literature_report(agent: Runnable, arxiv_url_or_id: str) -> Iterable[dict[str, Any]]:
    prompt = (
        "Generate an evidence-grounded literature report for the following arXiv URL or arXiv ID:\n"
        f"{arxiv_url_or_id}\n\n"
        "First gather paper metadata, then retrieve relevant open resources and related work, and finally output a structured English report with references."
    )
    inputs = {
        "messages": [
            SystemMessage(content=LITERATURE_REPORT_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    }
    return agent.stream(inputs, stream_mode="values")
