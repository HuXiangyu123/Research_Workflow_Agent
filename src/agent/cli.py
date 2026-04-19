from __future__ import annotations

import argparse
import sys
from dotenv import load_dotenv

from src.agent.llm import build_deepseek_chat
from src.agent.react_agent import build_react_agent
from src.agent.report import generate_literature_report
from src.agent.settings import Settings

import os
import re
import time
import json
from src.agent.callbacks import AgentProgressCallback
from src.tools.arxiv_paper import _extract_arxiv_id
from src.memory import ConversationStore, LongTermMemory
from src.ingest.ingestor import ingest_from_seeds
from src.ingest.indexer import build_index
from src.validators.citations_validator import has_citations_section

def _save_report_to_file(report_content: str, arxiv_input: str) -> str:
    """保存报告到 output 目录，文件名为 'arxivID报告.md'"""
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    arxiv_id = _extract_arxiv_id(arxiv_input)
    
    if arxiv_id:
        filename = f"{arxiv_id}报告.md"
    else:
        # Fallback: 尝试提取标题作为文件名
        title_match = re.search(r'^#\s+(.+)$', report_content, re.MULTILINE)
        if title_match:
            # 清理文件名中的非法字符
            clean_title = re.sub(r'[\\/*?:"<>|]', "", title_match.group(1).strip())
            filename = f"{clean_title}.md"
        else:
            filename = f"report_{int(time.time())}.md"
    
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    return os.path.abspath(filepath)

from langchain_core.messages import HumanMessage, AIMessage

def run_repl() -> None:
    print("文献报告 Agent 已启动 (输入 'quit' 或 'exit' 退出)")
    print("-" * 50)
    settings = Settings.from_env()
    llm = build_deepseek_chat(settings)
    agent = build_react_agent(llm)
    
    # 初始化回调
    progress_callback = AgentProgressCallback()

    user_memory = LongTermMemory(path=os.path.join("data", "user_memory.json"))
    session_id = str(int(time.time()))
    conversation = ConversationStore(path=os.path.join("data", "conversations", f"{session_id}.json"))
    
    # In-memory history for current session (LangChain messages)
    chat_history_messages = []

    while True:
        try:
            user_input = input("\nUser (输入 arXiv 链接 或 提问): ").strip()
        except EOFError:
            return

        if user_input.lower() in {"quit", "exit"}:
            return
        if not user_input:
            continue

        if user_input.startswith("/remember "):
            item = user_input[len("/remember ") :].strip()
            user_memory.add(item)
            print("已记录到长期记忆。")
            continue
        if user_input == "/memory":
            text = user_memory.to_prompt()
            print(text if text else "长期记忆为空。")
            continue
        if user_input == "/memory_clear":
            user_memory.clear()
            print("长期记忆已清空。")
            continue
            
        print("\n🚀 开始处理，请稍候...")
        try:
            # 1. Update persisted conversation log (JSON)
            conversation.append("User", user_input)
            
            # 2. Prepare context
            extra_context = "\n\n".join(
                [x for x in [user_memory.to_prompt(), conversation.to_prompt()] if x.strip()]
            ).strip()

            # 3. Call Agent
            # If history is empty, assume it's a new report generation request (or general question)
            # If history is not empty, pass it to support chat
            
            # If input looks like an Arxiv ID/URL, we might want to clear history or treat as new topic?
            # For now, we just append to history to support "compare with previous paper" scenarios.
            
            response_text = generate_literature_report(
                agent, 
                arxiv_url_or_id=user_input, # This arg name is a bit misleading now, it's just "input"
                callbacks=[progress_callback],
                extra_system_context=extra_context if not chat_history_messages else None, # Only inject context once in system prompt or let agent handle?
                chat_history=chat_history_messages if chat_history_messages else None
            )
            
            # 4. Update Chat History
            chat_history_messages.append(HumanMessage(content=user_input))
            chat_history_messages.append(AIMessage(content=response_text))
            
            # 5. Handle Report Saving (only if it looks like a report)
            # Simple heuristic: markdown title + section headings + a references appendix.
            is_report = (
                "# " in response_text
                and "## " in response_text
                and ("## References" in response_text or "## 引用" in response_text)
            )
            
            if is_report:
                 # 保存到文件
                filepath = _save_report_to_file(response_text, user_input)
                conversation.append("Assistant", f"报告已生成并保存至: {filepath}")
                print(f"\n✅ 报告已保存至: {filepath}")
            else:
                conversation.append("Assistant", response_text)

            print("\n" + "="*50)
            print(response_text)
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"\n❌ 处理失败: {str(e)}")

def run_eval(cases_path: str, out_path: str):
    print(f"Starting evaluation using {cases_path}...")
    
    if not os.path.exists(cases_path):
        print(f"Error: Cases file {cases_path} not found.")
        return

    settings = Settings.from_env()
    llm = build_deepseek_chat(settings)
    agent = build_react_agent(llm)
    
    results = []
    
    with open(cases_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            case = json.loads(line)
            input_url = case["input"]
            print(f"Evaluating: {input_url}")
            
            start_time = time.time()
            try:
                report = generate_literature_report(agent, arxiv_url_or_id=input_url)
                duration = time.time() - start_time
                
                # Checks
                checks = {
                    "has_citations": has_citations_section(report),
                    "must_include": {},
                    "min_citations_met": False
                }
                
                # Check must_include
                for kw in case.get("must_include", []):
                    checks["must_include"][kw] = kw.lower() in report.lower()
                
                # Check min citations (simple heuristic)
                citation_count = report.lower().count("http") # rough count of links in citation section? 
                # Better: count list items in citation section
                # For MVP, let's assume valid if has section
                if checks["has_citations"]:
                    # Try to count bullets in last section
                    lines = report.split('\n')
                    # find last header
                    last_header_idx = -1
                    for i, l in enumerate(lines):
                        if l.startswith('#') and ('引用' in l or 'Reference' in l):
                            last_header_idx = i
                    
                    if last_header_idx != -1:
                        # count bullets
                        count = 0
                        for l in lines[last_header_idx:]:
                            if l.strip().startswith('-') or l.strip().startswith('*') or re.match(r'\d+\.', l.strip()):
                                count += 1
                        checks["min_citations_met"] = count >= case.get("min_citations", 0)
                        checks["citation_count"] = count
                
                passed = checks["has_citations"] and all(checks["must_include"].values()) and checks["min_citations_met"]
                
                result = {
                    "input": input_url,
                    "passed": passed,
                    "duration": duration,
                    "checks": checks,
                    "report_snippet": report[:200]
                }
                results.append(result)
                print(f"  -> {'PASS' if passed else 'FAIL'} ({duration:.2f}s)")
                
            except Exception as e:
                print(f"  -> ERROR: {e}")
                results.append({"input": input_url, "passed": False, "error": str(e)})

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation complete. Results saved to {out_path}")

def main() -> None:
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Literature Report Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # REPL (Default)
    subparsers.add_parser("repl", help="Start interactive REPL")
    
    # Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest seeds into corpus")
    ingest_parser.add_argument("--force", action="store_true", help="Force re-ingestion")
    
    # Build Index
    index_parser = subparsers.add_parser("build-index", help="Build vector index from corpus")
    index_parser.add_argument("--force", action="store_true", help="Force rebuild")
    
    # Eval
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--cases", default="eval/cases.jsonl", help="Path to test cases")
    eval_parser.add_argument("--out", default="eval/reports/run.json", help="Output path for report")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        ingest_from_seeds(force=args.force)
    elif args.command == "build-index":
        build_index(force=args.force)
    elif args.command == "eval":
        run_eval(args.cases, args.out)
    else:
        # Default to REPL if no args or 'repl'
        run_repl()

if __name__ == "__main__":
    main()
