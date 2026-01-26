from __future__ import annotations

from dotenv import load_dotenv

from src.agent.llm import build_deepseek_chat
from src.agent.react_agent import build_react_agent
from src.agent.report import generate_literature_report
from src.agent.settings import Settings


import os
import re
import time
from src.agent.callbacks import AgentProgressCallback
from src.tools.arxiv_paper import _extract_arxiv_id

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

def run_repl() -> None:
    print("文献报告 Agent 已启动 (输入 'quit' 或 'exit' 退出)")
    print("-" * 50)
    settings = Settings.from_env()
    llm = build_deepseek_chat(settings)
    agent = build_react_agent(llm)
    
    # 初始化回调
    progress_callback = AgentProgressCallback()
    
    while True:
        user_input = input("\nUser (输入 arXiv 链接): ").strip()
        if user_input.lower() in {"quit", "exit"}:
            return
        if not user_input:
            continue
            
        print("\n🚀 开始生成报告，请稍候...")
        try:
            report = generate_literature_report(
                agent, 
                arxiv_url_or_id=user_input,
                callbacks=[progress_callback]
            )
            
            # 保存到文件
            filepath = _save_report_to_file(report, user_input)
            
            print("\n" + "="*50)
            print("✅ 报告生成成功！")
            print(f"📄 已保存至: {filepath}")
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"\n❌ 生成失败: {str(e)}")


def main() -> None:
    load_dotenv()
    run_repl()


if __name__ == "__main__":
    main()
