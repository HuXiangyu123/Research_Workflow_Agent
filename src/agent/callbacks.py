from typing import Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class AgentProgressCallback(BaseCallbackHandler):
    """
    用于 CLI 的简单进度显示回调。
    """
    def __init__(self):
        self.last_tool = None

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        print("🤖 AI 正在思考...", flush=True)

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        tool_name = serialized.get("name", "Unknown Tool")
        self.last_tool = tool_name
        print(f"🛠️  正在调用工具: {tool_name} (Input: {input_str[:50]}...)", flush=True)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        print(f"✅ 工具调用完成: {self.last_tool}", flush=True)

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        print(f"❌ 发生错误: {error}", flush=True)
