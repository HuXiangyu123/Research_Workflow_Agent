from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


_TRANSIENT_STORES: dict[str, dict[str, Any]] = {}


def _store_for(path: str) -> dict[str, Any]:
    return _TRANSIENT_STORES.setdefault(path, {"items": [], "messages": [], "summary": ""})


def _looks_like_secret(text: str) -> bool:
    s = text.lower()
    if "api_key" in s or "apikey" in s or "token" in s or "secret" in s:
        return True
    if "sk-" in s:
        return True
    return False


@dataclass
class LongTermMemory:
    path: str

    def load(self) -> list[str]:
        data = _store_for(self.path)
        items = data.get("items", [])
        return [x for x in items if isinstance(x, str)]

    def add(self, item: str) -> None:
        item = (item or "").strip()
        if not item or _looks_like_secret(item):
            return
        data = _store_for(self.path)
        items = data.get("items", [])
        if not isinstance(items, list):
            items = []
        if item not in items:
            items.append(item)
        data["items"] = items

    def clear(self) -> None:
        _store_for(self.path)["items"] = []

    def to_prompt(self, max_items: int = 20) -> str:
        items = self.load()[:max_items]
        if not items:
            return ""
        lines = ["用户运行期记忆（来自当前进程内偏好与显式记录）："]
        for it in items:
            lines.append(f"- {it}")
        return "\n".join(lines)


@dataclass
class ConversationStore:
    path: str

    def load(self) -> dict[str, Any]:
        data = _store_for(self.path)
        return {
            "messages": list(data.get("messages", [])),
            "summary": data.get("summary", ""),
        }

    def append(self, role: str, content: str) -> None:
        role = (role or "").strip()
        content = (content or "").strip()
        if not role or not content:
            return
        data = self.load()
        messages = data.get("messages", [])
        if not isinstance(messages, list):
            messages = []
        messages.append({"role": role, "content": content, "ts": int(time.time())})
        _store_for(self.path)["messages"] = messages

    def set_summary(self, summary: str) -> None:
        _store_for(self.path)["summary"] = (summary or "").strip()

    def get_summary(self) -> str:
        data = self.load()
        summary = data.get("summary", "")
        return summary if isinstance(summary, str) else ""

    def recent(self, max_messages: int = 8) -> list[dict[str, Any]]:
        data = self.load()
        messages = data.get("messages", [])
        if not isinstance(messages, list):
            return []
        return messages[-max_messages:]

    def to_prompt(self, max_messages: int = 8) -> str:
        summary = self.get_summary()
        recent_msgs = self.recent(max_messages=max_messages)
        if not summary and not recent_msgs:
            return ""

        lines: list[str] = ["对话历史上下文："]
        if summary:
            lines.append("历史摘要：")
            lines.append(summary)
        if recent_msgs:
            lines.append("最近对话：")
            for m in recent_msgs:
                role = m.get("role", "")
                content = m.get("content", "")
                if isinstance(role, str) and isinstance(content, str):
                    lines.append(f"{role}: {content}")
        return "\n".join(lines)
