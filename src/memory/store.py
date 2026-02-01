from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        _ensure_dir(parent)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


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
        data = _read_json(self.path, default={"items": []})
        items = data.get("items", [])
        return [x for x in items if isinstance(x, str)]

    def add(self, item: str) -> None:
        item = (item or "").strip()
        if not item or _looks_like_secret(item):
            return
        data = _read_json(self.path, default={"items": []})
        items = data.get("items", [])
        if not isinstance(items, list):
            items = []
        if item not in items:
            items.append(item)
        _write_json(self.path, {"items": items})

    def clear(self) -> None:
        _write_json(self.path, {"items": []})

    def to_prompt(self, max_items: int = 20) -> str:
        items = self.load()[:max_items]
        if not items:
            return ""
        lines = ["用户长期记忆（来自历史偏好与显式记录）："]
        for it in items:
            lines.append(f"- {it}")
        return "\n".join(lines)


@dataclass
class ConversationStore:
    path: str

    def load(self) -> dict[str, Any]:
        return _read_json(self.path, default={"messages": [], "summary": ""})

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
        data["messages"] = messages
        _write_json(self.path, data)

    def set_summary(self, summary: str) -> None:
        data = self.load()
        data["summary"] = (summary or "").strip()
        _write_json(self.path, data)

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

