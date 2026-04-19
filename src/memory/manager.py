"""Agent memory adapter backed by LangGraph checkpoint interfaces.

The runtime-facing API is intentionally small and transient. Earlier versions
persisted semantic/episodic/preference memory as JSON files under ``.memory``;
that violates the current project rules. This module now keeps short-lived
memory in process and exposes a LangGraph ``BaseCheckpointSaver`` for graph
state ownership. Durable memory should be added through PostgreSQL-backed
services only.

用法：
    from src.memory import get_memory_manager
    mm = get_memory_manager(workspace_id="ws_xxx")
    mm.add_sensory("search_result", {"query": "...", "results": [...]})
    context = mm.build_context(topic="RAG")  # 查询相关 runtime memory
    mm.inject_into_prompt(messages, context)  # 注入 LLM prompt
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
import numpy as np

from src.agent.checkpointing import get_langgraph_checkpointer

logger = logging.getLogger(__name__)


def _looks_like_secret(text: str) -> bool:
    s = text.lower()
    return any(k in s for k in ["api_key", "apikey", "token", "secret", "sk-"])


# ─── Runtime Event Buffer ───────────────────────────────────────────────────


@dataclass
class RuntimeEvent:
    """Single runtime event: tool output, user input, or external signal."""

    event_id: str
    event_type: str          # "tool_output" | "user_input" | "external_signal"
    content: str | dict
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return json.dumps(self.content, ensure_ascii=False, indent=2)[:500]

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class RuntimeEventBuffer:
    """
    Runtime event buffer for the current process/session.

    容量：最近 50 个事件（环形缓冲）
    TTL：当前会话内有效
    """

    MAX_EVENTS = 50

    def __init__(self, workspace_id: str):
        self._workspace_id = workspace_id
        self._events: list[RuntimeEvent] = []
        self._event_counter = 0

    def add(self, event_type: str, content: str | dict, metadata: dict | None = None) -> str:
        if _looks_like_secret(str(content)):
            return ""
        event_id = f"se_{self._workspace_id}_{self._event_counter}"
        self._event_counter += 1
        event = RuntimeEvent(
            event_id=event_id,
            event_type=event_type,
            content=content,
            metadata=metadata or {},
        )
        self._events.append(event)
        if len(self._events) > self.MAX_EVENTS:
            self._events.pop(0)
        return event_id

    def add_tool_output(self, tool_name: str, output: str | dict) -> str:
        return self.add("tool_output", output, {"tool": tool_name})

    def add_user_input(self, text: str) -> str:
        return self.add("user_input", text)

    def recent(self, n: int = 10, event_type: str | None = None) -> list[RuntimeEvent]:
        events = self._events[-n:] if n > 0 else self._events
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events

    def to_prompt(self, max_events: int = 5) -> str:
        """格式化为 prompt 片段。"""
        recent = self.recent(max_events)
        if not recent:
            return ""
        lines = ["## 近期感官输入："]
        for e in recent:
            prefix = f"[{e.event_type}]"
            lines.append(f"{prefix} {e.to_text()[:300]}")
        return "\n".join(lines)


# ─── Runtime Working State ──────────────────────────────────────────────────


@dataclass
class RuntimeWorkingState:
    """
    Current-session LLM context and agent loop state.

    包含：
    - messages：当前上下文窗口内的对话消息
    - agent_state：当前 agent loop 的迭代状态
    - context_window：当前 prompt 的可用 token 预算
    """
    messages: list[dict] = field(default_factory=list)   # role/content 消息列表
    agent_state: dict = field(default_factory=dict)      # agent 循环状态（iter_count 等）
    context_budget: int = 6000                          # 当前可用 token 预算
    summary: str = ""                                    # 历史摘要（压缩后）

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content, "ts": int(time.time())})

    def set_summary(self, summary: str) -> None:
        self.summary = summary

    def to_prompt(self, max_messages: int = 10) -> str:
        """格式化为 prompt 片段。"""
        lines = ["## 工作记忆（当前会话）："]
        if self.summary:
            lines.append(f"摘要：{self.summary}")
        recent = self.messages[-max_messages:] if self.messages else []
        if recent:
            lines.append("最近对话：")
            for m in recent:
                lines.append(f"{m['role']}: {m['content'][:200]}")
        return "\n".join(lines)

    def update_agent_state(self, **kwargs) -> None:
        self.agent_state.update(kwargs)

    def get_agent_state(self, key: str, default: Any = None) -> Any:
        return self.agent_state.get(key, default)


# ─── Runtime Vector Cache ───────────────────────────────────────────────────


@dataclass
class RuntimeMemoryEntry:
    """Runtime vector cache entry: text + vector + metadata."""

    entry_id: str
    text: str
    vector: list[float]
    memory_type: str         # "domain_knowledge" | "research_fact" | "preference"
    workspace_id: str
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "text": self.text,
            "vector": self.vector,
            "memory_type": self.memory_type,
            "workspace_id": self.workspace_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class RuntimeVectorCache:
    """
    Vector-based runtime knowledge cache.

    依赖：src.embeddings.get_embedding_client
    存储：进程内 transient cache；长期存储必须走 PostgreSQL 服务层
    检索：cosine similarity（numpy 实现）
    """

    MAX_ENTRIES = 500

    def __init__(self, workspace_id: str, storage_dir: str | None = None):
        self._workspace_id = workspace_id
        self._entries: list[RuntimeMemoryEntry] = []
        self._loaded = False
        self._client = None

    def _ensure_loaded(self) -> None:
        self._loaded = True

    def _save(self) -> None:
        return

    def add(self, text: str, memory_type: str = "domain_knowledge", metadata: dict | None = None) -> str:
        """添加语义记忆条目（自动生成向量）。"""
        if not text or _looks_like_secret(text):
            return ""
        self._ensure_loaded()

        from src.embeddings import get_embedding_client

        try:
            client = get_embedding_client()
            vec = client.encode([text])[0].tolist()
        except Exception as exc:
            logger.warning("[RuntimeVectorCache] embedding failed: %s, using zero vector", exc)
            from src.embeddings import get_embedding_dimension

            dim = get_embedding_dimension()
            vec = [0.0] * dim

        entry_id = f"sm_{self._workspace_id}_{len(self._entries)}"
        entry = RuntimeMemoryEntry(
            entry_id=entry_id,
            text=text,
            vector=vec,
            memory_type=memory_type,
            workspace_id=self._workspace_id,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        if len(self._entries) > self.MAX_ENTRIES:
            self._entries.pop(0)
        self._save()
        return entry_id

    def search(self, query: str, top_k: int = 5, memory_type: str | None = None) -> list[RuntimeMemoryEntry]:
        """
        语义检索：返回与 query 最相似的 top_k 条记忆。

        使用 cosine similarity：cosine(a, b) = dot(a, b) / (||a|| * ||b||)
        由于向量已归一化，cosine_sim = dot(a, b)
        """
        self._ensure_loaded()
        if not self._entries:
            return []

        from src.embeddings import get_embedding_client

        try:
            query_vec = get_embedding_client().encode([query])[0]
        except Exception as exc:
            logger.warning("[RuntimeVectorCache] query embedding failed: %s", exc)
            return []

        query_vec = np.array(query_vec)
        scores: list[tuple[int, float]] = []
        for i, entry in enumerate(self._entries):
            if memory_type and entry.memory_type != memory_type:
                continue
            entry_vec = np.array(entry.vector)
            score = float(np.dot(query_vec, entry_vec))
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [self._entries[i] for i, _ in scores[:top_k]]

    def to_prompt(self, query: str | None = None, top_k: int = 5) -> str:
        """
        格式化为 prompt 片段。

        若提供 query，执行语义检索；否则返回所有条目。
        """
        if query:
            entries = self.search(query, top_k=top_k)
        else:
            entries = self._entries[-top_k:]
        if not entries:
            return ""

        lines = ["## 运行期记忆（向量检索）："]
        for e in entries:
            age = _format_age(e.created_at)
            lines.append(f"[{e.memory_type}] ({age}) {e.text[:300]}")
        return "\n".join(lines)


# ─── Runtime Episode Log ────────────────────────────────────────────────────


@dataclass
class RuntimeEpisode:
    """Single runtime task episode."""

    episode_id: str
    task_id: str
    workspace_id: str
    topic: str
    start_time: float
    end_time: float | None = None
    status: str = "running"   # "running" | "completed" | "failed"
    stages: list[dict] = field(default_factory=list)   # 各阶段执行记录
    outcome_summary: str = ""   # 结果摘要
    artifacts: list[str] = field(default_factory=list)   # 产出的 artifact IDs
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "workspace_id": self.workspace_id,
            "topic": self.topic,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "stages": self.stages,
            "outcome_summary": self.outcome_summary,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }


class RuntimeEpisodeLog:
    """
    Current-process task episode log.

    存储：进程内 transient cache；长期存储必须走 PostgreSQL 服务层
    索引：按 topic / status / time 过滤
    """

    MAX_EPISODES = 100

    def __init__(self, workspace_id: str, storage_dir: str | None = None):
        self._workspace_id = workspace_id
        self._episodes: list[RuntimeEpisode] = []
        self._loaded = False
        self._current_episode: RuntimeEpisode | None = None

    def _ensure_loaded(self) -> None:
        self._loaded = True

    def _save(self) -> None:
        return

    def start_episode(self, task_id: str, topic: str, metadata: dict | None = None) -> str:
        """开始一个新情景（任务开始时调用）。"""
        self._ensure_loaded()
        episode_id = f"ep_{task_id}_{len(self._episodes)}"
        episode = RuntimeEpisode(
            episode_id=episode_id,
            task_id=task_id,
            workspace_id=self._workspace_id,
            topic=topic,
            start_time=time.time(),
            metadata=metadata or {},
        )
        self._episodes.append(episode)
        self._current_episode = episode
        return episode_id

    def end_episode(self, status: str, outcome_summary: str, artifacts: list[str] | None = None) -> None:
        """结束当前情景（任务完成/失败时调用）。"""
        if self._current_episode is None:
            return
        self._current_episode.end_time = time.time()
        self._current_episode.status = status
        self._current_episode.outcome_summary = outcome_summary
        if artifacts:
            self._current_episode.artifacts.extend(artifacts)
        self._current_episode = None
        self._save()

    def add_stage(self, stage_name: str, stage_data: dict) -> None:
        """记录当前情景的一个阶段执行记录。"""
        if self._current_episode is None:
            return
        self._current_episode.stages.append({
            "stage": stage_name,
            "timestamp": time.time(),
            "data": stage_data,
        })

    def recent(self, n: int = 5, status: str | None = None) -> list[RuntimeEpisode]:
        """返回最近 n 个情景。"""
        self._ensure_loaded()
        eps = self._episodes[-n:] if n > 0 else self._episodes
        if status:
            eps = [e for e in eps if e.status == status]
        return eps

    def by_topic(self, topic: str) -> list[RuntimeEpisode]:
        """按主题搜索情景。"""
        self._ensure_loaded()
        return [e for e in self._episodes if topic.lower() in e.topic.lower()]

    def to_prompt(self, current_topic: str | None = None, n: int = 3) -> str:
        """
        格式化为 prompt 片段。

        若提供 current_topic，优先检索相似主题的情景。
        """
        self._ensure_loaded()
        if current_topic:
            episodes = self.by_topic(current_topic)[-n:]
        else:
            episodes = self.recent(n)
        if not episodes:
            return ""

        lines = ["## 历史情景记忆："]
        for e in episodes:
            duration = ""
            if e.end_time:
                duration = f"（{_format_age(e.start_time)}，持续 {_format_duration(e.end_time - e.start_time)}）"
            status_icon = "✓" if e.status == "completed" else "✗" if e.status == "failed" else "→"
            lines.append(f"{status_icon} [{e.topic}]{duration} — {e.outcome_summary[:100]}")
        return "\n".join(lines)


# ─── Runtime Preference Store ───────────────────────────────────────────────


class RuntimePreferenceStore:
    """
    Current-process user preference cache.

    存储：进程内 transient cache；长期存储必须走 PostgreSQL 服务层
    """

    def __init__(self, workspace_id: str, storage_dir: str | None = None):
        self._workspace_id = workspace_id
        self._data: dict = {}

    def set(self, key: str, value: str) -> None:
        if _looks_like_secret(value):
            return
        self._data[key] = {"value": value, "updated_at": time.time()}

    def get(self, key: str, default: str = "") -> str:
        item = self._data.get(key)
        return item.get("value", default) if item else default

    def to_prompt(self) -> str:
        if not self._data:
            return ""
        lines = ["## 用户偏好："]
        for key, item in self._data.items():
            val = item.get("value", "")
            if val and val not in ("true", "false"):
                lines.append(f"- {key}: {val}")
        return "\n".join(lines)


# ─── MemoryManager（统一入口）───────────────────────────────────────────────


class MemoryManager:
    """
    LangGraph checkpoint-aware runtime memory adapter.

    It preserves the existing add/search/build_context interface used by the
    v2 agents while avoiding custom durable memory stores. The saver is exposed
    so graph builders can share the same LangGraph checkpoint implementation.
    """

    def __init__(
        self,
        workspace_id: str,
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        self._ws_id = workspace_id
        self._checkpointer = checkpointer or get_langgraph_checkpointer(f"memory:{workspace_id}")
        self._events = RuntimeEventBuffer(workspace_id)
        self._working = RuntimeWorkingState()
        self._vectors = RuntimeVectorCache(workspace_id)
        self._episodes = RuntimeEpisodeLog(workspace_id)
        self._preferences = RuntimePreferenceStore(workspace_id)

    @property
    def checkpointer(self) -> BaseCheckpointSaver:
        return self._checkpointer

    # ── 感官记忆 ──────────────────────────────────────────────────────────────

    def add_sensory(self, event_type: str, content: str | dict, metadata: dict | None = None) -> str:
        """记录感官事件。"""
        return self._events.add(event_type, content, metadata)

    def add_tool_output(self, tool_name: str, output: str | dict) -> str:
        """记录工具输出。"""
        return self._events.add_tool_output(tool_name, output)

    # ── 工作记忆 ──────────────────────────────────────────────────────────────

    def add_message(self, role: str, content: str) -> None:
        """追加对话消息到工作记忆。"""
        self._working.add_message(role, content)

    def set_summary(self, summary: str) -> None:
        """设置会话摘要（压缩工作记忆）。"""
        self._working.set_summary(summary)

    def update_agent_state(self, **kwargs) -> None:
        """更新 agent 状态。"""
        self._working.update_agent_state(**kwargs)

    def get_agent_state(self, key: str, default: Any = None) -> Any:
        return self._working.get_agent_state(key, default)

    # ── 语义记忆 ──────────────────────────────────────────────────────────────

    def add_semantic(self, text: str, memory_type: str = "domain_knowledge", metadata: dict | None = None) -> str:
        """添加语义记忆。"""
        return self._vectors.add(text, memory_type, metadata)

    def search_semantic(self, query: str, top_k: int = 5, memory_type: str | None = None) -> list[RuntimeMemoryEntry]:
        """语义检索。"""
        return self._vectors.search(query, top_k, memory_type)

    # ── 情景记忆 ──────────────────────────────────────────────────────────────

    def start_episode(self, task_id: str, topic: str, metadata: dict | None = None) -> str:
        """开始新任务情景。"""
        return self._episodes.start_episode(task_id, topic, metadata)

    def end_episode(self, status: str, outcome_summary: str, artifacts: list[str] | None = None) -> None:
        """结束当前情景。"""
        self._episodes.end_episode(status, outcome_summary, artifacts)

    def add_stage(self, stage_name: str, stage_data: dict) -> None:
        """记录情景阶段。"""
        self._episodes.add_stage(stage_name, stage_data)

    # ── 偏好记忆 ──────────────────────────────────────────────────────────────

    def set_preference(self, key: str, value: str) -> None:
        self._preferences.set(key, value)

    def get_preference(self, key: str, default: str = "") -> str:
        return self._preferences.get(key, default)

    # ── 上下文构建 ────────────────────────────────────────────────────────────

    def build_context(
        self,
        topic: str | None = None,
        max_sensory: int = 5,
        max_semantic: int = 5,
        max_episodes: int = 3,
        include_preference: bool = True,
    ) -> str:
        """
        构建完整的 memory context 字符串，用于注入 LLM prompt。

        拼接顺序：preference → semantic → episodic → working → sensory
        """
        parts: list[str] = []

        if include_preference:
            pref = self._preferences.to_prompt()
            if pref:
                parts.append(pref)

        if topic:
            sem = self._vectors.to_prompt(query=topic, top_k=max_semantic)
            if sem:
                parts.append(sem)

        epi = self._episodes.to_prompt(current_topic=topic, n=max_episodes)
        if epi:
            parts.append(epi)

        work = self._working.to_prompt()
        if work:
            parts.append(work)

        sens = self._events.to_prompt(max_events=max_sensory)
        if sens:
            parts.append(sens)

        if not parts:
            return ""
        return "\n\n".join(parts)

    def inject_into_messages(
        self,
        messages: list[dict],
        topic: str | None = None,
        prepend_system: bool = True,
        memory_section: str = "## Memory Context",
    ) -> list[dict]:
        """
        将 memory context 注入到消息列表中。

        若 prepend_system=True，在 system message 末尾追加 memory context；
        否则作为 user message 插入。
        """
        context = self.build_context(topic=topic)
        if not context:
            return messages

        memory_text = f"{memory_section}\n\n{context}"

        if prepend_system and messages and messages[0].get("role") == "system":
            messages[0]["content"] = messages[0]["content"].rstrip() + "\n\n" + memory_text
        else:
            messages.insert(0, {"role": "system", "content": memory_text})

        return messages


# ─── 全局单例 ───────────────────────────────────────────────────────────────


_memory_managers: dict[str, MemoryManager] = {}


def get_memory_manager(workspace_id: str) -> MemoryManager:
    """获取或创建 workspace 的 MemoryManager 单例。"""
    if workspace_id not in _memory_managers:
        _memory_managers[workspace_id] = MemoryManager(workspace_id)
    return _memory_managers[workspace_id]


def reset_memory_manager(workspace_id: str) -> None:
    """重置指定 workspace 的 MemoryManager（测试用）。"""
    _memory_managers.pop(workspace_id, None)


# ─── 辅助函数 ───────────────────────────────────────────────────────────────


def _format_age(timestamp: float) -> str:
    """将时间戳格式化为相对时间。"""
    delta = time.time() - timestamp
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta / 60)}m ago"
    if delta < 86400:
        return f"{int(delta / 3600)}h ago"
    return f"{int(delta / 86400)}d ago"


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m"
    return f"{int(seconds / 3600)}h"
