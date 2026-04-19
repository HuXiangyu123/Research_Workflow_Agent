"""LangGraph checkpoint helpers.

Default behavior is intentionally local and test-friendly: use MemorySaver for
transient execution state. Set LANGGRAPH_CHECKPOINT_BACKEND=postgres to opt into
PostgresSaver backed by DATABASE_URL for durable graph checkpoints.
"""

from __future__ import annotations

import atexit
import os
from contextlib import AbstractContextManager
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

_CHECKPOINTERS: dict[tuple[str, str, str], BaseCheckpointSaver] = {}
_POSTGRES_CONTEXTS: list[AbstractContextManager] = []


def get_langgraph_checkpointer(namespace: str = "default") -> BaseCheckpointSaver:
    """Return the configured LangGraph checkpoint saver for a namespace."""
    backend = os.getenv("LANGGRAPH_CHECKPOINT_BACKEND", "memory").strip().lower()
    if backend in {"", "memory", "inmemory", "in_memory"}:
        key = ("memory", namespace, "")
        if key not in _CHECKPOINTERS:
            _CHECKPOINTERS[key] = MemorySaver()
        return _CHECKPOINTERS[key]

    if backend == "postgres":
        database_url = os.getenv("DATABASE_URL", "").strip()
        if not database_url:
            raise RuntimeError(
                "LANGGRAPH_CHECKPOINT_BACKEND=postgres requires DATABASE_URL to be set."
            )
        key = ("postgres", namespace, database_url)
        if key not in _CHECKPOINTERS:
            from langgraph.checkpoint.postgres import PostgresSaver

            context = PostgresSaver.from_conn_string(database_url)
            saver = context.__enter__()
            saver.setup()
            _POSTGRES_CONTEXTS.append(context)
            _CHECKPOINTERS[key] = saver
        return _CHECKPOINTERS[key]

    raise ValueError(
        "Unsupported LANGGRAPH_CHECKPOINT_BACKEND="
        f"{backend!r}; expected 'memory' or 'postgres'."
    )


def build_graph_config(
    namespace: str,
    *,
    thread_id: str | None = None,
    recursion_limit: int | None = None,
) -> dict:
    """Build a LangGraph runnable config with the required checkpoint thread id."""
    config: dict = {
        "configurable": {
            "thread_id": thread_id or f"{namespace}:{uuid4().hex}",
        }
    }
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    return config


def _close_postgres_contexts() -> None:
    while _POSTGRES_CONTEXTS:
        context = _POSTGRES_CONTEXTS.pop()
        context.__exit__(None, None, None)


atexit.register(_close_postgres_contexts)
