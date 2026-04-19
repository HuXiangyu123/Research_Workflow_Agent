# PaperReader Agent — Memory 与状态管理

## 1. 先给结论

如果只讲当前主路径，项目里的“memory / state”应该拆成四层：

1. 运行中任务缓存：`src/api/routes/tasks.py::_tasks`
2. LangGraph checkpoint：`MemorySaver` 或 `PostgresSaver`
3. Durable API snapshot：`persisted_tasks` / `persisted_reports`
4. 用户可见 artifacts：`output/workspaces/...`

不要把它讲成“已经有完整的长期语义记忆系统”。当前更准确的说法是：

> 运行时有轻量 memory adapter，但 durable 主体仍然是 PostgreSQL snapshots + workspace artifacts。

## 2. 状态层次图

```mermaid
flowchart TD
    A[Task execution] --> B[_tasks in memory]
    A --> C[LangGraph checkpointer]
    A --> D[PersistedTask / PersistedReport]
    A --> E[output/workspaces/...]
    B --> F[SSE]
    D --> G[/tasks and /tasks/{id}/result]
    E --> H[artifact preview and replay]
```

## 3. 用了什么方法（Use What）

### 3.1 运行期缓存

- 进程内 `_tasks` 字典
- 用来支撑当前 task 的快速读写与 SSE

### 3.2 图级状态持久化

- LangGraph `BaseCheckpointSaver`
- 当前支持 `MemorySaver` 与 `PostgresSaver`

### 3.3 API 级 durable persistence

- SQLAlchemy + PostgreSQL
- `PersistedTask`
- `PersistedReport`

### 3.4 可回放工件层

- `output/workspaces/<workspace_id>/tasks/<task_id>/...`

## 4. 当前项目怎么做（How To Do）

### 4.1 运行中 `_tasks`

```python
router = APIRouter(prefix="/tasks", tags=["tasks"])

_tasks: dict[str, TaskRecord] = {}

def _get_task_record(task_id: str) -> TaskRecord | None:
    task = _tasks.get(task_id)
    if task:
        return task
    task = load_task_snapshot(task_id)
    if task:
        _tasks[task.task_id] = task
    return task
```

代码位置：`src/api/routes/tasks.py`

它的定位是：

- 当前进程内缓存
- 低延迟支撑 SSE
- 不是 durable source of truth

### 4.2 checkpoint

```python
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

def get_langgraph_checkpointer(namespace: str = "default") -> BaseCheckpointSaver:
    backend = os.getenv("LANGGRAPH_CHECKPOINT_BACKEND", "memory").strip().lower()
    if backend in {"", "memory", "inmemory", "in_memory"}:
        key = ("memory", namespace, "")
        if key not in _CHECKPOINTERS:
            _CHECKPOINTERS[key] = MemorySaver()
        return _CHECKPOINTERS[key]

    if backend == "postgres":
        from langgraph.checkpoint.postgres import PostgresSaver
        context = PostgresSaver.from_conn_string(database_url)
```

代码位置：`src/agent/checkpointing.py`

这层的语义是：

- 图运行时状态
- 可切换 memory / postgres backend
- 符合 LangGraph 规范

### 4.3 数据库 snapshot

```python
class PersistedTask(Base):
    __tablename__ = "persisted_tasks"

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    input_type: Mapped[str] = mapped_column(String(32), nullable=False)
    input_value: Mapped[str] = mapped_column(Text, nullable=False)
    report_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    workspace_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
```

```python
def upsert_task_snapshot(task: TaskRecord) -> bool:
    if not _ensure_tables():
        return False

    payload = {
        "task_id": task.task_id,
        "status": task.status.value,
        "input_type": task.input_type,
        "input_value": task.input_value,
        "report_mode": task.report_mode,
        "source_type": task.source_type,
        "workspace_id": task.workspace_id,
        "result_markdown": task.result_markdown,
        "brief": _json_safe(task.brief),
        "search_plan": _json_safe(task.search_plan),
    }
```

代码位置：`src/db/task_persistence.py`

### 4.4 workspace artifacts

```python
def get_workspace_path(task_id: str, workspace_id: str | None = None) -> Path:
    if workspace_id:
        return get_workspace_root(workspace_id) / "tasks" / task_id
    return OUTPUT_ROOT / task_id

def write_report(task_id: str, report_markdown: str, *, workspace_id: str | None = None) -> Path:
    workspace = get_workspace_path(task_id, workspace_id=workspace_id)
    path = workspace / "report.md"
    path.write_text(report_markdown, encoding="utf-8")
```

代码位置：`src/agent/output_workspace.py`

## 5. `src/memory/manager.py` 现在应该怎么理解

这个文件还存在，但现在应当把它定义为：

- 轻量 runtime memory adapter
- 兼容层
- 进程内工作记忆与事件缓冲
- 不是 durable 主存储

它自己也在文件头写明了这一点：

```python
"""Agent memory adapter backed by LangGraph checkpoint interfaces.

The runtime-facing API is intentionally small and transient.
Earlier versions persisted semantic/episodic/preference memory as JSON files
under `.memory`; that violates the current project rules.
"""
```

代码位置：`src/memory/manager.py`

### 其中仍然有用的部分

```python
class RuntimeEventBuffer:
    MAX_EVENTS = 50

    def add(self, event_type: str, content: str | dict, metadata: dict | None = None) -> str:
        if _looks_like_secret(str(content)):
            return ""
        event_id = f"se_{self._workspace_id}_{self._event_counter}"
        self._event_counter += 1
```

```python
class RuntimeWorkingState:
    messages: list[dict] = field(default_factory=list)
    agent_state: dict = field(default_factory=dict)
    context_budget: int = 6000
    summary: str = ""
```

这些部分更像：

- 当前会话工作记忆
- prompt 注入辅助
- 运行时缓存

## 6. 当前这套状态体系的优点

- `_tasks` 让运行态交互快
- checkpoint 让图状态管理符合 LangGraph 规范
- PostgreSQL 让历史任务可恢复
- workspace 让用户能真实看到中间产物

## 7. 面试时怎么回答“memory 怎么做的”

不要直接说“我们有长期记忆系统”，推荐按下面回答：

1. 运行中任务状态放在内存 `_tasks`。
2. 图级状态由 LangGraph checkpointer 管。
3. durable snapshot 和 final report 放 PostgreSQL。
4. 用户可见工件放 workspace 文件夹。
5. `src/memory/manager.py` 目前主要是 transient runtime memory adapter，不是长期主存储。
