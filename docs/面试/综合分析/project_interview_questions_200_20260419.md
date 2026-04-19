# PaperReader Agent 项目定制面试题 200 问

> 更新时间：2026-04-19  
> 口径原则：只基于当前仓库真实实现，不沿用过时审计结论或宣传性说法。  
> 配套阅读：建议和 `docs/面试/tech_part/` 联读，尤其是 `03-工作流架构.md`、`04-多智能体协作.md`、`06-RAG检索架构.md`、`07-Grounding验证体系.md`。

## 总图

```mermaid
flowchart TD
    U[User / Frontend] --> A[/tasks]
    A --> B{source_type}
    B -->|arxiv/pdf| C[Report Graph]
    B -->|research| D[Research Graph]
    C --> E[workspace artifacts]
    D --> E
    C --> F[task snapshot + report record]
    D --> F
    E --> G[SSE report_snapshot / artifact_ready]
    F --> H[/tasks/{id} + /result]
    G --> U
    H --> U
```

## 答题模板

面试时建议统一按这四句答：

1. 先说这个模块解决什么问题。
2. 再说当前项目用了什么官方 API、组件或策略。
3. 再说仓库里现在怎么做，数据怎么流、状态怎么存。
4. 最后补代码位置、设计效果和已知边界。

## 必背代码片段

### 片段 1：任务入口

```python
@router.post("", response_model=CreateTaskResponse)
async def create_task(req: CreateTaskRequest) -> CreateTaskResponse:
    from src.agent.output_workspace import DEFAULT_WORKSPACE_USER, build_workspace_id

    source_type = req.source_type if req.source_type in {"arxiv", "pdf", "research"} else "arxiv"
    workspace_user = DEFAULT_WORKSPACE_USER or "user"
    workspace_id = (req.workspace_id or "").strip() or build_workspace_id(workspace_user)
    task = TaskRecord(
        input_type=req.input_type,
        input_value=req.input_value,
        report_mode=req.report_mode if req.report_mode in {"draft", "full"} else "draft",
        source_type=source_type,
        auto_fill=req.auto_fill,
        workspace_id=workspace_id,
    )
```

代码位置：`src/api/routes/tasks.py`

### 片段 2：Research Graph

```python
g.add_conditional_edges("clarify", _route_after_clarify, {
    "search_plan": "search_plan",
    END: END,
})
g.add_conditional_edges("search_plan", _route_after_search_plan, {
    "search": "search",
    END: END,
})
g.add_edge("search", "extract")
g.add_edge("extract", "extract_compression")
g.add_edge("extract_compression", "draft")
g.add_edge("draft", "review")
g.add_conditional_edges("review", _route_after_review, {
    "persist_artifacts": "persist_artifacts",
    END: END,
})
```

代码位置：`src/research/graph/builder.py`

### 片段 3：官方 Supervisor

```python
workflow = create_supervisor(
    worker_agents,
    model=supervisor_model,
    prompt=(
        "You supervise the research workflow. Hand off only to the next allowed "
        "stage, wait for the stage to finish, and stop when all allowed stages are done."
    ),
    output_mode="last_message",
    parallel_tool_calls=False,
)
return workflow.compile(checkpointer=get_langgraph_checkpointer("agent_supervisor"))
```

代码位置：`src/research/agents/supervisor.py`

### 片段 4：检索并发

```python
with ThreadPoolExecutor(max_workers=3) as pool:
    searxng_future = pool.submit(_run_searxng_queries, all_queries)
    arxiv_future = pool.submit(_run_arxiv_direct_search, all_queries, effective_year_filter)
    deepxiv_future = pool.submit(_run_deepxiv_queries, all_queries, effective_year_filter)

searxng_results, query_traces = searxng_future.result()
arxiv_direct_results = arxiv_future.result()
deepxiv_results = deepxiv_future.result()
```

代码位置：`src/research/graph/nodes/search.py`

---

## 一、项目定位与系统边界（Q001-Q020）

### Q001. 这个项目一句话怎么介绍最准确？
- 目标：把“研究主题或论文输入”变成一条可追踪、可恢复、可审查的报告生成链路。
- 用了什么：FastAPI 任务入口、LangGraph 双工作流、PostgreSQL 持久化、workspace 工件落盘、React 前端可视化。
- 当前怎么做：统一从 `/tasks` 创建任务，按 `source_type` 分发到 report graph 或 research graph，结果同时回写数据库和 `output/workspaces/`。
- 代码定位与效果：`src/api/routes/tasks.py`、`src/graph/builder.py`、`src/research/graph/builder.py`、`src/agent/output_workspace.py`；这样讲能把项目从“聊天机器人”拉回“研究工作流系统”。

### Q002. 这个项目解决的核心业务问题是什么？
- 目标：解决从论文/主题输入到英文研究报告输出之间的工程断层。
- 用了什么：检索、抽取、压缩、写作、验证、评测、SSE 预览和持久化。
- 当前怎么做：research 模式先澄清主题，再做检索计划、多源召回、PaperCard 抽取、上下文压缩、综述写作和 review gate。
- 代码定位与效果：`src/research/graph/nodes/*.py`、`src/research/services/*.py`；系统重点不只是“能写”，而是“写之前有证据，写之后可审查”。

### Q003. 为什么说它不是普通 QA-RAG？
- 目标：说明项目的中间对象和最终目标都不是问答式 RAG。
- 用了什么：`brief`、`search_plan`、`rag_result`、`paper_cards`、`compression_result`、`draft_report`、`review_feedback` 这些写作导向状态。
- 当前怎么做：检索结果不会直接拼成上下文回答，而是被整理成候选论文、卡片、taxonomy 和 evidence pools，再进入写作与验证。
- 代码定位与效果：`src/research/graph/nodes/search.py`、`src/research/graph/nodes/extract.py`、`src/research/graph/nodes/extract_compression.py`；这样能支撑长文综述，而不是一次性问答。

### Q004. 为什么项目必须拆成 report workflow 和 research workflow？
- 目标：解释为什么单篇论文报告和多论文综述不能共用一条图。
- 用了什么：双 `StateGraph`，分别服务 document-driven 和 topic-driven 输入。
- 当前怎么做：`arxiv/pdf` 走单篇报告链，`research` 走澄清-检索-写作链，两条图共享部分状态字段，但节点拓扑不同。
- 代码定位与效果：`src/graph/builder.py`、`src/research/graph/builder.py`、`src/api/routes/tasks.py`；拆分后每条链更清晰，前端图谱也更好解释。

### Q005. 为什么统一从 `/tasks` 进入，而不是继续强调 CLI？
- 目标：把长耗时任务统一成可追踪 API。
- 用了什么：异步任务模型、TaskRecord、SSE、结果接口和 workspace 产物。
- 当前怎么做：`create_task` 只负责接收任务与分配 `task_id/workspace_id`，真正执行在后台 `_run_graph` 中完成。
- 代码定位与效果：`src/api/routes/tasks.py`；这样前后端、数据库和工件目录都围绕 `task_id` 对齐，CLI 不再是主路径。

### Q006. 为什么任务必须异步化？
- 目标：避免检索、全文下载、LLM 抽取与写作把 HTTP 请求阻塞到超时。
- 用了什么：后台协程、任务状态、节点事件和 SSE 流。
- 当前怎么做：任务创建后立即返回 `task_id`，用户通过 `/tasks/{id}` 和 `/tasks/{id}/events` 观察执行过程。
- 代码定位与效果：`src/api/routes/tasks.py`；用户体验从“卡住等待”变成“边跑边看”，也更适合 agent 风格展示。

### Q007. 当前系统最重要的输入维度有哪些？
- 目标：讲清任务初始化时最关键的控制参数。
- 用了什么：`input_type`、`input_value`、`report_mode`、`source_type`、`workspace_id`、`auto_fill`。
- 当前怎么做：`CreateTaskRequest` 把它们标准化成任务记录，`source_type` 决定走哪条图，`report_mode` 决定 report graph 的生成分支。
- 代码定位与效果：`src/api/routes/tasks.py`；这些字段决定了后端拓扑和前端展示，不是普通表单参数。

### Q008. 为什么项目强制 PostgreSQL only？
- 目标：避免持久化实现出现开发/生产两套语义。
- 用了什么：`DATABASE_URL`、SQLAlchemy 2、任务快照与报告持久化。
- 当前怎么做：长生命周期数据通过 `src/db/task_persistence.py` 进入 PostgreSQL，SQLite 明确被 AGENTS 规则禁止。
- 代码定位与效果：`AGENTS.md`、`src/db/task_persistence.py`、`src/db/engine.py`；这样恢复、并发和部署路径一致，不会埋下 SQLite 假稳定问题。

### Q009. 为什么又要数据库，又要 `output/workspaces/`？
- 目标：区分 durable state 和 human-readable artifacts 的职责。
- 用了什么：PostgreSQL 持久快照，workspace 目录存放 json/md 中间工件。
- 当前怎么做：数据库存 `TaskRecord` 和 report 记录，workspace 存 `brief.json`、`search_plan.json`、`draft.md`、`report.md`、`revisions/*.md` 等。
- 代码定位与效果：`src/api/routes/tasks.py`、`src/db/task_persistence.py`、`src/agent/output_workspace.py`；一边保证可恢复，一边保证能直接看到过程产物。

### Q010. AGENTS 里的硬规则对系统影响最大的有哪些？
- 目标：说明仓库不是随意写，而是被架构约束收敛。
- 用了什么：PostgreSQL only、LangGraph official API、结果接口行为对齐、禁止自定义耐久 memory。
- 当前怎么做：research workflow 走 `StateGraph`，supervisor 使用官方 API，任务结果通过 `/tasks/{id}` 和 `/tasks/{id}/result` 同步暴露。
- 代码定位与效果：`AGENTS.md`、`src/research/graph/builder.py`、`src/research/agents/supervisor.py`；这些规则逼着实现往长期可维护方向收敛。

### Q011. 当前项目的主数据对象有哪些？
- 目标：让面试官知道系统不是靠一坨字符串在跑。
- 用了什么：`TaskRecord`、`ResearchBrief` 风格字典、`rag_result`、`paper_cards`、`compression_result`、`review_feedback`。
- 当前怎么做：这些对象在 state、任务快照和 workspace 文件之间来回流动，最终再汇总成 Markdown 报告。
- 代码定位与效果：`src/models/task.py`、`src/graph/state.py`、`src/api/routes/tasks.py`；对象化后才能做 review、恢复和评测。

### Q012. 当前系统对外的最终交付物是什么？
- 目标：区分机器内部状态和用户真正消费的结果。
- 用了什么：Markdown 报告、workspace 工件、任务结果 JSON、前端实时预览。
- 当前怎么做：最终交付通常是 `report.md` 或 `result_markdown`，同时附带 `draft.md`、`review_feedback.json` 等上下文工件。
- 代码定位与效果：`src/api/routes/tasks.py`、`src/agent/output_workspace.py`；用户既能看最终文稿，也能回看生成过程。

### Q013. 这套系统为什么适合做“项目展示”而不是只做“模型 Demo”？
- 目标：强调它的亮点在端到端工程而不是单个模型能力。
- 用了什么：任务系统、双工作流、SSE、workspace、quality gate、前端图视图。
- 当前怎么做：用户提交任务后可以看到节点推进、报告流式快照和最终产物，而不是只拿到一段最终文本。
- 代码定位与效果：`src/api/routes/tasks.py`、`frontend/src/components/GraphView.tsx`；展示效果完整，也更接近真实应用。

### Q014. 当前项目的主运行路径是什么？
- 目标：说明什么是“主链”，什么只是兼容层或旁路。
- 用了什么：`/tasks` API、report graph、research graph、workspace artifacts。
- 当前怎么做：新的任务流程以 API 为核心，CLI 不是主入口；research workflow 的主叙述是 official supervisor + staged workers。
- 代码定位与效果：`src/api/routes/tasks.py`、`src/research/agents/supervisor.py`；讲主链时要避免把历史遗留路径当成现状。

### Q015. 为什么这个项目比普通 Agent Demo 更重“中间状态”？
- 目标：解释为什么仓库里有这么多 json 和中间字段。
- 用了什么：状态图、workspace 工件、评测指标、节点事件。
- 当前怎么做：每个关键阶段都会产出结构化状态，部分还会实时落盘并通过 SSE 暴露给前端。
- 代码定位与效果：`src/api/routes/tasks.py`、`src/agent/output_workspace.py`；中间状态多的代价是复杂，收益是可调试、可评测、可演示。

### Q016. 当前项目最重要的非功能性需求是什么？
- 目标：补充性能之外的工程要求。
- 用了什么：可恢复、可观测、可验证、可扩展。
- 当前怎么做：通过 task snapshot、workspace artifacts、review gate、SSE 事件和模块化 graph 节点来满足。
- 代码定位与效果：`src/db/task_persistence.py`、`src/graph/callbacks.py`、`src/research/services/reviewer.py`；这使它更像生产系统雏形，而不是实验脚本。

### Q017. 为什么 workspace-first 对这个项目尤其重要？
- 目标：解释为什么要把过程产物显式写到磁盘。
- 用了什么：workspace 用户 ID、task 目录、artifact 文件命名和 revisions。
- 当前怎么做：任务运行时同步写入 `brief.json`、`paper_cards.json`、`draft.md`、`report.md` 等文件，SSE 直接读取这些文件变化。
- 代码定位与效果：`src/agent/output_workspace.py`、`src/api/routes/tasks.py`；这样即使接口不看日志，也能直接回放工作流。

### Q018. 当前项目里“多智能体”应该怎么定义？
- 目标：避免把它说成完全自由自治的 agent 社会。
- 用了什么：官方 `create_supervisor`、`create_react_agent` 和 canonical stage order。
- 当前怎么做：多 agent 协作严格围绕 `search_plan -> search -> draft -> review` 等阶段展开，不允许任意游走。
- 代码定位与效果：`src/research/agents/supervisor.py`；这种 staged multi-agent 更稳，也更符合工程要求。

### Q019. 当前系统最大的架构优势是什么？
- 目标：给面试中的总结句一个高质量落点。
- 用了什么：双图分治、显式状态、官方 LangGraph 编排、review gate、workspace 可视化。
- 当前怎么做：把“研究报告生成”拆成多个可解释阶段，每阶段都有输入、输出、落盘和部分评测。
- 代码定位与效果：`src/research/graph/builder.py`、`src/research/agents/supervisor.py`、`src/api/routes/tasks.py`；优势在于链路清晰，便于定位问题和持续优化。

### Q020. 当前系统最大的架构风险是什么？
- 目标：展示你既能讲优点，也能讲边界。
- 用了什么：长链路、多状态对象、多种兼容层和外部搜索源。
- 当前怎么做：虽然主路径已经清晰，但 supervisor 仍有迁移兼容层，skills 与主动聊天还在继续收敛，长文写作与引用分配仍是重点优化对象。
- 代码定位与效果：`src/research/agents/supervisor.py`、`src/skills/orchestrator.py`、`src/research/graph/nodes/draft.py`；这类回答能体现你对现状不是盲目乐观。

## 二、API、TaskRecord 与任务生命周期（Q021-Q040）

### Q021. `create_task` 真正负责什么？
- 目标：解释入口接口不是“直接跑模型”。
- 用了什么：请求校验、参数标准化、`TaskRecord` 创建、`workspace_id` 分配、快照写入和异步调度。
- 当前怎么做：接口收到请求后生成任务记录，先进入内存与数据库，再启动后台执行，不在请求线程里跑完整工作流。
- 代码定位与效果：`src/api/routes/tasks.py`；这样任务提交和任务执行被明确解耦。

### Q022. `CreateTaskRequest` 为什么值得单独讲？
- 目标：说明工作流分流入口完全由这个模型控制。
- 用了什么：Pydantic v2 的字段约束和默认值。
- 当前怎么做：`input_type/input_value` 描述输入，`source_type` 决定 graph，`report_mode` 决定 report 生成路径，`auto_fill` 决定 clarify 是否允许自动补全。
- 代码定位与效果：`src/api/routes/tasks.py`；这让接口语义足够明确，也减少前端猜测。

### Q023. 为什么 `workspace_id` 在任务创建时就生成？
- 目标：给中间工件和 SSE 监听一个稳定容器。
- 用了什么：`build_workspace_id()` 和任务级路径约定。
- 当前怎么做：如果请求没有传 `workspace_id`，系统立即生成新的 workspace；如果传了，就把新任务挂到既有 workspace 下。
- 代码定位与效果：`src/api/routes/tasks.py`、`src/agent/output_workspace.py`；这使“回到旧 workspace 继续看结果”成为可能。

### Q024. 为什么既保留 `_tasks` 内存表，又保留数据库快照？
- 目标：区分当前进程缓存和 durable source of truth。
- 用了什么：内存 dict、`load_task_snapshot()`、`upsert_task_snapshot()`。
- 当前怎么做：详情查询先看 `_tasks`，没有再从数据库恢复，恢复后会重新放回 `_tasks`。
- 代码定位与效果：`src/api/routes/tasks.py`、`src/db/task_persistence.py`；这样既保留低延迟，又支持进程重启后恢复查看。

### Q025. `TaskRecord` 的核心价值是什么？
- 目标：说明任务生命周期为什么能被统一管理。
- 用了什么：状态字段、时间戳、markdown 字段、中间状态字段、持久化状态和错误信息。
- 当前怎么做：无论 report 还是 research，最终都会被回填到 `TaskRecord`，前端接口和数据库都围绕它序列化。
- 代码定位与效果：`src/models/task.py`、`src/api/routes/tasks.py`；它是 API、DB、workspace 三者对齐的桥梁。

### Q026. 当前任务状态怎么流转？
- 目标：讲清不是只有“开始/结束”两个状态。
- 用了什么：`TaskStatus`、`current_stage`、节点事件和 completed/error 回写。
- 当前怎么做：任务从创建进入运行态，运行过程中阶段不断更新，最终要么完成并记录完成时间，要么失败并记录错误。
- 代码定位与效果：`src/models/task.py`、`src/api/routes/tasks.py`；这样前端可以展示运行中的 agent，而不是只能等最终结果。

### Q027. 为什么 `_task_payload` 很重要？
- 目标：解释为什么详情接口能反映这么多内部状态。
- 用了什么：统一任务序列化函数。
- 当前怎么做：它把 markdown、brief、search_plan、rag_result、paper_cards、review_feedback、persistence 状态等一起打平输出。
- 代码定位与效果：`src/api/routes/tasks.py`；只要这个函数维护好，`/tasks/{id}` 的语义就会稳定。

### Q028. 为什么 `/tasks/{id}` 和 `/tasks/{id}/result` 必须保持行为对齐？
- 目标：避免详情和结果接口互相打架。
- 用了什么：统一回填 `TaskRecord`，再从同一对象中投影不同视图。
- 当前怎么做：详情接口偏向“运行态 + 中间状态”，结果接口偏向“最终文稿 + 摘要状态”，但底层数据源是一致的。
- 代码定位与效果：`AGENTS.md`、`src/api/routes/tasks.py`；这避免了前端看到“任务完成但结果空白”的不一致体验。

### Q029. 为什么结果接口同时返回多个 Markdown 字段？
- 目标：处理双工作流和 full/draft 模式差异。
- 用了什么：`draft_markdown`、`full_markdown`、`result_markdown`、`report_context_snapshot`。
- 当前怎么做：research 常以 `result_markdown` 为主，report workflow 则可能根据模式使用 `draft/full` 的不同文本。
- 代码定位与效果：`src/api/routes/tasks.py`；字段看起来多，但能让前端和测试明确知道拿到的是哪一版文本。

### Q030. `/tasks/{id}/chat` 为什么只服务已完成任务？
- 目标：说明当前聊天不是 workflow steering，而是 post-report QA。
- 用了什么：状态检查、报告快照上下文、聊天历史落入任务对象。
- 当前怎么做：接口先确认任务完成，再读取 `report_context_snapshot` 或最终 markdown 作为对话上下文。
- 代码定位与效果：`src/api/routes/tasks.py`；这条路稳定，但也说明运行中交互和主动 skills 还不是主能力。

### Q031. `report_context_snapshot` 的意义是什么？
- 目标：把聊天上下文冻结成可复用对象，而不是实时拼装大量状态。
- 用了什么：任务完成后保存一份最终对话上下文快照。
- 当前怎么做：research task 会把最终报告文本写入 snapshot，后续聊天都围绕这份文本继续。
- 代码定位与效果：`src/api/routes/tasks.py`；聊天层不需要理解整个工作流细节，复杂度更可控。

### Q032. `/tasks/{id}/events` 的 SSE 事件为什么有两路来源？
- 目标：说明“阶段事件”和“文稿流”不是同一类信息。
- 用了什么：`task.node_events` 和 workspace 文件扫描。
- 当前怎么做：节点事件负责 `node_start/node_end/status_change`，文件扫描负责 `artifact_ready/report_snapshot`。
- 代码定位与效果：`src/api/routes/tasks.py`；用户既能看流程推进，也能看报告正文的实时变化。

### Q033. `_STREAMABLE_TASK_FILES` 的作用是什么？
- 目标：限定哪些工件会进入实时流。
- 用了什么：固定白名单文件集合。
- 当前怎么做：`brief.json`、`search_plan.json`、`rag_result.json`、`paper_cards.json`、`draft.md`、`report.md` 等文件一旦变化就被 SSE 轮询到并发给前端。
- 代码定位与效果：`src/api/routes/tasks.py`；这样前端拿到的是高价值工件，而不是整个目录的噪音。

### Q034. `_workspace_stream_events` 是怎么工作的？
- 目标：解释 live preview 背后不是魔法，而是文件变化扫描。
- 用了什么：按文件名白名单枚举、`mtime_ns` 去重、Markdown 和 JSON 分流。
- 当前怎么做：Markdown 文件会被读成 `report_snapshot` 事件，JSON 文件则被压成摘要形式的 `artifact_ready` 事件。
- 代码定位与效果：`src/api/routes/tasks.py`；这是当前报告流式预览的后端基础设施。

### Q035. 为什么 `report.md` 能作为“最终完成信号”？
- 目标：解释 SSE 里 `is_final` 的来源。
- 用了什么：文件名约定和 `path.name == "report.md"` 判断。
- 当前怎么做：当流式扫描到 `report.md` 时，SSE 事件会标记 `is_final=true`，前端可以据此切换为完成态。
- 代码定位与效果：`src/api/routes/tasks.py`；这个约定简单但有效，避免再单独做一套完成流格式。

### Q036. 节点事件是怎么收集的？
- 目标：说明 graph 过程为什么能被时间线展示。
- 用了什么：`NodeEventEmitter` 和 `instrument_node(...)`。
- 当前怎么做：图中的每个节点都会被包装，执行前后向 emitter 写事件，最终存进 `task.node_events`。
- 代码定位与效果：`src/graph/callbacks.py`、`src/graph/instrumentation.py`、`src/graph/builder.py`；这为时间线、调试和演示都提供了基础。

### Q037. 为什么 `current_stage` 要被显式写进状态？
- 目标：给前端、API、日志一个统一的进度指针。
- 用了什么：节点包装器在结果里补充 `current_stage`。
- 当前怎么做：research graph 的 `_with_current_stage()` 会在节点输出中写回阶段名，任务详情接口再暴露出来。
- 代码定位与效果：`src/research/graph/builder.py`、`src/api/routes/tasks.py`；这样不同层对“当前跑到哪里”不会各说各话。

### Q038. `awaiting_followup` 在 API 层有什么意义？
- 目标：把 clarify 的“问题还不清楚”显式变成任务状态。
- 用了什么：任务状态字段、follow-up 提示和 research graph 的早停路由。
- 当前怎么做：如果 clarify 判定仍需追问，任务不会继续检索，接口会把这一状态返给前端。
- 代码定位与效果：`src/research/graph/nodes/clarify.py`、`src/research/graph/builder.py`、`src/api/routes/tasks.py`；这能减少离题召回。

### Q039. 任务如何挂到旧 workspace 上？
- 目标：说明系统不是“一次任务一个完全隔离会话”。
- 用了什么：`CreateTaskRequest.workspace_id`。
- 当前怎么做：前端或调用方传入既有 `workspace_id` 后，新的任务工件会写进同一 workspace 的新 task 目录。
- 代码定位与效果：`src/api/routes/tasks.py`、`src/agent/output_workspace.py`；这支持连续研究、迭代优化和结果对比。

### Q040. 系统重启后如何查看旧任务？
- 目标：说明结果不是只活在内存里。
- 用了什么：`load_task_snapshot()` 和数据库持久快照。
- 当前怎么做：当内存 `_tasks` 里找不到任务时，接口会回退到 PostgreSQL 加载快照，再恢复成 `TaskRecord`。
- 代码定位与效果：`src/api/routes/tasks.py`、`src/db/task_persistence.py`；这让任务结果具备真正的 durable viewing 能力。

## 三、Report Workflow 单篇论文链路（Q041-Q060）

### Q041. Report workflow 的完整节点顺序是什么？
- 目标：给单篇报告链路一个完整口径。
- 用了什么：`input_parse -> ingest_source -> extract_document_text -> normalize_metadata -> retrieve_evidence -> classify_paper_type -> draft/report_frame/survey_intro_outline -> repair_report -> resolve_citations -> verify_claims -> apply_policy -> format_output`。
- 当前怎么做：图里通过条件边在 metadata 校验和生成模式上做分支，最后统一走验证和格式化。
- 代码定位与效果：`src/graph/builder.py`；这条链更适合单篇论文精读和标准化成文。

### Q042. `input_parse` 解决什么问题？
- 目标：先把原始输入变成系统能理解的结构。
- 用了什么：输入类型识别、文本或 arXiv 链接标准化。
- 当前怎么做：无论用户给的是 arXiv 标识、URL 还是文本，都会在图开头统一解析。
- 代码定位与效果：`src/graph/nodes/input_parse.py`、`src/graph/builder.py`；这一步越稳，后面 ingest 和 metadata 失败越少。

### Q043. `ingest_source` 负责什么？
- 目标：把外部输入拉进系统内部处理上下文。
- 用了什么：文档抓取、原始内容读取与初步结构化。
- 当前怎么做：对于论文链接或 PDF，它负责完成后续文本抽取所需的基础准备。
- 代码定位与效果：`src/graph/nodes/ingest_source.py`、`src/graph/builder.py`；这是 report workflow 的文档入口。

### Q044. `extract_document_text` 为什么要单独成节点？
- 目标：把文档解析从输入接收逻辑里解耦出来。
- 用了什么：正文抽取、文本清洗和下游可消费文本结构。
- 当前怎么做：抽出的正文会成为 metadata 识别、证据检索和写作的基础。
- 代码定位与效果：`src/graph/nodes/extract_document_text.py`、`src/graph/builder.py`；单独成节点后更便于替换 PDF/HTML 解析策略。

### Q045. `normalize_metadata` 的意义是什么？
- 目标：在进入生成前把论文元信息变规范。
- 用了什么：标题、作者、年份、来源、paper type 等字段归一化。
- 当前怎么做：如果 metadata 不足或者输入不合法，还可能触发 safe abort 路径而不是盲目继续。
- 代码定位与效果：`src/graph/nodes/normalize_metadata.py`、`src/graph/builder.py`；这一步直接影响后续引用和报告头部质量。

### Q046. 为什么 report graph 里要有 safe abort？
- 目标：说明系统不会对明显无效输入硬写一篇文稿。
- 用了什么：`_should_abort` 条件边和 `format_output` 兜底。
- 当前怎么做：一旦 metadata 或上游检查认为任务不适合继续，会直接跳到最终格式化输出错误或说明信息。
- 代码定位与效果：`src/graph/builder.py`；这能减少空报告、乱引用和模型胡编。

### Q047. `retrieve_evidence` 在单篇报告里扮演什么角色？
- 目标：说明即使是单篇报告，也不是只靠原文直接复述。
- 用了什么：证据检索、上下文组织和后续写作输入准备。
- 当前怎么做：在 metadata 正常后，先把与论文相关的证据组织好，再交给生成节点。
- 代码定位与效果：`src/graph/nodes/retrieve_evidence.py`、`src/graph/builder.py`；这让报告更像“基于证据的解释”，而不只是摘要改写。

### Q048. 为什么还要 `classify_paper_type`？
- 目标：因为不同论文类型对应不同生成路径。
- 用了什么：类型分类 + 条件路由。
- 当前怎么做：分类结果会决定走 `draft_report`、`report_frame` 还是 `survey_intro_outline` 等不同生成方式。
- 代码定位与效果：`src/graph/nodes/classify_paper_type.py`、`src/graph/builder.py`；通过分类，单篇报告和 survey 风格 full mode 能共存。

### Q049. `draft_report` 适合什么场景？
- 目标：快速得到一版单篇论文草稿。
- 用了什么：直接草拟、后续 repair、引用解析和 claim verification。
- 当前怎么做：当类型和模式选择到草稿路径时，系统先生成初稿，再进入修复和验证链。
- 代码定位与效果：`src/graph/nodes/draft_report.py`、`src/graph/builder.py`；这是 report workflow 的快速路径。

### Q050. `report_frame` 是干什么的？
- 目标：为 full 模式提供更强的结构化框架。
- 用了什么：分章节框架、报告结构约束和后续 citation resolving。
- 当前怎么做：常规 full 报告不直接走草稿，而是先通过 frame 构建更稳定的成文骨架。
- 代码定位与效果：`src/agent/report_frame.py`、`src/graph/builder.py`；框架先行有助于减少长文结构漂移。

### Q051. `survey_intro_outline` 为什么存在？
- 目标：支持 survey/full 模式下更偏综述的开篇结构。
- 用了什么：综述导向的引言与大纲生成。
- 当前怎么做：当 paper type 更偏 survey 且是 full 模式时，会走这一分支，而不是普通 report frame。
- 代码定位与效果：`src/graph/nodes/survey_intro_outline.py`、`src/graph/builder.py`；这样单篇综述类论文的生成风格更合理。

### Q052. `repair_report` 的职责是什么？
- 目标：在初稿之后补一次结构和表达修复。
- 用了什么：修补节点、后处理和局部纠偏。
- 当前怎么做：`draft_report` 生成后不会直接交付，而是先进入 repair，再进入 citation 解析和 claim verification。
- 代码定位与效果：`src/graph/nodes/repair_report.py`；它是 report workflow 里重要的质量缓冲层。

### Q053. `resolve_citations` 为什么在 report workflow 里很关键？
- 目标：把生成文本中的引用占位或粗糙引用转成更规范的引用结构。
- 用了什么：引用解析与报告回写。
- 当前怎么做：无论从 `repair_report` 还是 `report_frame/survey_intro_outline` 过来，最终都统一进入 `resolve_citations`。
- 代码定位与效果：`src/graph/nodes/resolve_citations.py`、`src/graph/builder.py`；这样不同生成分支的引用策略最终能收敛。

### Q054. `verify_claims` 在 report graph 里做什么？
- 目标：防止文稿里出现明显无依据的结论。
- 用了什么：claim verification 节点和报告校验输出。
- 当前怎么做：引用解析之后，系统会再次检查关键 claim 是否有证据支撑，再把结果交给 policy 层。
- 代码定位与效果：`src/graph/nodes/verify_claims.py`；这一步是把“能写”往“可信”方向再推进一层。

### Q055. `apply_policy` 有什么意义？
- 目标：把策略约束集中在专门节点，而不是散落在 prompt 各处。
- 用了什么：规则判断、结果状态调整和格式输出前的最终整理。
- 当前怎么做：claim verification 之后的结果先经过 policy，再进入最终格式化输出。
- 代码定位与效果：`src/graph/nodes/apply_policy.py`、`src/graph/builder.py`；策略层独立后，更容易迭代规则而不破坏主图。

### Q056. `format_output` 为什么是统一终点？
- 目标：给所有 report 路径一个一致的交付收口。
- 用了什么：Markdown 组装、结果字段整理和最终输出封装。
- 当前怎么做：不管上游是正常完成还是 safe abort，都会通过 `format_output` 形成接口与持久化层可消费的结构。
- 代码定位与效果：`src/graph/nodes/format_output.py`；这避免了多分支最终输出格式各不相同。

### Q057. report workflow 为什么也需要 checkpointer？
- 目标：让单篇报告链也能享受 LangGraph 的状态管理能力。
- 用了什么：`get_langgraph_checkpointer()` 和 `use_checkpointer=True`。
- 当前怎么做：构建 report graph 时可以挂上 checkpointer，和 research graph 保持一致的演进方向。
- 代码定位与效果：`src/graph/builder.py`、`src/agent/checkpointing.py`；这为未来恢复和更强可观测打基础。

### Q058. report graph 中 `instrument_node(...)` 的价值是什么？
- 目标：让 report workflow 也能产生统一的节点事件。
- 用了什么：节点包装、事件发射器和统一观测接口。
- 当前怎么做：几乎每个 report graph 节点都通过 `instrument_node` 注册到事件体系中。
- 代码定位与效果：`src/graph/builder.py`、`src/graph/instrumentation.py`；这保证前端时间线不只对 research graph 有效。

### Q059. full 模式和 draft 模式的最大区别是什么？
- 目标：区分“快速草稿”和“结构化成文”两类输出。
- 用了什么：`report_mode`、不同生成分支和不同 markdown 字段。
- 当前怎么做：draft 更偏快速生成再修补，full 更偏 frame/outline 驱动的结构化输出。
- 代码定位与效果：`src/api/routes/tasks.py`、`src/graph/builder.py`；答题时要说明这不是单纯的输出长度差异。

### Q060. report workflow 当前最大的工程边界是什么？
- 目标：展示你能看见单篇报告链还未完全解决的问题。
- 用了什么：引用解析、claim verification、policy 与格式化这些后处理层。
- 当前怎么做：虽然链路已经完整，但最终质量仍受上游证据质量、引用分布和长文结构控制影响。
- 代码定位与效果：`src/graph/nodes/resolve_citations.py`、`src/graph/nodes/verify_claims.py`；这类回答能顺势引到 research workflow 的更强质量控制。

## 四、Research Workflow 与 LangGraph 编排（Q061-Q080）

### Q061. Research workflow 的标准节点顺序是什么？
- 目标：给多论文综述主链一个准确口径。
- 用了什么：`clarify -> search_plan -> search -> extract -> extract_compression -> draft -> review -> persist_artifacts`。
- 当前怎么做：这条链被显式声明在 `StateGraph` 中，不依赖 Python for/while 模拟。
- 代码定位与效果：`src/research/graph/builder.py`；这符合 AGENTS 对 LangGraph 实现的硬要求。

### Q062. 为什么 research workflow 要先做 `clarify`？
- 目标：先把模糊话题变成结构化 brief，再做检索。
- 用了什么：clarify agent/node、follow-up 判断、置信度和自动补全控制。
- 当前怎么做：如果 brief 不充分，图会直接结束或等待追问，而不是盲目进入搜索。
- 代码定位与效果：`src/research/graph/nodes/clarify.py`、`src/research/graph/builder.py`；这一步直接影响 off-topic 比例。

### Q063. `clarify` 节点输出的核心对象是什么？
- 目标：说明 research graph 的真正输入不是原始字符串，而是结构化 brief。
- 用了什么：topic、sub-questions、scope、constraints、confidence、needs_followup 等信息。
- 当前怎么做：后续 `search_plan` 和 `search` 主要围绕 brief 展开，而不再直接解释用户原始输入。
- 代码定位与效果：`src/research/graph/nodes/clarify.py`、`src/research/research_brief.py`；brief 是 research workflow 的第一层抽象。

### Q064. `awaiting_followup` 为真时，图为什么直接结束？
- 目标：控制错误输入不要污染后续整条链。
- 用了什么：`_route_after_clarify()` 条件路由。
- 当前怎么做：如果没有 brief，或者 still needs followup，路由函数直接返回 `END`。
- 代码定位与效果：`src/research/graph/builder.py`；这让 early stop 成为图级行为，而不是节点内部偷偷 return。

### Q065. `auto_fill` 在 research workflow 里有什么作用？
- 目标：平衡人工确认和自动推进两种体验。
- 用了什么：任务创建参数和 clarify 节点中的 follow-up 处理逻辑。
- 当前怎么做：开启 `auto_fill` 时，即便 clarify 觉得信息不足，也会尽量自动补齐而不是停下等用户。
- 代码定位与效果：`src/api/routes/tasks.py`、`src/research/graph/nodes/clarify.py`；适合演示和全自动测试，但会带来范围漂移风险。

### Q066. `search_plan` 节点解决什么问题？
- 目标：把 brief 转成检索可执行计划。
- 用了什么：query groups、plan goal、time range、filters 和 fallback heuristic。
- 当前怎么做：系统先把研究问题拆成多个查询组，再进入并发检索，而不是直接拿 topic 做单次查询。
- 代码定位与效果：`src/research/graph/nodes/search_plan.py`；查询规划越清晰，后面的召回越稳定。

### Q067. 为什么 `search_plan` 支持 heuristic fallback？
- 目标：防止 planner 异常时整条研究链直接断掉。
- 用了什么：`to_fallback_plan()` 和低置信度/needs_followup 快速路径。
- 当前怎么做：当 brief 置信度低、需要追问或者显式要求 heuristic 时，节点会退到规则化 search plan。
- 代码定位与效果：`src/research/graph/nodes/search_plan.py`；这提升了鲁棒性，但效果通常不如高质量 planner。

### Q068. `research_depth` 为什么重要？
- 目标：让 research workflow 支持 plan-only 和 full 两种运行深度。
- 用了什么：`_route_after_search_plan()` 条件判断。
- 当前怎么做：如果 `research_depth != "full"`，图在 search_plan 后就结束，只返回规划结果。
- 代码定位与效果：`src/research/graph/builder.py`；这对调试、演示和澄清阶段都很有用。

### Q069. 为什么 research graph 是“显式线性图”而不是复杂 DAG？
- 目标：强调当前实现追求可解释和稳定，而不是拓扑炫技。
- 用了什么：线性阶段图 + 节点内部并发。
- 当前怎么做：图本身阶段顺序固定，并发主要藏在 `search` 和 `extract` 这类节点内部。
- 代码定位与效果：`src/research/graph/builder.py`、`src/research/graph/nodes/search.py`；这种设计更利于调试和业务收敛。

### Q070. `_with_current_stage()` 有什么工程价值？
- 目标：让每个节点天然回写统一的阶段名。
- 用了什么：节点函数包装器。
- 当前怎么做：包装器在节点输出是 dict 时自动补充 `current_stage`，后端详情接口和前端都能直接使用。
- 代码定位与效果：`src/research/graph/builder.py`；这是一个小设计，但对可观测性非常关键。

### Q071. 为什么 `extract_compression` 要单独成节点？
- 目标：把“抽取完成”和“压缩整理完成”分成两个不同里程碑。
- 用了什么：独立压缩节点、独立工件、独立评估对象。
- 当前怎么做：先产出 `paper_cards`，再压成 `taxonomy/compressed_cards/evidence_pools`，最后才交给 draft。
- 代码定位与效果：`src/research/graph/nodes/extract.py`、`src/research/graph/nodes/extract_compression.py`；这样压缩效果可观测、可替换、可测试。

### Q072. 为什么 `review` 是真正的 gate，而不是装饰节点？
- 目标：说明 review 结果会影响图是否继续。
- 用了什么：`_route_after_review()` 和 `review_passed` 布尔值。
- 当前怎么做：只有 `review_passed=True` 才会进入 `persist_artifacts`，否则直接结束在未通过状态。
- 代码定位与效果：`src/research/graph/builder.py`、`src/research/graph/nodes/review.py`；这保证 review 对最终成文有真实约束。

### Q073. `persist_artifacts` 为什么放在最后？
- 目标：把正式通过审查的结果和中间失败结果区分开。
- 用了什么：持久化节点、workspace 工件收口和最终报告写入。
- 当前怎么做：review 通过后再集中写最终产物，而 review 未通过时更多保留为 inspectable draft/revision。
- 代码定位与效果：`src/research/graph/nodes/persist_artifacts.py`；这样最终产物的语义更干净。

### Q074. Research graph 怎么接入 checkpointer？
- 目标：符合 LangGraph 的官方状态持久化接口。
- 用了什么：`get_langgraph_checkpointer("research_graph")`。
- 当前怎么做：`build_research_graph(..., use_checkpointer=True)` 时，编译图会挂上 saver；当前环境可以是 `MemorySaver` 或 `PostgresSaver`。
- 代码定位与效果：`src/research/graph/builder.py`、`src/agent/checkpointing.py`；这为未来更强恢复能力留下了官方接口。

### Q075. `AgentState` 在 research workflow 里扮演什么角色？
- 目标：给 LangGraph 一个统一状态模式。
- 用了什么：共享状态类型和任务层的 state template。
- 当前怎么做：API 层先构造包含 brief、paper_cards、draft_report、review_feedback 等字段的初始 state，再交给图执行。
- 代码定位与效果：`src/graph/state.py`、`src/api/routes/tasks.py`；这样节点之间交换的是结构化状态，不是一串 prompt。

### Q076. API 层是怎么把 research 任务分发给 graph 的？
- 目标：讲清从 HTTP 到 LangGraph 的桥接流程。
- 用了什么：`source_type == "research"` 分支、Supervisor、初始 state、事件发射器。
- 当前怎么做：`_run_graph` 中识别 research 任务后，构造 state、创建 supervisor 或 research graph，并在后台执行。
- 代码定位与效果：`src/api/routes/tasks.py`；这让 report 和 research 虽然图不同，但任务入口保持一致。

### Q077. review 失败时系统会丢掉草稿吗？
- 目标：说明失败并不等于什么都看不到。
- 用了什么：任务状态保留、revision 文件和未通过状态。
- 当前怎么做：即便不进入 `persist_artifacts`，`draft_markdown`、`review_feedback` 和部分 workspace 产物仍然可见。
- 代码定位与效果：`src/research/agents/supervisor.py`、`src/agent/output_workspace.py`；这非常适合二次迭代和人工复盘。

### Q078. 为什么 research workflow 强调“inspectable in task state”？
- 目标：避免用户只看到“failed”而不知道为什么失败。
- 用了什么：任务详情字段、review feedback、collaboration trace 和 workspace 文件。
- 当前怎么做：图停止后，接口仍然会回传已有的 brief、plan、rag_result、draft 和 review 信息。
- 代码定位与效果：`src/api/routes/tasks.py`；这个设计对调参、调 prompt 和调检索都很关键。

### Q079. plan-only 模式适合哪些场景？
- 目标：展示工作流不仅能产文，也能作为研究规划器。
- 用了什么：`research_depth=plan` 和 search_plan 早停。
- 当前怎么做：系统在澄清后生成检索计划并直接结束，可用于调试 query groups、向用户展示范围或做快速预案。
- 代码定位与效果：`src/research/graph/builder.py`、`src/research/graph/nodes/search_plan.py`；这比每次都跑全链更节省资源。

### Q080. Research workflow 当前最容易出问题的三个点是什么？
- 目标：体现你真的理解长链路中的脆弱点。
- 用了什么：clarify 范围漂移、search quality、review/grounding 不通过。
- 当前怎么做：系统已经用 follow-up、strict rerank、fulltext ingest 和 review gate 做了缓解，但仍需要在引用分布和长文写作上继续优化。
- 代码定位与效果：`src/research/graph/nodes/clarify.py`、`src/research/graph/nodes/search.py`、`src/research/graph/nodes/review.py`；这类回答能自然引到后面的 RAG 与质量体系。

## 五、多智能体协作与 Supervisor（Q081-Q100）

### Q081. 当前多智能体协作最准确的表述是什么？
- 目标：防止面试时把系统说成完全自治 agent society。
- 用了什么：官方 supervisor + staged workers + canonical node order。
- 当前怎么做：多 agent 协作围绕固定 research workflow 阶段推进，每个 worker 只负责一个阶段。
- 代码定位与效果：`src/research/agents/supervisor.py`；这是工程化 staged multi-agent，不是开放式自由对话。

### Q082. 为什么要用 `create_supervisor(...)`？
- 目标：满足 AGENTS 对“使用官方编排 API”的要求。
- 用了什么：`langgraph_supervisor.create_supervisor`。
- 当前怎么做：worker agents 会被组合进 supervisor graph，由 supervisor 决定 handoff 顺序并最终 compile。
- 代码定位与效果：`src/research/agents/supervisor.py`；这样底层协作逻辑不需要再手写 dispatch loop。

### Q083. `create_react_agent(...)` 在这里具体负责什么？
- 目标：说明 worker agent 不是自写 loop，而是官方构造器产物。
- 用了什么：单阶段工具 + 受限 prompt + `create_react_agent`。
- 当前怎么做：每个 worker 都只绑定自己的单一 stage tool，被要求调用一次后结束。
- 代码定位与效果：`src/research/agents/supervisor.py`；worker 权限非常明确，减少了跨阶段串台。

### Q084. 为什么还保留 `AgentSupervisor` facade？
- 目标：解释为什么仓库里还有一个 supervisor 类壳子。
- 用了什么：对外稳定接口、内部 backend 切换和 workspace 同步。
- 当前怎么做：API 层仍调用 `AgentSupervisor`，但它的核心协作路径已经委托给官方 supervisor API。
- 代码定位与效果：`src/research/agents/supervisor.py`；这是迁移期的 adapter，而不是旧逻辑主导。

### Q085. `CANONICAL_NODE_ORDER` 的作用是什么？
- 目标：为多 agent 协作定义清晰阶段秩序。
- 用了什么：固定节点序列元组。
- 当前怎么做：supervisor 只会在允许的下一个阶段之间 handoff，不会任意跳跃。
- 代码定位与效果：`src/research/agents/supervisor.py`；这让多 agent 仍然服从研究工作流的工程边界。

### Q086. 为什么仓库里还有 `LEGACY_NODE_TARGETS`？
- 目标：说明迁移并不是“一刀切”。
- 用了什么：旧 graph node 映射表。
- 当前怎么做：当某个阶段没有 v2 agent，或者配置要求 legacy 时，supervisor 会直接调用原有节点实现。
- 代码定位与效果：`src/research/agents/supervisor.py`；这提供了灰度迁移能力。

### Q087. `V2_AGENT_TARGETS` 代表什么？
- 目标：指出哪些阶段已经有更 agent 化的实现。
- 用了什么：模块路径、函数名和 agent paradigm 元数据。
- 当前怎么做：目前像 `search_plan`、`search`、`draft`、`review` 这些阶段都可以映射到 v2 backend。
- 代码定位与效果：`src/research/agents/supervisor.py`；这说明多 agent 迁移已经不是概念，而是实代码落地。

### Q088. `NodeBackendMode` 有什么意义？
- 目标：允许在 legacy、v2、auto 之间做阶段级选择。
- 用了什么：配置驱动的 backend mode。
- 当前怎么做：`_get_backend_mode()` 会先看节点级配置，再看全局执行模式，最后决定用 legacy 还是 v2。
- 代码定位与效果：`src/research/agents/supervisor.py`、`src/models/config.py`；这让迁移与回滚都更可控。

### Q089. `AUTO` 模式下 supervisor 是怎么回退的？
- 目标：体现多 agent 体系有鲁棒性设计。
- 用了什么：先尝试 `_run_v2()`，失败后 fallback 到 `_run_legacy()`。
- 当前怎么做：如果 v2 agent 调用异常，系统会记录 warning，再回退到 legacy node，保证任务不断链。
- 代码定位与效果：`src/research/agents/supervisor.py`；这就是“兼容层不是摆设”的具体体现。

### Q090. `_BoundToolCallingModel` 为什么存在？
- 目标：给 worker 一个确定性执行模型，避免无关自由发挥。
- 用了什么：最小化 deterministic chat model，只调用已绑定工具一次。
- 当前怎么做：worker 看到消息时，要么发起一次 tool call，要么在工具已执行后返回完成消息。
- 代码定位与效果：`src/research/agents/supervisor.py`；这让 worker 行为非常可控，便于测试和回放。

### Q091. `_PlannedHandoffModel` 的作用是什么？
- 目标：为 supervisor 侧的 handoff 提供可控、可测的模型行为。
- 用了什么：基于 `FakeMessagesListChatModel` 的确定性 handoff 模型。
- 当前怎么做：在需要计划化 handoff 时，supervisor 可以使用更可控的消息序列，而不是完全依赖随机模型响应。
- 代码定位与效果：`src/research/agents/supervisor.py`；这让多 agent 测试更稳。

### Q092. `build_official_supervisor_graph()` 的核心思想是什么？
- 目标：把多个阶段 worker 编进一个官方 supervisor graph。
- 用了什么：worker agents 列表、supervisor model、prompt 和 checkpointer。
- 当前怎么做：系统根据 planned nodes 生成 worker，再交给 `create_supervisor(...)` 编译成可执行图。
- 代码定位与效果：`src/research/agents/supervisor.py`；这使多 agent 协作本身也成为 LangGraph 图。

### Q093. 为什么 worker prompt 要写 “only your single tool exactly once”？
- 目标：严格控制 worker 职责边界。
- 用了什么：受限 prompt 和单工具绑定。
- 当前怎么做：每个阶段 worker 被限定只能操作自己的 stage tool，做完立即停，不参与开放式闲聊。
- 代码定位与效果：`src/research/agents/supervisor.py`；这样容易测试，也避免跨阶段副作用。

### Q094. `parallel_tool_calls=False` 反映了什么设计取向？
- 目标：说明系统优先选择确定性阶段协作，而不是任意并发。
- 用了什么：supervisor compile 时的配置项。
- 当前怎么做：即使是多 agent，阶段 handoff 也保持顺序推进，不让 supervisor 同时放飞多个 worker 并行乱改状态。
- 代码定位与效果：`src/research/agents/supervisor.py`；这和 research graph 的线性设计是一致的。

### Q095. 多 agent 结果为什么还要同步到 workspace？
- 目标：避免多 agent 跑完后只有内存状态，没有工件可看。
- 用了什么：`write_node_output()` 和 `_sync_node_to_workspace()`。
- 当前怎么做：每个阶段结果都能落成文件，前端和调试者可以看到不同阶段输出。
- 代码定位与效果：`src/research/agents/supervisor.py`、`src/agent/output_workspace.py`；这让多 agent 的“协作痕迹”具象化。

### Q096. review 失败时为什么还要追加 revision 文件？
- 目标：保留“最后一版被挡住的草稿”，方便迭代。
- 用了什么：`append_revision()` 和按标签命名 revision。
- 当前怎么做：当 review 失败时，supervisor 会把当前 draft 写成 `revisions/xxx_after_review.md`。
- 代码定位与效果：`src/research/agents/supervisor.py`、`src/agent/output_workspace.py`；这是后续问题修复和人工审阅的重要依据。

### Q097. `collaboration_trace` 有什么价值？
- 目标：记录多 agent 在协作过程中的阶段执行信息。
- 用了什么：trace 列表、节点名、backend mode、可能的时序记录。
- 当前怎么做：阶段执行后会把关键协作信息汇总进任务状态，供详情接口与调试使用。
- 代码定位与效果：`src/research/agents/supervisor.py`、`src/api/routes/tasks.py`；这样多 agent 不是黑盒。

### Q098. `register_backend()` 的适用场景是什么？
- 目标：说明系统支持节点级替换和实验。
- 用了什么：节点名到 `NodeBackend` 的覆盖注册。
- 当前怎么做：如果某个阶段想用实验版实现或特定测试替身，可以只替换这一节点，而不重写全链。
- 代码定位与效果：`src/research/agents/supervisor.py`；这是典型的迁移期与实验期架构设计。

### Q099. 当前多 agent 设计最大的收益是什么？
- 目标：总结为什么要做 staged workers 而不是一个大 prompt。
- 用了什么：阶段职责清晰、official supervisor、workspace 落盘、review gate。
- 当前怎么做：planner/retriever/analyst/reviewer 可以分别演进，而总流程仍保持一致。
- 代码定位与效果：`src/research/agents/supervisor.py`、`src/research/agents/*.py`；这样局部调优更容易，不必每次推倒整个 prompt。

### Q100. 当前多 agent 设计最大的边界是什么？
- 目标：承认它还不是完全成熟的 agent operating system。
- 用了什么：canonical order、兼容层、阶段受限工具和顺序 handoff。
- 当前怎么做：系统优先保证可控性与稳定性，所以自治程度被有意压低，主动技能使用和运行中交互仍在演进。
- 代码定位与效果：`src/research/agents/supervisor.py`、`src/skills/orchestrator.py`；这样回答更真实，也方便引出后续优化方向。

## 六、RAG 检索与候选论文质量控制（Q101-Q120）

### Q101. 当前项目的 RAG 主链为什么比普通检索问答更复杂？
- 目标：说明这里的 RAG 服务对象是长文综述，不是单轮回答。
- 用了什么：查询规划、多源检索、候选筛选、正文 ingest、PaperCard 抽取、上下文压缩和 review。
- 当前怎么做：主题先经过 `clarify/search_plan`，再进入 `search -> extract -> extract_compression -> draft -> review` 的完整链。
- 代码定位与效果：`src/research/graph/nodes/search_plan.py`、`src/research/graph/nodes/search.py`、`src/research/graph/nodes/extract.py`；RAG 在这里是写作流水线的一部分。

### Q102. `search_plan` 生成的 `query_groups` 有什么作用？
- 目标：把一个大主题拆成多个检索子问题。
- 用了什么：query groups、plan goal、时间范围和主题约束。
- 当前怎么做：后续 `search` 节点会把所有 query group 展开并发执行，而不是只用一句 topic 去搜。
- 代码定位与效果：`src/research/graph/nodes/search_plan.py`；这样召回范围更全，也更有层次。

### Q103. `search` 节点为什么强调三源并行召回？
- 目标：提升召回的广度、精度和补充性。
- 用了什么：SearXNG、arXiv direct、DeepXiv 并发查询。
- 当前怎么做：节点内部用 `ThreadPoolExecutor(max_workers=3)` 同时打三种来源，再统一 dedup 和 rerank。
- 代码定位与效果：`src/research/graph/nodes/search.py`；并行把延迟控制住，同时避免只依赖单一源的盲点。

### Q104. SearXNG 在这套 RAG 里负责什么？
- 目标：解释它是广覆盖召回源，不是最终权威源。
- 用了什么：多查询广泛网页搜索。
- 当前怎么做：SearXNG 提供更宽的候选发现能力，尤其适合补足官方 arXiv 接口之外的相关线索。
- 代码定位与效果：`src/research/graph/nodes/search.py`、`src/tools/search_tools.py`；它提高 recall，但需要后续严格筛选。

### Q105. arXiv direct 的角色是什么？
- 目标：提供更高质量的论文 metadata 与 arXiv 精准召回。
- 用了什么：直接 arXiv 搜索和 enrich。
- 当前怎么做：在并发检索中单独跑 arXiv 查询，后续又优先以 arXiv 信息补齐候选论文 metadata。
- 代码定位与效果：`src/research/graph/nodes/search.py`、`src/tools/arxiv_api.py`；它是质量最高的一层来源。

### Q106. DeepXiv 在当前流程里补什么？
- 目标：补充趋势、摘要、解释性信息和潜在候选发现。
- 用了什么：DeepXiv 查询客户端和附加元数据。
- 当前怎么做：DeepXiv 与其他两源并行，返回的论文或摘要信息会进入后续合并与 enrich 流程。
- 代码定位与效果：`src/research/graph/nodes/search.py`、`src/tools/deepxiv_client.py`；它的价值更偏补充而不是替代 arXiv。

### Q107. 年份过滤为什么重要？
- 目标：减少离题和过旧论文对综述的污染。
- 用了什么：effective year filter 和搜索计划里的时间范围。
- 当前怎么做：search 节点会根据 `search_plan` 或默认策略对多源搜索施加年份约束。
- 代码定位与效果：`src/research/graph/nodes/search.py`；这会直接影响 off-topic ratio 和综述时效性。

### Q108. 为什么候选合并后还要做 dedup？
- 目标：避免同一篇论文在不同源重复出现并挤占写作上下文。
- 用了什么：`arxiv_id` 和 `url` 双重去重。
- 当前怎么做：如果候选已有相同 arXiv ID 或 URL，就不会再次加入 combined list。
- 代码定位与效果：`src/research/graph/nodes/search.py`；dedup 后上下文预算能留给更多独立论文。

### Q109. `enrich_search_results_with_arxiv(...)` 做了什么？
- 目标：提升候选论文的元数据完整度。
- 用了什么：基于 arXiv 的 metadata enrich。
- 当前怎么做：多源合并后的候选会再统一用 arXiv 数据补齐标题、作者、年份、摘要等字段。
- 代码定位与效果：`src/research/graph/nodes/search.py`；候选卡片越完整，抽取和引用就越稳。

### Q110. 为什么 search 后还要做 strict-core rerank？
- 目标：让召回的“数量”进一步转化为“主题纯度”。
- 用了什么：core groups、fatal penalties 和主题范围过滤。
- 当前怎么做：候选列表会按照严格主题规则进行重排和剔除，特别是对 agentic scope、临床范围、时间范围等约束敏感。
- 代码定位与效果：`src/research/graph/nodes/search.py`；这对降低 off-topic ratio 很关键。

### Q111. `STRICT_CORE_FATAL_PENALTIES` 想解决什么问题？
- 目标：把最容易让综述跑偏的错误类型显式编码出来。
- 用了什么：诸如 `outside_requested_time_range`、`off_topic_core_intent`、`missing_agent_for_strict_scope` 等致命惩罚项。
- 当前怎么做：一旦候选命中这些惩罚，rerank 或过滤逻辑会明显降权甚至直接剔除。
- 代码定位与效果：`src/research/graph/nodes/search.py`；这是一种工程化主题治理，而不是全靠 prompt 自觉。

### Q112. 什么叫 fulltext-first ingest？
- 目标：说明系统优先消费全文而不是只靠 abstract。
- 用了什么：候选论文 ingest、全文下载、切分和本地语料层写入。
- 当前怎么做：search 阶段结束后会立即尝试 ingest 候选论文，优先把全文证据准备好，失败后才退到摘要级别。
- 代码定位与效果：`src/research/graph/nodes/search.py`、`src/corpus/ingest/loaders.py`；这是提高 grounded writing 的关键步骤。

### Q113. `fulltext_ratio` 指标为什么重要？
- 目标：量化本次综述到底有多少候选拿到了正文级证据。
- 用了什么：`fulltext_success / fulltext_attempted`。
- 当前怎么做：search 节点 ingest 完后就会计算这个比例，并把统计写回状态或评测结果。
- 代码定位与效果：`src/research/graph/nodes/search.py`、`src/eval/rag/runner.py`；fulltext_ratio 高，通常意味着后续写作和 grounding 更有底。

### Q114. `rag_result` 在 research workflow 里是什么？
- 目标：给 search 阶段的结构化输出一个明确名称。
- 用了什么：候选论文、检索轨迹、筛选结果、ingest 统计等集合。
- 当前怎么做：`rag_result` 会作为 extract 节点的重要输入，同时也会被落盘成 `rag_result.json`。
- 代码定位与效果：`src/api/routes/tasks.py`、`src/research/graph/nodes/search.py`；它是 search 阶段对下游的正式契约。

### Q115. 为什么说 planner/retriever fallback 问题会直接暴露在中间 run 上？
- 目标：解释为什么中间回归能帮助定位问题而不是只看最终分数。
- 用了什么：`paper_count`、`quality_gate_passed`、检索候选和 review 结果。
- 当前怎么做：如果 fallback plan 或 retriever 质量不好，往往很快体现为候选论文太少、paper_count 过低、quality gate 不通过。
- 代码定位与效果：`src/research/graph/nodes/search_plan.py`、`src/research/graph/nodes/search.py`、`eval/runner.py`；这就是中间 run 的诊断价值。

### Q116. 如何理解 off-topic ratio 的下降？
- 目标：把指标变化与架构改动联系起来。
- 用了什么：clarify、search_plan、strict-core rerank 和年份过滤。
- 当前怎么做：系统不只靠“搜更多”，而是通过澄清范围、查询分组和严格重排逐步减少离题候选。
- 代码定位与效果：`src/research/graph/nodes/clarify.py`、`src/research/graph/nodes/search_plan.py`、`src/research/graph/nodes/search.py`；off-topic ratio 低通常说明前半链路更稳了。

### Q117. `paper_count` 为什么是关键健康信号？
- 目标：衡量最终可写材料是否足够支撑综述。
- 用了什么：候选论文数、去重结果和下游抽取卡片数。
- 当前怎么做：search 后的 paper_count 太低，通常意味着后续写作只能反复引用少数论文，质量门也更容易触发。
- 代码定位与效果：`src/research/graph/nodes/search.py`、`eval/runner.py`；paper_count 是“可写性”的下限指标。

### Q118. 为什么正文级 ingest 对综述尤其重要？
- 目标：综述不仅要写 abstract，还要写方法、实验、局限和差异点。
- 用了什么：全文 chunk、结构化抽取和实验复盘式证据。
- 当前怎么做：拿到全文后，extract 阶段能从正文中抽出方法细节、数据集、指标、局限，而不是只停留在 TL;DR。
- 代码定位与效果：`src/corpus/ingest/loaders.py`、`src/research/graph/nodes/extract.py`；这会显著影响 report confidence。

### Q119. 为什么 `search` 阶段的产物也要进入 workspace？
- 目标：让检索问题在写作前就能被看见。
- 用了什么：`rag_result.json`、检索轨迹、artifact_ready SSE。
- 当前怎么做：一旦 search 节点结束，相关工件会落到 workspace，前端可以在生成早期就看到候选是否合理。
- 代码定位与效果：`src/api/routes/tasks.py`、`src/agent/output_workspace.py`；这样不用等到最终 report 才发现检索已经跑偏。

### Q120. 当前 RAG 链路仍然面临的核心问题是什么？
- 目标：诚实说明为什么分数提升后仍会被 review 挡住。
- 用了什么：高召回、高 fulltext 与 grounding/review 的更严格要求。
- 当前怎么做：即使候选和正文覆盖已提升，若引用分布不均、claim 没有充足支撑，review 仍可能判定 low confidence 或 ungrounded。
- 代码定位与效果：`src/research/graph/nodes/draft.py`、`src/research/graph/nodes/review.py`、`src/research/services/reviewer.py`；这也是后续优化重点。

## 七、Extract、Compression 与 Draft 写作（Q121-Q140）

### Q121. `extract_node` 的核心职责是什么？
- 目标：把检索候选变成可写作的结构化论文卡片。
- 用了什么：LLM 抽取、批量处理、fallback card 和规范字段。
- 当前怎么做：extract 会读取候选论文与正文证据，生成 `paper_cards` 供压缩和写作使用。
- 代码定位与效果：`src/research/graph/nodes/extract.py`；它是从“找到论文”到“理解论文”的关键转换层。

### Q122. `paper_cards` 一般承载哪些信息？
- 目标：说明下游写作不是直接再读原文，而是使用压缩后的结构化表示。
- 用了什么：标题、研究问题、方法、数据集、指标、结论、局限、URL 等字段。
- 当前怎么做：每篇候选论文在 extract 后都应得到一张卡片，供 comparison matrix、taxonomy 和 draft 使用。
- 代码定位与效果：`src/research/graph/nodes/extract.py`、`src/models/paper.py`；卡片化能显著减少后续上下文负担。

### Q123. 为什么 extract 需要 fallback simple card？
- 目标：避免抽取阶段因为单篇解析失败就整批中断。
- 用了什么：简化卡片回退策略。
- 当前怎么做：当 LLM 抽取或正文解析失败时，系统仍会尽量生成一版最小可用 card，而不是完全丢掉该论文。
- 代码定位与效果：`src/research/graph/nodes/extract.py`；这让链路更鲁棒，但 fallback card 的信息密度会更低。

### Q124. 为什么项目需要 comparison matrix 这类中间产物？
- 目标：在写作前先把论文之间的差异结构化。
- 用了什么：comparison matrix skill 和表格化对比。
- 当前怎么做：系统会把论文的任务、方法、数据集、指标和局限拉齐成矩阵，供写作阶段引用。
- 代码定位与效果：`.agents/skills/comparison_matrix_builder/SKILL.md`、`src/api/routes/tasks.py`；这样 draft 不必反复重读每张卡片再做比较。

### Q125. `extract_compression_node` 输出什么？
- 目标：给写作阶段准备一套更紧凑、更可控的上下文。
- 用了什么：`taxonomy`、`compressed_cards`、`section_evidence_pools` 等压缩结果。
- 当前怎么做：节点接收 `paper_cards` 后，按主题和章节组织内容，再把压缩结果写回状态。
- 代码定位与效果：`src/research/graph/nodes/extract_compression.py`、`src/research/services/compression.py`；这一步直接影响长文写作稳定性。

### Q126. `taxonomy` 为什么重要？
- 目标：防止综述按论文顺序平铺直叙。
- 用了什么：主题分类、方法分群和章节级组织。
- 当前怎么做：系统会从论文卡片中抽出结构性主题，作为综述正文的章节或段落骨架。
- 代码定位与效果：`src/research/graph/nodes/extract_compression.py`；taxonomy 越好，综述越像“主题综述”而不是“论文串烧”。

### Q127. `compressed_cards` 的设计价值是什么？
- 目标：降低上下文长度，同时保留写作最关键的信息。
- 用了什么：对原始 card 的进一步压缩与重组。
- 当前怎么做：draft 节点优先消费 compressed cards，而不是把所有原始 paper cards 全部再塞给模型。
- 代码定位与效果：`src/research/graph/nodes/extract_compression.py`、`src/research/services/compression.py`；这是控制上下文和超时的重要手段。

### Q128. `section_evidence_pools` 是为了解决什么问题？
- 目标：让每个章节都能找到对应证据，而不是整篇文章共用一桶材料。
- 用了什么：章节级 evidence map。
- 当前怎么做：压缩阶段按章节目标把相关论文和证据聚合成多个 pool，供 draft 按章节写作。
- 代码定位与效果：`src/research/graph/nodes/extract_compression.py`、`src/api/routes/tasks.py`；这有助于减少正文后半段逐渐失焦。

### Q129. 为什么引入 writing scaffold？
- 目标：让 draft 阶段在真正写之前先拿到英文综述的写作骨架。
- 用了什么：writing scaffold skill、outline 和 section evidence map。
- 当前怎么做：scaffold 会帮助定义章节结构、写作顺序和关键证据映射，再让 drafting agent 沿着框架写。
- 代码定位与效果：`.agents/skills/writing_scaffold_generator/SKILL.md`、`src/api/routes/tasks.py`；它能缓解“会写句子但不会组织综述”的问题。

### Q130. `draft_node` 的输入为什么不是单一字符串？
- 目标：说明成文阶段消费的是一组结构化对象。
- 用了什么：brief、search_plan、paper_cards、compression_result、taxonomy、evidence pools 等。
- 当前怎么做：draft 节点会把这些对象拼成写作上下文，而不是只把检索结果摘要直接丢给模型。
- 代码定位与效果：`src/research/graph/nodes/draft.py`；这也是系统和普通“搜一下再写”最大的不同点之一。

### Q131. 当前项目如何约束英文 survey 写作风格？
- 目标：让综述输出更接近英文学术写作，而不是泛化摘要。
- 用了什么：英文 prompt、survey 写作约束、template 文档和 writing scaffold。
- 当前怎么做：相关 prompt 与约束文件会强调英文标题、章节逻辑、证据驱动和引用规范。
- 代码定位与效果：`src/research/prompts/survey_writing.py`、`docs/template/survey_writing_constraints.md`；这类约束是提升文稿风格稳定性的关键。

### Q132. `draft_markdown` 和 `draft_report` 有什么区别？
- 目标：区分“面向人看的文本”和“面向系统的结构化内容”。
- 用了什么：Markdown 字符串和结构化报告对象。
- 当前怎么做：draft 节点既可能产出结构化 report 数据，也会同步生成可直接展示的 `draft.md` 文本。
- 代码定位与效果：`src/research/graph/nodes/draft.py`、`src/api/routes/tasks.py`；双轨设计有利于前端展示和后续验证。

### Q133. 为什么 report 生成阶段容易出现正文重复或结构漂移？
- 目标：说明长文写作不是 prompt 一写就稳。
- 用了什么：多阶段上下文拼接、章节重写、finalize 流水线。
- 当前怎么做：如果中间约束和输出收口不严，模型可能把标题、摘要或前文内容重复展开。
- 代码定位与效果：`src/research/graph/nodes/draft.py`、`src/agent/report_markdown.py`；这也是需要写作约束和流式中间预览的原因。

### Q134. 为什么当前综述容易“几篇论文被反复引用”？
- 目标：把引用不足问题归因到检索、压缩和写作分配三个层面。
- 用了什么：paper_count、section evidence pools 和 citation resolving。
- 当前怎么做：如果有效论文数不足，或者 evidence pool 分配不均，draft 往往会抓住几篇高权重论文反复用。
- 代码定位与效果：`src/research/graph/nodes/search.py`、`src/research/graph/nodes/extract_compression.py`、`src/research/graph/nodes/draft.py`；这正是引用分布优化的核心问题。

### Q135. 如何从工程上改进引用分布？
- 目标：给出比“改 prompt”更具体的回答。
- 用了什么：扩大高质量候选、提高 fulltext coverage、按章节分配证据、在 draft/review 阶段检查引用多样性。
- 当前怎么做：项目已经把候选质量和压缩结构前移，后续还需要让 draft 与 review 显式约束 citation spread。
- 代码定位与效果：`src/research/graph/nodes/search.py`、`src/research/graph/nodes/draft.py`、`src/research/services/reviewer.py`；引用问题本质上是前后链路共同造成的。

### Q136. 为什么综述写作要强调“主题式组织”而不是“按论文顺序复述”？
- 目标：体现你理解 survey 的学术写法。
- 用了什么：taxonomy、outline、comparison matrix 和 section evidence pools。
- 当前怎么做：项目试图先把论文按主题和方法组织，再写章节，这样每章可以综合多篇论文比较，而不是逐篇摘要。
- 代码定位与效果：`src/research/graph/nodes/extract_compression.py`、`.agents/skills/comparison_matrix_builder/SKILL.md`；这是从“资料堆砌”走向“综述分析”的关键。

### Q137. Context compression 为什么是长文综述的核心技术？
- 目标：避免长上下文塞满后模型超时、失焦或忘记前文。
- 用了什么：卡片压缩、章节 evidence pool、写作骨架和上下文截断控制。
- 当前怎么做：系统在 draft 前先压缩，而不是在最后才做无脑裁剪。
- 代码定位与效果：`src/research/services/compression.py`、`src/research/graph/nodes/extract_compression.py`；这对稳定长文输出极其关键。

### Q138. 为什么报告生成过程需要 live preview？
- 目标：让用户看到长文写作正在发生，而不是长时间转圈。
- 用了什么：workspace 落盘 + SSE `report_snapshot` 事件。
- 当前怎么做：`draft.md`、`report.md` 等 Markdown 文件一旦变化，就被后端扫描并推送给前端。
- 代码定位与效果：`src/api/routes/tasks.py`、`src/agent/output_workspace.py`；这提升体验，也方便发现写作早期就出现的问题。

### Q139. revision 文件对写作优化有什么帮助？
- 目标：保留 review 前后的成文轨迹。
- 用了什么：`append_revision()` 和 `revisions/*.md`。
- 当前怎么做：当 review 失败或过程需要保留阶段版本时，会把相应草稿写成 revision 文件。
- 代码定位与效果：`src/research/agents/supervisor.py`、`src/agent/output_workspace.py`；对比 revision 可以快速定位写作与审查分歧。

### Q140. 当前 draft 阶段最大的质量风险是什么？
- 目标：说明“检索好了”不等于“文稿自然就好了”。
- 用了什么：长上下文整合、引用分布、章节逻辑、claim grounding。
- 当前怎么做：目前主要风险是少数论文被高频重复引用、后半文渐失焦，以及部分 claim 超出证据覆盖。
- 代码定位与效果：`src/research/graph/nodes/draft.py`、`src/research/services/reviewer.py`；这也是为什么后面还需要 review 和评测层。

## 八、Grounding、Review 与评测体系（Q141-Q160）

### Q141. `review_node` 在 research workflow 里做什么？
- 目标：把 draft 从“可读”推进到“可接受”。
- 用了什么：reviewer service、grounding 检查、置信度与通过判定。
- 当前怎么做：draft 完成后，review 节点会结合证据与写作结果生成 `review_feedback` 并设置 `review_passed`。
- 代码定位与效果：`src/research/graph/nodes/review.py`、`src/research/services/reviewer.py`；它是真正的质量门，不是摆设。

### Q142. reviewer service 的输入一般有哪些？
- 目标：说明 review 不是只拿最终 report 硬看。
- 用了什么：draft 文稿、paper cards、compression result、grounding 结果和证据池。
- 当前怎么做：服务会综合草稿和检索/抽取阶段的证据对象进行审阅，不仅看文字是否通顺。
- 代码定位与效果：`src/research/services/reviewer.py`、`src/research/services/grounding.py`；这让 review 更像基于证据的质控。

### Q143. `review_feedback` 通常包含什么？
- 目标：明确 review 输出不只是一个 pass/fail。
- 用了什么：问题列表、关键 claim、置信度、修改建议和总评。
- 当前怎么做：review 结果会进入任务状态，也会被落盘成 `review_feedback.json` 供前端和人工复盘。
- 代码定位与效果：`src/api/routes/tasks.py`、`src/research/graph/nodes/review.py`；这使得 review 失败是可解释的。

### Q144. `review_passed` 的工程意义是什么？
- 目标：把 review 结果变成图路由条件。
- 用了什么：布尔 gate 与 `persist_artifacts` 条件边。
- 当前怎么做：通过则进入最终产物持久化，不通过则停止但保留 inspectable 草稿与反馈。
- 代码定位与效果：`src/research/graph/builder.py`；这保证了“未通过”不会被伪装成正式完成。

### Q145. claim verification 在项目里是什么角色？
- 目标：检查关键结论是否真被证据支撑。
- 用了什么：claim verification skill/节点、引用对齐和证据匹配。
- 当前怎么做：系统会把重要 claim 与引用和证据做核验，输出结构化验证结果。
- 代码定位与效果：`.agents/skills/claim_verification/SKILL.md`、`src/graph/nodes/verify_claims.py`；这一步直接影响 review 中的 grounded 判断。

### Q146. `grounding.py` 服务在验证体系里做什么？
- 目标：把“感觉像有依据”变成明确的 grounding 计算。
- 用了什么：证据匹配、引用检查、片段级支撑分析。
- 当前怎么做：review 和评测环节都会使用 grounding 相关服务判断 claim 是否真的能从材料中找到支撑。
- 代码定位与效果：`src/research/services/grounding.py`；这比只看模型自评分数更可靠。

### Q147. 什么叫 ungrounded claim？
- 目标：在面试里准确解释 review 失败原因。
- 用了什么：claim 提取、引用检查和证据比对。
- 当前怎么做：如果文稿中的某个结论没有对应论文或正文证据支撑，就会被标成 ungrounded。
- 代码定位与效果：`src/research/services/grounding.py`、`src/research/services/reviewer.py`；这通常不是语言问题，而是证据链断裂问题。

### Q148. 为什么 `report_confidence` 可能是 low？
- 目标：说明低置信度不一定表示模型崩了，可能是质量门更严格了。
- 用了什么：review 反馈、grounding 结果、claim 通过率和引用覆盖情况。
- 当前怎么做：如果关键论断里有较多未落地引用、支撑论文过少或 evidence 分布不均，review 会压低 confidence。
- 代码定位与效果：`src/research/services/reviewer.py`、`src/research/graph/nodes/review.py`；这是一种保守但工程上更负责的做法。

### Q149. 为什么会出现 rag_score 很高，但 report confidence 仍然低？
- 目标：说明“检索质量”和“写作 grounded 程度”不是同一个指标。
- 用了什么：RAG 指标看候选和正文覆盖，review 看最终文稿的 claim 与引用关系。
- 当前怎么做：即使 `paper_count` 和 `fulltext_ratio` 很好，只要最终文稿只引用少数论文或 claim 分配不均，confidence 依然会被压低。
- 代码定位与效果：`src/eval/rag/runner.py`、`src/research/services/reviewer.py`；这正好体现了多层评测的必要性。

### Q150. `eval/runner.py` 在整个系统里负责什么？
- 目标：把链路结果转成可比较的指标。
- 用了什么：评测运行器、case 输入、layer 执行和 gate 汇总。
- 当前怎么做：运行器会基于 case 执行多个评测层，得到 rag/report 等维度的分数和通过结论。
- 代码定位与效果：`eval/runner.py`；这让优化不只靠主观阅读，而是有量化回归依据。

### Q151. `hard_rules` layer 检查什么？
- 目标：把最基础、最不能犯的错误先挡住。
- 用了什么：引用、结构、格式和基本约束规则。
- 当前怎么做：hard rules 层会优先检查违反明确规则的情况，例如缺失必要段落或明显格式问题。
- 代码定位与效果：`eval/layers/hard_rules.py`；它是最低层的“硬闸门”。

### Q152. `grounding` layer 为什么独立于 hard rules？
- 目标：因为 groundedness 不是纯格式问题，而是证据问题。
- 用了什么：引用-证据对齐、claim 支撑率和文本核验。
- 当前怎么做：grounding 层关注的是文稿内容能否被论文和证据真正支撑，而不只是格式是否合规。
- 代码定位与效果：`eval/layers/grounding.py`；它更接近内容真实性。

### Q153. `gate.py` 在评测体系里是什么？
- 目标：把多个层的结果汇成最终通过/不通过判断。
- 用了什么：层级评测结果、阈值、综合 gate。
- 当前怎么做：不同 layer 的结果会在 gate 层被汇总，决定当前 run 是否算通过。
- 代码定位与效果：`eval/gate.py`；这样不是单一分数说了算，而是多约束联合判定。

### Q154. `rag_score`、`report_score`、`fulltext_ratio`、`off_topic_ratio` 应该怎么解读？
- 目标：把几个常见指标讲成人能理解的话。
- 用了什么：检索质量、成文质量、正文覆盖率和离题率。
- 当前怎么做：rag_score 更偏前半链质量，report_score 更偏最终文稿，fulltext_ratio 衡量证据深度，off_topic_ratio 衡量主题纯度。
- 代码定位与效果：`src/eval/rag/runner.py`、`eval/runner.py`；这几项一起看，才能知道问题出在前半链还是后半链。

### Q155. `paper_count` 和 `quality_gate_passed` 反映什么？
- 目标：把中间 run 的两个常用信号说清楚。
- 用了什么：候选论文数量和质量门是否通过。
- 当前怎么做：paper_count 太低通常意味着材料不够写，而 quality_gate_passed=false 说明即使有材料，产出也没达到当前质量线。
- 代码定位与效果：`eval/runner.py`、`src/research/graph/nodes/search.py`；这两个指标适合快速诊断 run 健康度。

### Q156. 如何分析基线 run、中间回归和最新 run 的差异？
- 目标：说明你会读连续迭代结果，而不是只看单次高分。
- 用了什么：对比 fulltext_ratio、off_topic_ratio、paper_count、review_passed 和 confidence。
- 当前怎么做：基线通常暴露检索与正文获取问题，中间回归暴露 fallback/候选不足问题，最新 run 说明检索变强后 bottleneck 转移到 review 和 grounding。
- 代码定位与效果：`eval/runner.py`、`src/research/services/reviewer.py`；这就是工程优化“瓶颈迁移”的典型案例。

### Q157. `001_after_review.md` 这类 revision 文件能说明什么？
- 目标：证明系统会保留 review 阶段挡下来的版本。
- 用了什么：revision 目录、按标签命名的中间 Markdown。
- 当前怎么做：review 没通过时，草稿会被追加到 revisions 中，供人工比较最终版与被挡版本的差异。
- 代码定位与效果：`src/research/agents/supervisor.py`、`src/agent/output_workspace.py`；这是优化文稿质量非常有价值的材料。

### Q158. 如果 review 失败，应该先查哪里？
- 目标：给出一个工程化排障顺序。
- 用了什么：`review_feedback.json`、`claim_verification.json`、`paper_cards.json`、`rag_result.json` 和 revision 文稿。
- 当前怎么做：先看失败 claim 和 reviewer 结论，再回查是否是候选少、引用少、章节 evidence 分配差，最后再改 draft 或上游检索。
- 代码定位与效果：`output/workspaces/...`、`src/research/services/reviewer.py`；这样定位比盲改 prompt 快得多。

### Q159. 当前最值得优先优化的质量问题是什么？
- 目标：给出与现状一致的优先级判断。
- 用了什么：当前 run 的 review 失败、low confidence 和引用稀疏问题。
- 当前怎么做：现在的重点不是“没料可写”，而是“写出来后引用分布和 grounded claims 仍不足”，因此要优先优化 citation spread 和 section evidence 使用。
- 代码定位与效果：`src/research/graph/nodes/draft.py`、`src/research/services/reviewer.py`；这是当前迭代最真实的瓶颈。

### Q160. 指标体系本身会变，这意味着什么？
- 目标：提醒面试官和团队不要过度迷信单次分数。
- 用了什么：评测层、阈值和 case 的持续迭代。
- 当前怎么做：指标更新后，需要重新解释历史 run 的意义，并把重点放回“为什么这个分数变了，对应哪个链路被更严格要求了”。
- 代码定位与效果：`eval/runner.py`、`eval/layers/*.py`；这体现了你能把评测当工具，而不是当结果本身。

## 九、Skills、MCP、Workspace、持久化与聊天（Q161-Q180）

### Q161. 当前项目里的 skills 应该怎么理解？
- 目标：说明 skills 不是装饰文件，而是流程能力单元。
- 用了什么：技能清单、manifest/registry、主动调用与流程内调用两种形态。
- 当前怎么做：例如 claim verification、comparison matrix、writing scaffold 等技能被设计成可在流程中产出结构化工件。
- 代码定位与效果：`.agents/skills/*/SKILL.md`、`src/skills/registry.py`、`src/skills/research_skills.py`；skills 让工作流能力更模块化。

### Q162. `SkillsRegistry` 的职责是什么？
- 目标：统一发现、注册和列出技能。
- 用了什么：manifest、目录扫描和 registry 查询接口。
- 当前怎么做：系统启动或调用时会从技能目录收集清单，再由 orchestrator 或 API 暴露给前端和流程层使用。
- 代码定位与效果：`src/skills/registry.py`、`src/skills/discovery.py`；这是 skills 成为正式能力的基础设施。

### Q163. `discovery.py` 解决什么问题？
- 目标：自动发现技能而不是写死在代码里。
- 用了什么：技能目录扫描、manifest 解析和路径约定。
- 当前怎么做：skills 目录中的能力可以被扫描出来并转成 registry 内的结构化对象。
- 代码定位与效果：`src/skills/discovery.py`；这样新增技能不必每次都硬编码到后端。

### Q164. 为什么 skills 需要 manifest 化？
- 目标：把“能力能做什么、入参是什么”从代码里抽出来。
- 用了什么：`skill_id`、描述、输入 schema 等元数据。
- 当前怎么做：SkillOrchestrator 读取这些元数据，把技能包装成可供 agent 选择的工具。
- 代码定位与效果：`src/models/skills.py`、`src/skills/registry.py`；manifest 化让技能能被列举、选择和校验。

### Q165. `SkillOrchestrator` 的显式调用模式是什么？
- 目标：说明用户可以明确点名某个 skill。
- 用了什么：`/skill_id` 前缀和参数解析。
- 当前怎么做：如果用户消息以 `/` 开头，orchestrator 会直接识别 skill_id，解析 JSON 或 key=value 参数，然后调用该技能。
- 代码定位与效果：`src/skills/orchestrator.py`；这是主动使用 skills 的最直接入口。

### Q166. 隐式调用模式是什么？
- 目标：解释“用户不点名 skill，系统也能判断要不要用”。
- 用了什么：轻量 LLM 决策、工具摘要和 `should_use_skills`。
- 当前怎么做：orchestrator 会先把可用技能摘要交给轻量模型，让模型决定是否需要构建 skill chain。
- 代码定位与效果：`src/skills/orchestrator.py`；这为报告完成后的扩展对话提供了空间。

### Q167. 为什么 skill orchestrator 要支持 chain？
- 目标：因为很多任务不是单技能能完成的。
- 用了什么：`skill_chain`、逐步调用、前一步结果传给下一步。
- 当前怎么做：当模型判断需要多个技能时，orchestrator 会依次调用，并把 `_previous_results` 注入后续上下文。
- 代码定位与效果：`src/skills/orchestrator.py`；这让技能更像小型工作流，而不是孤立工具。

### Q168. 技能结果为什么也值得落到 workspace？
- 目标：把“用过哪些技能、产出了什么”可视化出来。
- 用了什么：`comparison_matrix.json`、`writing_scaffold.json`、`claim_verification.json` 等流式文件。
- 当前怎么做：这些技能产物已经进入 `_STREAMABLE_TASK_FILES` 白名单，变化后能通过 SSE 被前端看到。
- 代码定位与效果：`src/api/routes/tasks.py`；用户能直接感知 skills 不是隐藏在 prompt 里的黑盒。

### Q169. `mcp_adapter.py` 的角色是什么？
- 目标：把 MCP 能力接进项目，而不是只在文档里提。
- 用了什么：MCP 适配层、技能或工具接入点。
- 当前怎么做：后端通过 adapter 对接 MCP 调用，把外部能力包装成项目内部可用工具。
- 代码定位与效果：`src/tools/mcp_adapter.py`；这为后续扩展更多研究辅助能力留了接口。

### Q170. 为什么还要有 `/api/routes/skills.py` 和 `/api/routes/mcp.py`？
- 目标：把技能和 MCP 作为一等后端能力暴露出来。
- 用了什么：FastAPI 路由、清单查询和执行接口。
- 当前怎么做：这些接口让前端或其他调用方可以获取可用技能/MCP 状态，并执行相应能力。
- 代码定位与效果：`src/api/routes/skills.py`、`src/api/routes/mcp.py`；这让 skills/MCP 不只是内部实现细节。

### Q171. 当前 MCP 集成的边界是什么？
- 目标：实话实说地说明项目还没把 MCP 用到极致。
- 用了什么：stdio 方式的接入和 adapter 层。
- 当前怎么做：目前 MCP 主要通过适配层接入能力，但更完整的服务化治理和更深的前端使用路径仍在演进。
- 代码定位与效果：`src/tools/mcp_adapter.py`、`src/mcp_servers/`；这样回答更贴近现状，不会过度宣传。

### Q172. `output/workspaces/` 的目录层级应该怎么讲？
- 目标：让面试官快速理解产物组织方式。
- 用了什么：workspace 级目录 + task 级目录 + artifacts/revisions。
- 当前怎么做：通常是 `output/workspaces/{workspace_id}/tasks/{task_id}/...`，任务级目录下存放各阶段 json 和 md。
- 代码定位与效果：`src/agent/output_workspace.py`；这个层级天然支持“一个 workspace 多个任务”的场景。

### Q173. `build_workspace_id()` 的价值是什么？
- 目标：让 workspace 成为稳定会话标识而不是随机文件夹名。
- 用了什么：用户前缀、时间戳和唯一后缀。
- 当前怎么做：系统在创建任务时会生成或接受 workspace_id，并据此组织后续所有工件。
- 代码定位与效果：`src/agent/output_workspace.py`、`src/api/routes/tasks.py`；这样更利于展示、回查和人工管理。

### Q174. `write_node_output()` 解决什么问题？
- 目标：把节点结果规范写进 workspace，而不是每个节点自己发散命名。
- 用了什么：统一文件名与写盘逻辑。
- 当前怎么做：supervisor 和图节点都能复用它，把阶段输出落成一致的 json/md 工件。
- 代码定位与效果：`src/agent/output_workspace.py`、`src/research/agents/supervisor.py`；这减少了工件组织混乱。

### Q175. `append_revision()` 的意义是什么？
- 目标：保留重要版本变更历史。
- 用了什么：revision 序号、标签和 Markdown 追加。
- 当前怎么做：系统在 review 前后或失败时写入 revision 文档，形成可比对版本序列。
- 代码定位与效果：`src/agent/output_workspace.py`、`src/research/agents/supervisor.py`；它是最轻量但很实用的版本追踪方式。

### Q176. 任务快照持久化怎么做？
- 目标：保证任务详情在进程重启后仍可访问。
- 用了什么：`upsert_task_snapshot()`、`load_task_snapshot()`、`list_task_snapshots()`。
- 当前怎么做：任务关键字段会被定期写入 PostgreSQL，查询时可恢复成 `TaskRecord`。
- 代码定位与效果：`src/db/task_persistence.py`、`src/api/routes/tasks.py`；这使 API 层具备 durable history。

### Q177. report record 和 task snapshot 为什么要分开？
- 目标：区分运行态快照和最终交付记录。
- 用了什么：独立的 task snapshot 与 task report 持久化接口。
- 当前怎么做：快照偏向整个任务运行状态，report record 更偏最终文稿和可消费结果。
- 代码定位与效果：`src/db/task_persistence.py`、`src/api/routes/tasks.py`；这样查询和恢复语义更清晰。

### Q178. 当前“回到之前 workspace”的能力是怎么实现的？
- 目标：说明这不是只靠内存记忆。
- 用了什么：output 目录记录、workspace_id 复用和任务快照恢复。
- 当前怎么做：只要知道 workspace_id 和 task_id，就可以从磁盘工件和数据库快照中恢复查看历史结果。
- 代码定位与效果：`src/agent/output_workspace.py`、`src/api/routes/tasks.py`；这是从“会话内记忆”升级到“持久工作空间”的关键。

### Q179. 完成后的聊天框当前主要围绕什么能力？
- 目标：定义 post-report chat 的现实边界。
- 用了什么：最终报告上下文、聊天历史和可选 skills。
- 当前怎么做：聊天更多是围绕已经生成的报告问答，而不是在任务运行中控制 workflow。
- 代码定位与效果：`src/api/routes/tasks.py`、`src/skills/orchestrator.py`；这为后续引入主动 skill chat 留了接口。

### Q180. skills、MCP、chat 这一层当前最大的优化方向是什么？
- 目标：把现状和未来路径区分开。
- 用了什么：系统内嵌流程 skills、用户主动 skills、MCP 扩展和后报告对话。
- 当前怎么做：主链已能产生技能工件，但用户在聊天框中显式/隐式调用技能的体验仍需要进一步打磨。
- 代码定位与效果：`src/skills/orchestrator.py`、`src/api/routes/skills.py`、`src/tools/mcp_adapter.py`；这是把系统从“自动生成器”升级为“可交互研究助手”的关键一步。

## 十、前端、SSE、测试、部署与演进（Q181-Q200）

### Q181. 前端当前使用什么技术栈？
- 目标：给前端层一个准确、现代的技术画像。
- 用了什么：React 19、TypeScript 5、Vite 8、Tailwind CSS 4、`@xyflow/react`。
- 当前怎么做：前端通过任务 API、SSE 和图视图组件展示双工作流与报告生成过程。
- 代码定位与效果：`frontend/package.json`、`frontend/src/components/*`；这套栈适合快速构建可视化 agent UI。

### Q182. `GraphView.tsx` 在项目里负责什么？
- 目标：说明前端不是只有一个文本面板。
- 用了什么：基于 `@xyflow/react` 的节点图展示。
- 当前怎么做：组件会根据任务类型和阶段状态渲染 report/research 对应的工作流图。
- 代码定位与效果：`frontend/src/components/GraphView.tsx`；这让用户能直观看到 agent workflow 在哪里运行。

### Q183. `ConfigPanel.tsx` 的作用是什么？
- 目标：解释前端如何暴露任务配置项。
- 用了什么：表单配置、任务参数输入和模式选择。
- 当前怎么做：用户在配置面板中选择输入类型、模式和相关参数，再提交到 `/tasks`。
- 代码定位与效果：`frontend/src/components/ConfigPanel.tsx`；这是把复杂后端参数变成可操作 UI 的入口。

### Q184. 前端为什么要区分不同任务类型的图谱？
- 目标：避免把 report 和 research 两条完全不同的链强行画成一张图。
- 用了什么：任务 `source_type` 与前端类型定义。
- 当前怎么做：GraphView 会根据任务数据切换不同节点集与边集，保持视觉表达与后端真实拓扑一致。
- 代码定位与效果：`frontend/src/types/task.ts`、`frontend/src/components/GraphView.tsx`；图谱忠于实际实现，用户理解成本更低。

### Q185. live preview 的后端基础设施是什么？
- 目标：说明“报告生成时的动态更新”是如何实现的。
- 用了什么：SSE、workspace 文件扫描和 `report_snapshot` 事件。
- 当前怎么做：后端持续检查 `draft.md/report.md` 等文件变化，一有更新就把全文快照流给前端。
- 代码定位与效果：`src/api/routes/tasks.py`；这是当前接近 ChatGPT 式正文渐进输出的主要机制。

### Q186. 为什么前端右侧报告区必须和 SSE 对齐？
- 目标：强调 UI 不是静态页面，而是消费后端实时流。
- 用了什么：任务状态请求 + SSE 实时事件。
- 当前怎么做：前端既要显示当前已知最终结果，也要持续接收 `artifact_ready/report_snapshot` 来刷新右侧内容。
- 代码定位与效果：`src/api/routes/tasks.py`、`frontend/src/components/*`；否则用户只会看到长时间转圈，没有过程反馈。

### Q187. paper card 点击标题跳正文链接为什么是一个合理的前端需求？
- 目标：说明前端不仅展示摘要，还应支持回到原始证据。
- 用了什么：paper card 中的 `url`/`source_url` 字段与前端可点击链接。
- 当前怎么做：后端在候选和 card 中保留论文链接，前端可以据此跳转到原始文章或 arXiv 页面。
- 代码定位与效果：`src/models/paper.py`、`src/research/graph/nodes/extract.py`、`frontend/src/components/*`；这能增强可验证性和可用性。

### Q188. 为什么前端展示必须和后端结构变化同步？
- 目标：说明 UI 不是最后随便补的壳。
- 用了什么：共享任务类型、阶段字段和 artifact 命名约定。
- 当前怎么做：当后端新增了 `extract_compression`、skills 工件或 workspace 恢复逻辑，前端也要同步更新节点图、右侧面板和历史记录入口。
- 代码定位与效果：`frontend/src/types/task.ts`、`frontend/src/types/phase34.ts`、`src/api/routes/tasks.py`；否则会出现“后端有能力、前端看不见”的脱节。

### Q189. 为什么在跑真实 E2E 前要先检查模型环境？
- 目标：避免把“外部模型不可用”误判成系统 bug。
- 用了什么：`tests/api/check_env_gpt_models.py`。
- 当前怎么做：AGENTS 明确要求先检查配置中的 GPT 模型是否可调用，再进行全流程测试。
- 代码定位与效果：`AGENTS.md`、`tests/api/check_env_gpt_models.py`；这能显著减少无效排障。

### Q190. 当前测试体系如何覆盖任务 API？
- 目标：说明项目不是全靠手工点前端。
- 用了什么：pytest、FastAPI `TestClient` 和 API 级测试。
- 当前怎么做：`tests/api/` 下会覆盖任务创建、持久化、agents 路由等关键接口行为。
- 代码定位与效果：`tests/api/test_tasks.py`、`tests/api/test_task_persistence.py`、`tests/api/test_agents.py`；这为接口回归提供了底线保障。

### Q191. 为什么 research graph 节点适合做单测？
- 目标：因为每个节点输入输出都比较明确。
- 用了什么：节点级 pytest、伪造 state 和局部依赖隔离。
- 当前怎么做：像 `test_search_node.py`、`test_review_node.py`、`test_clarify_node.py` 这类测试会直接针对节点行为断言。
- 代码定位与效果：`tests/research/graph/nodes/*`；这比只做大而全 E2E 更容易定位问题。

### Q192. supervisor / langgraph-backed agents 为什么也需要单测？
- 目标：多 agent 的稳定性不能只靠人工观察。
- 用了什么：backend mode、handoff 行为和 deterministic models 的测试。
- 当前怎么做：测试会覆盖 official supervisor 包装、legacy/v2 切换和节点执行逻辑。
- 代码定位与效果：`tests/research/agents/test_langgraph_backed_agents.py`、`tests/research/agents/test_supervisor.py`；这样迁移期更安全。

### Q193. 评测层本身也要测试吗？
- 目标：避免指标计算逻辑悄悄漂移。
- 用了什么：hard rules、runner 和 gate 层测试。
- 当前怎么做：评测相关逻辑也有对应测试，保证分数与判定的基本一致性。
- 代码定位与效果：`tests/eval/test_hard_rules.py`、`eval/*`；如果评测逻辑不稳，所有优化判断都会失真。

### Q194. 为什么真实接口的 E2E 仍然不可替代？
- 目标：说明单测不能完全覆盖外部依赖和真实时序。
- 用了什么：真实搜索源、真实模型、真实 workspace 输出和指标评测。
- 当前怎么做：项目会用实际 API 跑完整 workspace/task，再审阅 report、revision 和评测结果。
- 代码定位与效果：`tests/api/check_env_gpt_models.py`、`eval/runner.py`、`output/workspaces/`；这能发现 prompt、检索和外部依赖的真实联动问题。

### Q195. `.env` 配置在这个项目里为什么关键？
- 目标：系统高度依赖外部模型、数据库和搜索源配置。
- 用了什么：`load_dotenv(".env")`、模型 provider 配置、数据库 URL、搜索 API 参数。
- 当前怎么做：脚本和测试需要显式加载 `.env`，避免路径不一致导致配置失效。
- 代码定位与效果：`AGENTS.md`、`src/agent/settings.py`、测试脚本；配置加载稳定性直接影响整个端到端流程。

### Q196. 当前主 LLM provider 是怎么接入的？
- 目标：说明系统不是绑死某一家 SDK，而是走兼容接口。
- 用了什么：OpenAI-compatible chat clients、DeepSeek 作为主要提供方。
- 当前怎么做：`src/agent/llm.py` 和 `src/agent/settings.py` 负责构建 reasoning/quick 等不同模型客户端。
- 代码定位与效果：`src/agent/llm.py`、`src/agent/settings.py`；这种接法让模型替换成本更低。

### Q197. PostgreSQL 在部署上处于什么位置？
- 目标：强调它是主数据库，不是可选装饰。
- 用了什么：`DATABASE_URL`、SQLAlchemy engine、task/report persistence。
- 当前怎么做：所有长生命周期持久化都应连接 PostgreSQL，pgvector 是可选扩展，不能用 SQLite 替代。
- 代码定位与效果：`AGENTS.md`、`src/db/engine.py`、`src/db/task_persistence.py`；这决定了部署文档和本地开发都要围绕 PostgreSQL。

### Q198. 当前项目最适合怎样部署与演示？
- 目标：给出一个符合现状的落地口径。
- 用了什么：FastAPI 后端、React 前端、PostgreSQL、外部模型与搜索源配置、output/workspaces 挂载。
- 当前怎么做：启动后端和前端后，用户通过 Web 创建任务、看图谱、看 live preview、查历史 workspace。
- 代码定位与效果：`README.md`、`src/api/app.py`、`frontend/`；这比强调 CLI 更符合当前产品形态。

### Q199. 当前系统的可观测性主要靠什么？
- 目标：解释为什么出了问题可以比较快定位。
- 用了什么：节点事件、workspace 工件、任务快照、review feedback、评测结果。
- 当前怎么做：从 API 到前端到磁盘文件都有观察面，能同时看流程、文稿和指标。
- 代码定位与效果：`src/api/routes/tasks.py`、`src/agent/output_workspace.py`、`eval/runner.py`；这对长链路系统尤其重要。

### Q200. 如果最后让你总结这套系统的演进路线，你会怎么说？
- 目标：给面试一个高质量收尾。
- 用了什么：先完成双工作流和官方 supervisor 收敛，再把检索质量、压缩写作、grounding review、skills 交互和前端体验逐步补齐。
- 当前怎么做：当前主线已经从“有没有料”推进到“写出来是否可信、引用是否均衡、过程反馈是否足够好”，下一阶段重点是 citation spread、主动 skills 使用和更强流式成文体验。
- 代码定位与效果：`src/research/graph/nodes/search.py`、`src/research/graph/nodes/draft.py`、`src/research/services/reviewer.py`、`src/skills/orchestrator.py`；这说明系统正在从可跑走向可用、从可用走向可信。
