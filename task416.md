# Task416: 修复所有 docs/issue 问题

**创建时间**: 2026-04-16
**最后更新**: 2026-04-16

---

## 修复顺序（按时间倒序）

### Issue 1: `docs/issue/2026-04-16-langgraph-compliance-audit.md`
- 状态: 新建
- 优先级: P0
- 核心: 手工 AgentSupervisor 违反 LangGraph 规范，需迁移到 `langgraph_supervisor` 或 `langgraph_sdk.multi_agent`
- **暂不修复** — 迁移到官方 langgraph-supervisor 需要较大重构，当前 legacy/supervisor 架构仍可用

### Issue 2: `docs/issue/2026-04-16-langgraph-agent-workflow-audit.md`
- 状态: ✅ 已修复
- 优先级: P0
- 修复内容:
  - 删除了 `HandoffSupervisorState` + 所有 handoff 相关代码
  - 删除了 `build_official_supervisor_graph`、`build_handoff_agent`、`build_handoff_agents`、`_make_handoff_agent_node`、`collaborate_with_handoff` 等
  - 删除了 `_summarize_handoff_trace`、`_result_payload`、`_build_default_handoff_model`、`_build_handoff_user_message`、`_format_handoff_agent_message`、`_select_handoff_nodes` 等未用方法
  - 删除了 `SupervisorGraphState.payload` 字段
  - 删除了未用的 imports (json, Sequence, Annotated, AnyMessage, HumanMessage, add_messages)

### Issue 3: `docs/issue/2026-04-16-report-output-workspace-persistence-issue.md`
- 状态: 暂不修复
- 优先级: P1
- 核心: 报告产物缺少 `output/<task_id>/` 文件化工作区

### Issue 4: `docs/issue/2026-04-14-report-generation-quality-issues.md`
- 状态: ✅ 已修复
- 优先级: P0
- 修复内容:
  - 修复 `research_depth` 从 "plan" 改为 "full"，确保执行完整 graph（clarify→search_plan→search→extract→draft→review→persist）
  - 重写 draft_node system prompt:
    - 删除所有 "infer from abstract" 诱导摘要复读的指令
    - 增加 "analyze before write" 和 "synthesize, do NOT paraphrase" 原则
    - 扩展 introduction 字数上限至 1500-2500 字符（原 800-1200）
    - 扩展 methods 字数上限至 2000-3000 字符（原 1200-1800）
    - 扩展 discussion 字数上限至 1000-1500 字符（原 600-900）
    - 增加 introduction 至少引用 8 篇论文的要求
    - 增加 methods 至少引用 10 篇论文的要求
    - 增加 cross-reference papers 的指令
    - 将 max_tokens 从 8192 增加到 16384
    - timeout 从 240s 增加到 300s

### Issue 5: `docs/issue/2026-04-13-agent-architecture-issues.md`
- 状态: Issue 1 Fixed, Issue 2 Acknowledged
- 核心: Agent 层未被调用（已在之前修复，tasks.py 走 AgentSupervisor.collaborate）

### Issue 6: `docs/issue/2026-04-11-frontend-workflow-issues.md`
- 状态: ✅ 已修复（P1 部分）
- 优先级: P1
- 修复内容:
  - 替换 ReportPreview.tsx 中的 JsonSection，展示 brief 和 search_plan 为格式化的 UI 组件
  - 删除了未使用的 JsonSection 组件

### Issue 7: `docs/issue/2026-04-11-rag-report-quality-issues.md`
- 状态: 已修复（2026-04-12）

### Issue 8: `docs/issue/2026-04-10-research-task-blocking.md`
- 状态: 已修复

---

## 当前进度

- [ ] Issue 1: langgraph-compliance-audit (P0) — 暂不修复（需要较大重构）
- [x] Issue 2: langgraph-agent-workflow-audit (P0) — ✅ 已修复
- [ ] Issue 3: report-output-workspace-persistence (P1) — 暂不修复
- [x] Issue 4: report-generation-quality (P0) — ✅ 已修复
- [x] Issue 5: agent-architecture (已修复)
- [x] Issue 6: frontend-workflow (P1) — ✅ 已修复
- [x] Issue 7: rag-report-quality (已修复)
- [x] Issue 8: research-task-blocking (已修复)

---

## 修改文件清单

1. `src/api/routes/tasks.py` — `research_depth` 从 "plan" 改为 "full"
2. `src/research/graph/nodes/draft.py` — system prompt 优化，max_tokens 增加
3. `src/research/agents/supervisor.py` — 删除 handoff 冗余代码
4. `frontend/src/components/ReportPreview.tsx` — JSON 展示替换为格式化 UI
5. `eval/runner.py` — 扩展支持 research 模式和 paper_read 模式
6. `task416.md` — 本文件

---

## 测试命令

```bash
# Supervisor 测试
cd /Users/artorias/devpro/PaperReader_agent
python -m pytest tests/research/agents/test_supervisor.py -v

# 端到端测试 - paper_read 模式
python eval/runner.py --mode paper_read --cases eval/cases.jsonl --layer 1

# 端到端测试 - research 模式
python eval/runner.py --mode research --research-topic "AI Agent 在 Coding Agent 领域目前的发展" --layer 1

# 前端编译检查
cd frontend && npx tsc --noEmit
```

---

## Commit 历史

- `fix(supervisor): remove handoff mechanism and unused code` — 删除所有 handoff 冗余代码
- `fix(draft): improve system prompt to prevent abstract parroting` — 优化 draft prompt
- `fix(graph): research_depth=full to execute full pipeline` — 确保完整节点执行
- `fix(frontend): replace JSON preview with formatted brief/search_plan cards` — 优化前端展示
- `enhance(eval): support research mode and paper_read mode` — 扩展测试脚本
