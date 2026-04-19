# PaperReader Agent — 技术文档总览

> 这套文档只描述当前仓库真实实现，不再沿用旧设计想象图。  
> 阅读目标：把项目讲清楚，把代码讲清楚，把“用了什么方法、怎么做”讲清楚。

## 目录

```text
docs/面试/tech_part/
├── README.md
├── 01-项目概览.md
├── 02-技术栈.md
├── 03-工作流架构.md
├── 04-多智能体协作.md
├── 05-Memory系统.md
├── 06-RAG检索架构.md
├── 07-Grounding验证体系.md
├── 08-评测体系.md
├── 09-工具与Skills.md
├── 10-CircuitBreaker设计.md
├── 11-ContextCompression设计.md
└── 12-EntropyManagement设计.md
```

## 这次更新的写法约束

本目录现在统一采用下面这套结构：

1. 先说模块在项目里的真实职责。
2. 再给出工作流图或结构图。
3. 明确写出“用了什么方法（Use What）”。
4. 明确写出“当前项目怎么做（How To Do）”。
5. 每篇至少放 1 到 3 段真实代码块，而不是只写概念解释。

## 推荐阅读顺序

如果只想先把项目主链讲清楚，按下面顺序看：

1. `01-项目概览.md`
2. `03-工作流架构.md`
3. `06-RAG检索架构.md`
4. `04-多智能体协作.md`
5. `07-Grounding验证体系.md`
6. `09-工具与Skills.md`

## 当前最重要的五条技术主线

### 1. 任务 API + workspace-first

- 入口是真实 `/tasks`，不是旧 CLI。
- 用户先创建 task，再看 SSE，再看 `output/workspaces/` 和 `/tasks/{id}/result`。

### 2. 双工作流

- Report workflow：单篇论文报告。
- Research workflow：topic-driven survey 工作流。

### 3. RAG 不是单步问答

- 当前实现不是“检索后直接回答”。
- 而是 `clarify/search_plan -> search -> extract -> extract_compression -> draft -> review`。

### 4. 多智能体协作已经落到官方 supervisor

- 当前主口径应当是 `create_supervisor + create_react_agent + staged workers`。
- 不是“完全自由自治的 agent society”。

### 5. 质量门不是口头说说

- Draft 后面还有 grounding、claim verification、review gate、confidence 计算。
- 这也是为什么检索质量提升后，报告仍然可能被 review 挡住。

## 这套文档适合怎么用

### 用于面试

- 先读 `01/03/06/07`，把主链讲熟。
- 再按面试官方向补 `04/05/08/09/10/11/12`。

### 用于项目宣传

- `01` 负责讲项目是什么。
- `03` 负责讲双工作流。
- `04` 负责讲多 agent。
- `06` 负责讲 RAG 深度。
- `07` 负责讲可信输出。

### 用于代码走读

- 每篇文档都已经把关键代码段贴出来。
- 建议一边看文档，一边打开对应实现文件核对。

## 当前已知边界

- 这些文档以 2026-04-19 仓库状态为准。
- 部分辅助模块仍有迁移中的兼容层，例如 `src/memory/manager.py`、supervisor 的 facade。
- 文档会明确区分“当前主链已经使用”和“仓库中存在但不是主路径”。
