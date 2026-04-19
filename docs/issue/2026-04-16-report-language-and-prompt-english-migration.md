# Issue Report: 报告语言统一与 Prompt 英文化迁移分析

**日期**: 2026-04-16
**类型**: Output Contract / Prompt Architecture
**优先级**: P0
**状态**: 待修复

---

## 背景

用户要求已经明确收敛为两条硬约束：

1. 最终报告标题统一为英文。
2. 最终成文统一为英文。

在此基础上，所有参与研究工作流的 agent prompt 也应统一迁移为英文，否则输出语言契约会持续漂移，最终仍会出现中英混杂、标题和正文不一致、不同链路格式不一致的问题。

本报告基于 **2026-04-16 的真实 API 输出** 与仓库代码审查形成，不是推测性判断。

---

## 真实症状

### 症状 1：search survey 报告中英混写，且标题与正文语言不一致

真实输出文件：

- `output/workspaces/user_20260416T101117Z_d56d60/tasks/511bd283-9f25-45da-b1d8-43a600ccaca5/report.md`

直接表现：

- 标题是英文：`# AI agents for medical imaging diagnosis and triage`
- 一级 section heading 多为英文：`Abstract / Introduction / Methods / Evaluation`
- 但正文内容主体仍是中文，例如第 5、9、15 行开始的整段中文分析
- 同一个 section 内部还混有中文二级标题，如 `## 相关研究背景`

这说明当前系统不是“英文报告”，而是“英文壳 + 中文正文”的混合产物。

### 症状 2：single-paper 报告仍完全以中文结构输出

真实输出文件：

- `output/workspaces/user_20260416T101117Z_d56d60/tasks/acdd02d4-5b10-427f-bc08-335d4fd869ef/report.md`

直接表现：

- 顶层结构完全是中文：`## 标题 / ## 核心贡献 / ## 方法概述 / ## 关键实验 / ## 局限性 / ## 相关工作`
- grounding summary 里也混有中文元信息：`Tier A 来源占比`、`报告置信度`

这说明 single-paper 链路与 survey 链路根本没有共享同一套语言输出契约。

### 症状 3：survey 报告会在末尾再重复输出一遍正文摘要块

同一份 survey 输出在 `report.md` 末尾再次追加：

- `## 标题`
- `## 核心贡献`
- `## 方法概述`
- `## 关键实验`
- `## 局限性`

这不是模型随机重复，而是 renderer 明确在 markdown 组装阶段再追加了一套中文摘要块。

---

## 代码根因

### 根因 1：survey markdown renderer 在正文后硬编码追加第二套中文块

位置：

- `src/research/graph/nodes/draft.py:345-430`

关键事实：

- `_build_markdown()` 先按 `Abstract / Introduction / ... / Conclusion` 输出完整 survey
- 然后在 `401-429` 行再次追加固定中文块：
  - `## 标题`
  - `## 核心贡献`
  - `## 方法概述`
  - `## 关键实验`
  - `## 局限性`

这会直接导致：

1. 正文重复。
2. 最终 published markdown 天然变成中英混合格式。
3. 任何“最终只输出英文”的目标都会在 format 层被破坏。

### 根因 2：survey prompt 自身要求中文标题，且 user prompt 仍是中文

位置：

- `src/research/graph/nodes/draft.py:65-107`
- `src/research/graph/nodes/draft.py:123-139`

关键事实：

- `sections.title` 在 system prompt 中被要求是 `15-25 Chinese characters`
- survey drafting 的 user prompt 主体仍是中文，例如：
  - `请根据以上论文卡片生成详尽的综述草稿`
  - `不要输出任何 [...] 占位符`

即使 system prompt 框架里写了英文 section 名，title 与执行指令仍然把模型往中文报告拉回去。

### 根因 3：single-paper 报告链路从 system prompt 到 markdown renderer 都是中文契约

位置：

- `src/agent/report_frame.py:6-43`
- `src/agent/report_frame.py:46-83`
- `src/agent/report.py:24-46`
- `src/agent/prompts.py:1-16`

关键事实：

- `REGULAR_FULL_SYSTEM_PROMPT` 明确要求：`Write a complete Chinese literature reading report`
- `SURVEY_INTRO_OUTLINE_SYSTEM_PROMPT` 的 schema 也包含中文 section key：`论文信息 / 综述大纲 / 建议追问`
- chat prompt 继续要求 `Answer in Chinese`
- `_final_report_to_markdown()` 会直接把 `final_report.sections` 原样落盘，并追加：
  - `## 引用`
  - `## 引用可信度`
- 全局 prompt `LITERATURE_REPORT_SYSTEM_PROMPT` 也显式写了 `使用中文输出`

换言之，single-paper 路径当前不是“偶尔输出中文”，而是“被系统设计成输出中文”。

### 根因 4：agent prompt 层没有完成英文化迁移

位置示例：

- `src/research/agents/planner_agent.py:34-98`
- `src/research/agents/retriever_agent.py:68-86`
- `src/research/agents/analyst_agent.py:321-345`
- `src/research/agents/reviewer_agent.py:82-112`
- `src/research/graph/nodes/extract.py:223-243`
- `src/research/services/compression.py:234-239`
- `src/research/services/compression.py:327-331`

关键事实：

- Planner、Retriever、Analyst、Reviewer、Extract、Compression 等多处 prompt 仍用中文写 schema 和执行指令。
- 即使最终 renderer 改成英文，前序 agent 仍会产出中文 outline、中文 taxonomy、中文 claims/lessons、中文 fallback 文本。

因此“只改最终 markdown 标题”没有意义，必须把 prompt 层一起迁移。

### 根因 5：仓库内部默认策略仍把报告语言视为中文

位置：

- `src/skills/registry.py:350-359`

`_workspace_policy_loader_stub()` 默认返回：

- `"report_language": "zh"`

这会进一步说明整个技能/策略层默认值仍是中文，不符合当前产品方向。

---

## 为什么这已经是架构问题，而不是局部文案问题

当前至少存在四套彼此独立的“语言契约”：

1. single-paper system prompt 契约：中文
2. survey drafting 契约：混合，中英文并存
3. markdown renderer 契约：survey 英文正文 + 中文尾部摘要块
4. skill/workspace policy 默认契约：`report_language=zh`

这会导致：

- 同样的研究主题，single-paper 与 survey 输出语言不同
- 测试无法稳定断言输出格式
- reviewer/repair/fallback 路径一旦切换，语言立即漂移
- 后续想引入 English-only academic writing skill 或 MCP prompt，也没有单一入口约束可以依赖

所以这不是“换几个标题”的工作，而是 **Report Output Contract 重构**。

---

## 修复方向

### 1. 定义单一的英文报告契约

建议新增一份统一约束，至少包含：

- report title language: English
- report body language: English
- canonical section names: English
- citation appendix language: English
- chat continuation default language: English unless user explicitly overrides

该契约应同时覆盖：

- single-paper graph
- research survey graph
- chat continuation
- fallback / repair / grounding 输出

### 2. 将所有 agent prompt 统一迁移为英文

范围至少包括：

- `src/agent/*`
- `src/research/agents/*`
- `src/research/graph/nodes/*` 中面向 LLM 的 prompt
- `src/research/prompts/*`
- `src/research/services/compression.py`
- `src/skills/research_skills.py` 中 LLM prompt

要求不是“翻译一部分”，而是：

- system prompt 用英文
- JSON schema 字段说明用英文
- fallback 文本用英文
- self-reflection / reviewer / planner / analyst 的提示语全部用英文

### 3. 删除 survey renderer 中的中文兼容尾块

`src/research/graph/nodes/draft.py:401-429` 的中文摘要补丁必须从最终 published markdown 中移除。

如果需要保留旧版结构评测兼容性，应改为：

- 单独输出 machine-readable metadata
- 或写入 debug artifact
- 但不能拼进最终 `report.md`

### 4. 统一 single-paper 与 survey 的 markdown renderer

当前两条链路分别使用：

- survey: `src/research/graph/nodes/draft.py::_build_markdown`
- single-paper: `src/agent/report.py::_final_report_to_markdown`

必须抽成一套统一的 English-first renderer，避免两条链路继续各自长成不同格式。

### 5. 建立语言契约测试

至少增加以下自动化断言：

1. `report.md` 不应包含固定中文 heading：`标题 / 核心贡献 / 方法概述 / 关键实验 / 局限性 / 引用可信度`
2. survey 与 single-paper 都使用英文 section heading
3. survey 报告末尾不再重复输出摘要块
4. prompt 仓库扫描中不应再出现核心输出指令里的中文生成要求

---

## 验收标准

### 功能验收

1. `single-paper` 与 `search survey` 两条链路的最终 `report.md` 都为英文报告。
2. 报告标题、正文、引用附录、grounding summary 全部使用英文标签。
3. survey 不再在结尾重复追加第二套摘要块。
4. 所有 agent prompt 默认输出语言统一为英文。

### 代码验收

1. 不再存在显式要求“中文输出”的核心 prompt。
2. 不再存在单独的中文 markdown renderer 分支。
3. `report_language` 默认策略切换为 `en`，或改为显式配置项，禁止隐式写死为 `zh`。

### 回归验收

1. 用真实 API 重跑单论文任务与 survey 任务，输出目录仍在 `output/workspaces/*`。
2. 两类任务的 `report.md` 都可以通过“English-only heading”断言。

---

## 相关依赖

这项修复与以下议题强相关：

- `docs/issue/2026-04-16-survey-writing-quality-skills-mcp-gap-analysis.md`
- `docs/issue/2026-04-16-longform-generation-timeout-and-context-engineering.md`

原因很直接：

- 如果 prompt 不统一成英文，写作质量优化无法稳定收敛。
- 如果长文 fallback 仍是旧中文模板，超时后又会把系统拉回中英混杂状态。

