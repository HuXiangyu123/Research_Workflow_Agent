# Issue Report: Research Survey 生成链路问题与修复顺序（用户确认版）

**日期**: 2026-04-16
**类型**: Report Quality / Research Workflow
**优先级**: P0
**状态**: 核心执行链路已修复，剩余质量优化继续跟进

---

## 背景

当前 research 调研报告在学术写作质量上存在系统性缺陷，核心表现包括：

1. 报告正文偏摘要复读，缺少对论文正文内容的深度分析。
2. `future_work` 逻辑混淆，把“信息缺失/局限描述”直接改写为“需要进一步研究”。
3. 检索召回与主题匹配度不足，出现明显离题论文。
4. 信息读取策略过早截断，导致证据粒度不足。

本文件按**用户确认顺序**固化修复优先级，后续实现必须按以下顺序推进。

---

## 修复顺序（必须按此执行）

### 1. 正确实现下载 arXiv 正文（最高优先级）

**问题**:
- 现状主要基于 metadata/abstract 生成 paper_cards 与报告内容。
- 缺乏正文级证据，导致 methods/evaluation/discussion 深度不足。

**修复目标**:
- 对候选论文增加可控的正文下载与解析流程（PDF -> structured text/chunks）。
- 报告生成优先使用正文证据，abstract 仅作为回退。

**建议落点**:
- `src/research/graph/nodes/search.py`
- `src/research/graph/nodes/extract.py`
- `src/research/services/compression.py`
- `src/tools/arxiv_api.py`（如需增强）

**验收标准**:
- 至少 70% 入选论文包含正文 chunk（非仅 abstract）。
- methods/evaluation 段落可引用正文中的实验设置、指标或结果句。

---

### 2. 修正 `needs_followup=true` 的处理策略

**问题**:
- `clarify` 阶段标记 `needs_followup=true` 时，当前流程可能直接继续并生成报告。
- 用户意图是：前端长期无交互、或后端自动化测试场景下，允许 LLM 自动补全缺失信息后继续。

**修复目标**:
- 明确双路径策略：
  - 交互场景: 等待用户补充。
  - 无交互超时/自动测试场景: 启动 LLM auto-fill 补全并继续。

**建议落点**:
- `src/research/graph/nodes/clarify.py`
- `src/research/graph/nodes/search_plan.py`
- `src/research/graph/builder.py`
- `src/api/routes/tasks.py`（超时与模式判定）

**验收标准**:
- `needs_followup=true` 时不再默认无条件推进。
- 自动化场景有可审计日志: 何时触发 auto-fill、补了哪些字段、置信度如何。

---

### 3. 使用 web search 获取“综述学术写法/格式规范”并固化到写作约束

**问题**:
- 目前 survey 写作风格与学术综述规范不稳定，章节逻辑和学术表达不统一。

**修复目标**:
- 基于可追溯 web source 整理 survey 写作规范（结构、引用、对比、讨论、future work 写法）。
- 将规范显式注入 drafting prompt / writing scaffold，而不是依赖隐式模型先验。

**建议落点**:
- `src/research/graph/nodes/draft.py`
- `src/research/prompts/*`
- `src/skills/research_skills.py`（如使用 scaffold 技能）

**验收标准**:
- 报告中 introduction/methods/evaluation/discussion/future_work 具备明确学术功能分工。
- `future_work` 不再直接复写 limitations。

---

### 4. 采用流动读取策略保存更多文献信息（替代硬截断）

**问题**:
- 当前存在候选截断、摘要长度截断、批次输出不稳定等问题，导致信息损失与样本塌缩。

**修复目标**:
- 引入流动读取/分层缓存策略：优先保留更多论文的关键信息，再按 section 需求做渐进扩展。
- 避免“一刀切截断到固定前 N 篇 + 固定字符数”。

**建议落点**:
- `src/research/graph/nodes/extract.py`
- `src/research/services/compression.py`
- `src/research/graph/nodes/draft.py`

**验收标准**:
- 候选到 card 的转化率显著提升，不再出现明显批次丢失。
- 报告证据覆盖更均衡，不只集中在前几篇论文。

---

### 5. 对初步 search 结果做二次筛选（主题相关性强约束）

**问题**:
- 初步召回包含离题论文。示例: 在“AI agent 医疗应用”主题下，不应无差别保留联邦学习综述等弱相关论文。

**修复目标**:
- 增加 domain-aware rerank/filter: 主题词匹配 + 任务意图匹配 + 负例规则。
- 在进入 extract 前先做候选净化。

**建议落点**:
- `src/research/graph/nodes/search.py`
- `src/research/agents/retriever_agent.py`
- `src/research/graph/nodes/extract.py`

**验收标准**:
- 候选集与主题相关性可解释（保留/剔除有理由）。
- 关键主题（如医疗 agent）的离题论文比例显著下降。

---

## 补充约束

1. 实现顺序必须遵守本文件，禁止跳步先做低优先级优化。
2. 每一步完成后都要有可重复的评测或样例验证。
3. 修复过程中不得以“模板补丁”掩盖证据链问题（如仅改标题、不改证据来源）。

---

## 建议执行方式

1. 先完成步骤 1 与 2，打通“正文证据 + followup 策略”基础能力。
2. 再完成步骤 3，统一学术写作规范输入。
3. 随后完成步骤 4 与 5，提升信息保真与主题纯度。

以上顺序为当前版本的执行基线。

---

## 2026-04-16 修复落地说明

本 issue 对应的核心执行链路缺口已完成修复，重点是把 API 实际运行路径重新对齐到 research graph 的既定顺序，而不是只在 graph builder 中声明节点。

**已落地变更**:

1. `AgentSupervisor` 真实执行顺序已补齐 `extract_compression`，不再出现 API 任务跳过压缩/证据整理阶段的问题。
2. `clarify -> search_plan` 的 followup 分支已按状态短路:
   - 交互场景若 `awaiting_followup=true`，后续 stage 不再继续真实执行。
   - 非交互/自动场景仍可通过既有 auto-fill 逻辑补全后继续。
3. `review_passed=false` 时不再继续执行 `persist_artifacts`，避免把未通过审阅的报告当作最终产物落盘。
4. `/tasks` 与 `/tasks/{id}/result` 已补充 `compression_result`、`taxonomy`、`awaiting_followup`、`followup_resolution` 等审计字段，便于检查 workflow 是否按预期推进。

**对应代码**:

- `src/research/agents/supervisor.py`
- `src/api/routes/tasks.py`
- `src/models/task.py`

**已验证**:

- `pytest tests/research/agents/test_supervisor.py tests/research/graph/test_builder.py tests/research/graph/nodes/test_clarify_node.py tests/research/graph/nodes/test_search_node.py tests/research/graph/nodes/test_extract_node.py -q`
- `pytest tests/api/test_tasks.py tests/api/test_agents.py -q`
- `python tests/api/check_env_gpt_models.py`

说明: 本次修复解决的是“实际执行链路未按修复顺序落地”的阻塞问题。正文证据深度、综述写作质量、上下文压缩策略等仍按本文件既定顺序继续迭代，但不再被 supervisor 绕过或被错误 stage 顺序掩盖。

---

## 2026-04-16 API 验证追加记录

本 issue 在后续真实 API 端到端验证中继续收敛，重点不再是“节点没跑到”，而是“真实执行后候选、正文、抽取、审稿闸门是否健康”。

### 已新增修复

1. `PlannerAgent` 对 LLM 产出的轻微 schema 漂移增加容错：
   - `query_groups[].priority` 越界时自动夹到 `1..3`
   - 不再因为单个字段超界而整份 `SearchPlan` 退回 fallback
2. `RetrieverAgent` 的 fallback candidate assembly 不再裸传最薄 metadata：
   - 先调用 `enrich_search_results_with_arxiv(...)`
   - 为候选补齐 `arxiv_id / authors / pdf_url / published_date`
3. `search` 阶段正文 ingest 已重新打通：
   - 真实 run 中已出现 `fulltext=8/8`
   - `extract` 也能按 `8/8` 使用正文证据而非纯 abstract
4. `extract` 批量抽取不再丢卡：
   - 单批返回数量不足时自动 fallback 补齐
   - 原始 `title / authors / arxiv_id / url` 强制回填，避免模型改写 metadata

### 真实验证结论

1. 旧基线 workspace:
   - `user_20260416T115750Z_932f54 / c60a9959-6d70-403a-a9cc-a4bd29892bed`
   - `rag_score=47.1`
   - `report_score=65.0`
   - `off_topic_ratio=0.941`
   - `fulltext_ratio=0.0`
2. 中间回归 run 暴露了 planner/retriever 缺口：
   - `user_20260416T125922Z_87fec9 / 1ad3865a-a210-48a0-997e-ad48156c909d`
   - `paper_count=2`
   - `quality_gate_passed=false`
   - 证明仅修写作侧不足，前链路必须补 planner/retriever
3. 最新真实 run:
   - `user_20260416T130349Z_4de09c / 384972b1-3f61-4416-8f5d-71efee25a37b`
   - `rag_score=89.2`
   - `report_score=82.5`
   - `fulltext_ratio=1.0`
   - `paper_count=8`
   - `review_passed=false`

### 当前剩余问题

1. 候选相关性虽显著提升，但真实 run 仍混入了边界外论文，说明 strict-core rerank 还需继续收紧。
2. review 闸门已经开始真实阻断低置信度报告，这是正确方向；但目前 `3/6 claims ungrounded` 仍导致最终产物只能以 after-review revision 形式保留，不能视作最终合格报告。
3. 正文 ingest 已恢复，但 PostgreSQL 写入 coarse/fine chunk 时仍出现 `NUL (0x00)` 字符导致的事务回滚；虽然 snippets 已成功回填到候选并被下游使用，但 durable chunk persistence 仍需继续修复。
