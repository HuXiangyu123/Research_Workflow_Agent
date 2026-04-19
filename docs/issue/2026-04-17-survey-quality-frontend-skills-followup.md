# Issue Report: Survey 质量、前端可用性与主动 Skills 交互跟进

**日期**: 2026-04-17
**类型**: Survey Quality / Frontend UX / Skills Interaction
**优先级**: P0
**状态**: 分析完成，先修复 Problem 1

---

## 背景

当前 research survey 工作流已经能稳定产出：

- `brief.json`
- `search_plan.json`
- `rag_result.json`
- `paper_cards.json`
- `draft.md`
- `review_feedback.json`
- `report.md`

当前主要矛盾已经不是“跑不通”，而是：

1. survey 正文质量与引用分布仍不符合学术综述写法
2. 前端 paper card / 右侧面板能力退化
3. skills 体系缺少“系统内嵌”和“用户主动调用”分层，chat 没有承接主动 skills

---

## 最近产物的主客观质量分析

重点抽样：

- 基线：`output/workspaces/user_20260416T115750Z_932f54/tasks/c60a9959-6d70-403a-a9cc-a4bd29892bed`
- 中间回归：`output/workspaces/user_20260416T125922Z_87fec9/tasks/1ad3865a-a210-48a0-997e-ad48156c909d`
- 最新完整 run：`output/workspaces/user_20260416T130349Z_4de09c/tasks/384972b1-3f61-4416-8f5d-71efee25a37b`

### 客观指标

最新完整 run 的量化结果：

- `rag_score = 89.2`
- `report_score = 82.5`
- `paper_count = 8`
- `fulltext_ratio = 1.0`
- `off_topic_ratio = 0.0`
- `supported_ratio = 0.5`
- `review_passed = false`

这说明当前问题已经不是 retrieval 质量，而是“成文质量与 grounding 质量没有同步提升”。

### 主观质量

最新 `report.md` 的优点：

- 英文标题和章节结构已经基本稳定
- 主题相关性明显优于基线 run
- 主体内容不再是纯中文块或重复摘要尾巴

但核心缺陷仍然明显：

1. **引用分布失衡**
   - 正文共有 `8` 个唯一引用标签，但 main body 的重复集中度很高：
   - `[6]` 出现 `27` 次
   - `[7]` 出现 `27` 次
   - `[5]` 出现 `24` 次
   - `[1]` 出现 `23` 次
   - `[2]` 仅出现 `5` 次
   - `[3]` 和 `[8]` 各仅出现 `9` 次
2. **章节级 citation coverage 不均衡**
   - `Introduction / Taxonomy / Methods` 覆盖了 8 个标签
   - `Datasets / Evaluation / Discussion / Future Work` 只覆盖 5 个标签
   - 说明引用更多是“前面堆满”，后面几个 section 开始反复复用少数论文
3. **claim-level grounding 仍然差**
   - `review_feedback.json` 显示 `3/6 claims ungrounded`
   - `2/6 claims` 因 usable citation evidence 缺失无法验证
   - `report confidence = low`
4. **综述仍有“强结论、弱证据”倾向**
   - 正文像 survey
   - claim verification 看起来更像“跨论文推断”，而不是“被明确证据支持的综合结论”

结论：

- `report_score` 提升说明结构、长度、语言、topic purity 已经改善
- `report confidence` 仍是 `low`，说明最终报告的“论断-证据耦合”没有过关
- 当前最需要修的不是再去扩 RAG，而是 **survey writing contract + citation distribution + claim grounding discipline**

---

## Problem 1：Survey 写作与最终生成体验

### 真实症状

#### 症状 1：引用数量虽然不再极少，但分布仍不符合综述写法

当前 `src/research/graph/nodes/draft.py` 已有 `_ensure_minimum_citation_coverage()`，但它只做两件事：

- 保证总引用数达到一个下限
- 在 section 末尾 round-robin 补一句代表性引用

它没有处理：

- section 级 citation coverage
- 代表论文在不同 section 的合理分配
- 少数论文在全文中过度重复
- 中心 claim 与直接支撑 citation 的绑定

因此结果会变成：

- “看上去有不少引用”
- 但正文仍是少数几篇论文反复支撑全篇

#### 症状 2：写作 scaffold 虽已生成，但没有真正主导 section evidence allocation

当前 `draft_node` 会产出：

- `comparison_matrix`
- `writing_scaffold`
- `writing_outline`
- `section_evidence_map`

但 prompt 侧真正使用的只有：

- scaffold
- outline
- matrix

`section_evidence_map` 没有被写进 prompt 主体，也没有进入 post-processing citation redistribution。

这意味着：

- skill 已经给出了 section 证据分配
- 但 drafting 主链路没有真正消费这份分配结果

#### 症状 3：draft live preview 仍然偏“最后一下子出现”

当前 SSE 的 live report 展示依赖两个机制：

1. `NodeEventEmitter` 推送 node start/end / thinking
2. `/tasks/{id}/events` 轮询 workspace 文件变化并发 `report_snapshot`

但 research drafting 当前还是：

- draft node 内部先完整生成 `DraftReport`
- node 结束后 supervisor 才把 `draft.md` 写到 workspace

结果是：

- 前端在 draft 阶段只能看转圈和 thinking
- 正文没有 section 级逐步出现

这离用户预期的“像 ChatGPT 一样逐渐成文”还有明显差距。

### 根因

1. `survey writing contract` 只有通用写法要求，没有 citation diversity contract
2. `section_evidence_map` 生成了，但 prompt 和 repair 没吃进去
3. `draft.md` 只有 node 结束后才落盘，live preview 粒度太粗

### 修复方向

1. 从外部 academic review / survey guidance 提炼统一约束文件到 `docs/template`
2. 让 `survey_writing.py` 直接加载该约束文件作为 prompt contract 来源
3. 让 `draft.py` 显式消费 `section_evidence_map`
4. 在 draft 后处理里做 section-aware citation redistribution，而不是只补总数
5. 在 draft 阶段先写 scaffold preview，再写 cumulative section snapshots

---

## Problem 2：前端 paper card 链接与右侧功能退化

### 真实症状

当前 research 模式预览页里：

- `ReportPreview.tsx` 渲染了 `brief`、`search_plan`、`rag_result`、`researchMarkdown`
- 但没有渲染 `paper_cards` 列表
- 也没有任何 paper title -> original URL 的跳转逻辑

这意味着：

- 用户拿到了 `paper_cards.json`
- 但前端没有对应的 paper card 展示层
- 更不可能点击标题跳到原文 URL

### 右侧面板为什么显得“被破坏”

当前右侧 `WorkspaceInspectorPanel` 只剩三个 tab：

- `Artifacts`
- `Skills`
- `Events`

而 research 结果本身最需要看的其实是：

- 当前选中 task 的 paper cards
- 当前 report 对应的 citations / artifacts / revisions
- 与 task 强相关的主动 skill 操作

现在的右侧更像通用 inspector，不像“围绕当前 report 工作”的工作台。

### 根因

1. `paper_cards` 虽存在于 task payload，但没有前端组件消费
2. `PaperCardsSection` 这类研究态结果组件目前不存在
3. 右侧面板被统一成 workspace inspector 后，没有保留 report-centric 交互

### 后续修复方向

1. 增加 `PaperCardsSection`
2. paper title 支持 `url / arxiv_id` 跳转
3. 把右侧重新收敛为“当前 task 的 artifacts + paper cards + explicit skills”

---

## Problem 3：缺少“用户主动使用 skills”的完整链路

### 真实症状

当前 skill 体系虽然已经有：

- `visibility`
- `/api/v1/skills`
- `SkillPalette`
- `/api/v1/skills/run`

但用户真正最自然的入口其实应该是：**报告生成后的 chat**。

而当前 chat 实现是：

- `frontend/src/components/ChatPanel.tsx` 仅发送纯文本
- `src/api/routes/tasks.py::task_chat()` 直接把消息拼进 LLM 上下文
- 没有 skill planning / explicit skill selection / tool-style result injection

所以现在的 skill 使用方式是：

- 用户去右侧单独点一个 skill
- skill 跑完返回一段 JSON

这不是 report follow-up 的自然交互方式。

### 根因

1. `skills` 没有分成系统内嵌与用户主动调用两层
2. `SkillVisibility` 虽有 `AUTO / EXPLICIT / BOTH`，但 API 和 UI 没用它做信息架构
3. `task_chat()` 完全没有 skill routing

### 后续修复方向

建议将 skills 分成两类：

1. **Embedded workflow skills**
   - 由系统在 `draft / review / search` 节点自动调用
   - 例如 `comparison_matrix_builder`、`writing_scaffold_generator`、`claim_verification`
2. **Explicit user skills**
   - 由用户在 chat 中主动触发
   - 例如 “帮我做对比矩阵”“帮我检查某个 claim”“帮我抽一篇论文的实验设置”

建议后续聊天框逻辑：

- chat 先做 lightweight intent routing
- 命中 explicit skill 时执行 skill
- skill 结果以 artifact / markdown summary 形式回灌 chat
- 未命中时再走普通 LLM continuation

---

## 本轮实施范围

本轮先修 **Problem 1：survey 优化**，优先顺序如下：

1. 生成并接入 `docs/template` 下的 survey academic writing constraints
2. 修复 `draft.py` 对 `section_evidence_map` 的消费缺口
3. 增强 citation diversity / section coverage 后处理
4. 改善 draft 阶段的 workspace live snapshot

Problem 2 和 Problem 3 本轮先记录，不在本次代码修改里完成。
