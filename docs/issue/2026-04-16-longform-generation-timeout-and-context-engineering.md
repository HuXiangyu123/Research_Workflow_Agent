# Issue Report: 长文生成超时、阶段流式输出缺失与截断链路分析

**日期**: 2026-04-16
**类型**: Long-form Generation / Streaming / Context Engineering
**优先级**: P0
**状态**: 待修复

---

## 背景

当前 survey 工作流已经具备：

- 真实检索
- 正文下载与 chunking
- extract
- compression
- drafting
- review
- workspace 持久化

但它仍存在一个核心问题：**长文生成不是靠稳定的 section-by-section 产出完成，而是靠“大 prompt 一次成文 + 超时后模板回退”勉强过线。**

这会带来三个直接后果：

1. 上游 LLM 一旦 524 或超时，成文质量骤降。
2. 用户在生成过程中看不到逐段正文的流式进展。
3. 为了压上下文而做的多层截断，会让关键论文证据在成文前就被丢掉。

---

## 真实症状

### 症状 1：最终报告不是稳定长文生成，而是允许 fallback 模板顶替

真实运行中，survey drafting 已经出现过上游 `524` 超时；代码也明确允许：

- drafting 失败后直接进入 fallback
- 最终仍继续写出 `report.md`

这意味着当前“任务完成”并不等于“长文生成成功”，而是“图跑完了，哪怕最后用的是退化模板”。

### 症状 2：任务 SSE 只流转节点事件，不流正文内容

位置：

- `src/api/routes/tasks.py:266-311`

关键事实：

- `/tasks/{id}/events` 只把 `task.node_events` 往外发
- `_run_graph_sync()` 虽然调用了 `graph.stream(...)`
- 但只收集 node output 到 `state_snapshot`
- 没有把 section 级 markdown、token 流、custom progress 写进 SSE

所以现在所谓“流式”本质上还是：

- node_start
- node_end
- done

而不是“正文分段流式产出”。

### 症状 3：research 任务只有在整个 supervisor 返回后，才把最终 markdown 写回 task / workspace

位置：

- `src/api/routes/tasks.py:430-533`

关键事实：

- research 路径通过 `run_in_executor` 等待 `supervisor.collaborate(...)` 完成
- 然后才从 `supervisor_state` 里取 `draft_markdown / result_markdown`
- 然后才设置 `task.result_markdown`
- 然后才写 `report.md`

也就是说，即使中间节点已经产生部分正文，用户也无法在任务进行中看到“报告正在逐段成形”。

### 症状 4：正文证据虽然被下载，但在进入成文前已经被多层压缩和截断

从代码可以看出，关键证据会经历多层预算控制：

1. `search.py`  
   - 最多只对 `fulltext_top_n=16` 篇论文做正文下载  
   - 每篇最多取 `6` 个 snippet  
   - 每个 snippet 最多 `1400` chars  
   - 拼接后 `fulltext_excerpt` 再截到 `8000` chars
2. `extract.py`  
   - 总候选上限 `42`
   - 证据预算 `60000 chars`
   - 每篇喂给 LLM 的正文证据最多 `2800 chars`
   - fallback summary 再截到 `1500 chars`
3. `compression.py`  
   - 最多只处理 `28` 张卡
   - 总字符预算 `42000`
   - fallback `core_claim` 只保留 `300 chars`

这意味着“已经下载到的关键正文”并不等于“最终被用于写作的关键正文”。

### 症状 5：DeepXiv progressive reading 只用了 brief，没有真正用到 head/section

位置：

- `src/tools/deepxiv_client.py`
- `src/research/graph/nodes/extract.py`

关键事实：

- client 已提供：
  - `get_paper_brief()`
  - `get_paper_head()`
  - `get_paper_section()`
- 但 extract 主流程实际只批量使用了 `batch_get_briefs()`
- 并没有对重点论文做 targeted section reading

所以当前“progressive reading”还停留在 first step，不足以支撑高质量 methods / evaluation / discussion 写作。

---

## 外部规范依据

### LangGraph 官方流式能力

LangGraph 官方文档明确说明：

1. `stream()` / `astream()` 可以输出实时更新。
2. 可以同时使用 `updates / values / messages / custom / tasks` 等 stream mode。
3. 节点内部还能用 `get_stream_writer()` 发出自定义进度数据。

这意味着“阶段流式输出正文”不是需要自造协议的事情，而是可以直接建立在 LangGraph 官方 streaming 能力之上。

### 长任务上下文工程

Anthropic 在 context management 官方文章里给出的长任务原则很直接：

1. 接近 token limit 时，应自动清理 stale tool results，而不是不断堆上下文。
2. 关键发现应转存到 memory，而不是一直挤在主上下文窗口里。
3. 研究场景的典型做法就是：memory 保存 key findings，context editing 清掉旧搜索结果。

这和当前用户提出的“三点要求”高度一致：

1. 阶段流式输出正文。
2. 合理压缩上下文。
3. 重点论文继续提取、下载、分析，不要在大上下文里盲写。

---

## 代码根因

### 根因 1：draft 仍是一次性大 JSON 生成，timeout 风险天然高

位置：

- `src/research/graph/nodes/draft.py:110-145`

关键事实：

- 最多直接把 `cards[:20]` 送给 LLM
- 注释里自己估算上下文已达约 `26k tokens`
- 单次调用参数是：
  - `max_tokens=16384`
  - `timeout_s=300`

这本质上是“单轮长上下文 + 长输出 + 长超时”的 one-shot drafting。

一旦模型或网关抖动，最容易出的问题就是：

- 524
- 输出 JSON 被截断
- fallback 退化

### 根因 2：当前没有 section-by-section drafting graph

位置：

- `src/research/graph/builder.py:58-124`

主 research graph 只有一个 `draft` 节点，没有：

- section planner
- section drafter
- section reviser
- section merger

因此系统无法：

1. 单独重试 `methods`
2. 单独流式输出 `evaluation`
3. 单独 checkpoint `discussion`

它只能整体成功，或者整体 fallback。

### 根因 3：虽然用了 `graph.stream()`，但没有消费 `messages/custom` 级别的流

位置：

- `src/api/routes/tasks.py:289-311`

当前实现只把 node output 累加进 `state_snapshot`，并没有：

- `stream_mode=["updates", "messages", "custom"]`
- 节点正文 token 流
- 节点自定义 section progress
- 中间 markdown patch

所以“LangGraph 已有流式能力”并没有真正转成产品层的“正文边生成边可见”。

### 根因 4：workspace 同步发生在节点完成后，不是正文增量写入

位置：

- `src/research/agents/supervisor.py:443-512`

关键事实：

- `_build_node_tool()` 里只在 node 完成后 `_sync_node_to_workspace(...)`
- 这适合同步 `brief.json / search_plan.json / paper_cards.json`
- 但不适合长文 section 逐段写入

所以 workspace 当前更像“节点产物存档目录”，不是“长文生成现场”。

### 根因 5：全文下载后只保留少量 snippets，且截断逻辑分散

位置：

- `src/research/graph/nodes/search.py:308-390`
- `src/research/graph/nodes/search.py:510-586`
- `src/research/graph/nodes/extract.py:76-120`
- `src/research/graph/nodes/extract.py:223-340`
- `src/research/graph/nodes/extract.py:623-678`
- `src/research/services/compression.py:32-104`
- `src/research/services/compression.py:272-370`

问题不只是“有截断”，而是：

1. 截断发生在多层。
2. 每层都各自有 budget。
3. 没有统一的 truncation audit artifact。

最终用户无法知道：

- 哪篇重点论文被丢了
- 哪个 section 因 budget 被砍掉
- 证据是在哪一层消失的

### 根因 6：重点论文没有真正进入“持续深读”模式

当前链路更像：

`search metadata -> optional PDF snippets -> extract card -> compress -> one-shot draft`

缺失的是：

- 对关键论文做 `head` 读取
- 针对 `Methods / Experiments / Results / Limitations` 做 section pull
- 在 drafting 前生成“重点论文证据包”

所以即使系统下载了 PDF，也只是抽出少数片段，并没有让重点论文承担更高权重的成文责任。

---

## 修复方向

### 1. 将单个 `draft` 节点拆成 section graph

建议至少拆成：

1. `plan_sections`
2. `draft_abstract`
3. `draft_introduction`
4. `draft_methods`
5. `draft_evaluation`
6. `draft_discussion`
7. `merge_report`
8. `review_sections`

这样可以做到：

- 逐 section checkpoint
- 单 section retry
- 单 section streaming
- 失败时局部回退而不是整篇回退

### 2. 用 LangGraph 官方 streaming 能力做正文流式输出

建议：

- graph 运行时启用 `updates + messages + custom`
- section 节点里使用 `get_stream_writer()`
- SSE 直接向前端/日志发 section progress
- workspace 下写增量文件，例如：
  - `report_sections/introduction.md`
  - `report_sections/methods.md`
  - `report_sections/evaluation.md`

### 3. 建立 section-scoped context assembly

不要再把“整批 cards + taxonomy + pools”一次性喂给同一个大 prompt，而应改成：

- `introduction` 只看 scope + key papers + timeline
- `methods` 只看方法类证据池
- `evaluation` 只看 benchmark/result 证据池
- `discussion` 只看 gap / conflict / limitation 证据池

这样既能降 timeout，也能提升 section 聚焦度。

### 4. 升级重点论文深读策略

应把 progressive reading 真正走完：

1. brief：判断是否值得深读
2. head：识别章节结构
3. section：定向拉取 `Methods / Experiments / Results / Discussion`
4. PDF raw：必要时回退全文 chunk

建议只对 top key papers 做深读，不对所有论文平均用力。

### 5. 建立统一的 truncation audit

建议每次任务输出一个明确 artifact，例如：

- `context_budget.json`

至少记录：

- search 阶段下载正文的论文数
- extract 阶段保留/丢弃的候选
- compression 阶段处理/丢弃的卡片
- 每个 section 实际使用了哪些论文
- 每篇重点论文被截断了多少字符

### 6. 让 timeout 后可以继续，而不是直接模板回退

建议：

- section 级 checkpoint
- 遇到 524 时从上一个完成 section 继续
- 只对失败 section 重试
- 若重试仍失败，保留已完成 section，不重写整篇

---

## 验收标准

### 流式输出

1. research 任务执行中，workspace 能看到逐 section 产出的正文文件，而不是只在最后出现 `report.md`。
2. SSE 不再只包含 node start/end，还能看到 section progress 或正文 token/update 事件。

### 超时恢复

1. survey drafting 出现上游超时时，可以从上一个 section checkpoint 恢复。
2. 不再因为单 section 失败而整篇退回模板。

### 上下文与截断

1. 每次任务都会生成 `context_budget.json` 或同等诊断 artifact。
2. 用户可以定位“重点论文在哪一层被截断、被保留了多少正文证据”。

### 重点论文深读

1. top key papers 至少一部分会经过 targeted section reading，而不是只保留 brief 或摘要。
2. `methods / evaluation / discussion` 段落能够直接体现正文级证据，而非只体现摘要级描述。

---

## 外部参考

- LangGraph Streaming 文档  
  https://docs.langchain.com/oss/python/langgraph/streaming
- Anthropic, “Managing context on the Claude Developer Platform”  
  https://www.anthropic.com/news/context-management
