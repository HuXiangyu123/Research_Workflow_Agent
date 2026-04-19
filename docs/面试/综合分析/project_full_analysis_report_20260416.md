# PaperReader Agent — 项目全仓库深度分析报告（2026-04-16）

> 本报告基于 `docs/features_oncoming/` 和 `docs/issues/` 的最新分析，融合 Context Compression Pipeline、Entropy Management System、Circuit Breaker 三大新特性，以及 LangGraph 合规性问题、工作流架构冗余、Survey 写作质量缺陷等核心 Issue。
> 分析日期：2026-04-16 | 分析范围：`src/`、`frontend/`、`eval/`、`tests/`、`docs/` 及根目录配置文件

---

## 1. 项目概览

### 1.1 项目定位

**PaperReader Agent** 是一个面向科研场景的多阶段 LLM Agent 系统，核心功能是：用户输入研究主题（如"调研 AI Agent 在医疗领域的进展"），系统自动完成**需求澄清 → 检索规划 → 多源论文获取 → 结构化抽取 → 上下文压缩 → 综述生成 → 引用验证 → 报告持久化**的全流程，最终输出带可追溯引用的结构化 Markdown 综述报告。

**核心特征**：
- **多阶段 StateGraph 工作流**（Research Graph + Report Graph 双图并行）
- **Multi-Agent 协作**（ClarifyAgent、SearchPlanAgent、ReviewerAgent、Supervisor）
- **结构化输出 + 引用验证闭环**（citation resolve → claim verify → policy apply）
- **Context Compression Pipeline**（87% 压缩率，解决 context 窗口限制）
- **Entropy Management System**（检测代码腐化）
- **Circuit Breaker**（熔断器保护外部调用）
- **PostgreSQL 持久化**（任务快照、报告、向量 chunk）
- **前端可视化 + SSE 实时推送**（GraphView、TraceTimeline、ReviewFeedbackPanel）
- **三层 Eval 评测体系**（hard rules → LLM grounding judge → human review）

### 1.2 业务背景

科研人员/研究生在写综述论文时，面临的核心痛点：
- 需要从数十篇乃至上百篇论文中提取方法、数据集、贡献、局限
- 手工整理引用关系耗时且容易出错
- 生成的综述报告缺少引用可靠性验证（幻觉风险）

PaperReader Agent 试图解决这些问题。

### 1.3 技术栈

- **Backend**: Python 3.10+, FastAPI, Uvicorn, Pydantic v2, SQLAlchemy 2, LangGraph + LangChain Core
- **LLM**: DeepSeek via OpenAI-compatible API
- **Database**: PostgreSQL only（禁止 SQLite）
- **Retrieval**: FAISS + SearXNG
- **Frontend**: React 19, TypeScript 5, Vite 8, Tailwind CSS 4, @xyflow/react
- **Testing**: pytest, FastAPI TestClient

---

## 2. 核心架构

### 2.1 双 Graph 架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Research Graph (8 节点)                        │
│                                                                      │
│  clarify ──→ search_plan ──→ search ──→ extract ──→ compress        │
│                                        │              │              │
│                                        ↓              ↓              │
│                                      draft ──→ review ──→ persist  │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                         Report Graph (11 节点)                        │
│                                                                      │
│  input_parse → ingest_source → extract_document_text                 │
│           → normalize_metadata → retrieve_evidence                    │
│           → draft_report → repair_report → resolve_citations         │
│           → verify_claims → apply_policy → format_output            │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 节点职责

| 节点 | 核心功能 | 关键文件 |
|------|----------|----------|
| clarify | 需求澄清 → ResearchBrief | `src/research/graph/nodes/clarify.py` |
| search_plan | 查询规划 → SearchPlan | `src/research/graph/nodes/search_plan.py` |
| search | 三源并行检索（SearXNG + arXiv + DeepXiv） | `src/research/graph/nodes/search.py` |
| extract | 批量 LLM 抽取 PaperCards | `src/research/graph/nodes/extract.py` |
| **extract_compression** | 上下文压缩（Taxonomy + CompressedCards + EvidencePools） | `src/research/graph/nodes/extract_compression.py` |
| draft | 结构化报告生成 | `src/research/graph/nodes/draft.py` |
| review | 引用验证 + 质量闸门 | `src/research/graph/nodes/review.py` |
| persist_artifacts | 报告持久化 | `src/research/graph/nodes/persist_artifacts.py` |

---

## 3. 新增特性分析（Features Oncoming）

### 3.1 Context Compression Pipeline ✅ 已实现

**问题背景**：当从 50 篇论文生成综述时，context window 不够用。当前 `draft_node` 的做法是 `cards[:20]` 直接截断，导致后 20 篇论文被忽略。

**解决方案**：`extract_compression_node` 在 extract 和 draft 之间插入上下文压缩：

```
extract_node → extract_compression_node → draft_node
```

**压缩管道**（`src/research/services/compression.py`）：
1. `_build_taxonomy`：用 LLM 将论文按技术路线/子领域分类（~3k chars）
2. `_build_compressed_abstracts`：将每张卡片从 ~1500 chars 压缩到 ~300 chars（87% 压缩率）
3. `_build_evidence_pools`：按 section 分配论文，避免重复引用

**Token 预算分配**：
| Section | 建议约束 |
|---------|----------|
| introduction | 8000 tokens |
| background | 6000 tokens |
| taxonomy | 8000 tokens |
| methods | 10000 tokens |
| datasets | 5000 tokens |
| evaluation | 8000 tokens |
| discussion | 6000 tokens |
| future_work | 4000 tokens |
| conclusion | 3000 tokens |

### 3.2 Entropy Management System ✅ 已实现

**问题背景**：AI Agent 系统运行一段时间后，代码库和文档逐渐偏离原始设计意图。

**四大熵来源**：
1. **文档漂移**：代码改了，文档没改
2. **模式不一致**：不同 Agent 生成的代码风格不同
3. **死代码积累**：重命名节点后旧代码路径仍在
4. **约束侵蚀**：新 import 绕过 .cursorignore 规则

**已实现的检测器**（`src/entropy/`）：
| 检测器 | 功能 |
|--------|------|
| `ConstraintViolationDetector` | 检测硬约束违反（如 SQLite 引入） |
| `DeadCodeDetector` | 检测孤立文件、幽灵节点引用 |
| `DocDriftDetector` | 检测文档漂移 |

**CLI**：`python -m src.entropy.cli scan/check`

### 3.3 Circuit Breaker ✅ 已实现

**问题背景**：当前 agent 系统没有任何熔断机制，存在以下脆弱性：
- 异常静默吞掉（`except: logger.warning`）
- 无失败率追踪
- 无超时级联
- 无故障感知

**熔断器状态机**：
```
CLOSED → OPEN（连续失败 ≥ threshold）
OPEN → HALF_OPEN（timeout 冷却结束）
HALF_OPEN → CLOSED（测试成功）
HALF_OPEN → OPEN（测试失败）
```

**降级策略**：
| API | 降级行为 |
|-----|----------|
| LLM (Reason) | 返回错误，要求用户重试 |
| LLM (Quick) | 降级到 REASON_MODEL |
| SearXNG | 降级到 arXiv 直连搜索 |
| arXiv | 降级到本地 corpus 搜索 |

---

## 4. 核心 Issue 分析

### 4.1 LangGraph 规范合规性问题（P0）

**违规核心**：手工 `AgentSupervisor` 类替代官方 `create_supervisor`

规则原文：
> **Multi-agent supervisor**: Must use `langgraph.supervisor` or `langgraph_sdk.multi_agent`, never implement supervisor logic with Python class + dispatch pattern.

**违规详解**：
1. **手工 `AgentSupervisor` 类**：`src/research/agents/supervisor.py` 定义了 `class AgentSupervisor`，包含 `_node_backends: dict[str, NodeBackend]` dispatch 表
2. **手工 `_merge_state()` 替代官方 `Command` Handoff**：agent 间通过直接覆盖 dict 通信，而非 `Command(goto=agent, graph=Command.PARENT)`
3. **手工 `collaboration_trace` 替代官方 Message History**：messages 列表应该由 LangGraph 自动管理

**合规性评分**：
| 规范项 | 评分 |
|--------|------|
| Supervisor 使用官方 API | 0/10 |
| Worker 使用官方 API | 0/10 |
| Handoff 机制 | 0/10 |
| Message History 管理 | 0/10 |
| 单一 Checkpointer | 0/10 |
| 图编排（不用循环） | 5/10 |
| Checkpoint 接口 | 8/10 |
| TypedDict State | 6/10 |

**总分：19/80 (24%)**

### 4.2 工作流架构冗余问题（P0）

**三层架构冗余**：
1. **Graph 层**（`src/research/graph/builder.py`）：7 节点线性 StateGraph
2. **Supervisor 层**（`src/research/agents/supervisor.py`）：包装 Graph 层的元编排器
3. **Agent 层**（`src/research/agents/*.py`）：5 个独立 LangGraph Agent

**两套 State 定义造成状态冗余**：
```
# 第一处：Graph 层状态
src/graph/state.py::AgentState

# 第二处：Supervisor 层包装状态
src/research/agents/supervisor.py::SupervisorGraphState
  → workflow_state: dict[str, Any]  # 包裹 AgentState 的 untyped dict

# 第三处：每个 Agent 的独立状态
src/research/agents/*_agent.py::XXXGraphState
```

**Supervisor 两套执行路径并行存在**：
- 路径 A：`supervisor.collaborate()` → `build_graph()`（默认，活跃）
- 路径 B：`supervisor.collaborate_with_handoff()` → `build_official_supervisor_graph()`（从未被调用）

**冗余代码清单**：
| 文件 | 代码行 | 描述 |
|------|--------|------|
| `supervisor.py` | 122-129 | `HandoffSupervisorState` TypedDict，从未使用 |
| `supervisor.py` | 358-403 | handoff 相关方法，从未使用 |
| `supervisor.py` | 428-490 | 第二套 supervisor 实现，从未调用 |
| `supervisor.py` | 515-566 | `collaborate_with_handoff()`，从未被调用 |

### 4.3 Survey 写作质量缺陷（P0）

**症状 1：survey 仍是摘要复读，不是学术综述**
- `Abstract`、`Evaluation`、`关键实验` 大量直接复述候选论文摘要
- `Methods` 更像"从摘要里抽关键词后列清单"
- `Discussion` 主要是逐条罗列论文局限，而非跨论文综合分析

**症状 2：离题论文进入最终综述**
- 主题为 "AI agents for medical imaging diagnosis and triage" 的综述中混入联邦学习综述等弱相关论文

**症状 3：review 已经明确发现 ungrounded claims，但流程仍判定通过**
- `passed: true` 但 `9/9 claims are currently ungrounded`

**根因分析**：
1. fallback 与主 drafting 逻辑都仍偏"摘要拼接"
2. review gate 对 unsupported claims 过于宽松
3. skills 在主 research workflow 中几乎被绕开
4. 英文学术写作规范没有成为全链路能力

### 4.4 长文生成超时与阶段流式输出缺失（P0）

**症状**：
1. 最终报告不是稳定长文生成，而是允许 fallback 模板顶替
2. 任务 SSE 只流转节点事件，不流正文内容
3. research 任务只有在整个 supervisor 返回后才把最终 markdown 写回

**根因**：
1. draft 仍是一次性大 JSON 生成，timeout 风险天然高
2. 没有 section-by-section drafting graph
3. 虽然用了 `graph.stream()`，但没有消费 `messages/custom` 级别的流
4. workspace 同步发生在节点完成后，不是正文增量写入

### 4.5 报告语言与 Prompt 英文化缺失（P0）

**症状**：
1. survey 报告中英混写，标题与正文语言不一致
2. single-paper 报告仍完全以中文结构输出
3. survey 报告会在末尾重复输出一遍正文摘要块

**根因**：
1. survey markdown renderer 在正文后硬编码追加第二套中文块
2. survey prompt 自身要求中文标题
3. single-paper 报告链路从 system prompt 到 markdown renderer 都是中文契约
4. agent prompt 层没有完成英文化迁移

---

## 5. 修复优先级与验收标准

### 5.1 最高优先级（P0）

#### 1. 报告语言统一（英文契约）

**验收标准**：
- single-paper 与 search survey 两条链路的最终 `report.md` 都为英文报告
- 报告标题、正文、引用附录、grounding summary 全部使用英文标签
- survey 不再在结尾重复追加第二套摘要块
- 所有 agent prompt 默认输出语言统一为英文

#### 2. Survey 写作质量提升

**验收标准**：
- 正文主体按 themes / methods / debates / evaluation gaps 组织，而不是逐篇摘要
- `Discussion` 必须包含跨论文比较
- `Future work` 必须从 gap 推导，而不是把 limitation 改写

#### 3. review gate 强化

**验收标准**：
- 不允许再出现 `9/9 claims ungrounded` 但 `passed=true` 的结果
- report pass 前必须满足最小 grounding 阈值和主题相关性阈值

### 5.2 高优先级（P1）

#### 4. 长文生成稳定性

**验收标准**：
- section 级 checkpoint
- 遇到 524 时从上一个完成 section 继续
- 只对失败 section 重试

#### 5. Section-scoped 流式输出

**验收标准**：
- research 任务执行中，workspace 能看到逐 section 产出的正文文件
- SSE 不再只包含 node start/end

### 5.3 中期目标（P2）

#### 6. LangGraph 规范合规

**方案 A（推荐）**：使用官方 `langgraph_supervisor`
```python
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

workflow = create_supervisor(
    [clarify_agent, search_agent, draft_agent, review_agent],
    model=model,
    prompt="You coordinate a research workflow..."
)
```

#### 7. 工作流架构简化

**删除冗余**：
- `supervisor.py` 中所有 handoff 相关代码
- `LEGACY_NODE_ALIASES`
- `NodeBackend` Protocol 和未使用的 `_node_backends` 逻辑

---

## 6. 面试价值总结

### 6.1 最值得讲的技术亮点

1. **多阶段 StateGraph 工作流设计**：8 节点 Research Graph + 11 节点 Report Graph
2. **Context Compression Pipeline**：extract_compression_node，87% 压缩率
3. **引用验证闭环（Claim-Level Grounding）**：resolve_citations → verify_claims → apply_policy
4. **三源并行检索**：SearXNG + arXiv API + DeepXiv
5. **Entropy Management System**：检测代码腐化
6. **Circuit Breaker**：熔断器保护外部调用
7. **三层 Eval 体系**：hard rules → LLM judge → human review

### 6.2 最容易被追问的地方

1. **为什么用 LangGraph 而不是自己写 orchestrator？**
2. **Multi-Agent 在哪里？**（当前是"Agent-Enhanced Workflow"）
3. **为什么 Skill framework 有两层但只用了上层？**
4. **Review 失败后怎么处理？**
5. **Re-plan 机制呢？**
6. **LangGraph 合规性如何？**

### 6.3 最容易被质疑的地方

1. **"这是一个 workflow，不是一个 agent"**
2. **LangGraph 合规性评分只有 24%**
3. **Supervisor 有两套执行路径但只有一套被使用**
4. **Eval 的 Layer 2/3 都没在 CI 中运行**
5. **很多 Agent 文件是空壳**

---

## 附录：相关文档

- `docs/features_oncoming/context-compression-for-report-generation.md`
- `docs/features_oncoming/entropy-management.md`
- `docs/features_oncoming/circuit-breaker.md`
- `docs/issues/2026-04-16-langgraph-compliance-audit.md`
- `docs/issues/2026-04-16-langgraph-agent-workflow-audit.md`
- `docs/issues/2026-04-16-survey-writing-quality-skills-mcp-gap-analysis.md`
- `docs/issues/2026-04-16-longform-generation-timeout-and-context-engineering.md`
- `docs/issues/2026-04-16-research-survey-workflow-remediation-order.md`
- `docs/issues/2026-04-16-report-language-and-prompt-english-migration.md`
- `docs/issues/2026-04-16-report-output-workspace-persistence-issue.md`
