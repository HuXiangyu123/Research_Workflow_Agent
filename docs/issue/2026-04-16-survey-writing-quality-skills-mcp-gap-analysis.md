# Issue Report: Survey 写作质量、Skills 接入与 MCP 服务化缺口分析

**日期**: 2026-04-16
**类型**: Writing Quality / Skills / MCP Integration
**优先级**: P0
**状态**: 部分修复，仍有质量闸门未通过项

---

## 背景

当前 search survey 工作流已经可以真实跑通并在 workspace 下产出：

- `brief.json`
- `search_plan.json`
- `rag_result.json`
- `paper_cards.json`
- `draft.md`
- `review_feedback.json`
- `report.md`

问题不在“跑不通”，而在“成文质量、技能体系、MCP 体系都还没有真正进入主链路”。

用户提出的关注点可以拆成三层：

1. 综述写法仍像摘要拼接，存在乱编和离题。
2. 英文学术写作规范没有真正落到 agent 行为。
3. skills / MCP 看起来有框架，但没有按标准方式真正服务 research workflow。

---

## 真实症状

### 症状 1：survey 仍是摘要复读，不是学术综述

真实输出文件：

- `output/workspaces/user_20260416T101117Z_d56d60/tasks/511bd283-9f25-45da-b1d8-43a600ccaca5/report.md`

直接表现：

- `Abstract`、`Evaluation`、`关键实验` 大量直接复述候选论文摘要
- `Methods` 更像“从摘要里抽关键词后列清单”
- `Discussion` 主要是逐条罗列论文局限，而非跨论文综合分析

这不是 survey synthesis，而是 candidate-card restatement。

### 症状 2：离题论文进入最终综述

同一份真实输出中可以直接看到主题明显偏离的条目，例如：

- `Securing Generative AI Agentic Workflows: Risks, Mitigation, and a Proposed Firewall Architecture`
- `LLM Agents for Interactive Workflow Provenance: Reference Architecture and Evaluation Methodology`
- `Clinical Productivity System - A Decision Support Model`

它们被写进了一个主题为 `AI agents for medical imaging diagnosis and triage` 的综述正文。

这说明当前 rerank/filter 仍无法稳定执行“领域相关性强约束”。

### 症状 3：review 已经明确发现 ungrounded claims，但流程仍判定通过

真实输出文件：

- `output/workspaces/user_20260416T101117Z_d56d60/tasks/511bd283-9f25-45da-b1d8-43a600ccaca5/review_feedback.json`

关键结果：

- `passed: true`
- 但 issue 同时写明：`9/9 claims are currently ungrounded`

这意味着质量闸门目前无法真正阻止“证据不足但成文继续发布”的情况。

### 症状 4：skills 目录存在，但 skill 实体并未真正被 progressive-disclosure 使用

仓库中确实存在：

- `.agents/skills/lit_review_scanner/SKILL.md`
- `.agents/skills/claim_verification/SKILL.md`
- `.agents/skills/comparison_matrix_builder/SKILL.md`
- `.agents/skills/experiment_replicator/SKILL.md`
- `.agents/skills/writing_scaffold_generator/SKILL.md`

但实际扫描后发现：

- 每个 skill 目录下目前只有 `SKILL.md`
- 没有真正被使用的 `scripts/`、`references/`、`assets/`
- `SKILL.md` 也基本只有 frontmatter，没有技能正文工作流说明

这意味着当前 skill 更接近“元数据占位”，不是可执行的知识包。

### 症状 5：MCP 在代码里有 adapter 和 API，但没有成为工作流中的真实服务能力

截至 **2026-04-16**，仓库并非字面上的“只有 stdio”：

- `src/tools/mcp_adapter.py` 已包含 `StdioTransport`
- 同时也包含 `RemoteHttpTransport`

但问题在于：

1. 没有默认注册的 MCP server 配置。
2. 研究主链路没有稳定调用 MCP server。
3. 远程 HTTP transport 仍只是最薄的一层 POST client，不是完整的服务化接入。

因此从产品效果看，用户的判断仍成立：**主 workflow 目前没有真正依赖一个“在线 MCP 服务体系”在工作。**

---

## 外部规范依据

### 英文学术综述写法

外部写作指南对 literature review / review article 的要求非常一致：

1. Monash University 指出 literature review 通常应有 `introduction / body / conclusion`，正文可按 `themes / debates / methodological issues / concepts` 等方式组织，而不是只有单篇论文摘要堆叠。
2. UMass Amherst Writing Center 明确提醒：不要把 literature review 写成 annotated bibliography 或 laundry list，而应说明 studies 之间如何互相关联，并写 methods 或 results，而不是只说“有这些文献存在”。
3. Elsevier 的 review article guidance 也强调 review article 需要围绕领域问题进行综合、组织和批判性比较，而不是简单收集摘要。

换言之，当前产物的问题不是“风格不够好”，而是 **没有达到 review article 的基本体裁要求**。

### MCP 官方传输规范

MCP 官方 transport 规范说明：

1. 标准 transport 有两种：`stdio` 与 `Streamable HTTP`。
2. Streamable HTTP 侧重“独立运行的服务进程”，支持 `POST + GET`，并可选用 SSE 进行多消息流式回传。
3. 官方规范还要求 session / multiple connections / resumability 等面向服务端运行的能力。

这意味着，如果项目目标是“让 agent 项目调用相关 MCP”，那么仅有一个能 POST `/mcp` 的 client 壳还不够，仍需把 server 配置、服务生命周期、流式消息、会话管理真正接起来。

---

## 代码根因

### 根因 1：fallback 与主 drafting 逻辑都仍偏“摘要拼接”

位置：

- `src/research/graph/nodes/draft.py:110-150`
- `src/research/graph/nodes/draft.py:440-678`

关键事实：

- 主 drafting prompt 虽然写了 “Synthesize instead of paper-by-paper paraphrase”，但输入材料和 fallback 结构仍然强烈鼓励摘要复述。
- fallback `_fallback_draft()` 明确做了以下事情：
  - `abstract`：拼接前几篇摘要
  - `background`：按论文逐条贴摘要
  - `methods`：按方法关键词列清单
  - `evaluation`：逐篇贴摘要
  - `discussion`：逐条列局限性

因此一旦 LLM 质量不足或超时回退，产物天然退化为“摘要集锦”。

### 根因 2：review gate 对 unsupported claims 过于宽松

位置：

- `src/research/services/reviewer.py:132`
- `src/research/services/reviewer.py:342-361`

关键事实：

- `passed` 的判断条件是：只有 `BLOCKER` 或 `ERROR` 才算失败
- 对 `ungrounded claims` 的处理却只是 `WARNING`

这就解释了为什么真实输出里会出现：

- `9/9 claims are currently ungrounded`
- 但 `passed: true`

从学术综述质量角度，这个 gate 过宽，已经不能承担“发布前质量闸门”的职责。

### 根因 3：skills 在主 research workflow 中几乎被绕开

位置：

- `src/skills/orchestrator.py`
- `src/skills/registry.py`
- `src/research/graph/builder.py`
- `src/research/agents/supervisor.py`
- `src/research/agents/analyst_agent.py`

关键事实：

1. 主 research graph 是：
   - `clarify -> search_plan -> search -> extract -> extract_compression -> draft -> review -> persist_artifacts`
2. 这条主链并没有调用 `SkillOrchestrator.orchestrate()`
3. `AnalystAgent` 不是通过 registry 调 skill，而是直接 import `src.skills.research_skills` 里的 Python 函数
4. `SkillOrchestrator` 主要只在 `/api/v1/agents/run` 的 `preferred_skill_id` 分支里才有机会被触发

所以当前 skills 更像“独立 API 功能”，不是工作流原生能力。

### 根因 4：`.agents/skills/*` 只是最薄的 frontmatter 占位，不是完整 skill 包

位置：

- `.agents/skills/*/SKILL.md`
- `src/skills/discovery.py:203-209`

关键事实：

1. 当前五个 skill 目录只有 `SKILL.md`，没有实际脚本和引用资料内容。
2. `SkillMetadataParser.load_skill_content()` 甚至存在明显 bug：
   - 代码里直接访问 `skill_md`
   - 但该变量在函数体内未定义
3. `SkillsRegistry.list_meta()` 返回的只是 manifest 简表，不会注入完整 skill 正文。

这说明当前“skills 标准格式”只是目录形态像，但 progressive-disclosure 机制并没有真正完成。

### 根因 5：skills registry 自身仍有配置漂移和重复注册问题

位置：

- `src/skills/registry.py:190-228`
- `src/skills/registry.py:386-505`

关键事实：

1. `discover_from_filesystem()` 代码块在类内重复了一遍，说明这部分实现仍然粗糙。
2. registry 同时注册了旧 skill 名和新 ARIS 风格 skill 名，形成重复映射：
   - `research_lit_scan` 与 `lit_review_scanner`
   - `paper_plan_builder` 与 `comparison_matrix_builder`

这会让 skill 选择与后续 trace 变得不稳定，也不利于标准化路径。

### 根因 6：MCP adapter 还没有达到“可作为 research workflow 服务底座”的程度

位置：

- `src/tools/mcp_adapter.py:49-100`
- `src/tools/mcp_adapter.py:104-128`
- `src/api/routes/mcp.py:1-111`
- `src/skills/registry.py:86-147`

关键事实：

1. `StdioTransport` 已存在，可以启动本地子进程。
2. `RemoteHttpTransport` 也已存在，但只实现了对 `/mcp` 的 POST。
3. 主 workflow 并没有默认注册任何 MCP server，也没有将 MCP 能力接入 search / extract / draft / review 主节点。
4. skill backend 虽支持 `mcp_toolchain` / `mcp_prompt`，但仓库当前没有真正注册这些 backend 的 skill manifest。

这意味着 MCP 目前还是“可选实验接口”，不是可审计、可依赖、可观测的生产 research tool plane。

### 根因 7：英文学术写作规范虽然有一份 prompt block，但没有成为全链路能力

位置：

- `src/research/prompts/survey_writing.py`
- `src/research/graph/nodes/draft.py`

关键事实：

- 仓库已经有 `SURVEY_WRITING_RULES`
- 但这些规则只在 drafting prompt 局部注入
- 它没有进入：
  - outline 生成
  - section planning
  - evidence allocation
  - review rubric
  - rewrite / revise stage

所以外部写作规范没有闭环成“plan -> draft -> critique -> revise”的全链路写作系统。

---

## 修复方向

### 1. 把“academic review writing”从 prompt 文案升级成显式 workflow 能力

建议把写作阶段拆成明确能力单元：

1. scope-and-thesis planner
2. thematic organizer
3. evidence-backed section drafter
4. cross-paper synthesis critic
5. unsupported-claim blocker

每个能力单元都应输出可审计 artifact，而不是只靠一次性大 prompt。

### 2. 将 `.agents/skills` 变成真正可用的 skill 包

建议每个 skill 至少具备：

- `SKILL.md` 正文工作流说明
- `references/` 中的外部规范依据
- `scripts/` 中可复用的执行脚本或 prompt builder
- `assets/` 中的模板或 rubric

否则它们只是 manifest 占位，不足以支撑 agent 的标准技能调用。

### 3. 让 skills 真正进入主 graph，而不是只留在 API 边缘

建议在主研究链中显式接入：

- search 后：topic pruning / off-topic rejection skill
- extract 后：comparison matrix / experiment replicator skill
- draft 前：writing scaffold / section outline skill
- review 时：claim verification skill

同时把 skill trace 写回 workspace，让用户能看到某段成文是由哪个 skill 支撑的。

### 4. 建立真实的 MCP 服务面，而不只是 transport 类

优先级更高的 MCP server 类型建议是：

1. paper metadata / fulltext reader server
2. targeted section reader server
3. academic writing prompt/rubric server
4. citation / grounding checker server

并要求：

- 可注册
- 可启动
- 可在 workspace 维度追踪
- 可输出调用日志与失败信息

### 5. 强化 review gate

至少应改为：

- `unsupported_claim` 超过阈值时直接 fail
- off-topic paper 比例超阈值时 fail
- 若成文仍是 paper-by-paper list，review 应给出明确 revision action，而不是 warning 放行

---

## 验收标准

### 写作质量

1. 正文主体按 themes / methods / debates / evaluation gaps 组织，而不是逐篇摘要。
2. `Discussion` 必须包含跨论文比较，至少出现“agreement / disagreement / trade-off / evidence gap”中的多项。
3. `Future work` 必须从 gap 推导，而不是把 limitation 改写成将来时。

### skills

1. `.agents/skills/*` 不再只有 frontmatter，占位目录必须具备可执行内容。
2. 主 workflow 的关键阶段会产生可追踪的 skill invocation 记录。
3. `SkillMetadataParser.load_skill_content()` 可正常读取 SKILL 正文，不再是死代码。

### MCP

1. 至少接入一个真实、长期运行的 MCP server 到 research workflow。
2. MCP 调用不再只停留在 `/api/v1/mcp` 调试接口，而会出现在真实任务 trace 中。
3. 远程服务接入满足官方 Streamable HTTP 需要的基本服务语义，至少包括清晰的 endpoint 与流式/会话设计。

### review gate

1. 不允许再出现 `9/9 claims ungrounded` 但 `passed=true` 的结果。
2. report pass 前必须满足最小 grounding 阈值和主题相关性阈值。

---

## 2026-04-16 本轮修复与验证结果

### 已落地修复

1. `.agents/skills/*` 已改为英文标准 skill 包骨架，并补上 `references/` 与 `assets/`。
2. research 主链路已真实接入 skills：
   - `comparison_matrix_builder`
   - `writing_scaffold_generator`
   - `claim_verification`
   - `academic_review_writer_prompt`（通过 stdio MCP server）
3. `academic_writing` MCP server 已默认注册并在真实任务日志中启动调用。
4. `extract` 批量抽取已修复“每批只返回 1 张 card”问题：
   - 返回数量不足会自动 fallback 补齐
   - 原始 metadata 不再被 LLM 改写
5. `arXiv` metadata/fulltext 补齐链路已修复：
   - fallback candidates 会先 enrich `arxiv_id`
   - 真实 run 已出现 `fulltext=8/8`
6. review gate 已收紧：
   - 不再像早期版本那样在大量 unsupported claims 时直接 `passed=true`
   - 最新真实 run 中 `review_passed=false`

### 真实 run 量化结果

1. 基线 run:
   - workspace `user_20260416T115750Z_932f54`
   - task `c60a9959-6d70-403a-a9cc-a4bd29892bed`
   - `rag_score=47.1`
   - `report_score=65.0`
   - `supported_ratio=0.0`
   - `off_topic_ratio=0.941`
2. 最新完整 run:
   - workspace `user_20260416T130349Z_4de09c`
   - task `384972b1-3f61-4416-8f5d-71efee25a37b`
   - `rag_score=89.2`
   - `report_score=82.5`
   - `supported_ratio=0.5`
   - `fulltext_ratio=1.0`
   - `review_passed=false`

### 结论

1. “skills / MCP 没有进入主链路”的问题已被实跑日志证明为已修复。
2. “无正文证据，只能摘要拼接”的问题已经显著改善，当前瓶颈转为证据筛选与 claim grounding，而不是没有 fulltext。
3. “review gate 形同虚设”的问题已修正方向正确，系统开始拒绝低置信度成文。

### 剩余缺口

1. strict-core 主题边界仍需继续收紧：
   - 真实 run 中仍出现边界外论文被保留的情况
   - 已补充 token boundary + year strict drop 修复，但尚未再做一次完整 API run 验证
2. claim grounding 仍不足：
   - 最新 run 仍有 `3/6 claims ungrounded`
   - 说明 drafting 还会生成超出可证据支持范围的总结句
3. durable fulltext persistence 仍有数据库异常：
   - PostgreSQL chunk 写入过程中出现 `NUL (0x00)` 字符错误
   - 当前不阻塞 snippets 回填与写作，但阻塞了部分长期可检索持久化

## 外部参考

- Monash University, “Structuring a literature review”  
  https://www.monash.edu/student-academic-success/excel-at-writing/how-to-write/literature-review/structuring-a-literature-review
- UMass Amherst Writing Center, “Literature Reviews”  
  https://www.umass.edu/writing-center/resources/literature-reviews
- Elsevier Researcher Academy, “An editor’s guide to writing a review article”  
  https://researcheracademy.elsevier.com/writing-research/technical-writing-skills/editor-guide-writing-review-article
- Model Context Protocol Specification, “Transports”  
  https://modelcontextprotocol.io/specification/2025-11-25/basic/transports
