# AI Agent 互联网面经与参考帖整合

> 更新时间：2026-04-18  
> 用途：给本项目面试准备、技术宣传、问题设计提供“真实互联网讨论样本”，避免继续闭门造题。

## 说明

这份清单是按“公开可稳定复核、原链接可直接打开、与 AI Agent / RAG / LangGraph / AI 应用开发岗位面试强相关”三个标准筛出来的。

本轮我实际检索了牛客、知乎、小红书等常见社区。结论很明确：

- 公开可稳定复核的原帖，绝大多数来自牛客。
- 知乎和小红书在公开搜索页上的结果要么登录拦截更重，要么索引不稳定，不适合作为这次面试参考库的主来源。
- 因此下面这 20 条以牛客原帖为主，少量“技术准备帖”也保留，因为它们覆盖了面试官最常追问的 Agent 工程话题。

## 可直接参考的 20 条原帖

| 序号 | 类型 | 标题 | 原链接 | 为什么值得看 | 对本项目可借鉴点 |
|---|---|---|---|---|---|
| 1 | 面经 | 初创公司Agent面经 | https://www.nowcoder.com/discuss/874350322167136256 | 直接讨论 Agent 岗位一线提问方式，适合看“多 agent 设计”会被怎么追问。 | 用来反推本项目的 supervisor、skills、tool-use、状态流转讲法。 |
| 2 | 面经 | 字节-火山-AI应用开发工程师-一面 | https://www.nowcoder.com/discuss/873994968799506432 | 偏 AI 应用开发一面，通常会问服务拆分、接口、异步任务、模型调用。 | 对应 `src/api/routes/tasks.py` 的任务接口、后台执行、SSE。 |
| 3 | 面经 | 蚂蚁 AI应用开发 二面 | https://www.nowcoder.com/discuss/874214801747042304 | 二面通常比一面更关注系统边界、稳定性、评测与工程落地。 | 对应本项目的 review gate、workspace persistence、report confidence。 |
| 4 | 面经 | Shopee AI应用开发 一面 | https://www.nowcoder.com/discuss/874594765931634688 | 海外业务侧常追问检索、延迟、成本和稳定性。 | 对应 search node 的三路检索、压缩、降级与输出链路。 |
| 5 | 面经 | 广州美央创新科技有限公司-AI智能体应用开发管培生-AI一面 | https://www.nowcoder.com/discuss/874251962995306496 | 标题就带“AI 智能体应用开发”，岗位贴近本项目。 | 可用于整理“什么算 Agent 应用开发，不只是 prompt chaining”。 |
| 6 | 面经 | 京东 AI应用开发(后端) 一面(JDY) | https://www.nowcoder.com/discuss/874214932156342272 | 偏后端，会落到 API、存储、并发与可恢复性。 | 对应 `/tasks`、PostgreSQL 持久化、workspace artifacts。 |
| 7 | 面经 | 小红书AI后端开发一面 | https://www.nowcoder.com/discuss/873229176470331392 | 产品型团队常会问“生成质量如何保证”和“前后端如何联动”。 | 对应 review、grounding、SSE live preview、GraphView。 |
| 8 | 面经 | 美团 AI应用开发 一面 | https://www.nowcoder.com/discuss/873488423330271232 | 高频会问检索、缓存、任务编排、错误恢复。 | 对应 research/report 双 workflow 和 checkpoint 设计。 |
| 9 | 面试题整理 | AI-Agent 面试题汇总 - 大模型篇 | https://www.nowcoder.com/discuss/860538803759386624 | 适合抽取模型能力边界、推理/工具调用、上下文管理等高频问法。 | 对应 LLM provider、reason/quick model、report confidence 的问答设计。 |
| 10 | 面试题整理 | AI-Agent 面试题汇总 - 机器学习篇 | https://www.nowcoder.com/discuss/860539089378902016 | 适合补齐“只会 Agent 不会模型基础”这个常见短板。 | 可补本项目的 rerank、grounding judge、检索质量评测表述。 |
| 11 | 面试题整理 | AI-Agent 面试题汇总 - Python基础 | https://www.nowcoder.com/discuss/859866308081381376 | 很多 Agent 岗最终还是会追到 Python 工程能力。 | 对应 FastAPI、Pydantic、异步任务、文件与数据库协同。 |
| 12 | 面试复盘 | 28届实习拷打，一场面试，23个Agent问题 | https://www.nowcoder.com/discuss/864153617182355456 | 标题已经说明是高密度 Agent 连环追问。 | 非常适合拿来校验本项目面试题是否够硬。 |
| 13 | 面试题整理 | 都在找Agent开发，我整理了80道相关的Agent开发面试题。 | https://www.nowcoder.com/discuss/867373725035872256 | 可快速看市面上“Agent 开发”被如何粗粒度提问。 | 适合对照本项目，把泛化问题落回具体代码。 |
| 14 | 面试攻略 | 大模型Agent面试全攻略（附答题思路） | https://www.nowcoder.com/discuss/871718560224112640 | 除题目外还给答题思路，适合做结构化训练。 | 可映射到本项目的“背景-设计-风险-效果”回答模板。 |
| 15 | 面试攻略 | Agent 面试会问什么？ | https://www.nowcoder.com/discuss/871128857296785408 | 标题直接命中目标，适合盘点高频主题。 | 用来校准问题覆盖面，避免只盯 LangGraph 不讲产品链路。 |
| 16 | 面经整理 | Ai开发面经整理，拷打一小时汗流浃背! (附回答总结) | https://www.nowcoder.com/discuss/864551829986627584 | 有“被拷打”和“附回答总结”两个维度，比较适合做模拟问答。 | 可对照本项目的任务流、引用、输出和评测问法。 |
| 17 | 准备帖 | 手把手教你准备 Ai 面试（2026届校招版） | https://www.nowcoder.com/discuss/782208437957509120 | 不只是题库，更偏体系化准备。 | 适合把本项目面试资料组织成“先总览、再模块、再追问”的结构。 |
| 18 | 学习路线 | 26年全网最全Agent学习路线，拿走不谢! | https://www.nowcoder.com/discuss/864821937527128064 | 虽不是严格面经，但能反映候选人常见学习路径。 | 帮助区分“学过 Agent”与“真正做过多阶段工作流”的表达差异。 |
| 19 | 技术帖 | 用 LangGraph 搭一套企业级 Coding Workflow，聊聊我的思路 | https://www.nowcoder.com/discuss/857246908849352704 | 直接命中 LangGraph 工作流工程化。 | 对应本项目 `StateGraph`、supervisor、checkpoint、节点编排。 |
| 20 | 技术帖 | 想学AI Agent？先从LangChain入手 | https://www.nowcoder.com/discuss/860859066308923392 | 补齐 LangChain / tool / agent 基础认知。 | 对应本项目的 `create_react_agent`、skills registry、MCP adapter。 |

## 这 20 条帖子暴露出的高频面试主题

结合上面的样本，可以把真实互联网面经里的追问，归纳成下面 8 类：

1. 你做的到底是不是“真正的 Agent”，还是 prompt + API 的自动化脚本。
2. 多 agent 是怎么分工的，状态怎么流动，谁做 supervisor，为什么这样拆。
3. 检索怎么做，多源搜索怎么去重，怎样控制 off-topic 和 hallucination。
4. 生成质量怎么兜底，尤其是引用、grounding、review gate、confidence。
5. 为什么要工作流图而不是 Python 手写流程循环，为什么选 LangGraph。
6. 前后端怎么联动，运行过程怎么可视化，是否支持流式输出和中间产物查看。
7. 数据和产物怎么持久化，workspace 和数据库各存什么，失败后如何恢复。
8. 技术选型如何解释，包括模型接入、工具系统、skills、MCP、评测和超时处理。

## 用这份清单时的建议

- 如果准备项目面试，不要把这些帖子当标准答案库，而要把它们当“真实追问分布样本”。
- 对本项目来说，最值得重点准备的不是泛化 Agent 定义，而是下面五条主链路：
  - `/tasks` 异步任务入口与结果对齐
  - `research graph` 8 节点编排
  - `search -> extract -> compression -> draft -> review`
  - `workspace artifacts + PostgreSQL snapshots`
  - `SSE + live preview + post-report chat/skills`
- 后续更新 100 道项目定制面试题时，应优先覆盖这五条主链路，而不是继续罗列泛化八股。
