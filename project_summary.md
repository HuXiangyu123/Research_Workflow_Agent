# 项目进度总结报告：文献报告智能 Agent

## 1. 项目概况

本项目旨在构建一个智能文献报告生成 Agent，能够根据用户提供的 arXiv 链接，自动检索相关文献、生成结构化笔记，并提供可追溯的引用来源。项目采用模块化设计，基于 LangGraph 和 LangChain 框架，并集成 DeepSeek 大模型能力。

## 2. 核心应用需求与解决方案

| 需求点 | 解决方案 | 实现状态 |
| :--- | :--- | :--- |
| **自动化文献获取** | 通过 arXiv ID 自动获取论文元数据（标题、摘要、作者等）。 | ✅ 已实现 (src/tools/arxiv.py) |
| **关联文献扩展** | 利用 Semantic Scholar 检索相关工作，扩展阅读视野。 | ✅ 已实现 (src/tools/semantic_scholar.py) |
| **多源信息整合** | 结合 arXiv 元数据、Semantic Scholar 检索结果及网络抓取内容。 | ✅ 已实现 (src/agent/react_agent.py) |
| **结构化报告生成** | 生成包含核心贡献、方法、实验结果、相关工作及引用的 Markdown 报告。 | ✅ 已实现 (src/agent/report.py) |
| **可追溯引用** | 在报告末尾强制生成引用列表，尝试建立文本与来源的链接。 | ✅ 初步实现 (基于 Prompt 约束) |
| **服务化接口** | 提供标准 API 供前端或其他系统调用。 | ✅ 已实现 (FastAPI) |

## 3. 模块解析与代码结构

项目遵循清晰的分层架构：

### 3.1 `src/agent/` - 核心编排层
- **[react_agent.py](file:///e:/devproj/agent/src/agent/react_agent.py)**: 定义了基于 LangGraph 的 ReAct Agent。
  - 使用 `create_react_agent` 构建循环工作流。
  - 集成了 DeepSeek LLM 和自定义工具集。
- **[report.py](file:///e:/devproj/agent/src/agent/report.py)**: 封装了报告生成的高级逻辑。
  - 负责调用 Agent 执行任务。
  - 对生成结果进行后处理（如验证引用格式）。
- **[llm.py](file:///e:/devproj/agent/src/agent/llm.py)** & **[settings.py](file:///e:/devproj/agent/src/agent/settings.py)**: 负责模型配置与环境变量管理，确保 DeepSeek 接口的兼容性。
- **[cli.py](file:///e:/devproj/agent/src/agent/cli.py)**: 命令行交互入口，方便本地测试与调试。

### 3.2 `src/tools/` - 工具层
- **[arxiv.py](file:///e:/devproj/agent/src/tools/arxiv.py)**: `get_arxiv_paper_info` 工具，用于解析 arXiv 页面或 API 获取论文基础信息。
- **[semantic_scholar.py](file:///e:/devproj/agent/src/tools/semantic_scholar.py)**: `search_related_works` 工具，模拟搜索相关文献（当前为 Mock 实现，待接入真实 API）。
- **[web_fetch.py](file:///e:/devproj/agent/src/tools/web_fetch.py)**: `fetch_web_content` 工具，用于通用的网页内容抓取（当前为简单模拟）。

### 3.3 `src/api/` - 接口层
- **[app.py](file:///e:/devproj/agent/src/api/app.py)**: FastAPI 应用入口。
  - 提供 `POST /report` 接口，接收 arXiv 链接，返回生成的报告内容。
  - 支持异步非阻塞调用。

### 3.4 `src/validators/` & `src/retrieval/` - 质量控制层
- **[citations_validator.py](file:///e:/devproj/agent/src/validators/citations_validator.py)**: 用于检查生成的报告是否包含合规的引用部分。
- **[citations.py](file:///e:/devproj/agent/src/retrieval/citations.py)**: 定义引用数据结构（目前主要作为数据模型）。

## 4. 关键 Agent 技术要点

1.  **ReAct 范式 (Reasoning + Acting)**
    - 利用 LangGraph 的 `prebuilt.create_react_agent` 实现。
    - 模型根据当前任务状态，动态决定是“查 arXiv”、“搜相关工作”还是“生成最终回答”。
    - **DeepSeek 适配**: 通过 `ChatOpenAI` 兼容层，配置 `base_url` 和 `api_key` 完美适配 DeepSeek 模型，利用其强大的推理能力进行工具选择。

2.  **工具增强生成 (RAG 变体)**
    - Agent 不仅依赖训练数据，还实时调用外部工具（arXiv, Semantic Scholar）获取最新信息。
    - 将检索到的上下文（Context）注入到 Prompt 中，指导模型生成基于事实的报告。

3.  **流式输出 (Streaming)**
    - 支持 LangGraph 的 `stream` 模式，实时反馈 Agent 的思考过程（中间步骤）和最终结果，提升用户体验。

4.  **结构化输出约束**
    - 通过 Prompt Engineering (在 `report.py` 中) 强约束模型输出 Markdown 格式。
    - 包含“核心贡献”、“实验结果”、“引用”等特定章节要求。

## 5. 后续演进方向

-   **真实 API 接入**: 替换 Semantic Scholar 和 Web Fetch 的 Mock 实现为真实调用。
-   **更强的引用校验**: 实现基于嵌入（Embedding）的引用一致性检查，确保引用内容真实存在于检索文档中（Fact-Checking）。
-   **多步规划**: 从 ReAct 升级为 Plan-and-Solve 模式，先生成大纲再分段撰写，应对超长文献综述。
-   **前端界面**: 开发一个简单的 Web UI 可视化报告生成过程。
