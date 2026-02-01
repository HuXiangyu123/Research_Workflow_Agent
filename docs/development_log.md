# 开发日志 (Development Log)

本文档记录了项目从初始化到当前版本的详细开发历程、功能变更与技术决策。

## [0.6.0] - 2026-02-01
### 上下文工程 (Context Engineering)
- **多层上下文注入**: 实现了对话历史、用户长期记忆、文件系统上下文的三层管理架构。
- **Memory 模块**: 
  - 新增 `src/memory/store.py`，实现 `ConversationStore`（会话历史）和 `LongTermMemory`（长期偏好）。
  - 实现了基于 JSON 的本地持久化存储（`data/` 目录）。
  - 增加了敏感信息（如 API Key）的自动过滤机制。
- **文件系统工具**:
  - 新增 `src/tools/local_fs.py`，提供 `search_local_files` 和 `read_local_file` 工具。
  - 实现了安全路径检查，默认忽略 `.env`、`.git` 等敏感目录。
- **Agent 接入**:
  - 改造 `report.py` 和 `cli.py`，在生成报告时动态注入 `extra_system_context`。

## [0.5.0] - 2026-02-01
### 开源化准备 (Open Source Prep)
- **安全清理**: 
  - 移除了仓库中的 `.env` 文件，防止密钥泄露。
  - 更新 `.gitignore`，忽略 `output/`, `data/`, `.env` 等。
- **文档更新**:
  - 更新 `README.md`，提供开源可用的安装、配置与使用指南。
  - 补全 `pyproject.toml` 依赖列表。
- **依赖管理**: 确保所有工具依赖（feedparser, pypdf, python-multipart）均已声明。

## [0.4.0] - 2026-02-01
### 交互体验优化 (UX Improvements)
- **CLI 进度显示**:
  - 新增 `src/agent/callbacks.py`，实现 `AgentProgressCallback`。
  - CLI 运行时实时显示 AI 思考状态与工具调用详情。
- **自动保存**:
  - CLI 生成报告后自动保存为 Markdown 文件。
  - 实现了智能文件名生成：优先使用 arXiv ID（如 `1706.03762报告.md`），失败则回退到论文标题。
  - 自动创建 `output/` 目录存放结果。

## [0.3.0] - 2026-02-01
### 输入源扩展 (Input Extensions)
- **PDF 上传支持**:
  - API 新增 `POST /report/upload_pdf` 接口，支持直接上传 PDF 文件生成报告。
  - 集成 `pypdf` 进行 PDF 文本提取 (`src/tools/pdf.py`)。
- **arXiv 深度集成**:
  - 重构 `src/tools/arxiv_paper.py`，使用 arXiv 官方 API (`export.arxiv.org/api/query`) 替代网页抓取。
  - 实现了对 Atom XML Feed 的解析，精准获取 Title, Authors, Summary 和 PDF Link。
  - 移除了不稳定的 Semantic Scholar 模拟工具。

## [0.2.0] - 2026-02-01
### 架构重构 (Architecture Refactoring)
- **模块化设计**: 将单文件脚本重构为分层架构：
  - `src/agent/`: 核心编排（ReAct Loop, Prompts）。
  - `src/tools/`: 原子能力工具。
  - `src/api/`: FastAPI 服务入口。
  - `src/validators/`: 输出校验逻辑。
- **Web 服务化**:
  - 引入 FastAPI，创建 `src/api/app.py`。
  - 实现了异步非阻塞的报告生成接口。

## [0.1.0] - 2026-02-01
### 项目初始化 (Initialization)
- **环境搭建**: 使用 `conda` 创建 `agent-build` 环境。
- **基础 Agent**: 
  - 基于 `LangGraph` 的 `create_react_agent` 构建基础 ReAct 循环。
  - 集成 `DeepSeek` 模型（通过 ChatOpenAI 兼容接口）。
  - 实现了基础的网页抓取工具 (`fetch_webpage_text`)。
- **核心逻辑**: 定义了“文献报告”生成的 Prompt 和流程。
