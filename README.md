# PaperReader Agent

> 面向科研场景的多阶段 LLM Agent 系统：输入研究主题，自动完成需求澄清 → 检索规划 → 多源论文获取 → 结构化抽取 → 上下文压缩 → 综述生成 → Review 把关 → 报告持久化，输出带可追溯引用的结构化 Markdown 综述报告。

---

## 核心功能

| 功能 | 描述 |
|------|------|
| **多阶段工作流** | 8 节点 LangGraph StateGraph：clarify → search_plan → search → extract → compress → draft → review → persist |
| **多源检索** | arXiv API 直连 + 本地 BM25 + Dense Retriever，支持 Cross-Encoder 重排 |
| **上下文压缩** | extract_compression_node，87% 压缩率 |
| **引用验证闭环** | resolve_citations → verify_claims → apply_policy，claim-level grounding |
| **Source Tier 分类** | A/B/C/D 四级权威度评估 |
| **三层评测体系** | Layer 1 (hard rules) → Layer 2 (LLM judge) → Layer 3 (human review) |
| **Circuit Breaker** | 熔断器保护外部 API 调用 |
| **Entropy Management** | 代码腐化检测 |

---

## 技术栈

```
后端
├── LLM              DeepSeek / OpenAI / 字节方舟（通过 OpenAI-compatible API 统一接入）
├── Agent 编排        LangGraph StateGraph + LangChain Core
├── Web 框架         FastAPI + Uvicorn + Pydantic v2
├── 数据库           PostgreSQL + SQLAlchemy 2.0（pgvector 可选）
├── 关键词检索       PostgreSQL 全文搜索（BM25/ts_rank）
├── 向量检索         FAISS（本地）/ pgvector（可选）
├── Embedding        DashScope Qwen API / 本地模型
├── Reranker         Cross-Encoder (ms-marco-MiniLM-L-6-v2)
├── PDF 解析         pypdf
└── 异步 HTTP        httpx / asyncio

前端
├── 框架             React 19 + TypeScript 5
├── 图可视化         @xyflow/react
├── 样式             Tailwind CSS 4
└── Markdown         react-markdown + remark-gfm + rehype-katex + remark-math
```

---

## 快速开始

### 1. 环境配置

```bash
conda create -n paper-reader python=3.11 -y
conda activate paper-reader
pip install -e .
```

### 2. 配置

```bash
cp .env.example .env
```

编辑 `.env`：

```ini
# DeepSeek（默认）
DEEPSEEK_API_KEY=sk-xxxx
DEEPSEEK_API_BASE=https://api.deepseek.com

# 数据库
DATABASE_URL=postgresql://user:pass@127.0.0.1:5432/researchagent
```

### 3. 启动服务

**后端**：
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

**前端**：
```bash
cd frontend && npm install && npm run dev
```

### 4. 使用

打开浏览器访问 `http://localhost:5173`：

- **创建任务**：输入 arXiv 链接/ID 或研究主题
- **实时追踪**：SSE 推送，节点状态实时更新
- **可视化**：@xyflow/react 渲染 LangGraph DAG
- **查看报告**：Markdown 渲染，支持 LaTeX 公式

---

## 工作流图

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Research Graph (8 节点)                        │
│                                                                     │
│  ┌─────────┐   ┌──────────────┐   ┌───────┐   ┌────────────────┐   │
│  │clarify  │──→│ search_plan  │──→│search │──→│    extract     │   │
│  └─────────┘   └──────────────┘   └───┬───┘   └───────┬────────┘   │
│       │                                  │              │           │
│       │ needs_followup?                  │              ↓           │
│       ↓（若需要追问）                     │       ┌─────────────┐     │
│      END                           ┌─────▼─────┐  │  compress   │    │
│                                  │  三源并行   │  └──────┬──────┘     │
│                                  │  去重入库   │         │            │
│                                  └────────────┘         ↓            │
│                                                      ┌───────┐       │
│                                                      │ draft │       │
│                                                      └───┬───┘       │
│                                                          │           │
│                                             review_passed? ↓          │
│                                         ┌─────────────────┐           │
│                                         │     review      │           │
│                                         └────────┬────────┘           │
│                                                  │                    │
│                                                  ↓                    │
│                                         ┌─────────────────┐            │
│                                         │persist_artifacts│            │
│                                         └─────────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
PaperReader Agent/
├── src/
│   ├── agent/          # Agent 编排、Circuit Breaker、Checkpointing
│   ├── research/       # Research Graph (8 节点工作流)
│   │   ├── agents/     # 多 Agent 协作 (ClarifyAgent, SearchPlanAgent...)
│   │   ├── graph/     # StateGraph 构建、节点实现
│   │   └── services/   # ReviewerService, CompressionService
│   ├── tools/          # arXiv API、Web抓取、PDF解析
│   ├── corpus/         # 检索模块 (BM25、Dense Retriever、Cross-Encoder Reranker)
│   ├── embeddings/     # 向量生成 (Qwen API / 本地模型)
│   ├── skills/         # Skills 框架
│   ├── memory/         # 短期/工作区/长期三层记忆
│   ├── db/             # PostgreSQL 持久化
│   ├── entropy/        # Entropy Management System
│   └── api/            # FastAPI 入口
├── frontend/           # React 19 + @xyflow/react 可视化
├── eval/               # 三层评测框架
└── docs/               # 架构文档
```

---

## 技术亮点

1. **多阶段 LangGraph 工作流**：8 节点 DAG，支持条件路由和节点重入
2. **多源混合检索**：BM25 关键词 + Dense 向量 + Cross-Encoder 重排
3. **上下文压缩**：87% 压缩率，解决 context window 限制
4. **引用验证闭环**：claim-level grounding + source tier 权威度评估
5. **三层 Eval 体系**：hard rules → LLM judge → human review
6. **Circuit Breaker + Entropy Management**：生产级稳定性保障

---

## 文档

详细技术文档见 [docs/面试/tech_part/README.md](docs/面试/tech_part/README.md)

---

## License

MIT
