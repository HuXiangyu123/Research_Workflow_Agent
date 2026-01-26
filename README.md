# Literature Report Agent (LangGraph + DeepSeek)

一个文献报告 Agent：输入 arXiv 链接/ID（或上传 PDF），自动生成结构化 Markdown 报告，并附带可追溯引用。

## 功能

- arXiv 元信息获取：通过 arXiv API Query 拉取 title/authors/summary/pdf 链接等
- 报告生成：输出中文结构化笔记 + 引用列表（Markdown）
- CLI：交互式输入 arXiv 链接，报告自动保存到 output 目录
- API：FastAPI 提供 `/report` 与 `/report/upload_pdf`

## 环境

本项目默认使用 Conda 环境 `agent-build`。每次运行前请先激活环境：

```powershell
conda activate agent-build
```

## 安装

使用 `pyproject.toml` 管理依赖：

```powershell
pip install -e .
```

## 配置

1) 从模板复制配置文件：

```powershell
copy .env.example .env
```

2) 在 `.env` 中填入你的 DeepSeek Key（仅本地保存，禁止提交到仓库）：

```ini
DEEPSEEK_API_KEY=sk-xxxx
DEEPSEEK_API_BASE=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

说明：
- `.env` 已在 `.gitignore` 中忽略；请确保不要把真实密钥提交到 Git
- 若密钥曾经暴露，请立即在控制台轮换（rotate）并作废旧密钥

## 使用（CLI）

```powershell
conda activate agent-build
python -m src.agent.cli
```

- 输入示例：`https://arxiv.org/abs/1706.03762`
- 输出位置：`output/1706.03762报告.md`

## 使用（API）

启动服务：

```powershell
conda activate agent-build
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

生成报告（arXiv 链接/ID）：

```bash
curl -X POST "http://127.0.0.1:8000/report" \
  -H "Content-Type: application/json" \
  -d "{\"arxiv_url_or_id\":\"https://arxiv.org/abs/1706.03762\"}"
```

上传 PDF 生成报告：

```bash
curl -X POST "http://127.0.0.1:8000/report/upload_pdf" \
  -F "file=@./paper.pdf"
```

## 代码结构

- `src/agent`: Agent 编排与报告生成
- `src/tools`: arXiv / 网页抓取 / PDF 解析工具
- `src/api`: FastAPI 服务入口

## 未来更新计划

- 添加更多工具： / 引用校验 / 引用关系图
- 实现rag引入
- 实现更多模型接口
- 引入latex公式适配