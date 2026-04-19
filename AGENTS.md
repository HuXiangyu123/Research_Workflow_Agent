# AGENTS

## Project Stack

- Backend: Python 3.10+, FastAPI, Uvicorn, Pydantic v2, SQLAlchemy 2
- Agent orchestration: LangGraph + LangChain Core + OpenAI-compatible chat clients
- Primary LLM provider: DeepSeek through an OpenAI-compatible API
- Primary database: PostgreSQL only
- Optional vector extension: pgvector on PostgreSQL
- Local/vector retrieval: FAISS
- Search aggregation: SearXNG
- Frontend: React 19, TypeScript 5, Vite 8, Tailwind CSS 4, `@xyflow/react`
- Markdown/math rendering: `react-markdown`, `remark-gfm`, `remark-math`, `rehype-katex`, `katex`
- Testing: pytest, FastAPI `TestClient`

## Hard Rules

- Do not introduce SQLite for metadata, task state, report persistence, or test fixtures.
- All long-lived persistence must go through PostgreSQL using `DATABASE_URL`.
- Runtime in-memory stores are allowed only for transient execution state or local UI buffering; they are not a substitute for durable storage.
- If a feature needs persistence, implement it in `src/db/*` or a PostgreSQL-backed service layer instead of adding local `.sqlite` files.
- When writing scripts or tests that need environment config, load `.env` explicitly with `load_dotenv(".env")` to avoid implicit path issues.
- Before running any full-flow / end-to-end test, run `python tests/api/check_env_gpt_models.py` and ensure the configured GPT models are callable.
- When updating task/report persistence, keep `/tasks`, `/tasks/{id}`, and `/tasks/{id}/result` behavior aligned.
- Do not encode volatile workflow topology or stage-by-stage architecture notes in this file; keep that material in `docs/`.

## LangGraph Implementation Rules

- **Node orchestration**: Must use `langgraph.graph.StateGraph`, never use Python `for`/`while` loops to simulate graph execution.
- **State persistence**: Must use `langgraph.checkpoint.*` (PostgresSaver / MemorySaver), never use JSON files for durable state.
- **Agent building**: Prefer `langgraph.prebuilt.chat_agent`, `create_react_agent`, or `langgraph_sdk.Agent`, never build agent loops from scratch.
- **Memory**: Must use `langgraph.checkpoint.base.BaseCheckpointSaver`, never implement custom multi-layer memory (SensoryMemory / WorkingMemory / SemanticMemory / EpisodicMemory).
- **Multi-agent supervisor**: Must use `langgraph.supervisor` or `langgraph_sdk.multi_agent`, never implement supervisor logic with Python class + dispatch pattern.
- **Before implementing**: When adding new nodes, agents, or orchestration logic, always check if LangGraph / LangChain has an official API for it first. If yes, use the official API. If no, get approval before custom implementation.
- **Existing violations to fix**: `src/memory/manager.py` (custom JSON memory), `src/research/agents/supervisor.py` (manual for-loop collaboration), `src/research/agents/*.py` (custom agent class patterns) — these should be migrated to LangGraph/LangChain official patterns.
