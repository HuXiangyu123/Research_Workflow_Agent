# Architecture

- src/agent：工作流与 Agent 入口
- src/tools：联网、解析、抓取等工具
- src/api：FastAPI 服务入口
- src/retrieval：检索与引用结构

## RAG 资源目录规范 (data/)

```
data/
├── seeds/                  # 种子清单
│   └── seed.jsonl          # 需入库的 arXiv/URL 列表（版本化）
├── corpus/                 # 语料库
│   ├── raw/                # 原始文件 (PDF/HTML/MD)
│   ├── parsed/             # 解析后的中间态 (JSON/TXT)
│   └── chunks/             # 切分后的数据块 (JSONL/Parquet)
├── indexes/                # 检索索引
│   ├── bm25/               # 倒排索引 (Tantivy/Elastic)
│   ├── vector/             # 向量索引 (FAISS/Chroma)
│   └── rerank_cache/       # 重排序结果缓存
├── metadata/               # 元数据
│   └── meta.sqlite         # 文档状态、哈希、引用关系表
└── logs/                   # 检索与运行日志
```

