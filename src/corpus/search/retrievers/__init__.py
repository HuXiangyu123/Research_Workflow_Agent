"""Corpus Search 模块 — Paper-level Retrieval。

模块职责：
- 根据 query / sub_questions / filters 组合召回候选论文集
- 输出高召回论文候选池（InitialPaperCandidates），为模块 5 打底

目录结构：
    retrievers/
        models.py          — 数据模型（InitialPaperCandidates, MergedCandidate, RetrievalTrace...）
        query_prep.py      — Query Preparation
        keyword_retriever.py  — Keyword/BM25 recall
        dense_retriever.py    — Dense vector recall
        filter_compiler.py    — Metadata filter 编译
        candidate_merger.py   — 多路 merge + source attribution
        trace_builder.py      — Retrieval trace 记录
        paper_retriever.py    — 统一入口
"""
