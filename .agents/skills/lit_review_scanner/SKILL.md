---
name: lit_review_scanner
description: Run a literature scan over arXiv-oriented search results and return deduplicated candidate papers with lightweight metadata for downstream filtering.
backend: local_function
backend_ref: fn:lit_review_scanner
default_agent: retriever
output_artifact_type: rag_result
visibility: both
tags:
  - retrieval
  - multi-source
  - arxiv
  - academic
  - research
input_schema:
  topic:
    type: string
  sub_questions:
    type: array
  max_results:
    type: integer
  year_filter:
    type: string
---

# Literature Review Scanner

Use this skill when the workflow needs a broad first-pass paper list before stricter reranking.

## Workflow

1. Expand the topic with a small set of sub-question queries.
2. Search arXiv-oriented engines.
3. Deduplicate by URL.
4. Return a compact candidate list for stronger downstream filtering and extraction.
