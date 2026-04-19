---
name: comparison_matrix_builder
description: Build a structured comparison matrix from paper cards so the drafting stage can reason about methods, datasets, benchmarks, and limitations without rereading every abstract.
backend: local_function
backend_ref: fn:comparison_matrix_builder
default_agent: analyst
output_artifact_type: comparison_matrix
visibility: both
tags:
  - analysis
  - comparison
  - papers
  - survey
  - matrix
input_schema:
  paper_cards:
    type: array
  compare_dimensions:
    type: array
  format:
    type: string
---

# Comparison Matrix Builder

Use this skill before drafting whenever the workflow needs a stable, auditable summary of the paper set.

## Workflow

1. Normalize each paper card into a row keyed by paper title.
2. Collect methods, datasets, benchmark signals, and limitations from extracted fields.
3. Mark missing evidence instead of hallucinating it.
4. Return both machine-friendly rows and a human-readable table so downstream prompts can consume either form.

## Output Contract

- `matrix.rows`: structured rows with one paper per row.
- `matrix.table_text`: markdown-ready table string for debugging or prompt injection.
- `missing_fields`: explicit reminders about weak evidence coverage.
