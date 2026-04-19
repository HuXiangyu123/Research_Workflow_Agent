---
name: experiment_replicator
description: Inspect extracted paper evidence for datasets, metrics, baselines, and reproducibility signals so the survey can discuss evaluation rigor instead of only reporting claims.
backend: local_function
backend_ref: fn:experiment_replicator
default_agent: analyst
output_artifact_type: experiment_analysis
visibility: both
tags:
  - experiment
  - replication
  - analysis
  - reproducibility
  - datasets
input_schema:
  paper_cards:
    type: array
  focus_papers:
    type: array
---

# Experiment Replicator

Use this skill when the workflow needs a reproducibility-oriented reading of the corpus.

## Workflow

1. Extract datasets, metrics, baselines, and missing evaluation details from the paper cards.
2. Score reproducibility conservatively based on evidence completeness.
3. Surface missing protocol details explicitly so the final report can discuss evidence quality.
