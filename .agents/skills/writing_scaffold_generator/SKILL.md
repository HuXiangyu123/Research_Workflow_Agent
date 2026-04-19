---
name: writing_scaffold_generator
description: Generate an English survey-writing scaffold, section plan, and evidence map so the drafting stage can stay thematic and evidence-led.
backend: local_function
backend_ref: fn:writing_scaffold_generator
default_agent: analyst
output_artifact_type: report_outline
visibility: both
tags:
  - writing
  - scaffold
  - outline
  - survey
  - generation
input_schema:
  topic:
    type: string
  paper_cards:
    type: array
  comparison_matrix:
    type: object
  desired_length:
    type: string
---

# Writing Scaffold Generator

Use this skill immediately before drafting so the report is organized as a review article instead of a paper-by-paper list.

## Workflow

1. Read the topic and comparison matrix.
2. Derive section goals from recurring method families, benchmark patterns, and evidence gaps.
3. Produce an English section scaffold and evidence map.
4. Keep the output short enough for prompt injection but explicit enough to control structure.

## Output Contract

- `scaffold`: section-level writing plan.
- `outline`: ordered section headings.
- `section_evidence_map`: which papers should anchor each major section.
- `writing_guidance`: compact rules for the drafter.
