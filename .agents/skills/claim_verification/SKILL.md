---
name: claim_verification
description: Verify scientific claims against report citations and grounding evidence, then return structured claim-level status and aggregate grounding statistics.
backend: local_function
backend_ref: fn:claim_verification
default_agent: reviewer
output_artifact_type: verified_report
visibility: both
tags:
  - verification
  - claims
  - evidence
  - grounding
  - review
input_schema:
  draft_report:
    type: object
  evidence_sources:
    type: array
  claim_ids:
    type: array
---

# Claim Verification

Use this skill when the workflow needs a deterministic pass over grounded claims before deciding whether a report can pass review.

## Workflow

1. Read the structured report claims and citations.
2. Prefer existing grounding annotations when the report already contains `overall_status` or `supports`.
3. Fall back to citation reachability signals only when no claim-level grounding state exists.
4. Return both claim-level statuses and aggregate grounding statistics so the review gate can use thresholds directly.

## Output Contract

- `verified_claims`: one record per claim with `status`, `reason`, and cited labels.
- `grounding_stats`: total, grounded, partial, ungrounded, abstained, grounded ratio, supported ratio.
- `summary`: a short English status line for task/workspace traces.
