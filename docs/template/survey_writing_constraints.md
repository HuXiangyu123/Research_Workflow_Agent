# Survey Academic Writing Constraints

**Last updated**: 2026-04-17
**Purpose**: This file is the canonical writing contract for research-survey drafting in this repository. It is written for both humans and prompts.

---

## External Basis

The constraint set below is synthesized from:

- Elsevier review-paper guidance:
  - https://legacyfileshare.elsevier.com/promis_misc/ejrh-review-papers.pdf
- Computer Science Review guide for authors:
  - https://www.sciencedirect.com/journal/computer-science-review/publish/guide-for-authors
- Monash literature-review structuring guide:
  - https://www.monash.edu/student-academic-success/excel-at-writing/how-to-write/literature-review/structuring-a-literature-review
- Westcliff review-paper structure guide:
  - https://wijar.westcliff.edu/wp-content/uploads/2020/06/Strucuture-of-a-Review-Paper.pdf

Representative survey templates examined:

- Deep learning for autonomous driving:
  - https://arxiv.org/abs/1910.07738
  - https://ar5iv.org/pdf/1910.07738
- Video question answering survey:
  - https://arxiv.org/abs/2203.01225
  - https://arxiv.org/pdf/2203.01225
- Multimodal biomedicine survey:
  - https://arxiv.org/abs/2311.02332

---

## Non-Negotiable Output Contract

1. The report must be written in English.
2. The report must read as a review article, not as an annotated bibliography.
3. The body must synthesize across papers by themes, method families, datasets, benchmarks, or debates.
4. Claims must stay close to the evidence. Broad field-level claims need multiple directly relevant citations.
5. Background-only or adjacent papers may frame the scope, but they must not dominate methods, evaluation, discussion, or future-work sections.
6. Future work must come from literature gaps, unresolved trade-offs, missing evaluations, or deployment constraints.
7. The conclusion should synthesize takeaways without introducing brand-new evidence.

---

## Canonical Survey Skeleton

Recommended section order:

1. Title
2. Abstract
3. Introduction
4. Background
5. Taxonomy
6. Methods
7. Datasets
8. Evaluation
9. Discussion
10. Future Work
11. Conclusion
12. References

This repo's drafting and rendering should keep that order unless the user explicitly requests a different structure.

---

## Section Writing Moves

### Abstract

- State motivation, scope, corpus boundary, main synthesis, and why the review matters.
- Avoid citations unless the venue strongly expects them.

### Introduction

- Define the topic and explain the review boundary.
- Preview the organizing logic.
- Distinguish in-scope from adjacent work.

### Background

- Only include enabling context needed to understand the later synthesis.
- Do not let background material replace the core survey.

### Taxonomy

- Organize the field into coherent categories.
- Name representative papers for each category.
- Explain the categorization rule explicitly.

### Methods

- Compare architectures, tool use, multimodal fusion, orchestration patterns, and trade-offs.
- Avoid one-paper-per-paragraph summaries whenever possible.

### Datasets

- Summarize dataset families, evaluation settings, and metric choices.
- Explicitly flag missing benchmark detail.

### Evaluation

- Compare evidence quality, not just best reported numbers.
- Separate benchmark performance from clinical or deployment significance.

### Discussion

- Surface agreements, disagreements, trade-offs, evidence gaps, and reproducibility issues.
- This is where cross-paper synthesis should be strongest.

### Future Work

- Derive directions from gaps in the corpus.
- Do not simply rewrite limitation sentences in future tense.

### Conclusion

- Restate the achieved synthesis and practical implications.
- No new citations are required unless absolutely necessary.

---

## Citation and Evidence Contract

### Global Rules

1. Use as many unique in-scope citations as the corpus reasonably supports.
2. When the corpus is at least 8 papers, the main body should usually involve at least 6 unique citations.
3. No single citation should dominate the survey unless it is a foundational benchmark, benchmark paper, or canonical reference that is explicitly being discussed as such.
4. Use adjacent or background papers mainly in `Introduction` and `Background`, not as the main support for central survey claims.
5. A central cross-paper claim should usually cite 2-3 directly relevant papers, not one weakly related citation.

### Section Coverage Floors

The drafting system should try to satisfy these floors when enough papers are available:

| Section | Preferred unique citation floor |
| --- | --- |
| Introduction | 4 |
| Background | 2 |
| Taxonomy | 4 |
| Methods | 5 |
| Datasets | 3 |
| Evaluation | 3 |
| Discussion | 3 |
| Future Work | 2 |

These are soft floors, not rigid publication rules. They exist to stop citation collapse.

### Claim Rules

1. Claims must be verifiable and narrower than the entire field when the evidence is limited.
2. Avoid “the literature proves” style claims unless the cited evidence is broad and direct.
3. If the corpus is mixed, state the boundary explicitly instead of over-generalizing.

---

## Anti-Patterns

Avoid the following:

- Paper-by-paper restatement of abstracts
- Reusing the same 1-2 papers in every major section
- Letting background papers dominate the argument
- Turning limitations into trivial future-work bullets
- Making strong negative claims with only indirect evidence
- Writing datasets and evaluation sections without actual dataset or benchmark evidence

---

## Survey Template Patterns

### Template A: Taxonomy-Driven Technical Survey

Used by many ML / AI surveys.

1. Introduction and scope
2. Background / task definition
3. Taxonomy of methods
4. Detailed method families
5. Datasets and benchmarks
6. Comparative discussion
7. Challenges and future directions
8. Conclusion

Best for:

- fast-moving technical fields
- method-centric corpora
- multimodal / agent / systems reviews

### Template B: Benchmark-and-Evaluation-Centered Survey

1. Introduction and motivation
2. Problem setting
3. Dataset landscape
4. Algorithm families
5. Metrics and evaluation design
6. Comparative results and evidence quality
7. Open issues
8. Conclusion

Best for:

- benchmark-heavy corpora
- survey topics where evaluation methodology is the main bottleneck

### Template C: Clinical / Application Survey

1. Clinical task and deployment context
2. Data modalities and evidence sources
3. Taxonomy of methods
4. Application scenarios
5. Validation and deployment evidence
6. Risks, limitations, and interpretability
7. Future work
8. Conclusion

Best for:

- medical AI surveys
- workflow-oriented or deployment-oriented corpora

---

## Prompt Contract

The block below is intentionally compact so code can inject it directly into prompts.

<!-- SURVEY_PROMPT_RULES_START -->
SURVEY WRITING RULES:
1. Write in English and keep an academic review-article tone.
2. Organize the body by themes, method families, datasets, benchmarks, or debates, not by paper order.
3. Synthesize across papers and explain trade-offs, agreements, disagreements, and evidence gaps.
4. Use background or adjacent papers to frame scope only; do not let them dominate core method or evaluation claims.
5. Maintain citation diversity: do not let a small subset of papers dominate the whole report when the corpus provides broader support.
6. Ensure each major section is supported by section-appropriate citations; methods, datasets, evaluation, and discussion should not all rely on the same tiny citation subset.
7. Keep claims close to the cited evidence. Strong cross-paper claims should usually cite 2-3 directly relevant papers.
8. Future work must come from unresolved gaps, missing evaluations, reproducibility limits, or deployment constraints, not from paraphrased limitation sentences.
9. Prefer representative in-scope papers in the main synthesis, and use adjacent papers only when explicitly labeled as boundary-setting context.
10. The conclusion should synthesize the field status and implications without introducing brand-new evidence.
These rules are grounded in academic review-writing guidance and observed survey-paper structure, not generic blog style.
<!-- SURVEY_PROMPT_RULES_END -->
