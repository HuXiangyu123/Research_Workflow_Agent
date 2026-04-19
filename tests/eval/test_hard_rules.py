from eval.layers.hard_rules import (
    check_citation_format,
    check_cost_guard,
    check_min_citations,
    check_must_include,
    check_structure,
    run_layer1,
)

GOOD_REPORT = """# Attention Is All You Need

## Abstract and Motivation

Transformer architecture overview

## Methods
Self-attention mechanism

## Experiments

BLEU score improvements

## Discussion and Future Directions

High compute cost

## References

- [1] https://arxiv.org/abs/1706.03762 — original paper
- [2] https://doi.org/10.1234 — related work
- [3] https://github.com/tensorflow/tensor2tensor — implementation
"""


def test_structure_complete():
    result = check_structure(GOOD_REPORT)
    assert result["pass"] is True
    assert len(result["missing"]) == 0


def test_structure_missing():
    partial = "# Test\n\n## Abstract and Motivation\n\nContrib"
    result = check_structure(partial)
    assert result["pass"] is False
    assert len(result["missing"]) > 0


def test_citation_format_good():
    result = check_citation_format(GOOD_REPORT)
    assert result["pass"] is True
    assert result["citation_count"] >= 3


def test_citation_format_missing():
    result = check_citation_format("# Title\n\nNo citations here")
    assert result["pass"] is False


def test_must_include_pass():
    result = check_must_include(GOOD_REPORT, ["Transformer", "attention"])
    assert result["pass"] is True


def test_must_include_fail():
    result = check_must_include(GOOD_REPORT, ["nonexistent_keyword"])
    assert result["pass"] is False


def test_min_citations_pass():
    result = check_min_citations(GOOD_REPORT, 3)
    assert result["pass"] is True


def test_min_citations_fail():
    result = check_min_citations(GOOD_REPORT, 100)
    assert result["pass"] is False


def test_cost_guard_pass():
    result = check_cost_guard(10000)
    assert result["pass"] is True


def test_cost_guard_fail():
    result = check_cost_guard(100000)
    assert result["pass"] is False


def test_run_layer1_good():
    case = {"must_include": ["Transformer"], "min_citations": 2}
    result = run_layer1(GOOD_REPORT, case)
    assert result["pass"] is True


def test_run_layer1_error_case():
    case = {"expect_error": True}
    result = run_layer1(
        "Error generating report:\ninput_parse: cannot determine source", case
    )
    assert result["pass"] is True


def test_run_layer1_error_case_not_error():
    case = {"expect_error": True}
    result = run_layer1("# Title\n\nSome normal report", case)
    assert result["pass"] is False
