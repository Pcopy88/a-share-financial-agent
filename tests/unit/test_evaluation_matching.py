"""Development regressions for source-bound table relevance; no model or answer keys."""

from types import SimpleNamespace

import pytest


def candidate(text, *, table=True, report_id="r", page=9):
    return SimpleNamespace(chunk={"report_id": report_id, "text": text,
                                 "page_refs": [{"pdf_page_number": page}],
                                 "metadata": {"kind": "table" if table else "text"}})


def gold(quote, *, report_id="r", page=9):
    return [{"id": "g", "report_id": report_id, "pdf_page_number": page, "quote": quote}]


def match(*args, **kwargs):
    from annual_report_agent.evaluation.matching import match_gold
    return match_gold(*args, **kwargs)


def test_table_cells_match_actual_raw_page_without_treating_pipes_as_content():
    row = "茅台酒 145,928,075,955.31 8,662,079,388.78 94.06"
    chunk = candidate("分产品 | 营业收入 | 营业成本 | 毛利率（%）\n"
                      "茅台酒 | 145,928,075,955.31 | 8,662,079,388.78 | 94.06")
    assert match(chunk, gold(row), {"r": {9: "单位：元\n" + row}}) == {"g"}


def test_without_raw_pages_table_matching_keeps_legacy_strict_behavior():
    assert match(candidate("收入 | 100 | 80"), gold("收入 100 80")) == set()
    assert match(candidate("收入 100 80"), gold("收入 100 80")) == {"g"}


def test_non_table_pipes_are_not_normalized():
    assert match(candidate("A | B", table=False), gold("A B"), {"r": {9: "A B"}}) == set()


@pytest.mark.parametrize("pages", [None, {}, {"r": {8: "收入 100 80"}}, {"other": {9: "收入 100 80"}},
                                   {"r": {9: "收入 100 70"}}])
def test_table_formatted_match_requires_quote_in_exact_source_page(pages):
    assert match(candidate("收入 | 100 | 80"), gold("收入 100 80"), pages) == set()


@pytest.mark.parametrize("target", [gold("收入 100 80", report_id="other"),
                                    gold("收入 100 80", page=8)])
def test_hard_report_and_page_constraints_survive_format_matching(target):
    assert match(candidate("收入 | 100 | 80"), target,
                 {"r": {9: "收入 100 80", 8: "收入 100 80"}, "other": {9: "收入 100 80"}}) == set()


@pytest.mark.parametrize("text,quote", [
    ("收入 | 1 | 23", "收入 123"),
    ("收入 | 94 | .06", "收入 94.06"),
    ("收入 | 80 | 100", "收入 100 80"),
    ("A-B | 100", "AB 100"),
    ("收入 | -100", "收入 100"),
    ("收入 | 100.01", "收入 100"),
    ("净利 | 润 | 100", "净利润 100"),
])
def test_matching_never_joins_numeric_cells_reorders_columns_or_drops_punctuation(text, quote):
    assert match(candidate(text), gold(quote), {"r": {9: quote}}) == set()


def test_candidate_rows_cannot_be_spliced_into_a_single_source_row():
    assert match(candidate("收入 | 100\n80 | 20"), gold("收入 100 80"),
                 {"r": {9: "收入 100 80"}}) == set()


def test_real_multiline_quote_keeps_candidate_row_order_and_source_breaks():
    assert match(candidate("收入 | 100\n成本 | 80"), gold("收入 100 成本 80"),
                 {"r": {"9": "收入 100\n成本 80"}}) == {"g"}


def test_wrapped_chinese_label_and_partial_row_numbers_preserve_complete_numbers():
    text = "其他系列 酒 | 24,683,762,096.71 | 4,967,916,424.11 | 79.87"
    quote = "24,683,762,096.71 4,967,916,424.11 79.87"
    assert match(candidate(text), gold(quote), {"r": {9: "其他系列酒 " + quote}}) == {"g"}


def test_raw_page_check_also_does_not_join_distinct_numbers():
    assert match(candidate("收入 | 123"), gold("收入 123"), {"r": {9: "收入 1 23"}}) == set()


def test_literal_pipe_inside_cell_remains_significant():
    assert match(candidate("A|B | 100"), gold("A|B 100"), {"r": {9: "A|B 100"}}) == {"g"}
    assert match(candidate("A|B | 100"), gold("AB 100"), {"r": {9: "AB 100"}}) == set()
