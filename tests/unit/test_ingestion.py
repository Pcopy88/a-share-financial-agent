"""Behavioural checks for evidence-preserving parsing and bounded chunks."""
import re

import pytest
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


class WordTokenizer:
    """Deliberate test double; never claims to be BGE or used in live indexes."""
    version = "test:whitespace-v1"

    def offsets(self, text):
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def count(self, text):
        return len(self.offsets(text))


def modules():
    # A missing implementation is an assertion failure, not a collection error.
    from importlib.util import find_spec
    assert find_spec("annual_report_agent.schemas") is not None, "report schemas package missing"
    assert find_spec("annual_report_agent.schemas.report") is not None, "report schema missing"
    from annual_report_agent.ingestion.chunker import chunk_parents
    from annual_report_agent.ingestion.parser import parse_report
    from annual_report_agent.schemas.report import PageRef, ParentBlock, Report
    return PageRef, ParentBlock, Report, chunk_parents, parse_report


def test_text_token_budget_overlap_and_page_provenance():
    PageRef, Parent, _, split, _ = modules()
    text = "zero one two three four five six seven eight nine ten eleven"
    parent = Parent(id="p", report_id="r", section_path="A", text=text,
                    page_refs=[PageRef(pdf_page_number=1), PageRef(pdf_page_number=2)],
                    metadata={"segments": [
                        {"start": 0, "end": 27, "page_ref": {"pdf_page_number": 1}},
                        {"start": 28, "end": len(text), "page_ref": {"pdf_page_number": 2}},
                    ]})
    chunks = split([parent], WordTokenizer(), chunk_size=6, overlap=2)
    assert [c.text for c in chunks] == [
        "zero one two three four five", "four five six seven eight nine",
        "eight nine ten eleven"]
    assert [p.pdf_page_number for p in chunks[0].page_refs] == [1]
    assert [p.pdf_page_number for p in chunks[-1].page_refs] == [2]
    assert all(c.chunk_token_count <= 6 for c in chunks)
    assert all(c.tokenizer_version == "test:whitespace-v1" for c in chunks)


def test_chunks_do_not_overlap_question_boundaries():
    PageRef, Parent, _, split, _ = modules()
    parents = [Parent(id=str(i), report_id="r", section_path=f"问题{i}",
                      question_number=str(i), text=f"answer{i} one two three",
                      page_refs=[PageRef(pdf_page_number=i)]) for i in (1, 2)]
    chunks = split(parents, WordTokenizer(), chunk_size=3, overlap=1)
    assert all("answer1" not in c.text for c in chunks if c.parent_id == "2")
    assert {c.question_number for c in chunks} == {"1", "2"}


def test_table_chunks_repeat_context_and_never_drop_rows():
    PageRef, Parent, _, split, _ = modules()
    parent = Parent(id="t", report_id="r", kind="table", section_path="finance",
                    text="Statement\nCNY\nItem | 2026 | 2025\nRevenue | 10 | 8\nCost | 4 | 3",
                    page_refs=[PageRef(pdf_page_number=4)], metadata={
                        "table_header": "Statement", "unit": "CNY",
                        "rows": [["Item", "2026", "2025"], ["Revenue", "10", "8"], ["Cost", "4", "3"]],
                        "header_rows": [["Item", "2026", "2025"]]})
    chunks = split([parent], WordTokenizer(), chunk_size=12, overlap=2)
    assert len(chunks) == 2
    assert all("Item | 2026 | 2025" in c.text and "CNY" in c.text for c in chunks)
    assert sum("Revenue | 10 | 8" in c.text for c in chunks) == 1
    assert sum("Cost | 4 | 3" in c.text for c in chunks) == 1
    assert all(c.chunk_token_count <= 12 for c in chunks)


def test_overlong_table_cell_is_fully_covered_with_repeated_row_identity():
    PageRef, Parent, _, split, _ = modules()
    long_cell = " ".join(f"word{i}" for i in range(24))
    parent = Parent(id="t", report_id="r", kind="table", section_path="T", text="x",
                    page_refs=[PageRef(pdf_page_number=1)], metadata={
                        "rows": [["Item", "Description", "Amount"], ["Note", long_cell, "100"]],
                        "header_rows": [["Item", "Description", "Amount"]]})
    chunks = split([parent], WordTokenizer(), chunk_size=18, overlap=2)
    assert len(chunks) > 1
    assert all(c.metadata["row_fragment"] for c in chunks)
    assert all("Note" in c.text and "100" in c.text for c in chunks)
    fragments = [f for c in chunks for f in c.metadata["cell_fragments"]
                 if f["column_index"] == 1]
    reconstructed = "".join(long_cell[f["start_char"]:f["end_char"]] for f in fragments)
    assert reconstructed == long_cell
    assert all(c.chunk_token_count <= 18 for c in chunks)


def test_wide_table_header_splits_columns_with_header_sources():
    PageRef, Parent, _, split, _ = modules()
    headers = [["Item", *[f"Column {i}" for i in range(8)]]]
    parent = Parent(id="wide", report_id="r", kind="table", section_path="T", text="wide",
                    page_refs=[PageRef(pdf_page_number=9)], metadata={
                        "rows": [*headers, ["Revenue", *[str(i) for i in range(8)]]],
                        "header_rows": headers})
    chunks = split([parent], WordTokenizer(), chunk_size=12, overlap=2)
    assert all(c.chunk_token_count <= 12 for c in chunks)
    assert all("Revenue" in c.text for c in chunks)
    assert {f["column_index"] for c in chunks for f in c.metadata["cell_fragments"]} == set(range(9))
    assert all(c.metadata["header_column_indices"] for c in chunks)


def test_missing_tokenizer_is_explicit_and_does_not_download(tmp_path):
    modules()
    from annual_report_agent.ingestion.tokenizer import FixedTokenizer, TokenizerUnavailable
    with pytest.raises(TokenizerUnavailable, match="TOKENIZER_UNAVAILABLE"):
        FixedTokenizer.load(tmp_path, allow_download=False)
    assert not list(tmp_path.iterdir())


def test_pdf_raw_text_survives_cleaning_and_blank_page_is_not_success(tmp_path):
    _, _, Report, _, parse = modules()
    path = tmp_path / "fixture.pdf"
    pdf = canvas.Canvas(str(path))
    for number in (1, 2):
        pdf.drawString(50, 810, "Repeated report header")
        pdf.drawString(50, 700, f"Section {number}")
        pdf.drawString(50, 675, f"Useful body evidence {number}")
        pdf.drawString(285, 20, str(number))
        pdf.showPage()
    pdf.showPage()
    pdf.save()
    result = parse(Report(id="r", sha256="fixture", path=str(path), filename=path.name),
                   tokenizer=WordTokenizer())
    assert "Repeated report header" in result.pages[0].raw_text
    assert all("Repeated report header" not in p.text for p in result.parents)
    assert [p.printed_page_label for p in result.pages[:2]] == ["1", "2"]
    assert result.pages[2].parse_status == "needs_ocr"
    assert result.report.status == "partial"
    assert result.chunks and result.report.page_count == 3


def test_document_period_is_read_from_content_not_filename(tmp_path):
    _, _, Report, _, parse = modules()
    path = tmp_path / "2022年度报告.pdf"
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    pdf = canvas.Canvas(str(path))
    pdf.setFont("STSong-Light", 14)
    pdf.drawString(50, 700, "2026年半年度报告")
    pdf.drawString(50, 660, "报告期：2026年1月1日至2026年6月30日")
    pdf.save()
    result = parse(Report(id="r", sha256="fixture", path=str(path), filename=path.name),
                   tokenizer=WordTokenizer())
    assert result.report.doc_type == "semiannual_report"
    assert result.report.fiscal_year == 2026
    assert result.report.report_periods[0]["period_end"] == "2026-06-30"
    assert result.report.inquiry_round is None


def test_new_section_does_not_inherit_financial_statement_scope(tmp_path):
    _, _, Report, _, parse = modules()
    from reportlab.platypus import Table
    path = tmp_path / "different_tables.pdf"
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    pdf = canvas.Canvas(str(path))
    pdf.setFont("STSong-Light", 14)
    pdf.drawString(50, 760, "合并利润表")
    pdf.drawString(50, 740, "单位：元")
    def table(y):
        t = Table([["项目", "2026年", "2025年"], ["营业收入", "10", "8"]],
                  colWidths=[150, 100, 100], rowHeights=[25, 25],
                  style=[("GRID", (0, 0), (-1, -1), 1, "black"),
                         ("FONTNAME", (0, 0), (-1, -1), "STSong-Light")])
        t.wrapOn(pdf, 400, 100)
        t.drawOn(pdf, 50, y)
    table(650)
    pdf.drawString(50, 600, "三、其他资料")
    table(510)
    pdf.save()
    result = parse(Report(id="r", sha256="fixture", path=str(path), filename=path.name),
                   tokenizer=WordTokenizer())
    tables = [p for p in result.parents if p.kind == "table"]
    assert tables[0].metadata["scope"] == "consolidated"
    assert tables[1].metadata["scope"] is None


def test_parse_only_is_independent_of_tokenizer_and_chunking(tmp_path):
    _, _, Report, _, parse = modules()
    path = tmp_path / "parse_only.pdf"
    pdf = canvas.Canvas(str(path))
    pdf.drawString(50, 700, "A very long passage need not invoke any cloud or tokenizer")
    pdf.save()
    result = parse(Report(id="r", sha256="fixture", path=str(path), filename=path.name),
                   split_chunks=False)
    assert result.parents and not result.chunks
    assert "passage" in result.pages[0].raw_text


def test_table_failure_keeps_raw_page_text_for_diagnosis(tmp_path, monkeypatch):
    _, _, Report, _, parse = modules()
    from pdfplumber.page import Page as PlumberPage
    path = tmp_path / "table_error.pdf"
    pdf = canvas.Canvas(str(path))
    pdf.drawString(50, 700, "Evidence must survive a table detector failure")
    pdf.save()

    def fail_tables(self):
        raise ValueError("broken table layout")

    monkeypatch.setattr(PlumberPage, "find_tables", fail_tables)
    result = parse(Report(id="r", sha256="fixture", path=str(path), filename=path.name),
                   split_chunks=False)
    assert result.pages[0].parse_status == "failed"
    assert "Evidence must survive" in result.pages[0].raw_text
    assert result.report.status == "failed"


def test_manifest_period_aliases_match_financial_contract():
    _, _, Report, _, _ = modules()
    report = Report(id="r", sha256="x", path="x.pdf", filename="x.pdf",
                    report_periods=[{"label": "2026H1", "start": "2026-01-01",
                                     "end": "2026-06-30"}])
    assert report.report_periods == [{"label": "2026H1", "period_start": "2026-01-01",
                                      "period_end": "2026-06-30"}]
    with pytest.raises(ValueError, match="conflicting period"):
        Report(id="r", sha256="x", path="x.pdf", filename="x.pdf",
               report_periods=[{"start": "2026-01-01", "period_start": "2025-01-01"}])


def test_long_year_bearing_cells_are_data_not_headers():
    from annual_report_agent.ingestion.parser import _header_rows
    narrative = "2024年签订项目采购合同，合同金额按照框架协议确定。" * 20
    assert _header_rows([["", narrative, ""]]) == []
