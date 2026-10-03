"""Independent layout fixtures and optional real-page financial parsing regressions."""

import json
from pathlib import Path

import pdfplumber
import pytest
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

from annual_report_agent.finance.pdf_facts import extract_pdf_facts
from annual_report_agent.ingestion.parser import parse_report
from annual_report_agent.schemas.report import Report


def make_statement(path, *, vertical=True, ambiguous=False, left_border=False):
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    pdf = canvas.Canvas(str(path), pagesize=(600, 800))
    pdf.setFont("STSong-Light", 12)
    pdf.drawString(240, 755, "合并利润表")
    pdf.drawString(50, 730, "编制单位：测试公司")
    pdf.drawRightString(550, 730, "单位：人民币元")
    if vertical:
        for y in (710, 685, 540):
            pdf.line(40, y, 560, y)
        for x in (260, 330, 445):
            pdf.line(x, 540, x, 710)
        if left_border:
            pdf.line(40, 540, 40, 710)
    for x, text in [(120, "项目"), (275, "附注"), (348, "2026年度"),
                    (458, "2026年度" if ambiguous else "2025年度")]:
        pdf.drawString(x, 692, text)
    for y, label, note, current, previous in [
        (665, "其中：营业收入", "五、1", "1,234.56", "987.65"),
        (630, "其中：营业成本", "五、2", "600.10", "500.00"),
        (595, "五、净利润", "", "200.20", "150.15"),
        (560, "归属于母公司股东的净利润", "", "180.20", "140.15"),
    ]:
        pdf.drawString(50, y, label)
        pdf.drawString(277, y, note)
        pdf.drawRightString(435, y, current)
        pdf.drawRightString(550, y, previous)
    pdf.drawString(50, 518, "法定代表人：")
    pdf.drawString(245, 22, "第1页 共1页")
    pdf.save()


def parse(path, pages=None):
    return parse_report(Report(id="layout-test", sha256="layout-test", path=str(path), filename=path.name,
                               company="测试公司", doc_type="annual_report", fiscal_year=2026),
                        split_chunks=False, page_numbers=pages)


@pytest.mark.parametrize("vertical", [True, False])
def test_open_or_absent_borders_keep_all_financial_columns_and_provenance(tmp_path, vertical):
    path = tmp_path / "layout.pdf"
    make_statement(path, vertical=vertical)
    parsed = parse(path)
    tables = [parent for parent in parsed.parents if parent.kind == "table"]
    facts = extract_pdf_facts(parsed)
    assert len(facts) == 8
    assert {(f.metric_code, str(f.value), str(f.period_end)) for f in facts} >= {
        ("revenue", "1234.56", "2026-12-31"), ("revenue", "987.65", "2025-12-31"),
        ("operating_cost", "600.10", "2026-12-31"), ("net_profit", "200.20", "2026-12-31"),
    }
    assert all(f.validation_status == "pending" for f in facts)
    assert tables[0].metadata["rows"][0] == ["项目", "附注", "2026年度", "2025年度"]
    assert tables[0].metadata["row_bboxes"] and tables[0].metadata["cell_bboxes"]
    assert tables[0].metadata["unit"] == "元"
    assert parsed.pages[0].printed_page_label == "1"


def test_ambiguous_duplicate_period_headers_remain_unusable(tmp_path):
    path = tmp_path / "ambiguous.pdf"
    make_statement(path, ambiguous=True)
    parsed = parse(path)
    diagnostics = []
    assert extract_pdf_facts(parsed, diagnostics=diagnostics) == []
    assert any(parent.metadata.get("structure_status") == "needs_review" for parent in parsed.parents)
    assert any("uncertain_table_structure" in item["reasons"] for item in diagnostics)


def test_missing_right_border_does_not_drop_last_period_column(tmp_path):
    path = tmp_path / "missing_right.pdf"
    make_statement(path, left_border=True)
    parsed = parse(path)
    facts = extract_pdf_facts(parsed)
    assert len(facts) == 8
    assert {str(f.value) for f in facts if f.metric_code == "revenue"} == {"1234.56", "987.65"}


def test_recovered_header_keeps_its_source_page_on_ruled_continuation(tmp_path):
    from reportlab.platypus import Table

    path = tmp_path / "continued.pdf"
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    pdf = canvas.Canvas(str(path), pagesize=(600, 800))
    for page, rows in enumerate([
        [["项目", "附注", "2026年度", "2025年度"],
         ["营业收入", "五、1", "100.00", "80.00"],
         ["营业成本", "五、2", "60.00", "50.00"]],
        [["净利润", "", "20.00", "15.00"],
         ["归属于母公司股东的净利润", "", "18.00", "14.00"]],
    ], 1):
        pdf.setFont("STSong-Light", 12)
        if page == 1:
            pdf.drawString(240, 755, "合并利润表")
            pdf.drawString(450, 725, "单位：元")
        style = [("FONTNAME", (0, 0), (-1, -1), "STSong-Light")]
        if page == 1:
            style.extend([("LINEABOVE", (0, 0), (-1, 0), 1, "black"),
                          ("LINEBELOW", (0, 0), (-1, 0), 1, "black"),
                          ("LINEBELOW", (0, -1), (-1, -1), 1, "black"),
                          ("LINEAFTER", (0, 0), (-2, -1), 1, "black")])
        else:
            style.append(("GRID", (0, 0), (-1, -1), 1, "black"))
        table = Table(rows, colWidths=[220, 70, 115, 115], rowHeights=30, style=style)
        table.wrapOn(pdf, 520, 100)
        table.drawOn(pdf, 40, 700 - 30 * len(rows))
        pdf.drawString(290, 22, str(page))
        pdf.showPage()
    pdf.save()
    parsed = parse(path)
    continued = next(p for p in parsed.parents if p.kind == "table" and p.page_refs[0].pdf_page_number == 2)
    assert continued.metadata["header_inherited"] is True
    assert continued.metadata["header_page_refs"][0]["pdf_page_number"] == 1
    assert continued.metadata["title_page_refs"][0]["pdf_page_number"] == 1
    facts = [f for f in extract_pdf_facts(parsed) if f.metric_code == "net_profit"]
    assert {str(f.value) for f in facts} == {"20.00", "15.00"}
    assert all(f.provenance["pdf_page_number"] == 2 for f in facts)
    assert all(f.provenance["header_page_refs"][0]["pdf_page_number"] == 1 for f in facts)
    assert extract_pdf_facts(parse(path, [2])) == []


def test_existing_ruled_table_with_leading_padding_is_not_replaced(tmp_path):
    from reportlab.platypus import Table

    path = tmp_path / "padded.pdf"
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    pdf = canvas.Canvas(str(path), pagesize=(600, 800))
    pdf.setFont("STSong-Light", 12)
    pdf.drawString(240, 755, "合并利润表")
    rows = [["", "项目", "2026年度", "2025年度"],
            ["", "营业收入", "100.00", "80.00"], ["", "营业成本", "60.00", "50.00"]]
    table = Table(rows, colWidths=[20, 200, 130, 130], rowHeights=30,
                  style=[("GRID", (0, 0), (-1, -1), 1, "black"),
                         ("FONTNAME", (0, 0), (-1, -1), "STSong-Light")])
    table.wrapOn(pdf, 520, 100)
    table.drawOn(pdf, 40, 610)
    pdf.save()
    parsed = parse(path)
    parent = next(p for p in parsed.parents if p.kind == "table")
    assert parent.metadata["rows"] == rows
    assert parent.metadata["structure_status"] == "candidate"
    assert "table_detection" not in parent.metadata


@pytest.mark.parametrize("number", [173, 175, 177, 180, 181])
def test_real_prospectus_ruled_tables_preserve_original_cell_structure(number):
    path = Path("D:/Desktop/南方乳业：招股说明书.pdf")
    if not path.exists():
        pytest.skip("optional local prospectus development source is unavailable")
    with pdfplumber.open(path) as pdf:
        original = [table.extract() for table in pdf.pages[number - 1].find_tables()]
    parsed = parse(path, [number])
    tables = [p for p in parsed.parents if p.kind == "table"]
    assert [p.metadata["rows"] for p in tables] == original
    assert all("table_detection" not in p.metadata for p in tables)


@pytest.mark.parametrize("year,pages,current_revenue,current_cost,comparative_cost,beginning_date", [
    (2023, [114, 115, 116, 117], "203979266387.09", "141625549746.95", "139784387882.78", "2023-01-01"),
    (2024, [111, 112, 113, 114], "189163654064.64", "133496119623.36", "143953487220.05", "2024-01-01"),
])
def test_real_open_border_statements_keep_report_specific_comparatives(
    year, pages, current_revenue, current_cost, comparative_cost, beginning_date,
):
    path = Path(f"data/samples/annual/gree_{year}.pdf")
    if not path.exists():
        pytest.skip("optional local annual-report development source is unavailable")
    parsed = parse(path, pages)
    parsed.report.fiscal_year = year
    facts = extract_pdf_facts(parsed)
    amounts = {(f.metric_code, str(f.period_end)): str(f.value) for f in facts}
    assert amounts[("revenue", f"{year}-12-31")] == current_revenue
    assert amounts[("operating_cost", f"{year}-12-31")] == current_cost
    assert amounts[("operating_cost", f"{year-1}-12-31")] == comparative_cost
    assert ("total_assets", beginning_date) in amounts
    assert any(f.metric_code == "operating_cash_flow" for f in facts)
    assert all(f.validation_status == "pending" for f in facts)
    assert {f.provenance["pdf_page_number"] for f in facts} == set(pages)
    reference = json.loads(Path(f"evals/gree_{year}_reference.json").read_text(encoding="utf-8"))
    actual = {(f.metric_code, str(f.period_end), f.scope): f for f in facts}
    assert len(facts) == len(reference["facts"]) == 18
    for expected in reference["facts"]:
        fact = actual[expected["metric_code"], expected["period_end"], expected["scope"]]
        assert str(fact.value) == expected["value"]
        assert (fact.period_start.isoformat() if fact.period_start else None) == expected["period_start"]
        assert fact.unit == expected["unit"]
        assert fact.provenance["pdf_page_number"] == expected["source"]["pdf_page_number"]
