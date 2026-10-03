"""Real PDF regression: no API keys and no reference-answer injection."""
from pathlib import Path

import pytest


@pytest.mark.integration
def test_jiangfeng_same_page_separates_statement_scopes():
    from annual_report_agent.ingestion.parser import parse_report
    from annual_report_agent.ingestion.tokenizer import FixedTokenizer
    from annual_report_agent.schemas.report import Report
    path = Path(r"D:\Desktop\江丰电子：2026年半年度报告（更正后）.pdf")
    if not path.exists():
        pytest.skip("User-supplied PDF is not distributed with this repository")
    tokenizer = FixedTokenizer.load(allow_download=False)
    result = parse_report(Report(id="jf", sha256="integration", path=str(path), filename=path.name),
                          tokenizer=tokenizer, page_numbers=[1, *range(56, 68)])
    tables = [p for p in result.parents if p.kind == "table"]
    page62 = [p for p in tables if p.page_refs[0].pdf_page_number == 62]
    assert {p.metadata["scope"] for p in page62} == {"parent", "consolidated"}
    profit = next(p for p in page62 if p.metadata["scope"] == "consolidated")
    assert "合并利润表" in profit.metadata["table_header"]
    assert "2,737,309,612.09" in profit.text
    assert "1910489823.73" in profit.text.replace(",", "")
    assert result.report.doc_type == "semiannual_report"
    assert result.report.page_count == 193
    assert all(c.chunk_token_count <= 500 for c in result.chunks)
    assert all(c.tokenizer_version.startswith("BAAI/bge-m3@") for c in result.chunks)


@pytest.mark.integration
def test_merged_pdf_layout_is_preserved_but_flagged_for_review():
    from annual_report_agent.ingestion.parser import parse_report
    from annual_report_agent.ingestion.tokenizer import FixedTokenizer
    from annual_report_agent.schemas.report import Report
    path = Path(r"D:\Desktop\南方乳业：招股说明书.pdf")
    if not path.exists():
        pytest.skip("User-supplied PDF is not distributed with this repository")
    parsed = parse_report(Report(id="nf", sha256="fixture", path=str(path), filename=path.name),
                          tokenizer=FixedTokenizer.load(), page_numbers=[116])
    tables = [p for p in parsed.parents if p.kind == "table"]
    assert tables and not tables[0].metadata["header_rows"]
    assert tables[0].metadata["structure_status"] == "needs_review"
    assert "TABLE_STRUCTURE_UNCERTAIN:116" in parsed.warnings
    assert all(c.chunk_token_count <= 500 for c in parsed.chunks)
    assert any("采购合同" in c.text for c in parsed.chunks)
