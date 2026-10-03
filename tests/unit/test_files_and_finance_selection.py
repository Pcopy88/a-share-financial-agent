"""Storage names and selection rules must not trust user filenames or mixed periods."""

from datetime import date
from decimal import Decimal
from importlib.util import find_spec

import pytest


def test_upload_is_content_addressed_and_cannot_escape_workspace(tmp_path):
    assert find_spec("annual_report_agent.storage.files"), "file store missing"
    from annual_report_agent.storage.files import save_pdf
    first = save_pdf(tmp_path, "../../outside.pdf", b"%PDF-1.7\nvalid fixture", max_bytes=100)
    second = save_pdf(tmp_path, "renamed.PDF", b"%PDF-1.7\nvalid fixture", max_bytes=100)
    assert first == second
    assert first[0].parent == tmp_path.resolve()
    with pytest.raises(ValueError, match="PDF"):
        save_pdf(tmp_path, "trick.pdf", b"not a PDF", max_bytes=100)
    with pytest.raises(ValueError, match="large"):
        save_pdf(tmp_path, "large.pdf", b"%PDF" + b"0" * 101, max_bytes=100)


def fact(identifier, metric, value, year=2026, **overrides):
    from annual_report_agent.schemas.finance import FinancialFact
    fields = dict(id=identifier, metric_code=metric, value=Decimal(value), raw_value=value,
                  raw_unit="元", currency="CNY", company="甲", period_start=date(year, 1, 1),
                  period_end=date(year, 6, 30), scope="consolidated", report_id="a",
                  source_id="p1", source_evidence_ids=["p1"], validation_status="verified",
                  review_record={"method": "reference_comparison"})
    return FinancialFact(**(fields | overrides))


def task(metric):
    from annual_report_agent.schemas.query import SubTaskPlan
    return SubTaskPlan(subtask_id="q1", source_text="收入同比", query_text="收入同比",
                       rewritten_query="收入同比", report_ids=["a"], route="data",
                       metric_ids=[metric], fiscal_years=[2026],
                       report_periods=[{"start": "2026-01-01", "end": "2026-06-30"}])


def test_calculation_selects_comparable_prior_period_and_keeps_operands():
    assert find_spec("annual_report_agent.finance.service"), "finance orchestration missing"
    from annual_report_agent.finance.service import select_and_calculate
    current = fact("cur", "revenue", "120")
    prior = fact("prev", "revenue", "100", 2025)
    wrong = fact("annual", "revenue", "500", 2025, period_end=date(2025, 12, 31))
    result = select_and_calculate(task("revenue_yoy"), [current, prior, wrong])
    assert result.calculations[0].result == Decimal("20")
    assert {f.id for f in result.facts} == {"cur", "prev"}


def test_conflicting_sources_and_unverified_values_never_enter_calculator():
    assert find_spec("annual_report_agent.finance.service"), "finance orchestration missing"
    from annual_report_agent.finance.service import FinanceSelectionError, select_and_calculate
    current = fact("cur", "revenue", "120")
    conflicting = fact("conflict", "revenue", "130")
    prior = fact("prev", "revenue", "100", 2025)
    with pytest.raises(FinanceSelectionError, match="DATA_CONFLICT"):
        select_and_calculate(task("revenue_yoy"), [current, conflicting, prior])
    with pytest.raises(FinanceSelectionError, match="INSUFFICIENT_EVIDENCE"):
        select_and_calculate(task("revenue_yoy"), [current.model_copy(update={"validation_status": "pending"}), prior])
