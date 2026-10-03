from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest


def income_frame(**changes):
    row = {"SECURITY_CODE": "300666", "REPORT_DATE": "2024-12-31 00:00:00",
           "NOTICE_DATE": "2025-04-01 00:00:00", "UPDATE_DATE": "2026-04-01 00:00:00",
           "REPORT_TYPE": "年报", "CURRENCY": "CNY", "OPERATE_INCOME": "120.00",
           "OPERATE_COST": "70.00", "NETPROFIT": "20.00", "PARENT_NETPROFIT": "21.00"}
    row.update(changes)
    return pd.DataFrame([row])


def adapt(frame, **kwargs):
    from annual_report_agent.finance.akshare_adapter import adapt_statement

    return adapt_statement(frame, statement="income", stock_code="300666", year=2024,
                           company="样例", report_id="report-1", **kwargs)


def test_adapter_maps_explicit_fields_keeps_dates_and_never_assigns_pdf_evidence():
    result = adapt(income_frame())
    facts = {item.metric_code: item for item in result.facts}
    assert facts["revenue"].value == Decimal("120.00")
    assert facts["parent_net_profit"].value == Decimal("21.00")
    assert facts["net_profit"].value == Decimal("20.00")
    assert facts["revenue"].period_start == date(2024, 1, 1)
    assert facts["revenue"].period_end == date(2024, 12, 31)
    assert all(f.validation_status == "pending" and not f.source_evidence_ids for f in result.facts)
    assert result.snapshot["records"][0]["UPDATE_DATE"] == "2026-04-01 00:00:00"
    assert result.snapshot["snapshot_hash"]
    assert facts["revenue"].provenance["notice_date"] == "2025-04-01 00:00:00"
    assert "pdf_page_number" not in facts["revenue"].provenance


def test_missing_value_stays_missing_and_float_boundary_is_disclosed():
    result = adapt(income_frame(NETPROFIT=None, OPERATE_INCOME=120.15))
    facts = {item.metric_code: item for item in result.facts}
    assert facts["net_profit"].value is None
    assert facts["net_profit"].validation_status == "missing"
    assert facts["revenue"].value == Decimal("120.15")
    assert "upstream_float_requires_pdf_verification" in facts["revenue"].warnings


def test_missing_mapped_column_reports_schema_drift_and_keeps_snapshot():
    from annual_report_agent.finance.akshare_adapter import AdapterError

    with pytest.raises(AdapterError, match="FIELD_SCHEMA_CHANGED") as error:
        adapt(income_frame().drop(columns=["OPERATE_COST"]))
    assert error.value.snapshot["records"]


def test_frozen_schema_detects_added_columns():
    from annual_report_agent.finance.akshare_adapter import AdapterError

    expected = list(income_frame().columns)
    with pytest.raises(AdapterError, match="FIELD_SCHEMA_CHANGED"):
        adapt(income_frame(NEW_SOURCE_COLUMN="change"), expected_columns=expected)


@pytest.mark.parametrize("changes", [
    {"SECURITY_CODE": "600519"}, {"CURRENCY": "USD"}, {"REPORT_TYPE": "单季报"},
])
def test_adapter_refuses_wrong_company_currency_or_reporting_basis(changes):
    from annual_report_agent.finance.akshare_adapter import AdapterError

    with pytest.raises(AdapterError):
        adapt(income_frame(**changes))


def test_half_year_is_not_silently_treated_as_annual():
    result = adapt(income_frame(REPORT_DATE="2024-06-30", REPORT_TYPE="中报"))
    assert result.facts == []
    assert "requested_annual_period_not_found" in result.warnings


def test_non_shanghai_shenzhen_code_is_explicitly_unsupported():
    from annual_report_agent.finance.akshare_adapter import AdapterError, market_symbol

    assert market_symbol("300666") == "SZ300666"
    assert market_symbol("600519") == "SH600519"
    with pytest.raises(AdapterError, match="UNSUPPORTED_MARKET"):
        market_symbol("920001")


def test_fetch_keeps_successful_statement_when_another_source_fails():
    from annual_report_agent.finance.akshare_adapter import fetch_financial_facts

    def failed(**kwargs):
        raise RuntimeError("upstream unavailable")

    client = SimpleNamespace(__version__="test-source", stock_profit_sheet_by_report_em=lambda **_: income_frame(),
                             stock_balance_sheet_by_report_em=failed, stock_cash_flow_sheet_by_report_em=failed)
    result = fetch_financial_facts("300666", 2024, company="样例", report_id="report-1", ak_client=client)
    assert len(result.facts) == 4
    assert len(result.errors) == 2
    assert result.snapshots[0]["api_name"] == "stock_profit_sheet_by_report_em"


def test_repeat_fetch_preserves_each_observation_without_overwriting_snapshot(tmp_path):
    from annual_report_agent.finance.akshare_adapter import fetch_financial_facts
    def unavailable(**kwargs):
        raise RuntimeError("not available")
    client = SimpleNamespace(__version__="test", stock_profit_sheet_by_report_em=lambda **kw: income_frame(),
        stock_balance_sheet_by_report_em=unavailable, stock_cash_flow_sheet_by_report_em=unavailable)
    first = fetch_financial_facts("300666", 2024, company="样例", report_id="r",
                                  ak_client=client, snapshot_dir=tmp_path)
    first_file = next(tmp_path.glob("*.json"))
    first_content = first_file.read_bytes()
    second = fetch_financial_facts("300666", 2024, company="样例", report_id="r",
                                   ak_client=client, snapshot_dir=tmp_path)
    assert len(first.facts) == len(second.facts) == 4
    assert first_file.read_bytes() == first_content
    assert len(list(tmp_path.glob("*.json"))) == 2


@pytest.mark.parametrize(("statement", "fields", "expected"), [
    ("balance", {"TOTAL_ASSETS": "100", "TOTAL_LIABILITIES": "60",
                 "TOTAL_CURRENT_ASSETS": "50", "TOTAL_CURRENT_LIAB": "20"},
     {"total_assets": "100", "total_liabilities": "60", "current_assets": "50", "current_liabilities": "20"}),
    ("cash_flow", {"NETCASH_OPERATE": "-12"}, {"operating_cash_flow": "-12"}),
])
def test_remaining_statement_mappings_preserve_negative_cash_and_instant_period(statement, fields, expected):
    from annual_report_agent.finance.akshare_adapter import adapt_statement

    frame = income_frame().drop(columns=["OPERATE_INCOME", "OPERATE_COST", "NETPROFIT", "PARENT_NETPROFIT"])
    for key, value in fields.items():
        frame[key] = value
    result = adapt_statement(frame, statement=statement, stock_code="300666", year=2024,
                             company="样例", report_id="r1")
    assert {fact.metric_code: str(fact.value) for fact in result.facts} == expected
    assert all((fact.period_start is None) == (statement == "balance") for fact in result.facts)


def test_missing_optional_dependency_is_explicit_and_requires_no_install(monkeypatch):
    import annual_report_agent.finance.akshare_adapter as module

    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(module.AdapterError, match="uv sync --extra acquisition"):
        module.call_akshare("stock_profit_sheet_by_report_em", {"symbol": "SZ300666"})


def test_raw_snapshot_is_saved_without_replacing_existing_file(tmp_path):
    import json

    from annual_report_agent.finance.akshare_adapter import save_snapshot

    snapshot = adapt(income_frame()).snapshot
    target = save_snapshot(snapshot, tmp_path / "source.json")
    assert json.loads(target.read_text(encoding="utf-8"))["records"][0]["OPERATE_INCOME"] == "120.00"
    with pytest.raises(FileExistsError):
        save_snapshot(snapshot, target)


def pdf_revenue(**changes):
    from annual_report_agent.schemas.finance import FinancialFact

    data = dict(id="pdf-revenue", metric_code="revenue", value=Decimal("120.00"),
                raw_value="120.00", raw_unit="元", company="样例", period_start="2024-01-01",
                period_end="2024-12-31", scope="consolidated", report_id="report-1", source_id="parent-1",
                source_evidence_ids=["parent-1"], provenance={
                    "source_type": "pdf", "source_sha256": "a" * 64, "pdf_page_number": 10,
                    "table_header": "合并利润表", "row_label": "其中：营业收入", "column_label": "2024年度",
                    "row_index": 3, "column_index": 2, "bbox": [10, 10, 300, 400]})
    data.update(changes)
    return FinancialFact(**data)


def test_reconcile_requires_actual_pdf_provenance_and_promotes_only_pdf_side():
    from annual_report_agent.finance.akshare_adapter import reconcile_with_pdf

    api = adapt(income_frame()).facts[0]
    pdf = pdf_revenue()
    result = reconcile_with_pdf([api], [pdf])
    by_id = {f.id: f for f in result.facts}
    assert by_id[pdf.id].validation_status == "verified"
    assert by_id[api.id].validation_status == "pending"
    assert by_id[pdf.id].source_id == "parent-1"
    assert by_id[pdf.id].provenance == pdf.provenance
    assert by_id[pdf.id].review_record["api_fact_ids"] == [api.id]
    assert by_id[pdf.id].review_record["method"] == "api_pdf_exact_reconciliation"
    assert api.validation_status == pdf.validation_status == "pending"


@pytest.mark.parametrize("missing", ["source_sha256", "pdf_page_number", "table_header", "row_label",
                                    "column_label", "bbox", "row_index", "column_index"])
def test_equal_numbers_without_pdf_provenance_are_never_verified(missing):
    from annual_report_agent.finance.akshare_adapter import reconcile_with_pdf

    pdf = pdf_revenue()
    del pdf.provenance[missing]
    result = reconcile_with_pdf([adapt(income_frame()).facts[0]], [pdf])
    assert not any(f.validation_status == "verified" for f in result.facts)
    assert any(d["code"] == "PDF_PROVENANCE_INCOMPLETE" for d in result.diagnostics)


def test_equal_parsed_numbers_with_wrong_raw_cell_cannot_pass():
    from annual_report_agent.finance.akshare_adapter import reconcile_with_pdf

    result = reconcile_with_pdf([adapt(income_frame()).facts[0]], [pdf_revenue(raw_value="999")])
    assert all(f.validation_status != "verified" for f in result.facts)


def test_reconciliation_conflict_retains_both_sources_and_blocks_calculation():
    from annual_report_agent.finance.akshare_adapter import reconcile_with_pdf

    result = reconcile_with_pdf([adapt(income_frame()).facts[0]],
                                [pdf_revenue(value=Decimal("121"), raw_value="121")])
    assert len(result.facts) == 2
    assert {f.validation_status for f in result.facts} == {"conflict"}
    assert len({f.conflict_group_id for f in result.facts}) == 1
    assert result.facts[0].conflict_group_id
    assert result.diagnostics[0]["code"] == "API_PDF_VALUE_CONFLICT"


@pytest.mark.parametrize("changes", [{"company": "另一公司"}, {"scope": "parent"},
                                      {"currency": "USD"}, {"period_end": "2023-12-31"},
                                      {"report_id": "another-version"}])
def test_reconciliation_never_mixes_context(changes):
    from annual_report_agent.finance.akshare_adapter import reconcile_with_pdf

    result = reconcile_with_pdf([adapt(income_frame()).facts[0]], [pdf_revenue(**changes)])
    assert not any(f.validation_status == "verified" for f in result.facts)


def test_reconciliation_compares_normalized_amount_and_preserves_original_unit():
    from annual_report_agent.finance.akshare_adapter import reconcile_with_pdf

    result = reconcile_with_pdf([adapt(income_frame()).facts[0]],
                                [pdf_revenue(raw_value="0.012", raw_unit="万元")])
    pdf = next(f for f in result.facts if f.id == "pdf-revenue")
    assert pdf.validation_status == "verified"
    assert pdf.raw_unit == "万元" and pdf.raw_value == "0.012"


def test_reconciliation_existing_source_conflict_cannot_be_cleared_by_one_matching_api():
    from annual_report_agent.finance.akshare_adapter import reconcile_with_pdf

    result = reconcile_with_pdf([adapt(income_frame()).facts[0]],
                                [pdf_revenue(validation_status="conflict", conflict_group_id="existing")])
    assert not any(f.validation_status == "verified" for f in result.facts)


@pytest.mark.parametrize(("key", "value"), [("source_sha256", None), ("table_header", None),
                                           ("column_label", None), ("row_label", "营业成本"),
                                           ("table_header", "合并现金流量表")])
def test_reconciliation_malformed_or_wrong_pdf_location_stays_pending(key, value):
    from annual_report_agent.finance.akshare_adapter import reconcile_with_pdf

    pdf = pdf_revenue()
    pdf.provenance[key] = value
    result = reconcile_with_pdf([adapt(income_frame()).facts[0]], [pdf])
    assert not any(f.validation_status == "verified" for f in result.facts)


def test_reconciliation_rejects_api_fact_without_snapshot_field_origin():
    from annual_report_agent.finance.akshare_adapter import reconcile_with_pdf

    api = adapt(income_frame()).facts[0]
    api.provenance["json_pointer"] = "/records/0/NETPROFIT"
    result = reconcile_with_pdf([api], [pdf_revenue()])
    assert not any(f.validation_status == "verified" for f in result.facts)
