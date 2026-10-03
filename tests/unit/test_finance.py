"""Finance behavior tests: exact amounts, safe periods and evidence gates."""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest


def fact(metric, amount, *, previous=False, instant=False, **changes):
    from annual_report_agent.schemas.finance import FinancialFact

    year = 2025 if previous else 2026
    values = dict(
        id=f"{metric}-{year}", metric_code=metric, value=amount, raw_value=str(amount),
        raw_unit="元", company="测试公司", report_id="report-1", source_id="source-1",
        source_evidence_ids=["e-1"], scope="consolidated",
        period_start=None if instant else date(year, 1, 1),
        period_end=date(year, 6, 30), validation_status="verified",
        review_record={"method": "test_fixture", "basis": "synthetic hand-checked values"},
    )
    values.update(changes)
    return FinancialFact(**values)


@pytest.mark.parametrize(("raw", "unit", "expected"), [
    ("1,234.56", "元", "1234.56"), ("（1,234.56）", "万元", "-12345600"),
    ("1.20", "亿元", "120000000"), ("−3.2", "人民币元", "-3.2"),
    ("", "元", None), ("--", "元", None), ("NaN", "元", None),
])
def test_amount_normalization_preserves_missing_and_unit_scale(raw, unit, expected):
    from annual_report_agent.finance.normalizer import normalize_amount

    result = normalize_amount(raw, unit)
    assert result == (None if expected is None else Decimal(expected))


@pytest.mark.parametrize(("raw", "unit"), [(1.2, "元"), ("10", "未知"), ("12x", "元")])
def test_amount_rejects_lossy_or_ambiguous_input(raw, unit):
    from annual_report_agent.finance.normalizer import normalize_amount

    with pytest.raises(ValueError):
        normalize_amount(raw, unit)


def test_fact_decimal_json_roundtrip_does_not_pass_through_float():
    value = fact("revenue", "99999999999999999.99")
    assert value.model_dump(mode="json")["value"] == "99999999999999999.99"
    assert type(value).model_validate_json(value.model_dump_json()).value == value.value


@pytest.mark.parametrize(("formula", "operands", "expected", "unit"), [
    ("revenue_yoy", {"current_revenue": ("revenue", "120", False, False),
                     "previous_revenue": ("revenue", "100", True, False)}, "20", "percent"),
    ("parent_net_profit_yoy", {
        "current_parent_net_profit": ("parent_net_profit", "30", False, False),
        "previous_parent_net_profit": ("parent_net_profit", "10", True, False)},
     "200", "percent"),
    ("gross_margin", {"revenue": ("revenue", "120", False, False),
                      "operating_cost": ("operating_cost", "72", False, False)},
     "40", "percent"),
    ("net_margin", {"revenue": ("revenue", "120", False, False),
                    "net_profit": ("net_profit", "24", False, False)}, "20", "percent"),
    ("debt_asset_ratio", {"total_assets": ("total_assets", "120", False, True),
                          "total_liabilities": ("total_liabilities", "60", False, True)},
     "50", "percent"),
    ("current_ratio", {"current_assets": ("current_assets", "150", False, True),
                       "current_liabilities": ("current_liabilities", "50", False, True)},
     "3", "times"),
    ("operating_cash_flow_to_revenue", {
        "revenue": ("revenue", "120", False, False),
        "operating_cash_flow": ("operating_cash_flow", "-6", False, False)},
     "-5", "percent"),
    ("gross_margin_yoy_pp", {"current_revenue": ("revenue", "120", False, False),
                            "current_operating_cost": ("operating_cost", "72", False, False),
                            "previous_revenue": ("revenue", "100", True, False),
                            "previous_operating_cost": ("operating_cost", "70", True, False)},
     "10", "percentage_points"),
])
def test_whitelisted_formulas_use_exact_decimal_and_keep_operand_ids(formula, operands,
                                                                  expected, unit):
    from annual_report_agent.finance.calculator import calculate_metric

    inputs = {key: fact(metric, value, previous=previous, instant=instant)
              for key, (metric, value, previous, instant) in operands.items()}
    result = calculate_metric(formula, inputs)
    assert result.result == Decimal(expected)
    assert result.unit == unit
    assert set(result.operand_fact_ids) == {value.id for value in inputs.values()}
    assert result.formula_version
    assert "2026" in result.period_label


@pytest.mark.parametrize("base", ["0", "-10"])
def test_yoy_rejects_nonpositive_base_instead_of_infinity_or_misleading_growth(base):
    from annual_report_agent.finance.calculator import CalculationError, calculate_metric

    with pytest.raises(CalculationError, match="base"):
        calculate_metric("revenue_yoy", {"current_revenue": fact("revenue", "120"),
                         "previous_revenue": fact("revenue", base, previous=True)})


@pytest.mark.parametrize("change", [
    {"company": "另一公司"}, {"currency": "USD"}, {"scope": "parent"},
    {"period_end": date(2026, 12, 31)}, {"validation_status": "pending"},
    {"value": None, "validation_status": "missing"},
    {"conflict_group_id": "unresolved-conflict"},
    {"metric_code": "parent_net_profit"},
])
def test_ratio_refuses_mixed_scope_period_currency_missing_or_unverified(change):
    from annual_report_agent.finance.calculator import CalculationError, calculate_metric

    with pytest.raises(CalculationError):
        calculate_metric("net_margin", {"revenue": fact("revenue", "120"),
                                         "net_profit": fact("net_profit", "24", **change)})


def test_yoy_refuses_half_year_against_full_year():
    from annual_report_agent.finance.calculator import CalculationError, calculate_metric

    with pytest.raises(CalculationError, match="period"):
        calculate_metric("revenue_yoy", {"current_revenue": fact("revenue", "120"),
                         "previous_revenue": fact("revenue", "100", previous=True,
                                                  period_end=date(2025, 12, 31))})


def test_normalizer_keeps_both_conflicting_sources_without_mutating_inputs():
    from annual_report_agent.finance.normalizer import mark_conflicts

    first = fact("revenue", "120", id="pdf-fact", source_id="pdf")
    second = fact("revenue", "121", id="api-fact", source_id="api")
    output = mark_conflicts([first, second])
    assert [item.value for item in output] == [Decimal("120"), Decimal("121")]
    assert {item.validation_status for item in output} == {"conflict"}
    assert output[0].conflict_group_id == output[1].conflict_group_id
    assert first.validation_status == "verified"
    assert [item.source_id for item in output] == ["pdf", "api"]


def test_reference_loader_does_not_create_verified_facts():
    from annual_report_agent.finance.pdf_facts import load_reference

    path = Path(__file__).resolve().parents[2] / "evals/jiangfeng_2026h1_reference.json"
    reference = load_reference(path)
    assert reference["purpose"] == "development_reference_only"
    assert reference["facts"][0]["application_validation_status"] == "not_run"


def parsed_table(*, table_header="合并利润表", scope="consolidated", unit="元", rows=None):
    from annual_report_agent.schemas.report import Page, PageRef, ParentBlock, ParsedReport, Report

    rows = rows or [["项目", "2026 年半年度", "2025 年半年度"],
                    ["一、营业总收入", "120.00", "100.00"],
                    ["其中：营业收入", "120.00", "100.00"],
                    ["营业总成本", "90.00", "85.00"],
                    ["其中：营业成本", "72.00", "70.00"],
                    ["归属于母公司股东的净利润", "30.00", "10.00"],
                    ["四、净利润（净亏损以“－”号填列）", "24.00", "8.00"]]
    text = "\n".join(" | ".join(cell or "" for cell in row) for row in rows)
    return ParsedReport(
        report=Report(id="report-1", sha256="synthetic-sha", path="synthetic.pdf",
                      filename="synthetic.pdf", company="测试公司", fiscal_year=2026,
                      doc_type="semiannual_report", page_count=62,
                      report_periods=[{"label": "2026H1", "period_start": "2026-01-01",
                                       "period_end": "2026-06-30"}]),
        pages=[Page(report_id="report-1", pdf_page_number=62, raw_text=text,
                    parse_status="parsed")],
        parents=[ParentBlock(id="table-62", report_id="report-1", kind="table", text=text,
                             page_refs=[PageRef(pdf_page_number=62)],
                             metadata={"rows": rows, "header_rows": rows[:1],
                                       "scope": scope, "unit": unit, "bbox": [1, 2, 3, 4],
                                       "table_header": table_header})], chunks=[],
    )


def test_pdf_extraction_uses_rows_not_total_cost_or_parent_profit_aliases():
    from annual_report_agent.finance.pdf_facts import extract_pdf_facts

    facts = extract_pdf_facts(parsed_table())
    current = {item.metric_code: item for item in facts if item.period_end.year == 2026}
    assert current["revenue"].value == Decimal("120.00")
    assert current["operating_cost"].value == Decimal("72.00")
    assert current["parent_net_profit"].value == Decimal("30.00")
    assert current["net_profit"].value == Decimal("24.00")
    assert len(facts) == 8
    assert {item.validation_status for item in facts} == {"pending"}
    assert current["revenue"].provenance["pdf_page_number"] == 62
    assert current["revenue"].provenance["source_sha256"] == "synthetic-sha"


def test_pdf_extraction_uses_balance_dates_not_half_year_duration():
    from annual_report_agent.finance.pdf_facts import extract_pdf_facts

    facts = extract_pdf_facts(parsed_table(table_header="合并资产负债表", rows=[
        ["项目", "期末余额", "期初余额"], ["资产总计", "120", "100"],
        ["负债合计", "60", "40"], ["流动资产合计", "90", "80"],
        ["流动负债合计", "30", "20"],
    ]))
    assets = [item for item in facts if item.metric_code == "total_assets"]
    assert [(item.period_start, item.period_end) for item in assets] == [
        (None, date(2026, 6, 30)), (None, date(2025, 12, 31))]


@pytest.mark.parametrize("change", [
    {"scope": None}, {"rows": [["项目", "本期", "上期"], ["营业收入", "120", "100"]],
                     "table_header": "未分配利润"},
])
def test_ambiguous_pdf_table_stays_pending_without_invented_facts(change):
    from annual_report_agent.finance.pdf_facts import extract_pdf_facts

    diagnostics = []
    assert extract_pdf_facts(parsed_table(**change), diagnostics=diagnostics) == []
    assert diagnostics and diagnostics[0]["status"] == "pending"


def reference_for(extracted):
    return {"purpose": "development_reference_only", "report_id": "report-1",
            "source_sha256": "synthetic-sha", "facts": [
                {"id": "reference-revenue", "metric_code": "revenue", "value": "120.00",
                 "raw_value": "120.00", "unit": "元", "currency": "CNY",
                 "scope": "consolidated", "period_start": "2026-01-01",
                 "period_end": "2026-06-30", "report_id": "report-1",
                 "source": {"pdf_page_number": 62, "table": "合并利润表",
                            "column_label": "2026 年半年度"}, "human_reviewed": False}]}


def test_reference_verification_records_machine_comparison_and_keeps_original_pending():
    from annual_report_agent.finance.pdf_facts import extract_pdf_facts, verify_against_reference

    extracted = extract_pdf_facts(parsed_table())
    checked = verify_against_reference(extracted, reference_for(extracted))
    verified = [item for item in checked if item.validation_status == "verified"]
    assert len(verified) == 1
    assert verified[0].review_record["method"] == "reference_comparison"
    assert verified[0].review_record["human_reviewed"] is False
    assert verified[0].review_record["reference_fact_id"] == "reference-revenue"
    assert {item.validation_status for item in extracted} == {"pending"}


@pytest.mark.parametrize(("target", "key", "value"), [
    ("fact", "raw_value", "999.00"), ("fact", "value", "999.00"),
    ("fact", "period_end", "2026-12-31"), ("fact", "scope", "parent"),
    ("fact", "unit", "万元"), ("source", "pdf_page_number", 61),
    ("root", "source_sha256", "wrong-file-sha"),
])
def test_reference_mismatch_never_grants_verified(target, key, value):
    from annual_report_agent.finance.pdf_facts import extract_pdf_facts, verify_against_reference

    extracted = extract_pdf_facts(parsed_table())
    reference = reference_for(extracted)
    destination = {"root": reference, "fact": reference["facts"][0],
                   "source": reference["facts"][0]["source"]}[target]
    destination[key] = value
    assert all(item.validation_status != "verified"
               for item in verify_against_reference(extracted, reference))


def test_normalization_failure_keeps_raw_text_and_provenance_pending():
    from annual_report_agent.finance.normalizer import normalize_fact

    original = fact("revenue", "1").model_dump(exclude={"value", "raw_value", "raw_unit",
                                                       "validation_status", "warnings"})
    pending = normalize_fact(raw_value="1,234.00", raw_unit="未披露", **original)
    assert pending.value is None
    assert pending.raw_value == "1,234.00"
    assert pending.validation_status == "pending"
    assert pending.source_id == "source-1"
    assert pending.warnings


@pytest.mark.parametrize("value", ["0", "-1"])
def test_margin_denominator_must_be_positive(value):
    from annual_report_agent.finance.calculator import CalculationError, calculate_metric

    with pytest.raises(CalculationError, match="denominator"):
        calculate_metric("net_margin", {"net_profit": fact("net_profit", "-20"),
                                         "revenue": fact("revenue", value)})


def test_calculation_does_not_round_before_display_or_depend_on_global_context():
    from decimal import localcontext

    from annual_report_agent.finance.calculator import calculate_metric

    with localcontext() as context:
        context.prec = 6
        result = calculate_metric("net_margin", {"net_profit": fact("net_profit", "1"),
                                                 "revenue": fact("revenue", "3")})
    assert str(result.result) == "33.33333333333333333333333333333333333333"


def test_unknown_formula_is_rejected_without_interpreting_expression():
    from annual_report_agent.finance.calculator import CalculationError, calculate_metric

    with pytest.raises(CalculationError, match="unknown_formula"):
        calculate_metric("120 / 100 - 1", {})


def test_parent_standalone_profit_cannot_be_used_as_parent_attributable_profit():
    from annual_report_agent.finance.calculator import CalculationError, calculate_metric

    with pytest.raises(CalculationError, match="consolidated"):
        calculate_metric("parent_net_profit_yoy", {
            "current_parent_net_profit": fact("parent_net_profit", "30", scope="parent"),
            "previous_parent_net_profit": fact("parent_net_profit", "10", scope="parent",
                                                previous=True)})


def test_decimal_input_accepts_exact_scientific_representation():
    from annual_report_agent.finance.normalizer import normalize_amount

    assert normalize_amount(Decimal("1E+20"), "万元") == Decimal("1000000000000000000000000")


def test_verified_label_does_not_override_unknown_source_unit():
    from annual_report_agent.finance.calculator import CalculationError, calculate_metric

    with pytest.raises(CalculationError, match="unit"):
        calculate_metric("net_margin", {"net_profit": fact("net_profit", "20", raw_unit="unknown"),
                                         "revenue": fact("revenue", "100")})


def test_parent_income_statement_deduction_prefix_keeps_operating_cost():
    from annual_report_agent.finance.pdf_facts import extract_pdf_facts

    output = extract_pdf_facts(parsed_table(table_header="母公司利润表", scope="parent", rows=[
        ["项目", "2026 年半年度", "2025 年半年度"],
        ["减：营业成本", "72.00", "70.00"],
    ]))
    assert [(item.metric_code, item.value) for item in output] == [
        ("operating_cost", Decimal("72.00")), ("operating_cost", Decimal("70.00"))]
