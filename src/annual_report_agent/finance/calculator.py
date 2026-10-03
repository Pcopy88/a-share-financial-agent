"""Eight explicit formulas. The model chooses an ID; it never supplies executable code.

Operand keys are public in FORMULA_OPERANDS. Percent results are percentage numbers
(20 means 20%); gross-margin changes are percentage points, not growth percentages.
"""

import hashlib
import json
from decimal import Decimal, localcontext

from annual_report_agent.finance.normalizer import normalize_amount
from annual_report_agent.schemas.finance import Calculation, FinancialFact

FORMULA_VERSION = "1.0"
FORMULA_OPERANDS: dict[str, dict[str, str]] = {
    "revenue_yoy": {"current_revenue": "revenue", "previous_revenue": "revenue"},
    "parent_net_profit_yoy": {"current_parent_net_profit": "parent_net_profit",
                              "previous_parent_net_profit": "parent_net_profit"},
    "gross_margin": {"revenue": "revenue", "operating_cost": "operating_cost"},
    "net_margin": {"net_profit": "net_profit", "revenue": "revenue"},
    "debt_asset_ratio": {"total_liabilities": "total_liabilities", "total_assets": "total_assets"},
    "current_ratio": {"current_assets": "current_assets", "current_liabilities": "current_liabilities"},
    "operating_cash_flow_to_revenue": {"operating_cash_flow": "operating_cash_flow",
                                       "revenue": "revenue"},
    "gross_margin_yoy_pp": {"current_revenue": "revenue", "current_operating_cost": "operating_cost",
                            "previous_revenue": "revenue", "previous_operating_cost": "operating_cost"},
}


class CalculationError(ValueError):
    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _period(fact: FinancialFact):
    return fact.period_start, fact.period_end


def _validate_operands(formula_id: str, operands: dict[str, FinancialFact]) -> None:
    expected = FORMULA_OPERANDS.get(formula_id)
    if expected is None:
        raise CalculationError("unknown_formula", formula_id)
    if set(expected) != set(operands):
        raise CalculationError("operand_keys", f"required keys: {', '.join(expected)}")
    facts = list(operands.values())
    for key, fact in operands.items():
        if fact.metric_code != expected[key]:
            raise CalculationError("wrong_metric", f"{key} requires {expected[key]}")
        if fact.validation_status != "verified" or fact.conflict_group_id or fact.value is None:
            raise CalculationError("unverified_operand", fact.id)
        if not fact.review_record or not fact.source_evidence_ids:
            raise CalculationError("missing_verification_provenance", fact.id)
        if not fact.value.is_finite():
            raise CalculationError("nonfinite_operand", fact.id)
        if fact.unit != "元" or not fact.raw_unit:
            raise CalculationError("unknown_unit", fact.id)
        try:
            normalize_amount("1", fact.raw_unit)
        except ValueError as exc:
            raise CalculationError("unknown_unit", fact.id) from exc
    for attribute in ("company", "currency", "scope"):
        if len({getattr(fact, attribute) for fact in facts}) != 1:
            raise CalculationError("mixed_" + attribute)
    if not facts[0].company or not facts[0].currency:
        raise CalculationError("missing_company_or_currency")
    if formula_id == "parent_net_profit_yoy" and facts[0].scope != "consolidated":
        raise CalculationError("parent_profit_requires_consolidated_scope")
    instant = formula_id in {"debt_asset_ratio", "current_ratio"}
    for fact in facts:
        if instant:
            if fact.period_start is not None:
                raise CalculationError("instant_period_required")
        elif not (
            fact.period_start is not None
            and (fact.period_start.month, fact.period_start.day) == (1, 1)
            and fact.period_start.year == fact.period_end.year
            and (fact.period_end.month, fact.period_end.day) in {(6, 30), (12, 31)}
        ):
            raise CalculationError("unsupported_duration_period", "require full year or January-June")
    if "previous_revenue" in operands or "previous_parent_net_profit" in operands:
        current = [value for key, value in operands.items() if key.startswith("current_")]
        previous = [value for key, value in operands.items() if key.startswith("previous_")]
        if len({_period(value) for value in current}) != 1 or len(
            {_period(value) for value in previous}
        ) != 1:
            raise CalculationError("mixed_period")
        for current_date, previous_date in zip(_period(current[0]), _period(previous[0]), strict=True):
            if current_date is None or previous_date is None or not (
                current_date.year == previous_date.year + 1
                and (current_date.month, current_date.day) == (previous_date.month, previous_date.day)
            ):
                raise CalculationError("noncomparable_yoy_period")
    elif len({_period(value) for value in facts}) != 1:
        raise CalculationError("mixed_period")


def _ratio(numerator: Decimal, denominator: Decimal, *, growth=False) -> Decimal:
    if denominator <= 0:
        raise CalculationError("nonpositive_base" if growth else "nonpositive_denominator")
    return numerator / denominator


def calculate_metric(formula_id: str, operands: dict[str, FinancialFact]) -> Calculation:
    """Calculate only verified, compatible operands named by FORMULA_OPERANDS.

    Raises CalculationError with a stable code for missing data, unresolved conflict,
    mismatched periods/scope/metrics, nonpositive denominators, or unknown formulas.
    A caller must select the comparable restated facts before invoking this function.
    """
    _validate_operands(formula_id, operands)
    values = {key: fact.value for key, fact in operands.items()}
    unit = "percent"
    with localcontext() as context:
        context.prec = 40
        if formula_id in {"revenue_yoy", "parent_net_profit_yoy"}:
            metric = "revenue" if formula_id == "revenue_yoy" else "parent_net_profit"
            result = (_ratio(values[f"current_{metric}"], values[f"previous_{metric}"],
                             growth=True) - 1) * 100
        elif formula_id == "gross_margin":
            result = _ratio(values["revenue"] - values["operating_cost"], values["revenue"]) * 100
        elif formula_id == "net_margin":
            result = _ratio(values["net_profit"], values["revenue"]) * 100
        elif formula_id == "debt_asset_ratio":
            result = _ratio(values["total_liabilities"], values["total_assets"]) * 100
        elif formula_id == "current_ratio":
            result = _ratio(values["current_assets"], values["current_liabilities"])
            unit = "times"
        elif formula_id == "operating_cash_flow_to_revenue":
            result = _ratio(values["operating_cash_flow"], values["revenue"]) * 100
        else:  # gross_margin_yoy_pp; whitelist validation has already succeeded.
            current = _ratio(values["current_revenue"] - values["current_operating_cost"],
                             values["current_revenue"]) * 100
            previous = _ratio(values["previous_revenue"] - values["previous_operating_cost"],
                              values["previous_revenue"]) * 100
            result, unit = current - previous, "percentage_points"
    ordered = [operands[key] for key in FORMULA_OPERANDS[formula_id]]
    latest = max(ordered, key=lambda fact: fact.period_end)
    label = (f"{latest.period_start.isoformat()}/{latest.period_end.isoformat()}"
             if latest.period_start else latest.period_end.isoformat())
    fingerprint = json.dumps([formula_id, FORMULA_VERSION,
                              [(fact.id, str(fact.value)) for fact in ordered]], ensure_ascii=False)
    warnings = list(dict.fromkeys(warning for fact in ordered for warning in fact.warnings))
    if any(fact.restatement_status == "restated" for fact in ordered):
        warnings.append("uses_disclosed_restated_comparative")
    return Calculation(
        id="calc-" + hashlib.sha256(fingerprint.encode()).hexdigest()[:20],
        formula_id=formula_id, formula_version=FORMULA_VERSION,
        operand_fact_ids=[fact.id for fact in ordered], result=result, unit=unit,
        warnings=warnings, period_label=label,
    )
