"""Select traceable, comparable facts before calling the formula whitelist."""

from datetime import date

from pydantic import BaseModel, Field

from annual_report_agent.finance.calculator import FORMULA_OPERANDS, calculate_metric
from annual_report_agent.schemas.finance import Calculation, FinancialFact

DIRECT_METRICS = {"revenue", "operating_cost", "net_profit", "parent_net_profit", "total_assets",
                  "total_liabilities", "current_assets", "current_liabilities", "operating_cash_flow"}
INSTANT_METRICS = {"total_assets", "total_liabilities", "current_assets", "current_liabilities"}


class FinanceSelectionError(ValueError):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(f"{code}: {detail}")


class CalculationBundle(BaseModel):
    facts: list[FinancialFact] = Field(default_factory=list)
    calculations: list[Calculation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _current_period(task, facts: list[FinancialFact], instant: bool) -> tuple:
    periods = []
    for entry in task.report_periods:
        end = entry.get("end") or entry.get("period_end")
        start = entry.get("start") or entry.get("period_start")
        if end:
            periods.append((None if instant else date.fromisoformat(start) if start else None,
                            date.fromisoformat(end)))
    if periods:
        return max(periods, key=lambda pair: pair[1])
    periods = {(f.period_start, f.period_end) for f in facts
               if (f.period_start is None) == instant
               and (not task.fiscal_years or f.period_end.year in task.fiscal_years)}
    if not periods:
        raise FinanceSelectionError("INSUFFICIENT_EVIDENCE", "没有目标期间的财务事实")
    latest_year = max(end.year for _, end in periods)
    current = [period for period in periods if period[1].year == latest_year]
    if len(current) != 1:
        raise FinanceSelectionError("REPORT_AMBIGUOUS", "请选择明确的全年、半年或报表时点")
    return current[0]


def _pick(facts: list[FinancialFact], metric: str, period: tuple) -> FinancialFact:
    candidates = [f for f in facts if f.metric_code == metric
                  and (f.period_start, f.period_end) == period]
    if any(f.validation_status == "conflict" or f.conflict_group_id for f in candidates):
        raise FinanceSelectionError("DATA_CONFLICT", f"{metric} 存在未解决的冲突")
    candidates = [f for f in candidates if f.validation_status == "verified" and f.value is not None]
    if not candidates:
        raise FinanceSelectionError("INSUFFICIENT_EVIDENCE", f"{metric} 缺少核验通过的值")
    keys = {(f.value, f.unit, f.currency, f.company, f.scope) for f in candidates}
    if len(keys) != 1:
        raise FinanceSelectionError("DATA_CONFLICT", f"{metric} 的来源金额或口径不一致")
    return sorted(candidates, key=lambda fact: (fact.report_id, fact.id))[0]


def select_and_calculate(task, facts: list[FinancialFact]) -> CalculationBundle:
    eligible = [fact for fact in facts if fact.report_id in task.report_ids and fact.scope == task.scope
                and (not task.hard_filters.company or fact.company == task.hard_filters.company)]
    if not task.metric_ids:
        raise FinanceSelectionError("METRIC_UNSUPPORTED", "没有指定可支持的指标")
    selected, calculations = {}, []
    for metric in task.metric_ids:
        if metric not in DIRECT_METRICS and metric not in FORMULA_OPERANDS:
            raise FinanceSelectionError("METRIC_UNSUPPORTED", metric)
        instant = metric in INSTANT_METRICS or metric in {"debt_asset_ratio", "current_ratio"}
        relevant_metrics = {metric} if metric in DIRECT_METRICS else set(FORMULA_OPERANDS[metric].values())
        relevant = [f for f in eligible if f.metric_code in relevant_metrics]
        current = _current_period(task, relevant, instant)
        if metric in DIRECT_METRICS:
            picked = _pick(eligible, metric, current)
            selected[picked.id] = picked
            continue
        operands = {}
        for key, field in FORMULA_OPERANDS[metric].items():
            period = current
            if key.startswith("previous_"):
                period = tuple(value.replace(year=value.year - 1) if value else None for value in current)
            picked = _pick(eligible, field, period)
            operands[key] = picked
            selected[picked.id] = picked
        calculations.append(calculate_metric(metric, operands))
    return CalculationBundle(facts=list(selected.values()), calculations=calculations)
