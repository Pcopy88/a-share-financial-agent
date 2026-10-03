"""Normalize source amounts without losing their original representation."""

import hashlib
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation, localcontext

from annual_report_agent.schemas.finance import FinancialFact

_MULTIPLIERS = {"元": Decimal(1), "万元": Decimal(10000), "亿元": Decimal(100000000)}
_MISSING = {"", "-", "--", "—", "–", "nan", "none", "null", "n/a", "不适用"}


def normalize_amount(raw_value: str | Decimal | int | None, raw_unit: str = "元") -> Decimal | None:
    """Convert explicit yuan/ten-thousand-yuan/hundred-million-yuan to yuan.

    Blanks return None. Floats, unknown units and malformed numerals raise ValueError;
    callers keep those inputs as pending/missing facts instead of replacing them with zero.
    """
    if isinstance(raw_value, (float, bool)):
        raise ValueError("float amounts are lossy; supply the original decimal string")
    if raw_value is None:
        return None
    original_text = format(raw_value, "f") if isinstance(raw_value, Decimal) else str(raw_value)
    text = re.sub(r"\s+", "", original_text).replace("，", ",").replace("−", "-")
    if text.lower() in _MISSING:
        return None
    unit = re.sub(r"\s+", "", raw_unit or "").removeprefix("人民币")
    if unit not in _MULTIPLIERS:
        raise ValueError(f"unknown amount unit: {raw_unit!r}")
    if (text.startswith("(") and text.endswith(")")) or (
        text.startswith("（") and text.endswith("）")
    ):
        text = "-" + text[1:-1]
    if not re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text):
        raise ValueError(f"invalid financial numeral: {raw_value!r}")
    try:
        with localcontext() as context:
            context.prec = max(40, len(text) + 10)
            return Decimal(text.replace(",", "")) * _MULTIPLIERS[unit]
    except InvalidOperation as exc:
        raise ValueError("invalid financial amount") from exc


def normalize_fact(*, raw_value, raw_unit, **context) -> FinancialFact:
    """Keep raw input and evidence even when normalization cannot succeed."""
    warnings = list(context.pop("warnings", []))
    try:
        value = normalize_amount(raw_value, raw_unit)
        status = "missing" if value is None else "pending"
        if value is None:
            warnings.append("missing_source_amount")
    except ValueError as exc:
        value, status = None, "pending"
        warnings.append(str(exc))
    return FinancialFact(
        value=value, raw_value=None if raw_value is None else str(raw_value),
        raw_unit=raw_unit, validation_status=status, warnings=warnings, **context,
    )


def mark_conflicts(facts: list[FinancialFact]) -> list[FinancialFact]:
    """Return every source fact; mark unequal observations of the same financial concept."""
    output = [fact.model_copy(deep=True) for fact in facts]
    groups = defaultdict(list)
    for fact in output:
        if fact.value is not None and fact.validation_status != "superseded":
            key = (fact.company, fact.metric_code, fact.period_start, fact.period_end, fact.scope)
            groups[key].append(fact)
    for key, values in groups.items():
        if len({(fact.value, fact.currency, fact.unit) for fact in values}) > 1:
            group_id = "conflict-" + hashlib.sha256(repr(key).encode()).hexdigest()[:16]
            for fact in values:
                fact.validation_status = "conflict"
                fact.conflict_group_id = group_id
                fact.warnings.append("unresolved_source_conflict")
    return output
