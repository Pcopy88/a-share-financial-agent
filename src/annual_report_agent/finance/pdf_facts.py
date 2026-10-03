"""Read-only extraction and controlled comparison against development references."""

import hashlib
import json
import re
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

from annual_report_agent.finance.normalizer import mark_conflicts, normalize_fact
from annual_report_agent.schemas.finance import FinancialFact
from annual_report_agent.schemas.report import ParsedReport

_ROW_METRICS = {
    "营业收入": "revenue",
    "营业成本": "operating_cost",
    "净利润": "net_profit",
    "归属于母公司股东的净利润": "parent_net_profit",
    "归属于母公司所有者的净利润": "parent_net_profit",
    "归属于上市公司股东的净利润": "parent_net_profit",
    "资产总计": "total_assets",
    "资产总额": "total_assets",
    "负债合计": "total_liabilities",
    "负债总额": "total_liabilities",
    "流动资产合计": "current_assets",
    "流动负债合计": "current_liabilities",
    "经营活动产生的现金流量净额": "operating_cash_flow",
}
_STATEMENT_METRICS = {
    "利润表": {"revenue", "operating_cost", "net_profit", "parent_net_profit"},
    "资产负债表": {"total_assets", "total_liabilities", "current_assets", "current_liabilities"},
    "现金流量表": {"operating_cash_flow"},
}


def _compact(text) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _metric(label: str) -> str | None:
    label = re.sub(r"^[一二三四五六七八九十\d]+[、.．]", "", _compact(label))
    label = re.sub(r"^(?:其中|减|加)[:：]", "", label)
    label = re.split(r"[（(]", label, maxsplit=1)[0]
    return _ROW_METRICS.get(label)


def _report_duration(parsed: ParsedReport):
    periods = []
    for item in parsed.report.report_periods:
        try:
            start, end = date.fromisoformat(item["period_start"]), date.fromisoformat(item["period_end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start <= end:
            periods.append((start, end))
    return max(periods, key=lambda item: item[1]) if periods else None


def _column_period(label: str, statement: str, duration):
    """Resolve the column itself; relative labels require an explicit report period."""
    text = _compact(label)
    if statement == "资产负债表":
        explicit = re.search(r"(20\d{2})[年/-](\d{1,2})[月/-](\d{1,2})日?", text)
        if explicit:
            try:
                return None, date(*map(int, explicit.groups()))
            except ValueError:
                return None
        if duration:
            if any(word in text for word in ("期末余额", "本报告期末", "本年末")):
                return None, duration[1]
            if any(word in text for word in ("期初余额", "上年末", "上年度末")):
                return None, date(duration[1].year, 1, 1) - timedelta(days=1)
        return None
    year = re.search(r"(20\d{2})年?", text)
    if year:
        year_value = int(year.group(1))
        if any(token in text for token in ("半年度", "上半年", "H1", "1-6月", "1—6月", "1至6月")):
            return date(year_value, 1, 1), date(year_value, 6, 30)
        if "年度" in text and "半年度" not in text:
            return date(year_value, 1, 1), date(year_value, 12, 31)
    if duration:
        if any(word in text for word in ("本期发生额", "本期金额", "本报告期", "本年累计")):
            return duration
        if any(word in text for word in ("上期发生额", "上期金额", "上年同期")):
            try:
                return tuple(value.replace(year=value.year - 1) for value in duration)
            except ValueError:
                return None
    return None


def extract_pdf_facts(parsed: ParsedReport, *, diagnostics: list[dict] | None = None) -> list[FinancialFact]:
    """Read financial statement rows from ParsedReport; all extracted facts start pending.

    Parser metadata contract: rows/header_rows as lists of cell lists, table_header,
    scope, unit, bbox, and optional header_page_refs. Unknown scope/period is recorded
    in diagnostics as pending and never fabricated from the reference fixture.
    source_id/source_evidence_ids identify the existing table ParentBlock.
    """
    diagnostics = diagnostics if diagnostics is not None else []
    facts = []
    duration = _report_duration(parsed)
    for block in parsed.parents:
        if block.kind != "table":
            continue
        metadata = block.metadata
        header = metadata.get("table_header", "")
        statement = next((name for name in _STATEMENT_METRICS if name in header), None)
        scope = metadata.get("scope")
        rows = metadata.get("rows", [])
        reasons = []
        if statement is None:
            reasons.append("unrecognized_financial_statement")
        if scope not in {"consolidated", "parent"}:
            reasons.append("unknown_statement_scope")
        if not block.page_refs or not parsed.report.company:
            reasons.append("missing_page_or_company")
        if not rows:
            reasons.append("missing_table_rows")
        if metadata.get("structure_status") == "needs_review":
            reasons.append("uncertain_table_structure")
        if reasons:
            diagnostics.append({"source_id": block.id, "status": "pending", "reasons": reasons})
            continue
        width = max(len(row) for row in rows)
        header_rows = metadata.get("header_rows") or rows[:1]
        labels = [" ".join(dict.fromkeys(str(row[column] or "") for row in header_rows
                                         if column < len(row))) for column in range(width)]
        periods = {column: _column_period(label, statement, duration)
                   for column, label in enumerate(labels) if column > 0}
        periods = {column: period for column, period in periods.items() if period is not None}
        if not periods:
            diagnostics.append({"source_id": block.id, "status": "pending",
                                "reasons": ["unresolved_column_period"]})
            continue
        for row_index, row in enumerate(rows):
            if not row:
                continue
            metric = _metric(row[0])
            if metric not in _STATEMENT_METRICS[statement]:
                continue
            if metric == "parent_net_profit" and scope != "consolidated":
                continue
            for column, (start, end) in periods.items():
                if column >= len(row):
                    continue
                raw_value = row[column]
                label = labels[column]
                restatement = ("restated" if "调整后" in label else
                               "before_restatement" if "调整前" in label else None)
                identity = f"{parsed.report.sha256}:{block.id}:{row_index}:{column}:{raw_value}"
                facts.append(normalize_fact(
                    id="fact-" + hashlib.sha256(identity.encode()).hexdigest()[:24],
                    metric_code=metric, raw_value=raw_value, raw_unit=metadata.get("unit"),
                    company=parsed.report.company, currency=metadata.get("currency", "CNY"),
                    period_start=start, period_end=end, scope=scope, restatement_status=restatement,
                    report_id=parsed.report.id, source_id=block.id, source_evidence_ids=[block.id],
                    provenance={"source_type": "pdf", "source_sha256": parsed.report.sha256,
                                "pdf_page_number": block.page_refs[0].pdf_page_number,
                                "pdf_page_numbers": [page.pdf_page_number for page in block.page_refs],
                                "table_header": header, "row_label": row[0], "column_label": label,
                                "row_index": row_index, "column_index": column,
                                "bbox": metadata.get("bbox"),
                                "header_page_refs": metadata.get("header_page_refs", [])},
                ))
    return mark_conflicts(facts)


def load_reference(path: str | Path) -> dict:
    """Load raw reference metadata, never manufacture verified application facts."""
    with Path(path).open(encoding="utf-8") as stream:
        reference = json.load(stream)
    if not isinstance(reference, dict) or not isinstance(reference.get("facts"), list):
        raise ValueError("reference must contain a facts list")
    return reference


def _reference_matches(fact: FinancialFact, candidate: dict) -> bool:
    try:
        expected_value = Decimal(candidate["value"])
        expected_start = (date.fromisoformat(candidate["period_start"])
                          if candidate.get("period_start") else None)
        expected_end = date.fromisoformat(candidate["period_end"])
    except (KeyError, TypeError, ValueError, InvalidOperation):
        return False
    source = candidate.get("source", {})
    expected_table = _compact(source.get("table"))
    return all((
        fact.value == expected_value,
        _compact(fact.raw_value) == _compact(candidate.get("raw_value")),
        fact.metric_code == candidate.get("metric_code"),
        fact.scope == candidate.get("scope"),
        fact.currency == candidate.get("currency"),
        fact.unit == candidate.get("unit"),
        _compact(fact.raw_unit).removeprefix("人民币") == candidate.get("raw_unit", candidate.get("unit")),
        fact.period_start == expected_start,
        fact.period_end == expected_end,
        fact.provenance.get("pdf_page_number") == source.get("pdf_page_number"),
        bool(expected_table) and expected_table in _compact(fact.provenance.get("table_header")),
        _compact(fact.provenance.get("column_label")) == _compact(source.get("column_label")),
        bool(fact.company),
        not candidate.get("company") or fact.company == candidate["company"],
    ))


def verify_against_reference(extracted: list[FinancialFact], reference: dict | str | Path) -> list[FinancialFact]:
    """Compare actual extraction to a source-bound reference without copying reference values.

    This is deterministic development-reference verification, not an external audit or
    human review. Every decision retains the original extraction and reference identity.
    Different report_id naming schemes are permitted only when the PDF SHA-256 matches.
    """
    if isinstance(reference, (str, Path)):
        reference = load_reference(reference)
    output = []
    reference_hash = hashlib.sha256(json.dumps(reference, sort_keys=True, ensure_ascii=False,
                                               default=str).encode()).hexdigest()
    for original in extracted:
        fact = original.model_copy(deep=True)
        source_matches = bool(reference.get("source_sha256")) and (
            fact.provenance.get("source_sha256") == reference["source_sha256"]
        )
        candidates = [candidate for candidate in reference.get("facts", [])
                      if candidate.get("metric_code") == fact.metric_code]
        matches = [candidate for candidate in candidates if _reference_matches(fact, candidate)]
        eligible = (fact.value is not None and not fact.conflict_group_id
                    and fact.validation_status in {"pending", "verified"}
                    and bool(fact.source_evidence_ids))
        success = source_matches and eligible and len(matches) == 1
        record = {
            "method": "reference_comparison", "checked_at": datetime.now(UTC).isoformat(),
            "outcome": "matched" if success else "not_verified", "human_reviewed": False,
            "reference_purpose": reference.get("purpose"), "reference_sha256": reference_hash,
            "reference_report_id": reference.get("report_id"),
            "reference_fact_id": matches[0]["id"] if success else None,
            "source_sha256": fact.provenance.get("source_sha256"),
            "source_identity_matched": source_matches,
            "original_extraction": original.model_dump(mode="json", exclude={"review_record"}),
            "boundary": "development reference comparison; no external authenticity or human review claim",
        }
        fact.review_record = record
        if success:
            fact.validation_status = "verified"
        elif fact.validation_status == "verified":
            fact.validation_status = "pending"
        output.append(fact)
    return output
