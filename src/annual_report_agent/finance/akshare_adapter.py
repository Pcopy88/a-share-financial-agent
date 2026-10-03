"""Optional AKShare candidate data, isolated from verified PDF financial facts.

Primary references (checked 2026-10-03):
https://akshare.akfamily.xyz/data/stock/stock.html
https://github.com/akfamily/akshare/blob/main/akshare/stock_feature/stock_three_report_em.py

Mapping v1 covers mainland Shanghai/Shenzhen non-financial annual statements only.
The source's reportType=1 cumulative/consolidated endpoints return CNY yuan amounts.
API candidates always require a separate PDF reconciliation before use in calculations.
"""

import hashlib
import importlib.util
import json
import math
import multiprocessing
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from queue import Empty
from typing import Any
from uuid import uuid4

import pandas as pd

from annual_report_agent.finance.normalizer import mark_conflicts, normalize_amount, normalize_fact
from annual_report_agent.schemas.finance import FinancialFact

MAPPING_VERSION = "eastmoney-consolidated-annual-v1"
STATEMENTS = {
    "balance": {
        "api": "stock_balance_sheet_by_report_em", "endpoint": "zcfzbAjaxNew",
        "fields": {"TOTAL_ASSETS": "total_assets", "TOTAL_LIABILITIES": "total_liabilities",
                   "TOTAL_CURRENT_ASSETS": "current_assets", "TOTAL_CURRENT_LIAB": "current_liabilities"},
    },
    "income": {
        "api": "stock_profit_sheet_by_report_em", "endpoint": "lrbAjaxNew",
        "fields": {"OPERATE_INCOME": "revenue", "OPERATE_COST": "operating_cost",
                   "NETPROFIT": "net_profit", "PARENT_NETPROFIT": "parent_net_profit"},
    },
    "cash_flow": {
        "api": "stock_cash_flow_sheet_by_report_em", "endpoint": "xjllbAjaxNew",
        "fields": {"NETCASH_OPERATE": "operating_cash_flow"},
    },
}
_ALLOWED_APIS = {value["api"] for value in STATEMENTS.values()} | {"stock_zh_a_disclosure_report_cninfo"}
_META_FIELDS = {"SECURITY_CODE", "REPORT_DATE", "REPORT_TYPE", "CURRENCY", "NOTICE_DATE", "UPDATE_DATE"}


class AdapterError(ValueError):
    def __init__(self, code: str, detail: str = "", *, snapshot: dict | None = None):
        self.code, self.snapshot = code, snapshot
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass
class StatementImport:
    facts: list[FinancialFact]
    snapshot: dict
    warnings: list[str] = field(default_factory=list)


@dataclass
class FinancialImport:
    facts: list[FinancialFact] = field(default_factory=list)
    snapshots: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ReconciliationResult:
    """Both original source families, with only PDF-backed matches promoted."""

    facts: list[FinancialFact]
    diagnostics: list[dict] = field(default_factory=list)


def _reconciliation_key(fact):
    return (fact.company, fact.metric_code, fact.period_start, fact.period_end,
            fact.currency, fact.unit, fact.scope, fact.report_id)


def _raw_amount_matches(fact):
    try:
        return (fact.value is not None and fact.value.is_finite() and bool(fact.raw_unit)
                and normalize_amount(fact.raw_value, fact.raw_unit) == fact.value)
    except ValueError:
        return False


def _pdf_provenance_complete(fact):
    # Reuse the extractor's row and column semantics rather than a second label map.
    from annual_report_agent.finance.pdf_facts import _STATEMENT_METRICS, _column_period, _metric

    p = fact.provenance
    bbox = p.get("bbox")
    header = p.get("table_header", "")
    if not isinstance(header, str):
        return False
    statement = next((name for name in ("资产负债表", "利润表", "现金流量表") if name in header), None)
    if not statement or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    if fact.metric_code not in _STATEMENT_METRICS[statement]:
        return False
    if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in bbox):
        return False
    if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        return False
    if not (isinstance(p.get("pdf_page_number"), int) and p["pdf_page_number"] > 0):
        return False
    if not (isinstance(p.get("row_index"), int) and p["row_index"] >= 0
            and isinstance(p.get("column_index"), int) and p["column_index"] > 0):
        return False
    column = p.get("column_label", "")
    # Relative-only column labels cannot be independently checked from these inputs.
    if not isinstance(column, str) or not re.search(r"20\d{2}", column):
        return False
    period = _column_period(column, statement, None)
    if period != (fact.period_start, fact.period_end):
        return False
    return all((
        p.get("source_type") == "pdf", bool(re.fullmatch(r"[a-fA-F0-9]{64}", str(p.get("source_sha256", "")))),
        bool(fact.source_id), fact.source_id in fact.source_evidence_ids,
        _metric(p.get("row_label", "")) == fact.metric_code,
        ("合并" if fact.scope == "consolidated" else "母公司") in header,
        _raw_amount_matches(fact), bool(fact.company), bool(fact.currency),
    ))


def _api_provenance_complete(fact):
    p = fact.provenance
    spec = next((s for s in STATEMENTS.values() if s["api"] == p.get("api_name")), None)
    if spec is None:
        return False
    fields = [field for field, metric in spec["fields"].items() if metric == fact.metric_code]
    pointer = str(p.get("json_pointer", ""))
    return all((
        p.get("source_type") == "akshare_snapshot",
        bool(re.fullmatch(r"[a-fA-F0-9]{64}", str(p.get("snapshot_hash", "")))),
        fact.source_id == p.get("snapshot_hash"),
        bool(fields) and bool(re.fullmatch(r"/records/\d+/" + fields[0], pointer)),
        bool(p.get("fetched_at")), bool(p.get("source_url")), _raw_amount_matches(fact),
        p.get("mapping_version") == MAPPING_VERSION,
    ))


def reconcile_with_pdf(api_facts: list[FinancialFact], pdf_facts: list[FinancialFact]) -> ReconciliationResult:
    """Exactly reconcile independent API and PDF facts, preserving both sources.

    Return `.facts` (API records first, PDF records second) and `.diagnostics`.
    Only matching PDF records become verified; API records remain pending and never
    acquire PDF page numbers. Same report_id is required to avoid silently reconciling
    a restated report with an unrelated version. Relative-only PDF column labels or
    missing table geometry stay pending for review. This checks provenance consistency,
    not the external authenticity of a PDF or a substitute for a human audit.
    """
    api = [fact.model_copy(deep=True) for fact in api_facts]
    pdf = [fact.model_copy(deep=True) for fact in pdf_facts]
    diagnostics = []
    eligible_api = {}
    for fact in api:
        if fact.validation_status == "verified":
            fact.validation_status = "pending"
        if fact.validation_status in {"conflict", "missing", "superseded"} or fact.conflict_group_id:
            continue
        if not _api_provenance_complete(fact):
            diagnostics.append({"code": "API_PROVENANCE_INCOMPLETE", "fact_ids": [fact.id]})
            continue
        eligible_api.setdefault(_reconciliation_key(fact), []).append(fact)
    eligible_pdf = {}
    for fact in pdf:
        if fact.validation_status in {"conflict", "missing", "superseded"} or fact.conflict_group_id:
            diagnostics.append({"code": "PDF_FACT_NOT_ELIGIBLE", "fact_ids": [fact.id]})
            continue
        if not _pdf_provenance_complete(fact):
            fact.validation_status = "pending"
            diagnostics.append({"code": "PDF_PROVENANCE_INCOMPLETE", "fact_ids": [fact.id]})
            continue
        eligible_pdf.setdefault(_reconciliation_key(fact), []).append(fact)
    for key, pdf_group in eligible_pdf.items():
        api_group = eligible_api.get(key, [])
        if not api_group:
            diagnostics.append({"code": "NO_COMPARABLE_API_FACT", "fact_ids": [f.id for f in pdf_group]})
            continue
        group = api_group + pdf_group
        if len({f.value for f in group}) != 1:
            identity = "|".join(sorted(f.id for f in group))
            conflict = "conflict-" + hashlib.sha256(identity.encode()).hexdigest()[:20]
            for fact in group:
                fact.validation_status, fact.conflict_group_id = "conflict", conflict
                fact.review_record = {"method": "api_pdf_exact_reconciliation", "outcome": "conflict",
                                      "human_reviewed": False, "fact_ids": [f.id for f in group]}
            diagnostics.append({"code": "API_PDF_VALUE_CONFLICT", "fact_ids": [f.id for f in group],
                                "conflict_group_id": conflict})
            continue
        for fact in pdf_group:
            original = fact.model_dump(mode="json", exclude={"review_record"})
            fact.validation_status = "verified"
            fact.review_record = {
                "method": "api_pdf_exact_reconciliation", "outcome": "matched",
                "checked_at": datetime.now(UTC).isoformat(), "human_reviewed": False,
                "api_fact_ids": [f.id for f in api_group],
                "snapshot_hashes": sorted({f.provenance["snapshot_hash"] for f in api_group}),
                "pdf_source_sha256": fact.provenance["source_sha256"],
                "original_extraction": original,
                "boundary": "exact source reconciliation; no human or external authenticity review claim",
            }
        for fact in api_group:
            fact.review_record = {"method": "api_pdf_exact_reconciliation", "outcome": "matched",
                                  "human_reviewed": False, "pdf_fact_ids": [f.id for f in pdf_group],
                                  "boundary": "API candidate retained; PDF record is the calculable operand"}
        diagnostics.append({"code": "API_PDF_MATCHED", "fact_ids": [f.id for f in group]})
    return ReconciliationResult(api + pdf, diagnostics)


def market_symbol(stock_code: str) -> str:
    if not isinstance(stock_code, str) or not re.fullmatch(r"\d{6}", stock_code):
        raise AdapterError("INVALID_STOCK_CODE", "require six digits")
    if stock_code.startswith(("600", "601", "603", "605", "688", "689")):
        return "SH" + stock_code
    if stock_code.startswith(("000", "001", "002", "003", "300", "301")):
        return "SZ" + stock_code
    raise AdapterError("UNSUPPORTED_MARKET", "first adapter covers Shanghai/Shenzhen A-shares")


def _ak_worker(api_name, params, output):
    try:
        import akshare

        output.put(("ok", getattr(akshare, api_name)(**params), akshare.__version__))
    except Exception as exc:
        output.put(("error", type(exc).__name__, ""))


def call_akshare(api_name: str, params: dict, *, timeout: float = 30):
    """Bound upstream functions which do not consistently expose a network timeout.

    A separate spawn process is terminated on timeout; no global requests monkeypatch
    or abandoned background download thread is left behind.
    """
    if api_name not in _ALLOWED_APIS:
        raise AdapterError("UNKNOWN_SOURCE_API")
    if importlib.util.find_spec("akshare") is None:
        raise AdapterError("AKSHARE_NOT_INSTALLED", "run: uv sync --extra acquisition; local upload remains available")
    context = multiprocessing.get_context("spawn")
    output = context.Queue(maxsize=1)
    process = context.Process(target=_ak_worker, args=(api_name, params, output), daemon=True)
    process.start()
    try:
        status, payload, version = output.get(timeout=timeout)
        if status != "ok":
            raise AdapterError("SOURCE_UNAVAILABLE", payload)
        return payload, version
    except Empty as exc:
        raise AdapterError("SOURCE_TIMEOUT", "local PDF upload remains available") from exc
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        output.close()


def _snapshot_scalar(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, (Decimal, date, datetime)):
        return str(value)
    if hasattr(value, "item"):
        return _snapshot_scalar(value.item())
    return str(value)


def adapt_statement(frame: pd.DataFrame, *, statement: str, stock_code: str, year: int,
                    company: str, report_id: str, akshare_version: str = "unknown",
                    expected_columns: list[str] | None = None) -> StatementImport:
    """Map a returned DataFrame while keeping its complete available raw snapshot.

    Required-field changes fail closed. Pass a prior snapshot's columns to additionally
    reject added/removed source columns after freezing an integration baseline. This
    snapshot represents the fetched response, never a historical point-in-time view.
    """
    symbol = market_symbol(stock_code)
    if statement not in STATEMENTS or not 1990 <= year <= 2100 or not company.strip():
        raise AdapterError("INVALID_REQUEST")
    if not isinstance(frame, pd.DataFrame):
        raise AdapterError("FIELD_SCHEMA_CHANGED", "expected DataFrame")
    spec = STATEMENTS[statement]
    columns = list(frame.columns)
    records = [{str(key): _snapshot_scalar(value) for key, value in row.items()}
               for row in frame.to_dict(orient="records")]
    payload = json.dumps({"columns": columns, "records": records}, sort_keys=True,
                         ensure_ascii=False, allow_nan=False)
    snapshot_hash = hashlib.sha256(payload.encode()).hexdigest()
    snapshot: dict[str, Any] = {
        "source_type": "akshare_snapshot", "api_name": spec["api"],
        "query_params": {"symbol": symbol}, "akshare_version": akshare_version,
        "fetched_at": datetime.now(UTC).isoformat(), "snapshot_hash": snapshot_hash,
        "columns": columns, "records": records, "mapping_version": MAPPING_VERSION,
        "source_url": "https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/" + spec["endpoint"],
        "endpoint_contract": {"unit": "元", "currency": "CNY", "scope": "consolidated",
                              "reportType": "1", "scope_limit": "non_financial_annual_only"},
        "warning": "upstream latest snapshot; not historical point-in-time or PDF-verified data",
    }
    required = _META_FIELDS | set(spec["fields"])
    if not required.issubset(columns) or (expected_columns is not None and set(columns) != set(expected_columns)):
        raise AdapterError("FIELD_SCHEMA_CHANGED", f"required missing: {sorted(required - set(columns))}",
                           snapshot=snapshot)
    facts = []
    for row_index, row in enumerate(records):
        try:
            end = date.fromisoformat(str(row["REPORT_DATE"])[:10])
        except (TypeError, ValueError) as exc:
            raise AdapterError("FIELD_SCHEMA_CHANGED", "invalid REPORT_DATE", snapshot=snapshot) from exc
        if end != date(year, 12, 31):
            continue
        if str(row["SECURITY_CODE"]).zfill(6) != stock_code:
            raise AdapterError("SOURCE_COMPANY_MISMATCH", snapshot=snapshot)
        if row["CURRENCY"] != "CNY":
            raise AdapterError("UNSUPPORTED_CURRENCY", snapshot=snapshot)
        if row["REPORT_TYPE"] not in {"年报", "年度报告"}:
            raise AdapterError("UNSUPPORTED_REPORT_BASIS", snapshot=snapshot)
        for source_field, metric in spec["fields"].items():
            value = row[source_field]
            warnings = ["api_only_not_pdf_verified"]
            if isinstance(value, float):
                warnings.append("upstream_float_requires_pdf_verification")
            raw = None if value is None else str(value)
            facts.append(normalize_fact(
                id=f"api-{snapshot_hash[:20]}-{row_index}-{metric}", metric_code=metric,
                raw_value=raw, raw_unit="元", company=company, currency="CNY", scope="consolidated",
                period_start=None if statement == "balance" else date(year, 1, 1), period_end=end,
                report_id=report_id, source_id=snapshot_hash, source_evidence_ids=[], warnings=warnings,
                provenance={"source_type": "akshare_snapshot", "api_name": spec["api"],
                            "snapshot_hash": snapshot_hash, "json_pointer": f"/records/{row_index}/{source_field}",
                            "fetched_at": snapshot["fetched_at"], "source_url": snapshot["source_url"],
                            "notice_date": row["NOTICE_DATE"], "update_date": row["UPDATE_DATE"],
                            "mapping_version": MAPPING_VERSION, "akshare_version": akshare_version},
            ))
    return StatementImport(mark_conflicts(facts), snapshot,
                           [] if facts else ["requested_annual_period_not_found"])


def save_snapshot(snapshot: dict, destination: str | Path) -> Path:
    """Write one immutable raw-response snapshot; existing files are never overwritten."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(snapshot, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return destination


def fetch_financial_facts(stock_code: str, year: int, *, company: str, report_id: str,
                          ak_client=None, snapshot_dir: str | Path | None = None) -> FinancialImport:
    """Fetch each source once, retaining partial success; never block local PDF operation."""
    symbol = market_symbol(stock_code)
    result = FinancialImport()
    for statement, spec in STATEMENTS.items():
        try:
            if ak_client is None:
                frame, version = call_akshare(spec["api"], {"symbol": symbol})
            else:
                frame = getattr(ak_client, spec["api"])(symbol=symbol)
                version = getattr(ak_client, "__version__", "injected_source")
            imported = adapt_statement(frame, statement=statement, stock_code=stock_code,
                                       year=year, company=company, report_id=report_id,
                                       akshare_version=version)
            result.facts.extend(imported.facts)
            result.snapshots.append(imported.snapshot)
            result.warnings.extend(imported.warnings)
        except AdapterError as exc:
            result.errors.append({"api_name": spec["api"], "code": exc.code, "detail": str(exc)})
            if exc.snapshot:
                result.snapshots.append(exc.snapshot)
            if exc.code == "AKSHARE_NOT_INSTALLED":
                break
        except Exception as exc:
            result.errors.append({"api_name": spec["api"], "code": "SOURCE_UNAVAILABLE",
                                  "detail": type(exc).__name__})
    if snapshot_dir:
        for snapshot in result.snapshots:
            # The content hash can repeat; every fetch has its own immutable timestamped observation.
            observation_id = uuid4().hex
            save_snapshot(snapshot, Path(snapshot_dir) / f"{snapshot['snapshot_hash']}-{observation_id}.json")
    result.facts = mark_conflicts(result.facts)
    return result
