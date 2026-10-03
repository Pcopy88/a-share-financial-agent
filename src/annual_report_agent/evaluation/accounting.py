"""Observed provider usage only; no tokenizer estimates or built-in prices.

Pricing contract: {effective_date: YYYY-MM-DD, currency: CNY,
models: {model_id: {input_cache_hit_per_million: decimal_string,
input_cache_miss_per_million: decimal_string, output_per_million: decimal_string}}}.
Embedding uses total_tokens_per_million. Price dates are caller-supplied metadata,
not a claim that the prices remain current. Unknown usage never means free usage.
"""

from datetime import date
from decimal import Decimal, InvalidOperation, localcontext

_TOKEN_FIELDS = {
    "input_tokens": "prompt_tokens", "input_cache_hit_tokens": "prompt_cache_hit_tokens",
    "input_cache_miss_tokens": "prompt_cache_miss_tokens", "output_tokens": "completion_tokens",
    "total_tokens": "total_tokens",
}
_RATES = {"input_cache_hit_per_million", "input_cache_miss_per_million",
          "output_per_million", "total_tokens_per_million"}


def _pricing(pricing):
    if pricing is None:
        return None
    try:
        when = pricing["effective_date"]
        if date.fromisoformat(when).isoformat() != when:
            raise ValueError("date must use YYYY-MM-DD")
        currency, models = pricing["currency"], pricing["models"]
        if not isinstance(currency, str) or len(currency) != 3 or not currency.isupper():
            raise ValueError("currency must be an explicit uppercase three-letter code")
        if not isinstance(models, dict):
            raise ValueError("models must be a mapping")
        normalized = {}
        for model, rates in models.items():
            if not isinstance(rates, dict) or set(rates) - _RATES:
                raise ValueError("unknown pricing rate")
            normalized[model] = {}
            for key, value in rates.items():
                if isinstance(value, (float, bool)) or not isinstance(value, (str, int, Decimal)):
                    raise ValueError("prices must be exact decimal strings or integers")
                amount = Decimal(value)
                if not amount.is_finite() or amount < 0:
                    raise ValueError("prices must be finite and nonnegative")
                normalized[model][key] = amount
        return {"effective_date": when, "currency": currency, "models": normalized}
    except (KeyError, TypeError, InvalidOperation, AttributeError) as exc:
        raise ValueError("pricing requires effective_date, currency and explicit model rates") from exc


def _identity(record):
    if record.get("mode") == "mock" or str(record.get("model", "")).startswith("mock"):
        return "mock", record.get("model", "mock-embedding")
    if record.get("phase") in {"document_embedding", "query_embedding"}:
        # Current live indexer has one supported embedding profile, per its contract.
        if record.get("mode") == "live":
            return "dashscope", record.get("model", "text-embedding-v3")
    provider, model = record.get("provider"), record.get("model", record.get("model_id"))
    if provider in {"deepseek", "dashscope"}:
        return provider, model or "unknown"
    if isinstance(model, str) and model.startswith("deepseek"):
        return "deepseek", model
    if model == "text-embedding-v3":
        return "dashscope", model
    return "unknown", model or "unknown"


def _count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _record_tokens(record, provider):
    raw = record.get("usage")
    raw = raw if isinstance(raw, dict) else {}
    if provider == "dashscope":
        raw = {**raw, "total_tokens": record.get("total_tokens", record.get("usage_total_tokens", raw.get("total_tokens")))}
    tokens = {name: _count(raw.get(field)) for name, field in _TOKEN_FIELDS.items()}
    diagnostics = []
    for field in _TOKEN_FIELDS.values():
        if raw.get(field) is not None and _count(raw[field]) is None:
            diagnostics.append("INVALID_TOKEN_COUNT:" + field)
    if provider == "deepseek":
        prompt, hit, miss, output, total = (tokens[k] for k in _TOKEN_FIELDS)
        if None not in (prompt, hit, miss) and hit + miss != prompt:
            diagnostics.append("CACHE_PARTITION_MISMATCH")
        if None not in (prompt, output, total) and prompt + output != total:
            diagnostics.append("TOTAL_TOKEN_MISMATCH")
    return tokens, diagnostics


def _cost(tokens, provider, model, pricing, diagnostics):
    if pricing is None or diagnostics or provider not in {"deepseek", "dashscope"}:
        return None
    rates = pricing["models"].get(model, {})
    terms = ({"input_cache_hit_tokens": "input_cache_hit_per_million",
              "input_cache_miss_tokens": "input_cache_miss_per_million",
              "output_tokens": "output_per_million"} if provider == "deepseek"
             else {"total_tokens": "total_tokens_per_million"})
    if any(tokens[key] is None or rate not in rates for key, rate in terms.items()):
        return None
    with localcontext() as context:
        context.prec = 40
        return sum((Decimal(tokens[key]) * rates[rate] for key, rate in terms.items()), Decimal(0)) / Decimal(1_000_000)


def _decimal_string(value):
    return format(value, "f") if value is not None else None


def summarize_usage(records, pricing=None) -> dict:
    """Summarize existing N4/N8 and indexer usage records as JSON-safe primitives.

    Each record is one logical call and its final provider response. `attempts`
    includes its first HTTP request. Prior retry usage is absent in the current
    clients; retries therefore make the complete cost unknown. Missing attempts
    (legacy embedding records) preserve only a lower bound of one request.
    Fields ending `_known_sum` and cost.known_subtotal describe observed subsets.
    Completely failed calls absent from `records` cannot be reconstructed here.
    """
    price = _pricing(pricing)
    grouped = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("usage records must be mappings")
        grouped.setdefault(_identity(record), []).append(record)
    groups = []
    for (provider, model), items in sorted(grouped.items()):
        known_attempts, attempts_complete, attempts_without_usage = 0, True, 0
        token_rows, costs, diagnostics = [], [], []
        usage_complete = True
        for index, item in enumerate(items):
            attempts = _count(item.get("attempts"))
            if not attempts:
                attempts_complete = False
                attempts = None
            known_attempts += attempts or 1
            tokens, warnings = _record_tokens(item, provider)
            diagnostics.extend({"record_index": index, "code": w} for w in warnings)
            token_rows.append(tokens)
            required = (list(_TOKEN_FIELDS) if provider == "deepseek" else
                        ["total_tokens"] if provider == "dashscope" else [])
            observed = bool(required) and all(tokens[key] is not None for key in required) and not warnings
            if not observed or attempts != 1:
                usage_complete = False
            if attempts is not None:
                attempts_without_usage += attempts - int(observed)
            costs.append(_cost(tokens, provider, model, price, warnings))
        token_summary = {}
        for key in _TOKEN_FIELDS:
            known = [tokens[key] for tokens in token_rows if tokens[key] is not None]
            token_summary[key] = sum(known) if len(known) == len(items) else None
            token_summary[key + "_known_sum"] = sum(known) if known else None
            token_summary[key + "_unknown_records"] = len(items) - len(known)
        known_costs = [value for value in costs if value is not None]
        subtotal = sum(known_costs, Decimal(0)) if known_costs else None
        cost_complete = usage_complete and len(known_costs) == len(items)
        groups.append({"provider": provider, "model": model, "logical_calls": len(items),
                       "http_attempts": known_attempts if attempts_complete else None,
                       "http_attempts_lower_bound": known_attempts,
                       "retries": known_attempts - len(items) if attempts_complete else None,
                       "attempts_without_usage": attempts_without_usage if attempts_complete else None,
                       "usage_complete": usage_complete, "tokens": token_summary,
                       "cost": {"known_subtotal": _decimal_string(subtotal),
                                "total": _decimal_string(subtotal) if cost_complete else None,
                                "complete": cost_complete}, "diagnostics": diagnostics})
    known_costs = [Decimal(g["cost"]["known_subtotal"]) for g in groups if g["cost"]["known_subtotal"] is not None]
    subtotal = sum(known_costs, Decimal(0)) if known_costs else None
    complete = bool(groups) and all(g["cost"]["complete"] for g in groups)
    return {"schema_version": "1.0", "logical_calls": sum(g["logical_calls"] for g in groups),
            "groups": groups, "cost": {"currency": price["currency"] if price else None,
                "pricing_effective_date": price["effective_date"] if price else None,
                "known_subtotal": _decimal_string(subtotal), "total": _decimal_string(subtotal) if complete else None,
                "complete": complete},
            "coverage_boundary": "仅覆盖传入记录；当前记录只含最终响应usage，重试或未记录失败的用量/费用未知，known_subtotal不是完整账单。"}
