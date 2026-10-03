"""Quality metrics never treat missing labels or mock responses as real success."""

import math
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from statistics import mean


def retrieval_scores(ranked_matches: list[set[str]], gold: set[str], *, complete_gold=False) -> dict:
    """Score original TopK positions; duplicate gold does not earn extra credit.

    No candidate is removed or promoted based on answer labels. A candidate earns
    one relevant position only when it contributes at least one previously unseen
    gold item within the original first five positions.
    """
    if not gold:
        return dict(precision_at_5=None, recall_at_30=None,
                    gold_evidence_coverage_at_30=None, hit_at_5=None)
    seen, relevant_positions = set(), 0
    for matches in ranked_matches[:5]:
        novel = (set(matches) & gold) - seen
        if novel:
            relevant_positions += 1
        seen.update(set(matches) & gold)
    found = set().union(*(set(matches) & gold for matches in ranked_matches[:30]))
    coverage = len(found) / len(gold)
    return dict(precision_at_5=relevant_positions / 5,
                recall_at_30=coverage if complete_gold else None,
                gold_evidence_coverage_at_30=None if complete_gold else coverage,
                hit_at_5=float(bool(seen)))


def relative_change(baseline: float, result: float) -> float | None:
    return (result - baseline) / baseline * 100 if baseline != 0 else None


def numeric_correct(actual: dict, expected: dict, places: int = 2) -> bool:
    if any(actual.get(key) != expected.get(key) for key in ("unit", "period", "scope")):
        return False
    try:
        actual_value, expected_value = Decimal(str(actual["value"])), Decimal(str(expected["value"]))
        if not actual_value.is_finite() or not expected_value.is_finite():
            return False
        quant = Decimal(1).scaleb(-places)
        return actual_value.quantize(quant, rounding=ROUND_HALF_UP) == expected_value.quantize(
            quant, rounding=ROUND_HALF_UP)
    except (InvalidOperation, KeyError, ValueError):
        return False


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    offset = (len(values) - 1) * p
    low = int(offset)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (offset - low)


def _node_timings(rows: list[dict]) -> dict:
    """Aggregate invocation records, including failed runs and repeated subtasks.

    p50/p95 use linear interpolation and all observed nonnegative, finite invocation
    durations (successes and failures). Missing durations do not become zero and are
    counted separately. A missing/unsupported mode never enters live or mock results.
    """
    modes = {"mock": [], "live": []}
    for row in rows:
        mode = row.get("mode")
        mode = mode if mode in {"mock", "live"} else "unknown"
        modes.setdefault(mode, []).append(row)
    result = {}
    for mode, mode_rows in modes.items():
        nodes, invalid_records, runs_with_timings = {}, 0, 0
        for row in mode_rows:
            timings = row.get("node_timings")
            if not isinstance(timings, list):
                invalid_records += int(timings is not None)
                continue
            named_records = 0
            for timing in timings:
                if (not isinstance(timing, dict) or not isinstance(timing.get("node"), str)
                        or not timing["node"].strip()):
                    invalid_records += 1
                    continue
                named_records += 1
                stats = nodes.setdefault(timing["node"], {"count": 0, "succeeded": 0,
                    "failed": 0, "unknown_status": 0, "durations": []})
                stats["count"] += 1
                status = timing.get("status")
                stats[status if status in {"succeeded", "failed"} else "unknown_status"] += 1
                duration = timing.get("latency_seconds")
                if (isinstance(duration, (int, float)) and not isinstance(duration, bool)
                        and math.isfinite(duration) and duration >= 0):
                    stats["durations"].append(duration)
            runs_with_timings += int(named_records > 0)
        for stats in nodes.values():
            durations = stats.pop("durations")
            stats.update(latency_samples=len(durations),
                         unknown_latency_count=stats["count"] - len(durations),
                         p50=percentile(durations, .5), p95=percentile(durations, .95))
        result[mode] = {"status": "available" if nodes else "no_data",
                        "runs_with_timings": runs_with_timings,
                        "runs_without_timings": len(mode_rows) - runs_with_timings,
                        "invalid_records": invalid_records,
                        "nodes": dict(sorted(nodes.items()))}
    return result


def summarize(rows: list[dict]) -> dict:
    live = [row for row in rows if row.get("mode") == "live" and not row.get("error")]
    keys = set().union(*(row.get("metrics", {}).keys() for row in live))
    quality = {}
    for key in sorted(keys):
        values = [row["metrics"][key] for row in live
                  if isinstance(row.get("metrics", {}).get(key), (int, float))]
        quality[key] = {"mean": mean(values) if values else None, "applicable_cases": len(values)}
    return dict(runs=len(rows), mock_runs=sum(row.get("mode") == "mock" for row in rows),
                quality_applicable_runs=len(live), quality=quality,
                errors=sum(bool(row.get("error")) for row in rows),
                node_latency=_node_timings(rows),
                latency={mode: {"p50": percentile([r["latency_seconds"] for r in rows
                    if r.get("mode") == mode], .5), "p95": percentile([r["latency_seconds"] for r in rows
                    if r.get("mode") == mode], .95)} for mode in ("mock", "live")})
