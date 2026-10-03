import json
import sys
from types import SimpleNamespace

import pytest


def generation(usage=None, attempts=1, **extra):
    return {"stage": "N8", "model": "deepseek-flash", "attempts": attempts,
            "usage": usage, **extra}


def usage():
    return {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
            "prompt_cache_hit_tokens": 40, "prompt_cache_miss_tokens": 60}


def prices():
    # Hypothetical rates for tests only, not a published vendor price.
    return {"effective_date": "2026-10-03", "currency": "CNY", "models": {
        "deepseek-flash": {"input_cache_hit_per_million": "1", "input_cache_miss_per_million": "2",
                           "output_per_million": "3"},
        "text-embedding-v3": {"total_tokens_per_million": "0.5"}}}


def test_cache_partition_is_not_added_to_prompt_total_and_prices_are_decimal():
    from annual_report_agent.evaluation.accounting import summarize_usage

    result = summarize_usage([generation(usage())], prices())
    group = result["groups"][0]
    assert group["provider"] == "deepseek"
    assert group["tokens"]["input_tokens"] == 100
    assert group["tokens"]["total_tokens"] == 120
    assert group["cost"]["total"] == "0.00022"
    assert result["cost"]["total"] == "0.00022"
    json.dumps(result, allow_nan=False)


def test_retry_only_reports_final_response_usage_and_total_cost_is_unknown():
    from annual_report_agent.evaluation.accounting import summarize_usage

    result = summarize_usage([generation(usage(), attempts=3)], prices())
    group = result["groups"][0]
    assert group["http_attempts"] == 3 and group["retries"] == 2
    assert group["attempts_without_usage"] == 2
    assert not group["usage_complete"]
    assert group["cost"]["known_subtotal"] == "0.00022"
    assert group["cost"]["total"] is None
    assert result["cost"]["total"] is None


def test_missing_usage_is_unknown_not_free_or_zero_tokens():
    from annual_report_agent.evaluation.accounting import summarize_usage

    group = summarize_usage([generation()], prices())["groups"][0]
    assert group["tokens"]["input_tokens"] is None
    assert group["tokens"]["input_tokens_known_sum"] is None
    assert group["cost"]["known_subtotal"] is None
    assert group["cost"]["total"] is None


def test_existing_embedding_record_exposes_unknown_attempt_count():
    from annual_report_agent.evaluation.accounting import summarize_usage

    group = summarize_usage([{"phase": "query_embedding", "mode": "live", "total_tokens": 10}], prices())["groups"][0]
    assert group["provider"] == "dashscope"
    assert group["tokens"]["total_tokens"] == 10
    assert group["http_attempts"] is None and group["retries"] is None
    assert group["http_attempts_lower_bound"] == 1
    assert group["cost"]["known_subtotal"] == "0.000005"
    assert group["cost"]["total"] is None


def test_two_providers_aggregate_without_conflating_token_types():
    from annual_report_agent.evaluation.accounting import summarize_usage

    result = summarize_usage([generation(usage()),
        {"phase": "document_embedding", "mode": "live", "total_tokens": 10, "attempts": 1}], prices())
    assert len(result["groups"]) == 2 and result["logical_calls"] == 2
    assert result["cost"]["total"] == "0.000225"


def test_unknown_cache_split_prevents_cache_price_estimation():
    from annual_report_agent.evaluation.accounting import summarize_usage

    raw = usage()
    del raw["prompt_cache_hit_tokens"]
    result = summarize_usage([generation(raw)], prices())
    assert result["groups"][0]["tokens"]["input_tokens"] == 100
    assert result["cost"]["total"] is None


def test_inconsistent_usage_counts_are_reported_not_costed():
    from annual_report_agent.evaluation.accounting import summarize_usage

    result = summarize_usage([generation({**usage(), "prompt_cache_hit_tokens": 99})], prices())
    assert result["groups"][0]["diagnostics"]
    assert result["cost"]["total"] is None


@pytest.mark.parametrize("pricing", [{}, {"currency": "CNY", "models": {}},
    {"effective_date": "yesterday", "currency": "CNY", "models": {}},
    {"effective_date": "2026-10-03", "currency": "CNY", "models": {"deepseek-flash": {"output_per_million": 1.5}}}])
def test_pricing_requires_explicit_date_and_exact_nonnegative_decimal_rates(pricing):
    from annual_report_agent.evaluation.accounting import summarize_usage

    with pytest.raises(ValueError):
        summarize_usage([generation(usage())], pricing)


def test_no_pricing_and_mock_data_cannot_be_claimed_as_real_spend():
    from annual_report_agent.evaluation.accounting import summarize_usage

    assert summarize_usage([generation(usage())])["cost"]["total"] is None
    result = summarize_usage([{"phase": "query_embedding", "mode": "mock", "total_tokens": 10}], prices())
    assert result["groups"][0]["provider"] == "mock"
    assert result["cost"]["total"] is None


def test_resource_sampler_records_rss_without_importing_torch(monkeypatch):
    from annual_report_agent.evaluation.resources import ResourceSampler

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    values = iter([100, 240])
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=next(values)))
    with ResourceSampler(process=process) as sample:
        pass
    assert sample.result["peak_rss_bytes"] == 240
    assert sample.result["gpu_peak_allocated_bytes"] is None
    assert sample.result["cloud_gpu_memory_bytes"] is None
    assert "torch" not in sys.modules


def test_resource_sampler_only_reads_already_initialized_cuda(monkeypatch):
    from annual_report_agent.evaluation.resources import ResourceSampler

    calls = []
    cuda = SimpleNamespace(is_initialized=lambda: True, current_device=lambda: 0,
        reset_peak_memory_stats=lambda d: calls.append(("reset", d)),
        max_memory_allocated=lambda d: 80, max_memory_reserved=lambda d: 100)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    with ResourceSampler(process=SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=12))) as sample:
        pass
    assert calls == [("reset", 0)]
    assert sample.result["gpu_peak_allocated_bytes"] == 80
    assert sample.result["gpu_peak_reserved_bytes"] == 100


def test_uninitialized_cuda_is_not_initialized_and_context_preserves_exception(monkeypatch):
    from annual_report_agent.evaluation.resources import ResourceSampler

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)))
    with pytest.raises(RuntimeError, match="original"):
        with ResourceSampler(process=SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=12))) as sample:
            raise RuntimeError("original")
    assert sample.result["gpu_peak_reserved_bytes"] is None
