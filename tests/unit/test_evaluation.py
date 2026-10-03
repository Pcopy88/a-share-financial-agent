"""Small gold sets must not masquerade as exhaustive recall or mock quality scores."""

from importlib.util import find_spec

import pytest


def module():
    assert find_spec("annual_report_agent.evaluation.metrics"), "evaluation metrics missing"
    from annual_report_agent.evaluation import metrics
    return metrics


def test_precision_denominator_is_five_and_overlapping_hits_only_count_once():
    scores = module().retrieval_scores([{"a"}, {"a"}, set()], {"a", "b"})
    assert scores["precision_at_5"] == 0.2
    assert scores["gold_evidence_coverage_at_30"] == 0.5
    assert scores["recall_at_30"] is None
    assert scores["hit_at_5"] == 1


def test_empty_gold_and_zero_baseline_are_not_perfect_scores():
    metrics = module()
    scores = metrics.retrieval_scores([], set())
    assert scores["precision_at_5"] is None
    assert metrics.relative_change(0, 0.6) is None
    assert metrics.relative_change(0.6, 0.69) == pytest.approx(15.0)


def test_duplicate_chunks_keep_original_top_five_positions():
    scores = module().retrieval_scores([{"a"}] * 5 + [{"b"}], {"a", "b"})
    assert scores["precision_at_5"] == 0.2


def test_duplicate_gold_matches_never_promote_rank_31_into_top_30():
    scores = module().retrieval_scores([{"g1"}] * 30 + [{"g2"}], {"g1", "g2"},
                                       complete_gold=True)
    assert scores["recall_at_30"] == .5
    assert scores["precision_at_5"] == .2


def test_partial_gold_coverage_respects_original_ranking_and_overlap():
    scores = module().retrieval_scores([{"a"}, {"a", "b"}] + [{"a"}] * 28 + [{"c"}],
                                       {"a", "b", "c"})
    assert scores["gold_evidence_coverage_at_30"] == pytest.approx(2/3)
    assert scores["recall_at_30"] is None
    assert scores["precision_at_5"] == .4


def test_mock_runs_are_not_aggregated_into_real_retrieval_accuracy():
    metrics = module()
    rows = [{"mode": "mock", "latency_seconds": 0.1,
             "metrics": {"precision_at_5": 1.0}},
            {"mode": "live", "latency_seconds": 1.0,
             "metrics": {"precision_at_5": 0.4}}]
    result = metrics.summarize(rows)
    assert result["quality_applicable_runs"] == 1
    assert result["quality"]["precision_at_5"]["mean"] == 0.4
    assert result["mock_runs"] == 1


def test_numeric_target_must_match_unit_period_scope_and_value():
    metrics = module()
    expected = dict(value="30.21", unit="percent", period="2026H1", scope="consolidated")
    assert metrics.numeric_correct(expected, expected)
    assert not metrics.numeric_correct(expected, expected | {"scope": "parent"})
    assert not metrics.numeric_correct(expected, expected | {"period": "2026FY"})
    assert not metrics.numeric_correct(expected, expected | {"value": "0.3021"})


def test_node_timings_separate_modes_include_failed_attempts_and_preserve_quality():
    rows = [
        {"mode": "mock", "latency_seconds": 100, "metrics": {"accuracy": 1},
         "node_timings": [{"node": "N4", "latency_seconds": 100, "status": "succeeded"}]},
        {"mode": "live", "latency_seconds": 1, "metrics": {"accuracy": .5},
         "node_timings": [{"node": "N4", "latency_seconds": 1, "status": "succeeded"}]},
        {"mode": "live", "latency_seconds": 3, "error": "upstream_failure",
         "node_timings": [{"node": "N4", "latency_seconds": 3, "status": "failed"}]},
    ]
    result = module().summarize(rows)
    live = result["node_latency"]["live"]["nodes"]["N4"]
    assert live["count"] == 2 and live["succeeded"] == 1 and live["failed"] == 1
    assert live["p50"] == 2 and live["p95"] == pytest.approx(2.9)
    assert result["node_latency"]["mock"]["nodes"]["N4"]["p50"] == 100
    assert result["quality"]["accuracy"]["mean"] == .5
    assert result["errors"] == 1


def test_node_timings_count_every_subtask_and_do_not_substitute_run_latency():
    result = module().summarize([{"mode": "live", "latency_seconds": 90,
        "node_timings": [
            {"node": "N6", "latency_seconds": 0, "status": "succeeded", "subtask_id": "s1"},
            {"node": "N6", "latency_seconds": 4, "status": "succeeded", "subtask_id": "s2"},
            {"node": "N8", "latency_seconds": 8, "status": "succeeded", "subtask_id": "s2"}]}])
    nodes = result["node_latency"]["live"]["nodes"]
    assert nodes["N6"]["count"] == 2 and nodes["N6"]["p50"] == 2
    assert nodes["N8"]["p50"] == nodes["N8"]["p95"] == 8
    assert result["latency"]["live"]["p50"] == 90


def test_missing_node_timings_are_explicit_no_data_not_zero_latency():
    for rows in ([], [{"mode": "live", "latency_seconds": 5}],
                 [{"mode": "live", "latency_seconds": 5, "node_timings": []}]):
        live = module().summarize(rows)["node_latency"]["live"]
        assert live["status"] == "no_data" and live["nodes"] == {}
        assert live["runs_without_timings"] == len(rows)


def test_invalid_latencies_and_unknown_status_are_visible_without_nan_percentiles():
    import json

    rows = [{"mode": "live", "latency_seconds": 4, "node_timings": [
        {"node": "N7", "latency_seconds": float("nan"), "status": "failed"},
        {"node": "N7", "latency_seconds": -1, "status": "failed"},
        {"node": "N7", "status": "skipped"},
        {"node": "N7", "latency_seconds": True, "status": "succeeded"}]}]
    result = module().summarize(rows)
    node = result["node_latency"]["live"]["nodes"]["N7"]
    assert node["count"] == 4 and node["failed"] == 2 and node["unknown_status"] == 1
    assert node["latency_samples"] == 0 and node["unknown_latency_count"] == 4
    assert node["p50"] is None and node["p95"] is None
    json.dumps(result, allow_nan=False)


def test_unknown_mode_is_not_counted_as_live_and_malformed_record_is_recorded():
    rows = [{"latency_seconds": 2, "node_timings": [None,
        {"node": "N4", "latency_seconds": 2, "status": "succeeded"}]}]
    result = module().summarize(rows)
    assert result["node_latency"]["live"]["status"] == "no_data"
    unknown = result["node_latency"]["unknown"]
    assert unknown["invalid_records"] == 1 and unknown["nodes"]["N4"]["count"] == 1
