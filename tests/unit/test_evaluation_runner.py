import json
from types import SimpleNamespace

import pytest

from annual_report_agent.config import Settings
from annual_report_agent.ingestion.indexer import Candidate


class Index:
    def snapshot(self, ids):
        return {identifier: {"index_version": "v"} for identifier in ids}

    def retrieve(self, task, manifests=None, mode=None, limit=30):
        return [Candidate(chunk={"id": "c", "report_id": "r", "parent_id": "p",
            "text": "经营业务为电子材料", "page_refs": [{"pdf_page_number": 2}]}, index_version="v")]


def fixture(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({"id": "q1", "question": "主营业务？", "report_ids": ["r"],
        "split": "development", "route": "semantic", "gold_evidence": [{"id": "g1",
        "report_id": "r", "pdf_page_number": 2, "quote": "经营业务为电子材料"}]},
        ensure_ascii=False), encoding="utf-8")
    runtime = SimpleNamespace(settings=Settings(_env_file=None, data_dir=tmp_path / "data"),
                              index=Index(), reranker=None)
    return runtime, path


def test_retrieval_runner_writes_recomputable_mock_separate_results(tmp_path):
    from annual_report_agent.evaluation.runner import EvaluationRunner
    runtime, cases = fixture(tmp_path)
    result = EvaluationRunner(runtime, tmp_path / "out").run(cases, experiment="R0")
    rows = [json.loads(line) for line in (tmp_path / "out" / result["id"] / "rows.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    assert rows[0]["metrics"]["precision_at_5"] == .2
    assert result["summary"]["quality"] == {}
    assert result["summary"]["mock_runs"] == 1
    assert result["dataset_sha256"]
    assert rows[0]["metrics"]["recall_at_30"] is None


def test_r3_does_not_silently_become_r2_when_model_unavailable(tmp_path):
    from annual_report_agent.evaluation.runner import EvaluationRunner
    runtime, cases = fixture(tmp_path)
    result = EvaluationRunner(runtime, tmp_path / "out").run(cases, experiment="R3")
    assert result["summary"]["errors"] == 1


def test_frozen_evidence_ablation_requires_frozen_input(tmp_path):
    from annual_report_agent.evaluation.runner import EvaluationRunner
    runtime, cases = fixture(tmp_path)
    with pytest.raises(ValueError, match="frozen"):
        EvaluationRunner(runtime, tmp_path / "out").run(cases, experiment="M1")


def test_multi_subtask_ranking_gives_second_subtask_top_hit_a_slot(tmp_path):
    from annual_report_agent.evaluation.dataset import EvalCase
    from annual_report_agent.evaluation.runner import EvaluationRunner
    runtime, _ = fixture(tmp_path)
    original = runtime.index.retrieve
    def retrieve(task, **kwargs):
        if task.query_text == "第一问":
            return [Candidate(chunk={"id": f"noise{i}", "report_id": "r", "text": "无关",
                "parent_id": "p", "page_refs": [{"pdf_page_number": 2}]}, index_version="v") for i in range(30)]
        return original(task, **kwargs)
    runtime.index.retrieve = retrieve
    case = EvalCase(id="multi", question="第一问；第二问", report_ids=["r"], split="development",
        route="hybrid", canonical_subtasks=[{"query_text": "第一问"}, {"query_text": "第二问"}],
        gold_evidence=[{"id": "g", "report_id": "r", "pdf_page_number": 2, "quote": "经营业务为电子材料"}])
    row = EvaluationRunner(runtime, tmp_path / "out")._retrieval(case, case.request(), "R0", {})
    assert row["metrics"]["precision_at_5"] == .2


def test_frozen_inputs_must_match_current_question_and_scope(tmp_path):
    from annual_report_agent.evaluation.runner import EvaluationRunner
    from annual_report_agent.schemas.query import AnalysisRequest
    runtime, _ = fixture(tmp_path)
    runtime.repo = SimpleNamespace(build=lambda version: {"index_version": version})
    frozen = {"mode": "mock", "report_index_versions": {"r": "v"}, "plan": {
        "original_question": "旧问题", "report_ids": ["r"], "hard_filters": {}}}
    with pytest.raises(ValueError, match="question"):
        EvaluationRunner(runtime, tmp_path / "out")._check_frozen(frozen,
            AnalysisRequest(question="新问题", report_ids=["r"]))


def test_c0_records_only_actually_sent_evidence(tmp_path, monkeypatch):
    from annual_report_agent.evaluation.dataset import EvalCase
    from annual_report_agent.evaluation.runner import EvaluationRunner
    from annual_report_agent.schemas.answer import AnswerDraft, SubTaskResult
    from annual_report_agent.schemas.query import QueryPlan, SubTaskPlan
    runtime, _ = fixture(tmp_path)
    runtime.repo = SimpleNamespace(build=lambda version: {"index_version": version}, list=lambda *args: [])
    runtime.generation_client_for = lambda profile: None
    task = SubTaskPlan(subtask_id="s", source_text="问题", query_text="问题", rewritten_query="问题",report_ids=["r"])
    plan = QueryPlan(original_question="问题", rewritten_question="问题", report_ids=["r"], subtasks=[task])
    evidence = [{"id": role, "role": role, "report_id": "r", "parent_id": "p", "quote": "业务",
        "page_refs": [{"pdf_page_number": 1}], "index_version": "v"} for role in ["primary", "context"]]
    frozen = {"mode": "mock", "plan": plan.model_dump(), "report_index_versions": {"r": "v"},
              "fact_snapshot": {"r": []},
              "traces": [{"task": task.model_dump(), "evidence": evidence, "facts": [], "calculations": []}]}
    from annual_report_agent.evaluation.runner import freeze_runs
    freeze_runs({"q": frozen}, tmp_path / "freeze.json")
    frozen = json.loads((tmp_path / "freeze.json").read_text(encoding="utf-8"))["q"]
    monkeypatch.setattr("annual_report_agent.evaluation.runner.compose", lambda *a, **kw: AnswerDraft(subtask_id="s"))
    monkeypatch.setattr("annual_report_agent.evaluation.runner.verify", lambda *a, **kw: SubTaskResult(subtask_id="s",status="insufficient_evidence"))
    case = EvalCase(id="q", question="问题",report_ids=["r"],split="development")
    run = EvaluationRunner(runtime, tmp_path / "out")._frozen_generate(case, frozen,"C0","default")
    assert [item["role"] for item in run["traces"][0]["evidence"]] == ["primary"]
    assert run["answer"]["citations"] == []


def frozen_fixture(tmp_path, *, facts=None):
    from annual_report_agent.evaluation.dataset import EvalCase
    from annual_report_agent.evaluation.runner import freeze_runs
    from annual_report_agent.schemas.query import QueryPlan, SubTaskPlan
    runtime, _ = fixture(tmp_path)
    snapshot = {"r": facts or []}
    runtime.repo = SimpleNamespace(build=lambda v: {"index_version": v},
        active_index=lambda r: {"index_version": "v"}, list=lambda *args: [])
    runtime._version_facts = lambda report_id, version: snapshot[report_id]
    runtime.generation_client_for = lambda profile: None
    task = SubTaskPlan(subtask_id="s", source_text="问题", query_text="问题", rewritten_query="问题",report_ids=["r"])
    plan = QueryPlan(original_question="问题", rewritten_question="问题", report_ids=["r"], subtasks=[task])
    run = {"mode": "mock", "plan": plan.model_dump(), "report_index_versions": {"r": "v"},
           "fact_snapshot": snapshot, "traces": [{"task": task.model_dump(), "evidence": [],
                                                   "facts": [], "calculations": []}]}
    freeze_runs({"q": run}, tmp_path / "frozen.json")
    frozen = json.loads((tmp_path / "frozen.json").read_text(encoding="utf-8"))["q"]
    case = EvalCase(id="q", question="问题",report_ids=["r"],split="development")
    return runtime, case, frozen


def test_freeze_persists_full_fact_snapshot_and_digest(tmp_path):
    _, _, frozen = frozen_fixture(tmp_path, facts=[{"id": "f", "value": "100", "validation_status": "verified"}])
    assert frozen["frozen_schema_version"] == 2
    assert frozen["fact_snapshot"]["r"][0]["value"] == "100"
    assert len(frozen["fact_snapshot_sha256"]) == 64


@pytest.mark.parametrize("changed", ["index", "value", "validation_status", "new_fact"])
def test_rewrite_ablation_rejects_changed_active_index_or_facts(tmp_path, changed):
    from annual_report_agent.evaluation.runner import EvaluationRunner
    runtime, case, frozen = frozen_fixture(tmp_path, facts=[{"id": "f", "value": "100", "validation_status": "verified"}])
    if changed == "index":
        runtime.repo.active_index = lambda r: {"index_version": "v2"}
    else:
        current = [{"id": "f", "value": "101" if changed == "value" else "100",
                    "validation_status": "pending" if changed == "validation_status" else "verified"}]
        if changed == "new_fact":
            current.append({"id": "new", "value": "99"})
        runtime._version_facts = lambda *args: current
    with pytest.raises(ValueError, match="snapshot|index"):
        EvaluationRunner(runtime, tmp_path / "out")._check_frozen(frozen, case.request(), require_current_snapshot=True)


def test_legacy_frozen_inputs_explicitly_require_regeneration(tmp_path):
    from annual_report_agent.evaluation.runner import EvaluationRunner
    runtime, case, frozen = frozen_fixture(tmp_path)
    for key in ("frozen_schema_version", "fact_snapshot", "fact_snapshot_sha256"):
        frozen.pop(key, None)
    with pytest.raises(ValueError, match="regenerate"):
        EvaluationRunner(runtime, tmp_path / "out")._check_frozen(frozen, case.request())


def test_identical_facts_in_different_repository_order_are_same_snapshot(tmp_path):
    from annual_report_agent.evaluation.runner import EvaluationRunner
    facts = [{"id": "f1", "value": "100"}, {"id": "f2", "value": "80"}]
    runtime, case, frozen = frozen_fixture(tmp_path, facts=facts)
    runtime._version_facts = lambda *args: list(reversed(facts))
    EvaluationRunner(runtime, tmp_path / "out")._check_frozen(frozen, case.request(), require_current_snapshot=True)


@pytest.mark.parametrize("accepted", [False, True])
def test_frozen_generation_returns_only_accepted_calculations(tmp_path, monkeypatch, accepted):
    from annual_report_agent.evaluation.runner import EvaluationRunner
    from annual_report_agent.schemas.answer import AnswerDraft, SubTaskResult
    runtime, case, frozen = frozen_fixture(tmp_path)
    frozen["traces"][0]["calculations"] = [{"id": "c", "formula_id": "gross_margin",
        "formula_version": "1", "operand_fact_ids": [], "result": "20", "unit": "percent", "period_label": "2026"}]
    monkeypatch.setattr("annual_report_agent.evaluation.runner.compose", lambda *a, **kw: AnswerDraft(subtask_id="s"))
    monkeypatch.setattr("annual_report_agent.evaluation.runner.verify", lambda *a, **kw: SubTaskResult(
        subtask_id="s", status="answered" if accepted else "insufficient_evidence",
        calculation_ids=["c"] if accepted else []))
    run = EvaluationRunner(runtime, tmp_path / "out")._frozen_generate(case, frozen, "M1", "default")
    assert [item["id"] for item in run["answer"]["calculations"]] == (["c"] if accepted else [])
    assert run["fact_snapshot"] == frozen["fact_snapshot"]


def test_actual_q_run_snapshot_is_checked_before_scoring(tmp_path):
    from annual_report_agent.evaluation.runner import EvaluationRunner
    runtime, _, frozen = frozen_fixture(tmp_path)
    with pytest.raises(ValueError, match="snapshot"):
        EvaluationRunner(runtime, tmp_path / "out")._check_run_snapshot(
            frozen, {"report_index_versions": {"r": "v"}, "fact_snapshot": {"r": [{"id": "changed"}]}})


def test_frozen_fact_digest_rejects_changed_snapshot_file(tmp_path):
    from annual_report_agent.evaluation.runner import EvaluationRunner
    runtime, case, frozen = frozen_fixture(tmp_path)
    frozen["fact_snapshot"]["r"] = [{"id": "changed"}]
    with pytest.raises(ValueError, match="inconsistent"):
        EvaluationRunner(runtime, tmp_path / "out")._check_frozen(frozen, case.request())


def test_failed_frozen_generation_keeps_observed_usage_in_written_row(tmp_path, monkeypatch):
    from annual_report_agent.evaluation.runner import EvaluationRunner
    runtime, case, frozen = frozen_fixture(tmp_path)
    cases = tmp_path / "failed.jsonl"
    cases.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    observed = {"stage": "composer", "request_id": "paid", "model": "served-model",
                "system_fingerprint": "fp", "usage": {"total_tokens": 19}}
    class ProviderError(Exception):
        def code(self):
            return 1
    def reject(*args, usage_sink, **kwargs):
        usage_sink.append(observed)
        raise ProviderError("must not be written")
    monkeypatch.setattr("annual_report_agent.evaluation.runner.compose", reject)
    result = EvaluationRunner(runtime, tmp_path / "out").run(
        cases, experiment="M1", frozen_path=tmp_path / "frozen.json")
    row = json.loads((tmp_path / "out" / result["id"] / "rows.jsonl").read_text(encoding="utf-8"))
    assert row["usage"] == [observed]
    assert row["error"] == "ProviderError"
    assert "must not be written" not in str(row)
    assert result["effective_profile"]["mode"] == "mock"
    assert len(result["effective_profile"]["sha256"]) == 64
