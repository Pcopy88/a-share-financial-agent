"""Failure observations use fake clients and repositories, with no cloud calls."""
import json
from types import SimpleNamespace

import pytest

from annual_report_agent.config import Settings
from annual_report_agent.runtime import Runtime, _error
from annual_report_agent.schemas.query import SubTaskPlan


class MethodCodeError(Exception):
    def code(self):
        return 1


class Repo:
    def __init__(self):
        self.run = {"run_id": "run", "node_trace": [], "node_timings": []}

    def get(self, kind, identifier):
        return self.run if kind == "run" else {"id": identifier}

    def put(self, kind, identifier, value):
        if kind == "run":
            self.run = value

    def update_task(self, *args, **kwargs):
        pass


def setup_runtime():
    runtime = Runtime.__new__(Runtime)
    runtime.settings = Settings(_env_file=None, mode="live")
    runtime.repo = Repo()
    runtime.generation_client_for = lambda name: object()
    task = SubTaskPlan(subtask_id="s", source_text="业务", query_text="业务",
                       rewritten_query="业务", report_ids=["r"])
    state = {"run_id": "run", "task_id": "job", "request": {
        "question": "业务", "report_ids": ["r"]}, "manifests": [
        {"report_id": "r", "index_version": "v", "report_metadata": {"company": "公司"}}],
        "usage": [], "plan": {"subtasks": [task.model_dump()]}, "cursor": 0}
    return runtime, state


def test_callable_error_code_is_never_serialized_or_exposed():
    info = _error(MethodCodeError("provider detail"))
    assert info["code"] == "INTERNAL_ERROR"
    assert "provider detail" not in json.dumps(info)


@pytest.mark.parametrize("node,target", [("N4_route", "plan_query"), ("N8_compose", "compose")])
def test_response_usage_survives_subsequent_schema_rejection(monkeypatch, node, target):
    runtime, state = setup_runtime()
    observed = {"stage": target, "request_id": "paid", "usage": {"total_tokens": 31},
                "model": "served-model", "system_fingerprint": "fp"}
    def reject(*args, usage_sink, **kwargs):
        usage_sink.append(observed)
        raise ValueError("invalid business schema")
    monkeypatch.setattr("annual_report_agent.runtime." + target, reject)
    update = runtime._nodes({})[node](state)
    assert update["usage"] == [observed]
    assert runtime.repo.run["observed_usage"] == [observed]
    assert runtime.repo.run["node_timings"][-1]["status"] == "failed"


def test_embedding_usage_survives_vector_database_failure():
    runtime, state = setup_runtime()
    observed = {"stage": "query_embedding", "usage": {"total_tokens": 3}}
    runtime._index = SimpleNamespace(usage_records=[])
    def reject(*args, **kwargs):
        runtime._index.usage_records.append(observed)
        raise MethodCodeError("database failure")
    runtime._index.retrieve = reject
    update = runtime._nodes({})["N5_retrieve"](state)
    assert update["usage"] == [observed]
    assert update["subtask_error"]["code"] == "INTERNAL_ERROR"


@pytest.mark.parametrize("broken", [None, "http", "index"])
def test_runtime_closes_database_before_unlock_even_when_cleanup_raises(broken):
    runtime = Runtime.__new__(Runtime)
    events = []
    def step(name):
        events.append(name)
        if name == broken:
            raise ValueError("cleanup failure")
    runtime._closed = False
    runtime._executor = SimpleNamespace(shutdown=lambda **kw: step("worker"))
    runtime._http_clients = [SimpleNamespace(close=lambda: step("http"))]
    runtime._profile_clients = {}
    runtime._index = SimpleNamespace(close=lambda: step("index"))
    runtime._release_instance_lock = lambda: step("unlock")
    if broken:
        with pytest.raises(ValueError, match="cleanup"):
            runtime.close()
    else:
        runtime.close()
    assert events == ["worker", "http", "index", "unlock"]
