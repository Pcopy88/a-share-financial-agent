"""Protect mode isolation, scope restrictions and durable publication semantics."""

import pytest
from pydantic import ValidationError

from annual_report_agent.config import Settings
from annual_report_agent.schemas.query import AnalysisRequest, QueryPlan, SubTaskPlan
from annual_report_agent.storage.sqlite_repo import Repository


def task(**overrides):
    data = dict(subtask_id="q1", source_text="收入多少", query_text="收入多少",
                rewritten_query="收入多少", report_ids=["report-a"], route="semantic")
    return SubTaskPlan(**(data | overrides))


def plan(**overrides):
    data = dict(original_question="收入多少", rewritten_question="收入多少",
                report_ids=["report-a"], subtasks=[task()])
    return QueryPlan(**(data | overrides))


def test_missing_live_credentials_do_not_fall_back_to_mock(tmp_path):
    settings = Settings(_env_file=None, mode="live", data_dir=tmp_path,
                        deepseek_api_key="", dashscope_api_key="", embedding_url="")
    assert settings.mode == "live"
    assert set(settings.missing_cloud_config()) == {
        "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "DASHSCOPE_EMBEDDING_URL"
    }
    assert settings.runtime_dir != Settings(_env_file=None, mode="mock", data_dir=tmp_path).runtime_dir


def test_credentials_are_not_in_public_settings_or_repr():
    settings = Settings(_env_file=None, deepseek_api_key="TOP-SECRET",
                        dashscope_api_key="ALSO-SECRET")
    assert "TOP-SECRET" not in repr(settings)
    assert "SECRET" not in str(settings.public_dict())


def test_request_cannot_use_unknown_filter_or_empty_scope():
    with pytest.raises(ValidationError):
        AnalysisRequest(question="收入", report_ids=[], hard_filters={})
    with pytest.raises(ValidationError):
        AnalysisRequest(question="收入", report_ids=["a"], hard_filters={"sql": "1=1"})


def test_subtask_cannot_expand_report_scope():
    with pytest.raises(ValidationError, match="scope"):
        plan(subtasks=[task(report_ids=["foreign-report"])])


def test_subtasks_must_cover_unique_ids_and_preserve_source():
    with pytest.raises(ValidationError, match="unique"):
        plan(subtasks=[task(), task()])
    with pytest.raises(ValidationError, match="source_text"):
        plan(subtasks=[task(source_text="用户没有问的问题")])
    with pytest.raises(ValidationError):
        plan(subtasks=[task(subtask_id=f"q{i}") for i in range(4)])


def test_subtask_cannot_drop_explicit_filters():
    with pytest.raises(ValidationError, match="filter"):
        plan(hard_filters={"inquiry_round": 2}, subtasks=[task()])


def test_index_publication_is_atomic_and_requires_full_chunk_set(tmp_path):
    repo = Repository(tmp_path / "state.sqlite3")
    repo.save_build("v1", "r1", ["c1", "c2"], "profile", "mock")
    repo.mark_chunk_encoded("v1", "c1")
    with pytest.raises(ValueError, match="incomplete"):
        repo.publish_index("v1")
    assert repo.active_index("r1") is None
    repo.mark_chunk_encoded("v1", "c2")
    repo.publish_index("v1")
    repo.save_build("v2", "r1", ["c3", "c4"], "profile", "mock")
    repo.mark_chunk_encoded("v2", "c3")
    with pytest.raises(ValueError):
        repo.publish_index("v2")
    assert Repository(tmp_path / "state.sqlite3").active_index("r1")["index_version"] == "v1"


def test_embedding_cache_cannot_mix_purpose_or_profile(tmp_path):
    repo = Repository(tmp_path / "state.sqlite3")
    repo.cache_embedding("profile-a", "document", "abc", [1.0, 0.0])
    assert repo.cached_embedding("profile-a", "document", "abc") == [1.0, 0.0]
    assert repo.cached_embedding("profile-a", "query", "abc") is None
    assert repo.cached_embedding("profile-b", "document", "abc") is None


def test_restart_only_interrupts_unfinished_tasks(tmp_path):
    repo = Repository(tmp_path / "state.sqlite3")
    running = repo.create_task("analysis", "run-a")
    done = repo.create_task("analysis", "run-b")
    repo.update_task(running["task_id"], "running", 0.1)
    repo.update_task(done["task_id"], "running", 0.2)
    repo.update_task(done["task_id"], "succeeded", 1)
    repo.recover_interrupted()
    assert repo.get_task(running["task_id"])["status"] == "interrupted"
    assert repo.get_task(done["task_id"])["status"] == "succeeded"
