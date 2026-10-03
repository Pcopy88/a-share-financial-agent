"""Use real SQLite and persistent Chroma; only the paid encoder is simulated."""

from importlib.util import find_spec

import pytest


def components():
    assert find_spec("annual_report_agent.ingestion.indexer"), "indexer missing"
    from annual_report_agent.ingestion.indexer import KnowledgeIndex
    from annual_report_agent.retrieval.embedding_client import MockEmbeddingClient
    from annual_report_agent.schemas.query import SubTaskPlan
    from annual_report_agent.storage.sqlite_repo import Repository
    return KnowledgeIndex, MockEmbeddingClient, SubTaskPlan, Repository


def sample(report_id, company, texts):
    report = dict(id=report_id, company=company, sha256="hash-" + report_id,
                  doc_type="semiannual_report", fiscal_year=2026, inquiry_round=None)
    parents, chunks = [], []
    for i, text in enumerate(texts):
        parents.append(dict(id=f"{report_id}-p{i}", report_id=report_id,
                            text=text + "\n这是必要上下文。", section_path="经营分析",
                            question_number=None, page_refs=[dict(pdf_page_number=i+1)], metadata={}))
        chunks.append(dict(id=f"{report_id}-c{i}", report_id=report_id, parent_id=parents[-1]["id"],
                           text=text, section_path="经营分析", question_number=None,
                           page_refs=[dict(pdf_page_number=i+1)], tokenizer_version="fixture-v1",
                           chunk_token_count=len(text), metadata={}))
    return report, parents, chunks


def query(Task, report_ids, **kwargs):
    return Task(subtask_id="q1", source_text="收入", query_text="收入", rewritten_query="营业收入",
                report_ids=report_ids, **kwargs)


def test_scope_is_applied_before_dense_and_bm25_ranking(tmp_path):
    Index, Mock, Task, Repo = components()
    repo = Repo(tmp_path / "app.db")
    index = Index(repo, tmp_path / "vectors", Mock())
    index.build(*sample("a", "甲公司", ["营业收入增长", "半导体产品研发"]))
    index.build(*sample("b", "乙公司", ["营业收入增长营业收入营业收入"]))
    for mode in ("dense", "bm25", "hybrid"):
        found = index.retrieve(query(Task, ["a", "b"], hard_filters={"company": "甲公司"}), mode=mode)
        assert found and all(c.chunk["report_id"] == "a" for c in found)
    assert index.retrieve(query(Task, ["a"], hard_filters={"inquiry_round": 2})) == []


def test_runtime_status_and_file_location_do_not_create_new_source_index(tmp_path):
    Index, Mock, Task, Repo = components()
    repo = Repo(tmp_path / "app.db")
    index = Index(repo, tmp_path / "vectors", Mock())
    report, parents, chunks = sample("a", "甲", ["收入增长"])
    first = index.build(report | {"status": "parsed", "path": "old.pdf"}, parents, chunks)
    second = index.build(report | {"status": "ready", "path": "new.pdf",
                                   "capabilities": {"text_search": True}}, parents, chunks)
    assert first["index_version"] == second["index_version"]
    revised = index.build(report | {"company": "已更正公司"}, parents, chunks)
    assert revised["index_version"] != first["index_version"]


def test_index_survives_restart_and_expands_only_real_selected_pages(tmp_path):
    Index, Mock, Task, Repo = components()
    repo = Repo(tmp_path / "app.db")
    index = Index(repo, tmp_path / "vectors", Mock())
    index.build(*sample("a", "甲", ["营业收入增长"]))
    fresh = Index(Repo(tmp_path / "app.db"), tmp_path / "vectors", Mock())
    candidates = fresh.retrieve(query(Task, ["a"]))
    from annual_report_agent.retrieval.context import expand_context
    evidence = expand_context(candidates, repo, budget_tokens=1000)
    assert evidence[0].role == "primary"
    assert any("必要上下文" in e.quote for e in evidence)
    assert all(p["pdf_page_number"] == 1 for e in evidence for p in e.page_refs)


def test_failed_rebuild_preserves_old_index_and_resumes_cached_batches(tmp_path):
    Index, Mock, Task, Repo = components()

    class FailSecondBatch:
        def __init__(self):
            self.delegate = Mock()
            self.profile = self.delegate.profile
            self.fail = False
            self.calls = 0

        def encode_batch(self, texts, input_ids, purpose):
            self.calls += 1
            if self.fail and self.calls == 3:
                raise RuntimeError("upstream offline")
            return self.delegate.encode_batch(texts, input_ids, purpose)

    client, repo = FailSecondBatch(), Repo(tmp_path / "app.db")
    index = Index(repo, tmp_path / "vectors", client)
    old = index.build(*sample("a", "甲", ["旧版收入"]))
    new = sample("a", "甲", [f"新版收入段落{i}" for i in range(12)])
    client.fail = True
    with pytest.raises(RuntimeError, match="offline"):
        index.build(*new)
    assert repo.active_index("a")["index_version"] == old["index_version"]
    client.fail = False
    finished = index.build(*new)
    assert finished["index_version"] != old["index_version"]
    assert client.calls == 4  # 1 old + 1 successful batch + 1 failed + only 1 resumed batch
    assert repo.active_index("a")["index_version"] == finished["index_version"]


def test_parent_mismatch_cannot_be_published(tmp_path):
    Index, Mock, _, Repo = components()
    index = Index(Repo(tmp_path / "app.db"), tmp_path / "vectors", Mock())
    report, parents, chunks = sample("a", "甲", ["收入"])
    chunks[0]["parent_id"] = "foreign-parent"
    with pytest.raises(ValueError, match="parent"):
        index.build(report, parents, chunks)


@pytest.mark.parametrize("damage", ["collection", "encoded_row", "hnsw"])
def test_damaged_ready_index_recovers_from_cache_and_reuses_recovery(tmp_path, monkeypatch, damage):
    from chromadb.api.models.Collection import Collection
    from chromadb.errors import InternalError

    Index, Mock, Task, Repo = components()
    encoder, repo = Mock(), Repo(tmp_path / "app.db")
    index = Index(repo, tmp_path / "vectors", encoder)
    source = sample("a", "甲", ["收入增长", "研发费用"])
    old = index.build(*source)
    old_version = old["index_version"]
    if damage == "collection":
        index.client.delete_collection(old_version)
    elif damage == "encoded_row":
        index.client.get_collection(old_version).delete(ids=["a-c1"])
    else:
        original_query = Collection.query

        def broken_query(self, *args, **kwargs):
            if self.name == old_version:
                raise InternalError("Error loading hnsw index")
            return original_query(self, *args, **kwargs)

        monkeypatch.setattr(Collection, "query", broken_query)

    def no_paid_call(*args, **kwargs):
        pytest.fail("recovery/health check must use cached document vectors")

    monkeypatch.setattr(encoder, "encode_batch", no_paid_call)
    recovered = index.build(*source)
    assert recovered["index_version"] != old_version
    assert recovered["source_identity"] == old_version
    assert recovered["recovery_of"] == old_version
    assert repo.active_index("a")["index_version"] == recovered["index_version"]
    assert set(recovered["encoded_ids"]) == {"a-c0", "a-c1"}
    assert index.build(*source)["index_version"] == recovered["index_version"]
    if damage != "collection":
        assert index.client.get_collection(old_version).name == old_version
    assert len(index.usage_records) == 1


def test_interrupted_recovery_preserves_active_and_resumes_only_remaining_vectors(tmp_path, monkeypatch):
    from chromadb.api.models.Collection import Collection
    from chromadb.errors import InternalError

    Index, Mock, _, Repo = components()
    encoder, repo = Mock(), Repo(tmp_path / "app.db")
    index = Index(repo, tmp_path / "vectors", encoder)
    source = sample("a", "甲", [f"营业收入段落{i}" for i in range(12)])
    old = index.build(*source)
    original_query, original_upsert = Collection.query, Collection.upsert
    writes, fail = [], True

    def broken_query(self, *args, **kwargs):
        if self.name == old["index_version"]:
            raise InternalError("Error loading hnsw index")
        return original_query(self, *args, **kwargs)

    def interrupted_upsert(self, *args, **kwargs):
        if self.name != old["index_version"]:
            writes.append((self.name, list(kwargs["ids"])))
            if fail and len(writes) == 2:
                raise InternalError("disk temporarily unavailable")
        return original_upsert(self, *args, **kwargs)

    monkeypatch.setattr(Collection, "query", broken_query)
    monkeypatch.setattr(Collection, "upsert", interrupted_upsert)
    monkeypatch.setattr(encoder, "encode_batch", lambda *a, **k: pytest.fail("cached vectors required"))
    with pytest.raises(InternalError, match="disk temporarily"):
        index.build(*source)
    assert repo.active_index("a")["index_version"] == old["index_version"]
    recovery_version = writes[0][0]
    assert len(repo.build(recovery_version)["encoded_ids"]) == 10
    fail = False
    recovered = index.build(*source)
    assert recovered["index_version"] == recovery_version
    assert len(writes) == 3 and len(writes[-1][1]) == 2
    assert len(index.usage_records) == 2


def test_recovery_query_validation_failure_does_not_publish(tmp_path, monkeypatch):
    from chromadb.api.models.Collection import Collection
    from chromadb.errors import InternalError

    Index, Mock, _, Repo = components()
    repo = Repo(tmp_path / "app.db")
    index = Index(repo, tmp_path / "vectors", Mock())
    source = sample("a", "甲", ["营业收入"])
    old = index.build(*source)

    def broken_query(self, *args, **kwargs):
        raise InternalError("Error loading hnsw index")

    monkeypatch.setattr(Collection, "query", broken_query)
    with pytest.raises(InternalError, match="hnsw"):
        index.build(*source)
    assert repo.active_index("a")["index_version"] == old["index_version"]


def test_index_close_releases_public_chroma_client(tmp_path, monkeypatch):
    Index, Mock, _, Repo = components()
    index = Index(Repo(tmp_path / "app.db"), tmp_path / "vectors", Mock())
    calls = []
    real_close = index.client.close

    def close():
        calls.append(True)
        real_close()

    monkeypatch.setattr(index.client, "close", close)
    index.close()
    assert calls == [True]
