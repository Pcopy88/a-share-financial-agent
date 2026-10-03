import pytest

from annual_report_agent.ingestion.indexer import Candidate


def candidates():
    return [Candidate(chunk={"id": str(i), "text": text}, index_version="v")
            for i, text in enumerate(["低相关内容", "收入确认与营业收入"])]


class Backend:
    def __init__(self, scores=None):
        self.scores = scores or [.1, .9]
        self.calls = []

    def tokenizer(self, query, passage, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": list(range(len(query) + len(passage) + 4))}

    def compute_score(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return self.scores


def test_rank_is_finite_and_preserves_original_candidate():
    from annual_report_agent.retrieval.reranker import LocalReranker
    backend = Backend()
    original = candidates()
    result = LocalReranker(backend, max_length=100).rerank("收入", original)
    assert [c.chunk["id"] for c in result] == ["1", "0"]
    assert result[0].retrieval == "cross_encoder"
    assert original[0].score == 0
    assert backend.calls[0][1]["max_length"] == 100


def test_overlong_pair_never_silently_truncates_or_calls_model():
    from annual_report_agent.retrieval.reranker import LocalReranker, RerankerError
    backend = Backend()
    with pytest.raises(RerankerError, match="RERANK_INPUT_TOO_LONG"):
        LocalReranker(backend, max_length=8).rerank("收入", candidates())
    assert backend.calls == []


@pytest.mark.parametrize("scores", [[float("nan"), .5], [.5], [float("inf"), 1]])
def test_bad_scores_fail_explicitly(scores):
    from annual_report_agent.retrieval.reranker import LocalReranker, RerankerError
    with pytest.raises(RerankerError, match="RERANK_INVALID_OUTPUT"):
        LocalReranker(Backend(scores), max_length=100).rerank("收入", candidates())


def test_empty_input_does_not_load_or_call_backend():
    from annual_report_agent.retrieval.reranker import LocalReranker
    assert LocalReranker(Backend()).rerank("问题", []) == []
