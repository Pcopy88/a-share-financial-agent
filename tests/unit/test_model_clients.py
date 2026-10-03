"""Cloud contracts verified with HTTP transports; no credentials or live network."""

import importlib
import json
import math

import httpx
import pytest
from pydantic import ValidationError


def modules():
    return (
        importlib.import_module("annual_report_agent.schemas.model"),
        importlib.import_module("annual_report_agent.retrieval.embedding_client"),
        importlib.import_module("annual_report_agent.generation.cloud_client"),
    )


def vector(first=3.0, second=4.0):
    return [first, second] + [0.0] * 1022


def embedding_response(vectors=None, **extra):
    vectors = vectors if vectors is not None else [vector()]
    return {
        "request_id": "embed-request-1",
        "output": {
            "embeddings": [
                {"text_index": i, "embedding": value} for i, value in enumerate(vectors)
            ]
        },
        "usage": {"total_tokens": 17},
    } | extra


def profile(**changes):
    schema, _, _ = modules()
    return schema.EmbeddingProfile(
        **({"endpoint": "https://workspace.example/api/v1/embeddings"} | changes)
    )


def embedding_client(handler, sleep=lambda _: None, **changes):
    _, embedding, _ = modules()
    return embedding.EmbeddingClient(
        api_key="test-key-only",
        profile=profile(**changes),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleep,
    )


def completion(content='{"answer":"原文披露"}', finish_reason="stop", **changes):
    return {
        "id": "chatcmpl-123",
        "model": "deepseek-flash",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
    } | changes


def generation_client(handler, sleep=lambda _: None):
    _, _, cloud = modules()
    return cloud.DeepSeekClient(
        api_key="test-key-only", base_url="https://api.deepseek.com",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=sleep,
    )


def test_generation_preserves_provider_fingerprint_and_requested_model():
    client = generation_client(lambda request: httpx.Response(200,
        json=completion(system_fingerprint="provider-build-123")))
    result = client.generate_json("JSON only", "test")
    assert result.system_fingerprint == "provider-build-123"
    assert result.requested_model == "deepseek-flash"


def test_profile_identifies_vector_space_and_separates_mock():
    live = profile()
    assert live.profile_id == profile().profile_id
    assert live.profile_id != profile(endpoint="https://other.example/embed").profile_id
    assert live.profile_id != profile(mode="mock").profile_id
    assert "mock" in profile(mode="mock").profile_id
    with pytest.raises(ValidationError):
        profile(dimension=768)


@pytest.mark.parametrize("purpose", ["document", "query"])
def test_embedding_uses_native_contract_and_restores_order(purpose):
    requests = []

    def handler(request):
        requests.append(request)
        data = embedding_response([vector(), vector(0, 2)])
        data["output"]["embeddings"].reverse()
        return httpx.Response(200, json=data)

    result = embedding_client(handler).encode_batch(["营业收入", "风险因素"], ["c1", "c2"], purpose)
    assert result.input_ids == ["c1", "c2"]
    assert result.ordered_vectors[0][:2] == pytest.approx([0.6, 0.8])
    assert result.ordered_vectors[1][:2] == [0.0, 1.0]
    assert len(result.ordered_vectors[0]) == 1024
    assert result.request_id == "embed-request-1"
    assert result.usage_total_tokens == 17
    body = json.loads(requests[0].content)
    assert body == {"model": "text-embedding-v3", "input": {"texts": ["营业收入", "风险因素"]},
                    "parameters": {"dimension": 1024, "output_type": "dense", "text_type": purpose}}
    assert requests[0].headers["Authorization"] == "Bearer test-key-only"


@pytest.mark.parametrize("texts,ids,purpose", [
    ([], [], "document"), (["x"] * 11, [str(i) for i in range(11)], "document"),
    (["x"], [], "document"), (["x", "y"], ["same", "same"], "document"),
    (["   "], ["a"], "document"), (["x"], [""], "document"),
    (["x"], ["a"], "other"),
])
def test_invalid_batch_is_rejected_before_network(texts, ids, purpose):
    schema, _, _ = modules()

    def forbidden(request):
        pytest.fail("invalid input must not reach HTTP")

    with pytest.raises(schema.ModelClientError) as error:
        embedding_client(forbidden).encode_batch(texts, ids, purpose)
    assert error.value.code == "INVALID_INPUT"
    assert not error.value.retryable


@pytest.mark.parametrize("malformation", [
    "duplicate_index", "out_of_range", "missing", "float_index", "bool_index", "wrong_dimension",
    "zero", "infinity", "nan", "string_number", "bool_number",
])
def test_corrupt_embedding_response_cannot_be_published(malformation):
    schema, _, _ = modules()
    payload = embedding_response([vector(), vector()])
    entries = payload["output"]["embeddings"]
    if malformation == "duplicate_index":
        entries[1]["text_index"] = 0
    elif malformation == "out_of_range":
        entries[1]["text_index"] = 2
    elif malformation == "missing":
        entries.pop()
    elif malformation == "float_index":
        entries[1]["text_index"] = 1.0
    elif malformation == "bool_index":
        entries[1]["text_index"] = True
    elif malformation == "wrong_dimension":
        entries[0]["embedding"] = [1.0]
    elif malformation == "zero":
        entries[0]["embedding"] = [0.0] * 1024
    else:
        entries[0]["embedding"][0] = {
            "infinity": float("inf"), "nan": float("nan"),
            "string_number": "3.0", "bool_number": True,
        }[malformation]
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(200, text=json.dumps(payload))

    with pytest.raises(schema.ModelClientError) as error:
        embedding_client(handler).encode_batch(["x", "y"], ["a", "b"], "document")
    assert error.value.code == "INVALID_RESPONSE"
    assert len(attempts) == 1


def test_absent_usage_stays_unknown():
    payload = embedding_response()
    payload.pop("usage")
    result = embedding_client(lambda _: httpx.Response(200, json=payload)).encode_batch(
        ["营业收入"], ["c1"], "query")
    assert result.usage_total_tokens is None


@pytest.mark.parametrize("status,code", [
    (429, "BudgetLimitExceeded"), (429, "Throttling.AllocationQuota.BudgetLimitExceeded"),
    (429, "CommodityNotPurchased"), (429, "PrepaidBillOverdue"),
    (429, "PostpaidBillOverdue"), (401, "InvalidApiKey"),
    (400, "InvalidParameter"), (200, "InvalidParameter"),
])
def test_permanent_provider_errors_are_not_retried(status, code):
    schema, _, _ = modules()
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(status, json={"code": code, "message": "test-key-only"})

    with pytest.raises(schema.ModelClientError) as error:
        embedding_client(handler).encode_batch(["x"], ["a"], "document")
    assert not error.value.retryable
    assert len(attempts) == 1
    assert "test-key-only" not in str(error.value)


def test_rate_limit_and_timeout_retry_twice_then_return_success():
    attempts, sleeps = [], []

    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"},
                                  json={"code": "Throttling.RateQuota"})
        if len(attempts) == 2:
            raise httpx.ReadTimeout("test timeout", request=request)
        return httpx.Response(200, json=embedding_response())

    result = embedding_client(handler, sleep=sleeps.append).encode_batch(["x"], ["a"], "query")
    assert len(attempts) == 3
    assert len(sleeps) == 2 and sleeps[0] >= 2
    assert result.attempts == 3


def test_transient_failure_exhausts_at_three_attempts():
    schema, _, _ = modules()
    attempts, sleeps = [], []

    def handler(request):
        attempts.append(request)
        return httpx.Response(503, text="temporary failure")

    with pytest.raises(schema.ModelClientError) as error:
        embedding_client(handler, sleep=sleeps.append).encode_batch(["x"], ["a"], "query")
    assert error.value.retryable
    assert len(attempts) == 3 and len(sleeps) == 2


def test_mock_is_deterministic_normalized_and_cannot_masquerade_as_live():
    schema, embedding, _ = modules()
    first = embedding.MockEmbeddingClient().encode_batch(["营业收入增长"], ["a"], "document")
    second = embedding.MockEmbeddingClient().encode_batch(["营业收入增长"], ["a"], "query")
    unrelated = embedding.MockEmbeddingClient().encode_batch(["董事会议事规则"], ["b"], "query")
    assert first.ordered_vectors == second.ordered_vectors
    assert first.ordered_vectors != unrelated.ordered_vectors
    assert len(first.ordered_vectors[0]) == 1024
    assert math.hypot(*first.ordered_vectors[0]) == pytest.approx(1.0)
    assert "mock" in first.model_profile_id
    assert first.usage_total_tokens is None
    with pytest.raises(schema.ModelClientError):
        embedding.MockEmbeddingClient(profile=profile())


def test_generation_sends_explicit_json_non_thinking_contract():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=completion())

    result = generation_client(handler).generate_json("只返回 json 对象。", "解释收入", max_tokens=77)
    assert result.data == {"answer": "原文披露"}
    assert result.usage["total_tokens"] == 19
    assert result.request_id == "chatcmpl-123"
    assert result.model == "deepseek-flash"
    assert result.attempts == 1 and result.latency >= 0
    assert str(requests[0].url) == "https://api.deepseek.com/chat/completions"
    assert json.loads(requests[0].content) == {
        "model": "deepseek-flash", "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"}, "max_tokens": 77,
        "messages": [{"role": "system", "content": "只返回 json 对象。"},
                     {"role": "user", "content": "解释收入"}],
        "stream": False,
    }


@pytest.mark.parametrize("content,finish_reason", [
    ("", "stop"), ("   ", "stop"), (None, "stop"), ("not json", "stop"),
    ("[]", "stop"), ('{"answer": "x"}', "length"),
    ('{"answer": "x"}', "content_filter"), ('{"answer": NaN}', "stop"),
])
def test_generation_rejects_empty_non_object_nonfinite_and_truncated_outputs(content, finish_reason):
    schema, _, _ = modules()
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(200, json=completion(content, finish_reason))

    with pytest.raises(schema.ModelClientError) as error:
        generation_client(handler).generate_json("Return json.", "Question")
    assert not error.value.retryable
    assert len(attempts) == 1


def test_generation_retries_server_failure_and_exposes_attempt_count():
    attempts, sleeps = [], []

    def handler(request):
        attempts.append(request)
        if len(attempts) < 3:
            return httpx.Response(502, json={"error": {"code": "server_error"}})
        return httpx.Response(200, json=completion())

    result = generation_client(handler, sleep=sleeps.append).generate_json("Return json.", "Q")
    assert result.attempts == 3 and len(sleeps) == 2


def test_generation_rejects_provider_error_even_with_http_200():
    schema, _, _ = modules()
    with pytest.raises(schema.ModelClientError):
        generation_client(lambda _: httpx.Response(200, json={"error": {
            "code": "insufficient_balance", "message": "test-key-only"}
        })).generate_json("Return json.", "Q")


def test_placeholder_endpoint_and_empty_credentials_fail_before_network():
    schema, embedding, cloud = modules()
    with pytest.raises((ValidationError, schema.ModelClientError)):
        embedding.EmbeddingClient(api_key="x", profile=profile(
            endpoint="https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/embeddings"))
    with pytest.raises(schema.ModelClientError):
        embedding.EmbeddingClient(api_key="", profile=profile())
    with pytest.raises(schema.ModelClientError):
        cloud.DeepSeekClient(api_key="", base_url="https://api.deepseek.com")


@pytest.mark.parametrize("choice", [None, [], "broken choice"])
def test_malformed_generation_choice_raises_client_error(choice):
    schema, _, _ = modules()
    with pytest.raises(schema.ModelClientError) as error:
        generation_client(lambda _: httpx.Response(200, json=completion(
            choices=[choice]))).generate_json("Return json.", "Q")
    assert error.value.code == "INVALID_RESPONSE"


def test_json_overflow_number_is_rejected_even_nested_in_array():
    schema, _, _ = modules()
    with pytest.raises(schema.ModelClientError) as error:
        generation_client(lambda _: httpx.Response(200, json=completion(
            '{"values": [1e999]}'))).generate_json("Return json.", "Q")
    assert error.value.code == "INVALID_RESPONSE"


def test_authentication_status_cannot_be_overridden_by_retryable_service_code():
    schema, _, _ = modules()
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(403, json={"code": "Throttling.RateQuota"})

    with pytest.raises(schema.ModelClientError) as error:
        embedding_client(handler).encode_batch(["x"], ["a"], "query")
    assert not error.value.retryable
    assert len(requests) == 1
