"""Native DashScope embeddings. This module never reads environment credentials.

Batching, successful-batch caching and overlong-text splitting belong to callers.
The service's 8192-token limit cannot be measured with the local chunk tokenizer;
length errors are surfaced without truncation, fallback, or repeated paid requests.
"""

import hashlib
import json
import math
import random
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Literal

import httpx

from annual_report_agent.schemas.model import (
    EmbeddingBatchResult,
    EmbeddingProfile,
    ModelClientError,
)

Purpose = Literal["document", "query"]


def _validate_batch(texts: list[str], input_ids: list[str], purpose: str) -> None:
    if (not isinstance(texts, list) or not isinstance(input_ids, list)
            or not 1 <= len(texts) <= 10 or len(input_ids) != len(texts)
            or any(not isinstance(text, str) or not text.strip() for text in texts)
            or any(not isinstance(item, str) or not item.strip() for item in input_ids)
            or len(set(input_ids)) != len(input_ids) or purpose not in {"document", "query"}):
        raise ModelClientError("INVALID_INPUT", "Supply 1-10 nonempty texts and unique aligned IDs.")


def _delay(response: httpx.Response | None, attempt: int) -> float:
    delay = float(2 ** (attempt - 1)) + random.uniform(0, 0.25)
    if response is not None and (value := response.headers.get("Retry-After")):
        try:
            server_delay = float(value)
        except ValueError:
            try:
                parsed = parsedate_to_datetime(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                server_delay = (parsed - datetime.now(UTC)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                return delay
        if math.isfinite(server_delay):
            delay = max(delay, server_delay)
    return delay


class EmbeddingClient:
    """Live v3 client. At most 3 HTTP attempts; injected clients remain caller-owned."""

    def __init__(self, api_key: str, profile: EmbeddingProfile,
                 http_client: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ModelClientError("MISSING_CREDENTIALS", "DashScope API key is required.")
        if profile.mode != "live":
            raise ModelClientError("INVALID_PROFILE", "Use MockEmbeddingClient for mock mode.")
        self.profile = profile
        self._api_key = api_key
        self._http = http_client if http_client is not None else httpx.Client()
        self._owns_http = http_client is None
        self._sleep = sleep

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def encode_batch(self, texts: list[str], input_ids: list[str],
                     purpose: Purpose) -> EmbeddingBatchResult:
        _validate_batch(texts, input_ids, purpose)
        started = time.perf_counter()
        body = {"model": self.profile.model_id, "input": {"texts": texts}, "parameters": {
            "dimension": self.profile.dimension, "output_type": "dense", "text_type": purpose,
        }}
        for attempt in range(1, 4):
            response = None
            try:
                response = self._http.post(
                    self.profile.endpoint, json=body,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=httpx.Timeout(60, connect=10), follow_redirects=False,
                )
            except httpx.TransportError:
                error = ModelClientError("NETWORK_ERROR", "Embedding transport failed.", True)
            else:
                try:
                    payload = response.json()
                except (ValueError, UnicodeDecodeError):
                    payload = None
                code = payload.get("code") if isinstance(payload, dict) else None
                service_error = code not in (None, "", "Success", "success", 0)
                if response.is_success and not service_error:
                    return self._parse(payload, input_ids, response, attempt, started)
                code_text = str(code or f"HTTP_{response.status_code}")
                permanent = any(marker in code_text.lower() for marker in (
                    "budget", "notpurchased", "billoverdue", "balance", "arrears", "invalid",
                    "unauthorized", "forbidden",
                ))
                retryable = not permanent and response.status_code not in {400, 401, 403, 404} and (
                    response.status_code in {408, 429, 500, 502, 503, 504}
                    or code_text.startswith("Throttling.")
                )
                safe_code = code_text.replace(self._api_key, "[REDACTED]")[:160]
                error = ModelClientError(safe_code, "Embedding provider rejected the request.",
                                         retryable)
            if not error.retryable or attempt == 3:
                raise error from None
            self._sleep(_delay(response, attempt))
        raise AssertionError("unreachable")

    def _parse(self, payload, input_ids, response, attempt, started) -> EmbeddingBatchResult:
        try:
            if not isinstance(payload, dict):
                raise ValueError("body must be an object")
            entries = payload["output"]["embeddings"]
            if not isinstance(entries, list) or len(entries) != len(input_ids):
                raise ValueError("embedding count does not match inputs")
            ordered = [None] * len(input_ids)
            for entry in entries:
                index, values = entry["text_index"], entry["embedding"]
                if (type(index) is not int or not 0 <= index < len(input_ids)
                        or ordered[index] is not None):
                    raise ValueError("text_index must be unique and in range")
                if (not isinstance(values, list) or len(values) != self.profile.dimension
                        or any(type(value) not in (int, float) or not math.isfinite(value)
                               for value in values)):
                    raise ValueError("embedding must contain 1024 finite numbers")
                norm = math.hypot(*values)
                if not math.isfinite(norm) or norm == 0:
                    raise ValueError("embedding must have a finite nonzero norm")
                ordered[index] = [value / norm for value in values]
            usage = payload.get("usage")
            tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
            if tokens is not None and (type(tokens) is not int or tokens < 0):
                raise ValueError("usage must be a nonnegative integer")
            request_id = payload.get("request_id") or response.headers.get("x-request-id")
            if request_id is not None and not isinstance(request_id, str):
                raise ValueError("request_id must be a string")
            return EmbeddingBatchResult(
                ordered_vectors=ordered, input_ids=input_ids,
                model_profile_id=self.profile.profile_id, request_id=request_id,
                usage_total_tokens=tokens, attempts=attempt, latency=time.perf_counter() - started,
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ModelClientError("INVALID_RESPONSE", "Invalid embedding response structure.") from exc


class MockEmbeddingClient:
    """Offline lexical feature hashing for workflow tests, not a semantic v3 replica.

    SHA256 word/character features are deterministic across Python processes. These
    vectors must live in a mock index and must never be used as v3 quality evidence.
    """

    def __init__(self, profile: EmbeddingProfile | None = None):
        self.profile = profile if profile is not None else EmbeddingProfile(
            provider="mock", model_id="feature-hash-v1", endpoint="mock://local", mode="mock")
        if self.profile.mode != "mock":
            raise ModelClientError("INVALID_PROFILE", "Mock embeddings require a mock profile.")

    def encode_batch(self, texts: list[str], input_ids: list[str],
                     purpose: Purpose) -> EmbeddingBatchResult:
        _validate_batch(texts, input_ids, purpose)
        started = time.perf_counter()
        vectors = []
        for text in texts:
            text = text.lower()
            words = re.findall(r"[a-z0-9_]+", text)
            characters = re.findall(r"[^\s]", text)
            features = words + characters + [
                a + b for a, b in zip(characters, characters[1:], strict=False)
            ]
            values = [0.0] * self.profile.dimension
            for feature in features:
                digest = hashlib.sha256(feature.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.profile.dimension
                values[index] += 1.0
            norm = math.hypot(*values)
            vectors.append([value / norm for value in values])
        request_hash = hashlib.sha256(json.dumps([input_ids, texts, purpose],
                                                 ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        return EmbeddingBatchResult(
            ordered_vectors=vectors, input_ids=input_ids, model_profile_id=self.profile.profile_id,
            request_id=f"mock-{request_hash}", latency=time.perf_counter() - started,
        )
