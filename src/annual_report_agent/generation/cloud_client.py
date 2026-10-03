"""DeepSeek JSON adapter; business schema validation is the caller's responsibility."""

import json
import math
import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import httpx

from annual_report_agent.schemas.model import GenerationResult, ModelClientError


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


def _reject_constant(value: str):
    raise ValueError("JSON numbers must be finite")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON numbers must be finite")
    return parsed


class DeepSeekClient:
    """At most two transport retries; never repairs invalid output by another call.

    base_url is an HTTPS API root, e.g. https://api.deepseek.com or a root ending
    in /v1. Injected clients are caller-owned; call close() for an owned client.
    No environment reads, provider fallback, or SDK retries are performed here.
    """

    def __init__(self, api_key: str, base_url: str, model: str = "deepseek-flash",
                 http_client: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ModelClientError("MISSING_CREDENTIALS", "DeepSeek API key is required.")
        parsed = urlsplit(base_url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or any(char in base_url for char in "{}<>")):
            raise ModelClientError("INVALID_CONFIG", "base_url must be a complete HTTPS API root.")
        if not isinstance(model, str) or not model.strip():
            raise ModelClientError("INVALID_CONFIG", "A model identifier is required.")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._http = http_client if http_client is not None else httpx.Client()
        self._owns_http = http_client is None
        self._sleep = sleep

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def generate_json(self, system: str, user: str, max_tokens: int = 2048) -> GenerationResult:
        if (not isinstance(system, str) or not system.strip()
                or not isinstance(user, str) or not user.strip()
                or type(max_tokens) is not int or max_tokens <= 0):
            raise ModelClientError("INVALID_INPUT", "Nonempty prompts and positive max_tokens required.")
        if "json" not in (system + user).lower():
            system += "\nReturn one valid JSON object."
        body = {
            "model": self.model, "messages": [
                {"role": "system", "content": system}, {"role": "user", "content": user},
            ], "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"},
            "max_tokens": max_tokens, "stream": False,
        }
        started = time.perf_counter()
        for attempt in range(1, 4):
            response = None
            try:
                response = self._http.post(
                    self.base_url + "/chat/completions", json=body,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=httpx.Timeout(90, connect=10), follow_redirects=False,
                )
            except httpx.TransportError:
                error = ModelClientError("NETWORK_ERROR", "Generation transport failed.", True)
            else:
                try:
                    payload = response.json()
                except (ValueError, UnicodeDecodeError):
                    payload = None
                failure = payload.get("error") if isinstance(payload, dict) else None
                code = payload.get("code") if isinstance(payload, dict) else None
                if response.is_success and not failure and code in (None, "", "Success", 0):
                    return self._parse(payload, response, attempt, started)
                if isinstance(failure, dict):
                    code = failure.get("code") or failure.get("type") or code
                code_text = str(code or f"HTTP_{response.status_code}")
                permanent = any(marker in code_text.lower() for marker in (
                    "budget", "balance", "quota_exceeded", "invalid", "authentication",
                    "permission", "notpurchased", "billoverdue",
                ))
                retryable = not permanent and response.status_code in {408, 429, 500, 502, 503, 504}
                safe_code = code_text.replace(self._api_key, "[REDACTED]")[:160]
                error = ModelClientError(safe_code, "Generation provider rejected the request.",
                                         retryable)
            if not error.retryable or attempt == 3:
                raise error from None
            self._sleep(_delay(response, attempt))
        raise AssertionError("unreachable")

    def _parse(self, payload, response, attempt, started) -> GenerationResult:
        try:
            if not isinstance(payload, dict):
                raise ValueError("body must be an object")
            choices = payload["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("one choice is required")
            choice = choices[0]
            if not isinstance(choice, dict):
                raise ValueError("choice must be an object")
            if choice.get("finish_reason") != "stop":
                raise ModelClientError("INCOMPLETE_OUTPUT", "Generation did not finish normally.")
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty content")
            data = json.loads(content, parse_constant=_reject_constant, parse_float=_finite_float)
            if not isinstance(data, dict):
                raise ValueError("JSON output must be an object")
            model = payload["model"]
            if not isinstance(model, str) or not model.strip():
                raise ValueError("response model identity is missing")
            usage = payload.get("usage")
            if usage is not None and not isinstance(usage, dict):
                raise ValueError("usage must be an object")
            request_id = response.headers.get("x-request-id") or payload.get("id")
            if request_id is not None and not isinstance(request_id, str):
                raise ValueError("request_id must be a string")
            fingerprint = payload.get("system_fingerprint")
            if fingerprint is not None and not isinstance(fingerprint, str):
                raise ValueError("system_fingerprint must be a string when provided")
            return GenerationResult(data=data, usage=usage, request_id=request_id, model=model,
                                    requested_model=self.model, system_fingerprint=fingerprint,
                                    attempts=attempt, latency=time.perf_counter() - started)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ModelClientError("INVALID_RESPONSE", "Generation returned invalid JSON output.") from exc
