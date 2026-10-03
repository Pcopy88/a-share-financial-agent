"""Public contracts for cloud model clients and explicitly simulated embeddings."""

import hashlib
import json
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelClientError(RuntimeError):
    """Safe application error. Provider response bodies and API keys are not logged."""

    def __init__(self, code: str, message: str, retryable: bool = False):
        self.code = code
        self.message = message
        self.retryable = retryable
        super().__init__(f"{code}: {message}")


class EmbeddingProfile(BaseModel):
    """Identity of a vector space; purpose is tracked separately by the cache."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    provider: str = "dashscope"
    model_id: str = "text-embedding-v3"
    dimension: Literal[1024] = 1024
    endpoint: str
    mode: Literal["live", "mock"] = "live"
    normalization: Literal["l2"] = "l2"
    distance_metric: Literal["cosine"] = "cosine"

    @model_validator(mode="after")
    def validate_live_profile(self):
        if self.mode == "live":
            parsed = urlsplit(self.endpoint)
            if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                    or parsed.password or parsed.query or parsed.fragment
                    or any(char in self.endpoint for char in "{}<>")):
                raise ValueError("endpoint must be a complete HTTPS URL without placeholders")
            if self.provider != "dashscope" or self.model_id != "text-embedding-v3":
                raise ValueError("live embedding currently supports DashScope text-embedding-v3")
        return self

    @property
    def profile_id(self) -> str:
        canonical = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"{self.mode}-embedding-{digest}"


class EmbeddingBatchResult(BaseModel):
    """Validated L2 vectors aligned with input_ids, never guessed billing tokens."""

    model_config = ConfigDict(extra="forbid")
    ordered_vectors: list[list[float]]
    input_ids: list[str]
    model_profile_id: str
    request_id: str | None = None
    usage_total_tokens: int | None = Field(default=None, ge=0)
    attempts: int = Field(default=1, ge=1, le=3)
    latency: float = Field(default=0.0, ge=0)


class GenerationResult(BaseModel):
    """Parsed JSON object; downstream Pydantic schemas must validate its meaning.

    latency is wall-clock seconds including backoff, attempts includes the first call,
    and usage is the final successful provider payload, not an estimate or retry sum.
    Missing usage/request_id remains None. model is the response model identity.
    """

    model_config = ConfigDict(extra="forbid")
    data: dict[str, Any]
    usage: dict[str, Any] | None = None
    request_id: str | None = None
    model: str
    requested_model: str | None = None
    system_fingerprint: str | None = None
    latency: float = Field(ge=0)
    attempts: int = Field(ge=1, le=3)
