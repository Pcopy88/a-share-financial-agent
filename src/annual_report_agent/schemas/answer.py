"""Evidence and answer contracts with per-subtask attribution."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AnswerStatus = Literal["answered", "partial", "insufficient_evidence", "needs_clarification",
                       "unsupported", "not_executed", "failed"]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    report_id: str
    parent_id: str
    chunk_ids: list[str] = Field(default_factory=list)
    role: Literal["primary", "context"] = "primary"
    quote: str
    page_refs: list[dict]
    index_version: str
    section_path: str = ""
    metadata: dict = Field(default_factory=dict)


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subtask_id: str
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    calculation_ids: list[str] = Field(default_factory=list)


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subtask_id: str
    claims: list[Claim] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


class SubTaskResult(BaseModel):
    subtask_id: str
    status: AnswerStatus
    claims: list[Claim] = Field(default_factory=list)
    primary_evidence_ids: list[str] = Field(default_factory=list)
    context_evidence_ids: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    calculation_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error_code: str | None = None
    reason: str | None = None


class AnswerResult(BaseModel):
    run_id: str
    mode: Literal["mock", "live"]
    status: AnswerStatus
    original_question: str
    subtask_results: list[SubTaskResult]
    claims: list[Claim] = Field(default_factory=list)
    calculations: list[dict] = Field(default_factory=list)
    citations: list[Evidence] = Field(default_factory=list)
    query_plan: dict = Field(default_factory=dict)
    evidence_level: Literal["high", "medium", "low"] = "low"
    warnings: list[str] = Field(default_factory=list)
    usage: list[dict] = Field(default_factory=list)
