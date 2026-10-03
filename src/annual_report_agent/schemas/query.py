"""Validated user scope and subtask contracts, never executable filters."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Route = Literal["semantic", "data", "hybrid"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetadataFilters(StrictModel):
    company: str | None = None
    doc_type: str | None = None
    inquiry_round: int | None = Field(None, ge=1)
    revision: str | None = None
    fiscal_year: int | None = Field(None, ge=1990, le=2100)
    question_number: str | None = None
    section_path: str | None = None


class AnalysisRequest(StrictModel):
    question: str = Field(min_length=1, max_length=8000)
    report_ids: list[str] = Field(min_length=1, max_length=20)
    hard_filters: MetadataFilters = Field(default_factory=MetadataFilters)
    task_type: Literal["question", "report"] = "question"
    model_profile: str = "default"

    @model_validator(mode="after")
    def validate_question(self) -> Self:
        if not self.question.strip() or len(set(self.report_ids)) != len(self.report_ids):
            raise ValueError("question must be nonempty; report_ids must be unique")
        return self


class SubTaskPlan(StrictModel):
    subtask_id: str = Field(min_length=1)
    source_text: str = Field(min_length=1)
    query_text: str = Field(min_length=1)
    rewritten_query: str = Field(min_length=1)
    report_ids: list[str] = Field(min_length=1)
    route: Route = "semantic"
    hard_filters: MetadataFilters = Field(default_factory=MetadataFilters)
    soft_hints: dict = Field(default_factory=dict)
    fiscal_years: list[int] = Field(default_factory=list)
    report_periods: list[dict] = Field(default_factory=list)
    metric_ids: list[str] = Field(default_factory=list)
    comparison: str | None = None
    scope: Literal["consolidated", "parent"] = "consolidated"
    needs_clarification: bool = False
    clarification_reason: str | None = None
    unsupported_reason: str | None = None
    section_id: str | None = None


class QueryPlan(StrictModel):
    task_type: Literal["question", "report"] = "question"
    original_question: str
    rewritten_question: str
    report_ids: list[str] = Field(min_length=1)
    hard_filters: MetadataFilters = Field(default_factory=MetadataFilters)
    route: Route = "semantic"
    subtasks: list[SubTaskPlan] = Field(default_factory=list)
    sections: list[dict] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_reason: str | None = None

    @model_validator(mode="after")
    def preserve_scope(self) -> Self:
        if not self.subtasks and not self.needs_clarification:
            raise ValueError("a non-clarification plan requires subtasks")
        limit = 3 if self.task_type == "question" else 12
        if len(self.subtasks) > limit:
            raise ValueError(f"subtask limit is {limit}")
        ids = [task.subtask_id for task in self.subtasks]
        if len(ids) != len(set(ids)):
            raise ValueError("subtask IDs must be unique")
        required = self.hard_filters.model_dump(exclude_none=True)
        for task in self.subtasks:
            if not set(task.report_ids).issubset(self.report_ids):
                raise ValueError("subtask cannot expand report scope")
            if self.task_type == "question" and task.source_text not in self.original_question:
                raise ValueError("source_text must occur in original_question")
            filters = task.hard_filters.model_dump(exclude_none=True)
            if any(filters.get(key) != value for key, value in required.items()):
                raise ValueError("subtask cannot drop or change explicit filter")
        return self
