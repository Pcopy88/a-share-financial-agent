"""Financial amounts stay Decimal in Python and decimal strings in JSON."""

from datetime import date
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


class FinancialFact(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str
    metric_code: str
    value: Decimal | None
    raw_value: str | None
    unit: Literal["元"] = "元"
    raw_unit: str | None = None
    currency: str = "CNY"
    company: str
    period_start: date | None = None
    period_end: date
    scope: Literal["consolidated", "parent"]
    report_id: str
    source_id: str
    source_evidence_ids: list[str] = Field(default_factory=list)
    validation_status: Literal["pending", "verified", "conflict", "missing", "superseded"] = (
        "pending"
    )
    restatement_status: str | None = None
    conflict_group_id: str | None = None
    supersedes_fact_id: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    review_record: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("value", mode="before")
    @classmethod
    def reject_float_amounts(cls, value):
        if isinstance(value, (float, bool)):
            raise ValueError("financial amounts require decimal strings, Decimal, or integers")
        return value

    @field_serializer("value", when_used="json")
    def serialize_decimal(self, value):
        return None if value is None else format(value, "f")


class Calculation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    formula_id: str
    formula_version: str
    operand_fact_ids: list[str]
    result: Decimal
    unit: Literal["percent", "times", "percentage_points"]
    warnings: list[str] = Field(default_factory=list)
    period_label: str

    @field_serializer("result", when_used="json")
    def serialize_decimal(self, value):
        return format(value, "f")
