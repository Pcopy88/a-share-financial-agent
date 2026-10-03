"""PDF identity and lossless evidence records; metadata is not verification."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DocumentType = Literal[
    "annual_report", "semiannual_report", "prospectus", "inquiry_letter",
    "inquiry_response", "other",
]


class Report(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    id: str
    sha256: str
    path: str
    filename: str
    company: str | None = None
    stock_code: str | None = None
    doc_type: DocumentType = "other"
    fiscal_year: int | None = None
    report_periods: list[dict[str, Any]] = Field(default_factory=list)
    disclosure_date: str | None = None
    inquiry_round: int | None = None
    revision: str | None = None
    page_count: int = Field(default=0, ge=0)
    status: str = "uploaded"
    capabilities: dict[str, Any] = Field(default_factory=dict)
    source_url: str | None = None

    @field_validator("report_periods", mode="before")
    @classmethod
    def canonical_periods(cls, value):
        result = []
        for item in value:
            period = dict(item)
            for old, canonical in (("start", "period_start"), ("end", "period_end")):
                if old in period:
                    if canonical in period and period[canonical] != period[old]:
                        raise ValueError("conflicting period aliases")
                    period[canonical] = period.pop(old)
            result.append(period)
        return result


class PageRef(BaseModel):
    pdf_page_number: int = Field(ge=1)
    printed_page_label: str | None = None
    bbox: tuple[float, float, float, float] | None = None


class Page(BaseModel):
    report_id: str
    pdf_page_number: int = Field(ge=1)
    printed_page_label: str | None = None
    raw_text: str
    parse_status: Literal["parsed", "needs_ocr", "failed"]
    error: str | None = None


class ParentBlock(BaseModel):
    id: str
    report_id: str
    section_path: str = ""
    question_number: str | None = None
    report_periods: list[dict[str, Any]] = Field(default_factory=list)
    kind: Literal["text", "table"] = "text"
    text: str
    page_refs: list[PageRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    index_version: str | None = None


class Chunk(BaseModel):
    id: str
    report_id: str
    parent_id: str
    text: str
    page_refs: list[PageRef]
    section_path: str = ""
    question_number: str | None = None
    report_periods: list[dict[str, Any]] = Field(default_factory=list)
    chunk_token_count: int = Field(ge=0)
    tokenizer_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    index_version: str | None = None
    parser_version: str = "pdfplumber-structure-v1"


class ParsedReport(BaseModel):
    report: Report
    pages: list[Page]
    parents: list[ParentBlock]
    chunks: list[Chunk]
    warnings: list[str] = Field(default_factory=list)
