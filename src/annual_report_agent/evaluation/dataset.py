"""Read labels separately from model inputs; never ingest answer keys."""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from annual_report_agent.schemas.query import AnalysisRequest


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    question: str
    report_ids: list[str] = Field(min_length=1)
    route: str | None = None
    split: str
    filters: dict = Field(default_factory=dict)
    gold_evidence: list[dict] = Field(default_factory=list)
    gold_complete: bool = False
    expected_status: str | None = None
    expected_points: list[str] = Field(default_factory=list)

    def request(self, id_mapping: dict[str, str] | None = None) -> AnalysisRequest:
        mapping = id_mapping or {}
        return AnalysisRequest(question=self.question,
            report_ids=[mapping.get(identifier, identifier) for identifier in self.report_ids],
            hard_filters=self.filters)


def load_cases(path: str | Path, split: str | None = None) -> tuple[list[EvalCase], str]:
    raw = Path(path).read_bytes()
    cases = [EvalCase.model_validate(json.loads(line)) for line in raw.decode("utf-8-sig").splitlines()
             if line.strip()]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation case IDs must be unique")
    if any(case.split not in {"development", "test"} for case in cases):
        raise ValueError("every case must declare development/test split")
    return [case for case in cases if split is None or case.split == split], hashlib.sha256(raw).hexdigest()
