"""Immutable index manifest types."""

from typing import Literal

from pydantic import BaseModel, Field


class IndexConfig(BaseModel):
    chunk_size: int = 500
    overlap: int = 80
    parser_version: str = "pdfplumber-v1"
    metadata_version: str = "1"
    tokenizer_version: str
    distance_metric: Literal["cosine"] = "cosine"
    embedding_profile_hash: str


class IndexManifest(BaseModel):
    index_version: str
    report_id: str
    chunk_ids: list[str]
    embedding_profile_hash: str
    tokenizer_version: str
    distance_metric: Literal["cosine"] = "cosine"
    build_status: Literal["building", "ready", "failed"]
    mode: Literal["mock", "live"]
    encoded_ids: list[str] = Field(default_factory=list)
    report_metadata: dict = Field(default_factory=dict)
