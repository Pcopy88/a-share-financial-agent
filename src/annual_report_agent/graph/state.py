"""Serializable graph state. PDFs and vectors live in separate durable stores."""

from typing import TypedDict

NODE_NAMES = ("N1_acquire", "N2_parse", "N3_index", "N4_route", "N5_retrieve",
              "N6_rerank", "N7_calculate", "N8_compose", "N9_verify")


class GraphState(TypedDict, total=False):
    run_id: str
    task_id: str
    kind: str
    report_id: str
    request: dict
    manifests: list[dict]
    plan: dict
    cursor: int
    candidates: list[dict]
    evidence: list[dict]
    facts: list[dict]
    calculations: list[dict]
    draft: dict
    results: list[dict]
    all_evidence: list[dict]
    all_calculations: list[dict]
    answer: dict
    subtask_error: dict | None
    global_error: dict | None
    warnings: list[str]
    usage: list[dict]
    calls_used: int
    tokens_used: int
