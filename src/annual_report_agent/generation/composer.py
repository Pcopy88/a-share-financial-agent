"""N8 drafts from the exact current-subtask payload, with explicit mock extracts."""

from pydantic import ValidationError

from annual_report_agent.finance.calculator import FORMULA_OPERANDS
from annual_report_agent.generation.cloud_client import DeepSeekClient
from annual_report_agent.generation.prompts import (
    COMPOSER_SYSTEM,
    calculation_text,
    dump_json,
    fact_text,
    record_usage,
)
from annual_report_agent.schemas.answer import AnswerDraft, Claim, Evidence
from annual_report_agent.schemas.finance import Calculation, FinancialFact
from annual_report_agent.schemas.model import ModelClientError
from annual_report_agent.schemas.query import SubTaskPlan


def compose(original_question: str, task: SubTaskPlan, evidence: list[Evidence],
            facts: list[FinancialFact], calculations: list[Calculation],
            client: DeepSeekClient | None = None, mode: str = "mock", *,
            usage_sink: list[dict] | None = None) -> AnswerDraft:
    """Caller supplies only actually sent evidence; verifier rechecks every citation.

    Mock mode extracts supplied text and formats deterministic calculations. It is not
    an answer-quality baseline. Live drafts are not trusted until verify() succeeds.
    """
    if mode not in {"live", "mock"}:
        raise ModelClientError("INVALID_MODE", "Unknown composition mode.")
    evidence = [Evidence.model_validate(item) for item in evidence]
    facts = [FinancialFact.model_validate(item) for item in facts]
    calculations = [Calculation.model_validate(item) for item in calculations]
    if mode == "mock":
        claims = []
        if task.route in {"semantic", "hybrid"}:
            claims.extend(Claim(subtask_id=task.subtask_id, text="原文摘录：" + item.quote,
                                evidence_ids=[item.id]) for item in evidence if item.role == "primary")
        for calculation in calculations:
            claims.append(Claim(subtask_id=task.subtask_id, text=calculation_text(calculation),
                                calculation_ids=[calculation.id], fact_ids=calculation.operand_fact_ids))
        computed_metrics = set(task.metric_ids).intersection(FORMULA_OPERANDS)
        direct_metrics = set(task.metric_ids).difference(FORMULA_OPERANDS)
        if task.route in {"data", "hybrid"} and not computed_metrics:
            claims.extend(Claim(subtask_id=task.subtask_id, text=fact_text(fact), fact_ids=[fact.id],
                                evidence_ids=fact.source_evidence_ids)
                          for fact in facts if fact.validation_status == "verified" and fact.value is not None
                          and (not direct_metrics or fact.metric_code in direct_metrics))
        gaps = []
        if task.route in {"semantic", "hybrid"} and not any(e.role == "primary" for e in evidence):
            gaps.append("当前子问题缺少可引用的原文证据。")
        if task.route in {"data", "hybrid"} and not calculations and (computed_metrics or not claims):
            gaps.append("当前子问题缺少已核验事实或所需程序计算结果。")
        return AnswerDraft(subtask_id=task.subtask_id, claims=claims, missing_information=gaps)
    if client is None:
        raise ModelClientError("MISSING_CLIENT", "Live composition requires a cloud client.")
    payload = {"original_question": original_question, "current_subtask": task.model_dump(mode="json"),
               "untrusted_evidence": [item.model_dump(mode="json") for item in evidence],
               "facts": [item.model_dump(mode="json") for item in facts],
               "calculations": [item.model_dump(mode="json") for item in calculations],
               "schema": AnswerDraft.model_json_schema()}
    result = client.generate_json(COMPOSER_SYSTEM, dump_json(payload), max_tokens=4096)
    record_usage(usage_sink, result, "N8", task.subtask_id)
    try:
        answer = AnswerDraft.model_validate(result.data)
        if answer.subtask_id != task.subtask_id or any(
            claim.subtask_id != task.subtask_id for claim in answer.claims
        ):
            raise ValueError("draft crossed subtask boundary")
        return answer
    except (ValidationError, ValueError) as error:
        raise ModelClientError("INVALID_DRAFT", "Draft failed schema/subtask validation.") from error
