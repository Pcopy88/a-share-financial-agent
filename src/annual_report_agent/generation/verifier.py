"""N9 deterministic lineage/numeric validation and non-generative aggregation.

Passing this module proves citation membership, source-location and numeric consistency.
It does not prove unrestricted semantic entailment, causation or complete document coverage.
"""

import re
from datetime import date
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, ConfigDict, Field

from annual_report_agent.finance.calculator import FORMULA_OPERANDS, calculate_metric
from annual_report_agent.finance.normalizer import normalize_amount
from annual_report_agent.generation.prompts import calculation_text, fact_text
from annual_report_agent.schemas.answer import AnswerDraft, AnswerResult, Evidence, SubTaskResult
from annual_report_agent.schemas.finance import Calculation, FinancialFact
from annual_report_agent.schemas.query import QueryPlan, SubTaskPlan


class VerificationContext(BaseModel):
    """Exact payload sent for this task, plus trusted page text and pinned versions.

    evidence metadata must contain any active report filters. Unknown metadata is not
    treated as a match. pages uses 1-based physical PDF pages, not printed labels.
    """

    model_config = ConfigDict(extra="forbid")
    task: SubTaskPlan
    report_ids: list[str]
    report_index_versions: dict[str, str]
    evidence: list[Evidence] = Field(default_factory=list)
    facts: list[FinancialFact] = Field(default_factory=list)
    calculations: list[Calculation] = Field(default_factory=list)
    pages: dict[str, dict[int, str]] = Field(default_factory=dict)


def _compact(text):
    return re.sub(r"\s+", "", text)


def _numeric_keys(text: str) -> set[tuple[str, Decimal | str]]:
    """Unit-aware numeral presence check, not a semantic relation checker."""
    keys = set()
    keys.update(("chinese_number", value) for value in re.findall(
        r"[零〇一二两三四五六七八九十百千万亿点负]+(?:亿元|万元|元|倍|个百分点|年|个月)", text
    ))
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", text)
    keys.update(("date", date_text) for date_text in dates)
    for date_text in dates:
        text = text.replace(date_text, " ")
    for match in re.finditer(
        r"(?<![\dA-Za-z.])([+-]?(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?)"
        r"\s*(亿元|万元|元|个百分点|百分点|%|％|倍)?", text
    ):
        value = Decimal(match[1].replace(",", "").replace("，", ""))
        unit = match[2]
        if unit in {"元", "万元", "亿元"}:
            keys.add(("currency", value * {"元": 1, "万元": 10000, "亿元": 100000000}[unit]))
        elif unit in {"%", "％"}:
            keys.add(("percent", value))
        elif unit in {"个百分点", "百分点"}:
            keys.add(("percentage_points", value))
        else:
            keys.add(("times" if unit == "倍" else "number", value))
    return keys


def _unique(items, label):
    values = {item.id: item for item in items}
    if len(values) != len(items):
        raise ValueError(f"duplicate {label} identifiers in sent context")
    return values


def verify(draft: AnswerDraft, context: VerificationContext) -> SubTaskResult:
    """Reject unsupported claims individually; retain other validated claims as partial."""
    task = context.task
    base = {"subtask_id": task.subtask_id}
    if task.needs_clarification:
        return SubTaskResult(**base, status="needs_clarification", reason=task.clarification_reason)
    if task.unsupported_reason:
        return SubTaskResult(**base, status="unsupported", reason=task.unsupported_reason)
    if draft.subtask_id != task.subtask_id or not set(task.report_ids).issubset(context.report_ids):
        return SubTaskResult(**base, status="failed", error_code="INVALID_CONTEXT",
                             reason="子任务与本次核验范围不一致。")
    try:
        evidence = _unique(context.evidence, "evidence")
        facts = _unique(context.facts, "fact")
        calculations = _unique(context.calculations, "calculation")
    except ValueError as error:
        return SubTaskResult(**base, status="failed", error_code="INVALID_CONTEXT", reason=str(error))

    def check_evidence(identifier):
        if identifier not in evidence:
            raise ValueError("citation was not sent to current subtask")
        item = evidence[identifier]
        if (item.report_id not in task.report_ids or item.report_id not in context.report_ids
                or context.report_index_versions.get(item.report_id) != item.index_version):
            raise ValueError("evidence crossed report or pinned index scope")
        if item.metadata.get("subtask_id", task.subtask_id) != task.subtask_id:
            raise ValueError("evidence belongs to another subtask")
        for key, value in task.hard_filters.model_dump(exclude_none=True).items():
            actual = item.section_path if key == "section_path" else item.metadata.get(key)
            if actual != value:
                raise ValueError("evidence metadata does not satisfy hard filters")
        if not item.quote.strip() or not item.page_refs:
            raise ValueError("evidence has no original quote or physical page")
        raw_pages = []
        page_numbers = set()
        for reference in item.page_refs:
            page = reference.get("pdf_page_number")
            if type(page) is not int or page < 1:
                raise ValueError("physical page must be a positive integer")
            if page in page_numbers:
                continue
            page_numbers.add(page)
            raw = context.pages.get(item.report_id, {}).get(page)
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("cited original PDF page is unavailable")
            raw_pages.append(raw)
        if _compact(item.quote) not in _compact("\n".join(raw_pages)):
            raise ValueError("sent evidence quote cannot be located in the cited original pages")
        return item

    def check_fact(identifier):
        if identifier not in facts:
            raise ValueError("fact was not sent to current subtask")
        item = facts[identifier]
        if (item.validation_status != "verified" or item.conflict_group_id or item.value is None
                or not item.value.is_finite() or not item.review_record or not item.source_evidence_ids
                or item.report_id not in task.report_ids or item.scope != task.scope):
            raise ValueError("fact is unverified, conflicted or outside task scope")
        if normalize_amount(item.raw_value, item.raw_unit) != item.value:
            raise ValueError("fact normalized amount disagrees with its source value/unit")
        quotes = []
        for source_id in item.source_evidence_ids:
            source = check_evidence(source_id)
            if source.report_id != item.report_id:
                raise ValueError("fact is bound to another document's evidence")
            if source.metadata.get("company") and source.metadata["company"] != item.company:
                raise ValueError("fact company disagrees with source metadata")
            quotes.append(source.quote)
        raw_value = _compact(item.raw_value or "").replace(",", "").replace("，", "")
        # Whitespace separates table cells/year labels; deleting it would merge 2026 120.
        source_text = " ".join(quotes).replace(",", "").replace("，", "")
        if not raw_value or not re.search(
            r"(?<![\d.])" + re.escape(raw_value) + r"(?!\d|\.\d)", source_text
        ):
            raise ValueError("fact source value is absent from the sent evidence")
        if task.report_periods:
            actual = (item.period_start.isoformat() if item.period_start else None,
                      item.period_end.isoformat())
            allowed = {(period.get("period_start") or period.get("start"),
                        period.get("period_end") or period.get("end"))
                       for period in task.report_periods}
            if any(any(key.startswith("previous_") for key in FORMULA_OPERANDS.get(metric, {}))
                   for metric in task.metric_ids):
                allowed |= {tuple(date.fromisoformat(value).replace(year=date.fromisoformat(value).year - 1)
                                  .isoformat() if value else None for value in period)
                            for period in tuple(allowed)}
            if actual not in allowed:
                raise ValueError("fact period does not match explicit task periods")
        return item

    def check_calculation(identifier):
        if identifier not in calculations:
            raise ValueError("calculation was not sent to current subtask")
        item = calculations[identifier]
        if item.formula_id not in FORMULA_OPERANDS or not item.result.is_finite():
            raise ValueError("unknown or nonfinite calculation")
        if task.metric_ids and item.formula_id not in task.metric_ids:
            raise ValueError("calculation does not answer the requested metric")
        names = list(FORMULA_OPERANDS[item.formula_id])
        if len(names) != len(item.operand_fact_ids):
            raise ValueError("wrong operand count")
        operands = {key: check_fact(fact_id)
                    for key, fact_id in zip(names, item.operand_fact_ids, strict=True)}
        recomputed = calculate_metric(item.formula_id, operands)
        if any(getattr(recomputed, key) != getattr(item, key) for key in (
            "id", "result", "unit", "formula_version", "period_label", "operand_fact_ids"
        )):
            raise ValueError("calculation disagrees with deterministic recomputation")
        return item

    supported, errors = [], []
    selected_evidence, selected_facts, selected_calculations = set(), set(), set()
    for claim in draft.claims:
        try:
            if claim.subtask_id != task.subtask_id or not claim.text.strip():
                raise ValueError("claim belongs to another subtask or is empty")
            if not (claim.evidence_ids or claim.fact_ids or claim.calculation_ids):
                raise ValueError("claim has no supporting citations")
            claim_evidence, claim_facts, claim_calculations = set(), set(), set()
            support_texts = []
            for identifier in claim.evidence_ids:
                support_texts.append(check_evidence(identifier).quote)
                claim_evidence.add(identifier)
            for identifier in claim.fact_ids:
                fact = check_fact(identifier)
                support_texts.append(fact_text(fact))
                claim_facts.add(identifier)
                claim_evidence.update(fact.source_evidence_ids)
            for identifier in claim.calculation_ids:
                calculation = check_calculation(identifier)
                support_texts.append(calculation_text(calculation))
                claim_calculations.add(identifier)
                claim_facts.update(calculation.operand_fact_ids)
                for fact_id in calculation.operand_fact_ids:
                    claim_evidence.update(facts[fact_id].source_evidence_ids)
            allowed_numbers = set().union(*(_numeric_keys(text) for text in support_texts))
            if not _numeric_keys(claim.text).issubset(allowed_numbers):
                raise ValueError("claim introduced a number or unit absent from its cited support")
            computed_metrics = set(task.metric_ids).intersection(FORMULA_OPERANDS)
            if task.route == "data" and computed_metrics and not claim_calculations:
                raise ValueError("computed metric requires a program calculation citation")
            if task.route == "data" and task.metric_ids and not computed_metrics and not any(
                facts[identifier].metric_code in task.metric_ids for identifier in claim_facts
            ):
                raise ValueError("direct metric requires a verified fact citation")
            supported.append(claim)
            selected_evidence.update(claim_evidence)
            selected_facts.update(claim_facts)
            selected_calculations.update(claim_calculations)
        except (ValueError, KeyError, TypeError, InvalidOperation) as error:
            errors.append(str(error))
    warnings = ["semantic_entailment_not_proven: 已核验引用来源和数值一致性，未自动证明语义蕴含或全文穷尽。"]
    answered_metrics = {calculations[identifier].formula_id for identifier in selected_calculations}
    answered_metrics.update(facts[identifier].metric_code for identifier in selected_facts)
    missing_metrics = set(task.metric_ids) - answered_metrics
    if missing_metrics:
        warnings.append("requested_metrics_incomplete: 尚未形成已核验结果的指标：" + "、".join(sorted(missing_metrics)))
    warnings.extend(draft.missing_information)
    warnings.extend(errors)
    if supported:
        status = "partial" if errors or draft.missing_information or missing_metrics else "answered"
    else:
        status = "failed" if errors else "insufficient_evidence"
    return SubTaskResult(
        **base, status=status, claims=supported,
        primary_evidence_ids=[item.id for item in context.evidence
                              if item.id in selected_evidence and item.role == "primary"],
        context_evidence_ids=[item.id for item in context.evidence
                              if item.id in selected_evidence and item.role == "context"],
        fact_ids=[item.id for item in context.facts if item.id in selected_facts],
        calculation_ids=[item.id for item in context.calculations if item.id in selected_calculations],
        warnings=warnings, error_code="VERIFICATION_FAILED" if errors else None,
        reason="部分结论未通过核验。" if errors else "当前输入证据不足。" if not supported else None,
    )


def aggregate(plan: QueryPlan, results: list[SubTaskResult], run_id: str = "", mode: str = "mock",
              global_error: str | None = None) -> AnswerResult:
    """Preserve plan order and gaps; never invent a summary or execute missing tasks."""
    expected = {task.subtask_id for task in plan.subtasks}
    mapped, warnings = {}, []
    for result in results:
        if result.subtask_id not in expected or result.subtask_id in mapped:
            global_error = global_error or "duplicate_or_unplanned_subtask_result"
            warnings.append("重复或未计划的子任务结果被拒绝。")
            continue
        if any(claim.subtask_id != result.subtask_id for claim in result.claims):
            global_error = global_error or "cross_subtask_claim"
            result = result.model_copy(update={"status": "failed", "claims": [],
                                               "error_code": "CROSS_SUBTASK_CLAIM"})
        if result.status == "answered" and not result.claims:
            result = result.model_copy(update={"status": "insufficient_evidence",
                                               "reason": "没有经核验的结论。"})
        mapped[result.subtask_id] = result
    ordered = []
    for task in plan.subtasks:
        result = mapped.get(task.subtask_id)
        if result is None:
            status = ("needs_clarification" if task.needs_clarification else
                      "unsupported" if task.unsupported_reason else "not_executed")
            result = SubTaskResult(subtask_id=task.subtask_id, status=status,
                                   reason=task.clarification_reason or task.unsupported_reason
                                   or global_error or "该子问题尚未执行，不能据此判断资料是否披露。")
        ordered.append(result)
    claims = [claim for result in ordered for claim in result.claims
              if result.status in {"answered", "partial"}]
    if global_error:
        status = "failed"
        warnings.append(str(global_error))
    elif plan.needs_clarification:
        status = "needs_clarification"
    elif ordered and all(result.status == "answered" for result in ordered):
        status = "answered"
    elif claims:
        status = "partial"
    else:
        statuses = {result.status for result in ordered}
        status = next((item for item in ("needs_clarification", "unsupported", "failed", "not_executed",
                                        "insufficient_evidence") if item in statuses), "not_executed")
    if mode == "mock":
        warnings.append("mock_mode: 模拟规划和摘录仅验证流程，不代表真实云端理解或回答质量。")
    warnings.extend(warning for result in ordered for warning in result.warnings)
    return AnswerResult(run_id=run_id, mode=mode, status=status, original_question=plan.original_question,
                        subtask_results=ordered, claims=claims, query_plan=plan.model_dump(mode="json"),
                        evidence_level="medium" if claims and mode == "live" else "low",
                        warnings=list(dict.fromkeys(warnings)))
