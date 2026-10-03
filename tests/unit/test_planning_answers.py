"""Planning scope, evidence lineage and deterministic answer verification."""

import importlib
from decimal import Decimal

import pytest

from annual_report_agent.schemas.answer import AnswerDraft, Claim, Evidence, SubTaskResult
from annual_report_agent.schemas.finance import FinancialFact
from annual_report_agent.schemas.model import GenerationResult, ModelClientError
from annual_report_agent.schemas.query import AnalysisRequest, QueryPlan, SubTaskPlan


def modules():
    return tuple(importlib.import_module("annual_report_agent." + name) for name in (
        "graph.planner", "generation.composer", "generation.verifier"))


def reports():
    return [{"id": "r1", "company": "甲公司", "doc_type": "annual_report", "fiscal_year": 2026,
             "revision": "v1", "report_periods": []}]


def task(**changes):
    return SubTaskPlan(**({"subtask_id": "q1", "source_text": "收入如何确认？",
                          "query_text": "收入如何确认？", "rewritten_query": "收入如何确认？",
                          "report_ids": ["r1"]} | changes))


def evidence(**changes):
    return Evidence(**({"id": "e1", "report_id": "r1", "parent_id": "p1", "chunk_ids": ["c1"],
                        "quote": "产品交付并经客户验收后确认收入。", "page_refs": [{"pdf_page_number": 2}],
                        "index_version": "i1", "metadata": {"company": "甲公司"}} | changes))


def context(**changes):
    _, _, verifier = modules()
    return verifier.VerificationContext(**({
        "task": task(), "report_ids": ["r1"], "report_index_versions": {"r1": "i1"},
        "evidence": [evidence()], "facts": [], "calculations": [],
        "pages": {"r1": {2: "产品交付并经客户验收后确认收入。"}},
    } | changes))


def draft(text="产品交付并经客户验收后确认收入。", **changes):
    return AnswerDraft(subtask_id="q1", claims=[Claim(**({
        "subtask_id": "q1", "text": text, "evidence_ids": ["e1"],
    } | changes))])


class JsonClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def generate_json(self, system, user, max_tokens=2048):
        self.calls.append((system, user, max_tokens))
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return GenerationResult(data=response, model="fixture-model", request_id="req-1",
                                usage={"total_tokens": 20}, latency=0.1, attempts=1)


def plan_payload(request, **changes):
    subtask = task(source_text=request.question, query_text=request.question,
                   rewritten_query=request.question, hard_filters=request.hard_filters)
    return {"original_question": request.question, "rewritten_question": request.question,
            "report_ids": request.report_ids, "hard_filters": request.hard_filters.model_dump(),
            "subtasks": [subtask.model_dump()]} | changes


def test_mock_planning_preserves_original_condition_and_subquestion_order():
    planner, _, _ = modules()
    request = AnalysisRequest(question="2026年收入是否下降？公司未确认收入的原因是什么？", report_ids=["r1"])
    plan = planner.plan_query(request, reports(), None)
    assert plan.original_question == request.question
    assert len(plan.subtasks) == 2
    assert "是否" in plan.subtasks[0].rewritten_query
    assert "未" in plan.subtasks[1].rewritten_query
    assert all(t.source_text in request.question for t in plan.subtasks)
    assert "mock" in str(plan.subtasks[0].soft_hints)


@pytest.mark.parametrize("question", [
    "2026年上半年经营现金流净额占同期营业收入的比例是多少？",
    "2026年上半年经营活动产生的现金流量净额占营业收入的比重是多少？",
    "经营活动现金流量净额与当期营业收入之比是多少？",
])
def test_mock_cash_flow_share_plans_ratio_instead_of_revenue(question):
    planner, _, _ = modules()
    request = AnalysisRequest(question=question, report_ids=["r1"])
    plan = planner.plan_query(request, reports(), mode="mock")
    assert plan.route == "data"
    assert len(plan.subtasks) == 1
    planned = plan.subtasks[0]
    assert planned.route == "data"
    assert planned.metric_ids == ["operating_cash_flow_to_revenue"]
    assert planned.source_text == planned.query_text == question


@pytest.mark.parametrize("question", [
    "营业收入占经营现金流净额的比例是多少？",
    "经营现金流净额占净利润的比例是多少？",
    "经营现金流净额与营业收入分别是多少？",
])
def test_mock_cash_flow_ratio_does_not_reverse_operands_or_invent_ratio(question):
    planner, _, _ = modules()
    plan = planner.plan_query(AnalysisRequest(question=question, report_ids=["r1"]), reports(), mode="mock")
    assert all("operating_cash_flow_to_revenue" not in item.metric_ids for item in plan.subtasks)


def test_four_questions_are_clarified_without_silent_truncation():
    planner, _, _ = modules()
    plan = planner.plan_query(AnalysisRequest(question="收入？成本？现金流？风险？", report_ids=["r1"]), reports(), None)
    assert plan.needs_clarification and not plan.subtasks


def test_missing_selected_report_is_global_clarification():
    planner, _, _ = modules()
    plan = planner.plan_query(AnalysisRequest(question="风险是什么？", report_ids=["missing"]), reports(), None)
    assert plan.needs_clarification


def test_ambiguous_local_comparison_does_not_discard_other_question():
    planner, _, _ = modules()
    plan = planner.plan_query(AnalysisRequest(question="收入如何确认？现金流是否改善？", report_ids=["r1"]), reports(), None)
    assert not plan.needs_clarification
    assert not plan.subtasks[0].needs_clarification
    assert plan.subtasks[1].needs_clarification


@pytest.mark.parametrize("mutation", ["filter", "scope", "year", "negation", "presupposition", "omission"])
def test_live_plan_rejects_scope_and_meaning_drift_after_one_repair(mutation):
    planner, _, _ = modules()
    request = AnalysisRequest(question="甲公司2026年收入是否未下降？", report_ids=["r1"])
    payload = plan_payload(request)
    if mutation == "filter":
        payload["subtasks"][0]["hard_filters"] = {"section_path": "风险因素"}
    elif mutation == "scope":
        payload["report_ids"] = ["r1", "foreign"]
    elif mutation == "year":
        payload["subtasks"][0]["rewritten_query"] = "甲公司2025年收入是否未下降？"
    elif mutation == "negation":
        payload["subtasks"][0]["rewritten_query"] = "甲公司2026年收入是否下降？"
    elif mutation == "presupposition":
        payload["subtasks"][0]["rewritten_query"] = "甲公司2026年收入为什么未下降？"
    else:
        payload["subtasks"][0]["source_text"] = "收入"
        payload["subtasks"][0]["query_text"] = "收入"
        payload["subtasks"][0]["rewritten_query"] = "收入"
    client = JsonClient(payload)
    with pytest.raises(ModelClientError):
        planner.plan_query(request, reports(), client, mode="live")
    assert len(client.calls) == 2


def test_live_plan_normal_call_and_repair_usage_are_recorded_without_extra_schema_fields():
    planner, _, _ = modules()
    request = AnalysisRequest(question="收入如何确认？", report_ids=["r1"])
    client = JsonClient({"unexpected": True}, plan_payload(request))
    usage = []
    plan = planner.plan_query(request, reports(), client, mode="live", usage_sink=usage)
    assert len(plan.subtasks) == 1 and len(client.calls) == 2
    assert len(usage) == 2 and all(item["stage"] == "N4" for item in usage)
    assert "usage" not in plan.model_dump()


def test_live_transport_error_does_not_trigger_json_repair_or_mock_fallback():
    planner, _, _ = modules()
    request = AnalysisRequest(question="收入？", report_ids=["r1"])
    client = JsonClient(ModelClientError("NETWORK_ERROR", "offline", True))
    with pytest.raises(ModelClientError, match="NETWORK_ERROR"):
        planner.plan_query(request, reports(), client, mode="live")
    assert len(client.calls) == 1


def test_report_plan_uses_fixed_sections_and_bounded_subtasks():
    planner, _, _ = modules()
    plan = planner.plan_query(AnalysisRequest(question="生成分析报告", report_ids=["r1"], task_type="report"), reports(), None)
    assert [section["section_id"] for section in plan.sections] == [
        "scope", "overview", "metrics", "explanations", "risks", "gaps", "citations"]
    assert 1 <= len(plan.subtasks) <= 12
    assert all(t.section_id in {s["section_id"] for s in plan.sections} for t in plan.subtasks)


def test_mock_composer_uses_actual_evidence_and_explicit_missing_information():
    _, composer, _ = modules()
    answer = composer.compose("收入如何确认？", task(), [evidence()], [], [])
    assert answer.claims and evidence().quote in answer.claims[0].text
    assert answer.claims[0].evidence_ids == ["e1"]
    empty = composer.compose("收入如何确认？", task(), [], [], [])
    assert not empty.claims and empty.missing_information


def test_live_composer_prompt_delimits_untrusted_evidence_and_captures_usage():
    _, composer, _ = modules()
    client = JsonClient(draft().model_dump())
    usage = []
    result = composer.compose("收入如何确认？", task(), [evidence()], [], [], client, "live", usage_sink=usage)
    assert result.subtask_id == "q1"
    assert "不可信" in client.calls[0][0] and "json" in client.calls[0][0].lower()
    assert "产品交付" in client.calls[0][1]
    assert usage[0]["stage"] == "N8" and usage[0]["subtask_id"] == "q1"


def test_verifier_accepts_locatable_current_evidence_with_semantic_boundary_warning():
    _, _, verifier = modules()
    result = verifier.verify(draft(), context())
    assert result.status == "answered" and result.primary_evidence_ids == ["e1"]
    assert any("semantic" in warning for warning in result.warnings)


@pytest.mark.parametrize("case", ["unknown_id", "foreign_task", "wrong_report", "wrong_index", "wrong_page",
                                  "fabricated_quote", "filter_mismatch", "unsent_quote", "invented_number"])
def test_verifier_rejects_false_lineage_and_fabricated_numbers(case):
    _, _, verifier = modules()
    answer, ctx = draft(), context()
    if case == "unknown_id":
        answer.claims[0].evidence_ids = ["not-sent"]
    elif case == "foreign_task":
        answer.claims[0].subtask_id = "q2"
    elif case == "wrong_report":
        ctx.evidence[0].report_id = "foreign"
    elif case == "wrong_index":
        ctx.evidence[0].index_version = "old"
    elif case == "wrong_page":
        ctx.evidence[0].page_refs = [{"pdf_page_number": 3}]
    elif case == "fabricated_quote":
        ctx.evidence[0].quote = "款项全部收回后确认收入。"
    elif case == "filter_mismatch":
        ctx.task.hard_filters.company = "乙公司"
    elif case == "unsent_quote":
        ctx.pages["r1"][2] += "收入为900亿元。"
        answer.claims[0].text = "收入为900亿元。"
    else:
        answer.claims[0].text = "收入为999亿元。"
    result = verifier.verify(answer, ctx)
    assert result.status == "failed" and not result.claims


def financial_fixture():
    from annual_report_agent.finance.calculator import calculate_metric
    ev = evidence(quote="2026年收入120元，2025年收入100元。")
    facts = [FinancialFact(
        id=f"f{year}", metric_code="revenue", value=Decimal(value), raw_value=value,
        raw_unit="元", company="甲公司", period_start=f"{year}-01-01", period_end=f"{year}-12-31",
        scope="consolidated", report_id="r1", source_id="s1", source_evidence_ids=["e1"],
        validation_status="verified", review_record={"method": "fixture"},
    ) for year, value in [(2026, "120"), (2025, "100")]]
    calculation = calculate_metric("revenue_yoy", {"current_revenue": facts[0], "previous_revenue": facts[1]})
    return ev, facts, calculation


def test_verified_calculation_recomputes_and_supports_percent_claim():
    _, composer, verifier = modules()
    ev, facts, calculation = financial_fixture()
    numeric_task = task(route="data", metric_ids=["revenue_yoy"])
    answer = composer.compose("收入同比是多少？", numeric_task, [ev], facts, [calculation])
    ctx = context(task=numeric_task, evidence=[ev], facts=facts, calculations=[calculation],
                  pages={"r1": {2: ev.quote}})
    result = verifier.verify(answer, ctx)
    assert result.status == "answered"
    assert calculation.id in result.calculation_ids


@pytest.mark.parametrize("case", ["unverified", "wrong_result", "wrong_operand", "false_percent", "wrong_unit"])
def test_financial_claim_requires_verified_operands_and_recomputed_value(case):
    _, _, verifier = modules()
    ev, facts, calculation = financial_fixture()
    answer = draft("营业收入同比为20%。", calculation_ids=[calculation.id])
    if case == "unverified":
        facts[0].validation_status = "pending"
    elif case == "wrong_result":
        calculation.result = Decimal("30")
    elif case == "wrong_operand":
        calculation.operand_fact_ids[0] = "not-sent"
    elif case == "false_percent":
        answer.claims[0].text = "营业收入同比为21%。"
    else:
        answer.claims[0].text = "营业收入同比为20亿元。"
    result = verifier.verify(answer, context(evidence=[ev], facts=facts, calculations=[calculation],
                                             pages={"r1": {2: ev.quote}}))
    assert result.status == "failed" and not result.claims


@pytest.mark.parametrize("statuses,expected", [
    (["answered", "answered"], "answered"), (["answered", "insufficient_evidence"], "partial"),
    (["needs_clarification", "failed"], "needs_clarification"),
    (["unsupported", "failed"], "unsupported"), (["failed", "not_executed"], "failed"),
    (["not_executed", "insufficient_evidence"], "not_executed"),
    (["insufficient_evidence", "insufficient_evidence"], "insufficient_evidence"),
])
def test_aggregate_follows_documented_status_priority(statuses, expected):
    _, _, verifier = modules()
    tasks = [task(subtask_id="q1"), task(subtask_id="q2")]
    plan = QueryPlan(original_question="收入如何确认？", rewritten_question="收入如何确认？",
                     report_ids=["r1"], subtasks=tasks)
    results = [SubTaskResult(subtask_id=t.subtask_id, status=status,
               claims=[Claim(subtask_id=t.subtask_id, text="有据结论", evidence_ids=["e1"])]
               if status == "answered" else []) for t, status in zip(tasks, statuses, strict=True)]
    result = verifier.aggregate(plan, list(reversed(results)))
    assert result.status == expected
    assert [item.subtask_id for item in result.subtask_results] == ["q1", "q2"]


def test_missing_subtask_is_not_executed_and_global_error_preserves_completed_answer():
    _, _, verifier = modules()
    plan = QueryPlan(original_question="收入如何确认？", rewritten_question="收入如何确认？",
                     report_ids=["r1"], subtasks=[task(), task(subtask_id="q2")])
    done = SubTaskResult(subtask_id="q1", status="answered", claims=draft().claims)
    result = verifier.aggregate(plan, [done])
    assert result.status == "partial" and result.subtask_results[1].status == "not_executed"
    failed = verifier.aggregate(plan, [done], global_error="storage unavailable")
    assert failed.status == "failed" and failed.claims


def test_duplicate_results_cannot_hide_a_missing_subtask():
    _, _, verifier = modules()
    plan = QueryPlan(original_question="收入如何确认？", rewritten_question="收入如何确认？",
                     report_ids=["r1"], subtasks=[task(), task(subtask_id="q2")])
    done = SubTaskResult(subtask_id="q1", status="answered", claims=draft().claims)
    result = verifier.aggregate(plan, [done, done])
    assert result.status == "failed"
    assert result.subtask_results[1].status == "not_executed"


def test_live_plan_cannot_collapse_two_explicit_questions_into_one_task():
    planner, _, _ = modules()
    request = AnalysisRequest(question="收入如何确认？风险是什么？", report_ids=["r1"])
    client = JsonClient(plan_payload(request))
    with pytest.raises(ModelClientError, match="INVALID_PLAN"):
        planner.plan_query(request, reports(), client, "live")
    assert len(client.calls) == 2


def test_fact_raw_value_must_match_a_complete_source_numeral():
    _, _, verifier = modules()
    ev, facts, calculation = financial_fixture()
    ev.quote = "2026年收入1120元，2025年收入100元。"
    answer = draft("营业收入同比为20%。", calculation_ids=[calculation.id])
    result = verifier.verify(answer, context(evidence=[ev], facts=facts, calculations=[calculation],
                                             pages={"r1": {2: ev.quote}}))
    assert result.status == "failed" and not result.claims


def test_spelled_out_chinese_amount_cannot_bypass_numeric_guard():
    _, _, verifier = modules()
    result = verifier.verify(draft("收入增加一百亿元。"), context())
    assert result.status == "failed" and not result.claims


def test_direct_financial_metric_works_with_service_without_calculation():
    from annual_report_agent.finance.service import select_and_calculate
    planner, composer, verifier = modules()
    request = AnalysisRequest(question="2026年营业收入是多少？", report_ids=["r1"])
    planned = planner.plan_query(request, reports(), None).subtasks[0]
    assert planned.metric_ids == ["revenue"] and planned.route == "data"
    ev, facts, _ = financial_fixture()
    bundle = select_and_calculate(planned, facts)
    answer = composer.compose(request.question, planned, [ev], bundle.facts, bundle.calculations)
    result = verifier.verify(answer, context(task=planned, evidence=[ev], facts=bundle.facts,
                                             pages={"r1": {2: ev.quote}}))
    assert result.status == "answered" and result.fact_ids == ["f2026"]
    assert result.calculation_ids == []


def test_explicit_current_period_alias_allows_valid_yoy_comparative_operands():
    _, composer, verifier = modules()
    ev, facts, calculation = financial_fixture()
    planned = task(route="data", metric_ids=["revenue_yoy"],
                   report_periods=[{"start": "2026-01-01", "end": "2026-12-31"}])
    answer = composer.compose("收入同比多少？", planned, [ev], facts, [calculation])
    result = verifier.verify(answer, context(task=planned, evidence=[ev], facts=facts,
                      calculations=[calculation], pages={"r1": {2: ev.quote}}))
    assert result.status == "answered"


def test_live_plan_preserves_half_year_vs_annual_condition():
    planner, _, _ = modules()
    request = AnalysisRequest(question="2026年上半年收入是多少？", report_ids=["r1"])
    payload = plan_payload(request)
    payload["subtasks"][0]["rewritten_query"] = "2026年全年收入是多少？"
    with pytest.raises(ModelClientError, match="INVALID_PLAN"):
        planner.plan_query(request, reports(), JsonClient(payload), "live")


def test_explicit_companies_narrow_each_subtask_within_selected_scope():
    planner, _, _ = modules()
    source_reports = reports() + [{"id": "r2", "company": "乙公司", "doc_type": "annual_report",
                                   "fiscal_year": 2026, "revision": "v1"}]
    request = AnalysisRequest(question="甲公司收入如何确认？乙公司风险是什么？", report_ids=["r1", "r2"])
    plan = planner.plan_query(request, source_reports, None)
    assert plan.subtasks[0].report_ids == ["r1"]
    assert plan.subtasks[1].report_ids == ["r2"]


def test_hybrid_answer_without_requested_calculation_is_partial_not_complete():
    _, _, verifier = modules()
    planned = task(route="hybrid", metric_ids=["revenue_yoy"])
    result = verifier.verify(draft(), context(task=planned))
    assert result.status == "partial" and result.claims


def test_fact_numeric_lookup_preserves_separate_table_numbers_and_sentence_period():
    _, composer, verifier = modules()
    ev, facts, calculation = financial_fixture()
    ev.quote = "Revenue 2026 120; Revenue 2025 100."
    planned = task(route="data", metric_ids=["revenue_yoy"])
    answer = composer.compose("收入同比多少？", planned, [ev], facts, [calculation])
    result = verifier.verify(answer, context(task=planned, evidence=[ev], facts=facts,
                      calculations=[calculation], pages={"r1": {2: ev.quote}}))
    assert result.status == "answered"
