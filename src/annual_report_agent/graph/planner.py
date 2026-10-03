"""N4: bounded planning, explicit mock rules, and local scope/meaning guards.

The deterministic guards preserve concrete markers, not arbitrary semantic equivalence.
Mock rules demonstrate orchestration and must not be reported as LLM understanding.
"""

import re

from pydantic import ValidationError

from annual_report_agent.generation.cloud_client import DeepSeekClient
from annual_report_agent.generation.prompts import (
    PLANNER_SYSTEM,
    REPORT_SECTIONS,
    dump_json,
    record_usage,
)
from annual_report_agent.schemas.model import ModelClientError
from annual_report_agent.schemas.query import AnalysisRequest, QueryPlan, SubTaskPlan

_NEGATIONS = ("没有", "不是", "未", "不", "无", "否")
_METRICS = {
    "毛利率同比": "gross_margin_yoy_pp", "毛利率变化": "gross_margin_yoy_pp",
    "毛利率": "gross_margin", "净利率": "net_margin", "资产负债率": "debt_asset_ratio",
    "流动比率": "current_ratio", "经营现金流收入比": "operating_cash_flow_to_revenue",
}


def _parts(question: str) -> list[str]:
    return [part.strip() for part in re.findall(r"[^？?；;]+[？?；;]?", question) if part.strip()]


def _clarification(request, reason):
    return QueryPlan(task_type=request.task_type, original_question=request.question,
                     rewritten_question=request.question, report_ids=request.report_ids,
                     hard_filters=request.hard_filters, needs_clarification=True,
                     clarification_reason=reason)


def _scope(request, reports):
    by_id = {report["id"]: report for report in reports}
    if not set(request.report_ids).issubset(by_id):
        return [], "选中的资料不存在，请重新确认资料范围。"
    selected = [by_id[item] for item in request.report_ids]
    for key, value in request.hard_filters.model_dump(exclude_none=True).items():
        if key in {"section_path", "question_number"}:
            continue
        selected = [report for report in selected if report.get(key) == value]
    if not selected:
        return [], "已选资料与明确过滤条件不匹配，不能放宽范围。"
    companies = {r.get("company") for r in selected if r.get("company")}
    mentioned = {r.get("company") for r in reports
                 if r.get("company") and r["company"] in request.question}
    if mentioned and not mentioned.issubset(companies):
        return [], "问题所指公司与已选资料不一致。"
    if len(companies) > 1 and not mentioned and not re.search("对比|比较|分别|各公司", request.question):
        return [], "选中了多家公司，请明确本次分析对象或比较范围。"
    if mentioned:
        selected = [r for r in selected if r.get("company") in mentioned]
    # Explicit inquiry round can narrow known reports, but unknown identity never matches.
    round_match = re.search(r"第([一二三四五六七八九十\d]+)轮", request.question)
    if round_match:
        value = round_match[1]
        number = int(value) if value.isdigit() else "一二三四五六七八九十".find(value) + 1
        selected = [r for r in selected if r.get("inquiry_round") == number]
        if not selected:
            return [], "没有身份已确认且符合所问轮次的已选资料。"
    versions = {}
    for report in selected:
        key = tuple(report.get(field) for field in ("company", "doc_type", "fiscal_year", "inquiry_round"))
        versions.setdefault(key, set()).add(report.get("revision"))
    if not request.hard_filters.revision and any(len(values) > 1 for values in versions.values()):
        return [], "已选资料存在多个版本，请明确使用哪个版本。"
    return selected, None


def _intent(text):
    metrics = []
    compact = re.sub(r"\s+", "", text)
    if re.search(r"经营(?:活动)?(?:产生的)?现金流(?:量)?净额(?:占|与)"
                 r"(?:同期|当期|本期)?营业收入(?:的|之)?(?:比例|比率|比重|比)", compact):
        metrics.append("operating_cash_flow_to_revenue")
    for name, identifier in _METRICS.items():
        if name in text and not any(identifier in existing for existing in metrics):
            if identifier == "gross_margin" and "gross_margin_yoy_pp" in metrics:
                continue
            metrics.append(identifier)
    if re.search(r"收入.*(?:同比|增长|增幅|下降)", text):
        metrics.append("revenue_yoy")
    if re.search(r"归母.*(?:同比|增长|增幅|下降)", text):
        metrics.append("parent_net_profit_yoy")
    explanatory = bool(re.search("为什么|原因|解释|如何|怎么|咋|政策|确认|风险", text))
    if not metrics and not explanatory:
        for name, identifier in (
            ("归母净利润", "parent_net_profit"), ("营业收入", "revenue"), ("收入", "revenue"),
            ("营业成本", "operating_cost"), ("净利润", "net_profit"),
            ("流动资产", "current_assets"), ("流动负债", "current_liabilities"),
            ("总资产", "total_assets"), ("总负债", "total_liabilities"),
            ("经营活动现金流量净额", "operating_cash_flow"),
        ):
            if name in text:
                metrics.append(identifier)
                break
    numeric = bool(metrics) or bool(re.search("是多少|多少元|金额|数值", text))
    route = "hybrid" if numeric and explanatory else "data" if numeric else "semantic"
    return route, list(dict.fromkeys(metrics))


def _subscope(source, selected):
    mentioned = {r["company"] for r in selected if r.get("company") and r["company"] in source}
    return [r["id"] for r in selected if not mentioned or r.get("company") in mentioned]


def _mock_plan(request, selected):
    if request.task_type == "report":
        definitions = [
            ("overview", "经营概况和主要业务", "semantic", []),
            ("metrics", "营业收入同比", "data", ["revenue_yoy"]),
            ("metrics", "毛利率", "data", ["gross_margin"]),
            ("metrics", "资产负债率", "data", ["debt_asset_ratio"]),
            ("explanations", "管理层对经营变化的解释", "semantic", []),
            ("risks", "已披露的主要风险事项", "semantic", []),
        ]
        tasks = [SubTaskPlan(subtask_id=f"q{i}", source_text=request.question, query_text=text,
                            rewritten_query=text, report_ids=[r["id"] for r in selected], route=route,
                            hard_filters=request.hard_filters, metric_ids=metrics, section_id=section,
                            soft_hints={"planning_mode": "fixed_template"})
                 for i, (section, text, route, metrics) in enumerate(definitions, 1)]
        sections = [{"section_id": key, "title": title,
                     "subtask_ids": [t.subtask_id for t in tasks if t.section_id == key]}
                    for key, title in REPORT_SECTIONS]
        return QueryPlan(task_type="report", original_question=request.question,
                         rewritten_question=request.question, report_ids=[r["id"] for r in selected],
                         hard_filters=request.hard_filters, route="hybrid", subtasks=tasks, sections=sections)
    parts = _parts(request.question)
    if len(parts) > 3:
        return _clarification(request, "普通问答最多处理3项明确问题，请收窄本次范围。")
    tasks = []
    for index, source in enumerate(parts, 1):
        route, metrics = _intent(source)
        ambiguous = "改善" in source and not re.search(r"同比|环比|20\d{2}.*20\d{2}", source)
        rewritten = source.replace("咋", "如何").replace("有啥", "有哪些")
        tasks.append(SubTaskPlan(
            subtask_id=f"q{index}", source_text=source, query_text=source, rewritten_query=rewritten,
            report_ids=_subscope(source, selected), hard_filters=request.hard_filters, route=route,
            fiscal_years=[int(y) for y in re.findall(r"(?:19|20)\d{2}", source)], metric_ids=metrics,
            scope="parent" if "母公司单体" in source else "consolidated",
            needs_clarification=ambiguous,
            clarification_reason="请明确比较期间和‘改善’所指指标。" if ambiguous else None,
            soft_hints={"planning_mode": "mock_rules", "quality_validated": False},
        ))
    routes = {t.route for t in tasks}
    return QueryPlan(original_question=request.question, rewritten_question=request.question,
                     report_ids=[r["id"] for r in selected], hard_filters=request.hard_filters,
                     route=next(iter(routes)) if len(routes) == 1 else "hybrid", subtasks=tasks)


def _guard_text(source, rewritten, companies):
    numbers = set(re.findall(r"\d+(?:\.\d+)?", source))
    if numbers != set(re.findall(r"\d+(?:\.\d+)?", rewritten)):
        raise ValueError("rewrite changed numeric or year conditions")
    for marker in _NEGATIONS:
        if (marker in source) != (marker in rewritten):
            raise ValueError("rewrite changed negation")
    if "是否" in source and ("是否" not in rewritten or "为什么" in rewritten):
        raise ValueError("rewrite changed a yes/no question into a presupposition")
    if "为什么" in source and not re.search("为什么|原因", rewritten):
        raise ValueError("rewrite dropped the reason question")
    for pattern in (r"上半年|半年度|半年报|H1", r"下半年|H2", r"全年|年度|年报",
                    r"一季度|第一季度|Q1", r"二季度|第二季度|Q2", r"三季度|第三季度|Q3",
                    r"四季度|第四季度|Q4", r"期末|年末"):
        if bool(re.search(pattern, source, re.I)) != bool(re.search(pattern, rewritten, re.I)):
            raise ValueError("rewrite changed reporting-period granularity")
    for company in companies:
        if (company in source) != (company in rewritten):
            raise ValueError("rewrite changed company")


def _validate_plan(data, request, selected):
    plan = QueryPlan.model_validate(data)
    allowed = [r["id"] for r in selected]
    filters = request.hard_filters.model_dump(exclude_none=True)
    if (plan.original_question != request.question or plan.task_type != request.task_type
            or set(plan.report_ids) != set(allowed)
            or plan.hard_filters.model_dump(exclude_none=True) != filters):
        raise ValueError("plan altered original question or selected scope")
    companies = {r["company"] for r in selected if r.get("company")}
    _guard_text(request.question, plan.rewritten_question, companies)
    if plan.needs_clarification:
        if not plan.clarification_reason:
            raise ValueError("global clarification requires a reason")
        return plan
    if plan.sections:
        raise ValueError("ordinary questions must not add report sections")
    if len(plan.subtasks) < len(_parts(request.question)):
        raise ValueError("plan collapsed explicit questions instead of preserving each requirement")
    coverage = [False] * len(request.question)
    last_position = -1
    for subtask in plan.subtasks:
        if (set(subtask.report_ids) != set(_subscope(subtask.source_text, selected))
                or subtask.hard_filters.model_dump(exclude_none=True) != filters):
            raise ValueError("model may not introduce or remove hard scope constraints")
        start = request.question.find(subtask.source_text, last_position + 1)
        if start < 0:
            raise ValueError("subquestions must follow original order")
        for index in range(start, start + len(subtask.source_text)):
            coverage[index] = True
        last_position = start
        _guard_text(subtask.source_text, subtask.query_text, companies)
        _guard_text(subtask.source_text, subtask.rewritten_query, companies)
        if subtask.needs_clarification and not subtask.clarification_reason:
            raise ValueError("local clarification requires a reason")
        if subtask.section_id is not None:
            raise ValueError("ordinary question cannot select an invented report section")
        source_years = {int(y) for y in re.findall(r"(?:19|20)\d{2}", subtask.source_text)}
        if not set(subtask.fiscal_years).issubset(source_years):
            raise ValueError("model invented years")
        expected_scope = "parent" if "母公司单体" in subtask.source_text else "consolidated"
        if subtask.scope != expected_scope or subtask.report_periods:
            raise ValueError("periods/scope require deterministic confirmation")
    uncovered = "".join(char for char, seen in zip(request.question, coverage, strict=True) if not seen)
    if re.search(r"[\w\u4e00-\u9fff]", uncovered):
        raise ValueError("plan omitted part of the original question")
    return plan


def plan_query(request: AnalysisRequest, reports: list[dict], client: DeepSeekClient | None = None,
               mode: str = "mock", *, usage_sink: list[dict] | None = None) -> QueryPlan:
    """Normal live question: one generation; malformed plans receive at most one repair.

    Fixed report templates are deterministic in both modes. Date ranges inferred by a
    model are not executed; callers must obtain explicit period confirmation for finance.
    """
    if mode not in {"mock", "live"}:
        raise ModelClientError("INVALID_MODE", "Unknown planning mode.")
    selected, reason = _scope(request, reports)
    if reason:
        return _clarification(request, reason)
    if request.task_type == "report" or mode == "mock":
        return _mock_plan(request, selected)
    if client is None:
        raise ModelClientError("MISSING_CLIENT", "Live planning requires a cloud client.")
    if len(_parts(request.question)) > 3:
        return _clarification(request, "普通问答最多处理3项明确问题，请收窄本次范围。")
    payload = {"question": request.question, "allowed_report_ids": [r["id"] for r in selected],
               "allowed_filters": request.hard_filters.model_dump(exclude_none=True),
               "explicit_source_scopes": [{"source_text": source, "report_ids": _subscope(source, selected)}
                                          for source in _parts(request.question)],
               "reports": [{key: report.get(key) for key in (
                   "id", "company", "doc_type", "fiscal_year", "inquiry_round", "revision")}
                   for report in selected], "schema": QueryPlan.model_json_schema()}
    for attempt in range(2):
        try:
            result = client.generate_json(PLANNER_SYSTEM, dump_json(payload), max_tokens=3072)
            record_usage(usage_sink, result, "N4")
            return _validate_plan(result.data, request, selected)
        except ModelClientError as error:
            if error.code not in {"INVALID_RESPONSE", "INCOMPLETE_OUTPUT"}:
                raise
            reason = error.code
        except (ValidationError, ValueError) as error:
            reason = str(error)[:500]
        if attempt == 0:
            payload["repair_instruction"] = "上次输出未通过本地校验，请只修复结构与条件保真：" + reason
    raise ModelClientError("INVALID_PLAN", "Planning failed local validation after one repair.")
