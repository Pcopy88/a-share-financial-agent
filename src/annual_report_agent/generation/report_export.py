"""Deterministic Markdown export of an already verified AnswerResult.

No model calls, environment access, new calculations or new factual conclusions occur.
The optional repository is read only by exact report/fact IDs; configuration is never read.
"""

import html
import re
from decimal import Decimal, InvalidOperation
from urllib.parse import quote, urlsplit, urlunsplit

from pydantic import BaseModel, ValidationError

from annual_report_agent.generation.prompts import METRIC_LABELS, REPORT_SECTIONS, UNIT_LABELS
from annual_report_agent.schemas.answer import AnswerResult, SubTaskResult
from annual_report_agent.schemas.finance import Calculation

_STATUS = {
    "answered": "已回答", "partial": "部分回答", "insufficient_evidence": "资料不足",
    "needs_clarification": "待澄清", "unsupported": "暂不支持", "not_executed": "未执行", "failed": "失败",
}
_FORMULAS = {
    "revenue_yoy": "(本期收入 / 可比上期收入 - 1) × 100%",
    "parent_net_profit_yoy": "(本期归母净利润 / 可比上期归母净利润 - 1) × 100%",
    "gross_margin": "(营业收入 - 营业成本) / 营业收入 × 100%",
    "net_margin": "净利润 / 营业收入 × 100%",
    "debt_asset_ratio": "总负债 / 总资产 × 100%",
    "current_ratio": "流动资产 / 流动负债",
    "operating_cash_flow_to_revenue": "经营活动现金流量净额 / 营业收入 × 100%",
    "gross_margin_yoy_pp": "本期毛利率 - 可比上期毛利率（百分点）",
}
_PUBLIC_FIELDS = {
    "provider", "model", "model_id", "response_model", "model_profile_id", "profile_id",
    "embedding_profile_hash", "embedding_profile_id", "dimension", "mode", "normalization",
    "distance_metric", "endpoint", "base_url", "region", "revision", "version", "precision",
    "device", "tokenizer_revision", "tokenizer_sha256", "tokenizer_version", "parser_version",
    "metadata_version", "chunk_size", "overlap", "top_k", "retrieval_mode", "rerank_enabled",
    "context_token_budget", "thinking", "response_format", "max_tokens", "max_retries", "timeout",
    "date", "call_date", "system_fingerprint", "type", "index_version", "report_index_versions",
    "embedding", "generation", "reranker", "tokenizer", "retrieval", "chunking", "model_profiles",
    "parameters", "runtime", "settings", "profiles", "embedding_model", "embedding_dimension",
    "embedding_url", "deepseek_model", "deepseek_base_url", "max_run_calls", "max_run_tokens",
}
_SECRET_FIELD = re.compile(r"api.?key|authorization|password|secret|credential|access.?token", re.I)


def _dict(value):
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)


def _text(value) -> str:
    """Treat source text as text, never as Markdown/HTML instructions or links."""
    value = re.sub(r"\s+", " ", str(value)).strip()
    value = html.escape(value, quote=False)
    return re.sub(r"([\\`*\[\]_|#])", r"\\\1", value)


def _safe_url(value):
    try:
        parsed = urlsplit(str(value))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return "未记录可展示的服务地址"
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        if parsed.port:
            host += f":{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except ValueError:
        return "未记录可展示的服务地址"


def _public_metadata(value, prefix=""):
    output = []
    if not isinstance(value, dict):
        return output
    for key, item in value.items():
        if key not in _PUBLIC_FIELDS or _SECRET_FIELD.search(key):
            continue
        label = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            if key == "report_index_versions":
                output.extend((f"{label}.{identifier}", version)
                              for identifier, version in item.items() if isinstance(version, str))
            else:
                output.extend(_public_metadata(item, label))
        elif isinstance(item, (str, int, float, bool)) or item is None:
            if key in {"endpoint", "base_url", "embedding_url", "deepseek_base_url"}:
                item = _safe_url(item)
            output.append((label, "未知" if item is None else item))
    return output


def _secret_values(value):
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if _SECRET_FIELD.search(str(key)) and isinstance(item, str) and len(item) >= 4:
                found.add(item)
                if item.lower().startswith("bearer "):
                    found.add(item[7:])
            found.update(_secret_values(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_secret_values(item))
    return found


def _section(task):
    selected = task.get("section_id")
    if selected in {identifier for identifier, _ in REPORT_SECTIONS}:
        return selected
    text = task.get("source_text", "") + task.get("query_text", "")
    if task.get("route") == "data":
        return "metrics"
    if "风险" in text:
        return "risks"
    if any(marker in text for marker in ("原因", "解释", "为什么")):
        return "explanations"
    return "overview"


def _ordered_results(answer):
    tasks = answer.query_plan.get("subtasks", [])
    mapping = {item.subtask_id: item for item in answer.subtask_results}
    if len(mapping) != len(answer.subtask_results):
        raise ValueError("cannot export duplicate subtask results")
    if not tasks:
        return [({}, item) for item in answer.subtask_results]
    ordered = []
    for task in tasks:
        identifier = task["subtask_id"]
        item = mapping.get(identifier)
        if item is None:
            item = SubTaskResult(subtask_id=identifier, status="not_executed",
                                 reason="该子问题尚未执行，不能推断其事实结论。")
        ordered.append((task, item))
    return ordered


def _operand_lines(identifier, fact):
    if not fact:
        return [f"- 操作数 {_text(identifier)}：操作数记录缺失。"]
    if fact.get("validation_status") != "verified":
        return [f"- 操作数 {_text(identifier)}：尚未核验，未展示为可信数值。"]
    try:
        if isinstance(fact.get("value"), (float, bool)):
            raise ValueError("lossy financial value")
        amount = Decimal(str(fact["value"]))
        if not amount.is_finite():
            raise ValueError("nonfinite financial value")
    except (KeyError, ValueError, InvalidOperation):
        return [f"- 操作数 {_text(identifier)}：数值记录无效。"]
    period = (f"{fact['period_start']} 至 {fact.get('period_end', '未知')}"
              if fact.get("period_start") else f"时点 {fact.get('period_end', '未知')}")
    label = METRIC_LABELS.get(fact.get("metric_code"), fact.get("metric_code", "指标未知"))
    scope = "合并口径" if fact.get("scope") == "consolidated" else "母公司口径"
    sources = "、".join(str(item) for item in fact.get("source_evidence_ids", [])) or "未附原文引用"
    return [f"- 操作数 {_text(identifier)}：{_text(label)} {_text(format(amount, 'f'))}"
            f"{_text(fact.get('unit', ''))}；{_text(period)}；{scope}。",
            f"  原始值：{_text(fact.get('raw_value', '未知'))} {_text(fact.get('raw_unit', '未知'))}；"
            f"来源：{_text(sources)}。"]


def export_markdown(answer: AnswerResult, repo=None, *, metadata: dict | None = None,
                    facts: list | None = None, reports: list | None = None,
                    evidence_base_url: str = "/api/reports") -> str:
    """Return Markdown; caller owns file saving and HTTP download handling.

    Optional repo implements get(kind, id); only kinds 'report' and 'fact' are read.
    metadata may contain public model profiles/parameters; unknown and credential fields
    are excluded. Relative page links work through the API; pass its HTTP root for a
    standalone export. No filesystem paths or environment configuration are inspected.
    """
    answer = AnswerResult.model_validate(answer)
    parsed = urlsplit(evidence_base_url)
    if (parsed.scheme and parsed.scheme not in {"http", "https"}) or parsed.query or parsed.fragment:
        raise ValueError("evidence_base_url must be an HTTP(S) URL or absolute relative path")
    if (parsed.username or parsed.password or evidence_base_url.startswith("//")
            or not parsed.scheme and not evidence_base_url.startswith("/")):
        raise ValueError("evidence_base_url must not contain credentials or relative traversal")
    root = evidence_base_url.rstrip("/")
    ordered = _ordered_results(answer)
    successful = [item for _, item in ordered if item.status in {"answered", "partial"}]
    evidence_ids = {identifier for item in successful for identifier in (
        item.primary_evidence_ids + item.context_evidence_ids
        + [identifier for claim in item.claims if claim.subtask_id == item.subtask_id
           for identifier in claim.evidence_ids]
    )}
    calculation_ids = {identifier for item in successful for identifier in (
        item.calculation_ids + [identifier for claim in item.claims if claim.subtask_id == item.subtask_id
                                for identifier in claim.calculation_ids]
    )}
    fact_ids = {identifier for item in successful for identifier in (
        item.fact_ids + [identifier for claim in item.claims if claim.subtask_id == item.subtask_id
                         for identifier in claim.fact_ids]
    )}
    calculations = {item.get("id"): item for item in answer.calculations
                    if item.get("id") in calculation_ids}
    fact_ids.update(identifier for item in calculations.values() for identifier in item.get("operand_fact_ids", []))
    fact_map = {item["id"]: item for value in facts or [] if (item := _dict(value)).get("id") in fact_ids}
    report_ids = list(dict.fromkeys(answer.query_plan.get("report_ids", [])
                                   + [citation.report_id for citation in answer.citations
                                      if citation.id in evidence_ids]))
    report_map = {item["id"]: item for value in reports or [] if (item := _dict(value)).get("id") in report_ids}
    if repo is not None:
        for identifier in sorted(fact_ids):
            if identifier not in fact_map and (item := repo.get("fact", identifier)) is not None:
                fact_map[identifier] = _dict(item)
        for identifier in report_ids:
            if identifier not in report_map and (item := repo.get("report", identifier)) is not None:
                report_map[identifier] = _dict(item)
    evidence_ids.update(identifier for fact in fact_map.values()
                        for identifier in fact.get("source_evidence_ids", []))
    citations = [item for item in answer.citations if item.id in evidence_ids]
    blocks = {identifier: [] for identifier, _ in REPORT_SECTIONS}
    blocks["scope"].extend([
        f"原始问题：{_text(answer.original_question)}",
        f"本次结果：{_STATUS[answer.status]}。运行编号：{_text(answer.run_id or '未记录')}。",
        "以下内容仅整理本次已校验结果；未回答的项目会保留原因。",
    ])
    if answer.query_plan.get("needs_clarification"):
        blocks["scope"].append("待澄清：" + _text(answer.query_plan.get("clarification_reason") or "请明确分析范围。"))
    for identifier in report_ids:
        report = report_map.get(identifier, {})
        detail = "；".join(f"{label}：{_text(report[key])}" for key, label in (
            ("company", "公司"), ("doc_type", "文档类型"), ("revision", "文档版本"),
            ("fiscal_year", "内容年度"), ("disclosure_date", "披露日期"), ("sha256", "文件SHA256"),
        ) if report.get(key) is not None)
        blocks["scope"].append(f"- 资料 {_text(identifier)}" + (f"：{detail}。" if detail else "：未附额外元数据。"))
    for task, item in ordered:
        section = _section(task)
        label = task.get("query_text") or task.get("source_text") or item.subtask_id
        content = [f"### {_text(item.subtask_id)} · {_text(label)}", f"状态：{_STATUS[item.status]}。"]
        valid_claims = [claim for claim in item.claims
                        if item.status in {"answered", "partial"} and claim.subtask_id == item.subtask_id]
        for claim in valid_claims:
            content.append(_text(claim.text))
            sources = claim.evidence_ids + claim.fact_ids + claim.calculation_ids
            if sources:
                content.append("依据：" + "、".join(_text(identifier) for identifier in sources) + "。")
        if item.reason:
            content.append("说明：" + _text(item.reason))
        if not valid_claims and not item.reason:
            content.append("此项没有可输出的已校验结论。")
        blocks[section].extend(content)
        if item.status != "answered":
            blocks["gaps"].append(f"- {_text(item.subtask_id)}：{_STATUS[item.status]}。"
                                  + _text(item.reason or "此项仍有未完成内容。"))
        blocks["gaps"].extend(f"- {_text(item.subtask_id)} 核验说明：{_text(warning)}" for warning in item.warnings)
    if calculation_ids:
        blocks["metrics"].append("### 程序计算依据")
    for identifier in sorted(calculation_ids):
        raw = calculations.get(identifier)
        if raw is None:
            blocks["metrics"].append(f"- {_text(identifier)}：计算记录缺失，未展开。")
            continue
        try:
            calculation = Calculation.model_validate(raw)
            if not calculation.result.is_finite() or calculation.formula_id not in _FORMULAS:
                raise ValueError("invalid formula/result")
        except (ValidationError, ValueError):
            blocks["metrics"].append(f"- {_text(identifier)}：计算记录无效，未展开。")
            continue
        blocks["metrics"].extend([
            f"计算 {_text(identifier)}：{_text(METRIC_LABELS[calculation.formula_id])} "
            f"{_text(format(calculation.result, 'f'))}{UNIT_LABELS[calculation.unit]}。",
            f"公式：{_text(_FORMULAS[calculation.formula_id])}。公式版本：{_text(calculation.formula_version)}。",
            f"期间：{_text(calculation.period_label)}。",
        ])
        for operand in calculation.operand_fact_ids:
            blocks["metrics"].extend(_operand_lines(operand, fact_map.get(operand)))
    direct_facts = fact_ids - {identifier for item in calculations.values() for identifier in item.get("operand_fact_ids", [])}
    if direct_facts:
        blocks["metrics"].append("### 已核验事实依据")
        for identifier in sorted(direct_facts):
            blocks["metrics"].extend(_operand_lines(identifier, fact_map.get(identifier)))
    for citation in citations:
        links = []
        for reference in citation.page_refs:
            page = reference.get("pdf_page_number")
            if type(page) is not int or page < 1:
                continue
            printed = reference.get("printed_page_label")
            label = f"PDF物理页 {page}" + (f"（印刷页 {_text(printed)}）" if printed else "（印刷页未记录）")
            links.append(f"[{label}]({root}/{quote(citation.report_id, safe='')}/pages/{page})")
        blocks["citations"].extend([
            f"### {_text(citation.id)} · 资料 {_text(citation.report_id)}",
            "；".join(links) or "物理页引用缺失。",
            f"索引版本：{_text(citation.index_version)}。",
            "> " + _text(citation.quote),
        ])
    missing_citations = evidence_ids - {item.id for item in citations}
    if missing_citations:
        blocks["gaps"].append("引用记录缺失：" + "、".join(_text(item) for item in sorted(missing_citations)) + "。")
    blocks["gaps"].extend("- " + _text(warning) for warning in dict.fromkeys(answer.warnings))
    public = _public_metadata(metadata or {})
    public.extend((f"report_index_versions.{item.report_id}", item.index_version) for item in citations)
    if public:
        blocks["scope"].append("### 模型与运行参数")
        blocks["scope"].extend(f"- {_text(key)}：{_text(value)}" for key, value in dict(public).items())
    if answer.usage:
        blocks["scope"].append("### 调用记录")
        for usage in answer.usage:
            values = {key: usage[key] for key in (
                "stage", "subtask_id", "model", "request_id", "attempts", "latency") if key in usage}
            tokens = usage.get("usage")
            if isinstance(tokens, dict):
                values.update({key: value for key, value in tokens.items()
                               if key in {"total_tokens", "prompt_tokens", "completion_tokens",
                                          "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"}
                               and type(value) is int and value >= 0})
            blocks["scope"].append("- " + "；".join(f"{_text(key)}={_text(value)}" for key, value in values.items()))
    output = ["# 财务资料分析报告"]
    if answer.mode == "mock":
        output.append("**模拟模式（mock）：本报告用于流程演示，不代表真实云端模型的理解或回答质量。**")
    else:
        output.append("运行模式：live。核验范围以逐项结果及其说明为准。")
    for identifier, title in REPORT_SECTIONS:
        output.append("## " + title)
        output.extend(blocks[identifier] or [
            "本节未产生已校验内容；不能据此断言原文未披露。" if identifier != "gaps"
            else "本次结果未附额外缺口说明。"
        ])
    rendered = "\n\n".join(output) + "\n"
    for secret in sorted(_secret_values(metadata or {}), key=len, reverse=True):
        rendered = rendered.replace(secret, "[REDACTED]").replace(_text(secret), "[REDACTED]")
    return rendered
