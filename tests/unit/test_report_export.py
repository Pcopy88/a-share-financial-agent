"""Export only verified results; preserve gaps, provenance and Decimal values."""

import importlib
import re

import pytest

from annual_report_agent.schemas.answer import AnswerResult, Claim, Evidence, SubTaskResult
from annual_report_agent.schemas.query import QueryPlan, SubTaskPlan


def exporter():
    return importlib.import_module("annual_report_agent.generation.report_export").export_markdown


def result(mode="mock"):
    tasks = [SubTaskPlan(subtask_id="q1", source_text="业务与风险", query_text="业务", rewritten_query="业务",
                         report_ids=["r1"], section_id="overview"),
             SubTaskPlan(subtask_id="q2", source_text="业务与风险", query_text="风险", rewritten_query="风险",
                         report_ids=["r1"], section_id="risks")]
    plan = QueryPlan(task_type="report", original_question="业务与风险", rewritten_question="业务与风险",
                     report_ids=["r1"], subtasks=tasks)
    claim = Claim(subtask_id="q1", text="公司生产精密零件。", evidence_ids=["e1"])
    return AnswerResult(run_id="run-123", mode=mode, status="partial", original_question="业务与风险",
        subtask_results=[SubTaskResult(subtask_id="q1", status="answered", claims=[claim],
                                      primary_evidence_ids=["e1"]),
                         SubTaskResult(subtask_id="q2", status="insufficient_evidence",
                                       reason="当前发送的证据不足以回答风险问题。")],
        claims=[claim], query_plan=plan.model_dump(mode="json"), citations=[Evidence(
            id="e1", report_id="r1", parent_id="p1", quote="公司生产精密零件。", index_version="index-abc",
            page_refs=[{"pdf_page_number": 12, "printed_page_label": "1-10"}])])


def test_export_has_fixed_seven_sections_and_explicit_mock_label():
    text = exporter()(result())
    assert re.findall(r"^## (.+)$", text, re.M) == [
        "分析范围", "经营概况", "主要指标", "管理层解释", "风险事项", "资料缺口", "引用清单"]
    assert "模拟模式" in text and "mock" in text and "真实云端" in text
    assert "公司生产精密零件。" in text
    assert "q1" in text and "已回答" in text and "q2" in text and "资料不足" in text
    assert "没有风险" not in text


def test_citations_distinguish_physical_and_printed_pages_and_keep_index_version():
    text = exporter()(result(), evidence_base_url="http://127.0.0.1:8000/api/reports")
    assert "/api/reports/r1/pages/12" in text
    assert "PDF物理页 12" in text and "印刷页 1-10" in text
    assert "index-abc" in text and "e1" in text


def test_failed_or_orphan_claims_are_not_exported_as_supported_facts():
    answer = result()
    invalid = Claim(subtask_id="q2", text="公司绝对没有风险。", evidence_ids=["e1"])
    answer.subtask_results[1].status = "failed"
    answer.subtask_results[1].claims = [invalid]
    answer.claims.append(Claim(subtask_id="orphan", text="伪造总结。"))
    text = exporter()(answer)
    assert "公司绝对没有风险" not in text and "伪造总结" not in text
    assert "失败" in text


def test_missing_planned_subtask_remains_not_executed():
    answer = result()
    answer.subtask_results.pop()
    text = exporter()(answer)
    assert "q2" in text and "未执行" in text
    assert "没有披露" not in text


def test_decimal_calculation_exports_formula_operands_and_version_without_float_roundtrip():
    answer = result("live")
    answer.query_plan["subtasks"][0]["section_id"] = "metrics"
    answer.subtask_results[0].claims = [Claim(subtask_id="q1", text="营业收入同比为20.00%。",
                                             calculation_ids=["calc-1"], fact_ids=["f1", "f2"])]
    answer.subtask_results[0].calculation_ids = ["calc-1"]
    answer.subtask_results[0].fact_ids = ["f1", "f2"]
    answer.calculations = [{"id": "calc-1", "formula_id": "revenue_yoy", "formula_version": "1.0",
                            "operand_fact_ids": ["f1", "f2"], "result": "20.00000000000000000001",
                            "unit": "percent", "period_label": "2026-01-01/2026-12-31"}]
    facts = [{"id": "f1", "metric_code": "revenue", "value": "120.00000000000000000001",
              "raw_value": "120.00000000000000000001", "unit": "元", "raw_unit": "元",
              "currency": "CNY", "company": "甲公司", "scope": "consolidated", "report_id": "r1",
              "period_start": "2026-01-01", "period_end": "2026-12-31", "source_id": "e1",
              "source_evidence_ids": ["e1"], "validation_status": "verified"},
             {"id": "f2", "metric_code": "revenue", "value": "100", "raw_value": "100", "unit": "元",
              "raw_unit": "元", "currency": "CNY", "company": "甲公司", "scope": "consolidated",
              "report_id": "r1", "period_start": "2025-01-01", "period_end": "2025-12-31",
              "source_id": "e1", "source_evidence_ids": ["e1"], "validation_status": "verified"}]
    text = exporter()(answer, facts=facts)
    assert "20.00000000000000000001" in text
    assert "120.00000000000000000001" in text
    assert "本期收入 / 可比上期收入" in text and "1.0" in text
    assert "f1" in text and "f2" in text and "2025-01-01" in text
    assert "模拟模式" not in text


def test_missing_operand_is_disclosed_without_filling_a_value():
    answer = result()
    answer.subtask_results[0].calculation_ids = ["calc-1"]
    answer.calculations = [{"id": "calc-1", "formula_id": "revenue_yoy", "formula_version": "1.0",
                            "operand_fact_ids": ["missing-fact"], "result": "20", "unit": "percent",
                            "period_label": "2026"}]
    text = exporter()(answer)
    assert "missing-fact" in text and "操作数记录缺失" in text
    assert "补算" not in text


def test_model_metadata_keeps_public_parameters_without_credentials_or_sensitive_url_parts():
    text = exporter()(result(), metadata={
        "embedding": {"model_id": "text-embedding-v3", "dimension": 1024,
                      "endpoint": "https://user:password@service.example/embed?api_key=SECRET_VALUE#frag",
                      "api_key": "SECRET_VALUE", "profile_id": "profile-123"},
        "generation": {"model_id": "deepseek-flash", "thinking": {"type": "disabled"},
                       "headers": {"Authorization": "Bearer AUTH_SECRET"}},
        "chunking": {"chunk_size": 500, "overlap": 80},
        "DEEPSEEK_API_KEY": "TOP_SECRET", "environment": {"PASSWORD": "HIDDEN_SECRET"},
    })
    for secret in ["SECRET_VALUE", "AUTH_SECRET", "TOP_SECRET", "HIDDEN_SECRET", "user:password"]:
        assert secret not in text
    assert "text-embedding-v3" in text and "1024" in text and "profile-123" in text
    assert "deepseek-flash" in text and "500" in text and "80" in text


def test_export_reads_only_referenced_repo_objects_and_never_loads_configuration(tmp_path):
    from annual_report_agent.storage.sqlite_repo import Repository
    repo = Repository(tmp_path / "export.sqlite3")
    repo.put("report", "r1", {"id": "r1", "company": "甲公司", "revision": "2026修订",
                                "path": "C:/private/source.pdf", "api_key": "DO_NOT_EXPORT"})
    repo.put("report", "unselected", {"id": "unselected", "company": "乙公司"})
    repo.put("config", "secret", {"api_key": "NEVER_READ"})
    text = exporter()(result(), repo)
    assert "甲公司" in text and "2026修订" in text
    assert "乙公司" not in text and "DO_NOT_EXPORT" not in text and "private/source" not in text


def test_markdown_in_source_is_escaped_and_unsafe_link_root_is_rejected():
    answer = result()
    answer.citations[0].quote = "<script>alert(1)</script>\n[click](javascript:evil)"
    text = exporter()(answer)
    assert "<script>" not in text and "[click](javascript:" not in text
    with pytest.raises(ValueError):
        exporter()(answer, evidence_base_url="javascript:alert(1)")
