from pathlib import Path
from urllib.parse import urlsplit

import httpx
from streamlit.testing.v1 import AppTest


def test_four_pages_render_and_backend_outage_is_explained(monkeypatch):
    def request(method, url, **kwargs):
        path = url.split(":8000")[-1]
        data = {"/api/health": {"mode": "mock"}, "/api/reports": {"reports": []},
                "/api/announcements": {"candidates": []},
                "/api/model-profiles": {"default": "flash", "profiles": {"flash": {}}},
                "/api/evaluation-datasets": {"datasets": ["development.jsonl"]},
                "/api/evaluations": {"evaluations": []}}[path]
        return httpx.Response(200, json=data)
    monkeypatch.setattr(httpx, "request", request)
    path = Path(__file__).parents[2] / "src/annual_report_agent/ui/app.py"
    app = AppTest.from_file(str(path)).run(timeout=15)
    assert not app.exception
    for page in ["分析问答", "证据与指标", "评测与报告", "资料管理"]:
        app.sidebar.radio[0].set_value(page).run(timeout=15)
        assert not app.exception
    def offline(*args, **kwargs):
        raise httpx.ConnectError("connection refused")
    monkeypatch.setattr(httpx, "request", offline)
    app.run(timeout=15)
    assert not app.exception
    assert "操作未完成" in app.error[0].value


def test_numeric_answer_opens_only_used_verified_fact_source_page(monkeypatch):
    def fact(identifier, raw_value, page, **updates):
        return {"id": identifier, "metric_code": "revenue", "raw_value": raw_value,
                "value": "12000", "raw_unit": "万元", "unit": "元", "company": "测试公司",
                "period_start": "2026-01-01", "period_end": "2026-06-30", "scope": "consolidated",
                "report_id": "r1", "validation_status": "verified",
                "provenance": {"page_refs": [{"pdf_page_number": page, "printed_page_label": "62"}]},
                **updates}

    calculation = {"id": "c1", "result": "20.00", "unit": "percent",
                   "operand_fact_ids": ["used", "pending", "foreign", "not-sent"]}
    used = fact("used", "1,234.56", 2)
    direct = fact("direct", "789.01", 3, scope="parent",
                  provenance={"pdf_page_number": 3})
    run = {"run_id": "run1", "request": {"report_ids": ["r1"]},
           "answer": {"status": "answered", "mode": "mock", "evidence_level": "high",
                      "citations": [], "calculations": [calculation], "subtask_results": [
                          {"subtask_id": "q1", "status": "answered", "warnings": [],
                           "claims": [{"text": "同比增长20%。", "evidence_ids": [], "fact_ids": ["direct"]}]},
                          {"subtask_id": "q2", "status": "failed", "warnings": [],
                           "claims": [{"text": "", "evidence_ids": [], "fact_ids": ["failed"]}]}]},
           "traces": [{"task": {"subtask_id": "q1"}, "calculations": [calculation], "facts": [
               used, direct, fact("unused", "UNUSED-VALUE", 4),
               fact("pending", "PENDING-VALUE", 4, validation_status="pending"),
               fact("foreign", "FOREIGN-VALUE", 4, report_id="r2")]},
               {"task": {"subtask_id": "q2"}, "calculations": [],
                "facts": [fact("failed", "FAILED-VALUE", 4)]}]}
    report = {"id": "r1", "company": "测试公司", "filename": "sample.pdf", "status": "ready",
              "page_count": 5, "doc_type": "semiannual_report"}
    seen_pages = []

    def request(method, url, **kwargs):
        data = {"/api/health": {"mode": "mock"},
                "/api/reports": {"reports": [report, report | {"id": "r2"}]},
                "/api/model-profiles": {"default": "flash", "profiles": {"flash": {}}},
                "/api/runs/run1": run, "/api/reports/r1/facts": {"facts": []}}[urlsplit(url).path]
        return httpx.Response(200, json=data)

    def get(url, **kwargs):
        seen_pages.append(urlsplit(url).path)
        return httpx.Response(404)

    monkeypatch.setattr(httpx, "request", request)
    monkeypatch.setattr(httpx, "get", get)
    path = Path(__file__).parents[2] / "src/annual_report_agent/ui/app.py"
    app = AppTest.from_file(str(path))
    app.session_state["page"] = "分析问答"
    app.session_state["run_id"] = "run1"
    app.run(timeout=15)
    assert not app.exception and not app.error
    text = "\n".join(item.value for kind in (app.text, app.caption, app.markdown) for item in kind)
    assert "1,234.56" in text and "789.01" in text and "万元" in text
    assert "2026-01-01" in text and "2026-06-30" in text
    assert "合并" in text and "母公司" in text
    assert all(value not in text for value in ("UNUSED-VALUE", "PENDING-VALUE", "FOREIGN-VALUE", "FAILED-VALUE"))
    source_buttons = [button for button in app.button if "打开财务原值 PDF" in button.label]
    assert len(source_buttons) == 2
    next(button for button in source_buttons if "第 2 页" in button.label).click().run(timeout=15)
    assert not app.exception and not app.error
    assert app.session_state["page"] == "证据与指标"
    assert app.session_state["evidence_report"] == "r1"
    assert app.session_state["evidence_page"] == 2
    assert seen_pages == ["/api/reports/r1/pages/2"]
