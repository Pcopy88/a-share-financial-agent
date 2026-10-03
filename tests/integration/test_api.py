import io

from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from annual_report_agent.config import Settings


def test_api_upload_confirm_prepare_query_and_safe_errors(tmp_path):
    from annual_report_agent.api.app import create_app
    from annual_report_agent.runtime import Runtime
    class Counter:
        version = "test:chars"
        def count(self, text):
            return len(text)
        def offsets(self, text):
            return [(i, i + 1) for i in range(len(text))]
    settings = Settings(_env_file=None, mode="mock", data_dir=tmp_path / "data")
    runtime = Runtime(settings, tokenizer=Counter(), reference_paths=[])
    output = io.BytesIO()
    pdf = canvas.Canvas(output)
    pdf.drawString(50, 700, "Products and business source text")
    pdf.save()
    with TestClient(create_app(runtime=runtime)) as client:
        assert client.get("/api/health").json()["mode"] == "mock"
        response = client.post("/api/reports/upload", files={"file": ("fixture.pdf", output.getvalue())})
        assert response.status_code == 201
        report_id = response.json()["report_id"]
        assert client.get("/api/reports").json()["reports"][0]["id"] == report_id
        assert client.post(f"/api/reports/{report_id}/prepare").status_code == 409
        assert client.patch(f"/api/reports/{report_id}/metadata", json={
            "company": "示例公司", "confirmed": True}).status_code == 200
        response = client.post(f"/api/reports/{report_id}/prepare")
        assert response.status_code == 202
        runtime.wait_task(response.json()["task_id"], 20)
        page = client.get(f"/api/reports/{report_id}/pages/1?format=json")
        assert "Products" in page.json()["raw_text"]
        image = client.get(f"/api/reports/{report_id}/pages/1")
        assert image.status_code == 200 and image.headers["content-type"] == "image/png"
        assert client.get(f"/api/reports/{report_id}/pages/999").status_code == 404
        response = client.post("/api/analysis", json={"question": "业务是什么？", "report_ids": [report_id]})
        assert response.status_code == 202
        runtime.wait_task(response.json()["task_id"], 20)
        result = client.get("/api/analysis/" + response.json()["run_id"]).json()
        assert result["answer"]["mode"] == "mock"
        assert client.get("/api/reports/unknown").json()["error"]["code"] == "NOT_FOUND"
        forbidden = client.patch(f"/api/reports/{report_id}/metadata", json={"path": "C:/secret.pdf"})
        assert forbidden.status_code == 422


def test_export_reads_actual_run_operands_from_versioned_facts(tmp_path):
    from annual_report_agent.api.app import create_app
    from annual_report_agent.runtime import Runtime
    from annual_report_agent.schemas.finance import FinancialFact
    class Counter:
        version = "test:chars"
        def count(self, text):
            return len(text)
        def offsets(self, text):
            return [(i, i + 1) for i in range(len(text))]
    runtime = Runtime(Settings(_env_file=None, mode="mock", data_dir=tmp_path / "data"),
                      tokenizer=Counter(), reference_paths=[])
    output = io.BytesIO()
    pdf = canvas.Canvas(output)
    pdf.drawString(50, 700, "Revenue 120 100")
    pdf.save()
    with TestClient(create_app(runtime=runtime)) as client:
        report = runtime.upload_pdf("financial-fixture.pdf", output.getvalue())
        runtime.confirm_metadata(report["id"], {"company": "示例公司", "doc_type": "semiannual_report",
            "fiscal_year": 2026, "confirmed": True,
            "report_periods": [{"period_start": "2026-01-01", "period_end": "2026-06-30"}]})
        runtime.wait_task(runtime.submit_prepare(report["id"])["task_id"], 20)
        parent = runtime.repo.list("parent", report["id"])[0]
        facts = [FinancialFact(id=f"f{year}", metric_code="revenue", value=value, raw_value=value,
            raw_unit="元", company="示例公司", scope="consolidated", period_start=f"{year}-01-01",
            period_end=f"{year}-06-30", report_id=report["id"], source_id=parent["id"],
            source_evidence_ids=[parent["id"]], validation_status="verified",
            review_record={"method": "synthetic_test_fixture"},
            provenance={"pdf_page_number": 1, "row_label": "Revenue"})
            for year, value in ((2026, "120"), (2025, "100"))]
        runtime.publish_financial_facts(report["id"], facts)
        response = client.post("/api/analysis", json={"question": "2026年上半年收入同比增长多少？",
                                                     "report_ids": [report["id"]]})
        submitted = response.json()
        runtime.wait_task(submitted["task_id"], 20)
        assert runtime.get_run(submitted["run_id"])["answer"]["status"] == "answered"
        assert all(runtime.repo.get("fact", fact.id) is None for fact in facts)
        exported = client.get(f"/api/runs/{submitted['run_id']}/export")
        assert exported.status_code == 200
        assert "操作数记录缺失" not in exported.text
        assert "120元" in exported.text and "100元" in exported.text
        assert "2026-01-01" in exported.text and "2025-01-01" in exported.text
