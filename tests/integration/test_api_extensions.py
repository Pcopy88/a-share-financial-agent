import io
from types import SimpleNamespace

from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from annual_report_agent.api.app import create_app
from annual_report_agent.config import Settings
from annual_report_agent.runtime import Runtime


class Counter:
    version = "test:characters"
    def count(self, text):
        return len(text)
    def offsets(self, text):
        return [(i, i+1) for i in range(len(text))]


def test_extensions_export_and_validation_against_actual_runtime(tmp_path):
    settings = Settings(_env_file=None, mode="mock", data_dir=tmp_path / "data",
                        output_dir=tmp_path / "out")
    runtime = Runtime(settings, tokenizer=Counter(), reference_paths=[])
    stream = io.BytesIO()
    pdf = canvas.Canvas(stream)
    pdf.drawString(50, 750, "Revenue arises from semiconductor products.")
    pdf.save()
    with TestClient(create_app(runtime=runtime)) as client:
        assert "deepseek_flash" in client.get("/api/model-profiles").json()["profiles"]
        assert client.post("/api/evaluations", json={"dataset": "../../secret.jsonl"}).status_code == 422
        assert client.get("/api/announcements").json() == {"candidates": []}
        assert client.post("/api/announcements/unknown/download").status_code == 404
        response = client.post("/api/reports/upload", files={"file": ("fixture.pdf", stream.getvalue())})
        report_id = response.json()["report_id"]
        client.patch(f"/api/reports/{report_id}/metadata", json={"company": "演示公司", "confirmed": True})
        task = client.post(f"/api/reports/{report_id}/prepare").json()
        assert runtime.wait_task(task["task_id"], 20)["status"] == "succeeded"
        assert client.get(f"/api/reports/{report_id}/facts").json() == {"facts": []}
        response = client.post("/api/query", json={"question": "主营业务是什么？", "report_ids": [report_id]})
        task = response.json()
        runtime.wait_task(task["task_id"], 20)
        markdown = client.get(f"/api/runs/{task['run_id']}/export")
        assert markdown.status_code == 200
        assert "mock" in markdown.text
        assert "secret" not in markdown.text
        assert client.post(f"/api/reports/{report_id}/financial-import").status_code == 422


def test_financial_import_uses_only_active_version_facts(tmp_path, monkeypatch):
    from annual_report_agent.schemas.finance import FinancialFact
    settings = Settings(_env_file=None, mode="mock", data_dir=tmp_path / "data")
    runtime = Runtime(settings, tokenizer=Counter(), reference_paths=[])
    report = {"id": "r", "company": "Current", "stock_code": "600001", "fiscal_year": 2024,
              "doc_type": "annual_report", "report_periods": []}
    runtime.repo.put("report", "r", report)
    runtime.repo.save_build("active", "r", ["c"], "profile", "mock", report_metadata=report)
    runtime.repo.mark_chunk_encoded("active", "c")
    runtime.repo.publish_index("active")
    current = FinancialFact(id="same-cell", metric_code="revenue", value="10", raw_value="10",
        scope="consolidated", company="Current",
        period_start="2024-01-01", period_end="2024-12-31", report_id="r", source_id="parent",
        provenance={"source_type": "pdf"})
    runtime.repo.put("fact", "active:same-cell", current, "r")
    runtime.repo.put("fact", "zz-old:same-cell", current.model_copy(update={"company": "Previous"}), "r")
    monkeypatch.setattr("annual_report_agent.finance.akshare_adapter.fetch_financial_facts",
        lambda *a, **kw: SimpleNamespace(facts=[], errors=[], warnings=[]))
    with TestClient(create_app(runtime=runtime)) as client:
        task = client.post("/api/reports/r/financial-import").json()
        assert runtime.wait_task(task["task_id"], 20)["status"] == "succeeded"
        assert runtime.repo.get("fact", "active:same-cell")["company"] == "Current"
