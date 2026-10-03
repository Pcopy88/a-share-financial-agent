"""Run the real serial workflow, SQLite and index with explicitly mock cloud clients."""
import io
import json
import threading
from decimal import Decimal
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from annual_report_agent.config import Settings


class TestTokenizer:
    __test__ = False
    version = "test:characters"

    def count(self, text):
        return len(text)

    def offsets(self, text):
        return [(i, i + 1) for i in range(len(text))]


def pdf_bytes(lines=None):
    stream = io.BytesIO()
    pdf = canvas.Canvas(stream)
    for index, line in enumerate(lines or ["The company makes semiconductor materials.",
                                           "Revenue 2026 120; Revenue 2025 100."]):
        pdf.drawString(50, 700 - index * 25, line)
    pdf.save()
    return stream.getvalue()


def make_runtime(tmp_path, **kwargs):
    from annual_report_agent.runtime import Runtime
    settings = Settings(_env_file=None, mode=kwargs.pop("mode", "mock"),
                        data_dir=tmp_path / "data", output_dir=tmp_path / "out",
                        deepseek_api_key="", dashscope_api_key="", embedding_url="", **kwargs)
    return Runtime(settings, tokenizer=TestTokenizer(), reference_paths=[])


def ready(runtime):
    report = runtime.upload_pdf("fixture.pdf", pdf_bytes())
    runtime.confirm_metadata(report["id"], {"company": "示例公司", "doc_type": "semiannual_report",
        "fiscal_year": 2026, "confirmed": True,
        "report_periods": [{"period_start": "2026-01-01", "period_end": "2026-06-30"}]})
    task = runtime.submit_prepare(report["id"])
    assert runtime.wait_task(task["task_id"], timeout=20)["status"] == "succeeded"
    return report["id"]


def test_prepare_persists_pages_and_requires_identity_confirmation(tmp_path):
    runtime = make_runtime(tmp_path)
    try:
        content = pdf_bytes()
        report = runtime.upload_pdf("../../fixture.pdf", content)
        duplicate = runtime.upload_pdf("same.pdf", content)
        assert report["id"] == duplicate["id"]
        with pytest.raises(ValueError, match="METADATA_UNCONFIRMED"):
            runtime.submit_prepare(report["id"])
        with pytest.raises(ValueError, match="INVALID_METADATA"):
            runtime.confirm_metadata(report["id"], {"path": "C:/other.pdf", "confirmed": True})
        runtime.confirm_metadata(report["id"], {"company": "示例公司", "confirmed": True})
        task = runtime.submit_prepare(report["id"])
        task = runtime.wait_task(task["task_id"], timeout=20)
        assert task["status"] == "succeeded"
        assert [item["node"] for item in task["node_timings"]] == ["N1_acquire", "N2_parse", "N3_index"]
        assert runtime.repo.list("page", report["id"])[0]["raw_text"].startswith("The company")
        assert runtime.repo.active_index(report["id"])["mode"] == "mock"
        assert runtime.repo.get("report", report["id"])["capabilities"]["identity"]["confirmed"]
    finally:
        runtime.close()


def test_query_uses_n4_once_and_serial_subtasks_with_real_page_citations(tmp_path):
    runtime = make_runtime(tmp_path)
    try:
        identifier = ready(runtime)
        submitted = runtime.submit_query({"question": "主营业务是什么？主要材料是什么？",
                                          "report_ids": [identifier]})
        assert runtime.wait_task(submitted["task_id"], 20)["status"] == "succeeded"
        run = runtime.get_run(submitted["run_id"])
        assert run["node_trace"].count("N4_route") == 1
        assert run["node_trace"].count("N9_verify") == 2
        assert len(run["answer"]["subtask_results"]) == 2
        assert run["answer"]["mode"] == "mock"
        assert run["answer"]["citations"]
        assert all(e["index_version"] == run["report_index_versions"][identifier]
                   for e in run["answer"]["citations"])
        assert all(e["page_refs"][0]["pdf_page_number"] == 1
                   for e in run["answer"]["citations"])
        assert run["traces"][0]["task"]["subtask_id"] == "q1"
    finally:
        runtime.close()


def test_financial_fact_parent_ids_become_sent_evidence_ids(tmp_path):
    runtime = make_runtime(tmp_path)
    try:
        identifier = ready(runtime)
        version = runtime.repo.active_index(identifier)["index_version"]
        parent = runtime.repo.list("parent", identifier)[0]
        from annual_report_agent.schemas.finance import FinancialFact
        for year, value in ((2026, "120"), (2025, "100")):
            fact = FinancialFact(id=f"f{year}", metric_code="revenue", value=value,
                raw_value=value, raw_unit="元", company="示例公司", scope="consolidated",
                period_start=f"{year}-01-01", period_end=f"{year}-06-30", report_id=identifier,
                source_id=parent["id"], source_evidence_ids=[parent["id"]],
                validation_status="verified", review_record={"method": "test_fixture", "source": "PDF"},
                provenance={"pdf_page_number": 1, "row_label": "Revenue", "source_sha256": "fixture"})
            runtime.repo.put("fact", f"{version}:{fact.id}", fact, identifier)
        submitted = runtime.submit_query({"question": "2026年上半年收入同比增长多少？",
                                          "report_ids": [identifier]})
        runtime.wait_task(submitted["task_id"], 20)
        answer = runtime.get_run(submitted["run_id"])["answer"]
        assert answer["status"] == "answered", answer
        assert answer["calculations"][0]["result"] == "20.0"
        assert answer["citations"] and all(e["id"] != parent["id"] for e in answer["citations"])
    finally:
        runtime.close()


def test_single_worker_and_restart_recovery(tmp_path):
    runtime = make_runtime(tmp_path)
    gate, started = threading.Event(), threading.Event()
    order = []
    def first():
        order.append("first")
        started.set()
        gate.wait(10)
    try:
        first_task = runtime.submit_callable("test", "first", first)
        assert started.wait(5)
        second_task = runtime.submit_callable("test", "second", lambda: order.append("second"))
        assert order == ["first"]
        gate.set()
        runtime.wait_task(first_task["task_id"], 10)
        runtime.wait_task(second_task["task_id"], 10)
        assert order == ["first", "second"]
        abandoned = runtime.repo.create_task("prepare", "missing")
        runtime.repo.update_task(abandoned["task_id"], "running", .5)
    finally:
        gate.set()
        runtime.close()
    restarted = make_runtime(tmp_path)
    try:
        assert restarted.repo.get_task(abandoned["task_id"])["status"] == "interrupted"
    finally:
        restarted.close()


def test_missing_live_keys_and_mock_budget_are_explicit(tmp_path):
    live = make_runtime(tmp_path, mode="live")
    try:
        with pytest.raises(ValueError, match="CONFIGURATION_REQUIRED"):
            live.submit_query({"question": "业务", "report_ids": ["missing"]})
    finally:
        live.close()
    runtime = make_runtime(tmp_path, max_run_calls=1)
    try:
        identifier = ready(runtime)
        submitted = runtime.submit_query({"question": "业务？风险？", "report_ids": [identifier]})
        runtime.wait_task(submitted["task_id"], 20)
        answer = runtime.get_run(submitted["run_id"])["answer"]
        assert any(s["status"] == "not_executed" for s in answer["subtask_results"])
        assert all(s["status"] != "insufficient_evidence" for s in answer["subtask_results"])
        run = runtime.get_run(submitted["run_id"])
        assert [item["node"] for item in run["node_timings"]] == run["node_trace"]
        assert all(item["latency_seconds"] >= 0 for item in run["node_timings"])
        assert any(item["status"] == "failed" and item["subtask_id"] == "q1"
                   for item in run["node_timings"])
    finally:
        runtime.close()


def test_run_pins_report_metadata_and_new_query_rejects_stale_index(tmp_path):
    runtime = make_runtime(tmp_path)
    gate, started = threading.Event(), threading.Event()
    try:
        identifier = ready(runtime)
        def blocker():
            started.set()
            gate.wait(10)
        runtime.submit_callable("test", "blocker", blocker)
        assert started.wait(5)
        query = {"question": "主营业务是什么？", "report_ids": [identifier],
                 "hard_filters": {"company": "示例公司"}}
        submitted = runtime.submit_query(query)
        runtime.confirm_metadata(identifier, {"company": "新公司", "confirmed": True})
        gate.set()
        runtime.wait_task(submitted["task_id"], 20)
        assert runtime.get_run(submitted["run_id"])["answer"]["status"] == "answered"
        with pytest.raises(ValueError, match="INDEX_STALE"):
            runtime.submit_query(query)
    finally:
        gate.set()
        runtime.close()


def test_global_clarification_and_frozen_plan_use_no_unbounded_loop(tmp_path):
    runtime = make_runtime(tmp_path)
    try:
        identifier = ready(runtime)
        run = runtime.run_query_sync({"question": "一？二？三？四？", "report_ids": [identifier]})
        assert run["answer"]["status"] == "needs_clarification"
        assert run["node_trace"] == ["N1_acquire", "N4_route", "N9_verify"]
        from annual_report_agent.graph.planner import plan_query
        from annual_report_agent.schemas.query import AnalysisRequest
        request = AnalysisRequest(question="业务是什么？", report_ids=[identifier])
        plan = plan_query(request, [runtime.get_report(identifier)], mode="mock")
        run = runtime.run_query_sync(request, execution_options={
            "plan_override": plan, "retrieval_mode": "bm25", "context_expansion": False})
        assert run["traces"][0]["task"]["subtask_id"] == "q1"
        assert all(e["role"] == "primary" for e in run["traces"][0]["evidence"])
    finally:
        runtime.close()


def test_published_facts_are_version_bound_and_frozen_at_submission(tmp_path):
    from annual_report_agent.schemas.finance import FinancialFact
    runtime = make_runtime(tmp_path)
    gate, started = threading.Event(), threading.Event()
    try:
        identifier = ready(runtime)
        parent = runtime.repo.list("parent", identifier)[0]
        fact = FinancialFact(id="published", metric_code="revenue", value="120", raw_value="120",
            company="示例公司", period_start="2026-01-01", period_end="2026-06-30",
            scope="consolidated", report_id=identifier, source_id=parent["id"],
            provenance={"pdf_page_number": 1})
        receipt = runtime.publish_financial_facts(identifier, [fact])
        assert receipt["index_version"] == runtime.repo.active_index(identifier)["index_version"]
        def blocker():
            started.set()
            gate.wait(10)
        runtime.submit_callable("test", "blocker", blocker)
        assert started.wait(5)
        submitted = runtime.submit_query({"question": "收入同比增长多少？", "report_ids": [identifier]})
        runtime.publish_financial_facts(identifier, [fact.model_copy(update={"value": Decimal("999")})])
        assert runtime.get_run(submitted["run_id"])["fact_snapshot"][identifier][0]["value"] == "120"
        with pytest.raises(ValueError, match="FACT_REPORT_MISMATCH"):
            runtime.publish_financial_facts(identifier, [fact.model_copy(update={"report_id": "other"})])
    finally:
        gate.set()
        runtime.close()


def test_second_runtime_cannot_recover_another_live_instance(tmp_path):
    runtime = make_runtime(tmp_path)
    try:
        task = runtime.repo.create_task("prepare", "in-progress")
        runtime.repo.update_task(task["task_id"], "running", .5)
        with pytest.raises(ValueError, match="INSTANCE_ALREADY_RUNNING"):
            make_runtime(tmp_path)
        assert runtime.repo.get_task(task["task_id"])["status"] == "running"
    finally:
        runtime.close()
    restarted = make_runtime(tmp_path)
    try:
        assert restarted.repo.get_task(task["task_id"])["status"] == "interrupted"
    finally:
        restarted.close()


def test_fact_citation_uses_source_row_instead_of_first_equal_number(tmp_path):
    from annual_report_agent.schemas.finance import FinancialFact
    runtime = make_runtime(tmp_path)
    try:
        report = runtime.upload_pdf("rows.pdf", pdf_bytes(["Unrelated 120 100", "Revenue 120 100"]))
        runtime.confirm_metadata(report["id"], {"company": "示例公司", "confirmed": True})
        runtime.wait_task(runtime.submit_prepare(report["id"])["task_id"], 20)
        version = runtime.repo.active_index(report["id"])["index_version"]
        parent = runtime.repo.list("parent", report["id"])[0]
        facts = [FinancialFact(id=f"revenue-{year}", metric_code="revenue", value=value,
            raw_value=value, raw_unit="元", company="示例公司", period_start=f"{year}-01-01", period_end=f"{year}-06-30",
            scope="consolidated", report_id=report["id"], source_id=parent["id"],
            source_evidence_ids=[parent["id"]],
            validation_status="verified", review_record={"method": "fixture"},
            provenance={"pdf_page_number": 1, "row_label": "Revenue"})
            for year, value in ((2026, "120"), (2025, "100"))]
        for fact in facts:
            runtime.repo.put("fact", f"{version}:{fact.id}", fact, report["id"])
        run = runtime.run_query_sync({"question": "2026年上半年收入同比增长多少？", "report_ids": [report["id"]]})
        assert run["answer"]["status"] == "answered", run["answer"]
        assert all(e["quote"] == "Revenue 120 100" for e in run["answer"]["citations"])
    finally:
        runtime.close()


def test_mock_still_validates_generation_profile_allowlist(tmp_path):
    runtime = make_runtime(tmp_path)
    try:
        assert runtime.generation_client_for("default") is None
        with pytest.raises(ValueError, match="MODEL_PROFILE_INVALID"):
            runtime.generation_client_for("unconfigured-arbitrary-model")
    finally:
        runtime.close()


def test_prepare_reparses_old_parser_cache_and_recovers_missing_chunks(tmp_path, monkeypatch):
    import annual_report_agent.runtime as module
    runtime = make_runtime(tmp_path)
    try:
        identifier = ready(runtime)
        saved = runtime.repo.get("parsed", identifier)
        saved["parents"][0]["metadata"]["parser_version"] = "obsolete"
        runtime.repo.put("parsed", identifier, saved, identifier)
        original = module.parse_report
        calls = []
        def tracked(*args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)
        monkeypatch.setattr(module, "parse_report", tracked)
        task = runtime.wait_task(runtime.submit_prepare(identifier)["task_id"], 20)
        assert task["status"] == "succeeded"
        assert len(calls) == 1
        saved = runtime.repo.get("parsed", identifier)
        saved["chunks"] = []
        runtime.repo.put("parsed", identifier, saved, identifier)
        runtime.wait_task(runtime.submit_prepare(identifier)["task_id"], 20)
        assert runtime.repo.get("parsed", identifier)["chunks"]
        assert len(calls) == 1
    finally:
        runtime.close()


def test_all_failed_pages_are_not_reported_as_successful_ocr_prepare(tmp_path, monkeypatch):
    import annual_report_agent.runtime as module
    from annual_report_agent.schemas.report import Page, ParsedReport
    runtime = make_runtime(tmp_path)
    try:
        report = runtime.upload_pdf("fixture.pdf", pdf_bytes())
        runtime.confirm_metadata(report["id"], {"company": "示例公司", "confirmed": True})
        def failed(report, **kwargs):
            report.status = "failed"
            report.page_count = 1
            return ParsedReport(report=report, pages=[Page(report_id=report.id,
                pdf_page_number=1, raw_text="", parse_status="failed", error="fixture parse failure")],
                parents=[], chunks=[], warnings=["PAGE_FAILED:1"])
        monkeypatch.setattr(module, "parse_report", failed)
        task = runtime.wait_task(runtime.submit_prepare(report["id"])["task_id"], 20)
        assert task["status"] == "failed"
        assert task["error_code"] == "PDF_PARSE_FAILED"
        assert runtime.get_report(report["id"])["status"] == "failed"
    finally:
        runtime.close()


def test_runtime_uses_configured_tokenizer_cache_for_initial_and_cached_parse(tmp_path, monkeypatch):
    import annual_report_agent.runtime as module
    runtime = make_runtime(tmp_path)
    runtime.tokenizer = None
    loaded = []
    expected = runtime.settings.data_dir / "tokenizer"
    def load(cache_dir="data/tokenizer", **kwargs):
        loaded.append(cache_dir)
        if cache_dir != expected:
            raise ValueError("TOKENIZER_UNAVAILABLE: wrong cache directory")
        return TestTokenizer()
    monkeypatch.setattr(module.FixedTokenizer, "load", load)
    try:
        identifier = ready(runtime)
        runtime.tokenizer = None
        task = runtime.wait_task(runtime.submit_prepare(identifier)["task_id"], 20)
        assert task["status"] == "succeeded", task
        assert loaded == [expected, expected]
    finally:
        runtime.close()


def test_metadata_edit_is_rejected_while_prepare_is_queued_or_running(tmp_path):
    runtime = make_runtime(tmp_path)
    gate, entered = threading.Event(), threading.Event()
    try:
        report = runtime.upload_pdf("fixture.pdf", pdf_bytes())
        runtime.confirm_metadata(report["id"], {"company": "Original", "confirmed": True})
        build = runtime.index.build
        def blocked(*args, **kwargs):
            entered.set()
            gate.wait(10)
            return build(*args, **kwargs)
        runtime.index.build = blocked
        queued = runtime.repo.create_task("prepare", report["id"], mode="mock")
        with pytest.raises(ValueError, match="REPORT_BUSY"):
            runtime.confirm_metadata(report["id"], {"company": "Changed", "confirmed": True})
        runtime.repo.update_task(queued["task_id"], "interrupted", 0)
        task = runtime.submit_prepare(report["id"])
        assert entered.wait(10)
        with pytest.raises(ValueError, match="REPORT_BUSY"):
            runtime.confirm_metadata(report["id"], {"company": "Changed", "confirmed": True})
        gate.set()
        assert runtime.wait_task(task["task_id"], 20)["status"] == "succeeded"
        runtime.confirm_metadata(report["id"], {"company": "Changed", "confirmed": True})
        assert runtime.get_report(report["id"])["company"] == "Changed"
    finally:
        gate.set()
        runtime.close()


def test_real_jiangfeng_wrapped_parent_profit_row_remains_citable(tmp_path):
    from annual_report_agent.finance.pdf_facts import extract_pdf_facts, verify_against_reference
    from annual_report_agent.ingestion.parser import parse_report
    from annual_report_agent.schemas.report import Report
    manifest = json.loads(Path("evals/sample_manifest.json").read_text(encoding="utf-8-sig"))
    documents = manifest.get("documents", manifest.get("samples", []))
    source = next(item for item in documents if item["id"] == "jiangfeng_semiannual_2026h1")
    path = Path(source["path"])
    if not path.exists():
        pytest.skip("user-provided Jiangfeng PDF not present on this machine")
    runtime = make_runtime(tmp_path)
    try:
        uploaded = runtime.upload_pdf(path.name, path.read_bytes())
        values = {key: source[key] for key in ("company", "doc_type", "fiscal_year", "report_periods")}
        report = Report.model_validate(runtime.confirm_metadata(uploaded["id"], values | {"confirmed": True}))
        parsed = parse_report(report, tokenizer=TestTokenizer(), page_numbers=[62, 63])
        runtime.store_parsed(parsed)
        runtime.index.build(parsed.report, parsed.parents, parsed.chunks)
        reference = json.loads(Path("evals/jiangfeng_2026h1_reference.json").read_text(encoding="utf-8"))
        runtime.publish_financial_facts(report.id, verify_against_reference(extract_pdf_facts(parsed), reference))
        run = runtime.run_query_sync({"question": "2026年上半年归母净利润同比增长多少？", "report_ids": [report.id]})
        assert run["answer"]["status"] == "answered", run["answer"]
        assert all("归属于母公司股东的净利润" in item["quote"] and item["page_refs"][0]["pdf_page_number"] == 63
                   for item in run["answer"]["citations"])
        assert round(Decimal(run["answer"]["calculations"][0]["result"]), 2) == Decimal("113.61")
    finally:
        runtime.close()


def test_wrapped_fact_match_does_not_cross_the_next_table_row(tmp_path):
    from annual_report_agent.schemas.finance import FinancialFact
    runtime = make_runtime(tmp_path)
    try:
        fact = FinancialFact(id="f", metric_code="net_profit", value="120", raw_value="120",
            company="Example", scope="consolidated", period_end="2026-06-30", report_id="r",
            source_id="p", provenance={"row_label": "Net profit\n(after tax)"})
        rows = [["Net profit\n(after tax)", "120", "100"], ["Other gains", "120", "100"]]
        assert runtime._wrapped_fact_row(fact, rows, 0,
            "Net profit\nOther gains 120 100\n(after tax)\n") is None
    finally:
        runtime.close()
