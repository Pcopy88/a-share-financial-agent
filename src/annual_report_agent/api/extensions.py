"""Reports, public acquisition and evaluation routes using the single runtime queue."""

import json
import re
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from annual_report_agent.evaluation.runner import EXPERIMENTS, EvaluationRunner
from annual_report_agent.generation.profiles import public_profiles
from annual_report_agent.schemas.answer import AnswerResult


class SearchRequest(BaseModel):
    stock_code: str = Field(pattern=r"^\d{6}$")
    year: int = Field(ge=1990, le=2100)


class EvaluationRequest(BaseModel):
    dataset: str
    experiment: str = "M2"
    split: str = "development"
    repetitions: int = Field(1, ge=1, le=3)
    frozen_run_id: str | None = None
    model_profile: str = "default"


def attach_routes(app):
    runtime = app.state.runtime
    eval_directory = runtime.settings.results_dir / "evaluations"

    def require(kind, identifier):
        value = runtime.repo.get(kind, identifier)
        if value is None:
            raise HTTPException(404, detail="资源不存在")
        return value

    @app.get("/api/model-profiles")
    def profiles():
        return public_profiles(runtime.settings)

    @app.get("/api/reports/{report_id}/facts")
    def facts(report_id: str):
        require("report", report_id)
        return {"facts": runtime.repo.list("fact", report_id)}

    @app.get("/api/runs/{run_id}/export")
    def export(run_id: str):
        from annual_report_agent.generation.report_export import export_markdown
        run = runtime.get_run(run_id)
        if not run or not run.get("answer"):
            raise HTTPException(409, detail="尚无可导出的分析结果")
        sent_facts = {fact["id"]: fact for trace in run.get("traces", [])
                      for fact in trace.get("facts", [])}
        text = export_markdown(AnswerResult.model_validate(run["answer"]), repo=runtime.repo,
            metadata=run, facts=list(sent_facts.values()),
            reports=[manifest["report_metadata"] for manifest in run.get("manifests", [])])
        return Response(text, media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{run_id}.md"'})

    @app.post("/api/announcements/search")
    def search(request: SearchRequest):
        identifier = "search-" + uuid4().hex

        def job():
            from annual_report_agent.ingestion.acquisition import search_announcements
            candidates = search_announcements(request.stock_code, request.year)
            for candidate in candidates:
                runtime.repo.put("announcement", candidate.id, candidate)
            return {"candidate_ids": [item.id for item in candidates]}

        return runtime.submit_callable("announcements", identifier, job)

    @app.get("/api/announcements")
    def announcements():
        return {"candidates": runtime.repo.list("announcement")}

    @app.post("/api/announcements/{candidate_id}/download")
    def download(candidate_id: str):
        candidate = require("announcement", candidate_id)

        def job():
            from annual_report_agent.ingestion.acquisition import (
                AnnouncementCandidate,
                download_candidate,
            )
            selected = AnnouncementCandidate.model_validate(candidate)
            result = download_candidate(selected, runtime.settings.runtime_dir / "downloads", selected=True)
            content = Path(result.path).read_bytes()
            report = runtime.upload_pdf(selected.title + ".pdf", content, metadata={
                "company": selected.company, "stock_code": selected.stock_code,
                "fiscal_year": selected.report_year, "doc_type": "annual_report",
                "revision": selected.revision, "disclosure_date": selected.disclosure_date.isoformat()})
            runtime.repo.put("acquisition", report["id"], result, report["id"])
            return report

        return runtime.submit_callable("download", candidate_id, job)

    @app.post("/api/reports/{report_id}/financial-import")
    def financial_import(report_id: str):
        report = require("report", report_id)
        if not report.get("stock_code") or not report.get("fiscal_year") or report.get("doc_type") != "annual_report":
            raise HTTPException(422, detail="当前 AKShare 适配只支持已确认股票代码及年度的全年年报")
        manifest = runtime.repo.active_index(report_id)
        fields = {"company", "stock_code", "doc_type", "fiscal_year", "report_periods",
                  "inquiry_round", "revision", "disclosure_date"}

        def ensure_current():
            active = runtime.repo.active_index(report_id)
            current = runtime.get_report(report_id)
            if not manifest or not active or active["index_version"] != manifest["index_version"]:
                raise ValueError("INDEX_STALE: selected index changed or is missing")
            if any(current.get(key) != report.get(key)
                   or current.get(key) != manifest["report_metadata"].get(key) for key in fields):
                raise ValueError("INDEX_STALE: metadata changed; rebuild before importing facts")

        try:
            ensure_current()
        except ValueError as error:
            raise HTTPException(409, detail=str(error)) from error

        def job():
            from annual_report_agent.finance.akshare_adapter import (
                fetch_financial_facts,
                reconcile_with_pdf,
            )
            from annual_report_agent.schemas.finance import FinancialFact
            ensure_current()
            prior = {item["id"]: FinancialFact.model_validate(item)
                     for item in runtime._version_facts(report_id, manifest["index_version"])
                     if item.get("provenance", {}).get("source_type") == "pdf"}
            result = fetch_financial_facts(report["stock_code"], report["fiscal_year"],
                company=report["company"], report_id=report_id,
                snapshot_dir=runtime.settings.runtime_dir / "snapshots")
            ensure_current()
            reconciled = reconcile_with_pdf(result.facts, list(prior.values()))
            for fact in reconciled.facts:
                runtime.repo.put("fact", fact.id, fact, report_id)
            published = runtime.publish_financial_facts(report_id, reconciled.facts)
            return published | {"errors": result.errors, "warnings": result.warnings,
                                "reconciliation": reconciled.diagnostics}

        return runtime.submit_callable("financial_import", report_id, job)

    @app.get("/api/evaluation-datasets")
    def datasets():
        return {"datasets": sorted(path.name for path in Path("evals").glob("*.jsonl"))}

    @app.post("/api/evaluations")
    def evaluate(request: EvaluationRequest):
        registry = datasets()["datasets"]
        if request.dataset not in registry or request.experiment not in EXPERIMENTS:
            raise HTTPException(422, detail="未知题集或实验")
        frozen = None
        if request.frozen_run_id:
            if not re.fullmatch(r"eval-[a-f0-9]{32}", request.frozen_run_id):
                raise HTTPException(422, detail="冻结编号格式错误")
            require("evaluation", request.frozen_run_id)
            frozen = eval_directory / request.frozen_run_id / "frozen.json"
            if not frozen.is_file():
                raise HTTPException(422, detail="该实验没有可复用的冻结输入")
        if request.experiment in {"M1", "C0", "C1", "Q0", "Q1"} and frozen is None:
            raise HTTPException(422, detail="先运行 M2 并选择其冻结输入")
        resource_id = "evaluation-job-" + uuid4().hex

        def job():
            result = EvaluationRunner(runtime, eval_directory).run(Path("evals") / request.dataset,
                experiment=request.experiment, split=request.split, repetitions=request.repetitions,
                frozen_path=frozen, id_mapping=sample_id_mapping(runtime), model_profile=request.model_profile)
            runtime.repo.put("evaluation", result["id"], result)
            runtime.repo.put("evaluation_job", resource_id, {"eval_run_id": result["id"]})
            return result

        return runtime.submit_callable("evaluation", resource_id, job)

    @app.get("/api/evaluations")
    def evaluations():
        return {"evaluations": runtime.repo.list("evaluation")}

    @app.get("/api/evaluations/{eval_run_id}")
    def evaluation(eval_run_id: str):
        return require("evaluation", eval_run_id)

    @app.get("/api/evaluations/{eval_run_id}/rows")
    def rows(eval_run_id: str):
        require("evaluation", eval_run_id)
        if not re.fullmatch(r"eval-[a-f0-9]{32}", eval_run_id):
            raise HTTPException(422, detail="评测编号格式错误")
        path = eval_directory / eval_run_id / "rows.jsonl"
        if not path.is_file():
            raise HTTPException(404, detail="逐题结果文件不存在")
        return Response(path.read_bytes(), media_type="application/x-ndjson")


def sample_id_mapping(runtime):
    """Resolve stable sample aliases to uploaded SHA-based report IDs."""
    reports = {item["sha256"]: item["id"] for item in runtime.repo.list("report")}
    mapping = {}
    for path in Path("evals").glob("*sample_manifest.json"):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        samples = data if isinstance(data, list) else data.get("samples", data.get("documents", []))
        for sample in samples:
            if sample.get("sha256") in reports:
                mapping[sample.get("id", sample.get("report_id"))] = reports[sample["sha256"]]
    return mapping
