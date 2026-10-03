"""FastAPI routes accept identities, never arbitrary local file paths."""
import json
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from annual_report_agent.config import Settings
from annual_report_agent.runtime import Runtime, RuntimeErrorInfo
from annual_report_agent.schemas.query import AnalysisRequest


def create_app(settings=None, runtime=None):
    service = runtime or Runtime(settings or Settings())

    @asynccontextmanager
    async def lifespan(app):
        yield
        service.close()

    app = FastAPI(title="A股财务资料分析", lifespan=lifespan)
    app.state.runtime = service

    @app.exception_handler(RuntimeErrorInfo)
    async def application_error(request, error):
        code = 404 if error.code == "NOT_FOUND" else 422 if error.code in {
            "INVALID_METADATA", "INVALID_PDF", "INVALID_SOURCE_PATH"} else 409
        return JSONResponse(status_code=code, content={"error": {
            "code": error.code, "message": error.message, "details": error.details}})

    @app.exception_handler(RequestValidationError)
    @app.exception_handler(ValidationError)
    async def validation_error(request, error):
        return JSONResponse(status_code=422, content={"error": {
            "code": "INVALID_REQUEST", "message": "请求字段不符合接口约定。"}})

    @app.get("/api/health")
    def health():
        return service.health()

    @app.post("/api/reports/upload", status_code=201)
    async def upload(file: Annotated[UploadFile, File()], metadata: Annotated[str | None, Form()] = None):
        limit = service.settings.max_upload_mb * 1024 * 1024
        content = await file.read(limit + 1)
        try:
            fields = json.loads(metadata) if metadata else None
        except ValueError as error:
            raise RuntimeErrorInfo("INVALID_METADATA", "元数据必须为 JSON 对象。") from error
        report = service.upload_pdf(file.filename or "upload.pdf", content, fields)
        task = service.repo.create_task("upload", report["id"], mode=service.settings.mode)
        service.repo.update_task(task["task_id"], "running", 0)
        service.repo.update_task(task["task_id"], "succeeded", 1)
        return report | {"report_id": report["id"], "task_id": task["task_id"]}

    @app.get("/api/reports")
    def reports():
        return {"reports": service.repo.list("report")}

    @app.post("/api/reports/generate", status_code=202)
    def generate(request: AnalysisRequest):
        return service.submit_report(request)

    @app.get("/api/reports/{report_id}")
    def report_detail(report_id: str):
        return service.get_report(report_id)

    @app.patch("/api/reports/{report_id}/metadata")
    def metadata(report_id: str, updates: dict):
        return service.confirm_metadata(report_id, updates)

    @app.post("/api/reports/{report_id}/prepare", status_code=202)
    def prepare(report_id: str):
        return service.submit_prepare(report_id)

    @app.get("/api/reports/{report_id}/pages/{number}")
    def page(report_id: str, number: int, format: Literal["png", "json"] = Query("png")):
        if format == "json":
            return service.page(report_id, number)
        return Response(service.page(report_id, number, image=True), media_type="image/png")

    @app.get("/api/tasks/{task_id}")
    def task_detail(task_id: str):
        task = service.repo.get_task(task_id)
        if task is None:
            raise RuntimeErrorInfo("NOT_FOUND", "任务不存在。")
        return task

    @app.post("/api/query", status_code=202)
    @app.post("/api/analysis", status_code=202)
    def query(request: AnalysisRequest):
        return service.submit_query(request)

    @app.get("/api/runs/{run_id}")
    @app.get("/api/analysis/{run_id}")
    def run_detail(run_id: str):
        return service.get_run(run_id)

    @app.get("/api/runs/{run_id}/answer")
    def answer(run_id: str):
        run = service.get_run(run_id)
        return {"run_id": run_id, "status": run["status"], "answer": run["answer"]}

    try:
        from annual_report_agent.api.extensions import attach_routes
    except ModuleNotFoundError as error:
        if error.name != "annual_report_agent.api.extensions":
            raise
    else:
        attach_routes(app)
    return app
