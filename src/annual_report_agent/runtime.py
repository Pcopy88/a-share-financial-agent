"""Single-worker application runtime. Cloud simulation is explicit and isolated."""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import httpx

from annual_report_agent.config import Settings
from annual_report_agent.finance.pdf_facts import extract_pdf_facts, verify_against_reference
from annual_report_agent.finance.service import select_and_calculate
from annual_report_agent.generation.cloud_client import DeepSeekClient
from annual_report_agent.generation.composer import compose
from annual_report_agent.generation.verifier import VerificationContext, aggregate, verify
from annual_report_agent.graph.builder import build_workflow
from annual_report_agent.graph.planner import plan_query
from annual_report_agent.graph.state import NODE_NAMES
from annual_report_agent.ingestion.chunker import chunk_parents
from annual_report_agent.ingestion.indexer import Candidate, KnowledgeIndex
from annual_report_agent.ingestion.parser import PARSER_VERSION, parse_report
from annual_report_agent.ingestion.tokenizer import FixedTokenizer
from annual_report_agent.retrieval.context import evidence_id, expand_context
from annual_report_agent.retrieval.embedding_client import EmbeddingClient, MockEmbeddingClient
from annual_report_agent.schemas.answer import AnswerDraft, Evidence, SubTaskResult
from annual_report_agent.schemas.finance import Calculation, FinancialFact
from annual_report_agent.schemas.model import EmbeddingProfile
from annual_report_agent.schemas.query import AnalysisRequest, QueryPlan, SubTaskPlan
from annual_report_agent.schemas.report import ParsedReport, Report
from annual_report_agent.storage.files import render_page, save_pdf
from annual_report_agent.storage.process_lock import RuntimeLock
from annual_report_agent.storage.sqlite_repo import Repository, now


class RuntimeErrorInfo(ValueError):
    def __init__(self, code: str, message: str, details=None):
        self.code, self.message, self.details = code, message, details
        super().__init__(f"{code}: {message}")


def _error(error):
    code = getattr(error, "code", None)
    if isinstance(code, str) and code.strip():
        return {"code": code.strip(), "message": str(getattr(error, "message", str(error)))}
    text = str(error)
    prefix = text.split(":", 1)[0]
    if re.fullmatch(r"[A-Z][A-Z_]+", prefix):
        return {"code": prefix, "message": text}
    return {"code": "INTERNAL_ERROR", "message": "执行失败，请查看本地运行记录并重试。"}


def _dict(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else dict(value)


class Runtime:
    def __init__(self, settings: Settings, *, tokenizer=None, reference_paths=None, reranker=None):
        self.settings = settings
        self.settings.runtime_dir.mkdir(parents=True, exist_ok=True)
        self._instance_lock = RuntimeLock(self.settings.runtime_dir).acquire()
        try:
            self._initialize(tokenizer, reference_paths, reranker)
        except Exception:
            self._release_instance_lock()
            raise

    def _release_instance_lock(self):
        if self._instance_lock is not None:
            self._instance_lock.release()
            self._instance_lock = None

    def _initialize(self, tokenizer, reference_paths, reranker):
        settings = self.settings
        self.repo = Repository(settings.runtime_dir / "application.sqlite")
        self.repo.recover_interrupted()
        for run in self.repo.list("run"):
            if run.get("status") in {"queued", "running"}:
                run.update(status="interrupted", error_code="PROCESS_RESTARTED")
                self.repo.put("run", run["run_id"], run)
        self.tokenizer, self.reranker = tokenizer, reranker
        reference_dir = Path(__file__).resolve().parents[2] / "evals"
        defaults = [reference_dir / name for name in (
            "jiangfeng_2026h1_reference.json", "moutai_2023_reference.json",
            "moutai_2024_reference.json", "gree_2023_reference.json", "gree_2024_reference.json")]
        self.reference_paths = ([path for path in defaults if path.exists()]
                                if reference_paths is None else reference_paths)
        self._index = None
        self._generation_client = None
        self._profile_clients = {}
        self._http_clients = []
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="annual-report-worker")
        self._jobs, self._futures = {}, {}
        self._local = threading.local()
        self._facts_lock = threading.RLock()
        self._report_lock = threading.RLock()
        self._closed = False

    def _ensure_cloud(self):
        if self.settings.mode == "live" and self.settings.missing_cloud_config():
            raise RuntimeErrorInfo("CONFIGURATION_REQUIRED", "请先在本机填写云端环境配置。",
                                   self.settings.missing_cloud_config())

    def _reserve_call(self, estimated_tokens=0):
        budget = getattr(self._local, "budget", None)
        if budget is None:
            return
        if (budget["calls"] + 1 > self.settings.max_run_calls
                or budget["reserved_tokens"] + estimated_tokens > self.settings.max_run_tokens):
            raise RuntimeErrorInfo("RUN_BUDGET_EXCEEDED", "本次调用或 token 预算已耗尽。")
        budget["calls"] += 1
        budget["reserved_tokens"] += estimated_tokens

    def _before_http(self, request):
        # Called for each HTTP attempt, including the clients' bounded retries.
        payload = json.loads(request.content)
        estimate = len(request.content.decode("utf-8")) + int(payload.get("max_tokens", 0))
        self._reserve_call(estimate)

    @property
    def index(self):
        if self._index is None:
            if self.settings.mode == "mock":
                embedder = MockEmbeddingClient()
            else:
                self._ensure_cloud()
                transport = httpx.Client(timeout=self.settings.request_timeout,
                                         event_hooks={"request": [self._before_http]})
                self._http_clients.append(transport)
                embedder = EmbeddingClient(self.settings.dashscope_api_key.get_secret_value(),
                    EmbeddingProfile(endpoint=self.settings.embedding_url,
                                     model_id=self.settings.embedding_model), http_client=transport)
            self._index = KnowledgeIndex(self.repo, self.settings.runtime_dir / "vectors", embedder)
        return self._index

    def _generator(self):
        return self.generation_client_for("default")

    def generation_client_for(self, profile="default"):
        self._ensure_cloud()
        try:
            from annual_report_agent.generation.profiles import create_client, effective_profile
        except ModuleNotFoundError as error:
            if error.name != "annual_report_agent.generation.profiles":
                raise
        else:
            try:
                cache_key = effective_profile(self.settings, profile)["sha256"]
                if cache_key not in self._profile_clients:
                    client = create_client(self.settings, profile)
                    if client is not None:
                        client._http.event_hooks.setdefault("request", []).append(self._before_http)
                    self._profile_clients[cache_key] = client
            except ValueError as error:
                raise RuntimeErrorInfo("MODEL_PROFILE_INVALID", "模型档案不存在或配置不完整。") from error
            return self._profile_clients[cache_key]
        if self.settings.mode == "mock":
            if profile != "default":
                raise RuntimeErrorInfo("MODEL_PROFILE_INVALID", "模型档案不存在。")
            return None
        if self._generation_client is None:
            transport = httpx.Client(timeout=self.settings.request_timeout,
                                     event_hooks={"request": [self._before_http]})
            self._http_clients.append(transport)
            self._generation_client = DeepSeekClient(
                self.settings.deepseek_api_key.get_secret_value(), self.settings.deepseek_base_url,
                self.settings.deepseek_model, http_client=transport)
        return self._generation_client

    def health(self):
        return {"status": "ok", "mode": self.settings.mode,
                "missing_cloud_config": self.settings.missing_cloud_config(),
                "cloud_verified": False, "queue_workers": 1}

    def get_report(self, identifier):
        value = self.repo.get("report", identifier)
        if value is None:
            raise RuntimeErrorInfo("NOT_FOUND", "资料不存在。")
        return value

    def upload_pdf(self, filename: str, content: bytes, metadata=None):
        try:
            path, digest = save_pdf(self.settings.runtime_dir / "pdfs", filename, content,
                                    max_bytes=self.settings.max_upload_mb * 1024 * 1024)
        except ValueError as error:
            raise RuntimeErrorInfo("INVALID_PDF", str(error)) from error
        identifier = "r-" + digest[:24]
        existing = self.repo.get("report", identifier)
        if existing:
            return existing
        report = Report(id=identifier, sha256=digest, path=str(path),
                        filename=filename.replace("\\", "/").rsplit("/", 1)[-1],
                        capabilities={"identity": {"confirmed": False}})
        self.repo.put("report", identifier, report)
        if metadata:
            return self.confirm_metadata(identifier, metadata)
        return _dict(report)

    def confirm_metadata(self, identifier, updates):
        with self._report_lock:
            if any(task["resource_type"] == "prepare" and task["resource_id"] == identifier
                   and task["status"] in {"queued", "running"} for task in self.repo.list("task")):
                raise RuntimeErrorInfo("REPORT_BUSY", "资料正在准备，请完成后再修改身份信息。")
            return self._confirm_metadata(identifier, updates)

    def _confirm_metadata(self, identifier, updates):
        allowed = {"company", "stock_code", "doc_type", "fiscal_year", "report_periods",
                   "disclosure_date", "inquiry_round", "revision", "confirmed"}
        if not isinstance(updates, dict) or set(updates) - allowed:
            raise RuntimeErrorInfo("INVALID_METADATA", "仅可修改资料身份字段。")
        current = self.get_report(identifier)
        report = Report.model_validate(current | {k: v for k, v in updates.items() if k != "confirmed"})
        report.capabilities["identity"] = {"confirmed": updates.get("confirmed") is True,
                                            "confirmed_at": now() if updates.get("confirmed") is True else None}
        report.status = "metadata_confirmed" if updates.get("confirmed") is True else "uploaded"
        self.repo.put("report", identifier, report)
        return _dict(report)

    def _safe_pdf_path(self, report):
        path = Path(report["path"]).resolve()
        root = (self.settings.runtime_dir / "pdfs").resolve()
        if path.parent != root or path.name != report["sha256"] + ".pdf":
            raise RuntimeErrorInfo("INVALID_SOURCE_PATH", "资料路径不属于本地资料库。")
        return path

    def page(self, report_id, number, *, image=False):
        report = self.get_report(report_id)
        if number < 1 or (report["page_count"] and number > report["page_count"]):
            raise RuntimeErrorInfo("NOT_FOUND", "页码超出资料范围。")
        if image:
            try:
                return render_page(self._safe_pdf_path(report), number)
            except ValueError as error:
                raise RuntimeErrorInfo("NOT_FOUND", "页码超出资料范围。") from error
        page = self.repo.get("page", f"{report_id}:{number}")
        if page is None:
            raise RuntimeErrorInfo("NOT_FOUND", "页面尚未解析。")
        return page

    def register_job(self, kind, handler):
        self._jobs[kind] = handler

    def submit_callable(self, kind, resource_id, fn, *, task=None):
        if self._closed:
            raise RuntimeErrorInfo("RUNTIME_CLOSED", "服务已关闭。")
        task = task or self.repo.create_task(kind, resource_id, mode=self.settings.mode)
        self._futures[task["task_id"]] = self._executor.submit(self._work, task["task_id"], fn)
        return task

    def submit_job(self, kind, resource_id, payload):
        if kind not in self._jobs:
            raise RuntimeErrorInfo("UNKNOWN_JOB", "任务类型尚未注册。")
        return self.submit_callable(kind, resource_id, lambda: self._jobs[kind](payload))

    def _work(self, task_id, fn):
        self.repo.update_task(task_id, "running", 0)
        try:
            result = fn()
            current = self.repo.get_task(task_id)
            if current["status"] == "running":
                self.repo.update_task(task_id, "succeeded", 1, result=result)
        except Exception as error:
            info = _error(error)
            self.repo.update_task(task_id, "failed", 1, error_code=info["code"], message=info["message"])
        finally:
            self._local.budget = None

    def wait_task(self, task_id, timeout=60):
        deadline = time.monotonic() + timeout
        while True:
            task = self.repo.get_task(task_id)
            if task is None:
                raise RuntimeErrorInfo("NOT_FOUND", "任务不存在。")
            if task["status"] not in {"queued", "running"}:
                return task
            if time.monotonic() >= deadline:
                raise TimeoutError("task is still running")
            time.sleep(.02)

    def submit_prepare(self, report_id):
        with self._report_lock:
            report = self.get_report(report_id)
            if not report.get("capabilities", {}).get("identity", {}).get("confirmed"):
                raise RuntimeErrorInfo("METADATA_UNCONFIRMED", "请先确认公司、资料类型及期间。")
            task = self.repo.create_task("prepare", report_id, mode=self.settings.mode)
            self.submit_callable("prepare", report_id, lambda: self._prepare(report_id, task["task_id"]), task=task)
            return {"task_id": task["task_id"], "report_id": report_id, "status": task["status"]}

    def store_parsed(self, parsed: ParsedReport):
        self.repo.put("parsed", parsed.report.id, parsed, parsed.report.id)
        self.repo.put("report", parsed.report.id, parsed.report)
        for page in parsed.pages:
            self.repo.put("page", f"{parsed.report.id}:{page.pdf_page_number}", page, parsed.report.id)
        for parent in parsed.parents:
            self.repo.put("source_parent", parent.id, parent, parsed.report.id)

    def _prepare(self, report_id, task_id):
        state = {"kind": "ingest", "report_id": report_id, "task_id": task_id,
                 "warnings": [], "usage": [], "global_error": None}
        with build_workflow(self._nodes({}), self.settings.runtime_dir / "checkpoints.sqlite") as graph:
            result = graph.invoke(state, {"configurable": {"thread_id": task_id}, "recursion_limit": 12})
        if result.get("global_error"):
            info = result["global_error"]
            raise RuntimeErrorInfo(info["code"], info["message"])
        return {"report_id": report_id, "warnings": result.get("warnings", [])}

    def submit_query(self, request):
        self._ensure_cloud()
        request = AnalysisRequest.model_validate(request)
        run = self._new_run(request)
        task = self.repo.get_task(run["task_id"])
        self.submit_callable("analysis", run["run_id"],
                             lambda: self.run_query_sync(request, run["run_id"]), task=task)
        return {"task_id": run["task_id"], "run_id": run["run_id"], "status": "queued"}

    def submit_report(self, request):
        request = AnalysisRequest.model_validate(request)
        return self.submit_query(request.model_copy(update={"task_type": "report"}))

    def _version_facts(self, report_id, version):
        facts = {}
        with self._facts_lock:
            for item in self.repo.list("fact", report_id):
                saved = self.repo.get("fact", f"{version}:{item['id']}")
                if saved:
                    facts[item["id"]] = saved
        return list(facts.values())

    def publish_financial_facts(self, report_id, facts):
        """Publish internally reconciled candidates without changing their review status.

        This is not an API that accepts user-declared verified amounts. Existing runs
        retain their submission-time fact snapshot; new runs see this publication.
        """
        self.get_report(report_id)
        manifest = self.repo.active_index(report_id)
        if manifest is None:
            raise RuntimeErrorInfo("INDEX_NOT_READY", "请先准备资料，再发布财务事实。")
        values = [FinancialFact.model_validate(item) for item in facts]
        if any(item.report_id != report_id for item in values):
            raise RuntimeErrorInfo("FACT_REPORT_MISMATCH", "财务事实不属于当前资料。")
        with self._facts_lock:
            for fact in values:
                self.repo.put("fact", f"{manifest['index_version']}:{fact.id}", fact, report_id)
        return {"report_id": report_id, "index_version": manifest["index_version"],
                "fact_ids": [item.id for item in values]}

    def _new_run(self, request):
        from annual_report_agent.generation.profiles import effective_profile
        try:
            profile = effective_profile(self.settings, request.model_profile)
        except ValueError as error:
            raise RuntimeErrorInfo("MODEL_PROFILE_INVALID", "模型档案不存在或配置不完整。") from error
        run_id = uuid4().hex
        manifests = [manifest for identifier in request.report_ids
                     if (manifest := self.repo.active_index(identifier)) is not None]
        identity_fields = {"company", "stock_code", "doc_type", "fiscal_year", "report_periods",
                           "disclosure_date", "inquiry_round", "revision"}
        for manifest in manifests:
            current = self.get_report(manifest["report_id"])
            indexed = manifest["report_metadata"]
            if any(current.get(key) != indexed.get(key) for key in identity_fields):
                raise RuntimeErrorInfo("INDEX_STALE", "资料身份信息已修改，请先重新准备资料。")
        task = self.repo.create_task("analysis", run_id, mode=self.settings.mode)
        run = {"run_id": run_id, "task_id": task["task_id"], "status": "queued", "mode": self.settings.mode,
               "request": _dict(request), "manifests": manifests,
               "fact_snapshot": {m["report_id"]: self._version_facts(m["report_id"], m["index_version"])
                                 for m in manifests},
               "report_index_versions": {m["report_id"]: m["index_version"] for m in manifests},
               "answer": None, "node_trace": [], "node_timings": [], "observed_usage": [],
               "effective_profile": profile,
               "subtask_trace": [], "traces": [], "error_code": None,
               "created_at": now()}
        self.repo.put("run", run_id, run)
        return run

    def get_run(self, run_id):
        run = self.repo.get("run", run_id)
        if run is None:
            raise RuntimeErrorInfo("NOT_FOUND", "分析运行不存在。")
        return run

    def run_query_sync(self, request, run_id=None, *, execution_options=None):
        """For the serial worker/evaluator; callers must not run parallel to that worker."""
        self._ensure_cloud()
        request = AnalysisRequest.model_validate(request)
        run = self.get_run(run_id) if run_id else self._new_run(request)
        task = self.repo.get_task(run["task_id"])
        if task["status"] == "queued":
            self.repo.update_task(task["task_id"], "running", 0)
        run.update(status="running", started_at=now())
        self.repo.put("run", run["run_id"], run)
        self._local.budget = {"calls": 0, "reserved_tokens": 0}
        state = {"kind": "analysis", "run_id": run["run_id"], "task_id": run["task_id"],
                 "request": _dict(request), "manifests": run["manifests"], "cursor": 0,
                 "results": [], "all_evidence": [], "all_calculations": [], "usage": [],
                 "warnings": [], "global_error": None, "subtask_error": None}
        try:
            with build_workflow(self._nodes(execution_options or {}),
                                self.settings.runtime_dir / "checkpoints.sqlite") as graph:
                result = graph.invoke(state, {"configurable": {"thread_id": run["run_id"]},
                                              "recursion_limit": 100})
            answer = result.get("answer")
            if answer is None:
                raise RuntimeErrorInfo("WORKFLOW_INCOMPLETE", "工作流未形成逐项结果。")
            info = result.get("global_error")
            run = self.get_run(run["run_id"])
            run.update(status="failed" if info else "succeeded", answer=answer,
                       error_code=info["code"] if info else None, completed_at=now(),
                       calls_used=self._local.budget["calls"],
                       reserved_tokens=self._local.budget["reserved_tokens"])
            self.repo.put("run", run["run_id"], run)
            self.repo.update_task(run["task_id"], run["status"], 1,
                                 error_code=run["error_code"], message=info["message"] if info else None)
            return run
        except Exception as error:
            info = _error(error)
            run = self.get_run(run["run_id"])
            run.update(status="failed", error_code=info["code"], message=info["message"])
            self.repo.put("run", run["run_id"], run)
            self.repo.update_task(run["task_id"], "failed", 1,
                                 error_code=info["code"], message=info["message"])
            failure = RuntimeErrorInfo(info["code"], info["message"])
            failure.observed_usage = run.get("observed_usage", [])
            raise failure from error
        finally:
            self._local.budget = None

    def _nodes(self, options):
        handlers = {"N1_acquire": self._acquire, "N2_parse": self._parse, "N3_index": self._build,
                    "N4_route": lambda state: self._plan(state, options),
                    "N5_retrieve": lambda state: self._retrieve(state, options),
                    "N6_rerank": lambda state: self._context(state, options),
                    "N7_calculate": self._calculate,
                    "N8_compose": lambda state: self._compose(state, options), "N9_verify": self._verify}
        def wrap(name, handler):
            def invoke(state):
                if state.get("run_id"):
                    run = self.get_run(state["run_id"])
                    run["node_trace"].append(name)
                    self.repo.put("run", run["run_id"], run)
                self.repo.update_task(state["task_id"], "running", min(.95, .05 + .1 * state.get("cursor", 0)), stage=name)
                started, status = time.perf_counter(), "succeeded"
                update = {}
                try:
                    if state.get("global_error") and name != "N9_verify":
                        status = "skipped"
                        return {}
                    update = handler(state)
                    return update
                except Exception as error:
                    status = "failed"
                    info = _error(error)
                    return {"global_error" if name in {"N1_acquire", "N2_parse", "N3_index", "N4_route", "N9_verify"}
                            else "subtask_error": info, "usage": state.get("usage", [])}
                finally:
                    observed_usage = update.get("usage", state.get("usage", []))
                    tasks = state.get("plan", {}).get("subtasks", [])
                    cursor = state.get("cursor", 0)
                    timing = {"node": name,
                        "subtask_id": tasks[cursor]["subtask_id"] if cursor < len(tasks) else None,
                        "latency_seconds": time.perf_counter() - started, "status": status}
                    if state.get("run_id"):
                        run = self.get_run(state["run_id"])
                        run.setdefault("node_timings", []).append(timing)
                        run["observed_usage"] = observed_usage
                        self.repo.put("run", run["run_id"], run)
                    else:
                        task = self.repo.get_task(state["task_id"])
                        self.repo.update_task(state["task_id"], "running", task["progress"],
                            node_timings=task.get("node_timings", []) + [timing], observed_usage=observed_usage)
            return invoke
        return {name: wrap(name, handlers[name]) for name in NODE_NAMES}

    def _acquire(self, state):
        if state["kind"] == "ingest":
            self._safe_pdf_path(self.get_report(state["report_id"]))
        return {}

    def _parse(self, state):
        report = Report.model_validate(self.get_report(state["report_id"]))
        saved = self.repo.get("parsed", report.id)
        compatible = (saved and saved["report"]["sha256"] == report.sha256
                      and saved.get("parents") and all(
                          parent.get("metadata", {}).get("parser_version") == PARSER_VERSION
                          for parent in saved["parents"]))
        if self.tokenizer is None:
            self.tokenizer = FixedTokenizer.load(self.settings.data_dir / "tokenizer")
        if compatible:
            parsed = ParsedReport.model_validate(saved)
            parsed.report = report.model_copy(update={"page_count": len(parsed.pages), "status": parsed.report.status})
            # Reuse expensive page extraction, then apply the current token ruler
            # and chunk policy. A caller may intentionally cache parents only.
            parsed.chunks = chunk_parents(parsed.parents, self.tokenizer, 500, 80)
        else:
            parsed = parse_report(report, tokenizer=self.tokenizer)
        self.store_parsed(parsed)
        diagnostics = []
        facts = extract_pdf_facts(parsed, diagnostics=diagnostics)
        for path in self.reference_paths:
            reference = json.loads(Path(path).read_text(encoding="utf-8"))
            if reference.get("source_sha256") == parsed.report.sha256:
                facts = verify_against_reference(facts, reference)
        self.repo.put("source_facts", report.id, {"facts": [_dict(f) for f in facts], "diagnostics": diagnostics})
        return {"warnings": parsed.warnings}

    def _build(self, state):
        parsed = ParsedReport.model_validate(self.repo.get("parsed", state["report_id"]))
        if not parsed.chunks:
            if parsed.report.status == "failed" or not parsed.pages:
                raise RuntimeErrorInfo("PDF_PARSE_FAILED", "资料未得到可用文本页，请检查 PDF 解析错误。")
            return {"warnings": state.get("warnings", []) + ["PDF_NEEDS_OCR:未建立文字索引。"]}
        before = len(self.index.usage_records)
        usage = state.setdefault("usage", [])
        try:
            manifest = self.index.build(parsed.report, parsed.parents, parsed.chunks)
        finally:
            usage.extend(self.index.usage_records[before:])
        source = self.repo.get("source_facts", parsed.report.id) or {"facts": []}
        self.publish_financial_facts(parsed.report.id, source["facts"])
        report = self.get_report(parsed.report.id)
        report["status"] = "ready" if not parsed.warnings else "partial"
        report["capabilities"]["text_search"] = {"status": "ready", "mode": self.settings.mode}
        report["capabilities"]["verified_financial_facts"] = {
            "status": "available" if any(f["validation_status"] == "verified" for f in source["facts"]) else "unverified",
            "count": sum(f["validation_status"] == "verified" for f in source["facts"])}
        self.repo.put("report", parsed.report.id, report)
        return {"manifests": [manifest], "usage": usage}

    def _current(self, state):
        return SubTaskPlan.model_validate(state["plan"]["subtasks"][state["cursor"]])

    def _plan(self, state, options):
        request = AnalysisRequest.model_validate(state["request"])
        snapshots = {item["report_id"]: item["report_metadata"] | {"id": item["report_id"]}
                     for item in state["manifests"]}
        reports = [snapshots.get(identifier) or self.repo.get("report", identifier)
                   for identifier in request.report_ids]
        reports = [r for r in reports if r is not None]
        usage = state.setdefault("usage", [])
        if "plan_override" in options:
            plan = QueryPlan.model_validate(options["plan_override"])
            if plan.original_question != request.question or not set(plan.report_ids).issubset(request.report_ids):
                raise RuntimeErrorInfo("INVALID_PLAN", "冻结计划与请求范围不一致。")
        else:
            if self.settings.mode == "mock":
                self._reserve_call()
            plan = plan_query(request, reports, self.generation_client_for(request.model_profile),
                              self.settings.mode, usage_sink=usage)
        run = self.get_run(state["run_id"])
        run["plan"] = _dict(plan)
        self.repo.put("run", run["run_id"], run)
        return {"plan": _dict(plan), "usage": usage, "cursor": 0}

    def _retrieve(self, state, options):
        task = self._current(state)
        if not set(task.report_ids).issubset({m["report_id"] for m in state["manifests"]}):
            raise RuntimeErrorInfo("INDEX_NOT_READY", "所选资料尚无当前模式可用索引。")
        mode = options.get("retrieval_mode", self.settings.retrieval_mode)
        if self.settings.mode == "mock" and mode != "bm25":
            self._reserve_call()
        before = len(self.index.usage_records)
        usage = state.setdefault("usage", [])
        try:
            candidates = self.index.retrieve(task, state["manifests"], mode=mode)
        finally:
            usage.extend(self.index.usage_records[before:])
        return {"candidates": [_dict(c) for c in candidates],
                "usage": usage}

    def _pages(self, report_ids):
        return {identifier: {page["pdf_page_number"]: page["raw_text"]
                             for page in self.repo.list("page", identifier)} for identifier in report_ids}

    def _locate(self, needle, raw):
        indexes = [i for i, char in enumerate(raw) if not char.isspace()]
        compact = "".join(raw[i] for i in indexes)
        target = re.sub(r"[\s|]", "", needle)
        position = compact.find(target) if target else -1
        if position < 0:
            return None
        return indexes[position], indexes[position + len(target) - 1] + 1

    def _locatable_evidence(self, evidence, task):
        output, used, seen = [], 0, set()
        pages = self._pages(task.report_ids)
        for source in evidence:
            spans = []
            for reference in source.page_refs:
                number = reference["pdf_page_number"]
                raw = pages.get(source.report_id, {}).get(number, "")
                match = self._locate(source.quote, raw)
                if match is None:
                    matches = [m for line in source.quote.splitlines() if len(line.strip()) >= 6
                               if (m := self._locate(line, raw)) is not None]
                    if matches:
                        match = min(m[0] for m in matches), max(m[1] for m in matches)
                if match:
                    spans.append((raw[match[0]:match[1]], reference))
            if not spans:
                continue
            quote = "\n".join(item[0] for item in spans)
            identity = (source.report_id, quote)
            if identity in seen or used + len(quote) > self.settings.context_token_budget:
                continue
            used += len(quote)
            seen.add(identity)
            output.append(source.model_copy(update={
                "id": evidence_id(source.index_version, source.id, source.role, quote), "quote": quote,
                "page_refs": [item[1] for item in spans],
                "metadata": source.metadata | {"subtask_id": task.subtask_id,
                    "original_text_aligned": True, "partial_context": quote != source.quote}}))
        return output

    def _context(self, state, options):
        task = self._current(state)
        candidates = [Candidate.model_validate(c) for c in state.get("candidates", [])]
        if options.get("rerank_enabled", self.settings.rerank_enabled):
            if self.reranker is None:
                from annual_report_agent.retrieval.reranker import LocalReranker
                self.reranker = LocalReranker.from_config(self.settings)
            candidates = self.reranker.rerank(task.rewritten_query, candidates)
        evidence = expand_context(candidates, self.repo, self.settings.context_token_budget)
        if not options.get("context_expansion", True):
            evidence = [item for item in evidence if item.role == "primary"]
        located = self._locatable_evidence(evidence, task)
        warnings = list(state.get("warnings", []))
        if len(located) < len(evidence):
            warnings.append("部分候选无法定位为原文片段或超出上下文预算，未发送。")
        return {"evidence": [_dict(item) for item in located], "warnings": warnings}

    def _calculate(self, state):
        task = self._current(state)
        versions = {m["report_id"]: m["index_version"] for m in state["manifests"]}
        snapshot = self.get_run(state["run_id"]).get("fact_snapshot", {})
        facts = {}
        for identifier in task.report_ids:
            if identifier not in versions:
                raise RuntimeErrorInfo("INDEX_NOT_READY", "财务事实尚未绑定可用资料版本。")
            for item in snapshot.get(identifier, []):
                facts[item["id"]] = FinancialFact.model_validate(item)
        bundle = select_and_calculate(task, list(facts.values()))
        evidence = [Evidence.model_validate(e) for e in state.get("evidence", [])]
        bound = []
        for fact in bundle.facts:
            page_number = fact.provenance.get("pdf_page_number")
            if not page_number:
                raise RuntimeErrorInfo("FACT_SOURCE_MISSING", "财务事实缺少真实 PDF 来源页。")
            page = self.page(fact.report_id, page_number)
            raw = page["raw_text"]
            version = versions[fact.report_id]
            parent = self.repo.get("parent", f"{version}:{fact.source_id}") or {}
            rows = parent.get("metadata", {}).get("rows", [])
            row_index = fact.provenance.get("row_index")
            row_text = (" ".join(str(value or "") for value in rows[row_index])
                        if isinstance(row_index, int) and 0 <= row_index < len(rows) else "")
            match = self._locate(row_text, raw) if row_text else None
            if match is None:
                match = self._wrapped_fact_row(fact, rows, row_index, raw)
            if match is None:
                raise RuntimeErrorInfo("FACT_SOURCE_MISSING", "财务原值及行标识无法共同定位到原文。")
            start = raw.rfind("\n", 0, match[0]) + 1
            end = raw.find("\n", match[1])
            quote = raw[start:end if end >= 0 else len(raw)]
            manifest = next(m for m in state["manifests"] if m["report_id"] == fact.report_id)
            refs = [ref for ref in parent.get("page_refs", []) if ref["pdf_page_number"] == page_number]
            item = Evidence(id=evidence_id(version, fact.id, "primary", quote),
                report_id=fact.report_id, parent_id=fact.source_id, quote=quote,
                page_refs=refs or [{"pdf_page_number": page_number, "printed_page_label": page.get("printed_page_label")}],
                index_version=version, metadata=manifest["report_metadata"] | {
                    "subtask_id": task.subtask_id, "section_path": parent.get("section_path"),
                    "question_number": parent.get("question_number"), "fact_provenance": fact.provenance})
            evidence.append(item)
            bound.append(fact.model_copy(update={"source_evidence_ids": [item.id]}))
        if sum(len(item.quote) for item in evidence) > self.settings.context_token_budget:
            raise RuntimeErrorInfo("RUN_BUDGET_EXCEEDED", "事实原文超过上下文预算，请缩小问题范围。")
        return {"facts": [_dict(f) for f in bound], "calculations": [_dict(c) for c in bundle.calculations],
                "evidence": [_dict(e) for e in evidence]}

    def _wrapped_fact_row(self, fact, rows, row_index, raw):
        """Align a row whose values occur between wrapped label lines in page text."""
        label = fact.provenance.get("row_label", "")
        label_parts = [part.strip() for part in label.splitlines() if part.strip()]
        if not label_parts or not fact.raw_value:
            return None
        cells = (rows[row_index] if isinstance(row_index, int) and 0 <= row_index < len(rows)
                 else [label, fact.raw_value])
        required = label_parts + [str(value).strip() for value in cells[1:] if value and str(value).strip()]
        neighbours = []
        if isinstance(row_index, int):
            for neighbour in (row_index - 1, row_index + 1):
                if 0 <= neighbour < len(rows) and rows[neighbour] and rows[neighbour][0]:
                    neighbours.append(str(rows[neighbour][0]).splitlines()[0].strip())
        lines = raw.splitlines(keepends=True)
        offsets, offset = [], 0
        for line in lines:
            offsets.append(offset)
            offset += len(line)
        # In pdfplumber text, a two-line first cell may surround a numeric-only
        # middle line. Bound the window by cell wrapping and adjacent row labels.
        maximum = min(9, 2 * max(len(str(cell or "").splitlines()) for cell in cells) + 1)
        for index, line in enumerate(lines):
            if not self._locate(label_parts[0], line):
                continue
            for end in range(index + 1, min(len(lines), index + maximum) + 1):
                if end > index + 1 and any(self._locate(neighbour, lines[end - 1]) for neighbour in neighbours):
                    break
                window = "".join(lines[index:end])
                if all(self._locate(part, window) is not None for part in required):
                    return offsets[index], offsets[index] + len(window.rstrip("\r\n"))
        return None

    def _compose(self, state, options):
        task = self._current(state)
        if self.settings.mode == "mock":
            self._reserve_call()
        usage = state.setdefault("usage", [])
        draft = compose(state["request"]["question"], task,
            [Evidence.model_validate(e) for e in state.get("evidence", [])],
            [FinancialFact.model_validate(f) for f in state.get("facts", [])],
            [Calculation.model_validate(c) for c in state.get("calculations", [])],
            options.get("generation_client") or self.generation_client_for(state["request"].get("model_profile", "default")),
            self.settings.mode, usage_sink=usage)
        return {"draft": _dict(draft), "usage": usage}

    def _verify(self, state):
        request = AnalysisRequest.model_validate(state["request"])
        plan = QueryPlan.model_validate(state["plan"]) if state.get("plan") else QueryPlan(
            original_question=request.question, rewritten_question=request.question,
            report_ids=request.report_ids, needs_clarification=True, clarification_reason="规划未完成。")
        results = [SubTaskResult.model_validate(r) for r in state.get("results", [])]
        all_evidence = list(state.get("all_evidence", []))
        all_calculations = list(state.get("all_calculations", []))
        cursor = state.get("cursor", 0)
        if not state.get("global_error") and not plan.needs_clarification and cursor < len(plan.subtasks):
            task = plan.subtasks[cursor]
            info = state.get("subtask_error")
            if info:
                status = {"RUN_BUDGET_EXCEEDED": "not_executed", "INSUFFICIENT_EVIDENCE": "insufficient_evidence",
                          "METRIC_UNSUPPORTED": "unsupported", "REPORT_AMBIGUOUS": "needs_clarification",
                          "DATA_CONFLICT": "needs_clarification"}.get(info["code"], "failed")
                result = SubTaskResult(subtask_id=task.subtask_id, status=status,
                                       error_code=info["code"], reason=info["message"])
            else:
                context = VerificationContext(task=task, report_ids=request.report_ids,
                    report_index_versions={m["report_id"]: m["index_version"] for m in state["manifests"]},
                    evidence=state.get("evidence", []), facts=state.get("facts", []),
                    calculations=state.get("calculations", []), pages=self._pages(task.report_ids))
                draft = AnswerDraft.model_validate(state.get("draft") or {"subtask_id": task.subtask_id})
                result = verify(draft, context)
            results.append(result)
            all_evidence.extend(state.get("evidence", []))
            all_calculations.extend(state.get("calculations", []))
            run = self.get_run(state["run_id"])
            trace = {"subtask_id": task.subtask_id, "task": _dict(task),
                **{key: state.get(key, [] if key != "draft" else None) for key in (
                    "candidates", "evidence", "facts", "calculations", "draft")}, "result": _dict(result)}
            run["subtask_trace"].append(trace)
            run["traces"].append(trace)
            self.repo.put("run", run["run_id"], run)
            cursor += 1
        done = state.get("global_error") or plan.needs_clarification or cursor >= len(plan.subtasks)
        update = {"results": [_dict(r) for r in results], "cursor": cursor,
                  "all_evidence": all_evidence, "all_calculations": all_calculations,
                  "candidates": [], "evidence": [], "facts": [], "calculations": [],
                  "draft": {}, "subtask_error": None}
        if done:
            answer = aggregate(plan, results, state["run_id"], self.settings.mode,
                               global_error=state.get("global_error", {}).get("code") if state.get("global_error") else None)
            used = {identifier for r in results for identifier in r.primary_evidence_ids + r.context_evidence_ids}
            answer.citations = list({e["id"]: Evidence.model_validate(e) for e in all_evidence if e["id"] in used}.values())
            calculation_ids = {identifier for r in results for identifier in r.calculation_ids}
            answer.calculations = list({c["id"]: c for c in all_calculations if c["id"] in calculation_ids}.values())
            answer.usage = state.get("usage", [])
            answer.warnings.extend(state.get("warnings", []))
            update["answer"] = _dict(answer)
        return update

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self._executor.shutdown(wait=True)
                for client in self._http_clients:
                    client.close()
                for client in self._profile_clients.values():
                    if client is not None:
                        client.close()
            finally:
                try:
                    if self._index is not None:
                        self._index.close()
                finally:
                    self._release_instance_lock()
