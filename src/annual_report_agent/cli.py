"""Local command line; default commands never silently activate paid cloud calls."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from time import sleep

from annual_report_agent.config import Settings


def output(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def parser():
    root = argparse.ArgumentParser(description="A股财报资料分析：上传、检索、计算和证据核验")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="检查配置，不读取或显示密钥内容、不调用云端")
    commands.add_parser("api", help="启动 FastAPI（一个 worker）")
    commands.add_parser("ui", help="启动 Streamlit")
    commands.add_parser("start", help="同时启动后端和界面；Ctrl+C 停止两者")
    backup = commands.add_parser("backup", help="后端停止后，备份当前模式的数据库和文件")
    backup.add_argument("--to", required=True)
    restore = commands.add_parser("restore", help="恢复到新的空目录，不覆盖原库")
    restore.add_argument("--archive", required=True)
    restore.add_argument("--to", required=True)
    models = commands.add_parser("models").add_subparsers(dest="action", required=True)
    models.add_parser("prepare-tokenizer", help="下载固定版本 CPU tokenizer")
    models.add_parser("prepare-reranker", help="下载固定本地重排权重并执行两段文本测试")
    samples = commands.add_parser("samples", help="导入样本清单中的 PDF 和已核对的身份")
    samples.add_argument("--manifest", default="evals/sample_manifest.json")
    samples.add_argument("--split", choices=["development", "test"])
    samples.add_argument("--prepare", action="store_true", help="随后解析建库；live 模式会调用百炼")
    evaluation = commands.add_parser("evaluate", help="运行独立评测并保存逐题结果")
    evaluation.add_argument("--dataset", required=True)
    evaluation.add_argument("--experiment", default="M2", choices=["R0", "R1", "R2", "R3", "M1", "M2", "Q0", "Q1", "C0", "C1"])
    evaluation.add_argument("--split", default="development", choices=["development", "test"])
    evaluation.add_argument("--repetitions", type=int, default=1)
    evaluation.add_argument("--frozen")
    evaluation.add_argument("--model-profile", default="default")
    return root


def _ui_command(settings):
    return [sys.executable, "-m", "streamlit", "run", str(Path(__file__).parent / "ui" / "app.py"),
            "--server.address", "127.0.0.1", "--server.port", str(settings.ui_port),
            "--server.maxUploadSize", str(settings.max_upload_mb),
            "--server.headless", "true", "--browser.gatherUsageStats", "false"]


def _start(settings):
    processes = []
    logs = []
    folder = settings.results_dir / "server_logs"
    folder.mkdir(parents=True, exist_ok=True)
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        for name, command in (("api", [sys.executable, "-m", "annual_report_agent", "api"]),
                              ("ui", _ui_command(settings))):
            handle = (folder / f"{name}.log").open("a", encoding="utf-8")
            logs.append(handle)
            processes.append(subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                               creationflags=creationflags))
        output({"ui": f"http://127.0.0.1:{settings.ui_port}",
                "api": f"http://{settings.api_host}:{settings.api_port}/docs",
                "mode": settings.mode, "logs": str(folder), "stop": "Ctrl+C"})
        while all(process.poll() is None for process in processes):
            sleep(.5)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for handle in logs:
            handle.close()


def _samples(settings, args):
    from annual_report_agent.runtime import Runtime
    data = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    documents = data if isinstance(data, list) else data.get("documents", data.get("samples", []))
    runtime = Runtime(settings)
    try:
        result = []
        for document in documents:
            if args.split and document.get("split") != args.split:
                continue
            path = Path(document["path"])
            content = path.read_bytes()
            import hashlib
            if hashlib.sha256(content).hexdigest() != document["sha256"]:
                raise ValueError(f"sample SHA mismatch: {document['id']}")
            fields = {key: document[key] for key in ("company", "stock_code", "doc_type",
                "fiscal_year", "report_periods", "disclosure_date", "inquiry_round", "revision")
                if key in document}
            report = runtime.upload_pdf(path.name, content)
            runtime.confirm_metadata(report["id"], fields | {"confirmed": True})
            item = {"sample_id": document["id"], "report_id": report["id"]}
            if args.prepare:
                task = runtime.submit_prepare(report["id"])
                output(item | task)
                while True:
                    try:
                        status = runtime.wait_task(task["task_id"], timeout=30)
                    except TimeoutError:
                        output({"task_id": task["task_id"], "status": "running"})
                        continue
                    if status["status"] not in {"queued", "running"}:
                        item["status"] = status
                        break
            result.append(item)
        output(result)
        return int(any(item.get("status", {}).get("status") in {"failed", "interrupted"} for item in result))
    finally:
        runtime.close()


def main(argv=None):
    args = parser().parse_args(argv)
    settings = Settings()
    if args.command == "doctor":
        packages = {}
        for package in ("annual-report-agent", "chromadb", "langgraph", "akshare", "torch", "FlagEmbedding"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = "not installed"
        output({"python": sys.version, "mode": settings.mode, "packages": packages,
                "missing_cloud_config": settings.missing_cloud_config(),
                "runtime_dir": str(settings.runtime_dir), "live_verified": False})
        return 0
    if args.command == "api":
        import uvicorn
        uvicorn.run("annual_report_agent.api.app:create_app", factory=True,
                    host=settings.api_host, port=settings.api_port, workers=1)
        return 0
    if args.command == "ui":
        return subprocess.call(_ui_command(settings))
    if args.command == "start":
        return _start(settings)
    if args.command == "backup":
        from annual_report_agent.storage.backup import backup
        output({"archive": backup(settings.runtime_dir, args.to)})
        return 0
    if args.command == "restore":
        from annual_report_agent.storage.backup import restore
        output({"restored_directory": restore(args.archive, args.to),
                "next": "将恢复目录放到 ARA_DATA_DIR 对应的 mock/live 子目录，再启动服务。"})
        return 0
    if args.command == "models":
        if args.action == "prepare-tokenizer":
            from annual_report_agent.ingestion.tokenizer import FixedTokenizer
            tokenizer = FixedTokenizer.load(settings.data_dir / "tokenizer", allow_download=True)
            output({"tokenizer_version": tokenizer.version, "example_tokens": tokenizer.count("营业收入")})
        else:
            from annual_report_agent.ingestion.indexer import Candidate
            from annual_report_agent.retrieval.reranker import LocalReranker
            reranker = LocalReranker.from_config(settings, allow_download=True)
            candidates = [Candidate(chunk={"id": str(index), "text": text}, index_version="probe")
                for index, text in enumerate(["公司销售电子材料取得营业收入。", "办公室的墙壁为白色。"])]
            result = reranker.rerank("公司的收入来自哪里？", candidates)
            output({"profile": reranker.profile, "last_run": reranker.last_run,
                    "ranking": [item.model_dump() for item in result]})
        return 0
    if args.command == "samples":
        return _samples(settings, args)
    if args.command == "evaluate":
        from annual_report_agent.api.extensions import sample_id_mapping
        from annual_report_agent.evaluation.runner import EvaluationRunner
        from annual_report_agent.runtime import Runtime
        runtime = Runtime(settings)
        try:
            result = EvaluationRunner(runtime, settings.results_dir / "evaluations").run(args.dataset,
                experiment=args.experiment, split=args.split, repetitions=args.repetitions,
                frozen_path=args.frozen, id_mapping=sample_id_mapping(runtime), model_profile=args.model_profile)
            runtime.repo.put("evaluation", result["id"], result)
            output(result)
            return int(result["summary"]["errors"] > 0)
        finally:
            runtime.close()
    return 1
