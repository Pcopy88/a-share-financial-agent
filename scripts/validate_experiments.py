"""Offline acceptance: 50-case execution plus development-only ablation smoke runs.

This explicitly enables source-SHA-bound, assistant-reviewed financial references.
It measures the reference-assisted calculation track, not automatic extraction
accuracy. Never use held-out outcomes here to adjust production rules.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from annual_report_agent.api.extensions import sample_id_mapping
from annual_report_agent.config import Settings
from annual_report_agent.evaluation.dataset import load_cases
from annual_report_agent.evaluation.runner import EvaluationRunner
from annual_report_agent.runtime import Runtime


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main():
    settings = Settings()
    if settings.mode != "mock":
        raise SystemExit("This acceptance script only runs mock mode; use the documented live CLI separately.")
    folder = settings.results_dir / "acceptance"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path("evals/annual_sample_manifest.json").read_text(encoding="utf-8"))
    references = [Path(sample["reference_path"]) for sample in manifest["samples"]]
    runtime = Runtime(settings, reference_paths=references)
    record = {"checked_at": datetime.now(UTC).isoformat(), "mode": "mock",
              "cloud_quality_verified": False, "track": "reference_assisted_verified_facts",
              "comparisons": "two development cases, one repetition; pipeline smoke only",
              "preparations": [], "experiments": []}
    try:
        for sample in manifest["samples"]:
            content = Path(sample["path"]).read_bytes()
            if hashlib.sha256(content).hexdigest() != sample["sha256"]:
                raise ValueError("sample SHA mismatch")
            report = runtime.upload_pdf(Path(sample["path"]).name, content)
            if sample["split"] == "test":
                metadata = {key: sample[key] for key in ("company", "stock_code", "doc_type",
                    "fiscal_year", "report_periods", "disclosure_date")}
                runtime.confirm_metadata(report["id"], metadata | {"confirmed": True})
                task = runtime.submit_prepare(report["id"])
                while True:
                    try:
                        task = runtime.wait_task(task["task_id"], timeout=30)
                        break
                    except TimeoutError:
                        print(json.dumps({"preparing": sample["id"]}), flush=True)
                record["preparations"].append(task)
                if task["status"] != "succeeded":
                    raise ValueError("held-out source preparation failed")
        mapping = sample_id_mapping(runtime)
        runner = EvaluationRunner(runtime, settings.results_dir / "evaluations")
        full_dataset = Path("evals/annual_50_v1.jsonl")
        for split in ("development", "test"):
            result = runner.run(full_dataset, split=split, experiment="M2", id_mapping=mapping)
            runtime.repo.put("evaluation", result["id"], result)
            record["experiments"].append(result)
            print(json.dumps({"experiment": "M2", "split": split, "summary": result["summary"]}), flush=True)
        development, _ = load_cases(full_dataset, "development")
        smoke = [next(case for case in development if case.route == route)
                 for route in ("semantic", "hybrid")]
        subset = folder / "development_ablation_smoke.jsonl"
        subset.write_text("\n".join(case.model_dump_json() for case in smoke) + "\n", encoding="utf-8")
        baseline = runner.run(subset, experiment="M2", id_mapping=mapping)
        record["experiments"].append(baseline)
        frozen = settings.results_dir / "evaluations" / baseline["id"] / "frozen.json"
        for experiment in ("R0", "R1", "R2", "R3", "M1", "Q0", "Q1", "C0", "C1"):
            runtime.settings.rerank_enabled = experiment == "R3"
            result = runner.run(subset, experiment=experiment, id_mapping=mapping,
                                frozen_path=frozen if experiment in {"M1", "Q0", "Q1", "C0", "C1"} else None)
            runtime.repo.put("evaluation", result["id"], result)
            record["experiments"].append(result)
            write(folder / "experiment_workflow.json", record)
            print(json.dumps({"experiment": experiment, "errors": result["summary"]["errors"],
                              "id": result["id"]}), flush=True)
        record["completed_at"] = datetime.now(UTC).isoformat()
        write(folder / "experiment_workflow.json", record)
        return int(any(item["summary"]["errors"] for item in record["experiments"]))
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
