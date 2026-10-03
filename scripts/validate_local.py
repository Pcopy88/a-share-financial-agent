"""Reproduce the five provided half-year arithmetic cases through the full workflow."""

import argparse
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from annual_report_agent.api.extensions import sample_id_mapping
from annual_report_agent.config import Settings
from annual_report_agent.generation.report_export import export_markdown
from annual_report_agent.runtime import Runtime
from annual_report_agent.schemas.answer import AnswerResult


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="explicitly allow paid cloud calls")
    args = parser.parse_args()
    settings = Settings()
    if settings.mode == "live" and not args.live:
        raise SystemExit("live mode requires --live; stop API before this batch")
    runtime = Runtime(settings)
    rows = []
    try:
        mapping = sample_id_mapping(runtime)
        reference = json.loads(Path("evals/jiangfeng_2026h1_reference.json").read_text(encoding="utf-8"))
        report_id = mapping[reference["report_id"]]
        for case in reference["calculation_cases"]:
            run = runtime.run_query_sync({"question": case["question"], "report_ids": [report_id]})
            answer = run["answer"]
            results = [calculation for calculation in answer["calculations"]
                       if calculation["formula_id"] == case["formula_id"]]
            passed = (answer["status"] == "answered" and len(results) == 1
                and results[0]["unit"] == case["unit"]
                and Decimal(results[0]["result"]) == Decimal(case["reference_result_decimal"])
                and all(evidence["page_refs"] for evidence in answer["citations"]))
            rows.append({"case_id": case["id"], "passed": passed, "run": run})
        folder = settings.results_dir / "acceptance"
        folder.mkdir(parents=True, exist_ok=True)
        result = {"mode": settings.mode, "cloud_quality_verified": False,
                  "checked_at": datetime.now(UTC).isoformat(), "rows": rows}
        (folder / "jiangfeng_workflow.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        run = rows[0]["run"]
        facts = {item["id"]: item for trace in run["traces"] for item in trace["facts"]}
        markdown = export_markdown(AnswerResult.model_validate(run["answer"]), runtime.repo,
            facts=list(facts.values()), reports=[item["report_metadata"] for item in run["manifests"]])
        (folder / "jiangfeng_example.md").write_text(markdown, encoding="utf-8")
        print(json.dumps({"mode": settings.mode, "passed": sum(row["passed"] for row in rows),
                          "total": len(rows), "output": str(folder)}, ensure_ascii=False))
        return int(not all(row["passed"] for row in rows))
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
