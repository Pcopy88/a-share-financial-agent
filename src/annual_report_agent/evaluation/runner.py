"""Repeatable retrieval and generation experiments; labels never enter inference."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from annual_report_agent.evaluation.accounting import summarize_usage
from annual_report_agent.evaluation.dataset import load_cases
from annual_report_agent.evaluation.matching import match_gold
from annual_report_agent.evaluation.metrics import numeric_correct, retrieval_scores, summarize
from annual_report_agent.evaluation.resources import ResourceSampler
from annual_report_agent.generation.composer import compose
from annual_report_agent.generation.profiles import effective_profile
from annual_report_agent.generation.verifier import VerificationContext, aggregate, verify
from annual_report_agent.schemas.answer import Evidence
from annual_report_agent.schemas.finance import Calculation, FinancialFact
from annual_report_agent.schemas.query import QueryPlan, SubTaskPlan

EXPERIMENTS = {"R0", "R1", "R2", "R3", "M1", "M2", "Q0", "Q1", "C0", "C1"}


class FrozenInputError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def _plain(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _fact_snapshot_digest(snapshot):
    if not isinstance(snapshot, dict) or any(not isinstance(items, list) for items in snapshot.values()):
        raise FrozenInputError("FROZEN_REFREEZE_REQUIRED", "full fact snapshot missing; regenerate frozen inputs")
    normalized = {report_id: sorted(json.dumps(_plain(item), sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), default=str) for item in items) for report_id, items in snapshot.items()}
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def freeze_runs(runs: dict, destination: Path) -> str:
    """Freeze already executed input contexts, never desired answers or gold labels."""
    payload = {}
    for case_id, run in runs.items():
        answer = run.get("answer") or {}
        plan = run.get("plan") or answer.get("query_plan")
        if not plan or not run.get("traces"):
            raise ValueError("run must include its executed plan and traces")
        snapshot = run.get("fact_snapshot")
        digest = _fact_snapshot_digest(snapshot)
        if set(snapshot) != set(run["report_index_versions"]):
            raise FrozenInputError("FROZEN_REFREEZE_REQUIRED", "fact snapshot scope missing; regenerate frozen inputs")
        payload[case_id] = {"frozen_schema_version": 2, "mode": run["mode"], "plan": plan,
                            "traces": run["traces"], "fact_snapshot": snapshot,
                            "fact_snapshot_sha256": digest,
                            "report_index_versions": run["report_index_versions"]}
    _write(Path(destination), payload)
    return hashlib.sha256(Path(destination).read_bytes()).hexdigest()


class EvaluationRunner:
    def __init__(self, runtime, output_dir):
        self.runtime, self.output_dir = runtime, Path(output_dir)

    def run(self, dataset, *, experiment="M2", split="development", repetitions=1,
            frozen_path=None, id_mapping=None, model_profile="default", progress=None):
        if experiment not in EXPERIMENTS or split not in {"development", "test"}:
            raise ValueError("invalid experiment or split")
        if not 1 <= repetitions <= 3:
            raise ValueError("repetitions must be 1..3")
        if experiment in {"M1", "C0", "C1", "Q0", "Q1"} and frozen_path is None:
            raise ValueError("this experiment requires frozen executed inputs")
        cases, dataset_sha = load_cases(dataset, split)
        if not cases:
            raise ValueError("selected dataset split is empty")
        frozen, frozen_hash = {}, None
        if frozen_path:
            raw = Path(frozen_path).read_bytes()
            frozen, frozen_hash = json.loads(raw), hashlib.sha256(raw).hexdigest()
        identifier = "eval-" + uuid4().hex
        folder = self.output_dir / identifier
        folder.mkdir(parents=True)
        record = {"id": identifier, "mode": self.runtime.settings.mode, "experiment": experiment,
            "split": split, "repetitions": repetitions, "model_profile": model_profile,
            "dataset_sha256": dataset_sha, "frozen_sha256": frozen_hash,
            "started_at": datetime.now(UTC).isoformat(), "case_count": len(cases),
            "configuration": self.runtime.settings.public_dict(),
            "effective_profile": effective_profile(self.runtime.settings, model_profile),
            "semantic_scoring": "requires independent human review; not auto-awarded",
            "gold_matching": "source-bound literal quote coverage; tables require original-page token/row verification"}
        rows, executed = [], {}
        with (folder / "rows.jsonl").open("w", encoding="utf-8") as log:
            for repeat in range(repetitions):
                for case in cases:
                    started = perf_counter()
                    row = {"case_id": case.id, "repeat": repeat, "mode": record["mode"],
                           "metrics": {}, "error": None, "usage": []}
                    sampler = ResourceSampler()
                    sampler.__enter__()
                    try:
                        request = case.request(id_mapping)
                        request.model_profile = model_profile
                        if experiment.startswith("R"):
                            row.update(self._retrieval(case, request, experiment, id_mapping or {},
                                                       usage_sink=row["usage"]))
                        else:
                            if experiment in {"M1", "C0", "C1"}:
                                run = self._frozen_generate(case, frozen[case.id], experiment,
                                    model_profile, request=request, usage_sink=row["usage"])
                            else:
                                options = {}
                                if experiment in {"Q0", "Q1"}:
                                    baseline = frozen[case.id]
                                    self._check_frozen(baseline, request, require_current_snapshot=True)
                                    plan = QueryPlan.model_validate(baseline["plan"])
                                    if experiment == "Q0":
                                        for task in plan.subtasks:
                                            task.rewritten_query = task.query_text
                                    options = {"plan_override": plan}
                                run = self.runtime.run_query_sync(request, execution_options=options)
                                if experiment in {"Q0", "Q1"}:
                                    self._check_run_snapshot(baseline, run)
                            executed[case.id] = run
                            row.update(self._score_answer(case, run))
                    except Exception as error:
                        # No request headers, prompts or credentials in error text.
                        code = getattr(error, "code", None)
                        row["error"] = code.strip() if isinstance(code, str) and code.strip() else type(error).__name__
                        row["usage"].extend(getattr(error, "observed_usage", []))
                    finally:
                        sampler.__exit__(None, None, None)
                    row["latency_seconds"] = perf_counter() - started
                    row["resources"] = sampler.result
                    row["accounting"] = summarize_usage(row.get("usage", []))
                    rows.append(row)
                    log.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    log.flush()
                    if progress:
                        progress(len(rows) / (len(cases) * repetitions))
        if executed and all(run.get("traces") for run in executed.values()):
            record["frozen_output_sha256"] = freeze_runs(executed, folder / "frozen.json")
        record["summary"] = summarize(rows)
        record["accounting"] = summarize_usage([item for row in rows for item in row.get("usage", [])])
        record["status"] = "completed_with_errors" if record["summary"]["errors"] else "completed"
        record["finished_at"] = datetime.now(UTC).isoformat()
        _write(folder / "result.json", record)
        return record

    def _retrieval(self, case, request, experiment, id_mapping, *, usage_sink=None):
        if case.route == "data":
            return {"metrics": {}, "not_applicable": "pure calculation does not use retrieval"}
        canonical = getattr(case, "canonical_subtasks", None)
        if case.route == "hybrid" and not canonical:
            raise ValueError("hybrid retrieval experiment requires frozen canonical_subtasks")
        parts = canonical or [{"query_text": getattr(case, "canonical_query", case.question)}]
        mode = {"R0": "bm25", "R1": "dense", "R2": "hybrid", "R3": "hybrid"}[experiment]
        snapshot = self.runtime.index.snapshot(request.report_ids)
        usage = [] if usage_sink is None else usage_sink
        rankings = []
        for index, part in enumerate(parts):
            query = part["query_text"]
            task = SubTaskPlan(subtask_id=f"s{index}", source_text=query, query_text=query,
                rewritten_query=query, report_ids=request.report_ids, hard_filters=request.hard_filters,
                route="semantic")
            before_usage = len(getattr(self.runtime.index, "usage_records", []))
            try:
                ranked = self.runtime.index.retrieve(task, manifests=snapshot, mode=mode, limit=30)
            finally:
                usage.extend(getattr(self.runtime.index, "usage_records", [])[before_usage:])
            if experiment == "R3":
                reranker = getattr(self.runtime, "reranker", None)
                if reranker is None and self.runtime.settings.rerank_enabled:
                    from annual_report_agent.retrieval.reranker import LocalReranker
                    reranker = self.runtime.reranker = LocalReranker.from_config(self.runtime.settings)
                if reranker is None:
                    raise ValueError("local reranker not loaded; R3 cannot fall back to R2")
                ranked = reranker.rerank(query, ranked)
            rankings.append(ranked)
        # Interleave fixed subtask rankings. Concatenation would hide later subtask hits.
        candidates, seen = [], set()
        for position in range(max((len(ranking) for ranking in rankings), default=0)):
            for ranking in rankings:
                if position >= len(ranking):
                    continue
                candidate = ranking[position]
                key = (candidate.index_version, candidate.chunk["id"])
                if key not in seen:
                    candidates.append(candidate)
                    seen.add(key)
        gold = [unit | {"report_id": id_mapping.get(unit["report_id"], unit["report_id"])}
                for unit in case.gold_evidence]
        repo = getattr(self.runtime, "repo", None)
        page_texts = ({report_id: {page["pdf_page_number"]: page["raw_text"]
                       for page in repo.list("page", report_id)} for report_id in request.report_ids}
                      if repo is not None else None)
        matches = [match_gold(candidate, gold, page_texts) for candidate in candidates]
        all_gold = {unit.get("id", f"g{i}") for i, unit in enumerate(gold)}
        return {"metrics": retrieval_scores(matches, all_gold, complete_gold=case.gold_complete),
                "ranked_candidates": [_plain(candidate) for candidate in candidates],
                "matched_gold_ids": [sorted(match) for match in matches], "index_snapshot": snapshot,
                "subtask_aggregation": "deduplicated_round_robin_before_top_k",
                "usage": usage,
                "subtask_rankings": [[_plain(item) for item in ranked] for ranked in rankings]}

    def _check_frozen(self, frozen, request, *, require_current_snapshot=False):
        if frozen["mode"] != self.runtime.settings.mode:
            raise ValueError("frozen mode mismatch; never mix mock/live evidence")
        plan = frozen["plan"]
        if plan["original_question"] != request.question:
            raise ValueError("frozen question differs from current case")
        filters = {key: value for key, value in plan.get("hard_filters", {}).items() if value is not None}
        if set(plan["report_ids"]) != set(request.report_ids) or filters != request.hard_filters.model_dump(exclude_none=True):
            raise ValueError("frozen report scope differs from current case")
        if frozen.get("frozen_schema_version") != 2 or "fact_snapshot_sha256" not in frozen:
            raise FrozenInputError("FROZEN_REFREEZE_REQUIRED", "legacy frozen input; regenerate with full fact snapshot")
        if (_fact_snapshot_digest(frozen.get("fact_snapshot")) != frozen["fact_snapshot_sha256"]
                or set(frozen["fact_snapshot"]) != set(frozen["report_index_versions"])):
            raise FrozenInputError("FROZEN_SNAPSHOT_INVALID", "frozen fact snapshot is inconsistent; regenerate")
        for version in frozen["report_index_versions"].values():
            if not self.runtime.repo.build(version):
                raise ValueError("frozen evidence index missing")
        if require_current_snapshot:
            versions, facts = {}, {}
            for report_id in frozen["report_index_versions"]:
                manifest = self.runtime.repo.active_index(report_id)
                if manifest is None:
                    raise FrozenInputError("FROZEN_INDEX_CHANGED", "active index no longer exists; regenerate")
                versions[report_id] = manifest["index_version"]
                facts[report_id] = self.runtime._version_facts(report_id, manifest["index_version"])
            self._check_run_snapshot(frozen, {"report_index_versions": versions, "fact_snapshot": facts})

    @staticmethod
    def _check_run_snapshot(frozen, run):
        if run.get("report_index_versions") != frozen["report_index_versions"]:
            raise FrozenInputError("FROZEN_INDEX_CHANGED", "active index differs from frozen index; regenerate")
        if _fact_snapshot_digest(run.get("fact_snapshot")) != frozen["fact_snapshot_sha256"]:
            raise FrozenInputError("FROZEN_FACT_SNAPSHOT_CHANGED", "fact snapshot differs from frozen inputs; regenerate")

    def _frozen_generate(self, case, frozen, experiment, model_profile, request=None, *, usage_sink=None):
        self._check_frozen(frozen, request or case.request())
        plan = QueryPlan.model_validate(frozen["plan"])
        results, all_evidence, all_calculations = [], {}, {}
        usage = [] if usage_sink is None else usage_sink
        client = self.runtime.generation_client_for(model_profile)
        sent_traces = []
        for trace in frozen["traces"]:
            task = SubTaskPlan.model_validate(trace["task"])
            evidence = [Evidence.model_validate(item) for item in trace["evidence"]
                        if experiment != "C0" or item["role"] == "primary"]
            facts = [FinancialFact.model_validate(item) for item in trace["facts"]]
            calculations = [Calculation.model_validate(item) for item in trace["calculations"]]
            draft = compose(plan.original_question, task, evidence, facts, calculations, client=client,
                            mode=self.runtime.settings.mode, usage_sink=usage)
            pages = {report_id: {page["pdf_page_number"]: page["raw_text"]
                        for page in self.runtime.repo.list("page", report_id)}
                     for report_id in task.report_ids}
            result = verify(draft, VerificationContext(task=task, report_ids=plan.report_ids,
                report_index_versions=frozen["report_index_versions"], evidence=evidence,
                facts=facts, calculations=calculations, pages=pages))
            results.append(result)
            sent_traces.append(trace | {"evidence": [_plain(item) for item in evidence],
                                        "draft": _plain(draft), "result": _plain(result)})
            all_evidence.update({item.id: item for item in evidence})
            all_calculations.update({item.id: item for item in calculations})
        answer = aggregate(plan, results, mode=self.runtime.settings.mode)
        evidence_ids = {identifier for result in results
                        for identifier in result.primary_evidence_ids + result.context_evidence_ids}
        calculation_ids = {identifier for result in results for identifier in result.calculation_ids}
        answer.citations = [item for identifier, item in all_evidence.items() if identifier in evidence_ids]
        answer.calculations = [_plain(item) for identifier, item in all_calculations.items()
                               if identifier in calculation_ids]
        answer.usage = usage
        return {"answer": _plain(answer), "mode": self.runtime.settings.mode,
                "plan": _plain(plan), "traces": sent_traces,
                "fact_snapshot": frozen["fact_snapshot"],
                "report_index_versions": frozen["report_index_versions"]}

    @staticmethod
    def _score_answer(case, run):
        answer = _plain(run.get("answer") or {})
        metrics = {"status_match": float(answer.get("status") == case.expected_status)
                   if case.expected_status else None}
        plan = answer.get("query_plan") or run.get("plan") or {}
        metrics["route_accuracy"] = (float(plan.get("route") == case.route)
                                      if case.route in {"semantic", "data", "hybrid"} else None)
        # Semantic truth, compound completeness and citation support require reviewed labels.
        metrics.update(citation_correctness=None, conclusion_support=None, compound_correctness=None)
        targets = getattr(case, "expected_calculations", [])
        numeric_results = []
        operand_facts = {fact["id"]: fact for trace in run.get("traces", [])
                         for fact in trace.get("facts", [])}
        for expected in targets:
            matching = [item for item in answer.get("calculations", [])
                        if item["formula_id"] == expected["formula_id"]
                        and item["period_label"] == expected["period_label"]]
            correct = False
            for actual in matching:
                scopes = {operand_facts.get(identifier, {}).get("scope")
                          for identifier in actual["operand_fact_ids"]}
                correct |= numeric_correct(
                    {"value": actual["result"], "unit": actual["unit"],
                     "period": actual["period_label"], "scope": next(iter(scopes)) if len(scopes) == 1 else None},
                    {"value": expected["result"], "unit": expected["unit"],
                     "period": expected["period_label"], "scope": expected.get("scope", "consolidated")})
            numeric_results.append({"formula_id": expected["formula_id"],
                                    "period": expected["period_label"], "correct": correct})
        metrics["numeric_target_accuracy"] = (sum(item["correct"] for item in numeric_results) / len(targets)
                                               if targets else None)
        metrics["all_numeric_targets_correct"] = float(all(item["correct"] for item in numeric_results)) if targets else None
        return {"metrics": metrics, "answer": answer, "run_id": run.get("run_id"),
                "node_timings": run.get("node_timings", []),
                "numeric_targets": numeric_results,
                "review_status": "pending_independent_review", "usage": answer.get("usage", [])}
