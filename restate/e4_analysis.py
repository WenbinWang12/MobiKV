"""E4 CPU analysis of source-bound normalized records, never an experiment gate.

The executor supplies destination-local timestamps and actual counters. An
independent raw auditor must establish the five gates; this module checks their
contract, exact offered coverage and solo pairing, but does not re-open tensor
or socket evidence. No synthetic timing, outcome filtering or stage completion.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import random
import re
import statistics

if __package__:
    from .plan import digest, require, validate_plan
else:
    from e4_plan import digest, require, validate_plan


GATES = ("correctness_passed", "diagnostic_passed", "traffic_passed",
         "cleanup_passed", "timing_contract_passed")
IDENTITY = ("trial_id", "phase", "sample_role", "kind", "cell_id", "module",
            "method", "block_id", "input_family_id", "block_seed")
STATUSES = ("passed", "failed", "timeout", "cancelled", "not_started")
ENDPOINTS = ("holder", "destination")
SCHEMA_VERSION = 1
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2026113300


def record_contract():
    """JSON-serializable producer contract. Unknown failed values are null.

    Timestamp names ending _s in timing/requests/migration all refer to the
    destination's monotonic clock, never an absolute holder clock. The same
    accepted_s anchors each offered request; an executor-entered timestamp may
    be later and must not replace that common acceptance instant.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "identity": list(IDENTITY),
        "binding": ["plan_sha256", "trial_sha256", "run_id", "attempt_id",
                    "source_sha256 (nonempty filename->SHA256)", "cohort_sha256",
                    "model_revision", "model_manifest_sha256"],
        "status": list(STATUSES), "gates": list(GATES),
        "raw_artifacts": {
            "candidate": "{path,sha256}; unique actual main report, including failures",
            "diagnostic": "{path,sha256,contents_sha256:{transfer-role:actual diagnostic content SHA}}; null if unavailable",
            "references": "{decode-role:{path,sha256}|null}; exact planned roles",
        },
        "timing": {"accepted_s": "common destination acceptance before setup/dispatch",
                   "group_service_settled_s": "all own jobs/background drained, compact paced peer settle ACK received",
                   "group_ownership_settled_s": "subsequent disarm/audit/owned resources settled; not service makespan"},
        "requests": [{"role": "exact planned order", "spec_sha256": "digest(plan request)",
            "input_ids_sha256": "actual prefix+suffix token identity",
            "output_ids_sha256": "actual greedy generated token identity",
            "status": "actual status; retain all offered slots",
            "executor_entered_s": "actual postacceptance entry",
            "token_delivery_s": "all actual destination delivery timestamps, 64 on success",
            "own_settled_s": "this request's migration/socket/thread/prefix ownership cleanup completed",
            "cleanup_passed": "bool", "migration": "null for remote; migration schema for repair"}],
        "transfer_jobs": [{"role": "exact planned order", "spec_sha256": "digest(plan transfer)",
            "input_ids_sha256": "actual prefix token identity; main needs no KV content hash",
            "status": "actual status", "own_settled_s": "this transfer's actual ownership settlement",
            "cleanup_passed": "bool", "migration": "migration schema"}],
        "migration": {"first_gpu_ready_s": "actual first destination layer ready",
            "gpu_ready_s": "actual full destination prefix ready",
            "drained_s": "actual terminal/frame drain ACK completed",
            "payload_bytes": "successful KV payload, excludes headers/control",
            "complete_layers": "actual fully published layers, success=48"},
        "resources": {
            "application_sent_bytes": "{holder,destination}: successful application-frame bytes each direction, counted once",
            "compute_service_s": "{holder,destination}: ready-compute operation host-wall service ending at CUDA completion; not kernel time",
            "compute_queue_wait_s": "{holder,destination}: actual operation wait sums",
            "background": "{holder,destination}:{offered,completed,failed,pending}; all offered, even on failure",
            "contention": "{holder,destination}:{coeligible_grant_count,successful_coeligible_grant_count,bulk_max_eligible_wait_s}; simultaneous actual grant sample, not independent bin maxima",
        },
        "unknown_failure_fields": "null permitted, not fabricated zero; complete offered arrays still required",
        "source_of_gate_truth": "Independent source-bound raw evidence audit, not booleans invented by this analysis",
        "analysis_scope": "2340 offered full attempts; only 576 measured groups enter paired summaries; warmups retained separately",
        "forbidden_claims": ["raw tensor verification by this module", "physical NIC bytes", "GPU kernel service from wall clocks",
                             "p95/p99 request tails", "production SLO goodput", "automatic performance victory", "E4 completion"],
    }


def _sha(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            label + " must be a lowercase SHA256")


def _number(value, label, *, optional=False, positive=False):
    if optional and value is None:
        return
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), label + " must be finite and nonnegative")


def _integer(value, label, *, optional=False):
    if optional and value is None:
        return
    require(type(value) is int and value >= 0, label + " must be a nonnegative integer")


def _artifact(item, label, *, optional=False):
    if optional and item is None:
        return
    require(isinstance(item, dict) and isinstance(item.get("path"), str) and item["path"], label + " path missing")
    _sha(item.get("sha256"), label + " hash")


def _optional_sha(value, label, successful):
    if value is not None or successful:
        _sha(value, label)


def _migration(item, spec, accepted, settled, successful):
    if item is None and not successful:
        return
    require(isinstance(item, dict), "Actual migration record missing")
    keys = ("first_gpu_ready_s", "gpu_ready_s", "drained_s")
    require(all(k in item for k in keys + ("payload_bytes", "complete_layers")), "Incomplete migration schema")
    for key in keys:
        _number(item[key], key, optional=not successful)
    for key in ("payload_bytes", "complete_layers"):
        _integer(item[key], key, optional=not successful)
    known = [v for v in (accepted, *(item[k] for k in keys), settled) if v is not None]
    require(known == sorted(known), "Migration readiness/drain/own settlement order violated")
    if item["complete_layers"] is not None:
        require(item["complete_layers"] <= 48, "More complete layers than model")
    if item["payload_bytes"] is not None:
        require(item["payload_bytes"] <= spec["prefix_tokens"] * 196608, "Extra/unbound migration payload")
    if successful:
        require(item["complete_layers"] == 48 and item["payload_bytes"] == spec["prefix_tokens"] * 196608,
                "Successful repair/transfer must deliver all actual 48-layer KV payload")


def _job(item, spec, record, *, transfer=False):
    require(isinstance(item, dict) and item.get("role") == spec["role"], "Offered role/order changed")
    require(item.get("spec_sha256") == digest(spec), "Request/action/geometry identity changed")
    require(item.get("status") in STATUSES and type(item.get("cleanup_passed")) is bool, "Job status/cleanup missing")
    successful = item["status"] == "passed"
    require(record["status"] != "passed" or successful, "Successful trial contains unsuccessful offered work")
    _optional_sha(item.get("input_ids_sha256"), "Actual input token hash", successful)
    accepted = record["timing"]["accepted_s"]
    settled = item.get("own_settled_s")
    _number(settled, "Own settlement", optional=not successful)
    if successful:
        require(item["cleanup_passed"] is True and accepted is not None, "Successful job lacks actual cleanup/acceptance")
    if accepted is not None and settled is not None:
        require(settled > accepted, "Own settlement must follow acceptance")
    if not transfer:
        _optional_sha(item.get("output_ids_sha256"), "Actual output token hash", successful)
        entered = item.get("executor_entered_s")
        _number(entered, "Executor entry", optional=not successful)
        tokens = item.get("token_delivery_s")
        require(isinstance(tokens, list), "Retain actual delivered token timestamps, including empty failures")
        require(len(tokens) <= spec["executor_only"]["realized_stop_tokens"], "Extra generated tokens")
        for timestamp in tokens:
            _number(timestamp, "Token delivery")
        require(tokens == sorted(tokens), "Token delivery timestamps out of order")
        known = [x for x in (accepted, entered, *tokens, settled) if x is not None]
        require(known == sorted(known), "Request acceptance/entry/token/cleanup order violated")
        if successful:
            require(len(tokens) == spec["executor_only"]["realized_stop_tokens"], "Successful request incomplete")
            require(tokens[0] > accepted and tokens[-1] > tokens[0], "Nonpositive completion/ITL denominator")
    if transfer or spec["action"] == "repair":
        _migration(item.get("migration"), spec, accepted, settled, successful)
    else:
        require(item.get("migration") is None, "Remote-only job must have null migration, not fabricated zero service")


def _resources(value, trial, passed):
    require(isinstance(value, dict), "Actual resource accounting required")
    for key in ("application_sent_bytes", "compute_service_s", "compute_queue_wait_s", "background", "contention"):
        require(isinstance(value.get(key), dict) and set(value[key]) == set(ENDPOINTS), "Both endpoint resource counters required")
    for side in ENDPOINTS:
        _integer(value["application_sent_bytes"][side], "Successful application sends", optional=not passed)
        for key in ("compute_service_s", "compute_queue_wait_s"):
            _number(value[key][side], key, optional=not passed)
        bg = value["background"][side]
        require(isinstance(bg, dict) and set(bg) == {"offered", "completed", "failed", "pending"}, "Background offered denominators missing")
        for key in bg:
            _integer(bg[key], "Background " + key, optional=not passed)
        if all(v is not None for v in bg.values()):
            require(bg["offered"] == bg["completed"] + bg["failed"] + bg["pending"], "Dropped background work")
            if passed:
                require(bg["failed"] == bg["pending"] == 0, "Successful trial has unfinished/failed background")
            expected_side = trial["load_location"] if trial["kind"] == "group" else "none"
            if side != expected_side:
                require(bg["offered"] == 0, "Undeclared background load or loaded solo")
        c = value["contention"][side]
        require(isinstance(c, dict) and set(c) == {"coeligible_grant_count", "successful_coeligible_grant_count", "bulk_max_eligible_wait_s"},
                "Actual same-sample contention counters required")
        for key in ("coeligible_grant_count", "successful_coeligible_grant_count"):
            _integer(c[key], key, optional=not passed)
        _number(c["bulk_max_eligible_wait_s"], "Bulk eligible wait", optional=not passed)
        if c["coeligible_grant_count"] is not None and c["successful_coeligible_grant_count"] is not None:
            require(c["successful_coeligible_grant_count"] <= c["coeligible_grant_count"], "Successful coeligible grants exceed all coeligible grants")


def validate_record(record, trial, protocol):
    """Check one actual record's structural contract; does not read raw files."""
    require(isinstance(record, dict) and record.get("schema_version") == SCHEMA_VERSION, "Normalized record version missing")
    require(all(record.get(k) == trial[k] for k in IDENTITY), "Trial identity changed")
    require(record.get("plan_sha256") == protocol["plan_sha256"] and record.get("trial_sha256") == digest(trial), "Plan/trial binding mismatch")
    for key in ("run_id", "attempt_id"):
        require(isinstance(record.get(key), str) and record[key], "Run/attempt identity missing")
    _sha(record.get("cohort_sha256"), "Actual physical cohort hash")
    require(record.get("model_revision") == protocol["cohort"]["model_revision"]
            and record.get("model_manifest_sha256") == protocol["cohort"]["model_manifest_sha256"], "Wrong model/revision")
    sources = record.get("source_sha256")
    require(isinstance(sources, dict) and sources and all(isinstance(k, str) and k for k in sources), "Source closure missing")
    for value in sources.values():
        _sha(value, "Source file hash")
    require(record.get("status") in STATUSES and all(type(record.get(k)) is bool for k in GATES), "Actual status/five gates required")
    successful = record["status"] == "passed"
    require(record.get("E4_complete", False) is False, "Normalized records cannot grant stage completion")
    artifacts = record.get("raw_artifacts")
    require(isinstance(artifacts, dict), "Raw report bindings missing")
    _artifact(artifacts.get("candidate"), "Candidate")
    _artifact(artifacts.get("diagnostic"), "Separate diagnostic", optional=not successful)
    refs = artifacts.get("references")
    require(isinstance(refs, dict) and set(refs) == {s["role"] for s in trial["requests"]}, "Exact per-request HF references required")
    for ref in refs.values():
        _artifact(ref, "HF reference", optional=not successful)
    if successful and trial["transfer_jobs"]:
        contents = artifacts["diagnostic"].get("contents_sha256")
        require(isinstance(contents, dict) and set(contents) == {s["role"] for s in trial["transfer_jobs"]}, "Separate transfer diagnostic content evidence missing")
        for value in contents.values():
            _sha(value, "Actual diagnostic prefix content hash")
    timing = record.get("timing")
    require(isinstance(timing, dict) and all(k in timing for k in ("accepted_s", "group_service_settled_s", "group_ownership_settled_s")), "Separate service/ownership timings required")
    times = [timing[k] for k in ("accepted_s", "group_service_settled_s", "group_ownership_settled_s")]
    for value in times:
        _number(value, "Group timing", optional=not successful)
    known = [v for v in times if v is not None]
    require(known == sorted(known), "Group timing order violated")
    if successful:
        require(times[1] > times[0], "Nonpositive service makespan")
    for field, transfer in (("requests", False), ("transfer_jobs", True)):
        offered = record.get(field)
        require(isinstance(offered, list) and len(offered) == len(trial[field]), "Missing/dropped offered job")
        for item, spec in zip(offered, trial[field]):
            _job(item, spec, record, transfer=transfer)
            if item.get("own_settled_s") is not None and times[1] is not None:
                require(item["own_settled_s"] <= times[1], "Group service settled before own job cleanup")
    _resources(record.get("resources"), trial, successful)
    return True


def validate_records(records, protocol, *, require_complete=True, require_passed=False):
    """Inventory retains failed/offered work; final analyze requires every gate.

    Records may arrive in any order. Duplicate trial/attempt IDs, hidden retries,
    mixed cohorts/sources, missing planned slots and reused report paths fail.
    """
    validate_plan(protocol)
    require(isinstance(records, list), "Records must be a list")
    index = {t["trial_id"]: t for t in protocol["trials"]}
    seen, attempts, paths = set(), set(), set()
    binding = None
    failures = []
    job_statuses, background = Counter(), Counter()
    for record in records:
        key = record.get("trial_id") if isinstance(record, dict) else None
        require(key in index and key not in seen, "Foreign/duplicate trial: " + str(key))
        seen.add(key)
        validate_record(record, index[key], protocol)
        require(record["attempt_id"] not in attempts, "Duplicate attempt or hidden retry")
        attempts.add(record["attempt_id"])
        identity = (record["run_id"], record["cohort_sha256"], digest(record["source_sha256"]))
        if binding is None:
            binding = identity
        require(identity == binding, "Mixed run/cohort/source in one experiment")
        raw = record["raw_artifacts"]
        for artifact in [raw["candidate"], raw["diagnostic"], *raw["references"].values()]:
            if artifact is not None:
                require(artifact["path"] not in paths, "Raw artifact reused instead of independent execution")
                paths.add(artifact["path"])
        for item in record["requests"] + record["transfer_jobs"]:
            job_statuses[item["status"]] += 1
        for side in ENDPOINTS:
            for label, count in record["resources"]["background"][side].items():
                if count is not None:
                    background[label] += count
        failed_gates = [name for name in GATES if record[name] is not True]
        if record["status"] != "passed" or failed_gates:
            failures.append({"trial_id": key, "status": record["status"], "failed_gates": failed_gates,
                             "offered_jobs": [{"role": j["role"], "status": j["status"]} for j in record["requests"] + record["transfer_jobs"]]})
    missing = [t["trial_id"] for t in protocol["trials"] if t["trial_id"] not in seen]
    require(not require_complete or not missing, "Missing planned attempts; failures may not be omitted")
    require(not require_passed or not failures, "Unaccepted actual evidence; retain failure inventory, do not produce a paired formal summary")
    return {"expected_attempts": len(index), "recorded_attempts": len(records), "missing_trial_ids": missing,
        "counts_by_kind": dict(Counter(r["kind"] for r in records)),
        "counts_by_role_and_kind": {role: dict(Counter(r["kind"] for r in records if r["sample_role"] == role))
                                    for role in sorted({r["sample_role"] for r in records})},
        "offered_decoded_requests": sum(len(r["requests"]) for r in records),
        "offered_transfer_jobs": sum(len(r["transfer_jobs"]) for r in records),
        "offered_job_statuses": dict(job_statuses), "background": dict(background),
        "failures": failures, "all_offered_gates_passed": not failures and not missing,
        "raw_files_reopened_by_analysis": False, "E4_complete": False}


def request_metrics(item, accepted):
    tokens = item["token_delivery_s"]
    itls = [b - a for a, b in zip(tokens, tokens[1:])]
    return {"ttft_s": tokens[0] - accepted, "completion_s": tokens[-1] - accepted,
            "settled_s": item["own_settled_s"] - accepted,
            "mean_itl_s": statistics.fmean(itls), "max_itl_s": max(itls), "itl_count": len(itls),
            "itl_intervals_s": itls, "delivered_tokens": len(tokens)}


def migration_metrics(item, accepted):
    migration = item["migration"]
    if migration is None:
        return None
    return {"first_gpu_ready_s": migration["first_gpu_ready_s"] - accepted,
            "gpu_ready_s": migration["gpu_ready_s"] - accepted,
            "drained_s": migration["drained_s"] - accepted,
            "own_settled_s": item["own_settled_s"] - accepted,
            "payload_bytes": migration["payload_bytes"], "complete_layers": migration["complete_layers"]}


def jain_reciprocal_slowdown(slowdowns):
    require(bool(slowdowns) and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in slowdowns), "Positive actual slowdown denominators required")
    inv = [1. / v for v in slowdowns]
    return sum(inv) ** 2 / (len(inv) * sum(v * v for v in inv))


def _migration_pair(item, solo, accepted, solo_accepted):
    actual, isolated = migration_metrics(item, accepted), migration_metrics(solo, solo_accepted)
    if actual is None:
        require(isolated is None, "Migration work differs between group and solo")
        return None
    require(isolated is not None and actual["payload_bytes"] == isolated["payload_bytes"], "Actual migration work differs")
    output = {"actual": actual, "solo": isolated}
    for name in ("first_gpu_ready_s", "gpu_ready_s", "drained_s", "own_settled_s"):
        require(isolated[name] > 0, "Nonpositive actual solo migration denominator")
        output[name.removesuffix("_s") + "_slowdown"] = actual[name] / isolated[name]
        output[name.removesuffix("_s") + "_delay_s"] = actual[name] - isolated[name]
    return output


def normalize_group(group, solo_records, trial, trial_lookup):
    """Derive a group's metrics only from its exact separately executed solos.

    Caller validates records against the frozen protocol first. This helper is
    independently testable with a small genuine-shape CPU fixture.
    """
    require(trial["kind"] == group["kind"] == "group", "Group required")
    require(len(solo_records) == len(trial["solo_trial_ids"]) and
            {r["trial_id"] for r in solo_records} == set(trial["solo_trial_ids"]), "Wrong/missing explicit solo binding")
    by_role = {}
    for solo in solo_records:
        st = trial_lookup[solo["trial_id"]]
        require(st["paired_group_trial_id"] == trial["trial_id"] and st["scheduler"] == trial["scheduler"]
                and st["method"] == trial["method"] and st["block_id"] == trial["block_id"]
                and st["load_location"] == "none", "Solo action/scheduler/input block pairing changed")
        require(st["normalized_role"] not in by_role, "Duplicate solo role")
        by_role[st["normalized_role"]] = (solo, st)
    accepted = group["timing"]["accepted_s"]
    request_rows, transfer_rows = [], []
    for field, output in (("requests", request_rows), ("transfer_jobs", transfer_rows)):
        for item, spec in zip(group[field], trial[field]):
            solo, st = by_role[spec["role"]]
            require(st[field] == [spec] and len(solo[field]) == 1, "Solo role/action/P/S/G mismatch")
            si = solo[field][0]
            require(item["spec_sha256"] == si["spec_sha256"] == digest(spec)
                    and item["input_ids_sha256"] == si["input_ids_sha256"], "Actual solo/group input differs")
            row = {"role": spec["role"], "action": spec["action"], "solo_trial_id": solo["trial_id"],
                   "input_ids_sha256": item["input_ids_sha256"],
                   "migration": _migration_pair(item, si, accepted, solo["timing"]["accepted_s"])}
            if field == "requests":
                require(item["output_ids_sha256"] == si["output_ids_sha256"], "Actual solo/group greedy output differs")
                row["actual"] = request_metrics(item, accepted)
                row["solo"] = request_metrics(si, solo["timing"]["accepted_s"])
                for name in ("ttft_s", "completion_s", "settled_s", "mean_itl_s"):
                    require(row["solo"][name] > 0, "Nonpositive actual solo request denominator")
                    row[name.removesuffix("_s") + "_slowdown"] = row["actual"][name] / row["solo"][name]
                row["output_ids_sha256"] = item["output_ids_sha256"]
            else:
                content = group["raw_artifacts"]["diagnostic"]["contents_sha256"][spec["role"]]
                require(content == solo["raw_artifacts"]["diagnostic"]["contents_sha256"][spec["role"]], "Actual independent transfer diagnostic content differs")
                row["diagnostic_content_sha256"] = content
            output.append(row)
    service = group["timing"]["group_service_settled_s"] - accepted
    ownership = group["timing"]["group_ownership_settled_s"] - accepted
    completion = [r["completion_slowdown"] for r in request_rows]
    settled = [r["settled_slowdown"] for r in request_rows]
    b = [r["completion_slowdown"] for r in request_rows if r["role"] != "A"]
    resource = copy.deepcopy(group["resources"])
    migration_work = sum(item["migration"]["payload_bytes"] for item in group["requests"] + group["transfer_jobs"] if item["migration"] is not None)
    return {**{k: group[k] for k in IDENTITY}, "requests": request_rows, "transfer_jobs": transfer_rows,
        "group_service_settled_latency_s": service, "group_ownership_latency_s": ownership,
        "post_service_ownership_latency_s": ownership - service,
        "jain_reciprocal_completion_slowdown": jain_reciprocal_slowdown(completion),
        "jain_reciprocal_settled_slowdown": jain_reciprocal_slowdown(settled),
        "worst_completion_slowdown": max(completion), "worst_settled_slowdown": max(settled),
        "worst_B_completion_slowdown": max(b) if b else None,
        "migration_payload_bytes": migration_work,
        "equal_migration_work_comparison_eligible": trial["method"] != "no_migration_reference",
        "finite_group_generated_tokens_per_s": sum(r["actual"]["delivered_tokens"] for r in request_rows) / service,
        "finite_group_successful_requests_per_s": len(request_rows) / service,
        "resources": resource, "application_sent_bytes_total": sum(resource["application_sent_bytes"].values()),
        "physical_wire_bytes": None, "kernel_active_time_s": None,
        "actual_successful_coeligible_grants": sum(resource["contention"][s]["successful_coeligible_grant_count"] for s in ENDPOINTS),
        "contention_observed_at_successful_grant": any(resource["contention"][s]["successful_coeligible_grant_count"] > 0 for s in ENDPOINTS),
        "bulk_max_eligible_wait_s": max(resource["contention"][s]["bulk_max_eligible_wait_s"] for s in ENDPOINTS),
        "E4_complete": False}


def _metric_points(group):
    a = next(r for r in group["requests"] if r["role"] == "A")
    points = {"A_ttft_s": a["actual"]["ttft_s"], "A_completion_s": a["actual"]["completion_s"],
              "A_mean_itl_s": a["actual"]["mean_itl_s"], "A_max_itl_s": a["actual"]["max_itl_s"]}
    for key in ("group_service_settled_latency_s", "group_ownership_latency_s", "post_service_ownership_latency_s",
                "jain_reciprocal_completion_slowdown", "jain_reciprocal_settled_slowdown", "worst_completion_slowdown",
                "worst_settled_slowdown", "worst_B_completion_slowdown", "finite_group_generated_tokens_per_s",
                "finite_group_successful_requests_per_s", "application_sent_bytes_total", "actual_successful_coeligible_grants", "bulk_max_eligible_wait_s"):
        points[key] = group[key]
    migrations = [r["migration"] for r in group["requests"] + group["transfer_jobs"] if r["migration"] is not None]
    for key in ("gpu_ready", "drained", "own_settled"):
        points["max_migration_" + key + "_s"] = max((m["actual"][key + "_s"] for m in migrations), default=None)
        points["max_migration_" + key + "_slowdown"] = max((m[key + "_slowdown"] for m in migrations), default=None)
        points["max_migration_" + key + "_delay_s"] = max((m[key + "_delay_s"] for m in migrations), default=None)
    return points


def _percentile(sorted_values, fraction):
    position = (len(sorted_values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (position - lo)


def joint_bootstrap_indices(block_ids):
    require(len(block_ids) == len(set(block_ids)) == 8, "Exactly eight independent blocks required")
    rng = random.Random(BOOTSTRAP_SEED)
    return [tuple(rng.randrange(8) for _ in range(8)) for _ in range(BOOTSTRAP_RESAMPLES)]


def _interval(values, resamples):
    boot = sorted(statistics.median(values[i] for i in draw) for draw in resamples)
    return [_percentile(boot, .025), _percentile(boot, .975)]


def paired_contrast(candidate, baseline, resamples, *, practical_tie_applicable=False):
    require(len(candidate) == len(baseline) == 8 and all(math.isfinite(v) for v in candidate + baseline), "Eight finite paired block values required")
    differences = [a - b for a, b in zip(candidate, baseline)]
    relative = [(b - a) / b if b > 0 else None for a, b in zip(candidate, baseline)]
    relative_valid = all(v is not None for v in relative)
    effect = statistics.median(relative) if relative_valid else None
    return {"candidate_block_points": candidate, "baseline_block_points": baseline,
        "candidate_minus_baseline_block_points": differences, "median_difference": statistics.median(differences),
        "median_difference_descriptive_95pct_interval": _interval(differences, resamples),
        "relative_reduction_block_points": relative, "median_relative_reduction": effect,
        "median_relative_reduction_descriptive_95pct_interval": _interval(relative, resamples) if relative_valid else None,
        "relative_reduction_undefined_reason": None if relative_valid else "One or more actual baseline values are zero; no epsilon or dropped blocks",
        "practical_tie_within_5pct": abs(effect) <= .05 if practical_tie_applicable and effect is not None else None,
        "tie_scope": "Descriptive median paired relative change; not significance or a winner" if practical_tie_applicable else None,
        "automatic_performance_claim": False}


def _pareto(groups):
    """Raw per-block dominance, only equal actual offered migration service."""
    cells = defaultdict(list)
    for row in groups:
        if row["equal_migration_work_comparison_eligible"]:
            cells[(row["cell_id"], row["block_id"])].append(row)
    result = []
    for (cell, block), rows in sorted(cells.items()):
        require(len(rows) == 5 and len({r["migration_payload_bytes"] for r in rows}) == 1, "Pareto work/composition differs")
        vectors = {}
        for row in rows:
            p = _metric_points(row)
            vector = {k: p[k] for k in ("A_completion_s", "A_mean_itl_s", "group_service_settled_latency_s", "max_migration_drained_s")}
            for request in row["requests"]:
                if request["role"] != "A":
                    vector[request["role"] + "_completion_slowdown"] = request["completion_slowdown"]
            vectors[row["method"]] = vector
        for method, vector in vectors.items():
            dominators = [name for name, other in vectors.items() if name != method
                          and all(other[k] <= vector[k] for k in vector)
                          and any(other[k] < vector[k] for k in vector)]
            result.append({"cell_id": cell, "block_id": block, "method": method, "cost_vector": vector,
                           "strict_raw_dominators": dominators, "scope": "Descriptive single-block vector, no uncertainty-adjusted superiority"})
    return result


def analyze(records, protocol):
    """Strict complete full-matrix analysis. E4_complete remains false."""
    inventory = validate_records(records, protocol, require_complete=True, require_passed=True)
    require(protocol["mode"] == "e4_full", "Smoke/profile data cannot produce full E4 statistics")
    bootstrap = protocol["design"]["statistics_and_reporting"]["bootstrap"]
    require(bootstrap["resamples"] == BOOTSTRAP_RESAMPLES and bootstrap["seed"] == BOOTSTRAP_SEED
            and bootstrap["confidence_level_descriptive"] == .95, "Unsupported preregistered bootstrap changed")
    lookup = {t["trial_id"]: t for t in protocol["trials"]}
    actual = {r["trial_id"]: r for r in records}
    groups = [normalize_group(actual[t["trial_id"]], [actual[k] for k in t["solo_trial_ids"]], t, lookup)
              for t in protocol["trials"] if t["kind"] == "group"]
    measured = [r for r in groups if r["sample_role"] == "measured"]
    warmups = [r for r in groups if r["sample_role"] == "warmup"]
    require(len(groups) == 648 and len(measured) == 576 and len(warmups) == 72, "Wrong analysis denominators")
    block_ids = [b["block_id"] for b in protocol["design"]["inputs_repetition_and_order"]["heldout_blocks"]]
    resamples = joint_bootstrap_indices(block_ids)
    blocks = defaultdict(dict)
    for row in measured:
        key = (row["cell_id"], row["method"])
        require(row["block_id"] not in blocks[key], "Duplicate independent block")
        blocks[key][row["block_id"]] = _metric_points(row)
    summaries = []
    for (cell, method), rows in sorted(blocks.items()):
        require(set(rows) == set(block_ids), "Unbalanced paired blocks")
        summary = {"cell_id": cell, "method": method, "n_independent_blocks": 8,
                   "block_ids": block_ids, "equal_migration_work_comparison_eligible": method != "no_migration_reference", "metrics": {}}
        for metric in rows[block_ids[0]]:
            values = [rows[b][metric] for b in block_ids]
            require(all(v is None for v in values) or all(v is not None for v in values), "Partial undefined metric would drop blocks")
            summary["metrics"][metric] = {"block_points": values,
                "median": statistics.median(values) if values[0] is not None else None,
                "minimum": min(values) if values[0] is not None else None,
                "maximum": max(values) if values[0] is not None else None}
        summaries.append(summary)
    primary = []
    endpoints = {
        ("I02", "fine_fifo"): ("A_mean_itl_s", "A_completion_s", "max_migration_gpu_ready_s", "max_migration_own_settled_s", "max_migration_own_settled_slowdown"),
        ("C02", "strict_rpc_first"): ("A_mean_itl_s", "worst_B_completion_slowdown", "group_service_settled_latency_s", "max_migration_gpu_ready_s", "bulk_max_eligible_wait_s"),
        ("C02", "fixed_weighted_3to1"): ("A_mean_itl_s", "worst_B_completion_slowdown", "group_service_settled_latency_s", "max_migration_gpu_ready_s", "bulk_max_eligible_wait_s"),
    }
    for (cell, baseline), metrics in endpoints.items():
        for metric in metrics:
            candidate = [blocks[(cell, "age_bounded_rpc_first")][b][metric] for b in block_ids]
            reference = [blocks[(cell, baseline)][b][metric] for b in block_ids]
            primary.append({"cell_id": cell, "candidate": "age_bounded_rpc_first", "baseline": baseline, "metric": metric,
                "block_ids": block_ids, **paired_contrast(candidate, reference, resamples,
                    practical_tie_applicable=metric in ("A_mean_itl_s", "A_completion_s", "max_migration_gpu_ready_s", "max_migration_own_settled_s"))})
    first = records[0]
    return {"schema_version": 1, "stage": "E4", "mode": "full_analysis", "passed": True,
        "run_id": first["run_id"], "plan_sha256": protocol["plan_sha256"], "design_sha256": protocol["design_sha256"],
        "source_sha256": first["source_sha256"], "cohort_sha256": first["cohort_sha256"],
        "model_revision": first["model_revision"], "model_manifest_sha256": first["model_manifest_sha256"],
        "normalized_records_sha256": digest(records), "inventory": inventory,
        "measured_group_count": 576, "excluded_warmup_group_count": 72, "independent_block_count": 8,
        "measured_groups": measured, "excluded_warmup_groups": warmups, "cell_method_summaries": summaries,
        "primary_paired_contrasts": primary, "equal_migration_work_raw_pareto": _pareto(measured),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "block_ids": block_ids,
            "joint_draws_sha256": digest(resamples), "statistic": "Median paired block difference / relative reduction",
            "confidence_level_descriptive": .95, "same_draws_used_for_every_contrast": True,
            "secondary_summary_scope": "All 12 cells and 6 methods expose all 8 points and median/range; intervals are preregistered primary contrasts"},
        "contention_observed_groups": sum(r["contention_observed_at_successful_grant"] for r in measured),
        "zero_observed_contention_groups": sum(not r["contention_observed_at_successful_grant"] for r in measured),
        "negative_and_zero_results_retained": True, "no_migration_reference_equal_work_pareto_excluded": True,
        "raw_files_reopened_by_analysis": False, "raw_gate_scope": "Five actual gates and SHA-bound record inputs must be independently audited; this CPU analysis verifies their structure and pairings",
        "throughput_scope": "Actual finite-group completed tokens/requests per service wall time; not production SLO goodput",
        "queue_scope": "Host-wall ready-operation intervals ending at CUDA completion, not kernel-active duration",
        "timing_scope": "Only destination-local absolute times are subtracted; service settlement excludes subsequent unpaced audit/ownership exchange",
        "starvation_scope": "Actual maximum eligible wait is observed, not starvation-freedom or a tail guarantee",
        "uncertainty_scope": "Eight paired blocks, no power guarantee; requests/tokens/cells are not additional independent samples",
        "physical_wire_bytes": None, "selected_GPU_hours": None,
        "GPU_hours_scope": "Need separate actual per-host session ownership intervals; never derive them from per-group service makespan",
        "request_p95_p99_reported": False, "SLO_claim": False, "automatic_performance_claim": False,
        "E4_complete": False, "paper_stage_complete": False, "pending_profile_groups": 16,
        "pending_physical_rank_traces": 32, "pending_independent_review": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory-only", action="store_true", help="Retain failures without generating formal paired statistics")
    parser.add_argument("--contract", action="store_true", help="Write the producer contract only")
    args = parser.parse_args(argv)
    if args.contract:
        require(args.plan is None and args.records is None and not args.inventory_only, "Contract export has no result inputs")
        result = record_contract()
    else:
        require(args.plan is not None and args.records is not None, "Plan and actual records required")
        protocol = json.loads(args.plan.read_text(encoding="utf-8"))
        payload = json.loads(args.records.read_text(encoding="utf-8"))
        records = payload.get("records") if isinstance(payload, dict) else payload
        if isinstance(payload, dict):
            require(payload.get("plan_sha256") == protocol["plan_sha256"], "Records wrapper plan mismatch")
        result = validate_records(records, protocol) if args.inventory_only else analyze(records, protocol)
    require(not args.output.exists(), "Refuse to overwrite an existing analysis artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
