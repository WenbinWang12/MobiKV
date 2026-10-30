"""Experiments 3/4/5 analysis: block-paired contrasts per (cell, rate) and per RTT, plus action selection rates.

Same rules as experiment 1: relative improvement (B - M) / B on lower-is-better metrics,
(M - B) / B otherwise; zero baseline -> null relative value (absolute difference kept);
labels advantage / disadvantage / tied_or_unresolved with the 5% band and a 10,000-draw
block bootstrap; a pair with a missing/failed trial is 'incomplete', never dropped.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path
import sys

from mh_analysis import trial_metrics_e7, write_csv

DRAWS, SEED = 10000, 2026102802
LOWER = {"mean_completion_s", "p90_completion_s"}
METRICS = ("mean_completion_s", "slo_attainment", "goodput_rps", "p90_completion_s")
X3_PAIRS = [("full_controller", b) for b in ("ablate_no_contention", "static_holder", "static_recompute", "static_repair",
                                             "threshold_bstar")] + [("static_repair", "static_recompute"),
                                                                    ("static_holder", "static_recompute")]
X4_PAIRS = [("full_controller~noholder", "static_recompute"), ("full_controller~noholder", "static_repair"),
            ("full_controller~noholder", "ablate_no_contention~noholder"), ("full_controller", "full_controller~noholder")]
X5_PAIRS = [("static_repair", "static_holder"), ("full_controller", "static_repair"), ("full_controller", "static_holder")]

E2_WAIVER = dict(kind="historical_E2_raw_dependency_reuse",
    training_support_records_available=False, synthetic_records_created=False,
    coverage_claim_allowed=False,
    reason="Historical E2 raw dependency evidence is unavailable; this run neither reruns E2 nor establishes scientific acceptance.")
E3_WAIVER = dict(kind="historical_E3_raw_dependency_reuse", id="E3-destination-raw-loss-20261023",
    synthetic_records_created=False, coverage_claim_allowed=False,
    original_E3_CPU_auditor_rerun=False,
    reason="Historical E3 destination per-trial raw evidence was lost; these supplemental experiments do not restore it.")


def limitations():
    return dict(scientific_acceptance=False, coverage_claim_allowed=False, dependency_evidence_complete=False,
        E4_complete=False, paper_stage_complete=False, historical_E4_completion_asserted=False,
        e2_raw_dependency_waiver=copy.deepcopy(E2_WAIVER), e3_raw_dependency_waiver=copy.deepcopy(E3_WAIVER),
        raw_artifact_hashes_recomputed=False,
        raw_validation_scope="Analysis checks frozen plan/envelope identity, available record hashes, request specifications and numeric metrics. It does not repeat tensor/socket/diagnostic raw audits.")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def sha_format(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def record_errors(record, trial, plan_sha):
    """Producer v1 lacks per-record plan/trial hashes; envelope binds that schema.

    Explicit failed legacy attempts may omit stage, sample_role and hashes.
    Supplied contradictory identity/hash fields are still reported, never trusted.
    """
    errors = []
    failed = record.get("status") != "passed"
    required = ("trial_id", "cell_id", "method", "block_id") if failed else \
        ("trial_id", "stage", "cell_id", "method", "block_id", "sample_role")
    for key in required:
        if record.get(key) != trial.get(key):
            errors.append("record identity differs: " + key)
    for key in ("stage", "sample_role", "phase"):
        if key in record and record[key] != trial.get(key):
            errors.append("optional record identity differs: " + key)
    for key, expected in (("plan_sha256", plan_sha), ("trial_sha256", digest(trial))):
        if key in record and record[key] != expected:
            errors.append("record SHA differs: " + key)
    artifact = record.get("measured_artifact_sha256")
    if (not failed or artifact is not None) and not sha_format(artifact):
        errors.append("missing/invalid measured_artifact_sha256")
    if failed:
        return errors
    requests = record.get("requests")
    specs = trial["requests"]
    if not isinstance(requests, list) or len(requests) != len(specs):
        return errors + ["passed request cardinality differs from plan"]
    for actual, spec in zip(requests, specs):
        for key in ("role", "action", "holder", "prefix_tokens", "arrival_offset_s", "prefix_location",
                    "slo_ttft_s", "slo_mean_itl_s", "input_id"):
            if actual.get(key) != spec.get(key):
                errors.append("passed request specification differs: %s.%s" % (spec["role"], key))
        if actual.get("delivered_tokens") != spec.get("executor_only", {}).get("realized_stop_tokens", 64):
            errors.append("passed delivered token count differs")
        if not sha_format(actual.get("token_ids_sha256")):
            errors.append("missing/invalid token_ids_sha256")
    try:
        metrics = trial_metrics_e7(record)
        if any(type(metrics[k]) not in (int, float) or not math.isfinite(metrics[k]) or metrics[k] < 0 for k in METRICS):
            errors.append("nonfinite or negative primary metric")
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as error:
        errors.append("cannot derive metrics: " + str(error))
    return errors


def interval(values):
    rng = random.Random(SEED)
    meds = sorted(statistics.median(rng.choice(values) for _ in values) for _ in range(DRAWS))
    return meds[int(.025 * DRAWS)], meds[int(.975 * DRAWS) - 1]


def label(med, lo, hi):
    if med > .05 and lo > .05:
        return "advantage"
    if med < -.05 and hi < -.05:
        return "disadvantage"
    return "tied_or_unresolved"


def contrast(table, unit, focus, base, metric, blocks):
    rel, absdiff, complete, paired_blocks = [], [], True, []
    for b in blocks:
        f, v = table.get(unit + (b, focus)), table.get(unit + (b, base))
        if f is None or v is None:
            complete = False
            rel.append(None)
            absdiff.append(None)
            continue
        paired_blocks.append(b)
        fv, bv = f[metric], v[metric]
        absdiff.append(bv - fv if metric in LOWER else fv - bv)
        rel.append(None if bv == 0 else ((bv - fv) / bv if metric in LOWER else (fv - bv) / bv))
    row = dict(unit="|".join(map(str, unit)), method=focus, baseline=base, metric=metric, n_planned_blocks=len(blocks),
               n_paired_blocks=len(paired_blocks), blocks=blocks, paired_blocks=paired_blocks,
               block_relative=rel, block_absolute=absdiff,
               absolute_median=statistics.median(absdiff) if complete and absdiff else None,
               observed_absolute_median=statistics.median([v for v in absdiff if v is not None]) if paired_blocks else None,
               observed_absolute_scope="Diagnostic available-pair value only; primary absolute median requires all planned blocks")
    if not complete or any(x is None for x in rel):
        row.update(median=None, ci_low=None, ci_high=None, label="relative_undefined" if complete else "incomplete")
    else:
        lo, hi = interval(rel)
        med = statistics.median(rel)
        row.update(median=med, ci_low=lo, ci_high=hi, label=label(med, lo, hi))
    return row


def load(stage_dir):
    stage_dir = Path(stage_dir)
    plan_path = stage_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    require(sha_format(plan.get("plan_sha256")), "Missing/invalid frozen plan SHA")
    require(digest({k: v for k, v in plan.items() if k != "plan_sha256"}) == plan["plan_sha256"], "Frozen plan self-SHA differs")
    by_trial = {t["trial_id"]: t for t in plan["trials"]}
    require(len(by_trial) == len(plan["trials"]), "Duplicate planned trial IDs")
    require(plan.get("counts", {}).get("trials", len(by_trial)) == len(by_trial), "Frozen trial count differs")
    path = stage_dir / "normalized-records.json"
    payload = json.loads(path.read_text()) if path.exists() else None
    if payload is not None:
        require(payload.get("plan_sha256") == plan["plan_sha256"], "Normalized envelope plan SHA differs")
    records = payload["records"] if payload is not None else []
    grouped, ledger, integrity_errors = defaultdict(list), [], []
    for i, record in enumerate(records):
        tid, status = record.get("trial_id"), record.get("status")
        trial = by_trial.get(tid)
        errors = record_errors(record, trial, plan["plan_sha256"]) if trial is not None else ["unplanned trial ID"]
        if status not in ("passed", "failed", "timeout", "cancelled", "not_started"):
            errors.append("unknown attempt status")
        missing_hashes = [k for k in ("plan_sha256", "trial_sha256", "measured_artifact_sha256") if k not in record]
        entry = dict(record_index=i, trial_id=tid, attempt_id=record.get("attempt_id"), session_id=record.get("session_id"),
            status=status, in_frozen_plan=trial is not None, binding_errors=errors,
            observed_record_sha256=digest(record), expected_trial_sha256=digest(trial) if trial else None,
            record_plan_sha256=record.get("plan_sha256"), record_trial_sha256=record.get("trial_sha256"),
            measured_artifact_sha256=record.get("measured_artifact_sha256"), missing_producer_hash_fields=missing_hashes,
            legacy_failed_attempt=status != "passed" and bool(missing_hashes),
            sample_role=trial.get("sample_role") if trial else record.get("sample_role"),
            cell_id=trial.get("cell_id") if trial else record.get("cell_id"),
            method=trial.get("method") if trial else record.get("method"),
            block_id=trial.get("block_id") if trial else record.get("block_id"), error=record.get("error"))
        ledger.append(entry)
        grouped[tid].append((record, entry))
        if errors:
            integrity_errors.append(dict(record_index=i, trial_id=tid, errors=errors))
    passed, inventory = {}, []
    for trial in plan["trials"]:
        tid = trial["trial_id"]
        attempts = grouped[tid]
        successes = [(r, e) for r, e in attempts if r.get("status") == "passed" and not e["binding_errors"]]
        attempted_successes = sum(r.get("status") == "passed" for r, _ in attempts)
        ambiguous = attempted_successes > 1
        if ambiguous:
            integrity_errors.append(dict(trial_id=tid, errors=["multiple passed records; no outcome-selected overwrite allowed"]))
        valid = len(successes) == 1 and not ambiguous and not any(e["binding_errors"] for _, e in attempts)
        if valid:
            passed[tid] = successes[0][0]
        failures = [(r, e) for r, e in attempts if r.get("status") != "passed"]
        prior_failures = [e for _, e in failures if valid and e["record_index"] < successes[0][1]["record_index"]]
        inventory.append(dict(trial_id=tid, **{k: trial[k] for k in ("stage", "cell_id", "method", "block_id", "sample_role")},
            planned_requests=len(trial["requests"]), recorded_attempts=len(attempts), recorded_passed_attempts=attempted_successes,
            recorded_nonpassed_attempts=len(failures), valid_passed_slot=valid,
            recovered_after_failure=valid and bool(prior_failures), ever_failed=bool(failures),
            duplicate_success_ambiguous=ambiguous, attempt_record_indices=[e["record_index"] for _, e in attempts],
            attempt_statuses=[r.get("status") for r, _ in attempts],
            terminal_status="passed_after_failure" if valid and failures else "passed" if valid else
                "missing" if not attempts else "invalid_or_ambiguous" if any(e["binding_errors"] for _, e in attempts) or ambiguous else "failed",
            excludes_warmup=trial["sample_role"] == "warmup"))
    planned = [t for t in plan["trials"] if t["sample_role"] == "measured"]
    counted = Counter(r.get("status", "unknown") for r in records)
    n = len(records)
    audit = dict(**limitations(), plan_self_sha256_verified=True, normalized_envelope_sha256_verified=payload is not None,
        plan_file_sha256=sha(plan_path), normalized_records_file_sha256=sha(path) if path.exists() else None,
        n_planned_trials=len(plan["trials"]), n_planned_measured_trials=len(planned),
        n_planned_warmup_trials=sum(t["sample_role"] == "warmup" for t in plan["trials"]),
        n_planned_requests=sum(len(t["requests"]) for t in plan["trials"]),
        n_recorded_attempts=n, recorded_attempt_status_counts=dict(counted),
        n_recorded_nonpassed_attempts=sum(r.get("status") != "passed" for r in records),
        n_valid_passed_slots=len(passed), n_valid_passed_measured_slots=sum(t["trial_id"] in passed for t in planned),
        n_slots_ever_failed=sum(r["ever_failed"] for r in inventory),
        n_recovered_slots=sum(r["recovered_after_failure"] for r in inventory),
        n_retries=sum(max(0, r["recorded_attempts"] - 1) for r in inventory),
        attempt_pass_fraction=counted.get("passed", 0) / n if n else None,
        plan_slot_completion_fraction=len(passed) / len(plan["trials"]) if plan["trials"] else None,
        execution_complete=len(passed) == len(plan["trials"]) and not integrity_errors,
        failure_free=not any(r.get("status") != "passed" for r in records),
        integrity_passed=not integrity_errors, integrity_errors=integrity_errors,
        n_legacy_failed_attempts=sum(r["legacy_failed_attempt"] for r in ledger),
        attempts=ledger, trial_inventory=inventory,
        denominator_policy="All frozen slots and every observed attempt are reported. A valid unique retry success may supply the planned-slot metric; earlier failures remain in attempt failure rates and recovered-slot counts. Missing/invalid/ambiguous slots make paired effects incomplete.",
        metric_scope="Paired performance uses the unique validated passed outcome per frozen slot; attempt-level failures are separately retained, not assigned an invented latency or erased.",
        hash_scope="Verify plan self-digest and records-envelope plan digest, supplied per-record plan/trial digests, and request identities. Producer-v1 per-record digests may be absent; legacy failed records retain explicit missing-field flags. Raw measured/tensor hashes are retained, not independently rehashed here.",
        negative_zero_failed_results_retained=True)
    return plan, planned, passed, audit


def arm(trial):
    return trial["method"] + ("~noholder" if trial["cell_id"].endswith("@noholder") else "")


def selection(planned, key):
    out = {}
    for t in planned:
        k = key(t)
        row = out.setdefault(k, {"holder": 0, "remote": 0, "repair": 0, "recompute": 0})
        for r in t["requests"]:
            row[r["action"]] += 1
    return [dict(unit=k, **v, planned_request_denominator=sum(v.values()),
                 selection_scope="Frozen planned actions, including failed/unstarted trial slots; not a success-only runtime count",
                 remote_or_repair_share=(v["remote"] + v["repair"]) / sum(v.values()) if sum(v.values()) else None)
            for k, v in sorted(out.items())]


def trial_report(trial, record, slot, **unit):
    row = dict(trial_id=trial["trial_id"], sample_role=trial["sample_role"], **unit,
               status=slot["terminal_status"], valid_passed_slot=slot["valid_passed_slot"],
               recorded_attempts=slot["recorded_attempts"], recorded_nonpassed_attempts=slot["recorded_nonpassed_attempts"],
               recovered_after_failure=slot["recovered_after_failure"])
    row.update(trial_metrics_e7(record) if record is not None else {key: None for key in METRICS})
    return row


def analyze_x34(stage_dir):
    plan, planned, passed, audit = load(stage_dir)
    table, missing = {}, []
    for t in planned:
        cell, label_ = t["cell_id"].split("@")[:2]
        r = passed.get(t["trial_id"])
        if r is None:
            missing.append(t["trial_id"])
            continue
        table[(cell, label_, t["block_id"], arm(t))] = trial_metrics_e7(r)
    blocks = sorted({t["block_id"] for t in planned})
    units = sorted({tuple(t["cell_id"].split("@")[:2]) for t in planned})
    rows = [dict(contrast(table, u, f, b, m, blocks), experiment=x)
            for u in units for x, pairs in (("X3", X3_PAIRS), ("X4", X4_PAIRS)) for f, b in pairs for m in METRICS]
    sel = selection(planned, lambda t: "%s|%s" % ("|".join(t["cell_id"].split("@")[:2]), arm(t)))
    slots = {r["trial_id"]: r for r in audit["trial_inventory"]}
    per_trial = [trial_report(t, passed.get(t["trial_id"]), slots[t["trial_id"]],
        cell=t["cell_id"].split("@")[0], rate=t["cell_id"].split("@")[1], block=t["block_id"], arm=arm(t)) for t in planned]
    return dict(plan_sha256=plan["plan_sha256"], planned=len(planned), missing_or_failed=missing,
                identical_rows=plan.get("identical_rows"), **audit), rows, sel, per_trial


def analyze_x5(stage_dir):
    plan, planned, passed, audit = load(stage_dir)
    table, missing = {}, []
    for t in planned:
        rtt = t["cell_id"].split("@rtt")[1]
        r = passed.get(t["trial_id"])
        if r is None:
            missing.append(t["trial_id"])
            continue
        table[(rtt, t["block_id"], t["method"])] = trial_metrics_e7(r)
    blocks = sorted({t["block_id"] for t in planned})
    rtts = sorted({t["cell_id"].split("@rtt")[1] for t in planned}, key=lambda s: float(s.rstrip("ms")))
    rows = [dict(contrast(table, (rtt,), f, b, m, blocks), experiment="X5") for rtt in rtts for f, b in X5_PAIRS
            for m in METRICS]
    # degradation of each method relative to its own zero-RTT run in the same block
    zero_values = [r for r in rtts if float(r.removesuffix("ms")) == 0]
    require(len(zero_values) == 1, "X5 requires exactly one planned zero-RTT reference; the smallest nonzero RTT is not a substitute")
    zero = zero_values[0]
    for method in sorted({t["method"] for t in planned}):
        for rtt in [r for r in rtts if r != zero]:
            for metric in ("mean_completion_s", "slo_attainment"):
                shifted = {(rtt, b, method + "@0"): table[(zero, b, method)] for b in blocks if (zero, b, method) in table}
                merged = {**table, **shifted}
                rows.append(dict(contrast(merged, (rtt,), method, method + "@0", metric, blocks), experiment="X5-vs-rtt0"))
    slots = {r["trial_id"]: r for r in audit["trial_inventory"]}
    per_trial = [trial_report(t, passed.get(t["trial_id"]), slots[t["trial_id"]],
        rtt=t["cell_id"].split("@rtt")[1], block=t["block_id"], method=t["method"]) for t in planned]
    return dict(plan_sha256=plan["plan_sha256"], planned=len(planned), missing_or_failed=missing, **audit), rows, per_trial


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def export_attempts(out, prefix, summary):
    summary = dict(summary)
    for name in ("attempts", "trial_inventory"):
        rows = summary.pop(name)
        write_csv(out / (prefix + "_" + name + ".csv"), rows)
        write_json(out / (prefix + "_" + name + ".json"), rows)
        summary[name + "_file"] = prefix + "_" + name + ".json"
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    summary = limitations()
    summary.update(analysis_code_sha256=sha(__file__), all_attempts_and_failures_retained=True)
    if (args.root / "X34-full" / "plan.json").exists():
        s, rows, sel, per_trial = analyze_x34(args.root / "X34-full")
        write_csv(out / "x34_paired.csv", rows)
        write_csv(out / "x34_selection.csv", sel)
        write_csv(out / "x34_per_trial.csv", per_trial)
        write_json(out / "x34_paired.json", rows)
        s["labels"] = {"%s|%s|%s|%s" % (r["experiment"], r["unit"], r["method"], r["baseline"]): r["label"]
                       for r in rows if r["metric"] == "mean_completion_s"}
        summary["X34"] = export_attempts(out, "x34", s)
    if (args.root / "X5-full" / "plan.json").exists():
        s, rows, per_trial = analyze_x5(args.root / "X5-full")
        write_csv(out / "x5_paired.csv", rows)
        write_csv(out / "x5_per_trial.csv", per_trial)
        write_json(out / "x5_paired.json", rows)
        s["labels"] = {"%s|%s|%s|%s" % (r["experiment"], r["unit"], r["method"], r["baseline"]): r["label"]
                       for r in rows if r["metric"] == "mean_completion_s"}
        summary["X5"] = export_attempts(out, "x5", s)
    stages = [summary[k] for k in ("X34", "X5") if k in summary]
    require(stages, "No X34/X5 plans found")
    summary["integrity_passed"] = all(s["integrity_passed"] for s in stages)
    summary["execution_complete"] = all(s["execution_complete"] for s in stages)
    summary["n_recorded_attempts"] = sum(s["n_recorded_attempts"] for s in stages)
    summary["n_recorded_nonpassed_attempts"] = sum(s["n_recorded_nonpassed_attempts"] for s in stages)
    summary["failure_free"] = all(s["failure_free"] for s in stages)
    write_json(out / "summary.json", summary)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "labels"})
                      for k, v in summary.items()}, indent=1))
    return 0 if summary["integrity_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
