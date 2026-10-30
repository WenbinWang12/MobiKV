"""CPU-only E1 nominal fitting and descriptive analysis; no execution/acceptance.

Normalized record schema (one authoritative attempt per planned measured trial):
  schema_version=1, plan_sha256, every plan.IDENTITY_FIELDS, public_features,
  attempt_id (unique), executed_action, status='passed'|'failed', metrics={ttft_s,completion_s,
  settled_s}, correctness_passed/diagnostic_passed/traffic_passed/cleanup_passed/
  timing_contract_passed (all true for passed), provenance={cohort_id,
  model_revision,model_manifest_sha256,runtime_source_sha256:{filename:hash},
  measurement_sha256,diagnostic_sha256,reference_sha256}. A passed policy record
  also binds policy_artifact_sha256 to the actual training-fit artifact.
Failed records retain identity, source provenance and measurement hash; their
metrics/diagnostic/reference may be null. Do not pass warmups or mixed splits.
Retries require a separately reviewed authoritative-attempt selection; this API
rejects duplicates instead of choosing favorable results. Supplied hashes are
provenance links, not independent raw-file verification or scientific acceptance.
"""
from __future__ import annotations

import itertools
import math
import re
from typing import Any

import numpy as np

try:
    from .plan import CONTROL, FIXED_ACTIONS, IDENTITY_FIELDS, POLICY, PUBLIC_FEATURES, digest, trial_index, validate_plan
except ImportError:
    from plan import CONTROL, FIXED_ACTIONS, IDENTITY_FIELDS, POLICY, PUBLIC_FEATURES, digest, trial_index, validate_plan


GATES = ("correctness_passed", "diagnostic_passed", "traffic_passed", "cleanup_passed", "timing_contract_passed")
METRICS = ("ttft_s", "completion_s", "settled_s")
HEX = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any, name: str, positive: bool = False) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name} must be numeric")
    value = float(value)
    _require(math.isfinite(value) and (value > 0 if positive else value >= 0), f"invalid {name}")
    return value


def _hash(value: Any, name: str) -> None:
    _require(isinstance(value, str) and bool(HEX.fullmatch(value)), f"invalid {name}")


def nnls_small(matrix: Any, targets: Any) -> dict:
    """Exact active-subset enumeration for 1--3 columns; no scipy dependency.

    Full column rank is required so insufficient training geometry cannot silently
    produce an arbitrary coefficient vector. All candidates are nonnegative.
    The lowest squared error wins; ties use enumeration order deterministically.
    """
    x, y = np.asarray(matrix, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    _require(x.ndim == 2 and y.ndim == 1 and len(y) == len(x), "invalid NNLS shapes")
    _require(1 <= x.shape[1] <= 3 and len(y) >= x.shape[1], "NNLS supports 1--3 sufficiently sampled coefficients")
    _require(bool(np.isfinite(x).all() and np.isfinite(y).all()), "nonfinite NNLS data")
    _require(bool((x >= 0).all() and (y >= 0).all()), "negative NNLS features/targets")
    _require(np.linalg.matrix_rank(x) == x.shape[1], "rank-deficient training geometry")
    best = None
    for mask in range(1 << x.shape[1]):
        active = [j for j in range(x.shape[1]) if mask & (1 << j)]
        coefficient = np.zeros(x.shape[1], dtype=np.float64)
        if active:
            values = np.linalg.lstsq(x[:, active], y, rcond=None)[0]
            tolerance = 1e-12 * max(1.0, float(np.max(np.abs(values))))
            if np.any(values < -tolerance):
                continue
            coefficient[active] = np.maximum(values, 0.0)
        squared_error = float(np.sum((x @ coefficient - y) ** 2))
        if best is None or squared_error < best[0]:
            best = (squared_error, coefficient, mask)
    _require(best is not None and math.isfinite(best[0]), "no finite NNLS solution")
    return {"coefficients": best[1].tolist(), "sum_squared_error": best[0], "active_subset_mask": best[2], "rows": len(y), "rank": x.shape[1]}


def _cohort(provenance: dict, plan: dict) -> dict:
    expected = plan["design"]["cohort"]
    for field, design_field in (("cohort_id", "cluster_id"), ("model_revision", "model_revision"), ("model_manifest_sha256", "model_manifest_sha256")):
        _require(provenance.get(field) == expected[design_field], f"foreign {field}")
    sources = provenance.get("runtime_source_sha256")
    _require(isinstance(sources, dict) and bool(sources), "missing runtime source identity")
    for name, sha in sources.items():
        _require(isinstance(name, str) and bool(re.fullmatch(r"[A-Za-z0-9_.-]+\.(py|json)", name)), "unsafe source name")
        _hash(sha, "runtime source hash")
    return {key: provenance[key] for key in ("cohort_id", "model_revision", "model_manifest_sha256", "runtime_source_sha256")}


def validate_records(records: list[dict], plan: dict, phase: str, *, require_complete: bool = False, require_passed: bool = False) -> dict:
    """Bind every row to a measured planned trial; never silently filter data."""
    validate_plan(plan)
    _require(phase in ("training", "calibration", "test"), "invalid record split")
    _require(isinstance(records, list), "records must be a list")
    expected = trial_index(plan, phase)
    rows, attempts, identities = {}, set(), []
    cohort = None
    for record in records:
        _require(record.get("schema_version") == 1, "record schema mismatch")
        _require(record.get("plan_sha256") == plan["plan_sha256"], "foreign plan hash")
        trial_id = record.get("trial_id")
        _require(trial_id in expected, "foreign/split-leaking/warmup trial")
        _require(trial_id not in rows, "duplicate trial; no automatic retry selection")
        trial = expected[trial_id]
        for field in IDENTITY_FIELDS:
            _require(record.get(field) == trial[field], f"trial identity mismatch: {field}")
        _require(record.get("public_features") == trial["public_features"], "public feature mismatch/leakage")
        attempt = record.get("attempt_id")
        _require(isinstance(attempt, str) and bool(attempt) and attempt not in attempts, "missing/duplicate attempt ID")
        attempts.add(attempt)
        provenance = record.get("provenance", {})
        identity = _cohort(provenance, plan)
        if cohort is None:
            cohort = identity
        _require(identity == cohort, "mixed runtime/model cohorts")
        _hash(provenance.get("measurement_sha256"), "measurement hash")
        status = record.get("status")
        _require(status in ("passed", "failed"), "record must report a terminal state")
        if status == "passed":
            _require(all(record.get(gate) is True for gate in GATES), "passed record has a false/missing evidence gate")
            _hash(provenance.get("diagnostic_sha256"), "diagnostic hash")
            _hash(provenance.get("reference_sha256"), "reference hash")
            metrics = record.get("metrics", {})
            values = [_finite(metrics.get(name), name, positive=True) for name in METRICS]
            _require(values[0] <= values[1] <= values[2], "nonmonotonic latency metrics")
            _require(record.get("executed_action") in ("recompute", "native_fetch") if record["method"] == POLICY else record.get("executed_action") == record["method"], "executed action does not match the planned method")
        _require(not require_passed or status == "passed", "fit/calibration cannot drop failed trials")
        rows[trial_id] = record
        identities.append({"trial_id": trial_id, "attempt_id": attempt, "record_sha256": digest(record)})
    missing = sorted(set(expected) - set(rows))
    _require(not require_complete or not missing, "incomplete planned split coverage")
    return {"rows": rows, "cohort": cohort, "missing_trial_ids": missing, "failed_trial_ids": sorted(k for k, r in rows.items() if r["status"] == "failed"), "record_identities": sorted(identities, key=lambda r: r["trial_id"]), "expected_count": len(expected)}


def _bound_artifact(payload: dict) -> dict:
    result = dict(payload)
    result["artifact_sha256"] = digest(payload)
    return result


def _verify_artifact(artifact: dict, kind: str, plan: dict | None = None) -> None:
    payload = {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    _require(artifact.get("kind") == kind and artifact.get("artifact_sha256") == digest(payload), "artifact kind/hash mismatch")
    if plan is not None:
        _require(artifact.get("plan_sha256") == plan["plan_sha256"], "artifact belongs to another plan")


def fit_nominal(training_records: list[dict], plan: dict) -> dict:
    """Fit only the full prescribed training split; test/calibration rows reject."""
    evidence = validate_records(training_records, plan, "training", require_complete=True, require_passed=True)
    rows = sorted(evidence["rows"].values(), key=lambda r: r["trial_id"])
    recompute = [r for r in rows if r["method"] == "recompute"]
    fetch = [r for r in rows if r["method"] == "native_fetch"]
    xp = np.asarray([r["public_features"]["prefix_tokens"] / 1024 for r in recompute])
    transfer = np.asarray([8 * r["public_features"]["prefix_KV_bytes"] / r["public_features"]["application_cap_bps"] for r in fetch])
    yr = np.asarray([r["metrics"]["ttft_s"] for r in recompute])
    yf = np.asarray([r["metrics"]["ttft_s"] for r in fetch])
    models = {
        "recompute": nnls_small(np.column_stack((np.ones(len(xp)), xp, xp ** 2)), yr),
        "native_fetch": nnls_small(np.column_stack((np.ones(len(transfer)), transfer)), yf),
    }
    diagnostic_variants = {
        "recompute_linear_with_intercept": nnls_small(np.column_stack((np.ones(len(xp)), xp)), yr),
        "recompute_linear_no_intercept": nnls_small(xp[:, None], yr),
        "fetch_transfer_no_intercept": nnls_small(transfer[:, None], yf),
    }
    return _bound_artifact({
        "schema_version": 1, "kind": "E1_nominal_training_fit", "plan_sha256": plan["plan_sha256"],
        "design_sha256": plan["design_sha256"], "cohort": evidence["cohort"],
        "fit_split": "training", "feature_names": list(PUBLIC_FEATURES), "target_metric": "ttft_s",
        "models": models, "diagnostic_variants": diagnostic_variants,
        "solver": {"algorithm": "enumerate_all_nonnegative_active_subsets_lstsq", "numpy_version": np.__version__, "fit_source_sha256": _analysis_source_hash()},
        "training_record_identities": evidence["record_identities"],
        "training_records_sha256": digest(rows),
        "training_input_ids": sorted({r["input_id"] for r in rows}),
        "training_seeds": sorted({r["seed"] for r in rows}),
        "future_length_feature": False, "test_labels_used": False,
        "E1_complete": False, "execution_authorized": False,
    })


def _analysis_source_hash() -> str:
    import hashlib
    from pathlib import Path
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def predict_nominal(artifact: dict, public_features: dict) -> dict:
    """Strict feature allowlist; passing an executor's G or outcomes raises."""
    _verify_artifact(artifact, "E1_nominal_training_fit")
    _require(set(public_features) == set(PUBLIC_FEATURES), "unknown/missing policy feature; future information prohibited")
    p = _finite(public_features["prefix_tokens"], "prefix_tokens", True)
    kv = _finite(public_features["prefix_KV_bytes"], "prefix_KV_bytes", True)
    rate = _finite(public_features["application_cap_bps"], "application_cap_bps", True)
    _require(public_features["fixed_suffix_tokens"] == 16, "unmodeled suffix geometry")
    _require(kv == p * 196608, "model-specific BF16 KV geometry mismatch")
    r = artifact["models"]["recompute"]["coefficients"]
    f = artifact["models"]["native_fetch"]["coefficients"]
    _require(len(r) == 3 and len(f) == 2, "coefficient dimensions mismatch")
    for value in r + f:
        _finite(value, "coefficient")
    recompute = r[0] + r[1] * (p / 1024) + r[2] * (p / 1024) ** 2
    fetch = f[0] + f[1] * (8 * kv / rate)
    crossover = 8 * f[1] * kv / (recompute - f[0]) if recompute > f[0] and f[1] > 0 else None
    return {"recompute_ttft_s": recompute, "native_fetch_ttft_s": fetch, "nominal_crossover_bps": crossover, "chosen_action": "recompute" if recompute <= fetch else "native_fetch", "objective": "nominal_TTFT", "measured_execution": False}


def choose_action(artifact: dict, public_features: dict) -> str:
    return predict_nominal(artifact, public_features)["chosen_action"]


def descriptive(values: list[float]) -> dict:
    """No request-tail percentile: n, median, range and raw values only."""
    clean = [_finite(v, "sample", positive=True) for v in values]
    return {"n": len(clean), "values": clean, "median": float(np.median(clean)) if clean else None, "min": min(clean) if clean else None, "max": max(clean) if clean else None, "request_tail_claim": False}


def _residual_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "median_absolute_percentage_error": None, "p90_absolute_percentage_error": None, "descriptive_adequacy_pass": None}
    ape = np.asarray([r["absolute_percentage_error"] for r in rows])
    median, p90 = float(np.median(ape)), float(np.quantile(ape, .9))
    return {"n": len(rows), "median_absolute_percentage_error": median, "p90_absolute_percentage_error": p90, "descriptive_adequacy_pass": median <= .2 and p90 <= .5, "scope": "prediction-error percentile, not request-latency tail or coverage guarantee"}


def _residuals(records: list[dict], artifact: dict, plan: dict) -> dict:
    """Called only after record validation by calibration/test entry points."""
    cell_map = {c["cell_id"]: c for c in plan["cells"]}
    details = []
    for r in sorted(records, key=lambda row: row["trial_id"]):
        if r["status"] != "passed" or r["method"] not in ("recompute", "native_fetch"):
            continue
        predicted = predict_nominal(artifact, r["public_features"])[r["method"] + "_ttft_s"]
        observed = r["metrics"]["ttft_s"]
        cell = cell_map[r["cell_id"]]
        x = r["public_features"]["prefix_tokens"] / 1024
        seconds = 8 * r["public_features"]["prefix_KV_bytes"] / r["public_features"]["application_cap_bps"]
        variants = artifact["diagnostic_variants"]
        if r["method"] == "recompute":
            linear = variants["recompute_linear_with_intercept"]["coefficients"]
            alternative = {"recompute_linear_with_intercept": linear[0] + linear[1] * x, "recompute_linear_no_intercept": variants["recompute_linear_no_intercept"]["coefficients"][0] * x}
        else:
            alternative = {"fetch_transfer_no_intercept": variants["fetch_transfer_no_intercept"]["coefficients"][0] * seconds}
        details.append({"trial_id": r["trial_id"], "cell_id": r["cell_id"], "method": r["method"], "prefix_tokens": cell["prefix_tokens"], "rate_mbps": cell["application_cap_mbps_decimal"], "realized_G_for_retrospective_grouping_only": cell["realized_stop_tokens"], "added_delay_ms": cell["added_experiment_path_round_trip_delay_ms"], "load_location": cell["background_load_location"], "OOD_treatment": cell["background_load_location"] != "none" or cell["added_experiment_path_round_trip_delay_ms"] != 0, "predicted_ttft_s": predicted, "observed_ttft_s": observed, "signed_error_s": predicted - observed, "absolute_error_s": abs(predicted - observed), "relative_error": (predicted - observed) / observed, "absolute_percentage_error": abs(predicted - observed) / observed, "training_only_diagnostic_variants": {name: {"predicted_ttft_s": value, "absolute_percentage_error": abs(value - observed) / observed} for name, value in alternative.items()}})
    groups = {}
    for item in details:
        key = f"{item['method']}|P{item['prefix_tokens']}|r{item['rate_mbps']}|G{item['realized_G_for_retrospective_grouping_only']}|d{item['added_delay_ms']}|load={item['load_location']}"
        groups.setdefault(key, []).append(item)
    variants = {name: _residual_summary([r["training_only_diagnostic_variants"][name] for r in details if name in r["training_only_diagnostic_variants"]]) for name in artifact["diagnostic_variants"]}
    return {"rows": details, "overall": _residual_summary(details), "by_condition": {key: _residual_summary(value) for key, value in sorted(groups.items())}, "training_only_variant_comparison": variants, "OOD": _residual_summary([r for r in details if r["OOD_treatment"]]), "in_domain": _residual_summary([r for r in details if not r["OOD_treatment"]]), "realized_G_used_for_prediction": False}


def calibrate_nominal(calibration_records: list[dict], artifact: dict, plan: dict) -> dict:
    """No coefficient update. Bind calibration residuals to the frozen train fit."""
    _verify_artifact(artifact, "E1_nominal_training_fit", plan)
    evidence = validate_records(calibration_records, plan, "calibration", require_complete=True, require_passed=True)
    _require(evidence["cohort"] == artifact["cohort"], "calibration source cohort differs from training")
    return _bound_artifact({"schema_version": 1, "kind": "E1_nominal_calibration", "plan_sha256": plan["plan_sha256"], "training_artifact_sha256": artifact["artifact_sha256"], "cohort": evidence["cohort"], "record_identities": evidence["record_identities"], "residuals": _residuals(list(evidence["rows"].values()), artifact, plan), "coefficients_updated": False, "coverage_guarantee": False, "E1_complete": False})


def _pair_summary(a: list[float | None], b: list[float | None], draws: np.ndarray) -> dict:
    available = [i for i, (av, bv) in enumerate(zip(a, b)) if av is not None and bv is not None]
    if len(available) != len(a):
        return {"status": "incomplete_pairs", "paired_n": len(available), "required_n": len(a), "missing_block_indices": sorted(set(range(len(a))) - set(available)), "label": "unavailable", "approximate_95pct_interval": None}
    av, bv = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    reduction = (bv - av) / bv
    med = float(np.median(reduction))
    interval = np.quantile(np.median(reduction[draws], axis=1), [.025, .975]).tolist()
    reverse = (av - bv) / av
    reverse_interval = np.quantile(np.median(reverse[draws], axis=1), [.025, .975]).tolist()
    label = "A_advantage" if med > .05 and interval[0] > .05 else ("B_advantage" if float(np.median(reverse)) > .05 and reverse_interval[0] > .05 else "tied_or_unresolved")
    return {"status": "descriptive", "paired_n": len(a), "relative_reduction_A_vs_B": reduction.tolist(), "median_relative_reduction": med, "approximate_95pct_interval": interval, "reverse_median_relative_reduction": float(np.median(reverse)), "reverse_approximate_95pct_interval": reverse_interval, "label": label, "coverage_guarantee": False, "familywise_significance_claim": False}


def analyze_test(test_records: list[dict], artifact: dict, plan: dict, calibration: dict) -> dict:
    """Keep all missing/failed trials visible; full eight-block CIs only.

    Every pair/cell/metric uses the SAME resampled input-block index matrix.
    Partial data produce descriptive values and unavailable paired intervals;
    they never become a complete-data result by pooling cells or tokens.
    """
    _verify_artifact(artifact, "E1_nominal_training_fit", plan)
    _verify_artifact(calibration, "E1_nominal_calibration", plan)
    _require(calibration["training_artifact_sha256"] == artifact["artifact_sha256"], "calibration refers to another fit")
    evidence = validate_records(test_records, plan, "test")
    _require(evidence["cohort"] is None or evidence["cohort"] == artifact["cohort"], "test source cohort differs from training")
    for record in evidence["rows"].values():
        if record["method"] == POLICY and record["status"] == "passed":
            _require(record.get("policy_artifact_sha256") == artifact["artifact_sha256"], "policy trial used a different/missing fitted artifact")
            _require(record["executed_action"] == choose_action(artifact, record["public_features"]), "executed nominal choice differs from its frozen rule")
    members = [r for r in plan["input_registry"] if r["phase"] == "test" and r["sample_role"] == "measured"]
    methods = list(FIXED_ACTIONS) + [POLICY, CONTROL]
    seed = 2026110600
    draws = np.random.Generator(np.random.PCG64(seed)).integers(0, len(members), size=(10000, len(members)))
    lookup = {(r["cell_id"], r["input_id"], r["method"]): r for r in evidence["rows"].values()}
    cells = []
    for cell in plan["cells"]:
        item = {"cell_id": cell["cell_id"], "conditions": cell, "metrics": {}}
        for metric in METRICS:
            vectors = {}
            for method in methods:
                rows = [lookup.get((cell["cell_id"], member["input_id"], method)) for member in members]
                vectors[method] = [r["metrics"][metric] if r is not None and r["status"] == "passed" else None for r in rows]
            hindsight = []
            for block, member in enumerate(members):
                costs = {method: vectors[method][block] for method in FIXED_ACTIONS}
                if any(v is None for v in costs.values()):
                    hindsight.append({"input_id": member["input_id"], "status": "incomplete_fixed_action_pair", "value": None, "best_actions": []})
                else:
                    best = min(costs.values())
                    hindsight.append({"input_id": member["input_id"], "status": "offline_noisy_hindsight", "value": best, "best_actions": [method for method in FIXED_ACTIONS if costs[method] == best]})
            pairs = {f"{a}__vs__{b}": _pair_summary(vectors[a], vectors[b], draws) for a, b in itertools.combinations(methods, 2)}
            complete_cell = all(all(v is not None for v in vectors[method]) for method in FIXED_ACTIONS)
            medians = {method: float(np.median(vectors[method])) for method in FIXED_ACTIONS} if complete_cell else {}
            best_median = min(medians.values()) if medians else None
            item["metrics"][metric] = {"methods": {method: descriptive([v for v in vector if v is not None]) for method, vector in vectors.items()}, "paired_comparisons": pairs, "hindsight_by_block": hindsight, "best_cell_median_fixed_actions": [method for method, value in medians.items() if value == best_median], "hindsight_extra_GPU_trials": 0}
        cells.append(item)
    return _bound_artifact({
        "schema_version": 1, "kind": "E1_heldout_descriptive_analysis", "plan_sha256": plan["plan_sha256"],
        "training_artifact_sha256": artifact["artifact_sha256"], "calibration_artifact_sha256": calibration["artifact_sha256"],
        "observed_records": len(evidence["rows"]), "expected_records": evidence["expected_count"],
        "missing_trial_ids": evidence["missing_trial_ids"], "failed_trial_ids": evidence["failed_trial_ids"],
        "complete_planned_record_coverage": not evidence["missing_trial_ids"],
        "all_planned_records_passed": not evidence["missing_trial_ids"] and not evidence["failed_trial_ids"],
        "record_identities": evidence["record_identities"], "cells": cells,
        "nominal_residuals": _residuals(list(evidence["rows"].values()), artifact, plan),
        "bootstrap": {"unit": "heldout_input_seed_block", "n_blocks": len(members), "draws": 10000, "seed": seed, "generator": "numpy.PCG64", "index_matrix_sha256": digest(draws.tolist()), "shared_across_all_cells_methods_metrics": True, "scope": "small-sample descriptive intervals, no coverage theorem or multiplicity guarantee"},
        "request_p95_p99_reported": False, "E1_complete": False, "paper_stage_complete": False,
        "scope": "Record-provenance-checked CPU analysis only; does not verify raw artifacts, authorize GPU work, or replace scientific/timing/full-stage review.",
    })
