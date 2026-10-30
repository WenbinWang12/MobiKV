"""Deterministic E4 protocol materializer; never launches or accepts experiments."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import random

if __package__:
    from .scheduler import POLICIES, policy_config
else:
    from e4_scheduler import POLICIES, policy_config

DEFAULT_DESIGN = (Path(__file__).resolve().parents[1] / "e4_full_design.json" if __package__
                  else Path(__file__).resolve().parent / "e4_design_protocol.json")
MODES = ("full", "smoke", "profiles")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _validate_design(d):
    require(d.get("stage") == "E4" and d.get("E4_complete") is False and d.get("paper_stage_complete") is False, "Foreign/completed design")
    require(len(d["cells"]) == len({c["id"] for c in d["cells"]}) == 12, "Expected12 unique cells")
    require(tuple(m["id"] for m in d["methods"]) == POLICIES[:3] + ("strict_rpc_first", "age_bounded_rpc_first", "no_migration_reference"), "Scheduler methods changed")
    require(d["cohort"]["physical_hosts"] == d["cohort"]["selected_GPUs"] == 2 and d["cohort"]["tensor_parallel_size"] == 1, "Two-host TP1 cohort required")
    blocks = d["inputs_repetition_and_order"]["heldout_blocks"]
    require(len(blocks) == len({b["block_id"] for b in blocks}) == 8, "Eight paired blocks required")
    families = [b["family"] for b in blocks] + [d["inputs_repetition_and_order"][k]["family"] for k in ("excluded_warmup", "excluded_smoke", "excluded_calibration", "excluded_profile")]
    require(len(set(families)) == len(families), "Input split leakage")
    for cell in d["cells"]:
        require(cell["rate_bps"] in (100_000_000, 1_000_000_000) and cell["migration_P"] in (1024, 4096), "Undeclared geometry")
        require(cell["repair_requests"] in (0, 1, 3) and cell["transfer_only_jobs"] in (0, 1), "Undeclared offered work")
        require((cell["module"] == "transfer_isolation") == (cell["repair_requests"] == 0 and cell["transfer_only_jobs"] == 1), "Module/job mismatch")
        require(cell["load_location"] in ("none", "holder", "destination"), "Unknown load location")


def _seed(*parts):
    return int(digest(list(parts))[:16], 16)


def _request(block, role, prefix, action):
    identity = {"input_id": block["family"] + ":" + role, "input_family_id": block["family"],
                "seed": _seed(block["seed"], role), "prefix_tokens": prefix, "suffix_tokens": 16}
    return {"role": role, **identity, "prefix_key": "E4-" + digest(identity)[:32],
            "action": action, "public_output_cap": 128, "executor_only": {"realized_stop_tokens": 64}}


def _group(d, block, cell, method, phase, role):
    trial_id = ".".join((phase, cell["id"], block["block_id"], method, "group"))
    scheduler = "fine_fifo" if method == "no_migration_reference" else method
    requests = [_request(block, "A", 1024, "remote")]
    for i in range(cell["repair_requests"]):
        requests.append(_request(block, "B" + str(i + 1), cell["migration_P"],
                                 "remote" if method == "no_migration_reference" else "repair"))
    transfers = []
    if cell["transfer_only_jobs"] and method != "no_migration_reference":
        transfer = _request(block, "transfer-cache", cell["migration_P"], "transfer_only")
        transfer.pop("executor_only"); transfer.pop("public_output_cap")
        transfer["suffix_tokens"] = 0
        transfer["expected_KV_payload_bytes"] = cell["migration_P"] * 196608
        transfer["expected_complete_layers"] = 48
        transfers.append(transfer)
    return {"trial_id": trial_id, "phase": phase, "sample_role": role, "kind": "group",
        "block_id": block["block_id"], "input_family_id": block["family"], "block_seed": block["seed"],
        "cell_id": cell["id"], "module": cell["module"], "method": method,
        "scheduler": policy_config(scheduler, cell["rate_bps"]),
        "requests": requests, "transfer_jobs": transfers, "load_location": cell["load_location"],
        "load_seed": _seed(block["seed"], cell["id"], cell["load_location"], "background"),
        "added_RTT_ms": 0, "common_offered_barrier": True, "target_cache_initially_empty": True,
        "solo_trial_ids": [], "diagnostic_id": trial_id + ".diagnostic",
        "reference_ids": [trial_id + ".HF." + r["role"] for r in requests],
        "instrumented_excluded": phase == "profile", "execution_authorized": False,
        "paper_stage_complete": False, "E4_complete": False}


def _with_solos(group, solo_first, order_seed):
    group = copy.deepcopy(group)
    solos = []
    for request in group["requests"]:
        solo = copy.deepcopy(group)
        solo.update(trial_id=group["trial_id"].removesuffix("group") + "solo-" + request["role"],
                    kind="decoded_request_solo", requests=[copy.deepcopy(request)], transfer_jobs=[],
                    load_location="none", common_offered_barrier=False,
                    paired_group_trial_id=group["trial_id"], normalized_role=request["role"])
        solo["reference_ids"] = [solo["trial_id"] + ".HF." + request["role"]]
        solo["diagnostic_id"] = solo["trial_id"] + ".diagnostic"
        solos.append(solo)
    for transfer in group["transfer_jobs"]:
        solo = copy.deepcopy(group)
        solo.update(trial_id=group["trial_id"].removesuffix("group") + "solo-transfer",
                    kind="transfer_solo", requests=[], transfer_jobs=[copy.deepcopy(transfer)],
                    load_location="none", common_offered_barrier=False,
                    paired_group_trial_id=group["trial_id"], normalized_role="transfer-cache")
        solo["reference_ids"] = []
        solo["diagnostic_id"] = solo["trial_id"] + ".diagnostic"
        solos.append(solo)
    random.Random(order_seed).shuffle(solos)
    group["solo_trial_ids"] = [s["trial_id"] for s in solos]
    group["pair_order"] = "solo_before_group" if solo_first else "group_before_solo"
    for solo in solos:
        solo["pair_order"] = group["pair_order"]
    return solos + [group] if solo_first else [group] + solos


def _counts(trials):
    kinds = ("group", "decoded_request_solo", "transfer_solo")
    roles = sorted({t["sample_role"] for t in trials})
    return {"attempts": len(trials), "by_kind": {k: sum(t["kind"] == k for t in trials) for k in kinds},
        "by_role_and_kind": {r: {k: sum(t["sample_role"] == r and t["kind"] == k for t in trials) for k in kinds} for r in roles},
        "decoded_request_executions": sum(len(t["requests"]) for t in trials),
        "transfer_job_executions": sum(len(t["transfer_jobs"]) for t in trials),
        "same_shape_HF_references": sum(len(t["reference_ids"]) for t in trials)}


def _build(d, mode):
    _validate_design(d); require(mode in MODES, "Unknown E4 protocol mode")
    inputs = d["inputs_repetition_and_order"]
    methods = [m["id"] for m in d["methods"]]
    cells = list(d["cells"])
    if mode == "full":
        warm = inputs["excluded_warmup"]
        blocks = [(dict(warm, block_id="warmup"), "warmup", "warmup")] + [(b, "test", "measured") for b in inputs["heldout_blocks"]]
    else:
        key = "excluded_smoke" if mode == "smoke" else "excluded_profile"
        phase, role = ("smoke", "functional_smoke") if mode == "smoke" else ("profile", "instrumented_excluded")
        blocks = [(dict(inputs[key], block_id=phase), phase, role)]
        section = d["smoke_plan"] if mode == "smoke" else d["separate_profiles"]
        allowed = section["group_cells"] if mode == "smoke" else section["cells"]
        cells = [c for c in cells if c["id"] in allowed]
        if mode == "profiles":
            methods = list(section["methods"])
    trials = []
    base_order = list(methods)
    random.Random(inputs["order_seed"]).shuffle(base_order)
    for block_index, (block, phase, role) in enumerate(blocks):
        ordered_cells = list(cells)
        random.Random(_seed(inputs["order_seed"], block["block_id"], mode)).shuffle(ordered_cells)
        for cell_index, cell in enumerate(ordered_cells):
            # Cyclic method positions are balanced to one occurrence over six
            # blocks; extra two held-out blocks do not create new replicates.
            offset = (block_index + next(i for i, c in enumerate(cells) if c["id"] == cell["id"])) % len(methods)
            order = base_order[offset:] + base_order[:offset]
            for method_index, method in enumerate(order):
                group = _group(d, block, cell, method, phase, role)
                if mode == "profiles":
                    trials.append(group)
                else:
                    first = (block_index + cell_index + methods.index(method)) % 2 == 0
                    trials.extend(_with_solos(group, first, _seed(inputs["order_seed"], group["trial_id"], "solos")))
    require(len({t["trial_id"] for t in trials}) == len(trials), "Duplicate trial identity")
    counts = _counts(trials)
    expected = {"full": 2340, "smoke": 107, "profiles": 16}[mode]
    require(counts["attempts"] == expected, "Design/materialized attempt count mismatch")
    if mode == "full":
        n = d["resources_and_counts"]
        require(counts["attempts"] == n["total_full_main_attempts"], "Design total differs")
        for sample, prefix in (("measured", "measured"), ("warmup", "warmup")):
            for kind, label in (("group", "group"), ("decoded_request_solo", "decoded_request_solo"), ("transfer_solo", "transfer_solo")):
                require(counts["by_role_and_kind"][sample][kind] == n[prefix + "_" + label + "_attempts"], "Design per-kind count differs")
        require(counts["decoded_request_executions"] == n["total_full_decoded_request_executions"] == counts["same_shape_HF_references"], "Decoded/reference count differs")
    payload = {"schema_version": 1, "stage": "E4", "mode": "e4_" + mode,
        "design": copy.deepcopy(d), "design_sha256": digest(d), "cohort": copy.deepcopy(d["cohort"]),
        "trials": trials, "counts": counts, "calibration": copy.deepcopy(inputs["service_calibration"]),
        "controller_features": ["actual ready stream class", "actual pending ticket", "local eligible queue age", "actual requested/sent bytes", "configured fixed application cap"],
        "future_information_used": False, "execution_authorized": False,
        "executable_plan": False, "E4_complete": False, "paper_stage_complete": False,
        "scope": "Frozen CPU specification; executor, raw auditor and accepted prerequisite gates still required"}
    return dict(payload, plan_sha256=digest(payload))


def build_plan(design=DEFAULT_DESIGN, mode="full"):
    value = copy.deepcopy(design) if isinstance(design, dict) else json.loads(Path(design).read_text(encoding="utf-8"))
    return _build(value, mode)


def validate_plan(protocol):
    require(isinstance(protocol, dict) and protocol.get("stage") == "E4", "Not an E4 protocol")
    mode = protocol.get("mode", "").removeprefix("e4_")
    require(protocol == _build(protocol["design"], mode), "Protocol hash/order/identity/counts changed")
    return True


def trial_index(protocol, *, kind=None, sample_role=None):
    validate_plan(protocol)
    return {t["trial_id"]: t for t in protocol["trials"]
            if (kind is None or t["kind"] == kind) and (sample_role is None or t["sample_role"] == sample_role)}
