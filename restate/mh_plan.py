"""E6/E7 plan materialization: calibration, multi-holder (E6) and integrated serving (E7).

All policies receive the same information: a cost model fitted only on the
excluded calibration run (standalone medians, same-resource slowdown factors and
residual radii). Decisions are made before execution in arrival order and use
only already-arrived requests plus model predictions (no measured outcomes, no
future arrivals). Plans are deterministic given (design, model).
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import random
import statistics

from e4_scheduler import policy_config

KV_BYTES_PER_TOKEN = 196608          # Qwen2.5-14B BF16: 48 layers * 8 KV heads * 128 * 2 (K,V) * 2 bytes
HOLDER_ACTIONS = ("holder", "remote", "repair")
ALL_ACTIONS = HOLDER_ACTIONS + ("recompute",)
GBPS, SLOW = 1_000_000_000, 250_000_000
MAX_PER_HOLDER = 4
BACKGROUND_FAMILY = "MH-background-calibration-family"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def seed(*parts):
    return int(digest(list(parts))[:16], 16)


def request(family, role, prefix, action, holder, *, offset=0.0, **extra):
    identity = {"input_id": family + ":" + role, "input_family_id": family, "seed": seed(family, role),
                "prefix_tokens": prefix, "suffix_tokens": 16}
    return {"role": role, **identity, "prefix_key": "MH-" + digest(identity)[:32], "action": action,
            "holder": holder if action in HOLDER_ACTIONS else None, "arrival_offset_s": round(float(offset), 4),
            "public_output_cap": 128, "executor_only": {"realized_stop_tokens": 64}, **extra}


def trial(stage, phase, role, block, cell, method, requests, holders, *, destination_load=False, timeout=1200):
    trial_id = ".".join((stage, phase, cell, block, method))
    used = sorted({r["holder"] for r in requests if r["action"] in HOLDER_ACTIONS})
    hold = {}
    for h in used:
        rate, policy, load = holders[h][:3]
        extra = dict(zip(("chunk_bytes", "added_RTT_ms"), holders[h][3:]))
        hold[str(h)] = {"subtrial_id": trial_id + "@h" + str(h), "scheduler": policy_config(policy, rate, extra.get("chunk_bytes")),
                        "load_location": "holder" if load else "none", "load_seed": seed(trial_id, h, "load")}
        if "added_RTT_ms" in extra:
            hold[str(h)]["added_RTT_ms"] = extra["added_RTT_ms"]
    return {"trial_id": trial_id, "stage": stage, "phase": phase, "sample_role": role, "kind": "group",
            "block_id": block, "cell_id": cell, "method": method, "requests": requests, "holders": hold,
            "holder_paths": {str(h): dict(rate_bps=v[0], policy=v[1], loaded=bool(v[2]),
                                           **dict(zip(("chunk_bytes", "added_RTT_ms"), v[3:]))) for h, v in holders.items()},
            "destination_load": destination_load, "load_seed": seed(trial_id, "destination-load"), "timeout_s": timeout}


def holder_registries(trials, n_holders):
    from mh_destination import holder_subtrial
    out = {str(h): {} for h in range(n_holders)}
    for t in trials:
        for h in t["holders"]:
            sub = holder_subtrial(t, int(h))
            out[h][sub["trial_id"]] = sub
    return out


def finalize(stage, mode, run_id, trials, n_holders, extra=None):
    assert len({t["trial_id"] for t in trials}) == len(trials)
    payload = {"schema_version": 1, "stage": stage, "mode": mode, "run_id": run_id, "trials": trials,
               "holder_registries": holder_registries(trials, n_holders), "background_family": BACKGROUND_FAMILY,
               "trial_timeout_s": 1200, "n_holders": n_holders, "counts": {"trials": len(trials),
               "requests": sum(len(t["requests"]) for t in trials)}, **(extra or {})}
    payload["plan_sha256"] = digest(payload)
    return payload


# ---------------------------------------------------------------- calibration

def build_calibration(run_id, *, reps=2):
    """Standalone cost of every action on both holders, plus same-resource concurrency (k=2,4)."""
    trials = []
    for rep in range(reps):
        fam = "MH-cal-r%d" % rep
        for h in (0, 1):
            for prefix in (1024, 4096):
                for rate in (GBPS, SLOW):
                    for load in (False, True):
                        for action in HOLDER_ACTIONS:
                            cell = "solo-%s-h%d-P%d-%s-%s" % (action, h, prefix, "1G" if rate == GBPS else "250M",
                                                              "load" if load else "idle")
                            req = [request(fam + ":" + cell, "A", prefix, action, h)]
                            trials.append(trial("CAL", "calibration", "calibration", "r%d" % rep, cell, "solo", req,
                                                {h: (rate, "fine_fifo", load)}))
        for prefix in (1024, 4096):
            cell = "solo-recompute-P%d" % prefix
            trials.append(trial("CAL", "calibration", "calibration", "r%d" % rep, cell, "solo",
                                [request(fam + ":" + cell, "A", prefix, "recompute", None)], {}))
        for k in (2, 4):
            for action in ALL_ACTIONS:
                cell = "group-%s-k%d-P1024-1G-idle" % (action, k)
                reqs = [request(fam + ":" + cell, "R%d" % i, 1024, action, 0) for i in range(k)]
                trials.append(trial("CAL", "calibration", "calibration", "r%d" % rep, cell, "same_resource", reqs,
                                    {0: (GBPS, "fine_fifo", False)}))
    return finalize("CAL", "calibration", run_id, trials, 2)


def build_smoke_calibration(run_id):
    """A few trials covering every action/holder for functional checks before the real calibration."""
    fam = "MH-calsmoke"
    trials = []
    for h in (0, 1):
        for action in HOLDER_ACTIONS:
            cell = "smoke-%s-h%d" % (action, h)
            trials.append(trial("CAL", "smoke", "calibration_smoke", "s0", cell, "solo",
                                [request(fam + ":" + cell, "A", 1024, action, h)], {h: (GBPS, "fine_fifo", h == 0)}))
    trials.append(trial("CAL", "smoke", "calibration_smoke", "s0", "smoke-recompute", "solo",
                        [request(fam + ":rc", "A", 1024, "recompute", None)], {}))
    mixed = [request(fam + ":mix", "A", 1024, "holder", 0), request(fam + ":mix", "B", 1024, "repair", 1, offset=0.5),
             request(fam + ":mix", "C", 4096, "remote", 1, offset=1.0), request(fam + ":mix", "D", 1024, "recompute", None, offset=1.5)]
    trials.append(trial("CAL", "smoke", "calibration_smoke", "s0", "smoke-mixed", "mixed", mixed,
                        {0: (GBPS, "age_bounded_rpc_first", False), 1: (SLOW, "age_bounded_rpc_first", False)},
                        destination_load=True))
    return finalize("CAL", "smoke", run_id, trials, 2)


def fit_model(records):
    """Cost model from excluded calibration records only."""
    solo, groups, resid = {}, {}, {}
    for r in records:
        if r["status"] != "passed" or r["stage"] != "CAL" or r["sample_role"] != "calibration":
            continue
        if r["method"] == "solo":
            q = r["requests"][0]
            parts = r["cell_id"].split("-")
            if q["action"] == "recompute":
                key = "recompute|%d" % q["prefix_tokens"]
            else:
                key = "%s|%d|%s|%s" % (q["action"], q["prefix_tokens"], parts[4], parts[5])
            solo.setdefault(key, []).append(q)
        else:
            groups.setdefault(r["cell_id"], []).append(r)
    model = {"standalone": {}, "slowdown": {}, "radius": {}}
    for key, rows in solo.items():
        comp = [q["completion_s"] for q in rows]
        med = statistics.median(comp)
        model["standalone"][key] = dict(completion_s=med, ttft_s=statistics.median(q["ttft_s"] for q in rows),
                                        mean_itl_s=statistics.median(q["mean_itl_s"] for q in rows), n=len(rows))
        resid.setdefault(key.split("|")[0], []).extend(abs(c - med) / med for c in comp)
    for action, values in resid.items():
        values = sorted(values)
        model["radius"][action] = values[max(0, math.ceil(.9 * len(values)) - 1)] if values else 0.
    for cell, rows in groups.items():
        action, k = cell.split("-")[1], int(cell.split("-")[2][1:])
        base_key = "recompute|1024" if action == "recompute" else "%s|1024|1G|idle" % action
        base = model["standalone"][base_key]["completion_s"]
        ratios = [q["completion_s"] / base for r in rows for q in r["requests"]]
        model["slowdown"].setdefault(action, {"1": 1.0})[str(k)] = statistics.median(ratios)
    ttft_best = {p: min(v["ttft_s"] for k, v in model["standalone"].items() if k.split("|")[1] == str(p)
                        and (k.startswith("recompute") or k.endswith("1G|idle"))) for p in (1024, 4096)}
    itl_local = model["standalone"]["recompute|1024"]["mean_itl_s"]
    model["slo"] = {"ttft_s": {str(p): 2 * v for p, v in ttft_best.items()}, "mean_itl_s": 2 * itl_local,
                    "rule": "2x the best standalone TTFT for the prefix length; 2x the standalone local mean ITL"}
    prefill = model["standalone"]["recompute|4096"]["completion_s"] - 64 * itl_local
    model["b_star_bps"] = 8 * KV_BYTES_PER_TOKEN * 4096 / max(prefill, 1e-3)
    model["b_star_rule"] = "INFOCOM threshold: fetch/repair iff path rate >= KV bytes per token / prefill time per token"
    model["model_sha256"] = digest(model)
    return model


# ---------------------------------------------------------------- cost model use

def standalone(model, action, prefix, path, holder=None):
    if action == "recompute":
        return model["standalone"]["recompute|%d" % prefix]["completion_s"]
    rate, loaded = path[0], path[2]
    label = rate_label(rate) if model.get("exact_rate_keys") else ("1G" if rate >= GBPS else "250M")
    key = "%s|%d|%s|%s" % (action, prefix, label, "load" if loaded else "idle")
    if model.get("per_holder_keys"):
        key += "|h%d" % holder
    return model["standalone"][key]["completion_s"]


def slow(model, action, k, rate=None):
    by_rate = model.get("slowdown_by_rate", {}).get(action)
    table = by_rate[rate_label(rate)] if by_rate and rate is not None else model["slowdown"].get(action, {"1": 1.0})
    if k <= 1:
        return 1.0
    if str(k) in table:
        return table[str(k)]
    ks = sorted(int(x) for x in table)
    lo = max(x for x in ks if x <= k)
    hi = min((x for x in ks if x >= k), default=lo)
    if hi == lo:
        return table[str(lo)] * k / lo
    w = (k - lo) / (hi - lo)
    return table[str(lo)] * (1 - w) + table[str(hi)] * w


def resource(action, holder):
    return "T" if action == "recompute" else "H%d" % holder


def options(req, paths, available, allow_holder=True):
    out = [("recompute", None)]
    for h in available:
        for a in HOLDER_ACTIONS:
            if allow_holder or a != "holder":
                out.append((a, h))
    return out


def predict(model, req_prefix, choice, paths, k, *, contention=True, radius=False):
    action, h = choice
    base = standalone(model, action, req_prefix, None if h is None else paths[h], h)
    value = base * (slow(model, action, k, None if h is None else paths[h][0]) if contention else 1.0)
    if radius:
        value *= 1 + model["radius"].get(action, 0.)
    return value


# ---------------------------------------------------------------- E6 policies (all requests arrive together)

def e6_assign(method, model, prefixes, paths):
    n = len(prefixes)
    holders = sorted(paths)
    if method == "highest_bandwidth_static":
        best = max(holders, key=lambda h: (paths[h][0], -h))
        return [min(((a, best) for a in HOLDER_ACTIONS), key=lambda c: predict(model, p, c, paths, 1)) for p in prefixes]
    if method == "greedy_independent":
        return [min(options(None, paths, holders), key=lambda c: predict(model, p, c, paths, 1)) for p in prefixes]
    if method == "round_robin_spread":
        out = []
        for i, p in enumerate(prefixes):
            h = holders[i % len(holders)]
            out.append(min(((a, h) for a in HOLDER_ACTIONS), key=lambda c: predict(model, p, c, paths, 1)))
        return out
    if method == "queue_aware_no_link_model":
        out, load = [], {}
        for p in prefixes:
            def cost(c):
                res = resource(*c)
                k = load.get(res, 0) + 1
                return predict(model, p, c, paths, 1 if c[0] == "repair" else k)
            cands = [c for c in options(None, paths, holders) if c[1] is None or
                     sum(1 for o in out if o[1] == c[1]) < MAX_PER_HOLDER]
            choice = min(cands, key=cost)
            out.append(choice)
            load[resource(*choice)] = load.get(resource(*choice), 0) + 1
        return out
    if method == "joint":
        best, best_cost = None, math.inf
        for combo in itertools.product(options(None, paths, holders), repeat=n):
            counts = {}
            for c in combo:
                counts[resource(*c)] = counts.get(resource(*c), 0) + 1
            if any(v > MAX_PER_HOLDER for r, v in counts.items() if r != "T"):
                continue
            cost = sum(predict(model, p, c, paths, counts[resource(*c)]) for p, c in zip(prefixes, combo))
            if cost < best_cost - 1e-12:
                best, best_cost = list(combo), cost
        return best
    raise ValueError(method)


E6_METHODS = ("highest_bandwidth_static", "greedy_independent", "round_robin_spread", "queue_aware_no_link_model", "joint")
E6_CONDITIONS = {
    "symmetric": {0: (GBPS, "fine_fifo", False), 1: (GBPS, "fine_fifo", False)},
    "bw_asym": {0: (GBPS, "fine_fifo", False), 1: (SLOW, "fine_fifo", False)},
    "load_asym": {0: (GBPS, "fine_fifo", True), 1: (GBPS, "fine_fifo", False)},
    "conflict": {0: (GBPS, "fine_fifo", True), 1: (SLOW, "fine_fifo", False)},
}


def e6_cells():
    cells = []
    for cond in E6_CONDITIONS:
        for n in (2, 4):
            cells.append(dict(id="%s-n%d-P1024" % (cond, n), condition=cond, n=n, prefix=1024))
    for cond in ("symmetric", "conflict"):
        cells.append(dict(id="%s-n2-P4096" % cond, condition=cond, n=2, prefix=4096))
    return cells


def build_e6(run_id, model, mode):
    blocks = [("smoke", "smoke", "functional_smoke")] if mode == "smoke" else \
        [("warmup", "warmup", "warmup")] + [("b%02d" % i, "test", "measured") for i in range(8)]
    trials = []
    rng = random.Random(seed("E6-order", mode))
    for block, phase, role in blocks:
        cells = e6_cells()
        rng.shuffle(cells)
        for cell in cells:
            paths = E6_CONDITIONS[cell["condition"]]
            prefixes = [cell["prefix"]] * cell["n"]
            methods = list(E6_METHODS)
            rng.shuffle(methods)
            for method in methods:
                assignment = e6_assign(method, model, prefixes, paths)
                fam = "MH-E6-%s-%s" % (block, cell["id"])
                reqs = [request(fam, "R%d" % i, p, a, h) for i, (p, (a, h)) in enumerate(zip(prefixes, assignment))]
                trials.append(trial("E6", phase, role, block, cell["id"], method, reqs, paths))
    return finalize("E6", "e6_" + mode, run_id, trials, 2, {"model_sha256": model["model_sha256"],
                    "methods": list(E6_METHODS), "cells": e6_cells(), "conditions": {k: {str(h): list(v) for h, v in c.items()}
                    for k, c in E6_CONDITIONS.items()}, "oracle": "post-hoc best measured method per (block, cell)"})


# ---------------------------------------------------------------- E7 integrated serving

E7_METHODS = ("static_recompute", "static_holder", "static_repair", "threshold_bstar", "greedy_predicted",
              "full_controller", "ablate_no_uncertainty", "ablate_no_contention", "ablate_no_progress")
E7_NETWORKS = {
    "fast": {0: GBPS, 1: GBPS, "load": (False, False)},
    "mixed": {0: GBPS, 1: SLOW, "load": (True, False)},
}
E7_LOADS = {"low": 40.0, "high": 10.0}      # arrival window (s) for 8 requests


def e7_workload(block, cell):
    rng = random.Random(seed("E7-workload", block, cell))
    window = E7_LOADS[cell.split("-")[1]]
    times = sorted(rng.uniform(0, window) for _ in range(8))
    times = [t - times[0] for t in times]
    reqs = []
    for i, t in enumerate(times):
        prefix = rng.choice((1024, 4096))
        location = rng.choices(("both", "h0", "h1", "none"), weights=(4, 2, 2, 2))[0]
        reqs.append(dict(role="Q%d" % i, prefix=prefix, location=location, offset=t))
    return reqs


def e7_assign(method, model, workload, paths, allow_holder=True):
    if not allow_holder and method in ("static_holder", "least_loaded"):
        raise ValueError(method + " needs the holder-decode action")
    out, active = [], []
    scheduler = "age_bounded_rpc_first" if method in ("full_controller", "ablate_no_uncertainty", "ablate_no_progress",
                                                       "least_loaded") else "fine_fifo"
    for q in workload:
        avail = {"both": [0, 1], "h0": [0], "h1": [1], "none": []}[q["location"]]
        avail = [h for h in avail if sum(1 for o in out if o["holder"] == h) < MAX_PER_HOLDER]
        p = q["prefix"]
        if method == "static_recompute" or not avail:
            choice = ("recompute", None)
        elif method == "static_holder":
            choice = ("holder", avail[0])
        elif method == "static_repair":
            choice = ("repair", avail[0])
        elif method == "threshold_bstar":
            h = max(avail, key=lambda x: (paths[x][0], -x))
            choice = ("repair", h) if paths[h][0] >= model["b_star_bps"] else ("recompute", None)
        elif method == "least_loaded":
            # Experiment 1: same feasible set, arrival order and occupancy bookkeeping as full_controller,
            # but no cost comparison at selection. Fewest active requests wins; ties H0 < H1 < T; a holder
            # resource always runs holder decode.
            def n_active(c):
                return sum(1 for a in active if a["res"] == resource(*c) and a["finish"] > q["offset"])
            choice = min([("holder", h) for h in avail] + [("recompute", None)], key=n_active)
        else:
            contention = method not in ("greedy_predicted", "ablate_no_contention")
            radius = method in ("full_controller", "ablate_no_progress", "ablate_no_contention")

            def k_of(c):
                if not contention:
                    return 1
                res = resource(*c)
                if method == "ablate_no_progress":
                    busy = [a for a in active if a["res"] == res]          # every earlier assignment, finished or not
                else:
                    busy = [a for a in active if a["res"] == res and a["finish"] > q["offset"]]
                return len(busy) + 1
            choice = min(options(None, paths, avail, allow_holder), key=lambda c: predict(model, p, c, paths, k_of(c),
                                                                            contention=contention, radius=radius))
        k = 1 + sum(1 for a in active if a["res"] == resource(*choice) and a["finish"] > q["offset"])
        finish = q["offset"] + predict(model, p, choice, paths, k)
        active.append(dict(res=resource(*choice), finish=finish))
        out.append(dict(action=choice[0], holder=choice[1]))
    return out, scheduler


X1_METHODS = ("full_controller", "least_loaded", "ablate_no_contention", "static_holder")


def build_x1(run_id, model, mode, prereg_sha256):
    """Experiment 1: the E7 cells/workloads with the least_loaded fairness baseline (preregistered)."""
    plan = build_e7(run_id, model, mode, methods=X1_METHODS, order_tag="X1-order")
    by_unit = {}
    for t in plan["trials"]:
        by_unit.setdefault((t["block_id"], t["cell_id"]), {})[t["method"]] = [(r["action"], r["holder"]) for r in t["requests"]]
    same = sorted("%s|%s" % u for u, m in by_unit.items() if m["least_loaded"] == m["full_controller"])
    plan.pop("plan_sha256")
    plan.update(experiment="X1", preregistration_sha256=prereg_sha256, identical_assignment_units=same,
                identical_assignment_count=len(same), units=len(by_unit))
    plan["plan_sha256"] = digest(plan)
    return plan


def build_e7(run_id, model, mode, methods=E7_METHODS, order_tag="E7-order"):
    blocks = [("smoke", "smoke", "functional_smoke")] if mode == "smoke" else \
        [("warmup", "warmup", "warmup")] + [("b%02d" % i, "test", "measured") for i in range(8)]
    cells = ["%s-%s" % (net, load) for net in E7_NETWORKS for load in E7_LOADS]
    trials = []
    rng = random.Random(seed(order_tag, mode))
    for block, phase, role in blocks:
        order = list(cells)
        rng.shuffle(order)
        for cell in order:
            net = E7_NETWORKS[cell.split("-")[0]]
            workload = e7_workload(block, cell)
            order_methods = list(methods)
            rng.shuffle(order_methods)
            for method in order_methods:
                tmp_paths = {h: (net[h], "fine_fifo", net["load"][h]) for h in (0, 1)}
                assignment, scheduler = e7_assign(method, model, workload, tmp_paths)
                paths = {h: (net[h], scheduler, net["load"][h]) for h in (0, 1)}
                fam = "MH-E7-%s-%s" % (block, cell)
                reqs = []
                for q, a in zip(workload, assignment):
                    reqs.append(request(fam, q["role"], q["prefix"], a["action"], a["holder"], offset=q["offset"],
                                        prefix_location=q["location"], slo_ttft_s=model["slo"]["ttft_s"][str(q["prefix"])],
                                        slo_mean_itl_s=model["slo"]["mean_itl_s"]))
                trials.append(trial("E7", phase, role, block, cell, method, reqs, paths, timeout=1200))
    return finalize("E7", "e7_" + mode, run_id, trials, 2, {"model_sha256": model["model_sha256"], "methods": list(methods),
                    "cells": cells, "slo": model["slo"], "b_star_bps": model["b_star_bps"],
                    "oracle": "post-hoc best measured method per (block, cell)"})


# ---------------------------------------------------------------- experiments 3/4 (rate sweep, no holder decode) and 5 (RTT)

CHUNK_TIME_S = 16384 * 8 / 100_000_000      # 1.31072 ms: the E4 16 KiB chunk at 100 Mbps
X3_METHODS = ("full_controller", "ablate_no_contention", "static_holder", "static_recompute", "static_repair",
              "threshold_bstar")
X4_METHODS = ("full_controller", "ablate_no_contention")   # static_recompute/static_repair rows are shared with X3
X3_CELLS = ("fast-high", "fast-low")
X5_METHODS = ("full_controller", "static_repair", "static_holder")


def rate_label(rate):
    return {GBPS: "1G", SLOW: "250M"}.get(rate, "%dM" % round(rate / 1e6))


def scaled_chunk(rate):
    return max(16384, int(round(rate * CHUNK_TIME_S / 8)))


def build_x34_calibration(run_id, rates, reps=2):
    """Per-rate standalone cost of every holder action on both holders (scaled chunks, idle) and
    per-rate same-resource slowdown for the rate-sensitive actions remote/repair (k=2,4)."""
    trials = []
    for rep in range(reps):
        fam = "MH-x34cal-r%d" % rep
        for h in (0, 1):
            for prefix in (1024, 4096):
                for rate in rates:
                    for action in HOLDER_ACTIONS:
                        cell = "solo-%s-h%d-P%d-%s-idle" % (action, h, prefix, rate_label(rate))
                        trials.append(trial("CAL", "calibration", "calibration", "r%d" % rep, cell, "solo",
                                            [request(fam + ":" + cell, "A", prefix, action, h)],
                                            {h: (rate, "fine_fifo", False, scaled_chunk(rate))}))
        for rate in rates:
            for k in (2, 4):
                for action in ("remote", "repair"):
                    cell = "group-%s-k%d-P1024-%s-idle" % (action, k, rate_label(rate))
                    reqs = [request(fam + ":" + cell, "R%d" % i, 1024, action, 0) for i in range(k)]
                    trials.append(trial("CAL", "calibration", "calibration", "r%d" % rep, cell, "same_resource", reqs,
                                        {0: (rate, "fine_fifo", False, scaled_chunk(rate))}))
    return finalize("CAL", "x34_calibration", run_id, trials, 2, {"rates": list(rates), "chunk_time_s": CHUNK_TIME_S})


def fit_model_x34(records, base, rates, per_holder=False, loaded_holders=()):
    """Exact per-rate model: holder actions from the x34 calibration, recompute/holder slowdown, SLO and B*
    from the base (experiment-1) model, per-rate slowdown for remote/repair."""
    solo, groups, resid = {}, {}, {}
    for r in records:
        if r["status"] != "passed" or r["stage"] != "CAL" or r["sample_role"] != "calibration":
            continue
        if r["method"] == "solo":
            q = r["requests"][0]
            parts = r["cell_id"].split("-")
            key = "%s|%d|%s|%s" % (q["action"], q["prefix_tokens"], parts[4], parts[5])
            solo.setdefault(key + ("|" + parts[2] if per_holder else ""), []).append(q)
        else:
            groups.setdefault(r["cell_id"], []).append(r)
    model = {"standalone": {k: copy.deepcopy(v) for k, v in base["standalone"].items() if k.startswith("recompute|")},
             "slowdown": copy.deepcopy(base["slowdown"]), "slowdown_by_rate": {},
             "radius": {"recompute": base["radius"].get("recompute", 0.)}}
    for key, rows in solo.items():
        comp = [q["completion_s"] for q in rows]
        med = statistics.median(comp)
        model["standalone"][key] = dict(completion_s=med, ttft_s=statistics.median(q["ttft_s"] for q in rows),
                                        mean_itl_s=statistics.median(q["mean_itl_s"] for q in rows), n=len(rows))
        resid.setdefault(key.split("|")[0], []).extend(abs(c - med) / med for c in comp)
    for action, values in resid.items():
        values = sorted(values)
        model["radius"][action] = values[max(0, math.ceil(.9 * len(values)) - 1)] if values else 0.
    for cell, rows in groups.items():
        parts = cell.split("-")
        action, k, label = parts[1], int(parts[2][1:]), parts[4]
        b = model["standalone"]["%s|1024|%s|idle" % (action, label) + ("|h0" if per_holder else "")]["completion_s"]
        ratios = [q["completion_s"] / b for r in rows for q in r["requests"]]
        model["slowdown_by_rate"].setdefault(action, {}).setdefault(label, {"1": 1.0})[str(k)] = statistics.median(ratios)
    needed = {"%s|%d|%s|%s" % (a, p, rate_label(r), "load" if h in loaded_holders else "idle") + ("" if h is None else "|h%d" % h)
              for a in HOLDER_ACTIONS for p in (1024, 4096) for r in rates for h in ((0, 1) if per_holder else (None,))}
    missing = sorted(needed - set(model["standalone"]))
    if missing:
        raise ValueError("Calibration lacks exact keys: %s" % missing[:5])
    model.update(slo=copy.deepcopy(base["slo"]), b_star_bps=base["b_star_bps"], b_star_rule=base["b_star_rule"],
                 exact_rate_keys=True, base_model_sha256=base["model_sha256"], rates=[rate_label(r) for r in rates],
                 chunk_time_s=CHUNK_TIME_S)
    if per_holder:
        model["per_holder_keys"] = True
    if loaded_holders:
        model["loaded_holders"] = sorted(loaded_holders)
    model["model_sha256"] = digest(model)
    return model


def _blocks(mode):
    return [("smoke", "smoke", "functional_smoke")] if mode == "smoke" else \
        [("warmup", "warmup", "warmup")] + [("b%02d" % i, "test", "measured") for i in range(8)]


def _requests(model, block, cell, workload, assignment):
    fam = "MH-E7-%s-%s" % (block, cell)
    return [request(fam, q["role"], q["prefix"], a["action"], a["holder"], offset=q["offset"], prefix_location=q["location"],
                    slo_ttft_s=model["slo"]["ttft_s"][str(q["prefix"])], slo_mean_itl_s=model["slo"]["mean_itl_s"])
            for q, a in zip(workload, assignment)]


def build_x34(run_id, model, mode, rates, prereg_sha256):
    """Experiments 3+4 in one plan: symmetric fast cells x rate grid; X3 methods with holder decode,
    X4 adaptive methods without it (static_recompute/static_repair never use holder decode, so their
    rows serve both). Workloads are identical across rates within a block."""
    rng = random.Random(seed("X34-order", mode))
    trials, rows = [], {}
    for block, phase, role in _blocks(mode):
        units = [(rate, cell) for rate in rates for cell in X3_CELLS]
        rng.shuffle(units)
        for rate, cell in units:
            workload = e7_workload(block, cell)
            arms = [(m, True) for m in X3_METHODS] + [(m, False) for m in X4_METHODS]
            rng.shuffle(arms)
            for method, allow_holder in arms:
                tmp = {h: (rate, "fine_fifo", False, scaled_chunk(rate)) for h in (0, 1)}
                assignment, scheduler = e7_assign(method, model, workload, tmp, allow_holder)
                paths = {h: (rate, scheduler, False, scaled_chunk(rate)) for h in (0, 1)}
                cell_id = "%s@%s%s" % (cell, rate_label(rate), "" if allow_holder else "@noholder")
                trials.append(trial("E7", phase, role, block, cell_id, method,
                                    _requests(model, block, cell, workload, assignment), paths, timeout=1200))
                rows[(block, rate_label(rate), cell, method + ("" if allow_holder else "~noholder"))] = \
                    [(a["action"], a["holder"]) for a in assignment]
    identical = {}
    for (block, label, cell, method), value in rows.items():
        for other in ("static_recompute", "static_repair"):
            if method != other and rows.get((block, label, cell, other)) == value:
                identical.setdefault("%s==%s" % (method, other), []).append("%s|%s|%s" % (block, label, cell))
    return finalize("E7", "x34_" + mode, run_id, trials, 2, {
        "experiment": "X34", "model_sha256": model["model_sha256"], "rates_bps": list(rates),
        "rate_labels": [rate_label(r) for r in rates], "chunk_bytes": {rate_label(r): scaled_chunk(r) for r in rates},
        "b_star_bps": model["b_star_bps"], "methods_x3": list(X3_METHODS), "methods_x4_noholder": list(X4_METHODS),
        "cells": list(X3_CELLS), "slo": model["slo"], "preregistration_sha256": prereg_sha256,
        "identical_rows": {k: len(v) for k, v in identical.items()}, "identical_row_units": identical})


def build_x5(run_id, model, mode, rate, cell, rtts, prereg_sha256):
    """Experiment 5: one symmetric cell at one rate, both holders with the same added RTT; decisions use the
    RTT-unaware model, so assignments are identical across the RTT grid within a block."""
    rng = random.Random(seed("X5-order", mode))
    trials = []
    for block, phase, role in _blocks(mode):
        units = list(rtts)
        rng.shuffle(units)
        workload = e7_workload(block, cell)
        for rtt in units:
            methods = list(X5_METHODS)
            rng.shuffle(methods)
            for method in methods:
                tmp = {h: (rate, "fine_fifo", False, scaled_chunk(rate)) for h in (0, 1)}
                assignment, scheduler = e7_assign(method, model, workload, tmp)
                paths = {h: (rate, scheduler, False, scaled_chunk(rate), rtt) for h in (0, 1)}
                cell_id = "%s@%s@rtt%gms" % (cell, rate_label(rate), rtt)
                trials.append(trial("E7", phase, role, block, cell_id, method,
                                    _requests(model, block, cell, workload, assignment), paths, timeout=1200))
    return finalize("E7", "x5_" + mode, run_id, trials, 2, {
        "experiment": "X5", "model_sha256": model["model_sha256"], "rate_bps": rate, "cell": cell,
        "holder_added_rtt_ms_grid": [float(x) for x in rtts], "methods": list(X5_METHODS), "slo": model["slo"],
        "chunk_bytes": scaled_chunk(rate), "preregistration_sha256": prereg_sha256})
