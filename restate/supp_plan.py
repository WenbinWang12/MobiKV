"""Supplementary TMC plans (2026-10-30), plan-level only; the runtime is unchanged.

HET  heterogeneous holders: one holder GPU under the E7 background load (v2; an MPS cap on one holder failed
     the numerical gate in smoke), per-holder calibrated costs, symmetric network, E7 fast cells;
     full_controller against least_loaded (same feasible set and bookkeeping, no cost comparison) and the
     static arms.
LM   load-matched throttle: the C3T configuration (MPS cap on all three GPUs, C3T cost model) with every
     arrival offset stretched by one preregistered factor, so offered load relative to service time
     matches the native runs; the C3 arms at the low and high rate points.
M7B  second model: the E7 cells and methods of experiment 1 plus the static arms, on Qwen2.5-7B.
"""
from __future__ import annotations

import random

import mh_plan as P

HET_METHODS = ("full_controller", "least_loaded", "ablate_no_contention", "static_holder", "static_recompute",
               "threshold_bstar")
HET_CELLS = ("fast-high", "fast-low")
LM_X3 = ("full_controller", "static_holder", "static_recompute", "static_repair")
LM_X4 = ("full_controller",)
M7B_METHODS = ("full_controller", "least_loaded", "ablate_no_contention", "static_holder", "static_recompute",
               "static_repair", "threshold_bstar")


def scaled_workload(block, cell, scale):
    """The E7 workload with every arrival offset multiplied by scale (same draws, same order)."""
    return [dict(q, offset=q["offset"] * scale) for q in P.e7_workload(block, cell)]


def build_het_calibration(run_id, rate, loaded_holder, reps=2):
    """v2 (after the MPS smoke failed the numerical gate): every holder action standalone on both holders with
    holder `loaded_holder` under the E7 background load, plus per-rate remote/repair slowdown on idle holder 0."""
    trials = []
    for rep in range(reps):
        fam = "MH-hetcal-r%d" % rep
        for h in (0, 1):
            load = h == loaded_holder
            for prefix in (1024, 4096):
                for action in P.HOLDER_ACTIONS:
                    cell = "solo-%s-h%d-P%d-%s-%s" % (action, h, prefix, P.rate_label(rate), "load" if load else "idle")
                    trials.append(P.trial("CAL", "calibration", "calibration", "r%d" % rep, cell, "solo",
                                          [P.request(fam + ":" + cell, "A", prefix, action, h)],
                                          {h: (rate, "fine_fifo", load, P.scaled_chunk(rate))}))
        for k in (2, 4):
            for action in ("remote", "repair"):
                cell = "group-%s-k%d-P1024-%s-idle" % (action, k, P.rate_label(rate))
                reqs = [P.request(fam + ":" + cell, "R%d" % i, 1024, action, 0) for i in range(k)]
                trials.append(P.trial("CAL", "calibration", "calibration", "r%d" % rep, cell, "same_resource", reqs,
                                      {0: (rate, "fine_fifo", False, P.scaled_chunk(rate))}))
    return P.finalize("CAL", "het_calibration", run_id, trials, 2, {"rate": rate, "loaded_holder": loaded_holder,
                                                                     "chunk_time_s": P.CHUNK_TIME_S})


def build_het(run_id, model, mode, rate, loaded_holder, prereg_sha256):
    pathset = lambda policy: {h: (rate, policy, h == loaded_holder, P.scaled_chunk(rate)) for h in (0, 1)}
    rng = random.Random(P.seed("HET-order", mode))
    trials, rows = [], {}
    for block, phase, role in P._blocks(mode):
        cells = list(HET_CELLS)
        rng.shuffle(cells)
        for cell in cells:
            workload = P.e7_workload(block, cell)
            methods = list(HET_METHODS)
            rng.shuffle(methods)
            for method in methods:
                assignment, scheduler = P.e7_assign(method, model, workload, pathset("fine_fifo"))
                trials.append(P.trial("E7", phase, role, block, "%s@%s" % (cell, P.rate_label(rate)), method,
                                      P._requests(model, block, cell, workload, assignment), pathset(scheduler), timeout=1200))
                rows[(block, cell, method)] = [(a["action"], a["holder"]) for a in assignment]
    same = sorted("%s|%s" % (b, c) for (b, c, m), v in rows.items()
                  if m == "least_loaded" and rows[(b, c, "full_controller")] == v)
    return P.finalize("E7", "het_" + mode, run_id, trials, 2, {
        "experiment": "HET", "model_sha256": model["model_sha256"], "rate_bps": rate, "loaded_holder": loaded_holder,
        "heterogeneity": "E7 background load on one holder (time-shared GPU, unchanged kernels)",
        "methods": list(HET_METHODS), "cells": list(HET_CELLS),
        "slo": model["slo"], "b_star_bps": model["b_star_bps"], "preregistration_sha256": prereg_sha256,
        "identical_assignment_units": same, "identical_assignment_count": len(same)})


def build_lm(run_id, model, mode, rates, scale, prereg_sha256):
    rng = random.Random(P.seed("LM-order", mode))
    trials = []
    for block, phase, role in P._blocks(mode):
        units = [(rate, cell) for rate in rates for cell in P.X3_CELLS]
        rng.shuffle(units)
        for rate, cell in units:
            workload = scaled_workload(block, cell, scale)
            arms = [(m, True) for m in LM_X3] + [(m, False) for m in LM_X4]
            rng.shuffle(arms)
            for method, allow_holder in arms:
                tmp = {h: (rate, "fine_fifo", False, P.scaled_chunk(rate)) for h in (0, 1)}
                assignment, scheduler = P.e7_assign(method, model, workload, tmp, allow_holder)
                paths = {h: (rate, scheduler, False, P.scaled_chunk(rate)) for h in (0, 1)}
                cell_id = "%s@%s%s" % (cell, P.rate_label(rate), "" if allow_holder else "@noholder")
                trials.append(P.trial("E7", phase, role, block, cell_id, method,
                                      P._requests(model, block, cell, workload, assignment), paths, timeout=1200))
    return P.finalize("E7", "lm_" + mode, run_id, trials, 2, {
        "experiment": "LM", "model_sha256": model["model_sha256"], "rates_bps": list(rates), "arrival_scale": scale,
        "rate_labels": [P.rate_label(r) for r in rates], "methods_x3": list(LM_X3), "methods_x4_noholder": list(LM_X4),
        "cells": list(P.X3_CELLS), "slo": model["slo"], "b_star_bps": model["b_star_bps"],
        "preregistration_sha256": prereg_sha256})


DEP_METHODS = ("full_controller_dl", "full_controller", "static_holder", "static_repair", "static_recompute",
               "threshold_bstar")
DEP_CELLS = ("fast-high", "fast-low")


def release_estimate(model, prefix, choice, paths, k):
    """Predicted time from arrival until the holder is no longer needed (radius-inflated): completion for
    holder/remote, last-layer local cutover for repair, zero for recompute."""
    action, h = choice
    if action == "recompute":
        return 0.0
    if action == "repair":
        rate = paths[h][0]
        base = model["repair_release"]["%d|%s" % (prefix, P.rate_label(rate))]["release_s"]
        return base * P.slow(model, "repair", k, rate) * (1 + model["radius"]["repair_release"])
    return P.predict(model, prefix, choice, paths, k, radius=True)


def dep_assign(method, model, workload, paths, window):
    """full_controller_dl: full_controller's scoring, occupancy bookkeeping and scheduler, with every holder
    option whose predicted release exceeds the holder availability window removed. Other methods are the
    unchanged E7 policies (deadline-unaware)."""
    if method != "full_controller_dl":
        return P.e7_assign(method, model, workload, paths)
    out, active = [], []
    for q in workload:
        avail = {"both": [0, 1], "h0": [0], "h1": [1], "none": []}[q["location"]]
        avail = [h for h in avail if sum(1 for o in out if o["holder"] == h) < P.MAX_PER_HOLDER]
        p = q["prefix"]

        def k_of(c):
            return 1 + sum(1 for a in active if a["res"] == P.resource(*c) and a["finish"] > q["offset"])
        cands = [c for c in P.options(None, paths, avail) if release_estimate(model, p, c, paths, k_of(c)) <= window]
        choice = min(cands, key=lambda c: P.predict(model, p, c, paths, k_of(c), contention=True, radius=True))
        finish = q["offset"] + P.predict(model, p, choice, paths, k_of(choice))
        active.append(dict(res=P.resource(*choice), finish=finish))
        out.append(dict(action=choice[0], holder=choice[1]))
    return out, "age_bounded_rpc_first"


def dep_windows(model, rate):
    """Preregistered holder availability windows: 1.5x the standalone repair release and 1.5x the standalone
    holder-decode completion, both for the 4096-token prefix at the experiment rate."""
    label = P.rate_label(rate)
    return {"Tshort": 1.5 * model["repair_release"]["4096|%s" % label]["release_s"],
            "Tlong": 1.5 * model["standalone"]["holder|4096|%s|idle" % label]["completion_s"]}


def build_dep(run_id, model, mode, rate, prereg_sha256):
    windows = dep_windows(model, rate)
    rng = random.Random(P.seed("DEP-order", mode))
    trials = []
    for block, phase, role in P._blocks(mode):
        units = [(w, cell) for w in sorted(windows) for cell in DEP_CELLS]
        rng.shuffle(units)
        for w, cell in units:
            workload = P.e7_workload(block, cell)
            methods = list(DEP_METHODS)
            rng.shuffle(methods)
            for method in methods:
                tmp = {h: (rate, "fine_fifo", False, P.scaled_chunk(rate)) for h in (0, 1)}
                assignment, scheduler = dep_assign(method, model, workload, tmp, windows[w])
                paths = {h: (rate, scheduler, False, P.scaled_chunk(rate)) for h in (0, 1)}
                trials.append(P.trial("E7", phase, role, block, "%s@%s" % (cell, w), method,
                                      P._requests(model, block, cell, workload, assignment), paths, timeout=1200))
    return P.finalize("E7", "dep_" + mode, run_id, trials, 2, {
        "experiment": "DEP", "model_sha256": model["model_sha256"], "rate_bps": rate, "windows_s": windows,
        "methods": list(DEP_METHODS), "cells": list(DEP_CELLS), "slo": model["slo"], "b_star_bps": model["b_star_bps"],
        "preregistration_sha256": prereg_sha256})


def build_m7b(run_id, model, mode, prereg_sha256):
    plan = P.build_e7(run_id, model, mode, methods=M7B_METHODS, order_tag="M7B-order")
    plan.pop("plan_sha256")
    plan.update(experiment="M7B", preregistration_sha256=prereg_sha256, kv_bytes_per_token=P.KV_BYTES_PER_TOKEN)
    plan["plan_sha256"] = P.digest(plan)
    return plan
