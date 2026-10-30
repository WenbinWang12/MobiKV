"""Holder release times from the raw per-trial reports (DEP, 2026-10-30).

Release = time from the request's arrival on the target until the holder is no longer needed for it:
holder/remote -> completion; repair -> the latest first-local-use transition over all layers (after it no
remote-attention call is made; the KV migration has finished before it); recompute -> 0. Target clock only.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics

import mh_plan as P


def require(value, message):
    if not value:
        raise RuntimeError(message)


def candidate(stage_dir, record):
    files = sorted((Path(stage_dir) / "trials" / record["trial_id"]).glob("candidate-*.json"))
    match = [f for f in files if hashlib.sha256(f.read_bytes()).hexdigest() == record["measured_artifact_sha256"]]
    require(len(match) == 1, "No unique raw report for " + record["trial_id"])
    return json.loads(match[0].read_text())


def releases(raw, planned_requests):
    out = {}
    for spec in planned_requests:
        q = raw["requests"][spec["role"]]
        require(q["mh_action"] == spec["action"], "Raw action differs from plan: " + spec["role"])
        if spec["action"] in ("holder", "remote"):
            out[spec["role"]] = q["completion_s"]
        elif spec["action"] == "repair":
            transitions = q["state_report"]["transitions"]
            if transitions:
                out[spec["role"]] = max(t["monotonic_s"] for t in transitions) - q["arrival_s"]
            else:
                # v3: migration won the race, so every layer was already local at its first use and no remote call
                # was made; the holder was needed only until the last layer was published on the target.
                require(not q["remote_calls"], "Repair without cutovers but with remote calls: " + spec["role"])
                out[spec["role"]] = max(p["published_local_s"] for p in q["e4_layer_publications"]) - q["arrival_s"]
        else:
            out[spec["role"]] = 0.0
    return out


def fit_release(model, cal_dir, records, rates):
    """Adds the standalone repair release per (prefix, rate) and its 90th-percentile relative radius."""
    rows = {}
    for r in records:
        if r["status"] != "passed" or r["method"] != "solo" or r["requests"][0]["action"] != "repair":
            continue
        q = r["requests"][0]
        rel = releases(candidate(cal_dir, r), [dict(role=q["role"], action="repair")])[q["role"]]
        rows.setdefault("%d|%s" % (q["prefix_tokens"], r["cell_id"].split("-")[4]), []).append(rel)
    needed = {"%d|%s" % (p, P.rate_label(rate)) for p in (1024, 4096) for rate in rates}
    require(needed <= set(rows), "Calibration lacks repair release keys: %s" % sorted(needed - set(rows)))
    model = dict(model, repair_release={k: dict(release_s=statistics.median(v), n=len(v)) for k, v in sorted(rows.items())})
    resid = sorted(abs(x - statistics.median(v)) / statistics.median(v) for v in rows.values() for x in v)
    model["radius"] = dict(model["radius"], repair_release=resid[max(0, math.ceil(.9 * len(resid)) - 1)])
    model.pop("model_sha256")
    model["model_sha256"] = P.digest(model)
    return model
