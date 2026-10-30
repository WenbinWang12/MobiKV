"""Supplementary TMC analysis (HET / LM / M7B): the experiment-1/3/4 rules, reused from x34_analysis
(block-paired relative improvement, 10,000-draw block bootstrap, +-5% labels, missing pairs never dropped)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import x34_analysis as A
from mh_analysis import write_csv
import supp_release as R

PAIRS = {
    "HET": [("full_controller", b) for b in ("least_loaded", "ablate_no_contention", "static_holder", "static_recompute",
                                             "threshold_bstar")] + [("least_loaded", "static_holder")],
    "LM": [("static_repair", "static_recompute"), ("full_controller", "static_recompute"), ("full_controller", "static_repair"),
           ("full_controller~noholder", "static_recompute"), ("full_controller~noholder", "static_repair"),
           ("static_holder", "static_repair"), ("full_controller", "full_controller~noholder")],
    "M7B": [("full_controller", b) for b in ("least_loaded", "ablate_no_contention", "static_holder", "static_recompute",
                                             "static_repair", "threshold_bstar")] + [("static_repair", "static_recompute"),
                                                                                     ("least_loaded", "static_holder")],
    "DEP": [("full_controller_dl", b) for b in ("full_controller", "static_holder", "static_repair", "static_recompute",
                                                "threshold_bstar")] + [("static_repair", "static_holder")],
}
A.LOWER.add("violation_rate")


def dep_metrics(stage_dir, plan, trial, record, release_rows):
    """Share of the trial's requests whose measured holder release exceeds the cell's availability window."""
    window = plan["windows_s"][trial["cell_id"].split("@")[1]]
    rel = R.releases(R.candidate(stage_dir, record), trial["requests"])
    for spec in trial["requests"]:
        release_rows.append(dict(trial_id=trial["trial_id"], block=trial["block_id"], unit="|".join(unit_of(trial)),
                                 method=trial["method"], role=spec["role"], prefix=spec["prefix_tokens"], action=spec["action"],
                                 release_s=rel[spec["role"]], window_s=window, violated=rel[spec["role"]] > window,
                                 completion_s=next(q["completion_s"] for q in record["requests"] if q["role"] == spec["role"])))
    return dict(violation_rate=sum(rel[s["role"]] > window for s in trial["requests"]) / len(trial["requests"]))


def unit_of(trial):
    return tuple(trial["cell_id"].split("@")[:2])


def analyze(stage_dir, experiment):
    plan, planned, passed, audit = A.load(stage_dir)
    A.require(plan.get("experiment") == experiment, "Plan is not a %s plan" % experiment)
    table, missing, release_rows = {}, [], []
    for t in planned:
        r = passed.get(t["trial_id"])
        if r is None:
            missing.append(t["trial_id"])
            continue
        table[unit_of(t) + (t["block_id"], A.arm(t))] = A.trial_metrics_e7(r)
        if experiment == "DEP":
            table[unit_of(t) + (t["block_id"], A.arm(t))].update(dep_metrics(stage_dir, plan, t, r, release_rows))
    blocks = sorted({t["block_id"] for t in planned})
    units = sorted({unit_of(t) for t in planned})
    arms = {A.arm(t) for t in planned}
    metrics = A.METRICS + (("violation_rate",) if experiment == "DEP" else ())
    rows = [dict(A.contrast(table, u, f, b, m, blocks), experiment=experiment)
            for u in units for f, b in PAIRS[experiment] if f in arms and b in arms for m in metrics]
    if release_rows:
        write_csv(Path(stage_dir).parent / "analysis" / "dep_releases.csv", release_rows)
    sel = A.selection(planned, lambda t: "%s|%s" % ("|".join(unit_of(t)), A.arm(t)))
    slots = {r["trial_id"]: r for r in audit["trial_inventory"]}
    per_trial = [A.trial_report(t, passed.get(t["trial_id"]), slots[t["trial_id"]], unit="|".join(unit_of(t)),
                                block=t["block_id"], arm=A.arm(t)) for t in planned]
    extra = {k: plan[k] for k in ("identical_assignment_count", "identical_assignment_units", "arrival_scale",
                                  "loaded_holder", "heterogeneity", "windows_s") if k in plan}
    return dict(plan_sha256=plan["plan_sha256"], planned=len(planned), missing_or_failed=missing, **extra, **audit), rows, sel, per_trial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--experiment", choices=sorted(PAIRS), required=True)
    ap.add_argument("--stage", required=True)
    args = ap.parse_args()
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    s, rows, sel, per_trial = analyze(args.root / args.stage, args.experiment)
    prefix = args.experiment.lower()
    write_csv(out / (prefix + "_paired.csv"), rows)
    write_csv(out / (prefix + "_selection.csv"), sel)
    write_csv(out / (prefix + "_per_trial.csv"), per_trial)
    A.write_json(out / (prefix + "_paired.json"), rows)
    s["labels"] = {"%s|%s|%s" % (r["unit"], r["method"], r["baseline"]): r["label"]
                   for r in rows if r["metric"] == "mean_completion_s"}
    summary = A.limitations()
    summary.update(analysis_code_sha256=A.sha(__file__), all_attempts_and_failures_retained=True, stage=args.stage)
    summary[args.experiment] = A.export_attempts(out, prefix, s)
    summary.update(integrity_passed=s["integrity_passed"], execution_complete=s["execution_complete"],
                   failure_free=s["failure_free"])
    A.write_json(out / "summary.json", summary)
    print(json.dumps({k: v for k, v in s.items() if k not in ("labels", "attempts", "trial_inventory",
                                                              "identical_assignment_units")}, indent=1, default=str)[:3000])
    return 0 if s["integrity_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
