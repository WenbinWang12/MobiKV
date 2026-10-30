"""Experiment 1 analysis: preregistered block-paired contrasts and the C1 decision.

Reuses mh_analysis.trial_metrics_e7. Relative improvement of method M over baseline B
is (B - M) / B for lower-is-better metrics and (M - B) / B for higher-is-better ones.
A zero baseline gives a null relative value (absolute difference kept), never a
1e-9 denominator. Labels: advantage if median and 95% bootstrap lower bound > 5%,
disadvantage if median and upper bound < -5%, else tied_or_unresolved; a pair with
any null relative block is labelled relative_undefined. Every planned trial is
listed; a missing or failed trial makes its (cell, pair) undefined, not dropped.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from mh_analysis import trial_metrics_e7, write_csv

DRAWS, SEED = 10000, 2026102801
LOWER = {"mean_completion_s", "p90_completion_s"}
METRICS = ("mean_completion_s", "slo_attainment", "goodput_rps", "p90_completion_s")
PAIRS = (("full_controller", "least_loaded"), ("full_controller", "ablate_no_contention"),
         ("full_controller", "static_holder"), ("least_loaded", "static_holder"))


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


def analyze(plan, records):
    planned = [t for t in plan["trials"] if t["sample_role"] == "measured"]
    passed = {r["trial_id"]: r for r in records if r["status"] == "passed"}
    table, missing = {}, []
    for t in planned:
        r = passed.get(t["trial_id"])
        if r is None:
            missing.append(t["trial_id"])
            continue
        table[(t["cell_id"], t["block_id"], t["method"])] = trial_metrics_e7(r)
    cells = sorted({t["cell_id"] for t in planned})
    blocks = sorted({t["block_id"] for t in planned})
    rows = []
    for cell in cells:
        for focus, base in PAIRS:
            for metric in METRICS:
                rel, absdiff, complete = [], [], True
                for b in blocks:
                    f, v = table.get((cell, b, focus)), table.get((cell, b, base))
                    if f is None or v is None:
                        complete = False
                        continue
                    fv, bv = f[metric], v[metric]
                    absdiff.append(bv - fv if metric in LOWER else fv - bv)
                    rel.append(None if bv == 0 else ((bv - fv) / bv if metric in LOWER else (fv - bv) / bv))
                row = dict(cell=cell, method=focus, baseline=base, metric=metric, n_planned_blocks=len(blocks),
                           n_paired_blocks=len(absdiff), block_relative=rel, block_absolute=absdiff,
                           absolute_median=statistics.median(absdiff) if absdiff else None)
                if not complete or any(x is None for x in rel):
                    row.update(median=None, ci_low=None, ci_high=None,
                               label="relative_undefined" if complete else "incomplete")
                else:
                    lo, hi = interval(rel)
                    med = statistics.median(rel)
                    row.update(median=med, ci_low=lo, ci_high=hi, label=label(med, lo, hi))
                rows.append(row)
    primary = [r for r in rows if r["method"] == "full_controller" and r["baseline"] == "least_loaded"
               and r["metric"] == "mean_completion_s"]
    labels = [r["label"] for r in primary]
    adv, dis = labels.count("advantage"), labels.count("disadvantage")
    if len(primary) == 4 and labels.count("tied_or_unresolved") == 4:
        decision = "C1 rewrite: concurrency count alone suffices; no added value from the cost model"
    elif adv >= 2 and dis == 0:
        decision = "C1 holds with the least_loaded control"
    elif any(l in ("incomplete", "relative_undefined") for l in labels):
        decision = "undetermined: incomplete primary contrasts"
    else:
        decision = "C1 holds only in the advantage cells; report per cell, no general claim"
    replication = [r for r in rows if r["method"] == "full_controller" and r["baseline"] == "ablate_no_contention"
                   and r["metric"] == "mean_completion_s"]
    choices = {}
    for (cell, b, m), v in table.items():
        for c in v["choices"]:
            choices.setdefault(m, {}).setdefault(c.split("@")[0], 0)
            choices[m][c.split("@")[0]] += 1
    summary = dict(experiment="X1", plan_sha256=plan["plan_sha256"], preregistration_sha256=plan.get("preregistration_sha256"),
                   planned_measured_trials=len(planned), passed_measured_trials=len(planned) - len(missing),
                   missing_or_failed=missing, identical_assignment_count=plan.get("identical_assignment_count"),
                   units=plan.get("units"), primary_labels={r["cell"]: r["label"] for r in primary},
                   primary_medians={r["cell"]: r["median"] for r in primary}, c1_decision=decision,
                   replication_full_vs_no_contention={r["cell"]: r["label"] for r in replication},
                   replication_met=sum(r["label"] == "advantage" for r in replication) >= 3,
                   action_counts=choices, scientific_acceptance=False, coverage_claim_allowed=False)
    per_trial = [dict(cell=c, block=b, method=m, **v) for (c, b, m), v in sorted(table.items())]
    return summary, rows, per_trial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    records = json.loads(args.records.read_text())["records"]
    summary, rows, per_trial = analyze(plan, records)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "x1_paired.csv", rows)
    write_csv(args.out / "x1_per_trial.csv", per_trial)
    (args.out / "x1_paired.json").write_text(json.dumps(rows, indent=1))
    (args.out / "x1_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
