"""Experiment B (preregistered metrics, descriptive): cost-model prediction accuracy on executed E7-family plans.

Per request: the planner's own bookkeeping prediction for the executed assignment (predict() with the
concurrency k counted exactly as in e7_assign) against the measured completion. Per (unit, block): predicted
and measured trial-mean completion across methods -> Spearman correlation, top-1 agreement, full_controller
regret against the post-hoc best measured method. Usage: supp_predacc.py OUT_JSON NAME=STAGE_DIR=MODEL ...
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mh_plan as P  # noqa: E402


def q(values, p):
    values = sorted(values)
    return values[max(0, math.ceil(p * len(values)) - 1)] if values else None


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def predictions(model, trial):
    paths = {int(h): (v["rate_bps"], v["policy"], v["loaded"]) for h, v in trial["holder_paths"].items()}
    active, out = [], {}
    for r in trial["requests"]:
        choice = (r["action"], r["holder"])
        k = 1 + sum(1 for a in active if a["res"] == P.resource(*choice) and a["finish"] > r["arrival_offset_s"])
        pred = P.predict(model, r["prefix_tokens"], choice, paths, k)
        active.append(dict(res=P.resource(*choice), finish=r["arrival_offset_s"] + pred))
        out[r["role"]] = (r["action"], pred, model["radius"].get(r["action"], 0.0))
    return out


def analyze(stage_dir, model):
    plan = json.loads((Path(stage_dir) / "plan.json").read_text())
    recs = {r["trial_id"]: r for r in json.loads((Path(stage_dir) / "normalized-records.json").read_text())["records"]
            if r["status"] == "passed"}
    per_action, units = {}, {}
    for t in plan["trials"]:
        if t["sample_role"] != "measured" or t["trial_id"] not in recs:
            continue
        measured = {x["role"]: x["completion_s"] for x in recs[t["trial_id"]]["requests"]}
        pred = predictions(model, t)
        for role, (action, p, radius) in pred.items():
            m = measured[role]
            row = per_action.setdefault(action, dict(ape=[], covered=0, n=0))
            row["ape"].append(abs(p - m) / m)
            row["covered"] += abs(m - p) <= radius * p
            row["n"] += 1
        arm = t["method"] + ("~noholder" if t["cell_id"].endswith("@noholder") else "")
        unit = (t["cell_id"].replace("@noholder", ""), t["block_id"])
        units.setdefault(unit, {})[arm] = (statistics.mean(v[1] for v in pred.values()), statistics.mean(measured.values()))
    actions = {a: dict(n=v["n"], median_ape=statistics.median(v["ape"]), p90_ape=q(v["ape"], .9),
                       radius_coverage=v["covered"] / v["n"]) for a, v in sorted(per_action.items())}
    rho, top1, regret = [], [], []
    for unit, arms in units.items():
        names = sorted(arms)
        if len(names) < 3:
            continue
        pv, mv = [arms[n][0] for n in names], [arms[n][1] for n in names]
        r = spearman(pv, mv)
        if r is not None:
            rho.append(r)
        top1.append(names[pv.index(min(pv))] == names[mv.index(min(mv))])
        if "full_controller" in arms:
            regret.append((arms["full_controller"][1] - min(mv)) / min(mv))
    return dict(per_action=actions, n_units=len(units), spearman_median=statistics.median(rho) if rho else None,
                spearman_q25_q75=[q(rho, .25), q(rho, .75)] if rho else None, top1_agreement=sum(top1) / len(top1) if top1 else None,
                fc_regret_median=statistics.median(regret) if regret else None, fc_regret_p90=q(regret, .9),
                fc_is_best_share=sum(1 for x in regret if x <= 1e-12) / len(regret) if regret else None)


def main():
    out = {}
    for spec in sys.argv[2:]:
        name, stage_dir, model_path = spec.split("=")
        model = json.loads(Path(model_path).read_text())
        out[name] = analyze(stage_dir, model)
        out[name]["model_sha256"] = model["model_sha256"]
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
