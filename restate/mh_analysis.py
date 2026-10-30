"""E6/E7 CPU analysis of normalized records -> tables (CSV) and summary JSON.

Units are the eight held-out input blocks; methods are paired within (cell, block).
Relative improvement of method M over baseline B is (B - M) / B on a
lower-is-better metric (for SLO attainment/goodput it is (M - B) / max(B, eps) and
higher is better). Approximate 95% intervals come from a 10,000-draw bootstrap of
blocks; labels: advantage (median and lower bound > 5%), disadvantage (median and
upper bound < -5%), else tied_or_unresolved. Descriptive only: eight blocks give
no familywise or coverage guarantee. The oracle is the post-hoc best measured
method per (cell, block) and uses information no online method has.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics

DRAWS = 10000


def load(path):
    records = json.loads(Path(path).read_text())["records"]
    return [r for r in records if r["status"] == "passed" and r["sample_role"] == "measured"]


def interval(values, seed=2026102301):
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choice(values) for _ in values) for _ in range(DRAWS))
    return meds[int(.025 * DRAWS)], meds[int(.975 * DRAWS) - 1]


def label(med, lo, hi):
    if med > .05 and lo > .05:
        return "advantage"
    if med < -.05 and hi < -.05:
        return "disadvantage"
    return "tied_or_unresolved"


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def trial_metrics_e6(r):
    q = r["requests"]
    return dict(mean_completion_s=statistics.fmean(x["completion_s"] for x in q),
                makespan_s=max(x["completion_s"] for x in q), mean_ttft_s=statistics.fmean(x["ttft_s"] for x in q),
                mean_itl_s=statistics.fmean(x["mean_itl_s"] for x in q), max_itl_s=max(x["max_itl_s"] for x in q),
                migrated_bytes=sum(x["migrated_bytes"] for x in q), remote_calls=sum(x["remote_calls"] for x in q),
                choices=sorted("%s@%s" % (x["action"], x["holder"]) for x in q))


def trial_metrics_e7(r):
    q = r["requests"]
    met = [x["ttft_s"] <= x["slo_ttft_s"] and x["mean_itl_s"] <= x["slo_mean_itl_s"] for x in q]
    span = max(x["arrival_offset_s"] + x["settled_s"] for x in q) - min(x["arrival_offset_s"] for x in q)
    return dict(slo_attainment=sum(met) / len(met), goodput_rps=sum(met) / span, episode_span_s=span,
                mean_completion_s=statistics.fmean(x["completion_s"] for x in q),
                p90_completion_s=sorted(x["completion_s"] for x in q)[math.ceil(.9 * len(q)) - 1],
                mean_ttft_s=statistics.fmean(x["ttft_s"] for x in q), mean_itl_s=statistics.fmean(x["mean_itl_s"] for x in q),
                migrated_bytes=sum(x["migrated_bytes"] for x in q), remote_calls=sum(x["remote_calls"] for x in q),
                choices=sorted("%s@%s" % (x["action"], x["holder"]) for x in q))


def analyze(records, stage, focus):
    metric_fn = trial_metrics_e6 if stage == "E6" else trial_metrics_e7
    lower_better = {"mean_completion_s", "makespan_s", "mean_ttft_s", "mean_itl_s", "max_itl_s", "p90_completion_s",
                    "migrated_bytes", "remote_calls", "episode_span_s"}
    primary = ["mean_completion_s", "makespan_s", "mean_ttft_s"] if stage == "E6" else \
        ["slo_attainment", "goodput_rps", "mean_completion_s", "p90_completion_s"]
    table = {}
    for r in records:
        table[(r["cell_id"], r["block_id"], r["method"])] = metric_fn(r)
    cells = sorted({k[0] for k in table})
    blocks = sorted({k[1] for k in table})
    methods = sorted({k[2] for k in table})
    per_trial = [dict(cell=c, block=b, method=m, **table[(c, b, m)]) for (c, b, m) in sorted(table)]
    summary_rows = []
    for c in cells:
        for m in methods:
            rows = [table[(c, b, m)] for b in blocks if (c, b, m) in table]
            if rows:
                summary_rows.append(dict(cell=c, method=m, n_blocks=len(rows),
                                         **{k: statistics.median(x[k] for x in rows) for k in primary}))
    oracle_rows, paired = [], []
    for c in cells:
        for b in blocks:
            present = {m: table[(c, b, m)] for m in methods if (c, b, m) in table}
            if not present:
                continue
            key = primary[0]
            best = (min if key in lower_better else max)(present, key=lambda m: present[m][key])
            oracle_rows.append(dict(cell=c, block=b, oracle_method=best, oracle_value=present[best][key],
                                    focus_value=present.get(focus, {}).get(key), metric=key))
        for base in methods:
            if base == focus:
                continue
            for key in primary:
                values = []
                for b in blocks:
                    if (c, b, focus) in table and (c, b, base) in table:
                        f, v = table[(c, b, focus)][key], table[(c, b, base)][key]
                        if key in lower_better:
                            values.append((v - f) / v)
                        else:
                            values.append((f - v) / max(abs(v), 1e-9))
                if len(values) >= 4:
                    lo, hi = interval(values)
                    med = statistics.median(values)
                    paired.append(dict(cell=c, method=focus, baseline=base, metric=key, n_blocks=len(values), median=med,
                                       ci_low=lo, ci_high=hi, label=label(med, lo, hi), block_values=values))
    overall = {}
    for base in methods:
        if base == focus:
            continue
        for key in primary:
            labels = [p["label"] for p in paired if p["baseline"] == base and p["metric"] == key]
            overall["%s|%s" % (base, key)] = dict(advantage=labels.count("advantage"), disadvantage=labels.count("disadvantage"),
                                                  tied=labels.count("tied_or_unresolved"), cells=len(labels))
    gaps = [(o["focus_value"] - o["oracle_value"]) / o["oracle_value"] if primary[0] in lower_better else
            (o["oracle_value"] - o["focus_value"]) / max(o["oracle_value"], 1e-9)
            for o in oracle_rows if o["focus_value"] is not None and o["oracle_value"]]
    oracle_wins = {}
    for o in oracle_rows:
        oracle_wins[o["oracle_method"]] = oracle_wins.get(o["oracle_method"], 0) + 1
    return dict(per_trial=per_trial, summary=summary_rows, paired=paired, oracle=oracle_rows,
                overall=dict(focus=focus, label_counts=overall, oracle_best_method_counts=oracle_wins,
                             focus_oracle_gap_median=statistics.median(gaps) if gaps else None,
                             cells=cells, methods=methods, blocks=blocks, records=len(records)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--stage", choices=("E6", "E7"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    focus = "joint" if args.stage == "E6" else "full_controller"
    result = analyze(load(args.records), args.stage, focus)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("per_trial", "summary", "paired", "oracle"):
        write_csv(args.out / ("%s_%s.csv" % (args.stage.lower(), name)), result[name])
    (args.out / ("%s_summary.json" % args.stage.lower())).write_text(json.dumps(result["overall"], indent=1, sort_keys=True))
    print(json.dumps(result["overall"], indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
