"""Summarize the final simulation protocol without pooling mobility seeds.

Read a campaign tree CELL/METHOD/seed_N.json, or the released compact seed CSV.
Percentiles use evaluation-half follow-up requests, linear interpolation, and
stall-aware TPOT in milliseconds. Paired ratios use the campaign's rounded
Student-t critical values (20 seeds: 2.093; 10 seeds: 2.262).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

PERCENTILES = (50, 90, 99)
METRICS = [(metric, p) for metric in ("ttft", "tpot") for p in PERCENTILES]
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 12: 2.201, 15: 2.145, 20: 2.093}
FIELDS = ["campaign", "cell", "method", "seed", "n_followup"] + [
    f"{metric}_p{p}_{'s' if metric == 'ttft' else 'ms'}" for metric, p in METRICS
] + ["load", "share", "cap_mbps", "model"]


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("No eligible follow-up samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def field(metric: str, percentile: int) -> str:
    return f"{metric}_p{percentile}_{'s' if metric == 'ttft' else 'ms'}"


def read_run(path: Path, campaign: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    requests = [r for r in data["per_request"] if r[3] and r[2] >= 2]
    values = {
        "ttft": [float(r[4]) for r in requests],
        "tpot": [float(r[11]) * 1000 for r in requests if r[11] is not None],
    }
    args = data["args"]
    row = {
        "campaign": campaign, "cell": path.parent.parent.name,
        "method": path.parent.name, "seed": int(path.stem.removeprefix("seed_")),
        "n_followup": len(requests), "load": args["load"], "share": args["share"],
        "cap_mbps": args["cap"], "model": args["model"],
    }
    for metric, percentile in METRICS:
        row[field(metric, percentile)] = quantile(values[metric], percentile / 100)
    return row


def paired_interval(ratios: list[float]) -> dict:
    n = len(ratios)
    if n not in T975:
        raise ValueError(f"No campaign t critical value for {n} seeds; use 10 or 20 for paper runs")
    average = statistics.mean(ratios)
    half = T975[n] * statistics.stdev(ratios) / math.sqrt(n)
    return {"ratio": average, "lo": average - half, "hi": average + half, "n": n}


def summarize(rows: list[dict], ours: str = "mobikv", comparisons: str = "best") -> tuple[list, dict]:
    cells = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        seed = int(row["seed"])
        group = cells[row["cell"]][row["method"]]
        if seed in group:
            raise ValueError(f"Duplicate cell/method/seed: {row['cell']}/{row['method']}/{seed}")
        group[seed] = row
    if not cells:
        raise ValueError("No simulation runs selected")
    latencies, ratios = [], {}
    for cell, methods in sorted(cells.items()):
        if ours not in methods:
            raise ValueError(f"Missing {ours} in {cell}")
        expected_seeds = set(methods[ours])
        if any(set(seed_rows) != expected_seeds for seed_rows in methods.values()):
            raise ValueError(f"Unpaired or incomplete seed sets in {cell}")
        if len(expected_seeds) < 2:
            raise ValueError(f"At least two seeds are needed in {cell}")
        mean = {}
        for method, seed_rows in sorted(methods.items()):
            mean[method] = {}
            for metric, percentile in METRICS:
                key = field(metric, percentile)
                values = [float(row[key]) for row in seed_rows.values()]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Non-finite statistic in {cell}/{method}/{key}")
                mean[method][key] = statistics.mean(values)
                latencies.append({"cell": cell, "method": method, "metric": metric,
                                  "percentile": f"P{percentile}", "mean": mean[method][key],
                                  "std": statistics.stdev(values), "n_seed": len(values)})
        baselines = sorted(method for method in methods if method != ours)
        if not baselines:
            raise ValueError(f"No comparison methods in {cell}")
        results = []
        for metric, percentile in METRICS:
            key = field(metric, percentile)
            selected = [min(baselines, key=lambda method: mean[method][key])] if comparisons == "best" else baselines
            for baseline in selected:
                values = []
                for seed in sorted(expected_seeds):
                    denominator = float(methods[baseline][seed][key])
                    if denominator <= 0:
                        raise ValueError(f"Non-positive baseline statistic: {cell}/{baseline}/{seed}")
                    values.append(float(methods[ours][seed][key]) / denominator)
                result = {"metric": metric, "percentile": f"P{percentile}",
                          "baseline": baseline, "ours_mean": mean[ours][key],
                          "base_mean": mean[baseline][key], **paired_interval(values)}
                results.append(result)
        ratios[cell] = {"n_seed": len(expected_seeds), "comparisons": results}
    return latencies, {"ours": ours, "comparison": comparisons, "cells": ratios}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=Path, help="Campaign tree with CELL/METHOD/seed_N.json")
    source.add_argument("--seeds", type=Path, help="Released compact per-seed CSV")
    parser.add_argument("--campaign", help="Required CSV campaign selector")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ours", default="mobikv")
    parser.add_argument("--comparisons", choices=("best", "each"), default="best",
                        help="Use 'each' for the six individual ablations")
    args = parser.parse_args()
    if args.root:
        rows = [read_run(path, args.root.name) for path in sorted(args.root.glob("*/*/seed_*.json"))]
        precision = "Percentiles recomputed from per-request simulation outputs."
    else:
        if not args.campaign:
            parser.error("--seeds requires --campaign")
        with args.seeds.open(encoding="utf-8", newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["campaign"] == args.campaign]
        precision = "Released seed statistics are rounded to six decimals (seconds) and four decimals (milliseconds); final digits can differ from unrounded results."
    latencies, ratios = summarize(rows, args.ours, args.comparisons)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "latencies.csv", latencies, list(latencies[0]))
    write_csv(args.out / "seeds.csv", rows, FIELDS)
    ratios["precision"] = precision
    (args.out / "ratios.json").write_text(json.dumps(ratios, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Summarized {len(rows)} seed runs into {len(latencies)} latency cells: {args.out}")


if __name__ == "__main__":
    main()
