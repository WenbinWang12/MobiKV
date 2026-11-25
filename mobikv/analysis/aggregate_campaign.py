#!/usr/bin/env python3
"""Aggregate per-seed simulator outputs without pooling seeds.

The simulator JSON already contains one row per request. This tool preserves
those rows as JSONL, computes request-level P50/P90/P95/P99 for each seed, and
then computes mean and sample standard deviation across seed-level estimates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

PERCENTILES = (0.50, 0.90, 0.95, 0.99)


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def sample_std(values: list[float]) -> float | None:
    return float(stdev(values)) if len(values) > 1 else None


def method_name(path: Path, payload: dict) -> str:
    # Campaign runners keep summaries under <model>/<method>/seed_<n>/.
    # Prefer that label so two rows sharing the same simulator policy (for
    # example MobiKV and its ESP variant) remain distinct.
    parts = path.parts
    for index, part in enumerate(parts):
        if part.startswith("seed_") and index > 0:
            return parts[index - 1]
    policy = payload.get("args", {}).get("policy")
    return str(policy or path.stem.rsplit("_s", 1)[0])


def model_name(path: Path, payload: dict) -> str:
    parts = path.parts
    for index, part in enumerate(parts):
        if part.startswith("seed_") and index > 1:
            return parts[index - 2]
    return str(payload.get("args", {}).get("model", "unknown"))


def request_rows(payload: dict, path: Path) -> list[dict]:
    args = payload.get("args", {})
    model = model_name(path, payload)
    method = method_name(path, payload)
    seed = args.get("seed")
    rows = []
    for row in payload.get("per_request", []):
        if len(row) < 10:
            continue
        record = {
            "source": str(path),
            "model": model,
            "method": method,
            "seed": seed,
            "request_id": row[0],
            "session_id": row[1],
            "turn": row[2],
            "evaluated": bool(row[3]),
            "ttft_s": row[4],
            "total_s": row[5],
            "plan": row[6],
            "handoff": bool(row[7]),
            "cached_tokens": row[8],
            "remote": bool(row[9]),
            "tpot_b_s": row[10] if len(row) > 10 else None,
            "tpot_s_s": row[11] if len(row) > 11 else None,
            "output_tokens": row[12] if len(row) > 12 else None,
        }
        rows.append(record)
    return rows


def metric_values(rows: list[dict], metric: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(metric)
        if row["evaluated"] and row["turn"] >= 2 and value is not None:
            values.append(float(value))
    return values


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    campaign_dir = args.campaign_dir
    out_dir = args.out or campaign_dir / "aggregate"
    output_files = {out_dir.resolve()}
    inputs = []
    for path in sorted(campaign_dir.rglob("*.json")):
        if path.resolve() in output_files or path.name in {"manifest.json"}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("per_request"), list):
            inputs.append((path, payload))
    if not inputs:
        raise SystemExit(f"no simulator JSON with per_request rows under {campaign_dir}")

    all_rows = []
    seed_rows = []
    resource_rows = []
    for path, payload in inputs:
        rows = request_rows(payload, path)
        all_rows.extend(rows)
        model = model_name(path, payload)
        method = method_name(path, payload)
        seed = payload.get("args", {}).get("seed")
        for metric in ("ttft_s", "tpot_b_s", "tpot_s_s"):
            values = metric_values(rows, metric)
            for probability in PERCENTILES:
                seed_rows.append(
                    {
                        "model": model,
                        "method": method,
                        "seed": seed,
                        "metric": metric,
                        "percentile": f"P{int(probability * 100)}",
                        "estimate": quantile(values, probability),
                        "n_req": len(values),
                        "n_invalid": sum(
                            1
                            for row in rows
                            if row["evaluated"] and row["turn"] >= 2 and row.get(metric) is None
                        ),
                    }
                )
        resource = payload.get("gpu_util_fg", {})
        resource_rows.append(
            {
                "model": model,
                "method": method,
                "seed": seed,
                "gpu_util_mean": resource.get("mean"),
                "gpu_util_p90": resource.get("p90"),
                "gpu_util_max": resource.get("max"),
                "gpu_bg_s": payload.get("gpu_bg_s"),
                "egress_bg_s": payload.get("egr_bg_s"),
                "foreground_delay_from_background_s": payload.get("fg_delay_by_bg_s"),
                "wall_s": payload.get("wall_s"),
                "stats": json.dumps(payload.get("stats", {}), sort_keys=True),
            }
        )

    aggregate_rows = []
    groups = {}
    for row in seed_rows:
        if row["estimate"] is None:
            continue
        key = (row["model"], row["method"], row["metric"], row["percentile"])
        groups.setdefault(key, []).append(float(row["estimate"]))
    for (model, method, metric, percentile), values in sorted(groups.items()):
        aggregate_rows.append(
            {
                "model": model,
                "method": method,
                "metric": metric,
                "percentile": percentile,
                "mean": mean(values),
                "std": sample_std(values),
                "n_seed": len(values),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / "percentiles_by_seed.csv",
        seed_rows,
        ["model", "method", "seed", "metric", "percentile", "estimate", "n_req", "n_invalid"],
    )
    write_csv(
        out_dir / "aggregate.csv",
        aggregate_rows,
        ["model", "method", "metric", "percentile", "mean", "std", "n_seed"],
    )
    write_csv(
        out_dir / "resource_metrics.csv",
        resource_rows,
        [
            "model",
            "method",
            "seed",
            "gpu_util_mean",
            "gpu_util_p90",
            "gpu_util_max",
            "gpu_bg_s",
            "egress_bg_s",
            "foreground_delay_from_background_s",
            "wall_s",
            "stats",
        ],
    )
    with (out_dir / "request_metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    manifest = {
        "input_files": [str(path) for path, _ in inputs],
        "n_input_files": len(inputs),
        "n_request_rows": len(all_rows),
        "aggregation": "request percentiles per seed, then mean and sample std across seeds",
        "percentiles": [50, 90, 95, 99],
        "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path, _ in inputs},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
