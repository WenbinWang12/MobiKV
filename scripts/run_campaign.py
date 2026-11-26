#!/usr/bin/env python3
"""Generate or run the final table, gamma sweep, stress, and ablation campaigns."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mobikv.paper_config import (  # noqa: E402
    ABLATIONS, DEFAULT_COSTS, GAMMAS, METHODS, MODELS, OPERATING_POINTS, SEEDS,
    STRESS, STRESS_SEEDS, gamma1_cap, load_costs, simulator_args,
)


@dataclass(frozen=True)
class Job:
    out: Path
    args: list[str]

    @property
    def command(self) -> list[str]:
        return [sys.executable, "-B", str(ROOT / "mobikv" / "simulator.py"), *self.args]


def csv_values(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def make_jobs(args: argparse.Namespace) -> list[Job]:
    costs = load_costs(args.costs)
    models = csv_values(args.models)
    seeds = [int(v) for v in csv_values(args.seeds)] if args.seeds else list(STRESS_SEEDS if args.campaign == "stress" else SEEDS)
    methods = csv_values(args.methods) or (["mobikv", *ABLATIONS] if args.campaign == "ablation" else list(METHODS))
    allowed = {"mobikv", *ABLATIONS} if args.campaign == "ablation" else set(METHODS)
    if any(m not in allowed for m in methods):
        raise ValueError(f"Methods for {args.campaign} must be selected from {sorted(allowed)}")
    if len(set(seeds)) != len(seeds) or len(set(methods)) != len(methods):
        raise ValueError("Duplicate seeds or methods would overwrite outputs")
    if not seeds or not methods:
        raise ValueError("At least one seed and method are required")
    if models and (len(set(models)) != len(models) or any(m not in MODELS for m in models)):
        raise ValueError(f"Models must be unique selections from {MODELS}")
    jobs = []

    def add(group: str, cell: str, model: str, op: str, overrides: dict | None = None, trips: Path | None = None) -> None:
        for method in methods:
            for seed in seeds:
                out = args.out_root / group / cell / method / f"seed_{seed}.json"
                flags = simulator_args(
                    trace=args.trace, trips=trips or args.trips, costs_path=args.costs,
                    model=model, op=op, method=method, seed=seed, out=out,
                    overrides=overrides, costs=costs,
                )
                jobs.append(Job(out, flags))

    if args.campaign == "table":
        op = args.op or "full"
        for model in models or MODELS:
            add(f"table_{op}", model, model, op)
    elif args.campaign in ("rho", "stress"):
        if args.op not in (None, "full") or models not in (None, ["Qwen3-8B"]):
            raise ValueError("The final rho and stress campaigns use Qwen3-8B at the full-speed point")
        if args.campaign == "rho":
            for gamma in GAMMAS:
                cap = float(f"{gamma1_cap(costs, 'Qwen3-8B', 100) / gamma:.6f}")
                add("rho_full_Qwen3-8B", f"gamma_{gamma:g}", "Qwen3-8B", "full", {"cap": cap})
        else:
            scenarios = csv_values(args.scenarios) or list(STRESS)
            if len(set(scenarios)) != len(scenarios) or any(s not in STRESS for s in scenarios):
                raise ValueError(f"Scenarios must be unique selections from {list(STRESS)}")
            if "porto" in scenarios and not args.porto_trips:
                raise ValueError("The Porto scenario requires --porto-trips; alternatively select --scenarios explicitly")
            for scenario in scenarios:
                add("stress_full_Qwen3-8B", scenario, "Qwen3-8B", "full", STRESS[scenario], args.porto_trips if scenario == "porto" else None)
    else:
        cells = [("full", "Qwen2.5-7B"), ("full", "Qwen3-8B"), ("slow", "Qwen2.5-7B")]
        if args.op or models:
            cells = [(args.op or "full", model) for model in models or ["Qwen3-8B"]]
        for op, model in cells:
            add(f"ablation_{op}", model, model, op)
    return jobs


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", required=True, choices=("table", "rho", "stress", "ablation"))
    p.add_argument("--trace", required=True, type=Path)
    p.add_argument("--trips", required=True, type=Path)
    p.add_argument("--porto-trips", type=Path)
    p.add_argument("--costs", type=Path, default=DEFAULT_COSTS)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--op", choices=OPERATING_POINTS)
    p.add_argument("--models", help="Comma-separated model names")
    p.add_argument("--methods", help="Comma-separated final method or ablation IDs")
    p.add_argument("--seeds", help="Comma-separated seeds; default 41-60, or 41-50 for stress")
    p.add_argument("--scenarios", help="Comma-separated stress scenarios; default all 23")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true", help="Print commands without creating output files")
    return p


def run_job(job: Job) -> None:
    if job.out.exists():
        try:
            result = json.loads(job.out.read_text(encoding="utf-8"))
            expected = dict(zip(job.args[::2], job.args[1::2], strict=True))
            def same(actual: object, expected_value: str) -> bool:
                if str(actual) == expected_value:
                    return True
                try:
                    return float(actual) == float(expected_value)
                except (TypeError, ValueError):
                    return False

            if "per_request" in result and all(same(result.get("args", {}).get(k[2:]), v) for k, v in expected.items()):
                return
        except (ValueError, OSError):
            pass
        raise ValueError(f"Existing output is incomplete or has different arguments: {job.out}")
    job.out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    with job.out.with_suffix(".log").open("w", encoding="utf-8") as log:
        subprocess.run(job.command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)


def main() -> None:
    p = parser()
    args = p.parse_args()
    try:
        if args.workers < 1:
            raise ValueError("--workers must be positive")
        jobs = make_jobs(args)
        if args.dry_run:
            for job in jobs:
                print(subprocess.list2cmdline(job.command) if os.name == "nt" else shlex.join(job.command))
        else:
            inputs = set()
            for job in jobs:
                flags = dict(zip(job.args[::2], job.args[1::2], strict=True))
                inputs.update(flags[k] for k in ("--trace", "--trips", "--costs"))
            missing = sorted(path for path in inputs if not Path(path).is_file())
            if missing:
                raise ValueError(f"Input files do not exist: {missing}")
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(run_job, jobs))
        print(f"{len(jobs)} jobs {'planned' if args.dry_run else 'completed'}", file=sys.stderr)
    except ValueError as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
