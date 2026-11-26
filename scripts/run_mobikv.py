#!/usr/bin/env python3
"""Run one final-paper method with measured costs and the frozen operating point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mobikv.paper_config import ALL_METHODS, DEFAULT_COSTS, MODELS, OPERATING_POINTS, simulator_args  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=ALL_METHODS, default="mobikv")
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--trips", required=True, type=Path)
    parser.add_argument("--costs", default=DEFAULT_COSTS, type=Path)
    parser.add_argument("--model", choices=MODELS, default="Qwen3-8B")
    parser.add_argument("--op", choices=OPERATING_POINTS, default="full")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--out", required=True, type=Path)
    args, extra = parser.parse_known_args()
    from mobikv import simulator

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = ["mobikv.simulator", *simulator_args(
        trace=args.trace, trips=args.trips, costs_path=args.costs, model=args.model,
        op=args.op, method=args.method, seed=args.seed, out=args.out,
    ), *extra]
    simulator.main()


if __name__ == "__main__":
    main()
