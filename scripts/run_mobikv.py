#!/usr/bin/env python3
"""Run the simulator with one of the paper's named system presets.

All additional arguments are passed to :mod:`mobikv.simulator`, so a paper
scenario can be copied directly from a preregistration command line.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PRESETS = {
    "mobikv": [
        "--policy", "mobikv", "--route_first", "p2c", "--cont_min", "0", "--beta", "1",
    ],
    "mobikv-df": [
        "--policy", "mobikv", "--route_first", "p2c", "--cont_min", "0", "--beta", "1",
        "--bg_dfb", "1",
    ],
    "mobikv-rpp": [
        "--policy", "mobikv", "--route_first", "p2c", "--cont_min", "0", "--beta", "1",
        "--pp", "1", "--pp_cold", "1", "--pp_ctrl", "dpp", "--pp_V", "30",
        "--pp_eps", "0.05", "--pp_min_rel", "0.25", "--pp_qmin", "1024", "--pp_grid", "32",
    ],
    "mobikv-df-rpp": [
        "--policy", "mobikv", "--route_first", "p2c", "--cont_min", "0", "--beta", "1",
        "--pp", "1", "--pp_cold", "1", "--pp_ctrl", "dpp", "--pp_V", "30",
        "--pp_eps", "0.05", "--pp_min_rel", "0.25", "--pp_qmin", "1024", "--pp_grid", "32",
        "--bg_dfb", "1",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MobiKV's trace-driven simulator with a paper preset."
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="mobikv-df-rpp")
    args, rest = parser.parse_known_args()
    # simulator.main() owns validation and help for its full option set.
    from mobikv import simulator

    sys.argv = ["mobikv.simulator", *PRESETS[args.preset], *rest]
    simulator.main()


if __name__ == "__main__":
    main()
