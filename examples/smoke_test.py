#!/usr/bin/env python3
"""Small CPU-only smoke test for the released simulator."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def make_trace(path: Path) -> None:
    rows = []
    # Two short sessions with a continuation sharing a prefix.  The simulator
    # uses hash_ids to reconstruct the parent relation.
    for sid, start in enumerate((0, 2500)):
        h0 = [f"s{sid}-b{i}" for i in range(64)]
        rows.append({"timestamp": start * 1000, "input_length": 4096,
                     "output_length": 16, "hash_ids": h0})
        h1 = h0 + [f"s{sid}-new{i}" for i in range(1024)]
        rows.append({"timestamp": (start + 120) * 1000, "input_length": 5120,
                     "output_length": 16, "hash_ids": h1})
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def make_trips(path: Path) -> None:
    # A 10-second sampled path long enough for both train and evaluation pools.
    n = 120
    xy = np.stack([np.linspace(0, 3000, n), np.linspace(0, 1800, n)], axis=1).astype(np.float32)
    np.savez(path, trip_user=np.array([0, 1]), trip_mode=np.array(["walk", "walk"]),
             trip_len=np.array([n, n]), trip_off=np.array([0, n]),
             trip_test=np.array([False, True]), trip_vmed=np.array([1.0, 1.0]),
             trip_t0=np.array([0.0, 0.0]), xy=np.concatenate([xy, xy]))


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mobikv-smoke-") as tmp:
        tmp = Path(tmp)
        trace, trips, out = tmp / "trace.jsonl", tmp / "trips.npz", tmp / "out.json"
        make_trace(trace)
        make_trips(trips)
        cmd = [sys.executable, str(ROOT / "scripts" / "run_mobikv.py"),
               "--preset", "mobikv-df-rpp", "--trace", str(trace),
               "--trips", str(trips), "--costs", str(ROOT / "configs" / "costs_v1.json"),
               "--out", str(out), "--seed", "1", "--eval_start", "1", "--tpot", "1",
               "--side_cells", "3", "--mem_gb", "64"]
        subprocess.run(cmd, cwd=ROOT, check=True)
        result = json.loads(out.read_text(encoding="utf-8"))
        assert result["cont"]["n"] >= 1
        assert result["cont"]["ttft_p95"] is not None
        assert result["cont"]["tpot_s_p95"] is not None
        print("MobiKV smoke test passed")


if __name__ == "__main__":
    main()
