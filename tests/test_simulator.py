"""CPU-only tests for the released simulator."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_mobikv  # noqa: E402
import smoke_test  # noqa: E402


def test_package_imports():
    for name in ("mobikv", "mobikv.simulator", "mobikv.analysis.aggregate_campaign"):
        importlib.import_module(name)


def test_presets_extend_the_base_policy():
    base = run_mobikv.PRESETS["mobikv"]
    for name, flags in run_mobikv.PRESETS.items():
        assert flags[: len(base)] == base, name
    assert "--bg_dfb" in run_mobikv.PRESETS["mobikv-df-rpp"]
    assert "--pp" in run_mobikv.PRESETS["mobikv-df-rpp"]


@pytest.mark.parametrize("preset", sorted(run_mobikv.PRESETS))
def test_preset_runs_on_synthetic_trace(tmp_path, preset):
    trace, trips, out = tmp_path / "trace.jsonl", tmp_path / "trips.npz", tmp_path / "out.json"
    smoke_test.make_trace(trace)
    smoke_test.make_trips(trips)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_mobikv.py"),
        "--preset", preset,
        "--trace", str(trace),
        "--trips", str(trips),
        "--costs", str(ROOT / "configs" / "costs_v1.json"),
        "--out", str(out),
        "--seed", "1",
        "--eval_start", "1",
        "--tpot", "1",
        "--side_cells", "3",
        "--mem_gb", "64",
    ]  # fmt: skip
    subprocess.run(cmd, cwd=ROOT, check=True)
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["cont"]["n"] >= 1
    assert result["cont"]["ttft_p95"] is not None
