"""CPU checks of frozen campaign settings and the archived simulator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "scripts"))

from mobikv import paper_config as paper  # noqa: E402
from mobikv.simulator import Cost, load_sessions  # noqa: E402
import run_campaign  # noqa: E402
import smoke_test  # noqa: E402


def test_frozen_final_controller_and_baselines():
    ours = paper.METHODS["mobikv"]
    assert (ours["route_first"], ours["route_all"], ours["bg_mode_set"], ours["route_dec_w"], ours["pp_esp"]) == ("all", 1, "kv", 0.2, 1)
    assert (ours["pp_ctrl"], ours["pp_V"], ours["pp_eps"], ours["pp_min_rel"]) == ("dpp", 30, 0.05, 0.25)
    assert paper.METHODS["wang_ton19_mdp"]["mdp_k"] == 3
    assert paper.METHODS["ouyang_jsac18"]["oy_hops"] == 3
    assert paper.METHODS["loongserve_sosp24"]["route_all"] == 1
    assert len(paper.METHODS) == 7 and len(paper.ABLATIONS) == 6


def test_measured_cost_entry_hashes_and_state_models():
    costs = paper.load_costs()
    provenance = json.loads((ROOT / "configs/provenance.json").read_text())
    digest = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]
    actual = {k: digest(v) for k, v in costs["compute"].items()}
    actual.update({"kv:" + k: digest(v) for k, v in costs["kv_bytes_per_token"].items()})
    actual.update({"sm:" + k: digest(v) for k, v in costs["state_model"].items()})
    actual["transfer"] = digest(costs["transfer"])
    assert actual == provenance["entry_hashes"]
    assert len(costs["compute"]) == 16
    assert set(costs["model_contexts"]) == set(paper.MODELS)
    for model in paper.MODELS:
        cost = Cost(costs, model, 100, 10000)
        assert cost.nbytes(8192) == paper.state_bytes(costs, model, 8192)
    assert paper.operating_flags(costs, "Qwen3.8-27B", "full")["load"] == 0.4757
    assert paper.gamma1_cap(costs, "Qwen3-8B", 100) / 10000 == pytest.approx(1.228964670745373)
    assert all(not {"host", "model_path", "mps_pipe"} & set(meta) for meta in costs["meta"].values())


def campaign_args(tmp_path, campaign, *extra):
    return run_campaign.parser().parse_args([
        "--campaign", campaign, "--trace", "trace.jsonl", "--trips", "trips.npz",
        "--porto-trips", "porto.npz", "--out-root", str(tmp_path / "runs"), *extra,
    ])


@pytest.mark.parametrize(("campaign", "count"), [("table", 1120), ("rho", 2520), ("stress", 1610), ("ablation", 420)])
def test_complete_campaign_jobs(tmp_path, campaign, count):
    jobs = run_campaign.make_jobs(campaign_args(tmp_path, campaign))
    assert len(jobs) == count
    assert len({job.out for job in jobs}) == count
    assert not (tmp_path / "runs").exists()
    parsed = [dict(zip(job.args[::2], job.args[1::2], strict=True)) for job in jobs]
    assert {int(p["--seed"]) for p in parsed} == set(range(41, 51 if campaign == "stress" else 61))
    assert all(p["--area_all"] == "1" and p["--tpot"] == "1" for p in parsed)
    if campaign == "rho":
        caps = {float(p["--cap"]) for p in parsed}
        assert len(caps) == 18
        assert max(caps) / min(caps) == pytest.approx(40)
    if campaign == "stress":
        assert {p["--trips"] for p in parsed} == {"trips.npz", "porto.npz"}


def test_dry_run_prints_without_touching_input_or_output(tmp_path):
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_campaign.py"), "--campaign", "table",
        "--trace", "not-downloaded.jsonl", "--trips", "not-downloaded.npz",
        "--models", "Qwen3-8B", "--methods", "mobikv", "--seeds", "41",
        "--out-root", str(tmp_path / "runs"), "--dry-run",
    ], capture_output=True, text=True, check=True)
    assert "--route_dec_w 0.2" in result.stdout
    assert "1 jobs planned" in result.stderr
    assert not (tmp_path / "runs").exists()


@pytest.fixture(scope="module")
def simulated_methods(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("final-simulator")
    trace, trips = tmp / "trace.jsonl", tmp / "trips.npz"
    smoke_test.make_trace(trace)
    smoke_test.make_trips(trips)
    requests, sessions, _ = load_sessions(trace)
    assert sessions == 2 and [r["turn"] for r in requests] == [1, 2, 1, 2]
    results = {}
    for method in paper.ALL_METHODS:
        out = tmp / (method + ".json")
        command = [sys.executable, str(ROOT / "scripts/run_mobikv.py"), "--method", method,
                   "--trace", str(trace), "--trips", str(trips), "--out", str(out),
                   "--seed", "41", "--side_cells", "3"]
        subprocess.run(command, capture_output=True, text=True, check=True)
        results[method] = json.loads(out.read_text())
    repeat = tmp / "repeat.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/run_mobikv.py"), "--trace", str(trace),
                    "--trips", str(trips), "--out", str(repeat), "--seed", "41", "--side_cells", "3"],
                   capture_output=True, text=True, check=True)
    return results, json.loads(repeat.read_text())


def test_all_final_methods_execute_and_seed_repeats(simulated_methods):
    results, repeated = simulated_methods
    for result in results.values():
        followups = [r for r in result["per_request"] if r[3] and r[2] >= 2]
        assert len(followups) == 1
        assert all(np.isfinite(r[4]) and r[4] > 0 and np.isfinite(r[11]) and r[11] > 0 for r in followups)
        assert result["args"]["eval_start"] == 1800
    assert results["mobikv"]["per_request"] == repeated["per_request"]
    assert results["mobikv"]["stats"] == repeated["stats"]


def test_ablations_switch_the_intended_mechanism(simulated_methods):
    results, _ = simulated_methods
    assert results["no_parallel_prefill"]["args"]["pp"] == 0
    assert results["no_parallel_prefill"]["stats"].get("pp_split", 0) == 0
    assert results["no_replica"]["args"]["policy"] == "reactive"
    assert results["hybrid_refresh"]["args"]["bg_mode"] == "hybrid"
    assert results["nearby_execution"]["args"]["route_all"] == 0
    assert results["no_decode_aware"]["args"]["route_dec_w"] == 0
    assert results["no_cold_helper"]["args"]["pp_esp"] == 0
    assert results["mobikv"]["stats"].get("pp_split", 0) > 0


def test_campaign_does_not_skip_incompatible_output(tmp_path):
    job = run_campaign.make_jobs(campaign_args(tmp_path, "table", "--models", "Qwen3-8B", "--methods", "mobikv", "--seeds", "41"))[0]
    job.out.parent.mkdir(parents=True)
    job.out.write_text("{}")
    with pytest.raises(ValueError, match="incomplete or has different arguments"):
        run_campaign.run_job(job)
