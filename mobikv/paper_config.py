"""Frozen parameters of the final November 23 simulation campaign."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COSTS = ROOT / "configs" / "costs_a800.json"
MODELS = (
    "Qwen2.5-7B", "Qwen2.5-14B", "Qwen3-8B", "Llama-3.1-8B",
    "Mistral-7B-v0.3", "Gemma-2-9B-it", "Qwen3.5-9B", "Qwen3.8-27B",
)
OPERATING_POINTS = {"full": (100, 10000.0), "slow": (25, 930.0), "full1g": (100, 930.0)}
GAMMAS = (0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10)
SEEDS = tuple(range(41, 61))
STRESS_SEEDS = tuple(range(41, 51))

_ROUTING = {"route_first": "all", "route_all": 1, "cont_min": 0, "beta": 1.0}
_RPP = {
    "pp": 1, "pp_cold": 1, "pp_ctrl": "dpp", "pp_V": 30, "pp_eps": 0.05,
    "pp_min_rel": 0.25, "pp_qmin": 1024, "pp_grid": 32,
}
_REFRESH = {"bg_dfb": 1, "bg_mode_set": "kv"}
_FINAL = {"policy": "mobikv", **_ROUTING, **_RPP, **_REFRESH, "route_dec_w": 0.2, "pp_esp": 1}
METHODS = {
    "follow_me_cloud": {"policy": "follow_tx"},
    "wang_ton19_mdp": {"policy": "mdp_thr", "mdp_k": 3},
    "ouyang_jsac18": {"policy": "ouyang", "oy_V": 1, "oy_B": 0.1, "oy_hops": 3},
    "preble_iclr25": {"policy": "preble"},
    "mooncake_fast25": {"policy": "mooncake"},
    "loongserve_sosp24": {
        "policy": "reactive", "route_first": "all", "route_all": 1,
        "pp": 1, "pp_esp": 1, "pp_cold": 1, "pp_ctrl": "greedy",
        "pp_qmin": 1024, "pp_min_rel": 0.25,
    },
    "mobikv": _FINAL,
}
# These are the six single-element variants used by the final supplement.
# Omitting a flag restores the archived simulator's default for that mechanism.
ABLATIONS = {
    "no_parallel_prefill": {"policy": "mobikv", **_ROUTING, **_REFRESH, "route_dec_w": 0.2},
    "no_replica": {"policy": "reactive", **_ROUTING, **_RPP, "route_dec_w": 0.2, "pp_esp": 1},
    "hybrid_refresh": {k: v for k, v in _FINAL.items() if k != "bg_mode_set"},
    "nearby_execution": {**_FINAL, "route_first": "p2c", "route_all": 0},
    "no_decode_aware": {k: v for k, v in _FINAL.items() if k != "route_dec_w"},
    "no_cold_helper": {k: v for k, v in _FINAL.items() if k != "pp_esp"},
}
ALL_METHODS = {**METHODS, **ABLATIONS}
_BW = {"bw_var": 1, "bw_slow": 0.4, "bw_dwell_n": 1200, "bw_dwell_s": 600}
STRESS = {
    "static": {"speedup": 0}, "speed3": {"speedup": 3}, "speed10": {"speedup": 10},
    "porto": {}, "load0.5": {"load": 0.5}, "load1.5": {"load": 1.5}, "load2": {"load": 2.0},
    "mem16": {"mem_gb": 16}, "mem32": {"mem_gb": 32}, "mem128": {"mem_gb": 128},
    "rtt5": {"rtt_hop_ms": 5}, "rtt20": {"rtt_hop_ms": 20},
    "r300": {"r": 300}, "r1000": {"r": 1000},
    "err25": {"pred_err": 0.25}, "err50": {"pred_err": 0.5},
    "bwslow": {**_BW, "bw_lag": 30}, "bwblind": {**_BW, "bw_lag": -1},
    "comp2": {"kv_comp": 2}, "comp4": {"kv_comp": 4},
    "linkhet": {"link_het": 0.3, "link_het_mult": 0.35},
    "hfail10": {"helper_fail": 0.1}, "hfail30": {"helper_fail": 0.3},
}


def load_costs(path: str | Path = DEFAULT_COSTS) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def state_bytes(costs: dict, model: str, tokens: int) -> float:
    """Archived rule including sliding-window and recurrent state."""
    state = costs.get("state_model", {}).get(model, {})
    return (tokens * costs["kv_bytes_per_token"][model] + state.get("const", 0.0)
            + state.get("kappa_win", 0.0) * min(tokens, state.get("win", 0)))


def gamma1_cap(costs: dict, model: str, share: int) -> float:
    """Mb/s giving gamma_8k = 1, using the campaign's payload-time definition."""
    compute_s = dict(costs["compute"][f"{model}-g{share}"]["full"])[8192] / 1000
    return state_bytes(costs, model, 8192) * 8 / (1e6 * costs["transfer"]["eff"] * compute_s)


def operating_flags(costs: dict, model: str, op: str) -> dict:
    share, cap = OPERATING_POINTS[op]
    compute = costs["compute"]
    reference = dict(compute["Qwen2.5-7B-g25"]["full"])[8192]
    current = dict(compute[f"{model}-g{share}"]["full"])[8192]
    return {
        "model": model, "share": share, "nl": costs["meta"][f"{model}-g{share}"]["layers"],
        "load": float(f"{min(1.0, reference / current):.4f}"), "cap": cap,
        "area_all": 1, "mem_gb": 64, "tpot": 1,
    }


def simulator_args(
    *, trace: str | Path, trips: str | Path, costs_path: str | Path,
    model: str, op: str, method: str, seed: int, out: str | Path,
    overrides: dict | None = None, costs: dict | None = None,
) -> list[str]:
    flags = {
        "trace": str(trace), "trips": str(trips), "costs": str(costs_path),
        **operating_flags(costs or load_costs(costs_path), model, op),
        **ALL_METHODS[method], "seed": seed, **(overrides or {}), "out": str(out),
    }
    return [item for key, value in flags.items() for item in (f"--{key}", str(value))]
