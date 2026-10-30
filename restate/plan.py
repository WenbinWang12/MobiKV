"""CPU-only materialization of the reviewed E1 design; never launches work.

CLI: python -B plan.py --output ABSENT_FILE.json
The result is deliberately NOT an executable/frozen campaign. Token arrays,
runtime identity, E0/E1-smoke acceptance and treatment implementations must be
bound by a later, separately reviewed freeze operation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
from typing import Any


DEFAULT_DESIGN = Path(__file__).resolve().parents[1] / "e1_full_design.json"
FIXED_ACTIONS = ("recompute", "native_fetch", "holder", "remote", "repair")
POLICY = "fresh_nominal_recompute_fetch_threshold"
CONTROL = "native_hf_control"
PUBLIC_FEATURES = ("prefix_tokens", "prefix_KV_bytes", "application_cap_bps", "fixed_suffix_tokens")
IDENTITY_FIELDS = ("trial_id", "phase", "sample_role", "input_id", "input_family_id", "seed", "cell_id", "method")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_design(source: str | Path | dict = DEFAULT_DESIGN) -> dict:
    if isinstance(source, dict):
        return copy.deepcopy(source)
    return json.loads(Path(source).read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _cell_tuple(cell: dict, axes: list[str]) -> tuple:
    return tuple(cell[name] for name in axes)


def _validate_design(d: dict) -> None:
    _require(d["stage"] == "E1" and d["executable_plan"] is False, "not an E1 design-only document")
    _require(d["GPU_execution_authorized_by_this_file"] is False, "design must not authorize execution")
    _require(tuple(d["methods"]["fixed_actions"]) == FIXED_ACTIONS, "fixed actions changed")
    _require(d["methods"]["executed_policy_baseline"]["id"] == POLICY, "policy changed")
    _require(d["shared_request_semantics"]["public_output_cap"] == 128, "public cap changed")
    split = d["split_and_repetition_protocol"]
    seen_inputs, seen_seeds = set(), set()
    for phase, count in (("training", 2), ("calibration", 2), ("test", 8)):
        ids, seeds = split[phase]["input_ids"], split[phase]["seeds"]
        _require(len(ids) == len(seeds) == count, "split block count changed")
        _require(len(set(ids)) == count and len(set(seeds)) == count, "duplicate split member")
        _require(not seen_inputs.intersection(ids) and not seen_seeds.intersection(seeds), "cross-split input/seed leakage")
        _require(all(isinstance(x, str) and x for x in ids), "invalid input IDs")
        _require(all(type(x) is int for x in seeds), "invalid seeds")
        seen_inputs.update(ids)
        seen_seeds.update(seeds)
    _require(split["warmups"]["per_method_per_cell_per_phase"] == 1, "warmup count changed")


def _materialize(d: dict) -> dict:
    _validate_design(d)
    grid = d["full_grid"]
    axes = grid["axis_order"]
    expected_axes = ["prefix_tokens", "suffix_tokens", "realized_stop_tokens", "application_cap_mbps_decimal", "added_experiment_path_round_trip_delay_ms", "background_load_location"]
    _require(axes == expected_axes, "axis schema changed")
    section_sets = {
        name: set(itertools.product(*(grid[name][axis] for axis in axes)))
        for name in ("main_factorial", "delay_sensitivity", "load_location_sensitivity")
    }
    main, delay, load = (section_sets[n] for n in section_sets)
    expected_main = set(itertools.product([128, 1024, 4096], [16], [16, 64], [100, 400, 1000], [0], ["none"]))
    expected_delay = set(itertools.product([1024], [16], [16, 64], [100, 1000], [0, 2, 10], ["none"]))
    expected_load = set(itertools.product([1024], [16], [16, 64], [100, 400, 1000], [0], ["none", "holder", "destination"]))
    _require((main, delay, load) == (expected_main, expected_delay, expected_load), "reviewed 38-cell geometry changed")
    _require(grid["load_location_sensitivity"]["positive_nominal_offered_load_rho"] == 0.5, "load intensity changed")
    union = main | delay | load
    _require(len(union) == grid["unique_test_cells"] == 38, "grid count mismatch")
    cells = []
    for index, values in enumerate(sorted(union)):
        cell = dict(zip(axes, values))
        cell.update(cell_id=f"c{index:03d}", sections=[n for n, members in section_sets.items() if values in members])
        cell["nominal_offered_load_rho"] = 0.0 if cell["background_load_location"] == "none" else 0.5
        cells.append(cell)
    cell_by_tuple = {_cell_tuple(c, axes): c for c in cells}
    split = d["split_and_repetition_protocol"]
    calibration = {tuple(v) for v in split["calibration"]["cells_axis_order_as_above"]}
    _require(len(calibration) == 6 and calibration <= main, "calibration cells must be six main cells")
    phase_cells = {"training": main, "calibration": calibration, "test": union}
    rng = random.Random(split["ordering"]["seed"])
    trials, registry, phase_order = [], [], []
    occupied_seeds = {seed for phase in phase_cells for seed in split[phase]["seeds"]}

    def trial(phase: str, cell: dict, member: dict, method: str, role: str, position: int) -> dict:
        p = cell["prefix_tokens"]
        trial_id = f"{phase}.{cell['cell_id']}.{member['block_id']}.{method}"
        result = {
            "trial_id": trial_id, "phase": phase, "sample_role": role,
            "input_id": member["input_id"], "input_family_id": member["input_family_id"],
            "seed": member["seed"], "block_id": member["block_id"],
            "cell_id": cell["cell_id"], "method": method, "method_order_position": position,
            "public_features": {"prefix_tokens": p, "prefix_KV_bytes": grid["KV_prefix_payload_bytes"][str(p)], "application_cap_bps": cell["application_cap_mbps_decimal"] * 1_000_000, "fixed_suffix_tokens": cell["suffix_tokens"]},
            "public_output_cap": 128,
            "executor_only": {"realized_stop_tokens": cell["realized_stop_tokens"], "EOS_stopping": False},
            "treatment": {"added_path_delay_ms": cell["added_experiment_path_round_trip_delay_ms"], "load_location": cell["background_load_location"], "nominal_offered_load_rho": cell["nominal_offered_load_rho"]},
            "prefix_key": f"{member['input_family_id']}:seed{member['seed']}:P{p}",
            "diagnostic_id": trial_id + ".exact-route-diagnostic",
            "reference_id": trial_id + ".same-shape-HF-reference",
            "token_arrays_frozen": False,
        }
        # Method identity does not change cache/input identity or load arrivals.
        result["load_trace_key"] = digest({"seed": d["compute_contention_treatment"]["arrival_seed"], "phase": phase, "cell_id": cell["cell_id"], "block_id": member["block_id"]})
        return result

    for phase_index, phase in enumerate(("training", "calibration", "test")):
        methods = list(FIXED_ACTIONS) + ([POLICY] if phase == "test" else []) + [CONTROL]
        members = [dict(input_id=input_id, input_family_id=input_id, seed=seed, block_id=f"b{i:02d}", phase=phase, sample_role="measured") for i, (input_id, seed) in enumerate(zip(split[phase]["input_ids"], split[phase]["seeds"]))]
        registry.extend(members)
        warmup_id = f"{d['design_id']}-{phase}-warmup-excluded"
        warmup_seed = 2026110800 + phase_index
        _require(warmup_seed not in occupied_seeds, "warmup seed leaks into a measured split")
        _require(warmup_id not in {input_id for name in phase_cells for input_id in split[name]["input_ids"]}, "warmup family leaks into a measured split")
        warmup = dict(input_id=warmup_id, input_family_id=warmup_id, seed=warmup_seed, block_id="warmup", phase=phase, sample_role="warmup")
        registry.append(warmup)
        order = sorted(phase_cells[phase])
        rng.shuffle(order)
        start = len(trials)
        for values in order:
            cell = cell_by_tuple[values]
            base = methods[:]
            rng.shuffle(base)
            warmup_order = base[:]
            rng.shuffle(warmup_order)
            trials.extend(trial(phase, cell, warmup, method, "warmup", pos) for pos, method in enumerate(warmup_order))
            # Cyclic Latin rotations balance each method's position to within one.
            blocks = members[:]
            rng.shuffle(blocks)
            for member in blocks:
                shift = int(member["block_id"][1:]) % len(base)
                method_order = base[shift:] + base[:shift]
                trials.extend(trial(phase, cell, member, method, "measured", pos) for pos, method in enumerate(method_order))
        phase_order.append({"phase": phase, "start_inclusive": start, "stop_exclusive": len(trials), "prerequisite": "full_E0_and_new_E1_smoke" if phase == "training" else ("training_fit_frozen" if phase == "calibration" else "calibration_and_coefficients_frozen")})

    profiles = []
    profile = d["instrumented_breakdown_plan"]
    _require(profile["input_id"] not in {m["input_id"] for m in registry} and profile["seed"] not in {m["seed"] for m in registry}, "profile input/seed leaks into main records")
    for values in profile["cells_axis_order_as_above"]:
        cell = cell_by_tuple[tuple(values)]
        for method in profile["methods"]:
            profiles.append({"profile_id": f"profile.{cell['cell_id']}.{method}", "cell_id": cell["cell_id"], "method": method, "input_id": profile["input_id"], "seed": profile["seed"], "sample_role": "instrumented_excluded", "not_a_performance_repetition": True})
    measured = [t for t in trials if t["sample_role"] == "measured"]
    counts = {"unique_cells": len(cells), "training_measured": sum(t["phase"] == "training" for t in measured), "calibration_measured": sum(t["phase"] == "calibration" for t in measured), "test_measured": sum(t["phase"] == "test" for t in measured), "measured": len(measured), "warmups": len(trials) - len(measured), "candidates": len(trials), "diagnostics": len(trials), "HF_references": len(trials), "instrumented_profiles": len(profiles), "excluded_background_calibration_requests": 24}
    _require(counts == dict(unique_cells=38, training_measured=216, calibration_measured=72, test_measured=2128, measured=2416, warmups=410, candidates=2826, diagnostics=2826, HF_references=2826, instrumented_profiles=30, excluded_background_calibration_requests=24), "planned counts changed")
    result = {
        "schema_version": 1, "protocol": "E1_full_design_materialization", "stage": "E1",
        "design_id": d["design_id"], "design_sha256": digest(d), "design": d,
        "status": "unfrozen_CPU_preparation", "executable_plan": False,
        "GPU_execution_authorized": False, "E1_complete": False, "paper_stage_complete": False,
        "randomization_algorithm": "Python random.Random seeded MT19937; shuffled cells/blocks plus cyclic Latin method rotations; serialized full order is authoritative",
        "ordering_seed": split["ordering"]["seed"], "cells": cells, "input_registry": registry,
        "phase_order": phase_order, "trials": trials, "profiles": profiles, "counts": counts,
        "pending_freeze_requirements": ["full_E0_acceptance", "new_E1_smoke_acceptance", "source_model_host_identity", "actual_token_arrays_and_hashes", "verified_delay_mechanism", "verified_holder_and_destination_contention", "complete_request_timer", "finite_runtime_limits", "training_fit_then_calibration_hashes_before_test"],
        "resource_estimated_GPU_hours": None,
    }
    result["plan_sha256"] = digest(result)
    return result


def build_plan(design: str | Path | dict = DEFAULT_DESIGN) -> dict:
    """Return a deterministic, non-executable preparation artifact."""
    return _materialize(load_design(design))


def validate_plan(plan: dict) -> None:
    """Reconstruct every trial from the bound design, not just reported counts."""
    expected = _materialize(copy.deepcopy(plan["design"]))
    _require(canonical_bytes(plan) == canonical_bytes(expected), "plan differs from deterministic design materialization")


def trial_index(plan: dict, phase: str | None = None, role: str = "measured") -> dict[str, dict]:
    return {t["trial_id"]: t for t in plan["trials"] if t["sample_role"] == role and (phase is None or t["phase"] == phase)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(args.design)
    validate_plan(plan)
    # Exclusive creation: preparation must not overwrite a previously reviewed plan.
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(plan, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "plan_sha256": plan["plan_sha256"], "counts": plan["counts"], "executable_plan": False}))


if __name__ == "__main__":
    main()
