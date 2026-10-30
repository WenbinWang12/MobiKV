"""Deterministic CPU-only E2 protocol expansion. No launcher or stage acceptance."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random

DEFAULT_DESIGN = Path(__file__).resolve().parents[1] / "e2_full_design.json"
ACTIONS = ("continue_repair", "stop_migration_and_remote", "complete_fetch_then_local", "recompute_full_prefix_keep_delta")
POLICIES = ("initial_E1_nominal_rule", "training_tuned_state_blind_static", "same_information_greedy", "state_aware_guarded", "ablate_committed_progress", "ablate_remaining_horizon_update")
CONTINUOUS_METHODS = ("sticky_initial_E1_nominal_rule", "sticky_training_tuned_static", "same_information_greedy", "state_aware_guarded", "ablate_committed_progress", "ablate_remaining_horizon_update", "ablate_guard_hysteresis_only")
AXES = ("P", "S", "G", "D", "F", "cap_mbps_decimal", "added_RTT_ms", "load_location")
IDENTITY_FIELDS = ("trial_id", "phase", "sample_role", "input_id", "input_family_id", "seed", "block_id", "cell_id", "method", "module")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_design(source=DEFAULT_DESIGN):
    return copy.deepcopy(source) if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))


def _materialize(d):
    require(d["stage"] == "E2" and d["executable_plan"] is False and d["GPU_execution_authorized_by_this_file"] is False, "not an E2 CPU design")
    require(tuple(d["fixed_continuations"]["actions"]) == ACTIONS, "physical actions changed")
    require(tuple(d["models_and_executed_policy_comparisons"]["policies"]) == POLICIES, "policies changed")
    require(d["cohort"]["layers"] == 48 and d["request_and_information_contract"]["public_output_cap"] == 128, "geometry changed")
    grid = d["conditional_grid"]
    require(tuple(grid["axis_order"]) == AXES, "axis order changed")
    groups = {name: set(itertools.product(*(grid[name][axis] for axis in AXES))) for name in ("main_factorial", "checkpoint_age_sensitivity")}
    expected_main = set(itertools.product([1024, 4096], [16], [32, 64], [8], [12, 47], [100, 1000], [0], ["none"]))
    expected_age = set(itertools.product([1024], [16], [32, 64], [4, 16], [12, 47], [1000], [0], ["none"]))
    require(groups["main_factorial"] == expected_main and groups["checkpoint_age_sensitivity"] == expected_age, "reviewed conditional geometry changed")
    union = expected_main | expected_age
    cells = [dict(zip(AXES, values), cell_id=f"c{i:03d}", module="conditional") for i, values in enumerate(sorted(union))]
    by_tuple = {tuple(c[a] for a in AXES): c for c in cells}
    splits = d["conditional_splits_and_counts"]
    confirmation = d["predeclared_primary_counterexample"]["confirmation"]
    split_specs = {name: splits[name] for name in ("training", "calibration", "test")}
    split_specs["confirmation"] = confirmation
    all_ids, all_seeds = set(), set()
    for phase, n in (("training", 4), ("calibration", 4), ("test", 8), ("confirmation", 4)):
        spec = split_specs[phase]
        ids, seeds = spec["input_ids"], spec["seeds"]
        require(len(ids) == len(seeds) == len(set(ids)) == len(set(seeds)) == n, "invalid split size/duplicates")
        require(all(isinstance(v, str) and v for v in ids) and all(type(v) is int for v in seeds), "invalid input/seed")
        require(not all_ids.intersection(ids) and not all_seeds.intersection(seeds), "split leakage")
        all_ids.update(ids); all_seeds.update(seeds)
    calibration = set(map(tuple, splits["calibration"]["cells"]))
    primary = set(map(tuple, d["predeclared_primary_counterexample"]["states"]))
    require(len(calibration) == 8 and calibration <= expected_main and len(primary) == 2 and primary <= union, "invalid calibration/primary cells")
    phase_cells = {"training": union, "calibration": calibration, "test": union, "confirmation": primary}
    continuous = d["continuous_policy_module"]
    require(tuple(continuous["methods"]) == CONTINUOUS_METHODS and continuous["checkpoints_generated"] == [4, 16], "continuous methods/checkpoints changed")
    require(len(continuous["conditions"]) == 4 and continuous["generated_tokens_executor_only"] == 64, "continuous geometry changed")
    for i, condition in enumerate(continuous["conditions"]):
        cell = {a: condition.get(a) for a in AXES}
        cell.update(cell_id=f"u{i:03d}", module="continuous", G=64, D=4, F=24, condition_id=condition["id"], rho=condition.get("rho", 0))
        cells.append(cell)
    require([(c["P"], c["cap_mbps_decimal"], c["load_location"]) for c in cells[-4:]] == [(1024,1000,"none"),(1024,1000,"holder"),(1024,1000,"destination"),(4096,100,"none")], "continuous treatments changed")
    rng = random.Random(splits["ordering_seed"])
    trials, registry, phase_order = [], [], []

    def make_trial(phase, cell, member, method, position):
        trial_id = f"{phase}.{cell['cell_id']}.{member['block_id']}.{method}"
        geometry = {"prefix_tokens": cell["P"], "suffix_tokens": cell["S"], "public_output_cap": 128, "checkpoint_generated": cell["D"], "initial_committed_layers": cell["F"], "total_layers": 48, "prefix_KV_bytes": cell["P"] * 196608, "initial_committed_KV_bytes": cell["P"] * 4096 * cell["F"], "application_cap_bps": cell["cap_mbps_decimal"] * 1_000_000}
        return {
            "trial_id": trial_id, "phase": phase, "module": cell["module"], "sample_role": member["sample_role"],
            **{k: member[k] for k in ("input_id", "input_family_id", "seed", "block_id")},
            "cell_id": cell["cell_id"], "method": method, "method_order_position": position,
            "public_geometry": geometry,
            "executor_only": {"realized_stop_tokens": cell["G"], "EOS_stopping": False},
            "treatment": {"load_location": cell["load_location"], "rho": cell.get("rho", 0), "added_RTT_ms": cell["added_RTT_ms"]},
            "checkpoints_generated": [cell["D"]] if cell["module"] == "conditional" else [4,16],
            "prefix_key": f"{member['input_family_id']}:seed{member['seed']}:P{cell['P']}",
            "matched_group_id": f"{phase}.{cell['cell_id']}.{member['block_id']}",
            "load_trace_key": digest({"phase": phase, "cell_id": cell["cell_id"], "seed": member["seed"], "block_id": member["block_id"]}),
            "diagnostic_id": trial_id + ".exact-route-diagnostic", "reference_id": trial_id + ".same-shape-HF-reference",
            "token_arrays_frozen": False,
        }

    for phase_i, phase in enumerate(("training", "calibration", "test", "confirmation", "continuous")):
        spec = split_specs["test" if phase == "continuous" else phase]
        members = [dict(input_id=v, input_family_id=v, seed=s, block_id=f"b{i:02d}", phase=phase, sample_role="measured") for i,(v,s) in enumerate(zip(spec["input_ids"], spec["seeds"]))]
        registry.extend(members)
        warmup = dict(input_id=f"e2-full-v1-{phase}-warmup-excluded", input_family_id=f"e2-full-v1-{phase}-warmup-excluded", seed=2026111700+phase_i, block_id="warmup", phase=phase, sample_role="warmup")
        require(warmup["input_id"] not in all_ids and warmup["seed"] not in all_seeds, "warmup leakage")
        registry.append(warmup)
        selected_cells = cells[-4:] if phase == "continuous" else [by_tuple[v] for v in sorted(phase_cells[phase])]
        methods = list(CONTINUOUS_METHODS if phase == "continuous" else ACTIONS + (POLICIES if phase == "test" else ()))
        ordered = selected_cells[:]; rng.shuffle(ordered)
        start = len(trials)
        for cell in ordered:
            base = methods[:]; rng.shuffle(base)
            warm_order = base[:]; rng.shuffle(warm_order)
            trials.extend(make_trial(phase,cell,warmup,m,p) for p,m in enumerate(warm_order))
            block_order = members[:]; rng.shuffle(block_order)
            for member in block_order:
                shift = int(member["block_id"][1:]) % len(base)
                method_order = base[shift:] + base[:shift]
                trials.extend(make_trial(phase,cell,member,m,p) for p,m in enumerate(method_order))
        phase_order.append({"phase": phase, "start_inclusive": start, "stop_exclusive": len(trials), "prerequisite": "full_E1_and_new_E2_smoke" if phase == "training" else ("E2_training_fit_frozen" if phase == "calibration" else "E2_fit_and_calibration_frozen")})
    profiles = []
    for values in d["separate_profile_plan"]["conditions"]:
        cell = by_tuple[tuple(values)]
        for method in ACTIONS:
            profiles.append({"profile_id": f"profile.{cell['cell_id']}.{method}", "cell_id": cell["cell_id"], "method": method, "input_id": "e2-full-v1-profile-excluded", "seed": 2026111800, "sample_role": "instrumented_excluded"})
    measured = [t for t in trials if t["sample_role"] == "measured"]
    counts = {f"{phase}_measured": sum(t["phase"] == phase for t in measured) for phase in split_specs}
    counts.update(continuous_measured=sum(t["phase"] == "continuous" for t in measured), measured=len(measured), warmups=len(trials)-len(measured), candidates=len(trials), diagnostics=len(trials), HF_references=len(trials), instrumented_profiles=len(profiles), conditional_cells=24, continuous_cells=4)
    require(counts == dict(training_measured=384, calibration_measured=128, test_measured=1920, confirmation_measured=32, continuous_measured=224, measured=2688, warmups=404, candidates=3092, diagnostics=3092, HF_references=3092, instrumented_profiles=12, conditional_cells=24, continuous_cells=4), "planned counts changed")
    result = dict(schema_version=1, protocol="E2_full_design_materialization", stage="E2", design_id=d["design_id"], design_sha256=digest(d), design=d, status="unfrozen_CPU_preparation", executable_plan=False, GPU_execution_authorized=False, E2_complete=False, paper_stage_complete=False, cells=cells, trials=trials, profiles=profiles, input_registry=registry, phase_order=phase_order, counts=counts, resource_estimated_GPU_hours=None,
        randomization="Seeded MT19937 shuffled cells/blocks plus cyclic method rotations; serialized order is authoritative",
        pending_freeze_requirements=["reviewed_E1_full", "new_E2_smoke", "source_model_physical_identity", "token_arrays_and_hashes", "real_holder_delta_witness", "live_repeated_checkpoint_and_replay", "timing_and_socket_counter_boundaries", "finite_watchdogs", "E1_fit_acceptance", "E2_fit_and_calibration_before_test"])
    result["plan_sha256"] = digest(result)
    return result


def build_plan(design=DEFAULT_DESIGN):
    return _materialize(load_design(design))


def validate_plan(plan):
    require(canonical_bytes(plan) == canonical_bytes(_materialize(copy.deepcopy(plan["design"]))), "plan differs from deterministic design expansion")


def trial_index(plan, phase=None, role="measured"):
    return {t["trial_id"]: t for t in plan["trials"] if t["sample_role"] == role and (phase is None or t["phase"] == phase)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_plan(args.design); validate_plan(result)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False); stream.write("\n")
    print(json.dumps({"output":str(args.output.resolve()),"plan_sha256":result["plan_sha256"],"counts":result["counts"],"executable_plan":False}))


if __name__ == "__main__":
    main()
