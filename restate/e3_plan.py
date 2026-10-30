"""Deterministic CPU E3 plan; no launcher, experiment or completion gate."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random

DEFAULT_DESIGN = Path(__file__).resolve().parents[1] / "e3_full_design.json"
METHODS = ("sticky_training_selected_admission", "holder_decode_static", "always_repair",
           "same_information_greedy", "state_aware_guarded", "ablate_observation_age_guard")
ONLINE = METHODS[3:]
FIXED_ACTIONS = ("repair", "remote", "native_fetch", "recompute", "holder")
PUBLIC_FIELDS = frozenset(("prefix_tokens", "suffix_tokens", "public_output_cap", "prefix_KV_bytes", "initial_application_cap_bps"))


def require(condition, message):
    if not condition: raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()).hexdigest()


def _design(source):
    d = copy.deepcopy(source) if isinstance(source,dict) else json.loads(Path(source).read_text(encoding="utf-8"))
    require(d["design_id"]=="restate-E3-full-v1" and d["stage"]=="E3" and
            d["executable_plan"] is False and d["GPU_execution_authorized_by_this_file"] is False,"not the reviewed CPU design")
    require(tuple(d["methods"])==METHODS and len(d["cells"])==24,"methods/cells changed")
    contract=d["request_and_control_contract"]
    require(contract["S"]==16 and contract["public_output_cap"]==128,"request geometry changed")
    require(contract["decision_limits"]=={"minimum_applied_switch_dwell_s":.25,"maximum_applied_policy_changes":8,
             "relative_improvement_margin":.05,"telemetry_max_age_s":.5},"control constants changed")
    expected={"constant_fast":[[0,1000]],"constant_slow":[[0,100]],"drop":[[0,1000],[.5,100]],
              "drop_recover":[[0,1000],[.5,100],[1.5,1000]],
              "alternating":[[0,1000],[.5,100],[1.5,1000],[2.5,100],[3.5,1000]]}
    require(d["perturbation_protocol"]["rate_traces_seconds_Mbps"]==expected,"reviewed rate traces changed")
    primary=[c for c in d["cells"] if c["module"]=="primary_capacity_feedback"]
    expected_cells={(p,trace,feedback) for p in (1024,4096) for trace in tuple(expected)[:4]
                    for feedback in ("fresh","temporary_feedback_delay")}
    expected_cells|={(4096,"alternating",f) for f in ("fresh","temporary_feedback_delay")}
    require({(c["P"],c["rate_trace"],c["feedback"]) for c in primary}==expected_cells and len(primary)==18,"primary cells changed")
    require(all(c["S"]==16 and c["G_executor_only"]==64 and c["load"]=="none" and
                c["added_RTT_trace_ms"]==[[0,0]] for c in primary),"primary treatment changed")
    secondary=[c for c in d["cells"] if c not in primary]
    require({c["id"] for c in secondary}=={"p1024-horizon-shift-g16","p1024-horizon-shift-g96",
        "p1024-RTT-pulse-fresh","p1024-RTT-pulse-temporary_feedback_delay",
        "p1024-holder-load-step","p1024-destination-load-step"},"secondary cells changed")
    require(len({c["id"] for c in d["cells"]})==24,"duplicate cells")
    for c in secondary:
        require(c["P"]==1024 and c["S"]==16,"secondary geometry changed")
        if c["module"]=="secondary_horizon_OOD":
            require(c["G_executor_only"] in (16,96) and c["rate_trace"]=="drop_recover" and
                    c["feedback"]=="temporary_feedback_delay" and c["load"]=="none" and c["added_RTT_trace_ms"]==[[0,0]],"horizon treatment changed")
        elif c["module"]=="secondary_RTT_OOD":
            require(c["G_executor_only"]==64 and c["rate_trace"]=="constant_fast" and c["load"]=="none" and
                    c["added_RTT_trace_ms"]==[[0,0],[.5,10],[2.5,0]],"RTT treatment changed")
        else:
            require(c["module"]=="secondary_compute_OOD" and c["G_executor_only"]==64 and
                    c["rate_trace"]=="constant_fast" and c["feedback"]=="fresh" and c["added_RTT_trace_ms"]==[[0,0]] and
                    c["load"]=={"location":c["id"].split("-")[1],"arrival_start_seconds":.5,"rho":.5,"until":"trial_stop"},"load treatment changed")
    require(d["ordering_and_pairing"]==dict(methods_per_cell=6,blocks_per_cell=8,warmups_per_cell_method=1,
        order_seed=2026112300,rule=d["ordering_and_pairing"]["rule"],hindsight=d["ordering_and_pairing"]["hindsight"]),"ordering/counts changed")
    return d


def build_plan(source=DEFAULT_DESIGN):
    d=_design(source); split=d["frozen_models_and_split"]
    families=split["test_families"]
    require(families==[{"id":f"e3-full-v1-test-{i:02d}","seed":2026112200+i} for i in range(8)],"held-out families changed")
    require(split["warmup_family"]=={"id":"e3-full-v1-warmup","seed":2026112100} and
            split["functional_smoke_family"]=={"id":"e3-full-v1-smoke","seed":2026112101},"warmup/smoke families changed")
    rng=random.Random(2026112300); cells=copy.deepcopy(d["cells"]); rng.shuffle(cells)
    trials=[]
    for cell in cells:
        methods=list(METHODS); rng.shuffle(methods)
        members=list(enumerate(families)); rng.shuffle(members)
        warm=list(methods); rng.shuffle(warm)
        blocks=[(-1,split["warmup_family"],warm)]+[(i,f,methods[i%6:]+methods[:i%6]) for i,f in members]
        rate=d["perturbation_protocol"]["rate_traces_seconds_Mbps"][cell["rate_trace"]]
        for i,family,order in blocks:
            role="warmup" if i<0 else "measured"; block="warmup" if i<0 else f"b{i:02d}"
            group=f"{cell['id']}.{block}"
            # Pair exogenous input/arrival recipes; never assert equal reached states.
            exogenous={"rate_steps":[{"at_seconds":t,"rate_bits_per_second":r*1_000_000} for t,r in rate],
                "feedback_recipe":{"kind":cell["feedback"],"request_interval_s":[.5,1.5],"withhold_s":1.0},
                "added_RTT_steps_ms":copy.deepcopy(cell["added_RTT_trace_ms"]),"load":copy.deepcopy(cell["load"]),
                "background_arrival_seed":int(digest({"cell":cell["id"],"family":family["id"]})[:12],16)}
            for position,method in enumerate(order):
                trial_id=f"E3.{group}.{method}"
                public={"prefix_tokens":cell["P"],"suffix_tokens":16,"public_output_cap":128,
                        "prefix_KV_bytes":cell["P"]*196608,"initial_application_cap_bps":rate[0][1]*1_000_000}
                trials.append({"trial_id":trial_id,"phase":"test","sample_role":role,"cell_id":cell["id"],
                    "module":cell["module"],"block_id":block,"paired_group_id":group,"method":method,
                    "method_order_position":position,"input_family_id":family["id"],"input_id":family["id"],
                    "seed":family["seed"],"prefix_key":f"{family['id']}:P{cell['P']}","public_admission":public,
                    "executor_only":{"realized_stop_tokens":cell["G_executor_only"],"EOS_stopping":False,
                                     "exogenous":copy.deepcopy(exogenous)},
                    "decision_contract":{"online":method in ONLINE,"completed_token_stride":4,
                        "no_callback_at_executor_stop":True,"fixed_methods_no_dummy_telemetry":True},
                    "exogenous_pairing_sha256":digest(exogenous),"token_arrays_frozen":False,
                    "diagnostic_id":trial_id+".route-diagnostic","reference_id":trial_id+".HF-reference"})
    profiles=[{"profile_id":f"profile.{r['cell']}.{m}","cell_id":r["cell"],"method":m,
               "sample_role":"instrumented_excluded","seed":2026112400}
              for r in d["separate_profile_plan"]["selection"] for m in r["methods"]]
    counts={"cells":24,"methods":6,"held_out_blocks":8,"measured":sum(t["sample_role"]=="measured" for t in trials),
            "warmups":sum(t["sample_role"]=="warmup" for t in trials),"candidates":len(trials),
            "diagnostics":len(trials),"HF_references":len(trials),"profiles":len(profiles),"functional_smoke_cases":8}
    require(counts["measured"]==1152 and counts["warmups"]==144 and counts["candidates"]==1296 and counts["profiles"]==8,"coverage mismatch")
    result={"schema_version":1,"stage":"E3","protocol":"E3_full_CPU_plan","design":d,"design_sha256":digest(d),
        "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name("e3_controller.py"))},
        "dependency_source_sha256":{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("e2_controller.py","e2_plan.py","analysis.py","plan.py")},
        "trials":trials,"counts":counts,"profiles":profiles,"functional_smoke":copy.deepcopy(d["functional_smoke_before_full"]),
        "status":"CPU_plan_not_frozen_or_executed","executable_plan":False,"GPU_execution_authorized":False,"E3_complete":False,
        "frozen_dependencies_required":["accepted_E1_full_raw_training_all_five_actions","accepted_E2_full_fit_and_calibration",
            "new_E3_main14B_functional_smoke","source_model_context_and_token_banks","all_channel_actuator_epoch_and_actual_ACK_integration",
            "actual_E2_training_feature_support_artifact"],
        "controller_implementation_choices":{"maximum_applied_changes_all_online":8,"guarded_dwell_s":.25,
            "greedy_dwell_s":0,"guarded_fractional_margin":.05,"stale_limit_s":.5,
            "paused_current_predictor_proxy":"stop_migration_and_remote","residual_coverage_claim":False}}
    result["plan_sha256"]=digest(result)
    return result


def validate_plan(plan):
    require(plan==build_plan(plan["design"]),"E3 plan/source/coverage/hash differs from deterministic materialization")
    return True


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(); plan=build_plan(); validate_plan(plan)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(plan,stream,indent=2,allow_nan=False); stream.write("\n")
    print(json.dumps({"path":str(args.output),"plan_sha256":plan["plan_sha256"],"counts":plan["counts"],"E3_complete":False}))


if __name__=="__main__": main()
