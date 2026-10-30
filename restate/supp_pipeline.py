"""Supplementary TMC pipeline (HET / LM / M7B), same discipline as x34_pipeline: frozen preregistration and
inputs, calibration bound to the frozen plan, verified rank cleanup per stage, analysis over all slots.

HET  --base-model (native BASE) --rate R --slow-holder 1 --mps-percentage p: per-rate, per-holder calibration
     (x34 calibration grid) -> fit_model_x34(per_holder=True) -> HET plan.
LM   --cost-model (frozen C3T cost model) --native-model (native BASE) --rates: no calibration; the arrival
     scale is the preregistered ratio of mean standalone recompute completion (throttled / native).
M7B  --cal-plan smoke|full: build_smoke_calibration (functional only, no plan) or build_calibration ->
     fit_model -> M7B plan.
--mode smoke uses one calibration repetition and the one-block smoke plans; nothing from smoke is analysed
as a result.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import subprocess
import sys

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
import c3t_pipeline as c3t  # noqa: E402
import mh_pipeline as pipeline  # noqa: E402
import mh_plan as P  # noqa: E402
import supp_plan as S  # noqa: E402
import supp_release as R  # noqa: E402
from x34_pipeline import freeze_prereg  # noqa: E402


def mps_gate(root, stage, expected):
    """Every host's MPS default must equal the expected cap; hosts not listed must have no MPS daemon."""
    state = c3t.mps_state()
    want = {ip: expected.get(ip) for ip in state}
    pipeline.log(root, stage="mps", event=stage, state=state, expected=want)
    pipeline.require(all((v is None) == (want[ip] is None) and (v is None or abs(v - want[ip]) < 1e-6)
                         for ip, v in state.items()), "MPS state differs at %s: %s" % (stage, state))


def arrival_scale(throttled, native):
    """Preregistered: mean of the 1024- and 4096-token standalone recompute completions, throttled / native."""
    mean = lambda m: statistics.mean(m["standalone"]["recompute|%d" % p]["completion_s"] for p in (1024, 4096))
    return mean(throttled) / mean(native)


def run(args):
    root, exp, mode = args.root.resolve(), args.experiment, args.mode
    tag = root.name
    with pipeline.pipeline_lock(root):
        prereg_sha = freeze_prereg(args.prereg, root / "preregistration.md")
        inputs = dict(experiment=exp, mode=mode, preregistration_sha256=prereg_sha, holders=args.holders,
                      target_gpu=args.target_gpu, source_sha256=pipeline.source_closure(), kv_bytes_per_token=P.KV_BYTES_PER_TOKEN,
                      scientific_acceptance=False, coverage_claim_allowed=False)
        for name in ("base_model", "cost_model", "native_model"):
            path = getattr(args, name)
            if path is not None:
                value = pipeline.read(path)
                pipeline.validate_hash(value, "model_sha256")
                inputs[name + "_path"], inputs[name + "_file_sha256"] = str(path.resolve()), pipeline.sha(path)
                inputs[name + "_sha256"] = value["model_sha256"]
        mps = dict(item.split("=") for item in args.mps)
        mps = {ip: float(v) for ip, v in mps.items()}
        inputs.update(rate=args.rate, rates=args.rates, loaded_holder=args.loaded_holder, mps_expected=mps)
        pipeline.immutable_json(root / "frozen-input-bindings.json", inputs)
        mps_gate(root, "start", mps)
        reps = 1 if mode == "smoke" else 2
        model = None
        if exp == "HET":
            base = pipeline.read(args.base_model)
            calplan = S.build_het_calibration(tag + "-CAL", args.rate, args.loaded_holder, reps=reps)
            records = pipeline.run_stage(root, "CAL", calplan, args.holders, args.target_gpu)
            model = P.fit_model_x34(records, base, [args.rate], per_holder=True, loaded_holders=(args.loaded_holder,))
        elif exp == "DEP":
            base = pipeline.read(args.base_model)
            calplan = P.build_x34_calibration(tag + "-CAL", [args.rate], reps=reps)
            records = pipeline.run_stage(root, "CAL", calplan, args.holders, args.target_gpu)
            model = R.fit_release(P.fit_model_x34(records, base, [args.rate]), root / "CAL", records, [args.rate])
        elif exp == "M7B":
            if args.cal_plan == "smoke":
                pipeline.run_stage(root, "CAL-smoke", P.build_smoke_calibration(tag + "-CAL-smoke"), args.holders, args.target_gpu)
                pipeline.log(root, stage="pipeline", event="complete", smoke_calibration_only=True, scientific_acceptance=False)
                return
            calplan = P.build_calibration(tag + "-CAL", reps=reps)
            records = pipeline.run_stage(root, "CAL", calplan, args.holders, args.target_gpu)
            model = P.fit_model(records)
        if model is not None:
            pipeline.validate_hash(model, "model_sha256")
            pipeline.immutable_json(root / "cost-model.json", model)
            pipeline.immutable_json(root / "frozen-model-bindings.json", dict(
                preregistration_sha256=prereg_sha, calibration_plan_sha256=calplan["plan_sha256"],
                calibration_records_sha256=pipeline.sha(root / "CAL/normalized-records.json"),
                calibration_acceptance_sha256=pipeline.sha(root / "CAL/stage-acceptance.json"),
                model_sha256=model["model_sha256"], model_file_sha256=pipeline.sha(root / "cost-model.json"),
                source_sha256=pipeline.source_closure()))
            pipeline.log(root, stage="fit", event="model_saved", model_sha256=model["model_sha256"], b_star_bps=model["b_star_bps"])
        stage = "%s-%s" % (exp, mode)
        pipeline.require(pipeline.source_closure() == inputs["source_sha256"], "Frozen package changed before stage")
        if exp == "HET":
            plan = S.build_het(tag + "-" + stage, model, mode, args.rate, args.loaded_holder, prereg_sha)
        elif exp == "DEP":
            plan = S.build_dep(tag + "-" + stage, model, mode, args.rate, prereg_sha)
        elif exp == "LM":
            model, native = pipeline.read(args.cost_model), pipeline.read(args.native_model)
            scale = arrival_scale(model, native)
            pipeline.log(root, stage="arrival_scale", event="computed", scale=scale)
            plan = S.build_lm(tag + "-" + stage, model, mode, args.rates, scale, prereg_sha)
        else:
            plan = S.build_m7b(tag + "-" + stage, model, mode, prereg_sha)
        pipeline.log(root, stage=stage + "-plan", event="frozen", plan_sha256=plan["plan_sha256"], trials=len(plan["trials"]),
                     identical_assignment_count=plan.get("identical_assignment_count"))
        mps_gate(root, "before_" + stage, mps)
        pipeline.run_stage(root, stage, plan, args.holders, args.target_gpu)
        mps_gate(root, "after_" + stage, mps)
        out = subprocess.run([sys.executable, "-B", str(PKG / "supp_analysis.py"), "--root", str(root), "--experiment", exp,
                              "--stage", stage], cwd=str(PKG), capture_output=True, text=True, timeout=3600)
        pipeline.log(root, stage="analysis", event="done" if out.returncode == 0 else "failed", exit=out.returncode,
                     stdout=out.stdout[-2000:], stderr=out.stderr[-2000:])
        pipeline.require(out.returncode == 0, "Analysis process failed; pipeline cannot be complete")
        pipeline.require(pipeline.read(root / "analysis/summary.json")[exp]["missing_or_failed"] == [],
                         "Analysis reports missing/failed trials")
        pipeline.log(root, stage="pipeline", event="complete", experiment=exp, mode=mode, scientific_acceptance=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=("HET", "LM", "M7B", "DEP"), required=True)
    ap.add_argument("--mode", choices=("smoke", "full"), required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--holders", nargs="+", required=True)
    ap.add_argument("--target-gpu", type=int, default=0)
    ap.add_argument("--prereg", type=Path, required=True)
    ap.add_argument("--base-model", type=Path)
    ap.add_argument("--cost-model", type=Path)
    ap.add_argument("--native-model", type=Path)
    ap.add_argument("--rate", type=int)
    ap.add_argument("--rates", type=int, nargs="+")
    ap.add_argument("--loaded-holder", type=int, choices=(0, 1))
    ap.add_argument("--cal-plan", choices=("smoke", "full"), default="full")
    ap.add_argument("--mps", nargs="*", default=[], help="ip=percentage for every host that must run MPS")
    args = ap.parse_args()
    need = {"HET": ("base_model", "rate", "loaded_holder"), "LM": ("cost_model", "native_model", "rates"),
            "M7B": (), "DEP": ("base_model", "rate")}[args.experiment]
    missing = [n for n in need if getattr(args, n) is None]
    if missing:
        ap.error("missing for %s: %s" % (args.experiment, missing))
    run(args)


if __name__ == "__main__":
    main()
