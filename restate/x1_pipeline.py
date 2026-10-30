"""Experiment 1 pipeline on the target host: fresh CAL -> fit -> X1 full (least_loaded) -> analysis.

Reuses mh_pipeline.run_stage (resume skips passed trials, at most three sessions per
stage). The preregistration file is copied into the run root and its SHA-256 is
frozen into the X1 plan before any X1 trial runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
import mh_pipeline  # noqa: E402
import mh_plan  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--holders", nargs="+", required=True)
    ap.add_argument("--target-gpu", type=int, default=0)
    ap.add_argument("--prereg", type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    prereg = root / "preregistration.md"
    if not prereg.exists():
        shutil.copyfile(args.prereg, prereg)
    prereg_sha = hashlib.sha256(prereg.read_bytes()).hexdigest()
    tag = root.name
    model_path = root / "cost-model.json"
    if not model_path.exists():
        records = mh_pipeline.run_stage(root, "CAL", mh_plan.build_calibration(tag + "-CAL"), args.holders, args.target_gpu)
        model = mh_plan.fit_model(records)
        model_path.write_text(json.dumps(model, indent=1, sort_keys=True))
        mh_pipeline.log(root, stage="fit", event="model_saved", model_sha256=model["model_sha256"],
                        b_star_bps=model["b_star_bps"], slowdown=model["slowdown"])
    model = json.loads(model_path.read_text())
    plan = mh_plan.build_x1(tag + "-X1-full", model, "full", prereg_sha)
    mh_pipeline.log(root, stage="X1-plan", event="frozen", plan_sha256=plan["plan_sha256"], preregistration_sha256=prereg_sha,
                    trials=len(plan["trials"]), identical_assignment_count=plan["identical_assignment_count"],
                    units=plan["units"])
    mh_pipeline.run_stage(root, "X1-full", plan, args.holders, args.target_gpu)
    out = subprocess.run([sys.executable, "-B", str(PKG / "x1_analysis.py"), "--plan", str(root / "X1-full" / "plan.json"),
                          "--records", str(root / "X1-full" / "normalized-records.json"), "--out", str(root / "analysis")],
                         cwd=str(PKG), capture_output=True, text=True)
    mh_pipeline.log(root, stage="analysis", event="done" if out.returncode == 0 else "failed", exit=out.returncode,
                    stderr=out.stderr[-2000:])
    mh_pipeline.log(root, stage="pipeline", event="complete")


if __name__ == "__main__":
    main()
