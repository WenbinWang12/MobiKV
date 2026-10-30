"""Sequential MH stages with immutable evidence and verified rank settlement."""
from __future__ import annotations
import restate_config
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

PKG = Path(__file__).resolve().parent
PYTHON = restate_config.PYTHON
sys.path.insert(0, str(PKG))
import mh_plan

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value, message):
    if not value: raise RuntimeError(message)
def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-%d" % os.getpid())
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=1, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def immutable_json(path, value):
    path = Path(path)
    if path.exists(): require(read(path) == value, "Existing immutable binding differs: " + str(path))
    else: save(path, value)
    return sha(path)

def source_closure():
    return {p.relative_to(PKG).as_posix(): sha(p) for p in sorted(PKG.rglob("*.py")) if p.is_file()}

def validate_hash(value, key):
    require(isinstance(value, dict) and key in value, "Missing content hash " + key)
    require(value[key] == mh_plan.digest({k:v for k,v in value.items() if k != key}), "Content hash mismatch: " + key)

def log(root, **entry):
    root = Path(root); entry["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    event = entry.get("event", "")
    entry.setdefault("status", "failed" if "fail" in event else "complete" if event == "complete" else "running")
    with (root / "pipeline-status.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, allow_nan=False) + "\n"); f.flush(); os.fsync(f.fileno())
    save(root / "current-status.json", entry)
    print(json.dumps(entry), flush=True)

@contextmanager
def pipeline_lock(root):
    import fcntl
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    with (root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        except BaseException as error:
            log(root, stage="pipeline", event="failed", error=repr(error), traceback=traceback.format_exc(),
                raw_results_preserved=True, scientific_acceptance=False, coverage_claim_allowed=False)
            raise

def progress(result):
    path = Path(result) / "progress.json"
    return read(path) if path.exists() else {}

def records_for_plan(result, plan, complete=False):
    path = Path(result) / "normalized-records.json"
    if not path.exists():
        require(not complete, "Missing normalized records"); return []
    payload = read(path)
    require(payload.get("plan_sha256") == plan["plan_sha256"], "Records belong to a different plan")
    records = payload["records"]; ids = {t["trial_id"] for t in plan["trials"]}
    require(all(r.get("trial_id") in ids for r in records), "Foreign trial in retained records")
    passed = Counter(r["trial_id"] for r in records if r.get("status") == "passed")
    require(all(n == 1 for n in passed.values()), "Duplicate successful trial; hidden rerun")
    if complete: require(set(passed) == ids, "Not all planned trials have a successful record")
    return records

def session_acceptance(result, plan):
    result = Path(result); latest = result / "latest-launch.json"
    require(latest.exists(), "Missing launcher settlement evidence")
    launch = read(latest)
    require(launch.get("passed") is True and launch.get("target_exit") == 0
            and launch.get("all_owned_exited") is True and launch.get("cleanup_verified") is True,
            "Launcher did not verify all rank exits and cleanup")
    context_path = Path(launch["context"]); context = read(context_path)
    require(context["run_id"] == plan["run_id"] and context["plan_sha256"] == sha(result / "plan.json"),
            "Launcher context belongs to another plan")
    require(context["package_sha256"] == source_closure(), "Source closure changed since session")
    from mh_launch import verify_rank_reports
    reports = verify_rank_reports(result, context, context_path)
    records = records_for_plan(result, plan, complete=True)
    receipt = dict(passed=True, session_id=context["session_id"], run_id=plan["run_id"],
        plan_sha256=plan["plan_sha256"], plan_file_sha256=sha(result / "plan.json"),
        records_sha256=sha(result / "normalized-records.json"), context_sha256=sha(context_path),
        launch_sha256=sha(latest), rank_report_sha256=reports, source_sha256=context["package_sha256"],
        all_owned_exited=True, cleanup_verified=True, passed_trials=len(plan["trials"]),
        failed_attempt_records=sum(r.get("status") != "passed" for r in records),
        scientific_acceptance=False, coverage_claim_allowed=False)
    save(result / "stage-acceptance.json", receipt)
    return records

def run_stage(root, name, plan, holders, target_gpu, attempts=3):
    result = Path(root) / name; result.mkdir(parents=True, exist_ok=True)
    validate_hash(plan, "plan_sha256")
    require(len({t["trial_id"] for t in plan["trials"]}) == len(plan["trials"]), "Duplicate planned trials")
    plan_path = result / "plan.json"; immutable_json(plan_path, plan)
    immutable_json(result / "source-binding.json", dict(source_sha256=source_closure(),
        plan_sha256=plan["plan_sha256"], plan_file_sha256=sha(plan_path)))
    records_for_plan(result, plan)
    try:
        records = session_acceptance(result, plan)
        log(root, stage=name, event="stage_reused", trials=len(plan["trials"]), completed_evidence_verified=True)
        return records
    except RuntimeError: pass
    latest = result / 'latest-launch.json'
    if latest.exists():
        require(read(latest).get('all_owned_exited') is True,
                'Prior session still has running or unverified owned processes; no overlapping retry')
    last_error = None
    for attempt in range(attempts):
        state = progress(result)
        log(root, stage=name, event="session_start", attempt=attempt, passed=state.get("passed_trials", 0), expected=len(plan["trials"]))
        code = subprocess.run([PYTHON, "-B", str(PKG / "mh_launch.py"), "--plan", str(plan_path),
            "--result-dir", str(result), "--holders", *holders, "--target-gpu", str(target_gpu)], cwd=str(PKG)).returncode
        try:
            require(code == 0, "Launcher failed with exit %d" % code)
            records = session_acceptance(result, plan)
            log(root, stage=name, event="stage_complete", trials=len(plan["trials"]),
                failed_records=sum(r.get("status") != "passed" for r in records))
            return records
        except RuntimeError as error:
            last_error = str(error)
            log(root, stage=name, event="session_failed", attempt=attempt, exit=code, error=last_error,
                passed=progress(result).get("passed_trials", 0), raw_results_preserved=True)
            latest = result / "latest-launch.json"
            require(latest.exists() and read(latest).get("all_owned_exited") is True,
                    "Cannot retry while prior owned processes remain or their status is unknown")
            if attempt + 1 < attempts: time.sleep(20)
    raise RuntimeError("Stage %s incomplete or cleanup failed: %s" % (name, last_error))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True); ap.add_argument("--holders", nargs="+", required=True)
    ap.add_argument("--target-gpu", type=int, default=0); ap.add_argument("--stages", default="cal,e6smoke,e6full,e7smoke,e7full")
    args = ap.parse_args(); root = args.root.resolve()
    with pipeline_lock(root):
        stages, tag = args.stages.split(","), root.name; model_path = root / "cost-model.json"
        if "cal" in stages:
            records = run_stage(root, "CAL", mh_plan.build_calibration(tag + "-CAL"), args.holders, args.target_gpu)
            model = mh_plan.fit_model(records); immutable_json(model_path, model)
            log(root, stage="fit", event="model_saved", model_sha256=model["model_sha256"])
        model = read(model_path); validate_hash(model, "model_sha256")
        for stage, builder, mode in (("e6smoke", mh_plan.build_e6, "smoke"), ("e6full", mh_plan.build_e6, "full"),
                                     ("e7smoke", mh_plan.build_e7, "smoke"), ("e7full", mh_plan.build_e7, "full")):
            if stage in stages:
                name = stage.upper().replace("SMOKE", "-smoke").replace("FULL", "-full")
                run_stage(root, name, builder(tag + "-" + name, model, mode), args.holders, args.target_gpu)
        log(root, stage="pipeline", event="complete", stages=stages)

if __name__ == "__main__": main()
