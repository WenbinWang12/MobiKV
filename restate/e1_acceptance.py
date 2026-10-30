"""Read-only E0 prerequisite checks and CPU latency derivation for an E1 runner.

An accepted E0 suite does not establish correctness of a new E1 implementation.
The E1 caller must separately require its matching correctness report. This module
does not submit jobs, update a ledger, measure wire bytes, or infer tail statistics.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
import subprocess
import types

from collect_e0_status import summarize_suite


REPO = "Qwen/Qwen2.5-14B"
REVISION = "97e1e76335b7017d8f67c08a19d103c0504298c9"
MANIFEST_SHA = "0ef3c81db38c9d2fe14959dc6e8abb5780bbf826b264dbacbdbf80d586cc5e99"
NODES = ["babel-v5-20", "babel-v5-24"]
REVIEWED_DRIVER_SHA256 = {
    "e0_suite_runner.py": "ac85a896b34554ce9148a1016a059c3ce2f0fdaa80cc1eea804d79b6d241c9eb",
    "e0_suite.sbatch": "f6606770ef2c18d41261110ff5c0fbe4e434c5b06b3dab6ab4ffc6396e02b31e",
}
REQUIRED_CASES = {
    "native-local", "layered-local", "native-crossnode", "holder", "remote",
    "repair-progressive", "repair-before-decode", "repair-cancel-incomplete",
    "multi-remote", "multi-repair",
}


class E0PrerequisiteError(RuntimeError):
    """The recorded E0 prerequisite is unavailable, incomplete, or inconsistent."""


def derive_latency_metrics(accepted_s, token_delivery_s, settled_s, expected_tokens):
    """Derive seconds from one destination monotonic clock and ordered deliveries.

    Each delivery is an individually timestamped token, not an inferred token rate.
    The last-token latency and settled completion are distinct. A single token has
    no inter-token interval; neither a percentile nor statistical claim is derived.
    """
    def timestamp(value, label):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{label} must be a finite real timestamp")
        return float(value)

    if isinstance(expected_tokens, bool) or not isinstance(expected_tokens, int) or expected_tokens < 1:
        raise ValueError("expected_tokens must be a positive integer")
    if not isinstance(token_delivery_s, (list, tuple)) or len(token_delivery_s) != expected_tokens:
        raise ValueError("Token delivery count must match expected_tokens")
    accepted = timestamp(accepted_s, "accepted_s")
    settled = timestamp(settled_s, "settled_s")
    delivered = [timestamp(value, f"token_delivery_s[{index}]") for index, value in enumerate(token_delivery_s)]
    if delivered[0] < accepted or any(right <= left for left, right in zip(delivered, delivered[1:])):
        raise ValueError("Deliveries must start after acceptance and increase strictly")
    if settled < delivered[-1]:
        raise ValueError("settled_s must not precede the last token delivery")
    itl = [right - left for left, right in zip(delivered, delivered[1:])]
    metrics = {"token_count": expected_tokens, "ttft_s": delivered[0] - accepted,
            "itl_s": itl, "mean_itl_s": sum(itl) / len(itl) if itl else None,
            "token_delivery_span_s": delivered[-1] - delivered[0],
            "last_token_latency_s": delivered[-1] - accepted,
            "total_completion_s": settled - accepted,
            "post_last_token_settle_s": settled - delivered[-1]}
    if not all(math.isfinite(value) for value in [*itl, *[value for key, value in metrics.items()
                                                        if key != "itl_s" and value is not None]]):
        raise ValueError("Timestamp differences overflowed; derived latency must be finite")
    return metrics


def _require(condition, message):
    if not condition:
        raise E0PrerequisiteError(message)


def _digest(path):
    _require(path.is_file() and not path.is_symlink(), f"Missing or symlinked evidence file: {path}")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _read_object(path):
    _require(path.is_file() and not path.is_symlink(), f"Missing or symlinked JSON evidence: {path}")
    _require(path.stat().st_size <= 64 * 1024**2, f"Unexpectedly large JSON evidence: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def _scheduler_evidence(job):
    output = subprocess.check_output([
        "sacct", "-j", job, "--allocations", "--parsable2", "--noheader",
        "--format=JobIDRaw,State,ExitCode,AllocTRES,NodeList"], text=True, timeout=30)
    rows = [line.split("|") for line in output.splitlines() if line.split("|", 1)[0] == job]
    _require(len(rows) == 1 and len(rows[0]) >= 5, "Expected exactly one sacct allocation row for the suite")
    _, state, exit_code, tres, nodes = rows[0][:5]
    _require(state == "COMPLETED" and exit_code == "0:0", f"E0 suite is not successfully completed: {state} {exit_code}")
    _require(re.search(r"(?:^|,)gres/gpu=2(?:,|$)", tres) is not None, "E0 suite did not allocate exactly two GPUs")
    _require(nodes in (",".join(NODES), "babel-v5-[20,24]"), "Scheduler hosts differ from the reviewed A100 pair")
    return {"job_id": job, "state": state, "exit_code": exit_code, "allocated_tres": tres, "nodes": nodes}


def assert_e0_suite_passed(root, suite_job_id):
    """Return hashed E0 prerequisite evidence or raise E0PrerequisiteError.

    Only sacct is invoked. Files are read without editing progress.json or reports.
    The frozen reviewed driver's existing CPU validators are rerun over actual rank
    and process artifacts; a top-level passed flag alone is insufficient.
    """
    try:
        return _assert_e0_suite_passed(Path(root), suite_job_id)
    except E0PrerequisiteError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError, subprocess.SubprocessError) as error:
        raise E0PrerequisiteError(f"Cannot verify E0 prerequisite: {type(error).__name__}: {error}") from error


def _assert_e0_suite_passed(root, suite_job_id):
    _require(not isinstance(suite_job_id, bool) and re.fullmatch(r"[1-9][0-9]*", str(suite_job_id)), "Invalid suite job ID")
    job = str(suite_job_id)
    root = root.resolve(strict=True)
    submitted = root / "runs/submitted"
    _require(not (root / "runs").is_symlink() and not submitted.is_symlink(), "Submission directories must not be symlinks")
    scheduler = _scheduler_evidence(job)
    matches = []
    for path in submitted.glob("*/submission.json"):
        try:
            value = _read_object(path)
        except (E0PrerequisiteError, OSError, ValueError):
            continue
        if str(value.get("job_id", "")) == job:
            matches.append((path, value))
    _require(len(matches) == 1, "Expected one retained submission manifest for this suite job")
    submission_path, submission = matches[0]
    frozen = submission_path.parent
    _require(not frozen.is_symlink() and frozen.resolve().parent == submitted.resolve(), "Invalid frozen snapshot directory")
    _require(submission.get("frozen_directory") == str(frozen), "Submission frozen_directory differs from its location")
    _require(submission.get("submission_type") == "e0_suite" and submission.get("status") == "submitted" and
             submission.get("job_submitted") is True, "Not a submitted E0 suite manifest")
    allocation = submission.get("allocation", {})
    for key, expected in {"nodes": 2, "gpus_per_node": 1, "gpu": "A100_80GB", "nodelist": ",".join(NODES)}.items():
        _require(allocation.get(key) == expected, f"Incorrect E0 allocation: {key}")
    for key, expected in {"model_repo": REPO, "model_revision": REVISION, "model_manifest_sha256": MANIFEST_SHA}.items():
        _require(submission.get(key) == expected, f"Incorrect E0 model identity: {key}")
    _require(submission.get("plan_file") == "suite_cases.json", "Unexpected frozen plan filename")
    artifact_hashes = {}

    def retain(path):
        _require(path.resolve().is_relative_to(root), f"Evidence path leaves workspace: {path}")
        digest = _digest(path)
        artifact_hashes[str(path.relative_to(root))] = digest
        return digest

    retain(submission_path)
    sources = submission.get("source_sha256", {})
    _require(isinstance(sources, dict) and sources, "Submission source hashes are missing")
    for name, expected in sources.items():
        _require(isinstance(name, str) and Path(name).name == name and name not in (".", "..") and
                 isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "Invalid source hash entry")
        _require((frozen / name).stat().st_size <= 2 * 1024**2, f"Unexpectedly large frozen source: {name}")
        _require(retain(frozen / name) == expected, f"Frozen source changed: {name}")
    for name, expected in REVIEWED_DRIVER_SHA256.items():
        _require(sources.get(name) == expected, f"Suite driver needs renewed review: {name}")
    plan_path = frozen / "suite_cases.json"
    _require(retain(plan_path) == submission.get("plan_sha256"), "Frozen plan hash changed")
    plan = _read_object(plan_path)
    # Execute only the locally retained driver whose complete bytes match the reviewed hash.
    runner = types.ModuleType("_reviewed_e0_gate_runner")
    runner.__file__ = str(frozen / "e0_suite_runner.py")
    runner_bytes = (frozen / "e0_suite_runner.py").read_bytes()
    _require(hashlib.sha256(runner_bytes).hexdigest() == REVIEWED_DRIVER_SHA256["e0_suite_runner.py"], "Reviewed driver changed before validation")
    exec(compile(runner_bytes, runner.__file__, "exec"), runner.__dict__)
    runner.validate_plan(plan, frozen)
    _require({case["id"] for case in plan["cases"]} == REQUIRED_CASES and
             len(plan["cases"]) == len(REQUIRED_CASES) and
             not any(case.get("optional_if_missing") for case in plan["cases"]), "E1 prerequisite requires all ten reviewed E0 cases")
    summary = summarize_suite(root, job, submission, {"state": "COMPLETED"})
    _require(summary.get("suite_passed_selected_cases") is True and summary.get("planned_execution_count") == 12,
             "Suite selected cases are not all passed with complete matching evidence")
    directory = root / "results" / f"e0-suite-{job}"
    _require(not (root / "results").is_symlink() and not directory.is_symlink(), "Result directories must not be symlinks")
    retain(directory / "suite.json")
    retain(directory / "suite_manifest.json")
    protocols = {name: _read_object(frozen / name) for name in sources
                 if name.endswith("_precision_protocol.json") or name == "e0_precision_protocol.json"}
    plans = {case["id"]: case for case in plan["cases"]}
    for case in summary["cases"]:
        rawdir = Path(case["raw_artifact_directory"])
        _require(not rawdir.is_symlink() and rawdir.resolve().parent == directory.resolve(), "Invalid case result directory")
        retain(rawdir / "case_acceptance.json")
        reports, processes = {}, {}
        for rank in range(len(case["selected_nodes"])):
            for collection, name in ((reports, f"rank-{rank}.json"), (processes, f"rank-{rank}-process.json")):
                path = rawdir / name
                retain(path)
                collection[rank] = _read_object(path)
        verification = runner.validate_reports(plans[case["id"]], reports, processes,
                                               case["selected_nodes"], job, sources, protocols)
        _require(verification["passed"] is True, f"Raw case acceptance failed: {case['id']}: {verification['failures']}")
    for relative, expected in artifact_hashes.items():
        _require(_digest(root / relative) == expected, f"Evidence changed during verification: {relative}")
    return {"suite_job_id": job, "plan_sha256": submission["plan_sha256"], "frozen_directory": str(frozen),
            "checked_utc": datetime.now(timezone.utc).isoformat(),
            "e0_selected_suite_prerequisite_passed": True, "E0_complete": False,
            "E1_source_correctness_implied": False, "performance_evidence": False,
            "scope": "E0 prerequisite only; E1 additionally requires matching implementation correctness evidence.",
            "model_repo": REPO, "model_revision": REVISION, "model_manifest_sha256": MANIFEST_SHA,
            "allocation": allocation, "scheduler": scheduler, "case_executions_verified": len(summary["cases"]),
            "source_sha256": sources, "artifact_sha256": artifact_hashes,
            "gate_sha256": _digest(Path(__file__).resolve()),
            "collector_sha256": _digest(Path(__file__).with_name("collect_e0_status.py").resolve())}


def _self_test():
    """CPU-only negative fixtures; no real scheduler query or ledger writes."""
    import tempfile
    from unittest.mock import patch

    value = derive_latency_metrics(10, [11, 13, 16], 17, 3)
    assert value["ttft_s"] == 1 and value["itl_s"] == [2, 3] and value["total_completion_s"] == 7
    assert derive_latency_metrics(1, [2], 2, 1)["mean_itl_s"] is None
    for args in ((0, [], 1, 0), (0, [1], 2, 2), (0, [2, 1], 3, 2),
                 (0, [1, 1], 2, 2), (2, [1], 3, 1), (0, [2], 1, 1),
                 (False, [1], 2, 1), (0, [math.nan], 2, 1), (0, [1], math.inf, 1),
                 (-1e308, [1e308], 1e308, 1)):
        try:
            derive_latency_metrics(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid timing fixture accepted: {args}")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for state, code in (("PENDING", "0:0"), ("FAILED", "1:0"), ("COMPLETED", "0:0")):
            with patch.object(subprocess, "check_output", return_value=f"123|{state}|{code}|gres/gpu=2|babel-v5-[20,24]|\n"):
                try:
                    assert_e0_suite_passed(root, "123")
                except E0PrerequisiteError:
                    pass
                else:
                    raise AssertionError("Pending, failed, or missing artifacts must never authorize E1")
    return {"passed": True, "scope": "CPU timestamp checks and pending/failed/missing-artifact denial fixtures",
            "successful_E0_evidence_created": False,
            "positive_gate_requires": "Available real submitted suite, frozen sources, raw rank/process artifacts, and completed sacct record"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--suite-job")
    args = parser.parse_args()
    if args.self_test:
        if args.suite_job:
            parser.error("--self-test does not accept --suite-job")
        result = _self_test()
    else:
        if not args.suite_job:
            parser.error("Provide --suite-job for a real read-only gate, or --self-test")
        result = assert_e0_suite_passed(args.root, args.suite_job)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
