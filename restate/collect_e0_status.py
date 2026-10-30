"""Refresh the experiment ledger from Slurm and retained reports; never submit jobs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


def compact_checks(checks):
    fields = ("logits_pass", "tokens_match", "finite", "max_abs_error",
              "max_tolerance_ratio", "min_reference_top2_margin", "required_for_acceptance",
              "reference_length_pass", "delta_length_pass", "delta_released",
              "all_remote_pass", "exact_source_audit_match", "cutover_pass",
              "reference", "candidate", "passed", "exact_equal", "greedy_equal",
              "captured_layers_exact_equal", "captured_layers_finite", "final_cache_hash_equal")
    tensor_fields = ("finite", "exact_equal", "strict_1e_minus4_pass", "elements", "max_abs_error",
                     "rms_error", "reference_absmax", "candidate_absmax", "reference_rms", "candidate_rms",
                     "max_error_vocab_id", "max_error_feature_index", "reference_at_max_error",
                     "candidate_at_max_error", "max_tolerance_ratio")
    result = {}
    for name, value in checks.items():
        if isinstance(value, bool):
            result[name] = {"passed": value}
            continue
        small = {key: value[key] for key in fields if key in value}
        if isinstance(value.get("logits"), dict):
            small["logits"] = {key: value["logits"][key] for key in tensor_fields if key in value["logits"]}
        if "first_divergent_layer" in value:
            first = value["first_divergent_layer"]
            small["first_divergent_layer"] = None
            if isinstance(first, dict):
                detail = {key: first[key] for key in
                          ("layer", "first_different_field_in_declared_order", "different_fields") if key in first}
                detail["summaries"] = {field: {key: summary[key] for key in tensor_fields if key in summary}
                                       for field, summary in first.get("summaries", {}).items()}
                small["first_divergent_layer"] = detail
        result[name] = small
    return result


def compact_validation_report(report):
    fields = ("stage", "rank", "hostname", "passed", "error", "precision", "gpu", "model_repo",
              "model_revision", "model_manifest_sha256", "manifest_sha256",
              "same_shape_determinism_pass", "cross_shape_is_acceptance_gate",
              "network_exercised", "crossnode_evidence", "migration_copy_exercised",
              "hardware_overlap_evidence", "performance_evidence", "scope",
              "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes", "gpu_allocated_after_branches_bytes",
              "identity", "model_shape", "gpu_memory_peak", "required_checks", "numerical_pass",
              "delta_length_pass", "overlap", "boundary", "boundary_pass", "cleanup_pass",
              "cleanup", "migration_terminal", "exception_cleanup", "prefix_immutable_pass",
              "delta_storage_separation_pass", "cancellation", "source_sha256",
              "source_prefix_prefill_count", "persistent_gpu_prefix_copies", "prefix_unchanged",
              "shared_prefix_sha256_before", "shared_prefix_sha256_after",
              "b_rpc_calls_after_a_finish", "b_tokens_after_a_finish",
              "shared_prefix_released_after_both_requests", "shared_empty_prefix_store",
              "shared_prefix_released_after_both_requests_and_migration", "prefix_users_before_release",
              "b_remote_calls_after_a", "b_local_calls_after_a", "b_final_forward_all_local",
              "b_later_layers_cut_over_after_a", "deterministic_fixture",
              "ungated_concurrency_or_overlap_benefit_claim", "gates")
    small = {key: report[key] for key in fields if key in report}
    small["checks"] = compact_checks(report.get("checks", {}))
    if "branches" in report:
        small["branches"] = {name: {key: branch[key] for key in (
            "numerical", "passed", "mode_counts", "transition_count", "owned_delta_references_released",
            "no_replica_alias", "prefix_unchanged", "progressive_state_pass", "state_pass") if key in branch}
            for name, branch in report["branches"].items()}
    small["cases"] = [{"case_id": case.get("case_id"), "passed": case.get("passed"),
                       **{key: case[key] for key in ("integrity_pass", "restored_integrity_pass",
                           "local_prefix_matches_received", "cache_lengths_pass", "no_prefix_recompute_pass")
                          if key in case},
                       "checks": compact_checks(case.get("checks", {}))}
                      for case in report.get("cases", [])]
    small["requests"] = {}
    for name, value in report.get("requests", {}).items():
        fields = ("generated_tokens", "delta_length_pass", "finished", "replica_released",
                  "b_state_unchanged", "prefix_retained", "rpc_calls_after_a_finish",
                  "delta_released", "all_remote_pass", "exact_source_audit_match")
        request = {key: value[key] for key in fields if key in value}
        if "all_vocab_check" in value:
            request["checks"] = compact_checks({"reference": value["all_vocab_check"]})
        if "token_ids" in value:
            request["generated_tokens"] = len(value["token_ids"])
        request["rpc_count"] = len(value.get("rpc_audit", value.get("audit", [])))
        small["requests"][name] = request
    return small


def is_copy_submission(manifest):
    return (Path(manifest.get("program", "")).name == "e0_copy_model.py" or
            all(key in manifest for key in ("source", "target", "snapshot", "model_revision")))


def summarize_copy(root, job, manifest, record):
    """Copy success is staging evidence, never an E0 correctness validation result."""
    reports, errors = [], []
    fields = ("status", "rank", "slurm_job_id", "hostname", "source_host", "target_host",
              "model_path", "manifest_sha256", "manifest_payload_bytes", "model_file_bytes",
              "model_payload_bytes_transferred", "all_target_files_verified", "chunk_bytes",
              "transport", "bytes_are_payload_not_wire_measurements", "target_free_bytes_before",
              "required_free_bytes", "started_at_utc", "finished_at_utc", "error_type", "error",
              "copied_files", "reused_files")
    paths = [root / "results" / f"e0-copy-model-{job}-rank{rank}.json" for rank in (0, 1)]
    for rank, path in enumerate(paths):
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text())
            if not isinstance(raw, dict):
                raise ValueError("Rank report must be a JSON object")
        except (OSError, ValueError) as error:
            errors.append({"path": str(path), "error": f"{type(error).__name__}: {error}"})
            continue
        small = {key: raw[key] for key in fields if key in raw}
        small["report_identity_matches"] = (raw.get("rank") == rank and
                                              str(raw.get("slurm_job_id", "")) == job)
        reports.append(small)
    complete = len(reports) == 2 and all(report["report_identity_matches"] for report in reports)
    hashes = [report.get("manifest_sha256") for report in reports]
    same_manifest = (complete and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                                     for value in hashes) and len(set(hashes)) == 1)
    declared_hash = manifest.get("manifest_sha256")
    if declared_hash is not None:
        same_manifest = same_manifest and hashes[0] == declared_hash
    copy_pass = bool(complete and same_manifest and all(
        report.get("status") == "passed" and report.get("all_target_files_verified") is True
        for report in reports))
    record.update(copy_passed=copy_pass if complete else None,
                  source_snapshot=manifest.get("snapshot", manifest.get("frozen_directory")))
    return {"job_id": job, "stage": record.get("stage", "model_copy"),
            "evidence_type": "model_staging_copy_not_E0_validation",
            "slurm_state": record.get("state"), "program": "e0_copy_model.py",
            "source": manifest.get("source"), "target": manifest.get("target"),
            "model_revision": manifest.get("model_revision"),
            "source_snapshot": record["source_snapshot"],
            "files_sha256": manifest.get("files_sha256", {}),
            "all_reports_present": complete, "manifest_identity_matches": bool(same_manifest),
            "copy_passed": copy_pass if complete else None,
            "reports": reports, "report_read_errors": errors,
            "raw_artifact_paths": [str(path) for path in paths]}


def read_suite_object(path, errors):
    """Missing/incomplete reports are evidence gaps, never successful results."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Report must be a JSON object")
        return value
    except (OSError, ValueError) as error:
        errors.append({"path": str(path), "error": f"{type(error).__name__}: {error}"})
        return None


def summarize_suite(root, job, manifest, record):
    """Collect the frozen driver's acceptance, without declaring the E0 stage complete."""
    directory = root / "results" / f"e0-suite-{job}"
    errors, failures, cases, gaps = [], [], [], []
    runtime = {}
    for name in ("suite.json", "suite_manifest.json"):
        path = directory / name
        if path.exists():
            runtime[name] = read_suite_object(path, errors) or {}
        else:
            gaps.append(f"Awaiting {name}")
    suite = runtime.get("suite.json", {})
    retained = runtime.get("suite_manifest.json", {})
    plan, actual_plan_hash = {}, None
    try:
        frozen = Path(manifest["frozen_directory"])
        if frozen.resolve().parent != (root / "runs/submitted").resolve():
            raise ValueError("Frozen directory must be an immediate child of runs/submitted")
        name = manifest.get("plan_file", "suite_cases.json")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Frozen plan must be a plain filename")
        plan_path = frozen / name
        if plan_path.is_symlink():
            raise ValueError("Frozen plan must not be a symlink")
        data = plan_path.read_bytes()
        plan = json.loads(data)
        if not isinstance(plan, dict):
            raise ValueError("Frozen plan must be a JSON object")
        actual_plan_hash = hashlib.sha256(data).hexdigest()
    except (KeyError, TypeError, OSError, ValueError) as error:
        errors.append({"path": manifest.get("frozen_directory"), "error": f"{type(error).__name__}: {error}"})
        plan = {}
    sources = manifest.get("source_sha256", {})
    frozen_matches = bool(
        isinstance(sources, dict) and sources and actual_plan_hash == manifest.get("plan_sha256")
        and plan.get("source_sha256") == sources)
    reported_identity_matches = bool(
        all(str(value.get("slurm_job_id", "")) == job for value in runtime.values())
        and all(value.get(key) == manifest.get(key) and manifest.get(key) is not None
                for value in runtime.values()
                for key in ("frozen_directory", "plan_sha256", "model_revision", "model_manifest_sha256")))
    if "suite_manifest.json" in runtime:
        reported_identity_matches = bool(reported_identity_matches and
            retained.get("actual_frozen_source_sha256") == sources and retained.get("plan") == plan)
    identity_matches = bool(frozen_matches and reported_identity_matches and len(runtime) == 2)
    if not frozen_matches:
        failures.append("Frozen submission plan/source hashes disagree or cannot be verified")
    if not reported_identity_matches:
        failures.append("Existing suite reports have missing or different plan/source/model/job identities")
    suite_allocation = suite.get("allocation", {})
    nodes = suite_allocation.get("nodes", []) if isinstance(suite_allocation, dict) else []
    if not isinstance(nodes, list) or not nodes or not all(isinstance(node, str) for node in nodes):
        nodes = []
    allocation = manifest.get("allocation", {})
    expected_nodes = allocation.get("nodelist", "").split(",")
    if (not all(re.fullmatch(r"[A-Za-z0-9_.-]+", node) for node in expected_nodes) or
            len(expected_nodes) != allocation.get("nodes") or len(set(expected_nodes)) != len(expected_nodes)):
        expected_nodes = []
        failures.append("Frozen submission must declare its complete explicit host list")
    allocation_matches = nodes == expected_nodes if "allocation" in suite else None
    if allocation_matches is False:
        failures.append("Suite allocation does not match the submitted hosts")
    elif allocation_matches is None:
        gaps.append("Awaiting actual suite allocation")
    planned_cases = plan.get("cases", [])
    if not isinstance(planned_cases, list) or not all(isinstance(case, dict) for case in planned_cases):
        planned_cases = []
    by_id = {case.get("id"): case for case in planned_cases if isinstance(case.get("id"), str)}
    raw_cases = suite.get("cases", [])
    if not isinstance(raw_cases, list):
        failures.append("Suite cases must be a list")
        raw_cases = []
    actual_selections, optional_skips = [], set()
    for item in raw_cases:
        if not isinstance(item, dict):
            failures.append("Suite contains a malformed case item")
            continue
        planned = by_id.get(item.get("id"), {}) if isinstance(item.get("id"), str) else {}
        small = {key: item[key] for key in ("id", "sequence", "status", "selected_nodes",
                 "raw_artifact_directory", "returncode", "timed_out", "gate_failure", "started_utc", "finished_utc") if key in item}
        small.update(program=planned.get("program"), recorded_passed=item.get("passed"))
        if item.get("status") == "explicit_optional_program_absent":
            allowed = bool(planned.get("optional_if_missing") is True and
                           planned.get("program") == "e0_multi_repair.py")
            small.update(case_passed=None, optional_skip_authorized=allowed, reports=[])
            if allowed:
                optional_skips.add(item["id"])
            else:
                failures.append(f"Unauthorized optional skip: {item.get('id')}")
            cases.append(small)
            continue
        selected = item.get("selected_nodes", [])
        if not isinstance(selected, list) or not all(isinstance(node, str) for node in selected):
            selected = []
        actual_selections.append((item.get("id"), tuple(selected)))
        case_errors, case_failures, reports = [], [], []
        acceptance = None
        try:
            rawdir = Path(item["raw_artifact_directory"])
            if rawdir.resolve().parent != directory.resolve() or rawdir.is_symlink():
                raise ValueError("Case artifacts must be an immediate child of this suite result directory")
            acceptance = read_suite_object(rawdir / "case_acceptance.json", case_errors)
            for rank, host in enumerate(selected):
                raw = read_suite_object(rawdir / f"rank-{rank}.json", case_errors)
                if raw is None:
                    continue
                try:
                    compact = compact_validation_report(raw)
                except (AttributeError, KeyError, TypeError, ValueError) as error:
                    case_errors.append({"path": str(rawdir / f"rank-{rank}.json"),
                                        "error": f"{type(error).__name__}: {error}"})
                    continue
                compact["report_identity_matches"] = (raw.get("rank") == rank and
                    str(raw.get("slurm_job_id", "")) == job and
                    str(raw.get("hostname", "")).split(".")[0] == host.split(".")[0])
                reports.append(compact)
        except (KeyError, TypeError, OSError, ValueError) as error:
            case_errors.append({"path": item.get("raw_artifact_directory"), "error": f"{type(error).__name__}: {error}"})
        accepted = acceptance or {}
        checks = accepted.get("acceptance", {})
        if not isinstance(checks, dict):
            checks = {}
        if acceptance is None:
            case_failures.append("Missing or unreadable case_acceptance.json")
        elif acceptance != item:
            case_failures.append("Case acceptance and suite case record disagree")
        check_failures = checks.get("failures", [])
        if isinstance(check_failures, list):
            case_failures.extend(check_failures)
        else:
            case_failures.append("Malformed acceptance failure list")
        if checks.get("passed") is not True:
            case_failures.append("Driver acceptance is not passed")
        if item.get("status") != "passed" or item.get("passed") is not True:
            case_failures.append("Case has not reached passed status")
        if item.get("returncode") != 0 or item.get("timed_out") is not False:
            case_failures.append("Step exit/timing evidence is incomplete or unsuccessful")
        complete = bool(selected and len(reports) == len(selected))
        ranks_passed = complete and all(report.get("passed") is True and
                                       report["report_identity_matches"] for report in reports)
        if not ranks_passed:
            case_failures.append("A rank report is missing, failed, or has a different identity")
        recorded_errors = accepted.get("report_read_errors", item.get("report_read_errors", []))
        passed = bool(planned and acceptance is not None and acceptance == item and
                      item.get("status") == "passed" and item.get("passed") is True and
                      item.get("returncode") == 0 and item.get("timed_out") is False and
                      checks.get("passed") is True and not case_failures and not case_errors and
                      not recorded_errors and ranks_passed)
        small.update(case_passed=passed, acceptance=checks, check_failed_reasons=case_failures,
                     all_rank_reports_present=complete, reports=reports,
                     report_read_errors=case_errors, driver_report_read_errors=recorded_errors,
                     process_reports=accepted.get("process_reports", []))
        cases.append(small)
    expected_selections = []
    for case in planned_cases:
        if case.get("id") in optional_skips:
            continue
        selections = [[node] for node in expected_nodes] if case.get("scope") == "each_host" else [expected_nodes]
        expected_selections.extend((case.get("id"), tuple(selected)) for selected in selections)
    coverage_complete = bool(expected_selections and actual_selections == expected_selections)
    if actual_selections != expected_selections[:len(actual_selections)]:
        failures.append("Reported case/host selections differ from the frozen plan order")
    if not coverage_complete and suite.get("status") == "passed_selected_cases":
        failures.append("Not all frozen case/host selections have reports in the planned order")
    elif not coverage_complete:
        gaps.append("Awaiting remaining frozen case/host selections")
    all_cases_passed = bool(cases and all(case.get("case_passed") is True or
                                        case.get("optional_skip_authorized") is True for case in cases))
    passed = bool(suite.get("status") == "passed_selected_cases" and suite.get("passed") is True
                  and identity_matches and allocation_matches and coverage_complete and all_cases_passed
                  and not errors and not failures)
    terminal = suite.get("status") in ("failed", "passed_selected_cases")
    selected_pass = passed if terminal else None
    pending_without_report = (record.get("state", "").split(" ")[0] == "PENDING" and
                              "suite.json" not in runtime)
    status = suite.get("status", "pending_scheduler" if pending_without_report else "report_missing")
    if not suite and (errors or failures):
        status = "report_invalid"
    record.update(suite_passed_selected_cases=selected_pass,
                  source_snapshot=manifest.get("frozen_directory"))
    return {"job_id": job, "stage": record.get("stage", "E0_suite"), "submission_type": "e0_suite",
            "slurm_state": record.get("state"), "status": status,
            "suite_passed_selected_cases": selected_pass, "E0_complete": False,
            "E1_advanced": False, "performance_evidence": False,
            "gate_failure": suite.get("gate_failure"), "collection_failed_reasons": failures,
            "pending_evidence_gaps": gaps,
            "frozen_directory": manifest.get("frozen_directory"), "plan_sha256": manifest.get("plan_sha256"),
            "actual_plan_sha256": actual_plan_hash, "source_sha256": sources,
            "actual_frozen_source_sha256": retained.get("actual_frozen_source_sha256", {}),
            "frozen_identity_matches": identity_matches, "frozen_submission_identity_matches": frozen_matches,
            "allocation_matches": allocation_matches, "planned_nodes": expected_nodes,
            "model_revision": manifest.get("model_revision"), "model_manifest_sha256": manifest.get("model_manifest_sha256"),
            "evidence_job_ids": manifest.get("evidence_job_ids", []), "allocation": suite.get("allocation", {}),
            "planned_execution_count": len(expected_selections), "reported_execution_count": len(actual_selections),
            "case_coverage_complete": coverage_complete, "cases": cases, "report_read_errors": errors,
            "raw_artifact_directory": str(directory)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", action="append", default=[], help="job_id:stage_label")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    progress_file = root / "progress.json"
    progress = json.loads(progress_file.read_text())
    records = {entry["job_id"]: entry for entry in progress["cluster_jobs_submitted"]}
    for item in args.job:
        job, stage = item.split(":", 1)
        if not job.isdecimal():
            raise ValueError("Slurm job ID must contain only digits")
        records.setdefault(job, {"job_id": job, "stage": stage})
    output = subprocess.check_output([
        "sacct", "-j", ",".join(records), "--allocations", "--parsable2", "--noheader",
        "--format=JobID,State,ElapsedRaw,AllocTRES,NodeList,ExitCode"], text=True)
    for line in output.splitlines():
        job, state, elapsed, tres, nodes, code = line.split("|")[:6]
        if job not in records:
            continue
        gpus = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", tres)
        records[job].update(state=state, elapsed_seconds=int(elapsed),
            gpus=int(gpus.group(1)) if gpus else 0, allocation=tres, nodes=nodes, exit_code=code)
        records[job].pop("State", None)
    snapshots = {}
    for path in (root / "runs/submitted").glob("*/submission.json"):
        manifest = json.loads(path.read_text())
        if manifest.get("job_id") in records:
            snapshots[manifest["job_id"]] = manifest
    summaries, copy_summaries, suite_summaries, skipped_submissions = [], [], [], []
    for job, manifest in snapshots.items():
        if manifest.get("submission_type") == "e0_suite":
            suite_summaries.append(summarize_suite(root, job, manifest, records[job]))
            continue
        if is_copy_submission(manifest):
            copy_summaries.append(summarize_copy(root, job, manifest, records[job]))
            continue
        if not isinstance(manifest.get("allocation"), dict) or "nodes" not in manifest["allocation"] or "case_args" not in manifest:
            skipped_submissions.append({"job_id": job, "reason": "Unsupported submission schema; no validation result inferred"})
            continue
        directory = root / "results" / ("e0-validate-" + job)
        reports = []
        for path in sorted(directory.glob("*.json")):
            if not (re.fullmatch(r"rank-\d+\.json", path.name) or path.name == "request-isolation.json"):
                continue
            report = json.loads(path.read_text())
            reports.append(compact_validation_report(report))
        complete_reports = len(reports) == manifest["allocation"]["nodes"]
        normal_pass = complete_reports and all(report.get("passed") is True for report in reports)
        injected = "destination_after_first_commit" in manifest["case_args"]
        fault_pass = (injected and complete_reports and all(not report["passed"] and
            report.get("exception_cleanup", {}).get("passed", False) for report in reports)
            and any(report.get("error") == "RuntimeError: injected_destination_failure_after_first_commit"
                    for report in reports)
            and any(report.get("error", "").startswith("EOFError:") for report in reports))
        records[job].update(passed=normal_pass if complete_reports else None,
                            fault_cleanup_fixture_passed=fault_pass if injected else None,
                            source_snapshot=manifest["frozen_directory"])
        summaries.append({"job_id": job, "stage": records[job]["stage"],
            "slurm_state": records[job].get("state"), "program": manifest.get("program", "e0_repair.py"),
            "arguments": manifest["case_args"], "all_reports_present": complete_reports,
            "normal_execution_passed": normal_pass, "expected_failure_cleanup_passed": fault_pass,
            "reports": reports, "raw_artifact_directory": str(directory)})
    progress["cluster_jobs_submitted"] = list(records.values())
    progress["gpu_hours_consumed"] = sum(entry.get("elapsed_seconds", 0) * entry.get("gpus", 0)
        for entry in records.values()) / 3600
    progress["gpu_accounting_scope"] = "Slurm allocation time, including cancelled launch attempts and intentional fault tests; not kernel-active GPU time."
    progress["updated_utc"] = datetime.now(timezone.utc).isoformat()
    progress_file.write_text(json.dumps(progress, indent=2) + "\n")
    summary = {"updated_utc": progress["updated_utc"], "E0_complete": False,
               "performance_evidence": False, "stages": summaries,
               "model_copy_jobs": copy_summaries, "suite_jobs": suite_summaries,
               "skipped_submission_schemas": skipped_submissions}
    (root / "results/e0_validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"tracked_jobs": len(records), "validation_jobs": len(summaries),
                      "model_copy_jobs": len(copy_summaries), "suite_jobs": len(suite_summaries),
                      "gpu_allocation_hours": progress["gpu_hours_consumed"]}))


if __name__ == "__main__":
    main()
