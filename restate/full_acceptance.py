"""Independent, CPU-only audit of frozen E1 raw records on the destination.

The files contain strict numerical reports and SHA witnesses of logits, not the
full tensors. This gate checks those reports against frozen reviewed code; it
does not pretend to recompute tensor errors from their hashes. Rank-0 final
worker/network ownership must additionally be checked by the parent launcher.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re

REVISION = "97e1e76335b7017d8f67c08a19d103c0504298c9"
MODEL_MANIFEST = "0ef3c81db38c9d2fe14959dc6e8abb5780bbf826b264dbacbdbf80d586cc5e99"
LAYERS, VOCAB = 48, 152064
PUBLIC = {"prefix_tokens", "prefix_KV_bytes", "application_cap_bps", "fixed_suffix_tokens"}
LOCAL = {"recompute", "native_hf_control"}
HASH = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def finite(value, label, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            "Invalid finite " + label)
    return value


def exact(actual, expected, label):
    require(actual == expected, "Mismatch: " + label)


def tokens(value, count, label):
    require(isinstance(value, list) and len(value) == count and
            all(type(v) is int and 0 <= v < VOCAB for v in value), "Invalid tokens: " + label)


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


class Evidence:
    def __init__(self):
        self.hashes, self.stats = {}, {}

    def read(self, path, expected_sha=None, parent=None):
        path = Path(path)
        require(path.is_absolute(), "Evidence needs an absolute path")
        require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
                "Missing/symlinked evidence: " + str(path))
        if parent is not None:
            require(path.parent == Path(parent), "Evidence escapes its exact directory: " + str(path))
        stat = path.stat()
        require(stat.st_size <= 256 * 1024**2, "Unexpectedly large JSON: " + str(path))
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if expected_sha is not None:
            require(isinstance(expected_sha, str) and HASH.fullmatch(expected_sha), "Malformed expected hash")
            exact(sha, expected_sha, "artifact SHA " + str(path))
        self.hashes[str(path)], self.stats[str(path)] = sha, (stat.st_size, stat.st_mtime_ns)
        value = json.loads(raw, object_pairs_hook=no_duplicates,
            parse_constant=lambda text: (_ for _ in ()).throw(ValueError("Nonfinite JSON literal: " + text)))
        require(isinstance(value, dict), "Expected object evidence: " + str(path))
        return value

    def unchanged(self):
        for name, observed in self.stats.items():
            stat = Path(name).stat()
            exact((stat.st_size, stat.st_mtime_ns), observed, "evidence changed during audit")


def numerical(check, actual_tokens, reference_tokens, generated):
    require(all(check.get(k) is True for k in ("finite", "logits_pass", "tokens_match")), "Numerical flags failed")
    tokens(actual_tokens, generated, "candidate")
    tokens(reference_tokens, generated, "reference")
    exact(actual_tokens, reference_tokens, "greedy token IDs")
    for name in ("max_abs_error", "max_tolerance_ratio", "min_reference_top2_margin"):
        finite(check.get(name), name)
    errors = check.get("per_step_max_abs_error")
    require(isinstance(errors, list) and len(errors) == generated, "Missing per-token numerical errors")
    for value in errors: finite(value, "per-token error")
    exact(max(errors), check["max_abs_error"], "maximum numerical error")
    require(check["max_tolerance_ratio"] <= 1, "Original 1e-4 absolute/relative logit gate failed")


def timing(row, generated, timeout):
    times = row.get("token_delivery_s")
    require(isinstance(times, list) and len(times) == generated, "Missing token timestamps")
    names = ("accepted_s", "policy_started_s", "policy_finished_s", "batch_ready_s", "executor_entered_s")
    before = [finite(row.get(name), name) for name in names]
    after = [finite(row.get(name), name) for name in ("executor_settled_s", "batch_done_received_s", "settled_s")]
    for value in times: finite(value, "token delivery")
    ordered = before + times + after
    require(all(a <= b for a, b in zip(ordered, ordered[1:])), "Acceptance/setup/token/settlement order failed")
    exact(row["accepted_s"], row["policy_started_s"], "policy acceptance boundary")
    exact(row["batch_done_received_s"], row["settled_s"], "settlement includes received batch_done")
    require(row["settled_s"] - row["accepted_s"] <= timeout, "Trial deadline exceeded")
    for key, expected in {"ttft_s": times[0]-before[0], "completion_s": times[-1]-before[0],
            "settled_latency_s": after[-1]-before[0], "post_token_drain_s": after[-1]-times[-1],
            "policy_decision_s": before[2]-before[1], "request_setup_s": before[3]-before[2],
            "outer_setup_elapsed_s": before[4]-before[0]}.items():
        require(math.isclose(finite(row.get(key), key), expected, rel_tol=1e-12, abs_tol=1e-9), "Derived timing differs: " + key)
    exact(row.get("itl_s"), [b-a for a, b in zip(times, times[1:])], "ITL timestamps")
    require(row["ttft_s"] > 0 and row["completion_s"] > 0, "Zero foreground timing")


def route(row, case, diagnostic, layers=LAYERS):
    from campaign_client import validate_decode_state, state_signature
    state, calls = row["state_report"], row["remote_calls"]
    validate_decode_state(state, calls, case, layers)
    exact(state.get("logits_captured"), diagnostic, "main/diagnostic logits capture")
    require(row.get("background_drained") is True, "Request receiver survived settlement")
    if case["action"] == "remote":
        exact(len(calls), layers * (case["suffix"] + case["generated"] - 1), "persistent remote coverage")
    if case["action"] not in ("holder", "native_hf_control"):
        exact(row.get("cutovers"), state_signature(row, case, layers), "actual route signature")
    if case["action"] in ("native_fetch", "repair"):
        exact(row.get("committed_frontier"), layers, "settled full-prefix frontier")
        transfers = row.get("transfer_segments")
        require(isinstance(transfers, list) and len(transfers) == 1, "Expected one complete E1 prefix transfer")
        segment = transfers[0]
        for key, expected in {"kind": "migration_terminal", "start": 0, "end": layers,
                "next_layer": layers, "transfer_index": 1, "stopped": False,
                "payload_bytes": case["prefix"] * 196608}.items():
            exact(segment.get(key), expected, "prefix transfer " + key)
        ack = segment.get("receiver_drain_ack", {})
        require(ack.get("drained") is True, "Prefix receiver drain ACK missing")
        for key in ("start", "end", "next_layer", "transfer_index", "payload_bytes"):
            exact(ack.get(key), segment[key], "receiver drain ACK " + key)
    else:
        exact(row.get("transfer_segments"), [], "Unexpected KV migration")


def traffic(raw, row):
    source = row.get("source")
    holder = raw["holders"][0]["report"]
    require(holder.get("passed") is True and holder.get("alive_request_threads") == [], "Holder batch failed")
    channels = row["traffic"]
    if row["action"] in LOCAL:
        exact(channels, {"rpc": None, "bulk": None}, "local request channels")
        exact(holder.get("requests"), [], "local holder foreground work")
        require(holder.get("local_request") is True, "Local holder branch not observed")
        require(source is None and not row.get("channel_audits"), "Local action has unexpected source data")
    else:
        require(isinstance(source, dict) and holder.get("requests") == [source], "Source request artifact differs")
        require(source.get("passed") is True and not source.get("errors") and
                source.get("completed_frame_accounting_matches_actual_socket_bytes") is True, "Source frame gate failed")
        cleanup = source.get("cleanup", {})
        require(cleanup.get("migration_thread_alive") is False and all(cleanup.get(k) is True for k in
            ("sockets_closed", "listeners_closed", "delta_references_released")), "Source resource cleanup failed")
        used = set()
        for channel in ("rpc", "bulk"):
            left, right = channels[channel], source["traffic"][channel]
            require((left is None) == (right is None), "Peer channel presence differs")
            if left is None: continue
            used.add(channel)
            for key, peer in (("sent", "received"), ("received", "sent"),
                              ("payload_sent", "payload_received"), ("payload_received", "payload_sent")):
                require(type(left[key]) is int and left[key] >= 0 and left[key] == right[peer], "Peer socket/frame bytes differ")
            audit = row["channel_audits"][channel]
            require(audit.get("frame_accounting_matches_actual_socket_bytes") is True and audit.get("socket_closed") is True,
                    "Destination request socket cleanup/accounting failed")
            exact(audit["transport"]["sent_bytes"], left["sent"], "destination actual sent bytes")
            exact(audit["transport"]["received_bytes"], left["received"], "destination actual received bytes")
            for key in ("sent", "received"):
                exact(right[key], right["completed_frame_"+key], "source completed frame bytes")
                exact(right[key], source["transport"][channel][key+"_bytes"], "source socket bytes")
        exact(set(row["channel_audits"]), used, "audited channel set")
        exact(source["request_id"], row["request_id"], "source request ID")
        exact(source["remote_calls"], len(row["remote_calls"]), "source RPC count")
        exact([[r["layer"], r["position"]] for r in source["remote_audit"]], row["remote_calls"], "source RPC sequence")
        if row["action"] == "holder":
            exact(source["holder_token_ids"], row["token_ids"], "holder streamed tokens")
            exact(source["decode_state"], row["state_report"], "holder decode state")
    require(row.get("traffic_acceptance", {}).get("passed") is True, "Request traffic gate missing")
    control = raw["outer_control_acceptance"]
    require(control.get("passed") is True and control.get("physical_wire_bytes") is None, "Control-byte acceptance missing")
    left, right = control["destination"], control["holder"]
    for a, b in (("sent", "received"), ("received", "sent"), ("sent_frames", "received_frames"), ("received_frames", "sent_frames")):
        require(type(left[a]) is int and left[a] >= 0 and left[a] == right[b], "Control frame peer mismatch")
    totals = []
    for snapshot, expected_control in ((raw["coordinator"], left), (raw["source_final_coordinator"], right)):
        limiter = snapshot["limiter"]
        conns = limiter["connections"]
        require("outer-control" in conns, "Request budget omitted outer control")
        for direction in ("sent", "received"):
            exact(conns["outer-control"][direction+"_bytes"], expected_control[direction], "control actual bytes")
        total = {key: sum(finite(c[key], "connection "+key) for c in conns.values()) for key in ("sent_bytes", "received_bytes")}
        for key in total: exact(limiter["total"][key], total[key], "aggregate socket bytes")
        totals.append(total)
    exact(totals[0]["sent_bytes"], totals[1]["received_bytes"], "all-source received bytes")
    exact(totals[0]["received_bytes"], totals[1]["sent_bytes"], "all-source sent bytes")
    aggregate = raw["all_application_bytes_acceptance"]
    require(aggregate.get("passed") is True and aggregate.get("physical_wire_bytes") is None, "Aggregate byte gate failed")
    exact(aggregate["destination"], totals[0], "aggregate destination evidence")
    exact(aggregate["holder"], totals[1], "aggregate holder evidence")
    require(raw["coordinator_after_close"].get("closed") is True, "Trial coordinator not closed")


def queue_gate(queue):
    require(queue.get("backend") == "cuda_stream_completion" and queue.get("fifo") is True and
            queue.get("active_gpu_operations") == queue.get("queued_gpu_operations") == 0,
            "Compute queue is not real/completed FIFO")
    previous_ticket, previous_end = -1, 0
    for op in queue["operations"]:
        require(type(op["ticket"]) is int and op["ticket"] > previous_ticket, "Duplicate/unordered compute ticket")
        require(op.get("status") in ("completed", "yielded_for_io") and op.get("error") is None and
                op.get("completion_backend") == "cuda_stream_completion", "Compute failure or fixture backend")
        a, b, c = [finite(op.get(k), k) for k in ("arrival_monotonic_s", "start_monotonic_s", "end_monotonic_s")]
        require(a <= b <= c and previous_end <= b, "Overlapping or non-FIFO completed compute intervals")
        finite(op.get("service_s"), "completion-bounded service")
        require(math.isclose(finite(op.get("queue_wait_s"), "queue wait"), b-a, abs_tol=1e-9), "Queue wait mismatch")
        previous_ticket, previous_end = op["ticket"], c


def background(report, required, seed, calibration):
    if not required:
        require(report is None, "Unexpected background location")
        return
    require(isinstance(report, dict) and report.get("passed") is True and report.get("backend") == "cuda_stream_completion",
            "Background did not execute with real CUDA completion")
    require(report.get("failed") == report.get("pending") == 0 and report.get("threads_alive") == [], "Background failed or has live work")
    require(all(report.get("cleanup", {}).get(k) is True for k in
        ("producer_exited", "worker_exited", "all_accepted_jobs_completed")), "Background cleanup not complete")
    exact(report["seed"], seed, "paired load seed")
    exact(report["rho"], .5, "nominal load")
    mean = finite(calibration["mean_service_s"], "calibrated service")
    require(mean > 0, "Zero calibration service")
    exact(report["lambda_jobs_per_s"], .5/mean, "actual arrival rate")
    exact(report["lead_in_s"], 5*mean, "5C lead-in")
    for field in ("started_monotonic_s", "lead_in_finished_monotonic_s", "acceptance_monotonic_s", "stop_requested_monotonic_s", "drained_monotonic_s"):
        finite(report.get(field), field)
    require(report["started_monotonic_s"] <= report["lead_in_finished_monotonic_s"] <= report["acceptance_monotonic_s"] <=
            report["stop_requested_monotonic_s"] <= report["drained_monotonic_s"], "Background local-clock order")
    exact(report["arrivals"], len(report["records"]), "arrival denominator")
    exact(report["completed"], report["arrivals"], "background completion denominator")
    rng, schedule = random.Random(seed), report["started_monotonic_s"]
    for index, record in enumerate(report["records"]):
        schedule += rng.expovariate(report["lambda_jobs_per_s"])
        exact(record["scheduled_monotonic_s"], schedule, "deterministic Poisson arrivals")
        exact(record["job_id"], "background-job-"+str(index), "background job identity")
        require(record.get("status") == "completed" and record.get("error") is None, "Background request failed")
        require(record["arrival_monotonic_s"] >= schedule and record["start_monotonic_s"] >= record["arrival_monotonic_s"] and
                record["end_monotonic_s"] >= record["start_monotonic_s"], "Background request timing order")
        require(record["gpu_admissions"] > 0 and finite(record["gpu_service_s"], "background service") > 0, "Missing background service")
        state = record["result"]["state_report"]
        require(state.get("owned_delta_references_released") is True and state.get("logits_captured") is False,
                "Background delta/capture gate failed")
        exact(state["mode_counts"], {"local": LAYERS*47, "remote": 0}, "P1024/S16/G32 background work")
        exact(state["delta_lengths"], [[47,47]]*LAYERS, "background delta")
        exact(state["final_position"], 1071, "background final position")
        tokens(record["result"]["token_ids"], 32, "background")
        ops = [op for op in report["operations"] if op["owner"] == record["job_id"]]
        exact(len(ops), record["gpu_admissions"], "background actual admissions")
        require(math.isclose(sum(op["service_s"] for op in ops), record["gpu_service_s"], rel_tol=1e-12, abs_tol=1e-9), "Background service records differ")


def network(snapshot, delay, hostname):
    exact(snapshot.get("hostname"), hostname, "netem physical host")
    require(snapshot.get("eth0_root_unchanged") is True and snapshot.get("owned_filter_count") == 8,
            "Scoped netem ownership not validated")
    exact(snapshot.get("requested_added_rtt_ms"), delay, "requested network treatment")
    for key in ("applied_one_way_delay_ms", "observed_one_way_delay_ms"):
        require(math.isclose(finite(snapshot.get(key), key), delay/2, abs_tol=1e-6), "Observed netem delay differs")


def logits_witness(rows, generated):
    require(isinstance(rows, list) and bool(rows), "Missing full-vocabulary witnesses")
    count = 0
    for row in rows:
        shape = row.get("shape")
        require(isinstance(shape, list) and len(shape) == 2 and type(shape[0]) is int and shape[0] > 0 and
                shape[1] == VOCAB and row.get("dtype") == "torch.float32" and row.get("bytes") == shape[0]*4*VOCAB,
                "Logit witness is not complete FP32 vocabularies")
        count += shape[0]
        require(isinstance(row.get("sha256"), str) and HASH.fullmatch(row["sha256"]), "Malformed logit witness SHA")
    exact(count, generated, "logit witness token rows (holder may send one G-by-vocab matrix)")


def trial_evidence(raw, trial, context, context_sha, reference, inputs, calibrations, diagnostic):
    for key, value in {"trial_id": trial["trial_id"], "trial": trial, "source_run_id": context["run_id"],
            "source_sha256": context["source_sha256"], "context_sha256": context_sha,
            "protocol_sha256": context["protocol_sha256"], "diagnostic": diagnostic}.items():
        exact(raw.get(key), value, "raw identity "+key)
    require(raw.get("passed") is True and raw.get("correctness_passed") is True, "Raw trial failed")
    exact(len(raw.get("requests", [])), 1, "request count")
    exact(len(raw.get("holders", [])), 1, "holder count")
    exact(raw["holders"][0]["holder"], context["hosts"][0], "holder host")
    row = raw["requests"][0]
    action = row["action"]
    from plan import POLICY
    require(action in ("recompute", "native_fetch") if trial["method"] == POLICY else action == trial["method"], "Planned action differs")
    p, g = trial["public_features"]["prefix_tokens"], trial["executor_only"]["realized_stop_tokens"]
    expected_case = {"id": trial["trial_id"], "input_id": trial["input_id"], "seed": trial["seed"],
        "prefix": p, "suffix": 16, "generated": g, "action": action, "prefix_key": trial["prefix_key"], "holder": 0}
    exact(row["case"], expected_case, "execution geometry")
    exact(row["input_ids"], inputs["token_ids"], "frozen input arrays")
    exact(row["input_ids_sha256"], digest(row["input_ids"]), "input token SHA")
    tokens(row["input_ids"], p+16, "request inputs")
    require(row.get("passed") is True and row.get("correctness_passed") is True, "Request correctness failed")
    numerical(row["numerical_check"], row["token_ids"], reference["token_ids"], g)
    exact(row["numerical_check"], reference["numerical_check"], "reference numerical report")
    timing(row, g, context["trial_timeout_s"])
    route(row, expected_case, diagnostic)
    traffic(raw, row)
    require(all(raw.get("cleanup", {}).get(k) is True for k in ("request_channels_closed", "migration_drained",
        "backgrounds_drained", "queues_idle", "trial_control_disarmed")), "Raw cleanup incomplete")
    require(not raw.get("cleanup_errors"), "Cleanup errors were reported")
    queue_gate(raw["source_queue_events"]); queue_gate(raw["destination_queue_events"])
    env, treatment = raw["environment"], trial["treatment"]
    arm = env["arm"]
    exact(arm["trial_id"], trial["trial_id"] + (".diagnostic" if diagnostic else ".candidate"), "arm identity")
    exact(arm["rate_bps"], trial["public_features"]["application_cap_bps"], "application cap")
    exact(arm["load_location"], treatment["load_location"], "load location")
    seed = int(trial["load_trace_key"][:16], 16)
    exact(arm["load_seed"], seed, "paired load seed")
    exact(env["destination_acceptance_s"], row["accepted_s"], "acceptance clock")
    for snapshot in (raw["coordinator"], raw["source_final_coordinator"]):
        exact(snapshot["limiter"]["rate_bits_per_second"], arm["rate_bps"], "actual pacing rate")
    for loc in ("destination", "holder"):
        background(raw["background_"+loc], treatment["load_location"] == loc, seed, calibrations[loc])
    network(env["destination_network_before"], treatment["added_path_delay_ms"], context["hosts"][1])
    network(env["source_network_after"], treatment["added_path_delay_ms"], context["hosts"][0])
    if diagnostic:
        logits_witness(raw["diagnostic_logits"], g)
        exact(raw["numerical_check"], reference["numerical_check"], "diagnostic numerical report")
    else:
        require("diagnostic_logits" not in raw and "logits" not in raw and "logits" not in row, "Main interval retained full logits")
    return row


def coverage(wrapper, manifest, records):
    from full_protocol import validate_wrapper, smoke_trials
    validate_wrapper(wrapper)
    trials = wrapper["full_protocol"]["trials"] if wrapper["mode"] == "e1_full" else smoke_trials(wrapper["full_protocol"])
    expected = 2826 if wrapper["mode"] == "e1_full" else 32
    exact(len(trials), expected, "frozen exact coverage")
    ids = [t["trial_id"] for t in trials]
    require(len(set(ids)) == expected, "Duplicate planned trial IDs")
    exact([r["trial_id"] for r in manifest], ids, "all raw manifest trials/order including warmups")
    exact([r["trial_id"] for r in records], ids, "all normalized trials/order including warmups")
    if wrapper["mode"] == "e1_full":
        exact(Counter((t["phase"],t["sample_role"]) for t in trials), Counter({("training","measured"):216,
            ("training","warmup"):108,("calibration","measured"):72,("calibration","warmup"):36,
            ("test","measured"):2128,("test","warmup"):266}), "full split counts")
        require(not wrapper.get("smoke_trials") and not wrapper.get("smoke_fixture_fit"), "Smoke substitution in full run")
    return trials


def audit(context_path):
    evidence = Evidence()
    context_path = Path(context_path).absolute()
    context = evidence.read(context_path)
    context_sha = evidence.hashes[str(context_path)]
    wrapper = evidence.read(context["plan_path"], context["plan_sha256"])
    frozen = evidence.read(Path(context["snapshot_dir"])/"frozen.json")
    exact(frozen["source_sha256"], context["source_sha256"], "frozen/context source map")
    for name, expected in frozen["files_sha256"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe frozen source path")
        path = Path(context["snapshot_dir"])/relative
        require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)), "Missing/symlinked frozen source")
        exact(hashlib.sha256(path.read_bytes()).hexdigest(), expected, "frozen closure file "+name)
    require(context.get("stage") == "E1" and context.get("scheduler") == "direct" and context.get("topology") == "two_physical_hosts"
            and context.get("mode") == wrapper["mode"] and context.get("world_size") == 2 and context.get("dtype") == "bfloat16",
            "Wrong direct E1 cohort")
    exact(context["model_revision"], REVISION, "fixed 14B revision")
    exact(context["model_manifest_sha256"], MODEL_MANIFEST, "fixed model manifest")
    context["trial_timeout_s"] = wrapper["limits"]["trial_timeout_s"]
    context["protocol_sha256"] = wrapper["full_protocol"]["plan_sha256"]
    for key in ("hostname", "boot_id", "dmi_uuid", "gpu_uuid", "internal_ip"):
        require(context["ranks"][0][key] != context["ranks"][1][key], "Two-host physical identity is not distinct: "+key)
    runtime = Path(context["runtime_dir"])
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted({*runtime.glob("*.py"), *runtime.glob("*_protocol.json")})
              if not p.is_symlink()}
    exact(actual, context["source_sha256"], "complete frozen runtime closure")
    exact(hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), actual["full_acceptance.py"], "executing auditor source")
    from e0_holder_decode import ATOL, RTOL
    require(ATOL == RTOL == 1e-4, "Published numerical tolerance changed")
    model_manifest = evidence.read(Path(context["model_path"])/"e0_model_manifest.json", MODEL_MANIFEST)
    exact(model_manifest["revision"], REVISION, "actual local model manifest revision")
    directory = Path(context["result_dir"])
    report = evidence.read(directory/"rank-1.json", parent=directory)
    worker = evidence.read(directory/"worker-1.json", parent=directory)
    for key in ("run_id", "stage", "mode", "scheduler", "topology", "cluster_id", "plan_sha256", "source_sha256"):
        exact(report.get(key), context[key], "rank identity "+key)
    for item in (report, worker):
        require(item.get("passed") is True and item.get("rank") == 1 and item.get("hostname") == context["hosts"][1]
                and item.get("run_id") == context["run_id"] and item.get("context_sha256") == context_sha, "Local rank/worker identity failed")
    require(worker.get("exit_code") == 0 and worker.get("owned_child_exited") is True and not worker.get("cleanup_error"), "Worker did not exit cleanly")
    expected_rank = context["ranks"][1]
    exact(report["physical_identity"], {k: expected_rank[k] for k in ("hostname","boot_id","dmi_uuid","internal_ip")}, "actual physical identity")
    for key in ("hostname","boot_id","dmi_uuid","internal_ip","gpu_uuid"):
        exact(worker["preflight"][key], expected_rank[key], "worker observed "+key)
    require(report.get("gpu_uuid") == expected_rank["gpu_uuid"] and report.get("device_count") == 1 and report.get("native_bfloat16") is True,
            "Actual selected GPU differs")
    exact(report.get("gpu"), expected_rank["gpu_name"], "observed GPU model")
    identity = report["identity"]
    for key in ("model_revision", "model_manifest_sha256", "source_sha256", "plan_sha256", "run_id"):
        exact(identity[key], context[key], "model/runtime identity "+key)
    require(identity.get("TP") == 1 and identity.get("layers") == LAYERS and identity.get("dtype") == "bfloat16", "Model execution geometry differs")
    flags = identity["backend_flags"]
    require(flags.get("attention") == "eager" and flags.get("float32_matmul_precision") == "highest" and all(flags.get(k) is False for k in
        ("matmul_allow_tf32","cudnn_allow_tf32","allow_bf16_reduced_precision_reduction","allow_fp16_reduced_precision_reduction")), "Numerical backend changed")
    peer = report["peer_identity_acceptance"]
    require(peer.get("passed") is True and peer.get("rank") == 0 and peer.get("hostname") == context["hosts"][0]
            and peer.get("gpu_uuid") == context["ranks"][0]["gpu_uuid"], "Physical/model peer handshake failed")
    exact(peer.get("physical_identity"), {k:context["ranks"][0][k] for k in ("hostname","boot_id","dmi_uuid","internal_ip")}, "peer observed physical identity")
    require(all(report.get("cleanup",{}).get(k) is True for k in ("all_controls_closed","listener_closed")) and not report["cleanup"].get("errors"), "Rank sockets not closed")
    require(report.get("network_cleanup",{}).get("restored") is True and report["network_cleanup"].get("eth0_root_unchanged") is True
            and not report["network_cleanup"].get("errors"), "Network restoration failed")
    from scoped_netem import config_only, validate_plan as validate_network_plan
    journal = evidence.read(directory/"netem-rank-1.json", parent=directory)
    validate_network_plan(journal["plan"])
    require(journal.get("status") == "cleaned", "Network journal is not cleaned")
    exact(journal["plan"]["run_id"], context["run_id"], "network journal run")
    exact(journal["plan"]["local_ip"], context["ranks"][1]["internal_ip"], "network local address")
    exact(journal["plan"]["peer_ip"], context["ranks"][0]["internal_ip"], "network peer address")
    exact(journal["plan_sha256"], digest(journal["plan"]), "network plan SHA")
    exact(journal["source_sha256"], actual["scoped_netem.py"], "network implementation SHA")
    exact(journal["cleanup"], report["network_cleanup"], "observed final network cleanup")
    exact(config_only(journal["cleanup"]["after"]), config_only(journal["before"]), "network baseline restoration")
    require(report.get("full_background_cleanup",{}).get("prefix_references_released") is True and
            report["full_background_cleanup"].get("background_threads_exited") is True, "Final destination background cleanup missing")
    require(len(report.get("holder_shutdown",[])) == 1 and report["holder_shutdown"][0].get("shared_prefix_references_released") is True,
            "Holder prefix shutdown not acknowledged")
    require(report.get("candidate_session_passed") is True and report.get("E1_complete") is False and report.get("paper_stage_complete") is False,
            "Session status/limited completion scope differs")
    full, protocol = report["full_session"], wrapper["full_protocol"]
    exact(full["mode"], wrapper["mode"], "session mode")
    exact(full["protocol_sha256"], protocol["plan_sha256"], "session protocol")
    require(full.get("pending_profiles") == 30 and full.get("E1_complete") is False, "Profile gate was skipped")
    records_obj = evidence.read(full["records_artifact"], full["records_sha256"], directory)
    manifest_obj = evidence.read(full["measurements_manifest_artifact"], full["measurements_manifest_sha256"], directory)
    for obj in (records_obj, manifest_obj): exact(obj["protocol_sha256"], protocol["plan_sha256"], "raw index protocol")
    records, manifest = records_obj["records"], manifest_obj["candidates"]
    trials = coverage(wrapper, manifest, records)
    exact(full["expected_candidates"], len(trials), "reported expected coverage")
    exact(full["completed_candidates"], len(trials), "reported completed coverage")
    arrays = evidence.read(report["frozen_inputs"]["path"], report["frozen_inputs"]["sha256"], directory)
    require(arrays.get("before_any_request_timing") is True, "Inputs were not frozen before requests")
    for key, expected in (("protocol_sha256",protocol["plan_sha256"]),("model_revision",REVISION),("model_manifest_sha256",MODEL_MANIFEST)):
        exact(arrays.get(key), expected, "frozen input bank "+key)
    tokens(arrays["background_token_ids"], 1040, "background input bank")
    exact(arrays["background_token_ids_sha256"], digest(arrays["background_token_ids"]), "background input bank SHA")
    calibration = evidence.read(directory/"background-calibration.json", parent=directory)
    calibrations = {"destination": calibration["destination"], "holder": calibration["holder"]["calibration"]}
    for cal in calibrations.values():
        require(cal.get("GPU_service_evidence") is True and cal.get("backend") == "cuda_stream_completion" and
                cal.get("warmup_jobs") == 2 and cal.get("measured_jobs") == 10 and len(cal["records"]) == 12, "Background calibration incomplete")
        exact([r["phase"] for r in cal["records"]], ["warmup"]*2+["calibration"]*10, "background calibration split")
        require(all(r["status"] == "completed" and r["gpu_admissions"] > 0 and r["gpu_service_s"] > 0 for r in cal["records"]), "Background calibration contains failed/no-work jobs")
        exact(cal["mean_service_s"], sum(r["gpu_service_s"] for r in cal["records"][2:])/10, "isolated measured service mean")
        for record in cal["records"]:
            ops = [o for o in cal["operations"] if o["owner"] == record["job_id"]]
            exact(len(ops), record["gpu_admissions"], "calibration actual admissions")
            require(all(o.get("completion_backend") == "cuda_stream_completion" and o.get("status") == "completed" for o in ops), "Non-CUDA/failed calibration operation")
            require(math.isclose(sum(o["service_s"] for o in ops),record["gpu_service_s"],rel_tol=1e-12,abs_tol=1e-9), "Calibrated service differs from completed operations")
            state=record["result"]["state_report"]
            require(state.get("owned_delta_references_released") is True and state.get("logits_captured") is False, "Calibration request cache/capture failed")
            exact(state["mode_counts"], {"local":LAYERS*47,"remote":0}, "Actual calibration request work")
        require({o["owner"] for o in cal["operations"]} == {r["job_id"] for r in cal["records"]}, "Calibration was not isolated")
    import analysis
    from plan import IDENTITY_FIELDS, POLICY
    selected_actions, numerical_max = Counter(), 0
    for trial, entry, normalized in zip(trials, manifest, records):
        require(set(trial["public_features"]) == PUBLIC, "Policy feature leakage")
        folder = directory/"trials"/trial["trial_id"]
        loaded = {}
        for label, pathkey, hashkey, filename in (("candidate","raw_measured_path","raw_measured_sha256","candidate.json"),
                ("diagnostic","diagnostic_path","diagnostic_sha256","diagnostic.json"),("reference","reference_path","reference_sha256","reference.json"),
                ("normalized","normalized_record_path","normalized_record_sha256","normalized-record.json")):
            exact(Path(entry[pathkey]), folder/filename, "exact trial artifact path")
            loaded[label] = evidence.read(entry[pathkey], entry[hashkey], folder)
        exact(loaded["normalized"], normalized, "normalized raw/index copy")
        ref = loaded["reference"]
        require(ref.get("passed") is True and ref.get("untimed_same_shape_reference") is True and ref.get("trial_id") == trial["trial_id"], "Reference failed/identity differs")
        for key, expected in (("source_run_id",context["run_id"]),("source_sha256",context["source_sha256"]),
                ("model_revision",REVISION),("model_manifest_sha256",MODEL_MANIFEST)):
            exact(ref.get(key), expected, "reference "+key)
        inputkey = digest({k: trial[k] for k in ("input_id","input_family_id","seed","prefix_key")})
        inputs = arrays["arrays"][inputkey]
        exact(inputs["token_ids_sha256"], digest(inputs["token_ids"]), "input bank SHA")
        exact(ref["input_ids_sha256"], inputs["token_ids_sha256"], "reference input identity")
        g = trial["executor_only"]["realized_stop_tokens"]
        logits_witness(ref["logits"], g)
        refstate = ref["state_report"]
        require(refstate.get("owned_cache_references_released") is True and refstate.get("logits_captured") is True, "HF reference cache cleanup/capture missing")
        end = trial["public_features"]["prefix_tokens"]+16+g-1
        exact(refstate["final_cache_lengths"], [end]*LAYERS, "reference final KV length")
        exact(refstate["final_position"], end, "reference final position")
        rows = [trial_evidence(loaded[name], trial, context, context_sha, ref, inputs, calibrations, name=="diagnostic") for name in ("candidate","diagnostic")]
        a,b = rows
        for key in ("token_ids","input_ids","input_ids_sha256","cutovers","remote_calls","action"):
            exact(a.get(key), b.get(key), "candidate/exact-route replay "+key)
        exact(normalized.get("status"), "passed", "normalized status")
        exact(normalized.get("attempt_id"), context["run_id"]+":"+trial["trial_id"], "single authoritative attempt")
        exact(normalized.get("plan_sha256"), protocol["plan_sha256"], "normalized protocol identity")
        for key in IDENTITY_FIELDS: exact(normalized[key], trial[key], "normalized planned identity")
        exact(normalized["public_features"], trial["public_features"], "normalized public features")
        exact(normalized["executed_action"], a["action"], "actual policy/action dispatch")
        require(all(normalized.get(key) is True for key in analysis.GATES), "Normalized gates failed")
        for key, rowkey in (("ttft_s","ttft_s"),("completion_s","completion_s"),("settled_s","settled_latency_s")):
            exact(normalized["metrics"][key], a[rowkey], "raw-derived normalized metric")
        provenance = normalized["provenance"]
        for key, expected in (("measurement_sha256",entry["raw_measured_sha256"]),("diagnostic_sha256",entry["diagnostic_sha256"]),
                ("reference_sha256",entry["reference_sha256"]),("runtime_source_sha256",context["source_sha256"]),
                ("cohort_id",context["cluster_id"]),
                ("model_revision",REVISION),("model_manifest_sha256",MODEL_MANIFEST)):
            exact(provenance[key], expected, "normalized raw provenance")
        selected_actions[a["action"]] += 1
        numerical_max = max(numerical_max, a["numerical_check"]["max_tolerance_ratio"])
    if wrapper["mode"] == "e1_full":
        phases = {phase: [r for r in records if r["sample_role"]=="measured" and r["phase"]==phase] for phase in ("training","calibration","test")}
        fit = evidence.read(full["training_fit_path"], full["training_fit_sha256"], directory)
        cal = evidence.read(full["calibration_path"], full["calibration_sha256"], directory)
        result = evidence.read(full["analysis_path"], full["analysis_sha256"], directory)
        exact(fit, analysis.fit_nominal(phases["training"], protocol), "independently refitted training evidence")
        exact(cal, analysis.calibrate_nominal(phases["calibration"], fit, protocol), "independent held-out calibration")
        exact(result, analysis.analyze_test(phases["test"], fit, protocol, cal), "independent full held-out analysis")
        require(result["all_planned_records_passed"] is True, "Full held-out coverage failed")
    else:
        fit = wrapper["smoke_fixture_fit"]["artifact"]
        require(wrapper["smoke_fixture_fit"].get("CPU_fixture_only") is True and fit.get("smoke_only") is True,
                "Untrained smoke fixture must be explicitly labeled")
        for key in ("training_fit_path","calibration_path","analysis_path"):
            require(full.get(key) is None, "Smoke produced scientific fitting/analysis artifacts")
        choices = [r for r in records if r["method"] == POLICY]
        exact(len(choices), 2, "actual policy smoke coverage")
        exact({r["executed_action"] for r in choices}, {"recompute","native_fetch"}, "both actual threshold branches")
    for record in records:
        if record["method"] == POLICY:
            exact(record.get("policy_artifact_sha256"), fit["artifact_sha256"], "policy bound training artifact")
            exact(record["executed_action"], analysis.choose_action(fit, record["public_features"]), "no-future-feature policy choice")
    evidence.unchanged()
    return {"schema_version":1,"passed":True,"mode":context["mode"],"run_id":context["run_id"],
        "result_dir":str(directory),"rank1_report_sha256":evidence.hashes[str(directory/"rank-1.json")],
        "worker1_sha256":evidence.hashes[str(directory/"worker-1.json")],
        "network_journal_sha256":evidence.hashes[str(directory/"netem-rank-1.json")],
        "context_sha256":context_sha,"protocol_sha256":protocol["plan_sha256"],"source_sha256":context["source_sha256"],
        "ranks":context["ranks"],"model_revision":REVISION,"model_manifest_sha256":MODEL_MANIFEST,
        "candidate_count":len(trials),"diagnostic_count":len(trials),"reference_count":len(trials),
        "measured_count":sum(t["sample_role"]=="measured" for t in trials),
        "warmup_count":sum(t["sample_role"]=="warmup" for t in trials),"executed_actions":dict(selected_actions),
        "max_reported_tolerance_ratio":numerical_max,"records_sha256":full["records_sha256"],
        "raw_manifest_sha256":full["measurements_manifest_sha256"],"audited_artifact_count":len(evidence.hashes),
        "audited_artifact_hashes_sha256":digest(evidence.hashes),"E1_complete":False,"paper_stage_complete":False,
        "pending_profiles":30,"rank0_final_worker_and_network_cleanup_checked_here":False,
        "evidence_scope":"Every raw trial/hash, exact-route numerical report and logit SHA witness, actual application-byte counters, local rank/worker and cleanup; full tensor logits are not persisted or recomputed by this audit",
        "remaining_gates":["parent validates rank-0 final ownership/model/network cleanup","30 separately accepted instrumented profiles"],
        "auditor_sha256":actual["full_acceptance.py"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.context)
        context = json.loads(args.context.read_text())
        destination = Path(context["result_dir"])/"full_raw_acceptance.json"
        if destination.exists():
            exact(json.loads(destination.read_text()), result, "immutable previous independent audit")
        else:
            with destination.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except BaseException as error:
        print(json.dumps({"passed":False,"E1_complete":False,"error":f"{type(error).__name__}: {error}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
