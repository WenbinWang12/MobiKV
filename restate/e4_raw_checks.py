"""Independent CPU validation of actual E4 reports before normalization.

No tensors are reconstructed from SHA witnesses. Original reviewed numerical
reports, complete routes, per-layer readiness, real socket counters, FIFO
records and ownership evidence are checked together and bound to actual JSON
files. Parent acceptance must additionally verify the frozen context/source
files and both final worker/network-ownership reports.
"""
from __future__ import annotations
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import random

if __package__:
    from . import analysis, transfer_only
    from .plan import digest, require, validate_plan
    from .runtime import full_acceptance as common
else:
    import e4_analysis as analysis
    import e4_transfer_only as transfer_only
    from e4_plan import digest, require, validate_plan
    import full_acceptance as common

LAYERS, VOCAB = 48, 152064
ROLES = ("holder", "destination")


def exact(actual, expected, label):
    require(actual == expected, "E4 " + label + " differs")


def finite(value, label):
    return common.finite(value, "E4 " + label)


def ordered(values, label):
    for value in values: finite(value, label)
    require(values == sorted(values), "E4 native-clock order: " + label)


def integer(value, label):
    analysis._integer(value, label)
    return value


def close(actual, expected, label):
    require(math.isclose(finite(actual, label), expected, rel_tol=1e-12, abs_tol=1e-9), "E4 derived " + label + " differs")


def service_duration(operation, label):
    """Verify the actual recorded service clock, separately from FIFO epochs."""
    exact(operation.get("service_duration_clock"), "perf_counter", label + " clock")
    start = finite(operation.get("service_start_perf_counter_s"), label + " perf start")
    end = finite(operation.get("service_end_perf_counter_s"), label + " perf end")
    require(start <= end, "E4 service clock order: " + label)
    exact(finite(operation.get("service_s"), label), end - start, "derived " + label)


def _context(context, protocol):
    exact(context.get("stage"), "E4", "context stage")
    exact(context.get("mode"), "e4_profile" if protocol["mode"] == "e4_profiles" else protocol["mode"], "context mode")
    exact(context.get("scheduler"), "direct", "scheduler")
    exact(context.get("topology"), "two_physical_hosts", "physical topology")
    exact(context.get("dtype"), "bfloat16", "actual precision cohort")
    ranks = context["ranks"]
    require(len(ranks) == 2 and [r["rank"] for r in ranks] == [0, 1], "Two real endpoint ranks required")
    for field in ("hostname", "boot_id", "dmi_uuid", "internal_ip", "gpu_uuid"):
        require(all(isinstance(r[field], str) and r[field] for r in ranks) and ranks[0][field] != ranks[1][field], "Distinct physical " + field + " required")
    for field in ("model_revision", "model_manifest_sha256"):
        exact(context[field], protocol["cohort"][field], field)
    require(isinstance(context.get("source_sha256"), dict) and context["source_sha256"], "Frozen source closure required")
    for value in context["source_sha256"].values(): analysis._sha(value, "Source SHA")
    return digest({key: context[key] for key in ("cluster_id", "scheduler", "topology", "ranks", "dtype", "model_revision", "model_manifest_sha256")})


def identity(raw, trial, protocol, context, *, diagnostic, reference_role=None):
    exact(raw.get("stage"), "E4", "raw stage")
    exact(raw.get("trial_id"), trial["trial_id"], "raw trial")
    exact(raw.get("protocol_sha256"), protocol["plan_sha256"], "raw active protocol")
    exact(raw.get("source_run_id"), context["run_id"], "raw run")
    exact(raw.get("source_sha256"), context["source_sha256"], "raw source closure")
    for field in ("model_revision", "model_manifest_sha256"):
        exact(raw.get(field), context[field], "raw " + field)
    analysis._sha(raw.get("context_sha256"), "Actual context file SHA")
    require(raw.get("passed") is True, "Raw execution failed")
    if reference_role is None:
        exact(raw.get("trial"), trial, "actual frozen trial")
        exact(raw.get("trial_sha256"), digest(trial), "actual trial SHA")
        exact(raw.get("diagnostic"), diagnostic, "main/diagnostic scope")
        require(raw.get("CPU_component_only") is False and not raw.get("errors"), "CPU fixture/errors cannot certify GPU execution")
        require(raw.get("E4_complete") is False and raw.get("physical_wire_bytes") is None, "Invalid stage/physical-wire claim")
    else:
        exact(raw.get("role"), reference_role, "HF reference role")
        require(raw.get("untimed_same_shape_reference") is True, "Separate same-shape HF reference required")


def artifacts_gate(measured, diagnostic, references, artifacts):
    """Re-open every exact raw JSON, compare file SHA and its supplied object."""
    evidence = common.Evidence()
    bound = [(artifacts["candidate"], measured), (artifacts["diagnostic"], diagnostic)]
    exact(set(artifacts["references"]), set(references), "reference artifact roles")
    bound.extend((artifacts["references"][role], value) for role, value in references.items())
    paths = []
    for artifact, value in bound:
        analysis._artifact(artifact, "Actual raw artifact")
        path = Path(artifact["path"])
        require(str(path) == value.get("artifact"), "Artifact points to a different raw object")
        paths.append(str(path))
        exact(evidence.read(path, artifact["sha256"]), value, "Actual file body")
    require(len(paths) == len(set(paths)), "Diagnostic/HF executions cannot reuse a raw file")
    evidence.unchanged()
    return dict(evidence.hashes)


def _frame(frame, *, full=True):
    keys = ("frames", "application_frame_bytes", "application_payload_bytes")
    for key in keys: integer(frame[key], "Frame " + key)
    require(frame["application_payload_bytes"] <= frame["application_frame_bytes"], "Frame payload exceeds application bytes")
    if full:
        for key in ("application_header_bytes", "length_prefix_bytes"): integer(frame[key], key)
        exact(frame["application_frame_bytes"], sum(frame[k] for k in ("application_payload_bytes", "application_header_bytes", "length_prefix_bytes")), "header/payload/length framing")
        exact(frame["length_prefix_bytes"], 4 * frame["frames"], "JSON header length prefixes")


def route_gate(row, source, spec, diagnostic):
    """Validate every token/layer, not only a reported RPC total."""
    p, delta = spec["prefix_tokens"], spec["suffix_tokens"] + spec["executor_only"]["realized_stop_tokens"] - 1
    end, state, calls = p + delta, row["state_report"], row["remote_calls"]
    exact(row["action"], spec["action"], "actual decode action")
    exact(row["diagnostic"], diagnostic, "decode diagnostic flag")
    require(row.get("passed") is True and row.get("background_drained") is True, "Decode/receiver failed")
    require(state.get("owned_delta_references_released") is True, "Destination delta references survived")
    exact(state["delta_lengths"], [[delta, delta]] * LAYERS, "destination deltas")
    exact(state["final_position"], end, "destination final position")
    exact(state["prefix_length"], p, "destination prefix")
    exact(state["suffix_forward_calls"], spec["suffix_tokens"], "suffix forward count")
    exact(state["generation_forward_calls"], spec["executor_only"]["realized_stop_tokens"] - 1, "generation forward count")
    exact(state["logits_captured"], diagnostic, "main has no full-vocabulary capture")
    require(isinstance(calls, list) and all(isinstance(x, list) and len(x) == 2 and type(x[0]) is int and
        0 <= x[0] < LAYERS and type(x[1]) is int and p <= x[1] < end for x in calls), "RPC position/layer domain")
    exact(calls, sorted(calls, key=lambda x: (x[1], x[0])), "tokenwise RPC order")
    require(len(set(map(tuple, calls))) == len(calls), "Duplicate remote RPC")
    remote_layers = {layer for layer, _ in calls}; switches = {}
    for item in state["transitions"]:
        layer, position = item["layer"], item["position"]
        require(type(layer) is int and layer in remote_layers and layer not in switches and
                type(position) is int and p < position < end, "Invalid/repeated route cutover")
        switches[layer] = position
    exact(list(switches.items()), sorted(switches.items(), key=lambda x: (x[1], x[0])), "actual transition order")
    expected = [[layer, position] for position in range(p, end) for layer in range(LAYERS)
                if layer in remote_layers and position < switches.get(layer, end)]
    exact(calls, expected, "contiguous replica RPCs; no post-cutover calls")
    exact(state["mode_counts"], {"local": LAYERS * delta - len(calls), "remote": len(calls)}, "all layer operations")
    signature = [switches.get(layer, end) if layer in remote_layers else p for layer in range(LAYERS)]
    exact(row["cutovers"], signature, "derived actual route signature")
    if spec["action"] == "remote": exact(signature, [end] * LAYERS, "persistent remote action")
    audit = state["decode_audit"]
    require(len(audit) == delta, "Missing actual token-forward audit")
    for offset, entry in enumerate(audit):
        exact((entry["input_tokens"], entry["absolute_position"], entry["delta_length"]), (1, p + offset, offset + 1), "decode token audit")
        require(type(entry["frontier"]) is int and 0 <= entry["frontier"] <= LAYERS, "Invalid actual decode frontier")
        finite(entry["monotonic_s"], "decode audit timestamp")
    ordered([x["monotonic_s"] for x in audit], "decode audit")
    exact(source["request_id"], row["request_id"], "source request owner")
    exact(source["prefix_key"], spec["prefix_key"], "source immutable prefix key")
    exact(source["prefix_length"], p, "source prefix length")
    exact(source["action"], spec["action"], "source action")
    exact(source["remote_calls"], len(calls), "source RPC count")
    exact([[x["layer"], x["position"]] for x in source["remote_audit"]], calls, "actual two-endpoint RPC sequence")
    exact(source["destination_reported_token_ids"], row["token_ids"], "actual final token receipt")
    per_layer = [sum(layer == i for layer, _ in calls) for i in range(LAYERS)]
    attention = source["attention_state"]
    for key, value in {"prefix_length": p, "calls_per_layer": per_layer, "next_positions": [p + v for v in per_layer],
                       "delta_lengths": per_layer, "total_calls": len(calls), "closed": True,
                       "owned_delta_references_released": True, "borrowed_prefix_references_dropped": True}.items():
        exact(attention.get(key), value, "holder replica " + key)
    return signature


def decode_traffic(row, source, spec, destination_ip):
    require(source.get("passed") is True and not source.get("errors") and
            source.get("completed_frame_accounting_matches_actual_socket_bytes") is True, "Source decode protocol failed")
    cleanup = source["cleanup"]
    require(cleanup.get("migration_thread_alive") is False and all(cleanup.get(k) is True for k in
            ("sockets_closed", "listeners_closed", "delta_references_released")), "Source request cleanup failed")
    # The accepted client opens bulk even for remote-only requests. Its actual
    # hello/ACK bytes count; absence of migration means zero tensor payload.
    names = ["rpc", "bulk"]
    exact(set(row["channel_audits"]), set(names), "destination actual channel set")
    exact(set(source["transport"]), set(names), "source actual channel set")
    for name in ("rpc", "bulk"):
        left, right = row["traffic"][name], source["traffic"][name]
        if name not in names:
            require(left is None and right is None, "Unexpected channel for remote-only work")
            continue
        require(source["peer_addresses"][name][0] == destination_ip, "Request socket peer is not the actual destination host")
        audit = row["channel_audits"][name]
        require(audit.get("socket_closed") is True and audit.get("frame_accounting_matches_actual_socket_bytes") is True, "Destination socket cleanup/frame mismatch")
        for direction, peer in (("sent", "received"), ("received", "sent")):
            integer(left[direction], "Actual frame bytes")
            exact(left[direction], right[peer], "peer actual bytes")
            exact(left["payload_" + direction], right["payload_" + peer], "peer payload bytes")
            exact(audit["transport"][direction + "_bytes"], left[direction], "destination socket vs frames")
            exact(source["transport"][name][peer + "_bytes"], right[peer], "source socket vs frames")
            frames = source["frames"][name][peer]; _frame(frames)
            exact(frames["application_frame_bytes"], right[peer], "source completed frames")
            exact(frames["application_payload_bytes"], right["payload_" + peer], "source completed payload")
            exact(frames["frames"], left[direction + "_frames"], "peer exact frame counts")
        expected_payload = (len(row["remote_calls"]) * (40 + 8 + 8) * 128 * 2,
                            len(row["remote_calls"]) * 40 * 128 * 2) if name == "rpc" else (0, spec["prefix_tokens"] * 196608 if spec["action"] == "repair" else 0)
        exact((left["payload_sent"], left["payload_received"]), expected_payload, "actual full Q/K/V or prefix payload")


def request_timing(row, spec, accepted, service_settled):
    exact(row["accepted_s"], accepted, "common destination acceptance")
    times = row["token_delivery_s"]
    require(len(times) == 64, "Exactly 64 own delivered token times required")
    ordered([accepted, row["executor_entered_s"], *times, row["protocol_settled_s"], row["own_settled_s"], service_settled], "own request timing")
    exact(row["protocol_settled_s"], row["settled_s"], "original protocol settlement preserved")
    for key, value in {"ttft_s": times[0]-accepted, "completion_s": times[-1]-accepted,
        "settled_latency_s": row["settled_s"]-accepted, "outer_setup_elapsed_s": row["executor_entered_s"]-accepted}.items():
        close(row[key], value, key)
    exact(row["itl_s"], [b-a for a, b in zip(times, times[1:])], "own ITLs")
    require(row.get("own_prefix_references_released") is True and row.get("own_receiver_thread_exited") is True,
            "Request prefix/receiver survived own settlement")
    exact(row["own_prefix_reference_count"], 96 if spec["action"] == "repair" else 0, "actual owned prefix weak-reference count")
    finish = row["e4_finish"]
    ordered([times[-1], finish["started_local_s"], finish["finished_local_s"], row["protocol_settled_s"]], "actual request finish ACK")
    exact(finish["acknowledgement"], row["finish_ack"], "finish ACK artifact")
    ack = finish["acknowledgement"]
    for key, expected in {"kind": "finish_ack", "request_id": row["request_id"], "remote_calls": len(row["remote_calls"]),
                          "next_layer": 48 if spec["action"] == "repair" else 0, "owned_delta_references_released": True}.items():
        exact(ack.get(key), expected, "finish " + key)


def repair_gate(row, source, spec):
    if spec["action"] != "repair":
        for field in ("transfer_segments", "e4_layer_publications", "e4_receiver_drains"):
            exact(row[field], [], "remote-only " + field)
        exact(source["transfers"], [], "remote-only source transfers")
        exact(source["gpu_d2h_copies"], 0, "remote-only prefix DtoH")
        exact(row["committed_frontier"], 0, "remote-only prefix frontier")
        return None
    pbytes = spec["prefix_tokens"] * 4096
    exact(row["committed_frontier"], LAYERS, "complete destination prefix")
    require(len(row["transfer_segments"]) == len(source["transfers"]) == len(row["e4_receiver_drains"]) == 1, "One complete prefix transfer required")
    segment, origin, drain = row["transfer_segments"][0], source["transfers"][0], row["e4_receiver_drains"][0]
    fields = dict(transfer_index=1, start=0, end=48, next_layer=48, payload_bytes=48*pbytes)
    for key, value in fields.items():
        exact(segment[key], value, "destination terminal " + key)
        exact(origin["control"][key] if key == "payload_bytes" else origin[key], value, "source terminal " + key)
    require(segment["stopped"] is False and segment["kind"] == "migration_terminal", "Premature repair cancellation")
    control = origin["control"]
    require(control["state"] == "complete" and control["complete_layers"] == 48 and control["in_flight_layer"] is None
            and control["terminal_sent"] is True and control["receiver_drained_ack"] is True and control["error"] is None,
            "Source transfer was not truly drained")
    exact(drain["fields"], fields, "actual destination drained identity")
    exact(drain["acknowledgement"], segment["receiver_drain_ack"], "actual drain ACK")
    require(drain["acknowledgement"]["drained"] is True and drain["receiver_thread_exited"] is True, "Receiver drain not complete")
    exact(drain["acknowledgement"].get("kind"),"migration_drained_ack","actual drain RPC kind")
    exact(drain["acknowledgement"].get("request_id"),row["request_id"],"actual drain owner")
    for key, value in fields.items(): exact(drain["acknowledgement"][key], value, "drain ACK field")
    exact(source["gpu_d2h_copies"], 96, "actual source copy count")
    exact(source["gpu_d2h_bytes"], 48*pbytes, "actual source copied bytes")
    publications = row["e4_layer_publications"]
    require(len(publications) == 48, "All actual layer readiness records required")
    for layer, pub in enumerate(publications):
        for key, value in {"layer": layer, "transfer_index": 1, "bytes": pbytes, "HtoD_copies": 2, "content_hash_captured": False}.items():
            exact(pub[key], value, "actual publication " + key)
        ordered([row["accepted_s"], pub["copy_started_local_s"], pub["GPU_ready_local_s"], pub["published_local_s"], drain["started_local_s"],
                 drain["finished_local_s"], row["e4_finish"]["started_local_s"]], "GPU-ready publication before drain")
    ordered([pub["published_local_s"] for pub in publications], "contiguous layer publications")
    # Source clocks stay source-local. They are never compared to destination.
    events = source["frame_events"]
    require(len(events) == 96, "Actual source frame begin/finish events missing")
    for layer in range(48):
        start, end = events[2*layer:2*layer+2]
        exact((start["kind"], end["kind"], start["layer"], end["layer"]), ("begin", "finish", layer, layer), "source complete frame events")
        exact(end["payload_bytes"], pbytes, "source finished frame payload")
    ordered([event["source_local_s"] for event in events], "source frame events")
    return dict(first_gpu_ready_s=publications[0]["GPU_ready_local_s"], gpu_ready_s=publications[-1]["GPU_ready_local_s"],
                drained_s=drain["finished_local_s"], payload_bytes=48*pbytes, complete_layers=48)


def _queue(queue, allowed, *, start, end):
    common.queue_gate(queue)
    for op in queue["operations"]:
        require(op["owner"] in allowed or op["owner"].startswith("background-job-"), "Unbound compute owner or hidden reference in main work")
        ordered([start, op["arrival_monotonic_s"], op["start_monotonic_s"], op["end_monotonic_s"], end], "actual operation scope")
        service_duration(op, "completion-bounded queue service")
    return {"service_s": sum(op["service_s"] for op in queue["operations"]), "wait_s": sum(op["queue_wait_s"] for op in queue["operations"])}


def _background_jobs(records, operations):
    ids = [r["job_id"] for r in records]
    require(len(ids) == len(set(ids)), "Duplicate offered background job")
    exact(set(op["owner"] for op in operations), set(ids), "actual background compute ownership")
    for r in records:
        require(r["status"] == "completed" and r["error"] is None, "Failed background job")
        ordered([r["scheduled_monotonic_s"], r["arrival_monotonic_s"], r["start_monotonic_s"], r["end_monotonic_s"]], "background job")
        ops = [o for o in operations if o["owner"] == r["job_id"]]
        exact(len(ops), r["gpu_admissions"], "background admitted operations")
        exact(Counter(op["kind"] for op in ops),Counter({"layered_token":47,"greedy_selection":32}),"actual cached-local background operation counts")
        require(ops and r["gpu_service_s"] > 0, "Background did no real admitted work")
        close(r["gpu_service_s"], sum(o["service_s"] for o in ops), "background service")
        s = r["result"]["state_report"]
        for key, value in {"owned_delta_references_released": True, "logits_captured": False,
                          "mode_counts": {"local": 48*47, "remote": 0}, "delta_lengths": [[47,47]]*48, "final_position": 1071}.items():
            exact(s.get(key), value, "background state " + key)
        common.tokens(r["result"]["token_ids"], 32, "background")


def calibration_gate(calibration):
    require(calibration.get("backend") == "cuda_stream_completion" and calibration.get("GPU_service_evidence") is True,
            "Calibration is not actual CUDA work")
    exact((calibration["warmup_jobs"], calibration["measured_jobs"]), (2, 10), "isolated calibration counts")
    records, operations = calibration["records"], calibration["operations"]
    require(len(records) == 12, "Missing actual calibration jobs")
    for i, row in enumerate(records):
        phase = "warmup" if i < 2 else "calibration"
        exact(row["phase"], phase, "calibration split")
        exact(row["job_id"], "background-" + phase + "-" + str(i), "calibration owner")
    _background_jobs(records, operations)
    common.queue_gate(dict(backend="cuda_stream_completion", fifo=True, active_gpu_operations=0,
                           queued_gpu_operations=0, operations=operations))
    for op in operations: service_duration(op, "calibration operation service")
    close(calibration["mean_service_s"], sum(r["gpu_service_s"] for r in records[2:])/10, "isolated ten-request service mean")
    workload = calibration["workload"]
    for key, expected in {"prefix":1024,"suffix":16,"generated":32,"model_shared":True,"prefix_shared_with_foreground":False,
                          "delta_per_request":True,"request_worker_count":1}.items(): exact(workload[key], expected, "calibration workload")
    analysis._sha(workload["input_sha256"], "Background input")


def background_gate(report, calibration, queue, *, required, seed, acceptance, service_settled):
    ops = [op for op in queue["operations"] if op["owner"].startswith("background-job-")]
    if not required:
        require(report is None and not ops, "Unexpected background or loaded solo")
        return dict(offered=0, completed=0, failed=0, pending=0)
    require(report.get("passed") is True and report.get("running") is False and report.get("error") is None,
            "Background not cleanly drained")
    exact(report["backend"], "cuda_stream_completion", "background backend")
    exact(report["workload"], calibration["workload"], "background workload identity")
    exact(report["seed"], seed, "background arrival seed")
    exact(report["rho"], .5, "background load")
    exact(report["lambda_jobs_per_s"], .5/calibration["mean_service_s"], "actual calibrated arrival rate")
    exact((report["lead_in_s"], report["lead_service_multiples"]), (0., 0), "E4 zero-lead timed arrivals")
    exact(report["acceptance_monotonic_s"], acceptance, "local background acceptance anchor")
    ordered([acceptance, report["started_monotonic_s"], report["lead_in_finished_monotonic_s"],
             report["stop_requested_monotonic_s"], report["drained_monotonic_s"], service_settled], "background local drain")
    require(report["threads_alive"] == [] and report["failed"] == report["pending"] == 0 and
            all(report["cleanup"].get(k) is True for k in ("producer_exited","worker_exited","all_accepted_jobs_completed")), "Background cleanup failed")
    exact(report["arrivals"], len(report["records"]), "all offered background arrivals")
    exact(report["completed"], report["arrivals"], "background completion denominator")
    exact(report["operations"], ops, "background operations vs full actual queue")
    _background_jobs(report["records"], ops)
    rng, scheduled = random.Random(seed), report["started_monotonic_s"]
    for index, r in enumerate(report["records"]):
        scheduled += rng.expovariate(report["lambda_jobs_per_s"])
        exact(r["scheduled_monotonic_s"], scheduled, "actual seeded Poisson schedule")
        exact(r["job_id"], "background-job-"+str(index), "trial background job identity")
    return dict(offered=report["arrivals"],completed=report["completed"],failed=0,pending=0)


def network_gate(snapshot, hostname):
    exact(snapshot["hostname"], hostname, "scoped network physical host")
    require(snapshot["eth0_root_unchanged"] is True and snapshot["owned_filter_count"] == 20, "E4 ten-port/two-direction rule ownership differs")
    for key in ("requested_added_rtt_ms", "applied_one_way_delay_ms", "observed_one_way_delay_ms"):
        exact(snapshot[key], 0, "actual zero added delay")
    require(snapshot["physical_wire_bytes"] is None, "No physical NIC counter claim")
    analysis._sha(snapshot["plan_sha256"], "Scoped network plan")


def budget_gate(raw, trial, sources):
    """Reconcile every owned connection, including paced outer control."""
    dest, holder = raw["final_coordinator"], raw["source"]["final_coordinator"]
    expected = {"destination": {"outer-control"}, "holder": {"outer-control"}}
    for spec in trial["requests"]:
        row, source = raw["requests"][spec["role"]], sources[spec["role"]]
        for name in row["channel_audits"]:
            expected["destination"].add(row["request_id"]+"/"+name)
            expected["holder"].add(row["request_id"]+":"+name)
            for snapshot, key, counters in ((dest,row["request_id"]+"/"+name,row["channel_audits"][name]["transport"]),
                                            (holder,row["request_id"]+":"+name,source["transport"][name])):
                for direction in ("sent_bytes","received_bytes"):
                    exact(snapshot["limiter"]["connections"][key][direction], counters[direction], "every decode connection in one shared budget")
    for spec in trial["transfer_jobs"]:
        for side, snapshot, endpoint in (("destination",dest,raw["transfers"][spec["role"]]),("holder",holder,sources[spec["role"]])):
            for name in ("rpc","bulk"):
                key=endpoint["request_id"]+"/transfer-"+name; expected[side].add(key)
                for direction in ("sent_bytes","received_bytes"):
                    exact(snapshot["limiter"]["connections"][key][direction], endpoint["socket_accounting"][name][direction], "every transfer connection in shared budget")
    controls = {"destination":raw["control_accounting"], "holder":raw["source"]["control_accounting"]}
    totals, contention = {}, {}
    for side, snapshot, closed in (("destination",dest,raw["coordinator_after_close"]),("holder",holder,raw["source"]["coordinator_after_close"])):
        exact(snapshot["configuration"], trial["scheduler"], "actual scheduler configuration")
        exact(closed["configuration"], trial["scheduler"], "closed scheduler configuration")
        require(snapshot["active"] is None and snapshot["pending"] == [] and closed["closed"] is True, "Budget still has actual queued/in-flight work")
        conns, limiter = snapshot["limiter"]["connections"], snapshot["limiter"]
        exact(set(conns), expected[side], "actual shared-budget connection set")
        exact(set(snapshot["streams"]), expected[side], "scheduler stream set")
        exact(limiter["rate_bits_per_second"], trial["scheduler"]["rate_bps"], "whole-direction cap bits/s")
        exact(limiter["chunk_bytes"], trial["scheduler"]["chunk_bytes"], "actual chunking")
        require(limiter["pacing_enabled"] is True and snapshot["physical_wire_bytes"] is None, "Missing actual application pacing")
        control = controls[side]
        require(control["passed"] is True and control["physical_wire_bytes"] is None, "Control accounting failed")
        for direction in ("sent","received"):
            exact(control["frames"][direction], control["socket"][direction+"_bytes"], "control actual frame/socket bytes")
            exact(conns["outer-control"][direction+"_bytes"], control["frames"][direction], "control uses same actual budget")
            exact(control["frames"][direction+"_frames"], 4, "paced epoch/batch/settle/disarm control frames")
        totals[side] = {direction:sum(integer(c[direction],"actual connection bytes") for c in conns.values()) for direction in ("sent_bytes","received_bytes")}
        for conn in conns.values():
            exact(integer(conn["reserved_bytes"],"actual paced bytes"),conn["sent_bytes"],"every successful byte reserved in shared cap")
        exact(limiter["total"]["reserved_bytes"],totals[side]["sent_bytes"],"global cap reservation count")
        for key,value in totals[side].items():
            exact(limiter["total"][key],value,"whole-budget total")
            exact(closed["limiter"]["total"][key],value,"close did not send unaccounted bytes")
        counts={key:integer(snapshot[key],key) for key in ("coeligible_grant_count","successful_coeligible_grant_count")}
        require(counts["successful_coeligible_grant_count"] <= counts["coeligible_grant_count"] <= snapshot["total_grants"], "Invalid same-grant coeligibility counters")
        bulk = [s for s in snapshot["streams"].values() if s["kind"]=="bulk"]
        counts["bulk_max_eligible_wait_s"]=max((finite(s["maximum_eligible_queue_age_s"],"bulk eligible wait") for s in bulk),default=0.)
        contention[side]=counts
    for direction,peer in (("sent","received"),("received","sent")):
        exact(controls["destination"]["frames"][direction],controls["holder"]["frames"][peer],"control peer bytes")
        exact(totals["destination"][direction+"_bytes"],totals["holder"][peer+"_bytes"],"all actual application bytes")
    # The compact settle snapshot is the service boundary. The only later paced
    # frame is destination->holder disarm; the large audit is explicitly unpaced.
    ds,hs=raw["settlement_coordinator"],raw["source"]["trial_report"]["service_settlement_coordinator"]
    for side,snapshot,final in (("destination",ds,dest),("holder",hs,holder)):
        exact(set(snapshot["limiter"]["connections"]),expected[side],"service-boundary connections")
        for key,conn in snapshot["limiter"]["connections"].items():
            for direction in ("sent_bytes","received_bytes"):
                old,new=conn[direction],final["limiter"]["connections"][key][direction]
                if key=="outer-control" and direction==("sent_bytes" if side=="destination" else "received_bytes"):
                    require(new>old,"Missing actual post-service disarm frame")
                else: exact(new,old,"no request/large audit bytes after service snapshot")
    for direction,peer in (("sent_bytes","received_bytes"),("received_bytes","sent_bytes")):
        exact(sum(c[direction] for c in ds["limiter"]["connections"].values()),
              sum(c[peer] for c in hs["limiter"]["connections"].values()),"service-boundary peer bytes")
    return dict(application_sent_bytes={s:totals[s]["sent_bytes"] for s in ROLES},contention=contention)


def raw_trial_gate(raw, trial, context, calibration, diagnostic):
    offered=trial["requests"]+trial["transfer_jobs"]
    exact(set(raw["requests"]),{s["role"] for s in trial["requests"]},"all offered decode roles")
    exact(set(raw["transfers"]),{s["role"] for s in trial["transfer_jobs"]},"all offered transfer roles")
    exact(set(raw["inputs"]),{s["role"] for s in offered},"all actual input arrays")
    group, source = raw["group"],raw["source"]["trial_report"]
    require(raw["source"].get("passed") is True and raw["source"].get("CPU_component_only") is False,
            "Disarmed source audit is not a successful actual GPU execution")
    require(group["passed"] is True and group["errors"]==[] and group["alive_threads"]==[] and source["passed"] is True
            and source["errors"]==[] and source["alive_request_threads"]==[] and source["CPU_component_only"] is False,
            "Actual offered group/source cleanup failed")
    exact((group["offered_count"],source["offered_count"]),(len(offered),len(offered)),"all offered work denominators")
    exact(source["trial_id"],trial["trial_id"],"source trial identity")
    exact(source["trial_sha256"],digest(trial),"source trial hash")
    exact(source["scheduler"],trial["scheduler"],"source scheduler identity")
    epoch=source["epoch"]
    ordered([epoch["marker_received_local_s"],epoch["source_epoch_s"],epoch["ack_send_started_local_s"],epoch["ack_sent_local_s"]],
            "actual holder epoch ACK")
    for key,value in epoch.items():
        if key!="ack_sent_local_s":exact(raw["source_epoch_ack"][key],value,"actual observed epoch ACK")
    exact(raw["source_epoch_ack"]["trial_id"],trial["trial_id"],"epoch ACK trial")
    exact(raw["source_epoch_ack"]["trial_sha256"],digest(trial),"epoch ACK frozen trial")
    require(len(source["outcomes"])==len(offered),"Missing source outcome")
    sources={};ids=[]
    for entry,spec in zip(source["outcomes"],offered):
        exact(entry["role"],spec["role"],"actual source offered role")
        exact(entry["kind"],"transfer_only" if spec["action"]=="transfer_only" else "decode","source job kind")
        require(entry["status"]=="passed" and entry["passed"] is True,"Unsuccessful offered source job")
        ordered([source["epoch"]["source_epoch_s"],entry["thread_arrived_local_s"],entry["released_local_s"],entry["thread_finished_local_s"],
                 source["settlement"]["source_work_settled_local_s"]],"source request lifecycle")
        sources[spec["role"]]=entry["raw"];ids.append(entry["request_id"])
        exact(entry["request_id"],entry["raw"]["request_id"],"source outcome owner")
    require(len(ids)==len(set(ids)),"Request IDs/deltas are not isolated")
    exact([o["job_id"] for o in group["outcomes"]],ids,"two-endpoint offered order")
    exact(set(raw["batch_ready"]["ports"]),set(ids),"actual offered setup endpoints")
    exact(raw["batch_done"]["offered_count"],len(offered),"compact source offered count")
    require(raw["batch_done"]["passed"] is True and raw["batch_done"]["alive_request_threads"]==[],"Compact source terminal is not settled")
    for compact,entry in zip(raw["batch_done"]["requests"],source["outcomes"]):
        for key in ("request_id","role","kind","passed","status"):
            exact(compact[key],entry[key],"compact/source actual terminal identity")
    exact(len(raw["batch_done"]["requests"]),len(offered),"complete compact terminal coverage")
    accepted,settled=group["accepted_s"],group["group_service_settled_s"]
    exact(raw["accepted_s"],accepted,"common group acceptance")
    ordered([accepted,group["setup_started_s"],raw["epoch_send_started_s"],raw["epoch_ack_received_s"],group["setup_finished_s"],
             raw["batch_done_received_s"],settled,group["group_finished_s"],raw["ownership_finished_s"]],"actual group service/ownership separation")
    require(settled-accepted<=1200,"Common service deadline exceeded")
    close(group["group_service_settled_latency_s"],settled-accepted,"group service latency")
    require(group["finalization"]["passed"] is True,"Group finalization failed")
    exact(group["finalization"]["service_settled_s"],settled,"actual compact-settle boundary")
    for entry in group["outcomes"]:
        require(entry["passed"] is True and entry["status"]=="passed","Dropped/failed destination work")
        exact(entry["accepted_s"],accepted,"own common acceptance")
        ordered([entry["thread_arrived_s"],accepted,entry["dispatch_released_s"],entry["executor_entered_s"],entry["settled_s"],
                 entry["protocol_returned_s"],entry["thread_finished_s"],settled],"actual own thread lifecycle")
    st=source["settlement"];wire=raw["source_settlement"]
    for key in ("offered_count","completed_jobs"):
        exact(st[key],len(offered),"source settled denominator")
    for key in ("source_threads_exited","background_drained","source_queue_idle"):
        require(st[key] is True,"Source service not fully drained")
    for key,value in st.items():
        if key!="ack_sent_local_s":exact(wire[key],value,"actual compact settlement ACK")
    ordered([source["source_jobs_finished_local_s"],st["settle_received_local_s"],st["source_work_settled_local_s"],
             st["ack_sent_local_s"],source["group_disarmed_local_s"]],"source compact-settle then audit")
    cleanup=raw["cleanup"]
    require(cleanup["errors"]==[] and cleanup["destination_job_threads_exited"] is True and cleanup["trial_coordinator_closed"] is True
            and cleanup["destination_background_running"] is False,"Destination per-trial resources not released")
    prepared={p["prefix_key"]:p for p in raw["prepared_prefixes"]}
    exact(set(prepared),{s["prefix_key"] for s in offered},"actual prepared immutable prefixes")
    requests,transfers=[],[]
    for spec,outcome in zip(offered,group["outcomes"]):
        role=spec["role"];inputs=raw["inputs"][role]
        exact(set(inputs),{"prefix_ids","suffix_ids"},"actual token arrays")
        common.tokens(inputs["prefix_ids"],spec["prefix_tokens"],"actual prefix")
        common.tokens(inputs["suffix_ids"],spec["suffix_tokens"],"actual suffix")
        prefix=prepared[spec["prefix_key"]]
        exact(prefix["token_ids_sha256"],digest(inputs["prefix_ids"]),"prepared real prefix token hash")
        exact((prefix["prefix_length"],prefix["layers"],prefix["cache_bytes"]),(spec["prefix_tokens"],48,spec["prefix_tokens"]*196608),"prepared actual KV geometry")
        # Source preparation is explicitly outside acceptance; its one-time
        # real content hash identifies the immutable borrowed prefix. No main
        # timed migration or receiver content hash is inferred from it.
        analysis._sha(prefix["cache_sha256"],"Untimed prepared immutable KV content")
        if spec["action"]=="transfer_only":
            row=raw["transfers"][role];origin=sources[role]
            accepted_pair=transfer_only.reconcile(origin,row)
            require(accepted_pair["GPU_execution_evidence"] is True and accepted_pair["CPU_component_only"] is False,"CPU transfer cannot certify GPU work")
            exact(row["spec"]["diagnostic"],diagnostic,"transfer diagnostic treatment")
            exact(row["spec"]["prefix_key"],spec["prefix_key"],"transferred prefix identity")
            exact(row["spec"]["prefix_tokens"],spec["prefix_tokens"],"transferred prefix geometry")
            exact(row["accepted_local_s"],accepted,"transfer common acceptance")
            own=row["settled_local_s"]
            received=row["source_finish_receive"]
            ordered([row["drain_send"]["finished_local_s"],received["started_local_s"],received["finished_local_s"],own],
                    "actual transfer drain/finish ACK receipt then owned cleanup")
            migration=dict(first_gpu_ready_s=row["layers"][0]["copy"]["ready_local_s"],gpu_ready_s=row["all_layers_ready_local_s"],
                           drained_s=received["finished_local_s"],payload_bytes=accepted_pair["payload_bytes"],complete_layers=48)
            transfers.append(dict(role=role,spec_sha256=digest(spec),input_ids_sha256=digest(inputs["prefix_ids"]),status="passed",
                                  own_settled_s=own,cleanup_passed=True,migration=migration))
        else:
            row=raw["requests"][role];origin=sources[role]
            exact(origin["prefix_cache_sha256"],prefix["cache_sha256"],"actual borrowed immutable prefix identity")
            route_gate(row,origin,spec,diagnostic);decode_traffic(row,origin,spec,context["ranks"][1]["internal_ip"])
            request_timing(row,spec,accepted,settled);migration=repair_gate(row,origin,spec)
            own=row["own_settled_s"]
            requests.append(dict(role=role,spec_sha256=digest(spec),input_ids_sha256=digest(inputs["prefix_ids"]+inputs["suffix_ids"]),
                output_ids_sha256=digest(row["token_ids"]),status="passed",executor_entered_s=row["executor_entered_s"],
                token_delivery_s=copy.deepcopy(row["token_delivery_s"]),own_settled_s=own,cleanup_passed=True,migration=migration))
        exact(outcome["settled_s"],own,"own job, not group, settlement")
    network_gate(raw["network_state"],context["ranks"][1]["hostname"])
    network_gate(raw["source"]["network_state"],context["ranks"][0]["hostname"])
    queues={"holder":raw["source"]["source_queue_events"],"destination":raw["destination_queue_events"]}
    starts={"holder":source["epoch"]["source_epoch_s"],"destination":accepted}
    ends={"holder":st["source_work_settled_local_s"],"destination":settled}
    resource=budget_gate(raw,trial,sources)
    resource.update(compute_service_s={},compute_queue_wait_s={},background={})
    for side in ROLES:
        metrics=_queue(queues[side],set(ids),start=starts[side],end=ends[side])
        resource["compute_service_s"][side]=metrics["service_s"];resource["compute_queue_wait_s"][side]=metrics["wait_s"]
        bg=raw["source"]["background_report"] if side=="holder" else raw["background_report"]
        resource["background"][side]=background_gate(bg,calibration[side],queues[side],required=trial["load_location"]==side,
            seed=trial["load_seed"],acceptance=starts[side],service_settled=ends[side])
    for spec in trial["requests"]:
        rid=raw["requests"][spec["role"]]["request_id"]
        for side in ROLES:
            ops=[op for op in queues[side]["operations"] if op["owner"]==rid]
            if side=="holder":
                exact(Counter(op["kind"] for op in ops),Counter({"holder_setup":1,"holder_attention":len(raw["requests"][spec["role"]]["remote_calls"])}),"actual source setup/attention operations")
            else:
                exact(Counter(op["kind"] for op in ops),Counter({"layered_token":79+len(raw["requests"][spec["role"]]["remote_calls"]),
                      "greedy_selection":64}),"actual destination forward segments and greedy operations")
    return dict(requests=requests,transfer_jobs=transfers,resources=resource,
                timing=dict(accepted_s=accepted,group_service_settled_s=settled,group_ownership_settled_s=raw["ownership_finished_s"]))


def _base(trial,protocol,context):
    return dict(schema_version=1,**{k:trial[k] for k in analysis.IDENTITY},plan_sha256=protocol["plan_sha256"],trial_sha256=digest(trial),
        run_id=context["run_id"],attempt_id=context["run_id"]+":"+trial["trial_id"],source_sha256=copy.deepcopy(context["source_sha256"]),
        cohort_sha256=_context(context,protocol),model_revision=context["model_revision"],model_manifest_sha256=context["model_manifest_sha256"],E4_complete=False)


def normalize(trial,active_protocol,context,measured,diagnostic,references_by_role,raw_artifacts,background_calibration):
    validate_plan(active_protocol)
    require(trial in active_protocol["trials"],"Trial is outside active frozen E4 plan")
    record=_base(trial,active_protocol,context)
    exact(set(references_by_role),{s["role"] for s in trial["requests"]},"independent HF role set")
    identity(measured,trial,active_protocol,context,diagnostic=False)
    identity(diagnostic,trial,active_protocol,context,diagnostic=True)
    exact(measured["context_sha256"],diagnostic["context_sha256"],"main/diagnostic context file SHA")
    exact(measured["inputs"],diagnostic["inputs"],"actual main/diagnostic inputs")
    calibration={"holder":background_calibration["source"]["calibration"],"destination":background_calibration["destination_calibration"]}
    require(background_calibration.get("CPU_component_only") is False,"Actual GPU isolated calibration required")
    for value in calibration.values():calibration_gate(value)
    main=raw_trial_gate(measured,trial,context,calibration,False)
    raw_trial_gate(diagnostic,trial,context,calibration,True)
    for spec in trial["requests"]:
        role=spec["role"];m=measured["requests"][role];d=diagnostic["requests"][role];ref=references_by_role[role]
        identity(ref,trial,active_protocol,context,diagnostic=False,reference_role=role)
        exact(ref["context_sha256"],measured["context_sha256"],"HF actual context")
        exact(ref["inputs"],measured["inputs"][role],"HF actual input arrays")
        exact(ref["input_ids_sha256"],digest(ref["inputs"]["prefix_ids"]+ref["inputs"]["suffix_ids"]),"HF actual input SHA")
        for key in ("token_ids","cutovers","remote_calls"):
            exact(m[key],d[key],"independent diagnostic actual "+key)
        require("diagnostic_logits" not in m and not m.get("logits"),"Main retained full vocabulary")
        common.logits_witness(d["diagnostic_logits"],64);common.logits_witness(ref["logits"],64)
        common.numerical(d["numerical_check"],d["token_ids"],ref["token_ids"],64)
        exact(ref["numerical_check"],d["numerical_check"],"actual reference numerical report")
        exact(m["numerical_check"],d["numerical_check"],"same diagnostic gate applied to main tokens/route")
        end=spec["prefix_tokens"]+79;s=ref["state_report"]
        for key,value in {"owned_cache_references_released":True,"final_cache_lengths":[end]*48,"final_position":end,
            "prefix_prefill_tokens":spec["prefix_tokens"],"suffix_forward_calls":16,"generation_forward_calls":63,"logits_captured":True}.items():
            exact(s[key],value,"actual same-shape HF state")
    contents={role:digest([layer["destination_hashes"] for layer in report["layers"]]) for role,report in diagnostic["transfers"].items()}
    exact(raw_artifacts["diagnostic"].get("contents_sha256",{}),contents,"actual separate diagnostic KV content identity")
    hashes=artifacts_gate(measured,diagnostic,references_by_role,raw_artifacts)
    record.update(main,status="passed",**{gate:True for gate in analysis.GATES},raw_artifacts=copy.deepcopy(raw_artifacts),
        raw_reopened_sha256=hashes,context_sha256=measured["context_sha256"],
        evidence_scope="Actual saved numerical reports/SHA witnesses, route/readiness/frame/queue/lifecycle evidence; no tensor reconstruction from hashes",
        parent_gates_pending=["Frozen context/source file SHA and two physical worker identities", "Final shared-prefix/network/session ownership cleanup", "16 profile groups and 32 rank traces"])
    analysis.validate_record(record,trial,active_protocol)
    return record


def failed_record(trial,active_protocol,context,measured,diagnostic,references_by_role,raw_artifacts,background_calibration,*,error):
    """Retain every offered role after failure; never manufacture success/zero.

    Original possibly malformed/partial values stay in SHA-bound raw artifacts.
    Unknown or unaudited normalized metrics are null; delivered token times are
    retained only when finite and monotonic. This record never passes a gate.
    """
    require(isinstance(error,str) and error,"Preserve actual failure reason")
    record=_base(trial,active_protocol,context)
    jobs={}
    for field,raw_field in (("requests","requests"),("transfer_jobs","transfers")):
        jobs[field]=[]
        for spec in trial[field]:
            raw=(measured or {}).get(raw_field,{}).get(spec["role"],{})
            item=dict(role=spec["role"],spec_sha256=digest(spec),input_ids_sha256=None,status="failed" if raw else "not_started",
                      own_settled_s=None,cleanup_passed=False,migration=None)
            if field=="requests":
                times=raw.get("token_delivery_s",[])
                valid=isinstance(times,list) and len(times)<=64 and all(type(t) in (int,float) and math.isfinite(t) and t>=0 for t in times)
                item.update(output_ids_sha256=None,executor_entered_s=None,token_delivery_s=copy.deepcopy(times) if valid and times==sorted(times) else [])
            jobs[field].append(item)
    resource={key:{side:None for side in ROLES} for key in ("application_sent_bytes","compute_service_s","compute_queue_wait_s")}
    resource["background"]={side:{k:None for k in ("offered","completed","failed","pending")} for side in ROLES}
    resource["contention"]={side:{k:None for k in ("coeligible_grant_count","successful_coeligible_grant_count","bulk_max_eligible_wait_s")} for side in ROLES}
    record.update(jobs,status="failed",**{gate:False for gate in analysis.GATES},raw_artifacts=copy.deepcopy(raw_artifacts),
        timing={k:None for k in ("accepted_s","group_service_settled_s","group_ownership_settled_s")},resources=resource,error=error,
        failed_values_preserved_in_raw=True,scope="All offered slots retained; unaudited metrics null; no successful timing inference")
    analysis.validate_record(record,trial,active_protocol)
    return record
