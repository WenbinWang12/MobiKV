"""One paired E1 input group, five actual actions and a separate HF control.

Implementation preparation only until the E0 stage and new runner are validated.
Two Slurm ranks: holder=0, destination=1, one A100 80GB each. All request
connections, payloads and completion acknowledgements use one shared outgoing
application-byte budget per rank. Preparation/control traffic is outside the
request and never used to carry timed request data. No Gloo or timing replay.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import threading
import time
import traceback


MODEL_REVISION = "97e1e76335b7017d8f67c08a19d103c0504298c9"
MODEL_MANIFEST = "0ef3c81db38c9d2fe14959dc6e8abb5780bbf826b264dbacbdbf80d586cc5e99"
ACTION_NAMES = {"recompute", "native_fetch", "holder", "remote_attention", "fixed_progressive_repair"}
CONTROL_ACTION = "native_hf_same_shape_local_control"
EXECUTION_SOURCES = (
    "e1_runner.py", "e1_runtime.py", "e1_channels.py", "e1_acceptance.py",
    "e1_transport.py", "e1_wire.py", "e1_trial_plan.py", "e0_layered_runtime.py",
    "e0_cache_utils.py", "e0_holder_decode.py", "e0_native_fetch.py",
    "collect_e0_status.py", "e0_precision_protocol.json", "e0_native_precision_protocol.json",
    "e0_holder_precision_protocol.json", "e1_pilot_protocol.json",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def save(path, value):
    nonfinite = []

    def normalize(item, location):
        if isinstance(item, float) and not math.isfinite(item):
            nonfinite.append({"path": location, "value": repr(item)})
            return None
        if isinstance(item, dict):
            return {key: normalize(entry, location + "/" + str(key)) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(entry, location + "/" + str(index)) for index, entry in enumerate(item)]
        return item

    serialized = normalize(value, "")
    if nonfinite:
        serialized.update(passed=False, nonfinite_diagnostics=nonfinite)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(serialized, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    if nonfinite and value.get("status") != "failed":
        raise ValueError("Nonfinite report values preserved as null with explicit diagnostic paths")


def baseline_rtt(control, rank, samples=16):
    """Unpaced framed application RTT outside requests, on the destination clock."""
    values = []
    for index in range(samples):
        ping = {"kind": "baseline_rtt_ping", "index": index}
        pong = {"kind": "baseline_rtt_pong", "index": index}
        if rank == 1:
            started = time.monotonic()
            control.send_message(ping)
            message, tensors, _ = control.recv_message()
            elapsed = time.monotonic() - started
            if message != pong or tensors or not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("Invalid baseline RTT response or clock")
            values.append(elapsed)
        else:
            message, tensors, _ = control.recv_message()
            if message != ping or tensors:
                raise ValueError("Invalid baseline RTT request")
            control.send_message(pong)
    return {"scope": "unpaced_application_control_round_trip_including_framing_and_python",
            "added_delay_s": 0, "outside_request_intervals": True,
            "measurement_host": "destination", "sample_count": samples,
            "round_trip_s": values if rank == 1 else None,
            "hardware_network_rtt_isolated": False}


def stage_gate(root, suite_job):
    from e1_acceptance import assert_e0_suite_passed
    evidence = assert_e0_suite_passed(root, suite_job)
    path = root / "results/e0_stage_acceptance.json"
    stage = json.loads(path.read_text())
    if (stage.get("passed") is not True or stage.get("E0_complete") is not True
            or str(stage.get("suite_job_id")) != str(suite_job)
            or stage.get("plan_sha256") != evidence["plan_sha256"]
            or not stage.get("coverage_review")):
        raise ValueError("A reviewed E0 stage acceptance bound to this exact suite is required")
    return {"suite": evidence, "stage_acceptance_sha256": digest(path)}


def load_group(plan_path, cell_id, phase, repetition):
    from e1_trial_plan import validate_plan, validate_protocol
    plan = json.loads(Path(plan_path).read_text())
    validate_plan(plan)
    protocol_path = Path(__file__).with_name("e1_pilot_protocol.json")
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol)
    if plan["provenance"]["protocol_file_sha256"] != digest(protocol_path):
        raise ValueError("Trial plan refers to a different pilot protocol")
    cells = [cell for cell in plan["cells"] if cell["cell_id"] == cell_id]
    if len(cells) != 1:
        raise ValueError("Select exactly one planned cell")
    groups = [group for group in cells[0]["groups"][phase] if group["repetition"] == repetition]
    if len(groups) != 1:
        raise ValueError("Select exactly one planned paired group")
    group = groups[0]
    records = [item for item in plan["inputs"] if item["input_id"] == group["input_id"]]
    if len(records) != 1:
        raise ValueError("Missing or duplicated input identity")
    return plan, cells[0], group, records[0]


def materialize(record, tokenizer, vocab_size):
    recipe = record["input_recipe"]
    ids = record["token_ids"]
    if ids is None:
        if recipe["kind"] != "tokenize_frozen_text_then_select_window":
            raise ValueError("Unsupported unmaterialized recipe")
        text = recipe["text_unit"] * recipe["text_repeats"]
        if hashlib.sha256(text.encode()).hexdigest() != recipe["input_text_utf8_sha256"]:
            raise ValueError("Input text identity changed")
        bank = tokenizer(text, add_special_tokens=False).input_ids
        count = recipe["window_tokens"]
        if len(bank) < count:
            raise ValueError("Tokenized input bank is too short")
        offset = record["input_seed"] % (len(bank) - count + 1)
        ids = bank[offset:offset + count]
    if len(ids) != recipe["window_tokens"] or any(type(x) is not int or not 0 <= x < vocab_size for x in ids):
        raise ValueError("Materialized tokens are outside the declared shape/vocabulary")
    if record["token_ids_sha256"] is not None and canonical_hash(ids) != record["token_ids_sha256"]:
        raise ValueError("Supplied token identity differs")
    return list(ids)


def listener(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.listen(4)
    sock.settimeout(180)
    return sock


def connect_raw(host, port):
    deadline = time.monotonic() + 180
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(180)
        try:
            sock.connect((host, port))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return sock
        except (ConnectionRefusedError, TimeoutError):
            sock.close()
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def expect(channel, expected, *, count=0):
    message, tensors, stats = channel.recv()
    if message != expected or len(tensors) != count:
        raise ValueError("Unexpected request message or tensor count: " + repr(message))
    return tensors, stats


class Background:
    def __init__(self, target, endpoint):
        self.errors = []
        self.endpoint = endpoint

        def guarded():
            try:
                target()
            except BaseException as error:
                self.errors.append(f"{type(error).__name__}: {error}")
                endpoint.close()

        self.thread = threading.Thread(target=guarded, name="e1-prefix-copy", daemon=True)
        self.thread.start()

    def join(self):
        self.thread.join(timeout=180)
        if self.thread.is_alive() or self.errors:
            self.endpoint.close()
            raise RuntimeError(f"Migration did not finish: {self.errors}")


def replay_store(layers, prefix_length, cutovers):
    """Diagnostic-only visibility at recorded layer/position cutovers.

    KV still travels through the real migration channel. A diagnostic can wait
    for a required layer; its latency is never a performance observation.
    """
    from e0_layered_runtime import PrefixStore

    class RecordedVisibility(PrefixStore):
        def __init__(self):
            super().__init__(layers)
            self.next_positions = [prefix_length] * layers
            self.ready_events = [threading.Event() for _ in range(layers)]
            self.expected_next_layer = 0

        def publish(self, layer, key, value, payload_bytes):
            super().publish(layer, key, value, payload_bytes)
            self.ready_events[layer].set()

        def get(self, layer):
            if type(layer) is not int or layer != self.expected_next_layer:
                raise ValueError("Diagnostic replay requires one get per layer in tokenwise layer order")
            self.expected_next_layer = (layer + 1) % layers
            position = self.next_positions[layer]
            self.next_positions[layer] += 1
            if position < cutovers[layer]:
                return None
            if not self.ready_events[layer].wait(timeout=180):
                raise TimeoutError("Diagnostic replay could not obtain the required migrated layer")
            return super().get(layer)

    return RecordedVisibility()


def cutover_signature(trial, cell, layers):
    start = cell["prefix_tokens"]
    end = start + cell["suffix_tokens"] + cell["generated_tokens"] - 1
    remote_layers = {layer for layer, _ in trial["remote_calls"]}
    transitions = {item["layer"]: item["position"] for item in trial["state_report"]["transitions"]}
    cutovers = [transitions.get(layer, end) if layer in remote_layers else start for layer in range(layers)]
    return {"prefix_tokens": start, "last_input_exclusive": end, "layer_first_local_position": cutovers,
            "sha256": canonical_hash(cutovers)}


def send_prefix(engine, pairs, channel, request_id):
    import torch
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    with torch.inference_mode():
        for index, (key, value) in enumerate(pairs):
            cpu = [torch.empty_like(t, device="cpu", pin_memory=True) for t in (key, value)]
            with torch.cuda.stream(stream):
                for target, tensor in zip(cpu, (key, value)):
                    target.copy_(tensor, non_blocking=True)
                ready = torch.cuda.Event(enable_timing=False)
                ready.record()
            ready.synchronize()
            channel.send({"kind": "prefix", "request_id": request_id, "layer": index}, cpu)
        channel.send({"kind": "prefix_complete", "request_id": request_id, "layers": engine.layers})


def receive_prefix(engine, store, prefix_length, channel, request_id):
    import torch
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    shape = (1, engine.model.config.num_key_value_heads, prefix_length, engine.head_dim)
    with torch.inference_mode():
        for index in range(engine.layers):
            tensors, stats = expect(channel, {"kind": "prefix", "request_id": request_id, "layer": index}, count=2)
            if any(tuple(t.shape) != shape or t.dtype != engine.dtype for t in tensors):
                raise ValueError("Received prefix shape or dtype differs")
            pinned = [t.pin_memory() for t in tensors]
            with torch.cuda.stream(stream):
                gpu = [torch.empty_like(t, device=engine.device) for t in pinned]
                for target, tensor in zip(gpu, pinned):
                    target.copy_(tensor, non_blocking=True)
                ready = torch.cuda.Event(enable_timing=False)
                ready.record()
            ready.synchronize()
            store.publish(index, *gpu, stats["application_payload_bytes"])
        expect(channel, {"kind": "prefix_complete", "request_id": request_id, "layers": engine.layers})


def holder_trial(engine, prefix_pairs, ids, cell, action, request_id, request_listener, diagnostic, peer_hostname):
    import torch
    from e1_channels import Endpoint
    from e1_runtime import HolderAttention
    endpoint = Endpoint(cell["rate_bits_per_second_per_direction"], timeout=180)
    background = None
    attention = None
    result = {"request_id": request_id, "action": action, "remote_calls": []}
    try:
        if action in ("recompute", CONTROL_ACTION):
            endpoint.declare_local_request(request_id, peer_hostname)
            endpoint.close()
            result.update(traffic=endpoint.snapshot(), background_drained=True, passed=True)
            return result
        rpc = endpoint.accept(request_listener, "rpc", request_id)
        bulk = endpoint.accept(request_listener, "bulk", request_id) if action in ("native_fetch", "fixed_progressive_repair") else None
        expect(rpc, {"kind": "begin", "request_id": request_id, "action": action,
                     "prefix_ids": ids[:cell["prefix_tokens"]], "suffix_ids": ids[cell["prefix_tokens"]:],
                     "generated": cell["generated_tokens"]})
        if action in ("native_fetch", "fixed_progressive_repair"):
            background = Background(lambda: send_prefix(engine, prefix_pairs, bulk, request_id), endpoint)
        if action == "holder":
            suffix = torch.tensor([ids[cell["prefix_tokens"]:]], device=engine.device, dtype=torch.long)
            store = engine.store_from_prefix(prefix_pairs)

            def send_token(step, token):
                rpc.send({"kind": "token", "request_id": request_id, "step": step, "token": token})

            decoded = engine.decode(store, cell["prefix_tokens"], suffix, cell["generated_tokens"],
                                    on_token=send_token, capture_logits=diagnostic)
            result["state_report"] = decoded["state_report"]
            result["token_ids"] = decoded["token_ids"]
            if diagnostic:
                rpc.send({"kind": "diagnostic_logits", "request_id": request_id}, [torch.cat(decoded["logits"], dim=0)])
            store.values.clear()
            del store, suffix, decoded
        if action in ("remote_attention", "fixed_progressive_repair"):
            attention = HolderAttention(engine.model, prefix_pairs, cell["prefix_tokens"])
        with torch.inference_mode():
            while True:
                message, tensors, _ = rpc.recv()
                if message == {"kind": "finish", "request_id": request_id} and not tensors:
                    break
                if attention is None or set(message) != {"kind", "request_id", "layer", "position"}:
                    raise ValueError("Unexpected attention request")
                if message["kind"] != "attention" or message["request_id"] != request_id or len(tensors) != 3:
                    raise ValueError("Attention request identity differs")
                layer, position = message["layer"], message["position"]
                query, key, value = [tensor.to(engine.device) for tensor in tensors]
                output = attention.call(layer, position, query, key, value)
                rpc.send({"kind": "attention_result", "request_id": request_id,
                          "layer": layer, "position": position}, [output.cpu()])
                result["remote_calls"].append([layer, position])
                query = key = value = output = None
        if background is not None:
            background.join()
        if attention is not None:
            result["attention_state"] = attention.summarize()
            result["attention_cleanup"] = attention.close()
        torch.cuda.synchronize()
        rpc.send({"kind": "finished", "request_id": request_id})
        endpoint.close()
        result["traffic"] = endpoint.snapshot()
        result["background_drained"] = background is None or not background.thread.is_alive()
        result["passed"] = True
        return result
    finally:
        endpoint.close()
        if attention is not None:
            attention.close()
        if background is not None and background.thread.is_alive():
            background.thread.join(timeout=10)


def destination_trial(engine, prefix, suffix, ids, cell, action, request_id, host, port, diagnostic, replay_cutovers=None, peer_hostname=None):
    import torch
    from e0_layered_runtime import PrefixStore
    from e1_channels import Endpoint
    from e1_acceptance import derive_latency_metrics
    endpoint = None
    background = None
    store = None
    prefix_pairs = None
    tokens, delivery, remote_calls, token_frontiers = [], [], [], []
    result = {"request_id": request_id, "action": action, "diagnostic": diagnostic}

    def delivered(step, token):
        if step != len(tokens) or type(token) is not int or not 0 <= token < engine.model.config.vocab_size:
            raise ValueError("Invalid token delivery order/value")
        tokens.append(token)
        delivery.append(time.monotonic())
        token_frontiers.append(store.snapshot()[0] if store is not None else None)

    accepted = time.monotonic()
    try:
        endpoint = Endpoint(cell["rate_bits_per_second_per_direction"], timeout=180)
        rpc = bulk = None
        if action in ("recompute", CONTROL_ACTION):
            endpoint.declare_local_request(request_id, peer_hostname)
        else:
            rpc = endpoint.connect(host, port, "rpc", request_id)
            bulk = endpoint.connect(host, port, "bulk", request_id) if action in ("native_fetch", "fixed_progressive_repair") else None
            rpc.send({"kind": "begin", "request_id": request_id, "action": action,
                      "prefix_ids": ids[:cell["prefix_tokens"]], "suffix_ids": ids[cell["prefix_tokens"]:],
                      "generated": cell["generated_tokens"]})
        if action == "holder":
            for step in range(cell["generated_tokens"]):
                message, tensors, _ = rpc.recv()
                if (set(message) != {"kind", "request_id", "step", "token"} or
                        message.get("kind") != "token" or message.get("request_id") != request_id or
                        message.get("step") != step or tensors):
                    raise ValueError("Holder token stream identity or order differs")
                delivered(step, message["token"])
            logits = []
            if diagnostic:
                values, _ = expect(rpc, {"kind": "diagnostic_logits", "request_id": request_id}, count=1)
                if tuple(values[0].shape) != (cell["generated_tokens"], engine.model.config.vocab_size) or values[0].dtype != torch.float32:
                    raise ValueError("Diagnostic logits have the wrong shape/type")
                logits = [values[0]]
            decoded = {"token_ids": tokens.copy(), "logits": logits, "state_report": {"execution_host": "holder"}}
        elif action == CONTROL_ACTION:
            decoded = engine.HF_reference(prefix, suffix, cell["generated_tokens"],
                                          on_token=delivered, capture_logits=diagnostic)
        else:
            if action == "recompute":
                prefix_pairs = engine.prefix(prefix)
                store = engine.store_from_prefix(prefix_pairs)
            else:
                store = (replay_store(engine.layers, cell["prefix_tokens"], replay_cutovers)
                         if action == "fixed_progressive_repair" and replay_cutovers is not None else PrefixStore(engine.layers))
            if action == "native_fetch":
                receive_prefix(engine, store, cell["prefix_tokens"], bulk, request_id)
            if action == "fixed_progressive_repair":
                background = Background(lambda: receive_prefix(engine, store, cell["prefix_tokens"], bulk, request_id), endpoint)

            def remote(layer, position, query, key, value):
                rpc.send({"kind": "attention", "request_id": request_id, "layer": layer, "position": position},
                         [t.cpu().contiguous() for t in (query, key, value)])
                rows, _ = expect(rpc, {"kind": "attention_result", "request_id": request_id,
                                      "layer": layer, "position": position}, count=1)
                expected = (1, 1, engine.model.config.num_attention_heads, engine.head_dim)
                if tuple(rows[0].shape) != expected or rows[0].dtype != engine.dtype:
                    raise ValueError("Attention output shape/type differs")
                remote_calls.append([layer, position])
                return rows[0].to(engine.device)

            decoded = engine.decode(store, cell["prefix_tokens"], suffix, cell["generated_tokens"],
                                    remote_attention=remote if action in ("remote_attention", "fixed_progressive_repair") else None,
                                    on_token=delivered, capture_logits=diagnostic)
        if decoded["token_ids"] != tokens:
            raise ValueError("Generated and delivered tokens differ")
        result["frontier_at_final_token_return"] = store.snapshot()[0] if store is not None else None
        if rpc is not None:
            rpc.send({"kind": "finish", "request_id": request_id})
        if background is not None:
            background.join()
        if rpc is not None:
            expect(rpc, {"kind": "finished", "request_id": request_id})
        if store is not None:
            frontier, commits = store.snapshot()
            result.update(final_frontier=frontier, commits=commits)
            store.values.clear()
        if prefix_pairs is not None:
            prefix_pairs.clear()
        torch.cuda.synchronize()
        endpoint.close()
        settled = time.monotonic()
        result.update(token_ids=tokens, accepted_s=accepted, token_delivery_s=delivery, settled_s=settled,
                      token_frontiers=token_frontiers,
                      metrics=derive_latency_metrics(accepted, delivery, settled, cell["generated_tokens"]),
                      state_report=decoded["state_report"], remote_calls=remote_calls,
                      traffic=endpoint.snapshot(), background_drained=background is None or not background.thread.is_alive(),
                      timing_is_performance_evidence=False,
                      diagnostic_logit_payload_transferred=diagnostic and action == "holder")
        return result, decoded["logits"]
    finally:
        if endpoint is not None:
            endpoint.close()
        if background is not None and background.thread.is_alive():
            background.thread.join(timeout=10)


def validate_structural(action, destination, holder, cell, layers):
    from e1_channels import reconcile_pair
    if destination["remote_calls"] != holder["remote_calls"]:
        raise ValueError("The two attention endpoints have different RPC sequences")
    if not destination["background_drained"] or not holder["background_drained"]:
        raise ValueError("A request-owned background thread survived settling")
    for trial in (destination, holder):
        if trial["traffic"].get("no_interhost_request", False) != (action in ("recompute", CONTROL_ACTION)):
            raise ValueError("Only local actions may omit inter-host request traffic")
    total = layers * (cell["suffix_tokens"] + cell["generated_tokens"] - 1)
    if action == "remote_attention" and len(destination["remote_calls"]) != total:
        raise ValueError("Persistent remote action did not execute every layer remotely")
    if action in ("recompute", "native_fetch", "holder", CONTROL_ACTION) and destination["remote_calls"]:
        raise ValueError("Local/holder action unexpectedly used RPC attention")
    if action in ("native_fetch", "fixed_progressive_repair") and destination.get("final_frontier") != layers:
        raise ValueError("A prefix transfer did not commit all layers before settling")
    if action != CONTROL_ACTION:
        state = holder.get("state_report", {}) if action == "holder" else destination["state_report"]
        delta = cell["suffix_tokens"] + cell["generated_tokens"] - 1
        if (state.get("owned_delta_references_released") is not True or
                state.get("final_position") != cell["prefix_tokens"] + delta or
                state.get("delta_lengths") != [[delta, delta] for _ in range(layers)]):
            raise ValueError("Common-engine delta shape/position/ownership check failed")
        counts = state.get("mode_counts", {})
        if (set(counts) != {"local", "remote"} or any(type(v) is not int or v < 0 for v in counts.values()) or
                counts["local"] + counts["remote"] != total or counts["remote"] != len(destination["remote_calls"])):
            raise ValueError("Not all layer calls are accounted for")
        calls = destination["remote_calls"]
        start, end = cell["prefix_tokens"], cell["prefix_tokens"] + delta
        if any(type(layer) is not int or not 0 <= layer < layers or type(position) is not int or
               not start <= position < end for layer, position in calls):
            raise ValueError("RPC layer or position is outside its domain")
        if calls != sorted(calls, key=lambda item: (item[1], item[0])):
            raise ValueError("RPC sequence differs from tokenwise layer execution order")
        remote_layers = {layer for layer, _ in calls}
        cutovers = {}
        for item in state.get("transitions", []):
            layer, position = item["layer"], item["position"]
            if (type(layer) is not int or layer in cutovers or layer not in remote_layers or
                    type(position) is not int or not start < position < end):
                raise ValueError("Duplicate or invalid remote-to-local transition")
            cutovers[layer] = position
        if list(cutovers.items()) != sorted(cutovers.items(), key=lambda item: (item[1], item[0])):
            raise ValueError("Cutover sequence differs from tokenwise layer execution order")
        actual = {(layer, position) for layer, position in calls}
        if len(actual) != len(calls):
            raise ValueError("Duplicated RPC position")
        if action in ("remote_attention", "fixed_progressive_repair"):
            # A layer can be local on its first call if migration won the race.
            expected = {(layer, position) for layer in remote_layers
                        for position in range(cell["prefix_tokens"], cutovers.get(layer, cell["prefix_tokens"] + delta))}
            if actual != expected:
                raise ValueError("Remote delta positions have holes or outlive cutover")
            if holder.get("attention_cleanup", {}).get("owned_delta_references_released") is not True:
                raise ValueError("Holder replica cleanup failed")
    elif destination["state_report"].get("owned_cache_references_released") is not True:
        raise ValueError("Native control retained its request cache")
    return reconcile_pair(destination["traffic"], holder["traffic"])


def validate_numerical(check, tokens, generated, reference_tokens):
    if not all(check.get(key) is True for key in ("finite", "logits_pass", "tokens_match")):
        raise ValueError("Numerical acceptance flags are incomplete")
    metrics = [check.get(key) for key in ("max_abs_error", "max_tolerance_ratio", "min_reference_top2_margin")]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in metrics):
        raise ValueError("Missing, nonfinite or invalid numerical metrics")
    errors = check.get("per_step_max_abs_error")
    if (not isinstance(errors, list) or len(errors) != generated or
            any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in errors) or
            max(errors) != check["max_abs_error"] or check["max_tolerance_ratio"] > 1):
        raise ValueError("Per-token errors or strict tolerance acceptance are inconsistent")
    if (not isinstance(tokens, list) or len(tokens) != generated or
            any(type(token) is not int or token < 0 for token in tokens) or tokens != reference_tokens):
        raise ValueError("Generated tokens differ from the complete same-shape reference")


def performance_gate(root, job, identity, cell, layers):
    """Require both complete diagnostic reports with this exact common runner/input."""
    import subprocess
    if not str(job).isdigit():
        raise ValueError("A completed correctness job ID is required for performance")
    state = subprocess.check_output(["sacct", "-j", str(job), "--allocations", "-n", "-P",
                                     "--format=JobID,State,ExitCode"], text=True).strip().splitlines()
    if len(state) != 1 or state[0].split("|")[:3] != [str(job), "COMPLETED", "0:0"]:
        raise ValueError("Common-runner correctness job did not finish successfully")
    paths = [root / "results" / f"e1-group-{job}" / f"rank-{rank}.json" for rank in (0, 1)]
    reports = [json.loads(path.read_text()) for path in paths]
    for rank, report in enumerate(reports):
        if (report.get("passed") is not True or report.get("mode") != "correctness" or
                report.get("rank") != rank or report.get("slurm_job_id") != str(job) or report.get("identity") != identity):
            raise ValueError("Common-runner correctness identity, mode or outcome differs")
    target = reports[1]
    expected_order = identity["action_order"] + [CONTROL_ACTION]
    if (len(reports[0]["trials"]) != len(expected_order) or
            [trial.get("action") for trial in target["trials"]] != expected_order or
            [trial.get("action") for trial in reports[0]["trials"]] != expected_order):
        raise ValueError("Correctness evidence does not cover all five actions and the HF control")
    for trial, holder in zip(target["trials"], reports[0]["trials"]):
        check = trial.get("numerical_check", {})
        if not all(check.get(key) is True for key in ("finite", "logits_pass", "tokens_match")) or trial.get("passed") is not True:
            raise ValueError("Common-runner numerical acceptance is incomplete")
        validate_numerical(check, trial.get("token_ids"), cell["generated_tokens"], target["trials"][-1].get("token_ids"))
        if (holder.get("passed") is not True or holder.get("request_id") != trial.get("request_id") or
                trial.get("structural_checks_passed") is not True or
                trial.get("traffic_reconciliation") != validate_structural(trial["action"], trial, holder, cell, layers)):
            raise ValueError("Common-runner structural evidence is inconsistent")
        if trial["action"] == "fixed_progressive_repair" and trial.get("cutover_signature") != cutover_signature(trial, cell, layers):
            raise ValueError("Correctness cutover signature differs from its actual validated path")
    return {"job_id": str(job), "report_sha256": {path.name: digest(path) for path in paths},
            "expected_tokens_by_action": {trial["action"]: trial["token_ids"] for trial in target["trials"]},
            "repair_cutover_signature": next(trial["cutover_signature"] for trial in target["trials"]
                                               if trial["action"] == "fixed_progressive_repair")}


def load_replay_candidate(root, job, identity, cell, layers):
    import subprocess
    if not str(job).isdigit():
        raise ValueError("A numeric performance candidate job ID is required")
    state = subprocess.check_output(["sacct", "-j", str(job), "--allocations", "-n", "-P",
                                     "--format=JobID,State,ExitCode"], text=True).strip().splitlines()
    if len(state) != 1 or state[0].split("|")[:3] != [str(job), "COMPLETED", "0:0"]:
        raise ValueError("Performance candidate job has not completed")
    paths = [root / "results" / f"e1-group-{job}" / f"rank-{rank}.json" for rank in (0, 1)]
    reports = [json.loads(path.read_text()) for path in paths]
    order = identity["action_order"] + [CONTROL_ACTION]
    for rank, report in enumerate(reports):
        if (report.get("rank") != rank or report.get("slurm_job_id") != str(job) or
                report.get("passed") is not True or report.get("mode") != "performance" or report.get("identity") != identity or
                [trial.get("action") for trial in report.get("trials", [])] != order):
            raise ValueError("Performance candidate has incomplete or inconsistent evidence")
    for target, holder in zip(reports[1]["trials"], reports[0]["trials"]):
        if target.get("passed") is not True or holder.get("passed") is not True:
            raise ValueError("A candidate trial failed")
        validate_structural(target["action"], target, holder, cell, layers)
    trial = next(trial for trial in reports[1]["trials"] if trial["action"] == "fixed_progressive_repair")
    signature = cutover_signature(trial, cell, layers)
    if trial.get("cutover_signature") != signature:
        raise ValueError("Candidate cutover identity is inconsistent")
    return {"job_id": str(job), "report_sha256": {path.name: digest(path) for path in paths}, "cutover_signature": signature}


def run(args, report):
    import subprocess
    import sys
    from e1_wire import FramedConnection
    from e1_transport import PacedSocket, RateLimiter
    root = Path(args.root).resolve()
    rank = int(os.environ.get("SLURM_PROCID", "-1"))
    if rank not in (0, 1) or os.environ.get("SLURM_NTASKS") != "2":
        raise ValueError("Run only as two scheduler tasks, one GPU each")
    frozen = Path(os.environ["RESTATE_E1_DIR"]).resolve()
    if frozen != Path(__file__).resolve().parent:
        raise ValueError("Runner must execute its immutable submitted snapshot")
    submission = json.loads((frozen / "submission.json").read_text())
    if str(submission.get("job_id")) != report["slurm_job_id"]:
        raise ValueError("Submission/job identity mismatch")
    if submission.get("arguments") != sys.argv[1:]:
        raise ValueError("Runtime arguments differ from the immutable submission")
    for name, expected in submission["source_sha256"].items():
        if Path(name).name != name or digest(frozen / name) != expected:
            raise ValueError("Frozen source identity changed: " + name)
    if not set(EXECUTION_SOURCES).issubset(submission["source_sha256"]):
        raise ValueError("Incomplete frozen execution source closure")
    hosts = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True).split()
    if sorted(hosts) != ["babel-v5-20", "babel-v5-24"]:
        raise ValueError("Use the same two validated A100 physical hosts")
    report["E0_evidence"] = stage_gate(root, args.e0_suite_job)
    plan, cell, group, record = load_group(frozen / args.plan, args.cell, args.phase, args.repetition)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from e0_holder_decode import verify_manifest, compare
    from e1_runtime import Engine
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one visible allocated GPU is required")
    torch.cuda.set_device(0)
    gpu = torch.cuda.get_device_name(0)
    if "A100" not in gpu or torch.cuda.get_device_properties(0).total_memory < 79_000_000_000:
        raise ValueError("The main protocol requires A100 80GB")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    model_path = Path(args.model).resolve()
    manifest, manifest_sha = verify_manifest(model_path)
    if manifest["revision"] != MODEL_REVISION or manifest_sha != MODEL_MANIFEST:
        raise ValueError("Main model manifest differs")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, attn_implementation="eager").to("cuda:0").eval()
    engine = Engine(model)
    ids = materialize(record, tokenizer, model.config.vocab_size)
    all_ids = torch.tensor([ids], dtype=torch.long, device=engine.device)
    prefix, suffix = all_ids[:, :cell["prefix_tokens"]], all_ids[:, cell["prefix_tokens"]:]
    identity = {"source_sha256": {name: digest(frozen / name) for name in EXECUTION_SOURCES},
                "model_revision": MODEL_REVISION, "model_manifest_sha256": MODEL_MANIFEST,
                "plan_sha256": digest(frozen / args.plan), "cell_id": cell["cell_id"],
                "pair_id": group["pair_id"], "input_ids_sha256": canonical_hash(ids),
                "input_ids": ids, "action_order": group["action_order"], "gpu": gpu,
                "dtype": "bfloat16", "tensor_parallel": 1, "attention": "eager",
                "hosts": sorted(hosts),
                "torch_version": torch.__version__, "transformers_version": transformers.__version__,
                "cuda_version": torch.version.cuda, "tf32": False,
                "bf16_reduced_precision_reduction": False, "fp16_reduced_precision_reduction": False,
                "repair_schedule": "start_at_begin_ascending_layers_drain_after_last_token",
                "rate_scope": "shared_per_direction_application_socket_bytes", "rate": cell["rate_bits_per_second_per_direction"],
                "framing_integrity_mode": "none", "instrumentation": "NullTimeline_no_profiler_events"}
    identity["local_action_transport"] = "no_interhost_request_connections_or_completion_round_trip"
    report.update(identity=identity, gpu=gpu, cell=cell["cell_id"], pair_id=group["pair_id"],
                  precision="bfloat16", hardware_role="holder" if rank == 0 else "destination",
                  performance_evidence=False,
                  sample_role=args.phase, E1_complete=False)
    evidence = performance_gate(root, args.correctness_job, identity, cell, engine.layers) if args.mode == "performance" else None
    report["correctness_evidence"] = evidence
    if args.replay_performance_job and args.mode != "correctness":
        raise ValueError("Recorded cutover replay is a separate correctness run only")
    replay = load_replay_candidate(root, args.replay_performance_job, identity, cell, engine.layers) if args.replay_performance_job else None
    report["replay_candidate"] = replay
    # Warm both tokenwise engines and full-prefix kernels, then remove request KV.
    warm_pairs = engine.prefix(prefix)
    warm_store = engine.store_from_prefix(warm_pairs)
    engine.decode(warm_store, cell["prefix_tokens"], suffix, 2, capture_logits=False)
    warm_store.values.clear()
    warm_pairs.clear()
    del warm_store, warm_pairs
    engine.HF_reference(prefix, suffix, 2, capture_logits=False)
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_port = int(os.environ["MASTER_PORT"])
    control_listener = request_listener = None
    raw_control = None
    control = None
    try:
        if rank == 0:
            control_listener, request_listener = listener(base_port), listener(base_port + 1)
            raw_control, _ = control_listener.accept()
            raw_control.settimeout(180)
        else:
            raw_control = connect_raw(os.environ["MASTER_ADDR"], base_port)
        control = FramedConnection(PacedSocket(raw_control, RateLimiter(0), "outside-request-control"), integrity_mode="none")
        hello = {"kind": "run_identity", "identity": identity, "hostname": socket.gethostname(), "mode": args.mode}
        control.send_message(hello)
        peer, tensors, _ = control.recv_message()
        if (peer.get("identity") != identity or peer.get("mode") != args.mode or tensors or
                peer.get("hostname") == socket.gethostname()):
            raise ValueError("The peer is not a matching worker on a distinct host")
        report["peer_hostname"] = peer["hostname"]
        report["baseline_rtt"] = baseline_rtt(control, rank)
        diagnostic = args.mode == "correctness"
        reference = engine.HF_reference(prefix, suffix, cell["generated_tokens"]) if diagnostic and rank == 1 else None
        for index, action in enumerate(group["action_order"] + [CONTROL_ACTION]):
            request_id = f"e1-{report['slurm_job_id']}-{index}-{action}"
            if rank == 0:
                pairs = engine.prefix(prefix)
                torch.cuda.synchronize()
                control.send_message({"kind": "ready", "request_id": request_id})
                trial = holder_trial(engine, pairs, ids, cell, action, request_id, request_listener, diagnostic, peer["hostname"])
                report["trials"].append(trial)
                control.send_message({"kind": "holder_report", "request_id": request_id, "report": trial})
                message, tensors, _ = control.recv_message()
                if message != {"kind": "accepted", "request_id": request_id} or tensors:
                    raise ValueError("Destination rejected a paired trial")
                pairs.clear()
                del pairs
            else:
                ready, tensors, _ = control.recv_message()
                if ready != {"kind": "ready", "request_id": request_id} or tensors:
                    raise ValueError("Holder prefix was not declared ready")
                trial, logits = destination_trial(engine, prefix, suffix, ids, cell, action, request_id,
                                                  os.environ["MASTER_ADDR"], base_port + 1, diagnostic,
                                                  replay["cutover_signature"]["layer_first_local_position"] if replay else None,
                                                  peer_hostname=peer["hostname"])
                trial.update(passed=False, sample_role=args.phase,
                             measurement_kind="native_engine_control" if action == CONTROL_ACTION else "action",
                             timing_is_performance_evidence=False)
                report["trials"].append(trial)
                message, tensors, _ = control.recv_message()
                if message.get("kind") != "holder_report" or message.get("request_id") != request_id or tensors:
                    raise ValueError("Missing holder report after settling")
                holder = message["report"]
                trial["traffic_reconciliation"] = validate_structural(action, trial, holder, cell, engine.layers)
                trial["structural_checks_passed"] = True
                if action == "fixed_progressive_repair":
                    trial["cutover_signature"] = cutover_signature(trial, cell, engine.layers)
                    if replay is not None and trial["cutover_signature"] != replay["cutover_signature"]:
                        raise ValueError("Diagnostic cutover schedule differs from the performance candidate")
                trial["holder_state"] = {key: value for key, value in holder.items() if key not in ("traffic", "remote_calls")}
                if diagnostic:
                    check = compare(torch, torch.cat(reference["logits"], dim=0), torch.cat(logits, dim=0),
                                    reference["token_ids"], trial["token_ids"])
                    trial["numerical_check"] = check
                    if not all(check[key] for key in ("finite", "logits_pass", "tokens_match")):
                        raise ValueError("Strict common-engine numerical acceptance failed")
                    validate_numerical(check, trial["token_ids"], cell["generated_tokens"], reference["token_ids"])
                elif trial["token_ids"] != evidence["expected_tokens_by_action"][action]:
                    raise ValueError("Performance output differs from accepted correctness output")
                trial["passed"] = True
                if not diagnostic:
                    same_path = action != "fixed_progressive_repair" or trial["cutover_signature"] == evidence["repair_cutover_signature"]
                    trial["all_vocab_path_validation"] = "matching_prior_diagnostic" if same_path else "pending_recorded_cutover_diagnostic"
                    trial["timing_is_performance_evidence"] = args.phase == "measured" and same_path
                control.send_message({"kind": "accepted", "request_id": request_id})
            save(args.output / f"rank-{rank}.json", report)
        report.update(passed=True, status="passed_selected_group",
                      performance_evidence=rank == 1 and args.mode == "performance" and args.phase == "measured"
                        and all(trial.get("timing_is_performance_evidence") for trial in report["trials"]),
                      gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    finally:
        if control is not None:
            control.close()
        elif raw_control is not None:
            raw_control.close()
        if control_listener is not None:
            control_listener.close()
        if request_listener is not None:
            request_listener.close()


def entrypoint():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--phase", choices=("warmup", "measured"), required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--mode", choices=("correctness", "performance"), required=True)
    parser.add_argument("--e0-suite-job", required=True)
    parser.add_argument("--correctness-job")
    parser.add_argument("--replay-performance-job")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if Path(args.plan).name != args.plan or not str(args.e0_suite_job).isdigit():
        parser.error("Plan must be a frozen filename and E0 suite ID must be numeric")
    rank = int(os.environ.get("SLURM_PROCID", "-1"))
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit() or rank not in (0, 1):
        parser.error("A real Slurm job and two ranks are required")
    expected_output = args.root.resolve() / "results" / f"e1-group-{job}"
    if args.output.resolve() != expected_output:
        parser.error("Output must be this job's isolated result directory")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"rank": rank, "slurm_job_id": job, "hostname": socket.gethostname(),
              "mode": args.mode, "passed": False, "status": "running", "trials": [],
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "physical_wire_bytes": None, "wire_measurement_available": False}
    code = 1
    try:
        run(args, report)
        code = 0
    except BaseException as error:
        report.update(passed=False, status="failed", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save(args.output / f"rank-{rank}.json", report)
    return code


if __name__ == "__main__":
    raise SystemExit(entrypoint())
