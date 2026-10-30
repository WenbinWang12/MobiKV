"""Destination execution for the bounded E0--E7 campaign.

All action costs are measured executions. A separate replay validates the exact
remote/local layer path of every measured candidate; diagnostic time is discarded.
"""
from __future__ import annotations

from dataclasses import asdict
import copy
import json
import math
import socket
import threading
import time
import weakref

from e0_layered_runtime import PrefixStore
from e1_wire import FramedConnection
from e1_runner import canonical_hash, replay_store, validate_numerical
from e2_control import (ProgressState, ExecutionCapabilities, EmpiricalRemainingEstimator,
                        CostObservation, ProgressPolicy)
from e3_perturbation import RatePlan, VariableRateLimiter
from e4_e6_mechanisms import SharedBandwidthCoordinator


def coordinator(rate, priority, plan=None):
    if plan:
        variable = RatePlan(plan["steps"], plan_id=plan["plan_id"], seed=plan["seed"])
        return SharedBandwidthCoordinator(limiter=VariableRateLimiter(variable,
            epoch_monotonic=time.monotonic(), chunk_bytes=16384), priority=priority)
    return SharedBandwidthCoordinator(rate, chunk_bytes=16384, priority=priority)


class Channel:
    def __init__(self, host, port, budget, request, kind, *, expected_peer=None, timeout=180, owned_resources=None):
        raw = socket.create_connection((host, port), timeout=timeout)
        raw.settimeout(timeout)
        self.socket = self.connection = None
        self.request = request
        self.counts = {"sent": 0, "received": 0, "payload_sent": 0, "payload_received": 0,
                       "sent_frames": 0, "received_frames": 0}
        try:
            if owned_resources is not None:
                owned_resources.register(raw)
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socket = budget.wrap(raw, request + "/" + kind, kind)
            if owned_resources is not None:
                owned_resources.register(self.socket)
            self.connection = FramedConnection(self.socket, integrity_mode="none")
            self.send("begin" if kind == "rpc" else "hello")
            message, tensors = self.recv(tensor_format="bytes")
            if message.get("kind") != ("begin_ready" if kind == "rpc" else "bulk_ready") or tensors:
                raise ValueError("Request setup acknowledgement differs")
            if expected_peer is not None and any(message.get(field) != value for field, value in expected_peer.items()):
                raise ValueError("Holder prefix/model setup identity differs")
            self.setup_ack = message
        except BaseException:
            if self.connection is not None:
                self.connection.close()
            elif self.socket is not None:
                self.socket.close()
            else:
                raw.close()
            raise

    def send(self, kind, tensors=(), **fields):
        stats = self.connection.send_message({"kind": kind, "request_id": self.request, **fields}, tensors)
        self.counts["sent"] += stats["application_frame_bytes"]
        self.counts["payload_sent"] += stats["application_payload_bytes"]
        self.counts["sent_frames"] += 1

    def recv(self, *, tensor_format="torch"):
        message, tensors, stats = self.connection.recv_message(tensor_format=tensor_format)
        self.counts["received"] += stats["application_frame_bytes"]
        self.counts["payload_received"] += stats["application_payload_bytes"]
        self.counts["received_frames"] += 1
        if message.get("request_id") != self.request:
            raise ValueError("Cross-request frame received")
        return message, tensors

    def command(self, kind, expected, **fields):
        self.send(kind, **fields)
        message, tensors = self.recv(tensor_format="bytes")
        if message.get("kind") != expected or tensors:
            raise ValueError("Unexpected control acknowledgement: " + repr(message))
        return message

    def close(self):
        self.connection.close()

    def audit(self):
        actual = self.socket.snapshot()
        if actual["sent_bytes"] != self.counts["sent"] or actual["received_bytes"] != self.counts["received"]:
            raise ValueError("Completed client frames differ from actual socket byte counters")
        return {"frame_accounting_matches_actual_socket_bytes": True, "socket_closed": self.connection.fileno() == -1,
                "transport": actual}


def validate_control_script(case):
    script = case.get("control_script", [])
    if not isinstance(script, list):
        raise ValueError("control_script must be a list")
    previous, stopped = 0, False
    result = []
    for entry in script:
        if not isinstance(entry, dict) or set(entry) != {"generated_count", "command"}:
            raise ValueError("Control entries require exactly generated_count and command")
        count, command = entry["generated_count"], entry["command"]
        if (type(count) is not int or not previous < count <= case["generated"] or
                command not in ("pause", "resume", "stop") or stopped):
            raise ValueError("Control script order, token boundary, or post-stop command is invalid")
        if case.get("checkpoint_generated") is not None and count <= case["checkpoint_generated"]:
            raise ValueError("Scripted migration controls must follow the matched-state checkpoint")
        previous, stopped = count, command == "stop"
        result.append(dict(entry))
    if result and case["action"] in ("holder", "native_hf_control", "recompute", "local_reuse", "remote_attention"):
        raise ValueError("This action has no asynchronous migration for a control script")
    if type(case.get("initially_paused", False)) is not bool:
        raise ValueError("initially_paused must be a boolean")
    return tuple(result)


def validate_migration_terminal(message, *, start, end, next_layer, transfer_index, payload_bytes):
    expected = {"kind": "migration_terminal", "start": start, "end": end,
                "next_layer": next_layer, "transfer_index": transfer_index, "payload_bytes": payload_bytes}
    if any(message.get(field) != value for field, value in expected.items()):
        raise ValueError("Migration terminal does not match complete received frames")
    if (any(type(message.get(field)) is not int for field in ("start", "end", "next_layer", "transfer_index", "payload_bytes"))
            or type(message.get("stopped")) is not bool or not start <= next_layer <= end
            or (not message["stopped"] and next_layer != end)):
        raise ValueError("Migration terminal bounds/completion status are invalid")
    return {key: message[key] for key in ("transfer_index", "start", "end", "next_layer", "payload_bytes")}


def validate_control_ack(message, command, transfer_index):
    if (message.get("kind") != command + "_ack" or type(message.get("applied")) is not bool or
            type(message.get("transfer_index")) is not int or message["transfer_index"] != transfer_index):
        raise ValueError("Migration control acknowledgement identity/applied status differs")
    migration = message.get("migration", {})
    mode = migration.get("state")
    if message["applied"]:
        expected = {"pause": ("paused", "complete"), "resume": ("running", "complete"),
                    "stop": ("stopping", "stopped", "complete")}[command]
        if mode not in expected:
            raise ValueError("Applied control acknowledgement has an inconsistent migration state")
    elif not message.get("noop_reason"):
        raise ValueError("A control no-op must report its reason")
    if not message.get("ack_scope"):
        raise ValueError("Migration control acknowledgement must distinguish command receipt from frame drain")
    return mode


def send_migration_control(rpc, command, transfer_index, generated_count, *, diagnostic=False, reason="script"):
    sent_at = time.monotonic()
    acknowledgement = rpc.command(command, command + "_ack")
    received_at = time.monotonic()
    mode = validate_control_ack(acknowledgement, command, transfer_index)
    event = {"generated_count": generated_count, "command": command, "reason": reason,
             "sent_s": sent_at, "ack_s": received_at, "elapsed_s": received_at - sent_at,
             "applied": acknowledgement["applied"], "acknowledgement": acknowledgement,
             "timing_is_measurement": not diagnostic}
    return mode, event


def state_signature(result, case, layers):
    start = case["prefix"]
    end = start + case["suffix"] + case["generated"] - 1
    remote = {layer for layer, _ in result["remote_calls"]}
    switches = {entry["layer"]: entry["position"] for entry in result["state_report"]["transitions"]}
    return [switches.get(layer, end) if layer in remote else start for layer in range(layers)]


def validate_decode_state(state, remote_calls, case, layers):
    delta = case["suffix"] + case["generated"] - 1
    start, end = case["prefix"], case["prefix"] + delta
    if case["action"] == "native_hf_control":
        if (remote_calls or state.get("owned_cache_references_released") is not True or
                state.get("final_cache_lengths") != [end] * layers or state.get("final_position") != end):
            raise ValueError("Native HF control cache cleanup/length differs")
        return
    if (state.get("owned_delta_references_released") is not True or state.get("final_position") != end or
            state.get("delta_lengths") != [[delta, delta] for _ in range(layers)]):
        raise ValueError("Campaign decoder delta length/position/cleanup differs")
    counts = state.get("mode_counts", {})
    if (set(counts) != {"local", "remote"} or any(type(value) is not int or value < 0 for value in counts.values())
            or counts["local"] + counts["remote"] != layers * delta or counts["remote"] != len(remote_calls)):
        raise ValueError("Campaign layer calls are not fully accounted for")
    if any(type(layer) is not int or not 0 <= layer < layers or type(position) is not int or not start <= position < end
           for layer, position in remote_calls):
        raise ValueError("Campaign RPC layer/position domain differs")
    if remote_calls != sorted(remote_calls, key=lambda item: (item[1], item[0])) or len(set(map(tuple, remote_calls))) != len(remote_calls):
        raise ValueError("Campaign RPC sequence is duplicated or out of tokenwise order")
    remote_layers = {layer for layer, _ in remote_calls}
    cutovers = {}
    for item in state.get("transitions", []):
        layer, position = item["layer"], item["position"]
        if (type(layer) is not int or layer not in remote_layers or layer in cutovers or
                type(position) is not int or not start < position < end):
            raise ValueError("Campaign cutover is duplicate or invalid")
        cutovers[layer] = position
    if list(cutovers.items()) != sorted(cutovers.items(), key=lambda item: (item[1], item[0])):
        raise ValueError("Campaign cutovers are out of tokenwise order")
    expected = {(layer, position) for layer in remote_layers for position in range(start, cutovers.get(layer, end))}
    if set(map(tuple, remote_calls)) != expected:
        raise ValueError("Campaign remote delta has holes or RPC after cutover")
    if case["action"] in ("recompute", "local_reuse", "native_fetch", "holder") and remote_calls:
        raise ValueError("A fully local/holder action used remote attention")


def _e2_destination_view(snapshot):
    """Canonical tensor/history/route view; never physical replay migration."""
    from campaign_decode import _EVENT_FIELDS, _PHYSICAL_VIEW_FIELDS
    if snapshot.get("content_hashes_captured") is not True:
        raise ValueError("E2 content matching requires actual destination tensor hashes")
    for key in ("delta_sha256", "committed_prefix_sha256"):
        value = snapshot.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Missing destination content SHA")
    original = {k: v for k, v in snapshot.items() if k not in _EVENT_FIELDS}
    if canonical_hash(original) != snapshot.get("state_sha256"):
        raise ValueError("Destination checkpoint was changed after capture")
    return {k: copy.deepcopy(v) for k, v in original.items() if k not in _PHYSICAL_VIEW_FIELDS}


def _e2_socket_totals(budget):
    connections = budget.snapshot().get("limiter", {}).get("connections")
    if not isinstance(connections, dict) or not connections:
        raise ValueError("E2 prospective boundary requires actual shared socket counters")
    result = {"sent_bytes": 0, "received_bytes": 0}
    for item in connections.values():
        for name in result:
            value = item.get(name)
            if type(value) is not int or value < 0:
                raise ValueError("Invalid actual socket byte counter")
            result[name] += value
    return result


def apply_e2_transition(selected, generated_count, *, layers, snapshot,
                        start_transfer, finish_transfer, stop_transfer,
                        recompute_publish, mark_stopped):
    """Install one real continuation through injected execution operations.

    The hooks are the same socket/GPU operations used by execute, not a cost
    simulator. Keeping ordering here also permits CPU protocol-order tests.
    """
    before, operations = snapshot(), []
    if before["errors"]:
        raise RuntimeError("Cannot transition after migration error: " + repr(before["errors"]))
    if selected in ("continue_repair", "complete_fetch_then_local"):
        if before["stopped_irreversibly"]:
            raise ValueError("Stopped migration cannot be silently restarted")
        if selected == "continue_repair" and before["receiver_thread_alive"]:
            operations.append({"kind":"retain_actual_running_transfer","transfer":before["transfer"]})
        else:
            if before["transfer"] is not None:
                finish_transfer()
                if selected == "complete_fetch_then_local":
                    operations.append({"kind":"wait_and_drain_existing_transfer","terminal":snapshot()["terminal"]})
            frontier = snapshot()["physical_frontier"]
            if frontier < layers:
                blocking = selected == "complete_fetch_then_local"
                start_transfer(frontier, layers, wait=blocking)
                now = snapshot()
                operations.append({"kind":"fetch_and_drain_remaining","terminal":now["terminal"]} if blocking else
                                  {"kind":"start_remaining_transfer","transfer":now["transfer"]})
            elif selected == "continue_repair":
                operations.append({"kind":"already_all_local","observed_frontier":frontier})
            if selected == "complete_fetch_then_local" and snapshot()["physical_frontier"] != layers:
                raise RuntimeError("Fetch did not publish every GPU-ready prefix layer")
    elif selected in ("stop_migration_and_remote", "recompute_full_prefix_keep_delta"):
        if before["transfer"] is not None and not before["drained"]:
            acknowledgement = stop_transfer(generated_count, selected)
            operations.append({"kind":"source_stop_command","acknowledgement":copy.deepcopy(acknowledgement)})
            finish_transfer()
            operations.append({"kind":"complete_frame_receiver_drain","terminal":snapshot()["terminal"]})
        else:
            operations.append({"kind":"no_active_transfer_to_stop","actual":snapshot()})
        mark_stopped()
        if selected == "recompute_full_prefix_keep_delta":
            operations.append(recompute_publish())
    elif selected == "continue_local":
        if before["physical_frontier"] != layers:
            raise ValueError("continue_local requires every actual prefix tensor GPU-ready")
        operations.append({"kind":"retain_actual_all_local","observed_frontier":layers})
    else:
        raise ValueError("Unsupported E2 transition")
    return {"before":before,"operations":operations,"after":snapshot(),
        "application_scope":"chosen continuation installed or retained; individual source command ACKs preserve applied/noop truth"}


class E2CheckpointSession:
    """State/decision bridge. Transport and actual transitions are injected hooks.

    Policies receive only controller.STATE_FIELDS. Diagnostic replay has no
    policy callback and cannot claim matching physical in-flight migration.
    """
    def __init__(self, engine, case, request, budget, rpc, remote_calls, operations,
                 migration_snapshot, *, diagnostic, decide, ack, replay_decisions, checkpoint_marks):
        import e2_controller as controller
        self.controller = controller
        self.engine, self.case, self.request = engine, case, request
        self.budget, self.rpc, self.remote_calls = budget, rpc, remote_calls
        self.operations, self.migration_snapshot = operations, migration_snapshot
        self.diagnostic, self.decide, self.ack = diagnostic, decide, ack
        self.replay = replay_decisions
        self.checkpoint_marks = checkpoint_marks
        self.current = "continue_repair"
        self.checkpoints, self.decisions = [], []
        self.points = tuple(case["checkpoints_generated"])
        self.module = case["e2_module"]
        if self.module not in ("conditional", "continuous") or not self.points:
            raise ValueError("Unknown/empty E2 checkpoint module")
        expected_mode = "content" if diagnostic or self.module == "conditional" else "metadata"
        if case.get("checkpoint_witness_mode") != expected_mode:
            raise ValueError("E2 content/metadata capture mode differs from declared module")
        if diagnostic:
            if decide is not None or ack is not None or not isinstance(self.replay, list) or len(self.replay) != len(self.points):
                raise ValueError("E2 diagnostic requires all recorded decisions and no live policy")
        elif not callable(decide) or not callable(ack) or replay_decisions is not None:
            raise ValueError("E2 measured execution requires causal decision/ACK callbacks")

    def __call__(self, step, decoder, token_ids):
        from checkpoint_witness import validated_holder_state
        index, d = len(self.checkpoints), len(token_ids)
        if index >= len(self.points) or self.points[index] != d or step != d-1:
            raise ValueError("Unexpected E2 checkpoint order")
        destination = copy.deepcopy(decoder.campaign_checkpoint)
        audit_started = time.monotonic()
        destination_quiescence = None
        frontier = destination["prefix_frontier"]
        processed = self.case["suffix"]+d-1
        if destination["position"] != self.case["prefix"]+processed or destination["delta_lengths"] != [[processed, processed]]*self.engine.layers:
            raise ValueError("Actual destination delta/history checkpoint differs")
        if self.module == "continuous":
            prospective, before_bytes = self.checkpoint_marks[d]
        else:
            prospective = before_bytes = None
            migration = self.migration_snapshot()
            if migration["receiver_thread_alive"] or not migration["drained"]:
                raise ValueError("Conditional history must already be completely drained, not stopped by this audit")
            queue = self.engine.compute_queue.state() if self.engine.compute_queue is not None else None
            if queue is not None and (queue["active_gpu_operations"] or queue["queued_gpu_operations"]):
                raise ValueError("Conditional destination has pending GPU work")
            destination_quiescence = {"migration":migration,"gpu_queue":copy.deepcopy(queue),
                "no_receiver_thread_or_pending_frame":True,"observed_at_token_boundary":True}
        # Continuous main decisions need no holder feature. An observation RPC
        # here would impose a decision-only decode pause on sticky baselines.
        holder_mode = "content" if self.module == "conditional" else "metadata" if self.diagnostic else None
        witness = holder = None
        holder_audit_started = time.monotonic() if holder_mode is not None else None
        holder_audit_finished = None
        if holder_mode is not None:
            reply = self.rpc.command("e2_checkpoint_witness", "e2_checkpoint_witness_ack", mode=holder_mode,
                                     processed_delta_tokens=processed, committed_layers=frontier)
            witness = reply.get("witness")
            if not isinstance(witness, dict) or canonical_hash({k:v for k,v in witness.items() if k != "witness_sha256"}) != witness.get("witness_sha256"):
                raise ValueError("Holder checkpoint artifact hash differs")
            holder = witness["scientific_state"]
            if (witness.get("mode") != holder_mode or witness.get("provenance", {}).get("request_id") != self.request or
                    holder.get("prefix_token_ids_sha256") != canonical_hash(destination["prefix_token_ids"]) or
                    holder.get("model") != self.engine.model_identity or
                    holder.get("prefix_length") != self.case["prefix"] or
                    holder.get("processed_delta_tokens") != processed or
                    holder.get("committed_layers") != frontier or
                    holder.get("actual_RPC_history") != [{"layer":layer,"position":position} for layer,position in self.remote_calls]):
                raise ValueError("Holder witness does not bind this actual request/history/model")
            if holder_mode == "content":
                holder = validated_holder_state(witness)
            holder_audit_finished = time.monotonic()
        physical = self.migration_snapshot()
        legal = (["continue_local"] if frontier == self.engine.layers else
                 ["stop_migration_and_remote", "recompute_full_prefix_keep_delta"] if physical["stopped_irreversibly"] else
                 ["continue_repair", "stop_migration_and_remote", "complete_fetch_then_local", "recompute_full_prefix_keep_delta"])
        item_bytes = {"torch.bfloat16":2,"torch.float32":4}.get(str(self.engine.dtype))
        if item_bytes is None:
            raise ValueError("Unsupported E2 model dtype")
        per_layer = 2*self.engine.kv_heads*self.case["prefix"]*self.engine.head_dim*item_bytes
        state = {"prefix_tokens":self.case["prefix"],"suffix_tokens":self.case["suffix"],
            "public_output_cap":self.case["public_output_cap"],"generated_count":d,
            "committed_layers":frontier,"total_layers":self.engine.layers,
            "prefix_KV_bytes":per_layer*self.engine.layers,"committed_KV_bytes":per_layer*frontier,
            "application_cap_bps":self.case["application_cap_bps"],"load_location":self.case["load_location"],
            "processed_delta_tokens":processed,"position":decoder.position,"legal_actions":legal,
            "current_action":"continue_local" if frontier == self.engine.layers else self.current}
        if self.diagnostic:
            record = self.replay[index]
            if not isinstance(record, dict) or set(record) != {"state", "proposal", "ack"}:
                raise ValueError("Recorded E2 decision schema differs")
            recorded = record["state"]
            self.controller.validate_state(recorded)
            # The logical frontier was borrowed from actually received KV. No
            # diagnostic claim is made about main physical transfer/queue state.
            for field in state:
                if field not in ("legal_actions", "current_action") and state[field] != recorded[field]:
                    raise ValueError("Recorded checkpoint observable geometry differs")
            if self.module == "conditional" and state != recorded:
                raise ValueError("Conditional replay legal/current state differs")
            state = copy.deepcopy(recorded)
        self.controller.validate_state(state)
        if set(state) != self.controller.STATE_FIELDS:
            raise ValueError("Unexpected future-bearing E2 controller input")
        matched_payload = None
        if destination["content_hashes_captured"]:
            matched_payload = {"scope":"conditional_both_endpoints" if holder_mode == "content" else
                "diagnostic_destination_content_and_route_only", "destination":_e2_destination_view(destination)}
            if holder_mode == "content":
                matched_payload["holder"] = holder
            else:
                matched_payload["holder_RPC_history"] = holder["actual_RPC_history"]
        if self.module == "conditional":
            prospective = time.monotonic()
            before_bytes = _e2_socket_totals(self.budget)
        audit_finished = time.monotonic()
        if self.diagnostic:
            proposal = copy.deepcopy(record["proposal"])
            ack = record["ack"]
            if (set(ack) != {"applied","action","state_sha256","evidence_sha256"} or ack["applied"] is not True or
                    ack["action"] != proposal.get("action") or ack["state_sha256"] != canonical_hash(state)):
                raise ValueError("Recorded proposal/ACK is not bound to checkpoint state")
        else:
            proposal = json.loads(json.dumps(self.decide(copy.deepcopy(state)), allow_nan=False))
        policy_finished = time.monotonic()
        selected = proposal.get("action")
        if proposal.get("future_information_used") is not False or selected not in state["legal_actions"] or selected not in self.operations:
            raise ValueError("Policy proposed an illegal/unimplemented or future-informed action")
        before_delta = [(id(t),t.data_ptr(),tuple(t.shape)) for pair in decoder.delta for t in pair]
        before_position = decoder.position
        evidence = self.operations[selected](d)
        after_delta = [(id(t),t.data_ptr(),tuple(t.shape)) for pair in decoder.delta for t in pair]
        if before_delta != after_delta or decoder.position != before_position:
            raise RuntimeError("E2 transition changed the pending-token delta or position")
        evidence["delta_preservation"] = {"same_tensor_identity_storage_shape":True,
            "position_before":before_position,"position_after":decoder.position,
            "delta_lengths":[[int(t.shape[-2]) for t in pair] for pair in decoder.delta]}
        evidence.update(action=selected, generated_count=d, holder_checkpoint_witness_sha256=None if witness is None else witness["witness_sha256"],
                        diagnostic_replay=self.diagnostic,
                        predecision_physical_match_required=self.module == "conditional")
        actual_ack = {"applied":True,"action":selected,"state_sha256":canonical_hash(state),
                      "evidence_sha256":canonical_hash(evidence)}
        if not self.diagnostic:
            self.ack(copy.deepcopy(actual_ack))
        acknowledged = time.monotonic()
        self.current = selected
        self.decisions.append({"state":copy.deepcopy(state),"proposal":copy.deepcopy(proposal),"ack":actual_ack})
        checkpoint = {"state":state,"destination_snapshot":destination,"holder_witness":witness,
            "witness_mode":"content" if destination["content_hashes_captured"] else "metadata",
            "holder_witness_mode":holder_mode,"matched_state_payload":matched_payload,
            "matched_state_sha256":canonical_hash(matched_payload) if matched_payload is not None else None,
            "prospective_started_s":prospective,"policy_finished_s":policy_finished,
            "transition_ack_s":acknowledged,"prospective_destination_socket_totals":before_bytes,
            "transition_evidence":evidence,
            "destination_quiescence":destination_quiescence,
            "holder_audit_started_s":holder_audit_started,"holder_audit_finished_s":holder_audit_finished,
            "holder_audit_duration_s":None if holder_audit_started is None else holder_audit_finished-holder_audit_started,
            "checkpoint_audit_started_s":audit_started,"checkpoint_audit_finished_s":audit_finished,
            "synchronous_holder_observation_performed":holder_mode is not None,
            "state_scope":"recorded logical route; not equal physical migration" if self.diagnostic and self.module == "continuous" else "observed state",
            "audit_excluded_from_prospective_timer":self.module == "conditional"}
        self.checkpoints.append(checkpoint)
        return {"selected_continuation":selected,"e2_checkpoint_index":index,"actual_transition_evidence_sha256":actual_ack["evidence_sha256"]}


def e3_migration_knowledge(transfer,terminal,control_events,released_evidence,*,frontier,receiver_thread_alive):
    """Merge already received protocol events; never inspect withheld replies."""
    if transfer is None:
        raise ValueError("Online repair has no actual started transfer")
    mode="paused" if transfer["initially_paused"] else "running"
    flight=None; known=False; origin="transfer_started_ACK_inflight_not_observed"
    observed_at=-1.; evidence=None
    if released_evidence is not None:
        raw=released_evidence["source_evidence"].get("migration_evidence")
        if raw is not None:
            if raw["transfer_index"]!=transfer["transfer_index"] or raw["transfer_start"]!=transfer["start"]:
                raise ValueError("Released migration metadata belongs to another segment")
            evidence=copy.deepcopy(raw); state=raw["migration"]
            mode,flight=state["state"],state["in_flight_layer"]
            if flight is not None: flight+=transfer["start"]
            known=True; origin="released_telemetry"; observed_at=released_evidence["reply_received_destination_s"]
    if control_events and control_events[-1]["ack_s"]>=observed_at:
        event=control_events[-1]; raw=event["acknowledgement"]
        if raw["transfer_index"]!=transfer["transfer_index"]: raise ValueError("Control ledger segment differs")
        mode,flight=raw["migration"]["state"],raw["migration"]["in_flight_layer"]
        if flight is not None: flight+=transfer["start"]
        known=True; origin="actual_control_ACK"; observed_at=event["ack_s"]; evidence=copy.deepcopy(raw)
    if terminal is not None:
        mode="stopped" if terminal["stopped"] else "complete"
        flight=None; known=True; origin="actual_received_complete_frame_terminal"; evidence=copy.deepcopy(terminal)
    drained=transfer["drained"]
    if drained and (terminal is None or receiver_thread_alive):
        raise ValueError("Receiver drain lacks terminal/thread completion")
    if drained: flight=None; known=True
    return {"migration_mode":mode,"committed_layers":frontier,"in_flight_layer":flight,
        "receiver_drained":drained,"transfer_index":transfer["transfer_index"],"transfer_start":transfer["start"],
        "transfer":copy.deepcopy(transfer),"start_acknowledgement":copy.deepcopy(transfer["start_acknowledgement"]),
        "source_bookkeeping_origin":origin,"source_bookkeeping_evidence":evidence,
        "in_flight_layer_known":known,"receiver_thread_alive":receiver_thread_alive,
        "source_and_destination_are_not_atomic":True,"receiver_drain_ACK":copy.deepcopy(transfer.get("receiver_drain_acknowledgement"))}


def e3_local_state(destination, migration):
    """Allowlisted local geometry plus released/acknowledged source bookkeeping.

    A None in-flight marker can mean not yet observed; it never proves drain.
    Migration evidence remains independently auditable and non-atomic.
    """
    import e3_controller as controller
    p = destination["prefix_length"]
    f = destination["logical_committed_frontier"]
    d = destination["generated_count"]
    result = {"prefix_tokens":p,"suffix_tokens":len(destination["suffix_token_ids"]),
        "public_output_cap":128,"generated_count":d,"committed_layers":f,"total_layers":48,
        "prefix_KV_bytes":p*196608,"committed_KV_bytes":p*4096*f,
        "processed_delta_tokens":len(destination["suffix_token_ids"])+d-1,"position":destination["position"],
        "migration_mode":migration["migration_mode"],"in_flight_layer":migration["in_flight_layer"],
        "receiver_drained":migration["receiver_drained"]}
    controller.validate_local(result)
    return result


class E3OnlineSession:
    """Destination online callbacks, without future G/trace in policy arguments.

    observe_opportunity runs after emitting D and before checkpoint capture, so
    actual telemetry overhead precedes the local progress snapshot. Diagnostic
    uses real recorded-route tensors and copies policy history only as evidence;
    it does not re-execute timing-dependent commands or claim physical matching.
    """
    def __init__(self, *, points, snapshot, control, drain, fetch_remaining,
                 recompute_publish, decide, ack, telemetry, diagnostic=False,
                 replay_decisions=None, clock=time.monotonic):
        import e3_controller as controller
        from e3_transport_control import TransitionAdapter
        self.controller,self.clock=controller,clock
        self.points=tuple(points); self.snapshot=snapshot; self.diagnostic=diagnostic
        if diagnostic:
            if any(value is not None for value in (decide,ack,telemetry)) or not isinstance(replay_decisions,list) or len(replay_decisions)!=len(self.points):
                raise ValueError("E3 replay requires only the recorded decision population")
        elif not all(callable(value) for value in (decide,ack)) or telemetry is None or replay_decisions is not None:
            raise ValueError("Online E3 requires actual policy/ACK/telemetry callbacks")
        self.decide,self.ack,self.telemetry=decide,ack,telemetry
        self.replay_decisions=copy.deepcopy(replay_decisions)
        self.decisions,self.checkpoints=[],[]
        self._observation=None; self._telemetry_evidence=None; self._opportunity=None
        self.active_decoder=None
        def signature():
            if self.active_decoder is None: raise RuntimeError("Delta signature outside checkpoint")
            return {"position":self.active_decoder.position,
                "pairs":[[{"id":id(t),"data_ptr":t.data_ptr(),"shape":list(t.shape)} for t in pair]
                         for pair in self.active_decoder.delta]}
        self.adapter=TransitionAdapter(snapshot=lambda:snapshot(self._telemetry_evidence),
            control=control,drain=drain,fetch_remaining=fetch_remaining,
            recompute_publish=recompute_publish,delta_signature=signature,clock=clock)

    def observe_opportunity(self,d,rpc):
        if d != self.points[len(self.checkpoints)] or self._opportunity is not None:
            raise ValueError("E3 observation opportunity duplicated or out of order")
        started=self.clock()
        if not self.diagnostic:
            self._observation=self.telemetry.observe(rpc)
            self._telemetry_evidence=self.telemetry.released_evidence(self._observation)
        self._opportunity={"generated_count":d,"observation_started_s":started,
                           "observation_finished_s":self.clock(),"actual_telemetry_RPC":not self.diagnostic}

    def __call__(self,step,decoder,token_ids):
        index=len(self.checkpoints); d=len(token_ids)
        if index>=len(self.points) or self.points[index]!=d or step!=d-1 or self._opportunity is None:
            raise ValueError("Unexpected E3 checkpoint order")
        destination=copy.deepcopy(decoder.campaign_checkpoint)
        if destination["generated_count"]!=d or destination["emitted_token_ids"]!=list(token_ids):
            raise ValueError("E3 checkpoint token history differs")
        checkpoint={**self._opportunity,"destination_snapshot":destination,
            "physical_matched_state_claim":False,"main_checkpoint_content_equality_proven":False}
        self.active_decoder=decoder
        try:
            if self.diagnostic:
                record=copy.deepcopy(self.replay_decisions[index]); original=record["state"]
                self.controller.validate_local(original)
                if (original["generated_count"]!=d or original["position"]!=decoder.position or
                        original["committed_layers"]!=destination["logical_committed_frontier"] or
                        destination["content_hashes_captured"] is not True or
                        destination["delta_lengths"]!=[[original["processed_delta_tokens"]]*2 for _ in range(48)]):
                    raise ValueError("E3 diagnostic does not reproduce recorded observable geometry")
                if record["proposal"]["state_sha256"]!=canonical_hash(original) or record["ack"]["proposal_sha256"]!=record["proposal"]["proposal_sha256"]:
                    raise ValueError("E3 diagnostic decision/state binding changed")
                checkpoint.update(telemetry_evidence=None,transition_evidence=None,
                    actual_policy_executed=False,actual_commands_reexecuted=False,
                    scope="diagnostic actual destination tensor and recorded layer route only; copied main policy history; physical/in-flight state not matched")
            else:
                migration=self.snapshot(self._telemetry_evidence)
                local=e3_local_state(destination,migration)
                checkpoint.update(telemetry_evidence=copy.deepcopy(self._telemetry_evidence),
                    migration_knowledge=copy.deepcopy(migration),policy_started_s=self.clock())
                proposal=self.decide(copy.deepcopy(local),copy.deepcopy(self._observation),self.clock())
                checkpoint["policy_finished_s"]=self.clock()
                if proposal.get("state_sha256")!=canonical_hash(local):
                    raise ValueError("Policy did not bind exact E3 local state")
                actual_ack,evidence=self.adapter.execute(proposal,local)
                record=self.ack(copy.deepcopy(actual_ack))
                if (not isinstance(record,dict) or record.get("ack")!=actual_ack or record.get("proposal")!=proposal or
                        record.get("state")!=local or record.get("observation")!=self._observation):
                    raise ValueError("Persistent E3 policy ACK record differs from actual execution")
                checkpoint.update(transition_evidence=evidence,actual_policy_executed=True,
                    actual_commands_reexecuted=True,scope="continuous main metadata and actual command/drain evidence; no content-state equality claim")
            checkpoint["decision_record_sha256"]=canonical_hash(record)
            self.decisions.append(copy.deepcopy(record)); self.checkpoints.append(checkpoint)
            return {"E3_checkpoint_index":index,"generated_count":d,"diagnostic":self.diagnostic,
                    "decision_record_sha256":checkpoint["decision_record_sha256"]}
        finally:
            self.active_decoder=None; self._opportunity=None


def validate_e3_execution_contract(case,diagnostic,replay,e3_decide,e3_ack,e3_telemetry,e3_replay):
    """Fail before model/communication side effects, including fixed baselines."""
    e3_enabled = case.get("e3_online",False)
    if type(e3_enabled) is not bool or e3_enabled and case.get("e2_module") is not None:
        raise ValueError("E3 online must be explicit and cannot mix E2 checkpoint logic")
    if e3_enabled:
        if case["action"]!="repair" or case.get("control_script") or case.get("checkpoint_generated") is not None or case.get("initial_frontier",0)!=0:
            raise ValueError("E3 online starts actual repair from admission without E2/scripted reconstruction")
        if case.get("checkpoints_generated")!=list(range(4,case["generated"],4)) or not case["checkpoints_generated"]:
            raise ValueError("E3 opportunities must be executor D%4 strictly before actual stop")
        if diagnostic != (e3_replay is not None) or diagnostic and replay is None:
            raise ValueError("E3 diagnostic requires exact recorded decisions and layer cutovers")
        if diagnostic:
            if any(value is not None for value in (e3_decide,e3_ack,e3_telemetry)) or not isinstance(e3_replay,list) or len(e3_replay)!=len(case["checkpoints_generated"]):
                raise ValueError("E3 diagnostic requires only its complete main decision records")
        elif not callable(e3_decide) or not callable(e3_ack) or e3_telemetry is None:
            raise ValueError("Online E3 callbacks and actual telemetry are required")
    elif any(value is not None for value in (e3_decide,e3_ack,e3_telemetry,e3_replay)):
        raise ValueError("Fixed methods cannot receive dummy online callbacks or telemetry")
    elif "e3_online" in case and (case.get("checkpoints_generated") or case.get("checkpoint_generated") is not None):
        raise ValueError("Fixed E3 baselines must have no checkpoint work")
    return e3_enabled


def _e4_release_owned_prefix(store, result, owned_resources, *, publications, drains, finish, receiver):
    """No tensor content reads/copies: prove owned Python references are gone."""
    refs=[] if store is None else [weakref.ref(t) for p in store.values if p is not None for t in p]
    if store is not None:
        store.values.clear()
    released=all(reference() is None for reference in refs)
    result.update(e4_layer_publications=publications,e4_receiver_drains=drains,
        e4_finish=finish,protocol_settled_s=result.get("settled_s"),
        own_prefix_references_released=released,own_prefix_reference_count=len(refs),
        own_receiver_thread_exited=receiver is None or not receiver.is_alive(),
        own_settled_s=time.monotonic() if released else None,
        own_settlement_scope="native local time after protocol, receiver exit and request-owned prefix release")
    if not released:
        result["passed"]=False
    owned_resources.record_report(result)
    if not released:
        raise RuntimeError("E4 request-owned prefix tensor references survived cleanup")


def execute(engine, ids, case, request, host, ports, budget, *, diagnostic=False,
            replay=None, calibration=None, cached_pairs=None, keep_prefix=False,
            accepted_s=None, e2_decide=None, e2_ack=None, e2_replay=None,
            e3_decide=None,e3_ack=None,e3_telemetry=None,e3_replay=None,e3_smoke_replay=None,
            e4_owned_resources=None):
    if e4_owned_resources is not None and (keep_prefix or cached_pairs is not None or
            case.get("action") not in ("remote", "repair")):
        raise ValueError("E4 own-settlement requires an uncached remote/repair request")
    e3_enabled=validate_e3_execution_contract(case,diagnostic,replay,e3_decide,e3_ack,e3_telemetry,e3_replay)
    import torch
    from campaign_decode import decode_request
    e2_enabled = case.get("e2_module") is not None
    smoke_enabled=case.get("e3_smoke_fixture") is not None
    if smoke_enabled:
        if case.get("e3_execution_mode")!="e3_smoke" or diagnostic!=(e3_smoke_replay is not None):
            raise ValueError("Functional fixture requires explicit smoke mode and diagnostic evidence binding")
    elif e3_smoke_replay is not None:
        raise ValueError("Main fixture evidence cannot enter a non-fixture request")
    if e2_enabled:
        if case["action"] != "continuation" or case.get("control_script"):
            raise ValueError("E2 controls own the continuation; legacy scripted controls are disallowed")
        points = case.get("checkpoints_generated")
        if (not isinstance(points, list) or not points or any(type(d) is not int or not 0 < d < case["generated"] for d in points)
                or points != sorted(set(points))):
            raise ValueError("Invalid E2 checkpoint schedule")
        legacy = case.get("checkpoint_generated")
        if legacy is not None and legacy != points and not (len(points) == 1 and legacy == points[0]):
            raise ValueError("Conflicting E2 legacy and multi-checkpoint schedules")
        if diagnostic != (e2_replay is not None) or (diagnostic and replay is None):
            raise ValueError("E2 diagnostic requires both recorded decisions and layer cutovers")
    elif any(value is not None for value in (e2_decide, e2_ack, e2_replay)):
        raise ValueError("E2 callbacks require an explicitly declared E2 case")
    script = validate_control_script(case)
    if replay is not None and (not diagnostic or len(replay) != engine.layers or any(
            type(position) is not int or not case["prefix"] <= position <= case["prefix"] + case["suffix"] + case["generated"] - 1
            for position in replay)):
        raise ValueError("Recorded cutovers require a diagnostic replay and one valid position per layer")
    if len(ids) != case["prefix"] + case["suffix"]:
        raise ValueError("Input token count differs from declared prefix/suffix")
    entered_s = time.monotonic()
    if accepted_s is not None and (isinstance(accepted_s, bool) or not isinstance(accepted_s, (int, float))
                                   or not math.isfinite(accepted_s) or accepted_s > entered_s):
        raise ValueError("Outer request acceptance must be a finite earlier destination timestamp")
    began = entered_s if accepted_s is None else accepted_s
    action = case["action"]
    local = action in ("recompute", "local_reuse", "native_hf_control")
    prefix = torch.tensor([ids[:case["prefix"]]], device=engine.device)
    suffix = torch.tensor([ids[case["prefix"]:]], device=engine.device)
    rpc = bulk = thread = store = None
    errors, remote_calls, delivery, tokens, decisions, transfer_segments = [], [], [], [], [], []
    migration_active = False
    migration_terminal = None
    transfer_info = None
    last_control_mode = None
    control_events = []
    finish_ack = None
    e2_session = None
    e3_session = None
    smoke_fixture=None
    injected_application_fault=None
    e4_publications=[]
    e4_drain_events=[]
    e4_finish_event=None
    publication=threading.Condition()
    checkpoint_marks = {}
    stopped_irreversibly = False
    remaining_started = None
    prefill_duration = None
    result = {"request_id": request, "action": action, "diagnostic": diagnostic,
              "accepted_s": began, "passed": False, "physical_wire_bytes": None,
              "executor_entered_s": entered_s,
              "outer_setup_elapsed_s": entered_s - began,
              "acceptance_scope": "destination outer policy/setup boundary" if accepted_s is not None else "executor entry before token tensor creation",
              "wire_measurement_available": False,
              "measurement_kind": "diagnostic_replay" if replay is not None else "diagnostic" if diagnostic else "measured_candidate_pending_external_validation",
              "timing_eligible_after_external_validation": not diagnostic}

    def control_command(command, generated_count, *, reason="script"):
        nonlocal last_control_mode
        if transfer_info is None or rpc is None:
            raise ValueError("Control script reached a boundary without any started migration")
        last_control_mode, event = send_migration_control(rpc, command, transfer_info["transfer_index"], generated_count,
                                                         diagnostic=diagnostic, reason=reason)
        control_events.append(event)
        return event["acknowledgement"]

    def delivered(step, token):
        if step != len(tokens):
            raise ValueError("Delivered token order differs")
        tokens.append(token)
        delivery.append(time.monotonic())
        if smoke_fixture is not None:
            smoke_fixture.on_token(step,token)
        if e3_enabled and len(tokens) in case["checkpoints_generated"]:
            e3_session.observe_opportunity(len(tokens),rpc)
            if diagnostic:
                required=e3_replay[case["checkpoints_generated"].index(len(tokens))]["state"]["committed_layers"]
                if required>store.snapshot()[0]:
                    events=getattr(store,"ready_events",None)
                    if events is None or not events[required-1].wait(timeout=180):
                        raise TimeoutError("E3 diagnostic lacks the recorded actually published prefix view")
        if e2_enabled and len(tokens) in case["checkpoints_generated"]:
            # Continuous prospective time includes destination metadata capture,
            # source observation RPC and decision work. Conditional timing starts
            # after its deliberately expensive two-endpoint content audit.
            checkpoint_marks[len(tokens)] = (time.monotonic(), _e2_socket_totals(budget))
            if diagnostic:
                record = e2_replay[case["checkpoints_generated"].index(len(tokens))]
                required = record["state"]["committed_layers"]
                if required > store.snapshot()[0]:
                    events = getattr(store, "ready_events", None)
                    if events is None or not events[required-1].wait(timeout=180):
                        raise TimeoutError("Diagnostic checkpoint has not actually received its recorded visible prefix")
        for entry in script:
            if entry["generated_count"] == len(tokens):
                if replay is not None:
                    control_events.append({**entry, "applied": False, "reason": "recorded_cutover_diagnostic_skips_control",
                                           "timing_is_measurement": False})
                else:
                    control_command(entry["command"], len(tokens))

    def receive_segment(start, end, transfer_index):
        nonlocal migration_terminal
        torch.cuda.set_device(0)
        stream = torch.cuda.Stream()
        next_layer = start
        payload_bytes = 0
        with torch.inference_mode():
            while True:
                message, tensors = bulk.recv()
                if message["kind"] == "migration_terminal":
                    if tensors:
                        raise ValueError("Migration terminal cannot carry tensor payload")
                    validate_migration_terminal(message, start=start, end=end, next_layer=next_layer,
                                                transfer_index=transfer_index, payload_bytes=payload_bytes)
                    migration_terminal = message
                    transfer_segments.append(dict(message))
                    return
                if (message["kind"] != "prefix" or type(message.get("layer")) is not int or
                        message["layer"] != next_layer or message.get("transfer_index") != transfer_index or not next_layer < end):
                    raise ValueError("Migration prefix order differs")
                shape = (1, engine.kv_heads, case["prefix"], engine.head_dim)
                if len(tensors) != 2 or any(tuple(t.shape) != shape or t.dtype != engine.dtype for t in tensors):
                    raise ValueError("Migration shape/dtype differs")
                pinned = [t.pin_memory() for t in tensors]
                copy_started_s=time.monotonic() if e4_owned_resources is not None else None
                with torch.cuda.stream(stream):
                    pair = [t.to(engine.device, non_blocking=True) for t in pinned]
                    ready = torch.cuda.Event(enable_timing=False)
                    ready.record()
                ready.synchronize()
                gpu_ready_s=time.monotonic() if e4_owned_resources is not None else None
                layer_bytes = sum(t.numel() * t.element_size() for t in pair)
                store.publish(next_layer, *pair, layer_bytes)
                if e4_owned_resources is not None:
                    e4_publications.append(dict(layer=next_layer,transfer_index=transfer_index,
                        bytes=layer_bytes,HtoD_copies=2,copy_started_local_s=copy_started_s,
                        GPU_ready_local_s=gpu_ready_s,published_local_s=time.monotonic(),
                        readiness="actual CUDA event synchronized before PrefixStore.publish",
                        content_hash_captured=False))
                if smoke_enabled:
                    with publication: publication.notify_all()
                payload_bytes += layer_bytes
                next_layer += 1

    def start_transfer(start, end, wait=False, initially_paused=False):
        nonlocal thread, migration_active, migration_terminal, transfer_info, last_control_mode
        if start == end:
            return
        if thread is not None:
            finish_transfer()
        migration_terminal = None
        acknowledgement = rpc.command("transfer", "transfer_started", start=start, end=end, initially_paused=initially_paused)
        if (acknowledgement.get("start") != start or acknowledgement.get("end") != end or
                type(acknowledgement.get("transfer_index")) is not int or
                acknowledgement["transfer_index"] != (1 if transfer_info is None else transfer_info["transfer_index"] + 1)):
            raise ValueError("Started transfer identity differs")
        transfer_info = {"start": start, "end": end, "transfer_index": acknowledgement["transfer_index"],
                         "drained": False, "initially_paused": initially_paused,
                         "start_acknowledgement":copy.deepcopy(acknowledgement)}
        last_control_mode = "paused" if initially_paused else "running"
        def guarded():
            try:
                receive_segment(start, end, acknowledgement["transfer_index"])
            except BaseException as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                if smoke_enabled:
                    with publication: publication.notify_all()
                if bulk:
                    bulk.close()
        thread = threading.Thread(target=guarded, name="campaign-prefix-receiver", daemon=True)
        migration_active = True
        thread.start()
        if smoke_fixture is not None:
            smoke_fixture.after_transfer_started(copy.deepcopy(transfer_info))
        if wait:
            finish_transfer()

    def finish_transfer():
        nonlocal migration_active
        if thread is not None:
            timeout=(e4_owned_resources.remaining() if e4_owned_resources is not None else
                     180 if not smoke_enabled else max(0.,min(180.,case["e3_smoke_deadline_s"]-time.monotonic())))
            thread.join(timeout=timeout)
            if thread.is_alive() or errors:
                raise RuntimeError("Prefix receive did not drain: " + repr(errors))
            if migration_terminal is None:
                raise RuntimeError("Prefix receiver exited without a validated terminal")
            if not transfer_info["drained"]:
                fields = {key: migration_terminal[key] for key in ("transfer_index", "start", "end", "next_layer", "payload_bytes")}
                drain_started_s=time.monotonic()
                acknowledgement = rpc.command("migration_drained", "migration_drained_ack", **fields)
                drain_finished_s=time.monotonic()
                if acknowledgement.get("drained") is not True or any(acknowledgement.get(key) != value for key, value in fields.items()):
                    raise ValueError("Holder did not acknowledge the exact GPU-ready received segment")
                transfer_info["drained"] = True
                transfer_info["receiver_drain_acknowledgement"] = copy.deepcopy(acknowledgement)
                transfer_segments[-1]["receiver_drain_ack"] = acknowledgement
                if e4_owned_resources is not None:
                    e4_drain_events.append(dict(started_local_s=drain_started_s,
                        finished_local_s=drain_finished_s,receiver_thread_exited=True,
                        fields=fields,acknowledgement=copy.deepcopy(acknowledgement)))
            migration_active = False

    def remote(layer, position, query, key, value):
        rpc.send("attention", [t.cpu().contiguous() for t in (query, key, value)], layer=layer, position=position)
        message, tensors = rpc.recv()
        expected = (1, 1, engine.query_heads, engine.head_dim)
        if (message.get("kind") != "attention_result" or message.get("layer") != layer or
                message.get("position") != position or len(tensors) != 1 or
                tuple(tensors[0].shape) != expected or tensors[0].dtype != engine.dtype):
            raise ValueError("Remote attention result identity differs")
        remote_calls.append([layer, position])
        return tensors[0].to(engine.device)

    def e2_migration_snapshot():
        return {"physical_frontier":store.snapshot()[0],
            "receiver_thread_alive":thread is not None and thread.is_alive(),
            "drained":thread is None or transfer_info["drained"],
            "transfer":copy.deepcopy(transfer_info),"terminal":copy.deepcopy(migration_terminal),
            "last_control_mode":last_control_mode,"stopped_irreversibly":stopped_irreversibly,
            "errors":list(errors)}

    def e2_transition(selected, generated_count):
        def mark_stopped():
            nonlocal stopped_irreversibly
            stopped_irreversibly = True

        def recompute_publish():
            nonlocal prefill_duration
            recompute_started = time.monotonic()
            pairs = engine.prefix(prefix)
            try:
                prefill_duration = time.monotonic()-recompute_started
                first = store.snapshot()[0]
                for layer in range(first, engine.layers):
                    store.publish(layer, *pairs[layer], 0)
                return {"kind":"actual_full_prefix_recompute","prefix_tokens":case["prefix"],
                    "computed_layers":engine.layers,"newly_published_layers":list(range(first,engine.layers)),
                    "elapsed_s":prefill_duration,"retains_existing_delta":True}
            finally:
                pairs.clear()

        return apply_e2_transition(selected, generated_count, layers=engine.layers,
            snapshot=e2_migration_snapshot, start_transfer=start_transfer, finish_transfer=finish_transfer,
            stop_transfer=lambda d, action:control_command("stop",d,reason="E2_"+action),
            recompute_publish=recompute_publish, mark_stopped=mark_stopped)

    def e3_snapshot(released_evidence):
        if errors: raise RuntimeError("E3 receiver failed: "+repr(errors))
        return e3_migration_knowledge(transfer_info,migration_terminal,control_events,released_evidence,
            frontier=store.snapshot()[0],receiver_thread_alive=thread is not None and thread.is_alive())

    def e3_control(command):
        nonlocal stopped_irreversibly
        reply=control_command(command,len(tokens),reason="E3_online_"+command)
        if command=="stop" and reply["migration"]["state"] in ("stopping","stopped","complete"):
            stopped_irreversibly=True
        return reply

    def e3_drain():
        finish_transfer()
        return {"terminal":copy.deepcopy(migration_terminal),"transfer":copy.deepcopy(transfer_info),
                "receiver_thread_exited":thread is None or not thread.is_alive(),
                "prefix_frontier":store.snapshot()[0],"GPU_publication_precedes_terminal":True}

    def e3_fetch():
        if stopped_irreversibly: raise ValueError("E3 fetch cannot restart stopped migration")
        first=store.snapshot()[0]
        start_transfer(first,engine.layers,wait=True)
        return e3_drain()

    def e3_recompute():
        nonlocal prefill_duration
        began_recompute=time.monotonic(); pairs=engine.prefix(prefix)
        try:
            prefill_duration=time.monotonic()-began_recompute; first=store.snapshot()[0]
            for layer in range(first,engine.layers): store.publish(layer,*pairs[layer],0)
            return {"kind":"actual_full_prefix_recompute","prefix_tokens":case["prefix"],
                "computed_layers":engine.layers,"newly_published_layers":list(range(first,engine.layers)),
                "started_s":began_recompute,"finished_s":time.monotonic(),
                "elapsed_s":prefill_duration,"retains_existing_delta":True}
        finally: pairs.clear()

    def smoke_wait_frontier(frontier,deadline):
        with publication:
            ready=publication.wait_for(lambda:store.snapshot()[0]>=frontier or bool(errors),
                timeout=max(0.,deadline-time.monotonic()))
        if not ready or errors:
            raise RuntimeError("Functional fixture did not receive its actual GPU-ready prefix: "+repr(errors))

    def smoke_destination():
        snapshot=e2_migration_snapshot()
        snapshot["coordinator"]=budget.snapshot()
        return snapshot

    def checkpoint(step, decoder, token_ids):
        nonlocal remaining_started, prefill_duration
        remaining_started = time.monotonic()
        if thread is not None and (thread.is_alive() or not transfer_info["drained"]):
            raise ValueError("Matched-state checkpoint requires the initial transfer fully drained before decode")
        frontier = store.snapshot()[0]
        selected = case.get("continuation", "continue_repair")
        decision = None
        policy = None
        if selected in ("state_policy", "stale_policy", "no_uncertainty_policy"):
            if not calibration:
                raise ValueError("A policy requires actual earlier calibration measurements")
            processed = decoder.position - case["prefix"]
            per_layer = 2 * engine.kv_heads * case["prefix"] * engine.head_dim * 2
            state = ProgressState(request, case["prefix_key"], (per_layer,) * engine.layers,
                tuple(range(frontier)), frontier * per_layer, None, len(token_ids), processed,
                (processed,) * engine.layers,
                tuple(sum(layer == index for layer, _ in remote_calls) for index in range(engine.layers)),
                "paused" if frontier < engine.layers else "complete", True, True, True, True,
                sunk_elapsed_seconds=remaining_started - began)
            now = time.monotonic()
            age = case.get("observation_age_seconds", 100.0 if selected in ("stale_policy", "no_uncertainty_policy") else 0.0)
            if isinstance(age, bool) or not isinstance(age, (float, int)) or not math.isfinite(age) or not 0 <= age <= now:
                raise ValueError("Injected observation age must be finite, nonnegative, and representable")
            observation = CostObservation(now - age, now - age, calibration["local_step_seconds"],
                calibration["all_remote_step_seconds"], calibration["effective_prefix_payload_bytes_per_second"],
                full_prefix_recompute_seconds=calibration["full_prefix_recompute_seconds"],
                source=calibration["source"])
            policy = ProgressPolicy(max_observation_age=1000 if selected == "no_uncertainty_policy" else 1,
                                    minimum_dwell_seconds=0)
            estimate = EmpiricalRemainingEstimator(calibration["training_lengths"],
                training_split_id=calibration["training_split_id"]).estimate(len(token_ids))
            decision = policy.decide(state, ExecutionCapabilities(True, True, True, False, True),
                                     estimate, observation, now=now)
            selected = decision["action"]
            decisions.append({"at_generated": len(token_ids), "state": asdict(state),
                              "estimate": asdict(estimate), "observation": asdict(observation),
                              "injected_observation_age_seconds": age, "decision": decision})
        else:
            decisions.append({"at_generated": len(token_ids), "action": selected,
                              "selection": "predeclared counterfactual branch"})
        if replay is not None:
            # Replay validates the exact measured layer route. Diagnostic transfer
            # may proceed early; PrefixStore visibility enforces recorded cutovers.
            if frontier < engine.layers:
                start_transfer(frontier, engine.layers)
        elif selected == "continue_repair":
            start_transfer(frontier, engine.layers, initially_paused=case.get("initially_paused", False))
        elif selected == "complete_fetch_then_local":
            start_transfer(frontier, engine.layers, wait=True)
        elif selected in ("stop_migration_and_remote", "pause_migration_and_remote"):
            pass  # Initial matched-state transfer has already drained; no future frames start.
        elif selected == "recompute_full_prefix_keep_delta":
            started = time.monotonic()
            pairs = engine.prefix(prefix)
            prefill_duration = time.monotonic() - started
            for layer in range(frontier, engine.layers):
                store.publish(layer, *pairs[layer], 0)
            pairs.clear()
        elif selected != "continue_local":
            raise ValueError("Unsupported continuation: " + str(selected))
        if policy is not None:
            policy.acknowledge(decision, now=time.monotonic(), applied=replay is None)
            decisions[-1]["execution_acknowledged"] = replay is None
            decisions[-1]["diagnostic_route_replay_overrides_policy"] = replay is not None
        return {"selected_continuation": selected, "diagnostic_replay": replay is not None,
                "initial_frontier": frontier, "policy_decision": decision}

    try:
        if not local:
            peer = {"prefix_key": case["prefix_key"], "layers": engine.layers, "dtype": str(engine.dtype)}
            rpc = Channel(host, ports["rpc"], budget, request, "rpc", expected_peer=peer,
                timeout=180 if e4_owned_resources is None else e4_owned_resources.remaining(),
                owned_resources=e4_owned_resources)
            if action != "holder":
                bulk = Channel(host, ports["bulk"], budget, request, "bulk", expected_peer=peer,
                    timeout=180 if e4_owned_resources is None else e4_owned_resources.remaining(),
                    owned_resources=e4_owned_resources)
        if action == "holder":
            rpc.send("decode", suffix_ids=ids[case["prefix"]:], generated=case["generated"], diagnostic=diagnostic)
            rows = []
            for step in range(case["generated"]):
                message, tensors = rpc.recv(tensor_format="bytes")
                if message.get("kind") != "token" or message.get("step") != step or tensors:
                    raise ValueError("Holder token stream differs")
                delivered(step, message["token"])
            if diagnostic:
                message, tensors = rpc.recv()
                if message.get("kind") != "full_logits" or len(tensors) != 1:
                    raise ValueError("Missing holder diagnostic logits")
                rows = [tensors[0]]
            message, tensors = rpc.recv(tensor_format="bytes")
            if message.get("kind") != "decode_done" or message.get("generated") != case["generated"] or tensors:
                raise ValueError("Missing holder decode completion")
            decoded = {"token_ids": tokens.copy(), "logits": rows, "state_report": message.get("state_report", {})}
        elif action == "native_hf_control":
            decoded = engine.HF_reference(prefix, suffix, case["generated"], on_token=delivered, capture_logits=diagnostic)
        else:
            store = replay_store(engine.layers, case["prefix"], replay) if replay is not None else PrefixStore(engine.layers)
            if smoke_enabled:
                from e3_smoke_client import ClientSmokeFixture
                smoke_fixture=ClientSmokeFixture(case,rpc,
                    control=lambda command:control_command(command,len(tokens),reason="E3_functional_fixture"),
                    drain=e3_drain,snapshot_destination=smoke_destination,wait_frontier=smoke_wait_frontier,
                    deadline_monotonic=case["e3_smoke_deadline_s"],epoch_monotonic=budget.limiter.epoch_monotonic,
                    diagnostic=diagnostic,copied_evidence=e3_smoke_replay)
            if action in ("recompute", "local_reuse"):
                if action == "local_reuse" and cached_pairs is None:
                    raise ValueError("Cache-hit action lacks a real resident target prefix")
                started = time.monotonic()
                pairs = engine.prefix(prefix) if action == "recompute" else cached_pairs
                prefill_duration = time.monotonic() - started
                for index, pair in enumerate(pairs):
                    store.publish(index, *pair, 0)
                if action == "recompute":
                    pairs.clear()
            elif case.get("initial_frontier", 0):
                start_transfer(0, case["initial_frontier"], wait=True)
            if action == "native_fetch":
                start_transfer(store.snapshot()[0], engine.layers, wait=True)
            elif action == "repair":
                start_transfer(store.snapshot()[0], engine.layers,
                               initially_paused=case.get("initially_paused", False) and replay is None)
            if e3_enabled:
                e3_session=E3OnlineSession(points=case["checkpoints_generated"],snapshot=e3_snapshot,
                    control=e3_control,drain=e3_drain,fetch_remaining=e3_fetch,recompute_publish=e3_recompute,
                    decide=e3_decide,ack=e3_ack,telemetry=e3_telemetry,diagnostic=diagnostic,replay_decisions=e3_replay)
                views=({record["state"]["generated_count"]:record["state"]["committed_layers"] for record in e3_replay}
                       if diagnostic else None)
                decoded=decode_request(engine,store,prefix,suffix,case["generated"],remote_attention=remote,
                    on_token=delivered,on_checkpoint=e3_session,checkpoints_generated=case["checkpoints_generated"],
                    capture_logits=diagnostic,capture_checkpoint_hashes=diagnostic,
                    recorded_checkpoint_frontiers=views,
                    checkpoint_input_ids={"prefix":ids[:case["prefix"]],"suffix":ids[case["prefix"]:]})
                decisions=e3_session.decisions
                if len(decisions)!=len(case["checkpoints_generated"]):
                    raise RuntimeError("E3 did not execute every online opportunity")
            elif e2_enabled:
                actions = ("continue_repair","stop_migration_and_remote","complete_fetch_then_local",
                           "recompute_full_prefix_keep_delta","continue_local")
                operations = {name:(lambda d, name=name:e2_transition(name,d)) for name in actions}
                e2_session = E2CheckpointSession(engine,case,request,budget,rpc,remote_calls,operations,
                    e2_migration_snapshot,diagnostic=diagnostic,decide=e2_decide,ack=e2_ack,
                    replay_decisions=e2_replay,checkpoint_marks=checkpoint_marks)
                views = ({record["state"]["generated_count"]:record["state"]["committed_layers"]
                          for record in e2_replay} if diagnostic else None)
                decoded = decode_request(engine,store,prefix,suffix,case["generated"],remote_attention=remote,
                    on_token=delivered,on_checkpoint=e2_session,checkpoints_generated=case["checkpoints_generated"],
                    capture_logits=diagnostic,capture_checkpoint_hashes=case["checkpoint_witness_mode"] == "content",
                    recorded_checkpoint_frontiers=views,
                    checkpoint_input_ids={"prefix":ids[:case["prefix"]],"suffix":ids[case["prefix"]:]})
                decisions = e2_session.decisions
                if len(e2_session.checkpoints) != len(case["checkpoints_generated"]):
                    raise RuntimeError("E2 did not execute every declared checkpoint")
                remaining_started = e2_session.checkpoints[0]["prospective_started_s"]
            else:
                decoded = decode_request(engine, store, prefix, suffix, case["generated"],
                    remote_attention=None if local else remote, on_token=delivered,
                    on_checkpoint=checkpoint if case.get("checkpoint_generated") else None,
                    checkpoint_generated=case.get("checkpoint_generated"), capture_logits=diagnostic,
                    capture_checkpoint_hashes=bool(case.get("checkpoint_generated")))
        if decoded["token_ids"] != tokens:
            raise ValueError("Delivered token sequence differs from execution")
        validate_decode_state(decoded["state_report"], remote_calls, case, engine.layers)
        if smoke_fixture is not None:
            from e3_smoke_client import InjectedE3OwnedCleanupFault
            try:
                smoke_fixture.after_decode(decoded)
            except InjectedE3OwnedCleanupFault as error:
                # The application fault remains a failed raw request even when
                # its expected complete-frame cleanup succeeds.
                injected_application_fault=f"{type(error).__name__}: {error}"
                smoke_fixture.recover_injected_fault(error)
        if thread is not None:
            if thread.is_alive() and last_control_mode == "paused":
                control_command("stop", len(tokens), reason="cleanup_paused_migration_before_drain")
            finish_transfer()
        if rpc is not None:
            finish_started_s=time.monotonic()
            finish_ack = rpc.command("finish", "finish_ack", token_ids=tokens)
            if e4_owned_resources is not None:
                e4_finish_event=dict(started_local_s=finish_started_s,finished_local_s=time.monotonic(),
                    acknowledgement=copy.deepcopy(finish_ack))
            expected_next = 0 if migration_terminal is None else migration_terminal["next_layer"]
            if (finish_ack.get("remote_calls") != len(remote_calls) or finish_ack.get("next_layer") != expected_next
                    or finish_ack.get("owned_delta_references_released") is not True):
                raise ValueError("Holder final RPC/transfer/delta cleanup acknowledgement differs")
        torch.cuda.synchronize()
        for channel in (rpc, bulk):
            if channel:
                channel.close()
        channel_audits = {name: channel.audit() for name, channel in (("rpc", rpc), ("bulk", bulk)) if channel is not None}
        settled = time.monotonic()
        result.update(passed=injected_application_fault is None, token_ids=tokens, token_delivery_s=delivery, settled_s=settled,
            ttft_s=delivery[0] - began, completion_s=delivery[-1] - began, settled_latency_s=settled - began,
            itl_s=[right - left for left, right in zip(delivery, delivery[1:])],
            remaining_started_s=remaining_started,
            remaining_completion_s=delivery[-1] - remaining_started if remaining_started is not None else None,
            remaining_settled_s=settled - remaining_started if remaining_started is not None else None,
            history_and_reconstruction_s=remaining_started - began if remaining_started is not None else None,
            prefill_duration_s=prefill_duration, state_report=decoded["state_report"],
            checkpoint=decoded.get("checkpoint"), remote_calls=remote_calls, decisions=decisions,
            transfer_segments=transfer_segments, control_events=control_events, finish_ack=finish_ack,
            migration_controls_actually_applied=sum(event.get("applied") is True and event.get("reason") == "script"
                                                     for event in control_events),
            channel_audits=channel_audits,
            traffic={"rpc": dict(rpc.counts) if rpc else None, "bulk": dict(bulk.counts) if bulk else None},
            background_drained=thread is None or not thread.is_alive())
        if action not in ("holder", "native_hf_control"):
            result["cutovers"] = state_signature(result, case, engine.layers)
            result["committed_frontier"] = store.snapshot()[0]
        if e2_enabled:
            if diagnostic and result["cutovers"] != replay:
                raise ValueError("E2 diagnostic did not execute the recorded layer route")
            checkpoints = e2_session.checkpoints
            result.update(checkpoints=checkpoints, decoder_checkpoints=decoded["checkpoints"],
                next_token_after_decision_s=delivery[case["checkpoints_generated"][0]]-remaining_started,
                actual_policy_switch_count=sum(a["ack"]["action"] != b["ack"]["action"] and b["ack"]["action"] != "continue_local"
                    for a,b in zip(decisions,decisions[1:])),
                e2_acceptance={"state_structure_passed":True,"causal_information_passed":True,
                    "transition_ack_passed":True,
                    "content_witness_passed":all(c["matched_state_sha256"] is not None for c in checkpoints),
                    "checkpoint_replay_passed":diagnostic,
                    "content_witness_scope":"conditional_both_endpoints" if case["e2_module"] == "conditional" else
                        "diagnostic_destination_content_and_route_only" if diagnostic else "continuous_main_metadata_only",
                    "main_checkpoint_content_equality_proven":False})
        if "e3_online" in case:
            if diagnostic and action not in ("holder","native_hf_control") and result["cutovers"]!=replay:
                raise ValueError("E3 diagnostic did not execute the recorded actual layer route")
            result.update(e3_online=e3_enabled,e3_checkpoints=[] if e3_session is None else e3_session.checkpoints,
                decoder_checkpoints=decoded.get("checkpoints",[]),
                telemetry_history=[] if e3_telemetry is None else copy.deepcopy(e3_telemetry.history),
                telemetry_release_events=[] if e3_telemetry is None else copy.deepcopy(e3_telemetry.inbox.events),
                e3_scope="diagnostic actual destination content/route; copied policy history, no physical matching" if diagnostic else
                    "continuous real execution; strict released information; no synchronized clocks or matched physical state claim",
                main_checkpoint_content_equality_proven=False,diagnostic_commands_reexecuted=False,
                fixed_method_has_no_checkpoint_or_telemetry=not e3_enabled,
                actual_policy_switch_count=sum(row.get("actual_applied_change") is True for row in decisions),
                E3_complete=False)
        if injected_application_fault is not None:
            result.update(error=injected_application_fault,expected_injected_application_fault=True,
                original_application_failure_preserved=True,controlled_cleanup_after_failure_completed=True)
        if smoke_fixture is not None:
            result["smoke_fixture_evidence"]=smoke_fixture.finish(result)
        retained = None
        if keep_prefix and store is not None and store.snapshot()[0] == engine.layers:
            retained = list(store.values)
        return result, decoded["logits"], retained
    finally:
        for channel in (rpc, bulk):
            if channel:
                channel.close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=10)
            if thread.is_alive():
                # Never clear storage while the receiver may still publish into it.
                raise RuntimeError("Campaign receiver survived socket close and cleanup deadline")
        if e4_owned_resources is not None:
            _e4_release_owned_prefix(store,result,e4_owned_resources,publications=e4_publications,
                drains=e4_drain_events,finish=e4_finish_event,receiver=thread)
        elif store is not None:
            store.values.clear()


def verify_outputs(torch, reference, report, diagnostic_rows, performance_report=None):
    from e0_holder_decode import compare
    candidate = torch.cat(diagnostic_rows, dim=0)
    check = compare(torch, torch.cat(reference["logits"], dim=0), candidate,
                    reference["token_ids"], report["token_ids"])
    validate_numerical(check, report["token_ids"], len(reference["token_ids"]), reference["token_ids"])
    if performance_report is not None:
        if performance_report["token_ids"] != report["token_ids"]:
            raise ValueError("Measured output differs from separate diagnostic")
        if performance_report.get("cutovers") != report.get("cutovers"):
            raise ValueError("Diagnostic layer cutovers differ from the measured execution")
    return check
