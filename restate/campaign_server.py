"""Real GPU holder for the campaign executor; no scheduler or job submission.

serve_holder(engine, control, base_port) owns shared immutable prefixes. Control
prepare/batch-ready traffic is outside request measurements. Every request RPC
and bulk frame uses real TCP and a shared outgoing coordinator. Attention/model
operations are serialized by one GPU mutex; migration uses its own CUDA stream.
Application byte counters never certify physical network bytes or GPU overlap.
"""

from contextlib import contextmanager, nullcontext
import gc
import hashlib
import json
import math
import socket
import threading
import time
import traceback
import weakref

from e1_runtime import HolderAttention
from e0_cache_utils import cache_sha256, cache_to_cpu_pairs
from e1_wire import FramedConnection
from e2_control import MigrationControl
from e3_perturbation import RatePlan, VariableRateLimiter
from e4_e6_mechanisms import SharedBandwidthCoordinator


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _token_ids(value, name, vocab):
    if not isinstance(value, list) or not value or any(type(x) is not int or not 0 <= x < vocab for x in value):
        raise ValueError(f"{name} must be nonempty valid integer token IDs")
    return value


def _listener(port, timeout):
    result = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        result.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        result.bind(("0.0.0.0", port))
        result.listen(1)
        result.settimeout(timeout)
        return result
    except BaseException:
        result.close()
        raise


class _SharedGPU:
    def __init__(self, engine):
        self.engine, self.mutex, self.lock = engine, threading.Lock(), threading.Lock()
        self.active, self.queued, self.peak_queued = 0, 0, 0

    @contextmanager
    def acquire(self, report, deadline, kind="holder_setup"):
        queue = self.engine.compute_queue
        if queue is not None:
            with queue.bind(report.get("request_id", "holder-setup"), report, deadline), queue.operation(kind):
                yield
            return
        started = time.monotonic()
        with self.lock:
            self.queued += 1
            self.peak_queued = max(self.peak_queued, self.queued)
        acquired = False
        try:
            acquired = self.mutex.acquire(timeout=max(0.0, deadline - time.monotonic()))
        finally:
            with self.lock:
                self.queued -= 1
                if acquired: self.active += 1
        if not acquired:
            raise TimeoutError("GPU admission exceeded batch deadline")
        report["gpu_queue_wait_s"] += time.monotonic() - started
        report["gpu_admissions"] += 1
        torch = self.engine.torch
        try:
            with torch.cuda.device(self.engine.device), torch.cuda.stream(torch.cuda.default_stream(self.engine.device)), torch.inference_mode():
                yield
        finally:
            with self.lock: self.active -= 1
            self.mutex.release()

    def snapshot(self):
        if self.engine.compute_queue is not None:
            return self.engine.compute_queue.state()
        with self.lock:
            return {"active_gpu_operations": self.active, "queued_gpu_operations": self.queued,
                    "peak_queued_gpu_operations": self.peak_queued, "model_execution_serialized": True}


class _Request:
    def __init__(self, engine, gpu, prefix, spec, ports, coordinator, timeout, deadline, integrity):
        self.engine, self.gpu, self.prefix, self.spec = engine, gpu, prefix, spec
        self.request_id, self.coordinator = spec["request_id"], coordinator
        self.timeout, self.deadline, self.integrity = timeout, deadline, integrity
        self.lock = threading.Lock()
        self.listeners, self.channels, self.wrapped = {}, {}, {}
        self.thread = self.migration_thread = self.migration = self.attention = None
        self.migration_error = None
        self.frontier, self.transfer_index = 0, 0
        self.transfer_start=0
        self.e3_observation_sequence=-1
        self.smoke_lifecycle=None
        smoke=spec.get('e3_smoke_fixture')
        if smoke is not None:
            from e3_smoke_control import validate_fixture_spec
            validate_fixture_spec(getattr(engine,'e3_execution_mode',None),smoke)
        self.report = {"request_id": self.request_id, "prefix_key": spec["prefix_key"], "action": spec["action"],
            "initial_frontier_requested": spec.get("initial_frontier", 0),
            "prefix_length": prefix["length"], "prefix_cache_sha256": prefix.get("cache_sha256"),
            "passed": False, "gpu_queue_wait_s": 0.0, "gpu_admissions": 0, "remote_calls": 0,
            "remote_audit": [], "holder_token_ids": [], "transfers": [], "gpu_d2h_copies": 0,
            "gpu_d2h_bytes": 0, "gpu_d2h_wall_s": 0.0, "frames": {}, "errors": [],
            "physical_wire_bytes": None, "wire_measurement_available": False}
        self.report['frame_events']=[]
        try:
            for name, port in ports.items(): self.listeners[name] = _listener(port, timeout)
        except BaseException:
            self.abort("listener_initialization_failed")
            raise

    def _count(self, name, direction, stats):
        with self.lock:
            counters = self.report["frames"].setdefault(name, {}).setdefault(direction, {
                "frames": 0, "application_frame_bytes": 0, "application_payload_bytes": 0,
                "application_header_bytes": 0, "length_prefix_bytes": 0})
            counters["frames"] += 1
            for key in counters:
                if key != "frames": counters[key] += stats[key]

    def send(self, name, metadata, tensors=()):
        stats = self.channels[name].send_message({"request_id": self.request_id, **metadata}, tensors)
        self._count(name, "sent", stats)
        return stats

    def recv(self, name):
        message, tensors, stats = self.channels[name].recv_message(tensor_format="bytes")
        self._count(name, "received", stats)
        if message.get("request_id") != self.request_id:
            raise ValueError("Request ID mismatch")
        return message, tensors

    def _accept(self, name, expected_kind, reply_kind):
        raw, address = self.listeners[name].accept()
        raw.settimeout(self.timeout)
        try:
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self.lock:
                self.wrapped[name] = self.coordinator.wrap(raw, f"{self.request_id}:{name}", "rpc" if name == "rpc" else "bulk")
                self.channels[name] = FramedConnection(self.wrapped[name], integrity_mode=self.integrity)
        except BaseException:
            raw.close()
            raise
        message, tensors = self.recv(name)
        if message.get("kind") != expected_kind or tensors:
            raise ValueError(f"Invalid {name} opening handshake")
        self.report.setdefault("peer_addresses", {})[name] = list(address)
        self.send(name, {"kind": reply_kind, "prefix_key": self.spec["prefix_key"],
                         "layers": self.engine.layers, "dtype": str(self.engine.dtype)})

    def abort(self, reason):
        with self.lock:
            self.report["errors"].append(str(reason))
            channels, listeners = list(self.channels.values()), list(self.listeners.values())
            migration = self.migration
        if self.smoke_lifecycle is not None:
            self.smoke_lifecycle.abort(str(reason))
        if migration is not None: migration.fail(reason)
        for channel in channels:
            try: channel.close()
            except OSError: pass
        for listener in listeners:
            try: listener.close()
            except OSError: pass

    def _migration_worker(self, start, end, transfer, control):
        torch = self.engine.torch
        key = value = cpu_key = cpu_value = None
        try:
            with torch.cuda.device(self.engine.device), torch.inference_mode():
                stream = torch.cuda.Stream(device=self.engine.device)
                for layer in range(start, end):
                    if not control.begin_frame(layer - start, timeout=max(0.0, self.deadline - time.monotonic())):
                        break
                    with self.lock:
                        self.report['frame_events'].append({'kind':'begin','layer':layer,
                            'transfer_index':transfer,'source_local_s':time.monotonic(),'payload_bytes':None})
                    if self.smoke_lifecycle is not None:
                        self.smoke_lifecycle.after_begin_frame(layer-start)
                    key, value = self.prefix["pairs"][layer]
                    cpu_key = torch.empty(key.shape, dtype=key.dtype, device="cpu", pin_memory=True)
                    cpu_value = torch.empty(value.shape, dtype=value.dtype, device="cpu", pin_memory=True)
                    copied_at = time.monotonic()
                    with torch.cuda.stream(stream):
                        cpu_key.copy_(key, non_blocking=True)
                        cpu_value.copy_(value, non_blocking=True)
                    stream.synchronize()
                    byte_count = key.numel() * key.element_size() + value.numel() * value.element_size()
                    with self.lock:
                        self.report["gpu_d2h_copies"] += 2
                        self.report["gpu_d2h_bytes"] += byte_count
                        self.report["gpu_d2h_wall_s"] += time.monotonic() - copied_at
                    self.send("bulk", {"kind": "prefix", "layer": layer, "transfer_index": transfer}, [cpu_key, cpu_value])
                    control.finish_frame(layer - start, byte_count)
                    with self.lock:
                        self.report['frame_events'].append({'kind':'finish','layer':layer,
                            'transfer_index':transfer,'source_local_s':time.monotonic(),'payload_bytes':byte_count})
                    if self.smoke_lifecycle is not None:
                        self.smoke_lifecycle.after_finish_frame(layer-start,byte_count)
                    self.frontier = layer + 1
                    key = value = cpu_key = cpu_value = None
                terminal = control.terminal_message()
                self.send("bulk", {"kind": "migration_terminal", "transfer_index": transfer,
                    "start": start, "end": end, "next_layer": start + terminal["complete_layers"],
                    "stopped": terminal["kind"] == "migration_cancelled", "payload_bytes": terminal["payload_bytes"]})
                if self.smoke_lifecycle is not None:
                    self.smoke_lifecycle.after_terminal_sent(terminal)
                with self.lock:
                    self.report["transfers"].append({"start": start, "end": end, "transfer_index": transfer,
                        "next_layer": self.frontier, "control": control.snapshot()})
        except BaseException as error:
            self.migration_error = f"{type(error).__name__}: {error}"
            control.fail(self.migration_error)
            self.abort("migration_failed: " + self.migration_error)
        finally:
            key = value = cpu_key = cpu_value = None

    def _start_transfer(self, message):
        start, end = _integer(message.get("start"), "start"), _integer(message.get("end"), "end", 1)
        if "bulk" not in self.channels or start != self.frontier or not start < end <= self.engine.layers:
            raise ValueError("Transfer must extend the existing contiguous prefix frontier")
        if self.migration_thread is not None:
            self.migration_thread.join(timeout=max(0.0, min(1.0, self.deadline - time.monotonic())))
            if self.migration_thread.is_alive() or self.migration_error:
                raise RuntimeError("Previous transfer is still active or failed")
            if not self.migration.snapshot()["receiver_drained_ack"]:
                raise RuntimeError("Previous transfer must have a verified receiver drain acknowledgement")
        initially_paused = message.get("initially_paused", False)
        if type(initially_paused) is not bool:
            raise ValueError("initially_paused must be boolean")
        self.transfer_index += 1
        self.transfer_start=start
        self.migration = MigrationControl(self.request_id, end - start, initially_paused=initially_paused)
        if self.spec.get('e3_smoke_fixture') is not None:
            from e3_smoke_control import FrameLifecycleFixture
            if self.smoke_lifecycle is not None:
                raise ValueError('Lifecycle smoke has one real transfer only')
            self.smoke_lifecycle=FrameLifecycleFixture(mode=self.engine.e3_execution_mode,
                spec=self.spec['e3_smoke_fixture'],request_id=self.request_id,
                migration_control=self.migration,deadline_monotonic=self.deadline,
                transfer_index=self.transfer_index,transfer_start=start)
        self.migration_thread = threading.Thread(target=self._migration_worker,
            args=(start, end, self.transfer_index, self.migration), daemon=True)
        self.migration_thread.start()
        self.send("rpc", {"kind": "transfer_started", "start": start, "end": end, "transfer_index": self.transfer_index,
                          "initially_paused": initially_paused, "paused": initially_paused})

    def _drained(self, message):
        if self.migration_thread is None: raise ValueError("No transfer to acknowledge")
        self.migration_thread.join(max(0.0, self.deadline - time.monotonic()))
        if self.migration_thread.is_alive() or self.migration_error or not self.report["transfers"]:
            raise RuntimeError("Migration terminal was not sent cleanly")
        transfer = self.report["transfers"][-1]
        expected = {key: transfer[key] for key in ("transfer_index", "start", "end", "next_layer")}
        expected["payload_bytes"] = transfer["control"]["payload_bytes"]
        if any(type(message.get(key)) is not int or message[key] != value for key, value in expected.items()):
            raise ValueError("Receiver drain acknowledgement differs from the actual migration terminal")
        self.migration.acknowledge_drained({"kind": "migration_drained", "request_id": self.request_id,
            "complete_layers": expected["next_layer"] - expected["start"], "payload_bytes": expected["payload_bytes"]})
        transfer["control"] = self.migration.snapshot()
        if self.smoke_lifecycle is not None:
            self.smoke_lifecycle.after_receiver_drained({'kind':'migration_drained',
                'request_id':self.request_id,'complete_layers':expected['next_layer']-expected['start'],
                'payload_bytes':expected['payload_bytes']})
        self.send("rpc", {"kind": "migration_drained_ack", **expected, "drained": True})

    def _attention(self, message, tensors):
        if len(tensors) != 3 or self.spec["action"] == "holder":
            raise ValueError("Attention expects exactly Q/K/V for a remote request")
        layer, position = message.get("layer"), message.get("position")
        if any("torch." + t.dtype != str(self.engine.dtype) for t in tensors):
            raise ValueError("Attention wire dtype/device mismatch")
        torch = self.engine.torch
        tensors = [torch.frombuffer(t.data, dtype=getattr(torch, t.dtype)).reshape(t.shape).detach() for t in tensors]
        before = self.report["gpu_queue_wait_s"]
        gpu_tensors = output = cpu_output = None
        with self.gpu.acquire(self.report, self.deadline, "holder_attention"):
            gpu_tensors = [tensor.to(self.engine.device) for tensor in tensors]
            output = self.attention.call(layer, position, *gpu_tensors)
            cpu_output = output.to(device="cpu", copy=True)
        self.send("rpc", {"kind": "attention_result", "layer": layer, "position": position}, [cpu_output])
        self.report["remote_calls"] += 1
        self.report["remote_audit"].append({"layer": layer, "position": position,
            "gpu_queue_wait_s": self.report["gpu_queue_wait_s"] - before})
        gpu_tensors = output = cpu_output = None

    def _decode(self, message, tensors):
        if self.spec["action"] != "holder" or tensors or self.report["holder_token_ids"]:
            raise ValueError("Exactly one holder decode is permitted")
        suffix = _token_ids(message.get("suffix_ids"), "suffix_ids", self.engine.model.config.vocab_size)
        generated = _integer(message.get("generated"), "generated", 1)
        if suffix != self.spec["suffix_ids"] or generated != self.spec["generated"] or type(message.get("diagnostic", False)) is not bool:
            raise ValueError("Holder decode differs from the batch request")
        diagnostic = message.get("diagnostic", False)
        torch = self.engine.torch
        store = result = rows = suffix_gpu = None
        def token(step, value):
            self.report["holder_token_ids"].append(value)
            self.send("rpc", {"kind": "token", "step": step, "token": value})
        try:
            queue = self.engine.compute_queue
            # Legacy no-queue path is preserved; queued execution yields between
            # token forwards and does not hold admission while streaming tokens.
            scope = (queue.bind(self.request_id, self.report, self.deadline) if queue is not None
                     else self.gpu.acquire(self.report, self.deadline))
            with scope:
                store = self.engine.store_from_prefix(self.prefix["pairs"])
                suffix_gpu = torch.tensor([suffix], dtype=torch.int64, device=self.engine.device)
                result = self.engine.decode(store, self.prefix["length"], suffix_gpu, generated,
                                            on_token=token, capture_logits=diagnostic)
            self.report["decode_state"] = result["state_report"]
            if diagnostic:
                rows = torch.cat(result["logits"], dim=0)
                self.send("rpc", {"kind": "full_logits", "generated": generated}, [rows])
            self.send("rpc", {"kind": "decode_done", "generated": generated, "state_report": self.report["decode_state"]})
        finally:
            if store is not None: store.values.clear()
            store = result = rows = suffix_gpu = None

    def run(self):
        self.report["started_monotonic_s"] = time.monotonic()
        finished = False
        try:
            self._accept("rpc", "begin", "begin_ready")
            if "bulk" in self.listeners: self._accept("bulk", "hello", "bulk_ready")
            with self.gpu.acquire(self.report, self.deadline):
                self.attention = HolderAttention(self.engine.model, self.prefix["pairs"], self.prefix["length"],
                                                 compute_queue=self.engine.compute_queue)
            while True:
                if time.monotonic() >= self.deadline: raise TimeoutError("Request exceeded batch deadline")
                message, tensors = self.recv("rpc")
                kind = message.get("kind")
                if kind == "attention": self._attention(message, tensors)
                elif kind == "decode": self._decode(message, tensors)
                elif tensors: raise ValueError("Control RPC must not carry tensors")
                elif kind == "transfer": self._start_transfer(message)
                elif kind == "migration_drained": self._drained(message)
                elif kind == "e2_checkpoint_witness":
                    from checkpoint_witness import capture_holder_witness
                    mode = message.get("mode")
                    if mode not in ("metadata", "content"):
                        raise ValueError("Explicit checkpoint witness mode required")
                    witness = capture_holder_witness(self, mode=mode,
                        model_identity=self.engine.model_identity,
                        processed_delta_tokens=message.get("processed_delta_tokens"),
                        committed_layers=message.get("committed_layers"))
                    self.report.setdefault("checkpoint_witnesses", []).append(witness)
                    self.send("rpc", {"kind": "e2_checkpoint_witness_ack", "witness": witness})
                elif kind == 'e3_observe':
                    from e3_transport_control import digest
                    actuator=getattr(self.engine,'e3_actuator',None)
                    sequence=message.get('sequence')
                    if actuator is None or type(sequence) is not int or sequence<=self.e3_observation_sequence:
                        raise ValueError('Missing E3 actuator or repeated observation sequence')
                    self.e3_observation_sequence=sequence
                    evidence=actuator.current_observation()
                    evidence['migration_evidence']=None if self.migration is None else {
                        'transfer_index':self.transfer_index,'transfer_start':self.transfer_start,
                        'migration':self.migration.snapshot(),'source_snapshot_local_s':time.monotonic()}
                    reply={'kind':'e3_observation','sequence':sequence,'evidence':evidence,'evidence_sha256':digest(evidence)}
                    self.report.setdefault('e3_observations',[]).append(reply)
                    self.send('rpc',reply)
                elif kind in ('e3_smoke_snapshot','e3_smoke_fault'):
                    if self.smoke_lifecycle is None or getattr(self.engine,'e3_execution_mode',None)!='e3_smoke':
                        raise ValueError('Smoke RPC requires a source-bound excluded fixture')
                    if kind=='e3_smoke_fault':
                        reason=message.get('reason')
                        if reason!='InjectedE3OwnedCleanupFault: after complete G8 decode with live migration' or self.spec['generated']!=8:
                            raise ValueError('Foreign application fault identity or output count')
                        self.smoke_lifecycle.release_after_injected_fault(reason)
                    self.send('rpc',{'kind':kind+'_ack','fixture':self.smoke_lifecycle.snapshot()})
                elif kind in ("pause", "resume", "stop"):
                    if self.migration is None: raise ValueError("No migration exists")
                    from e3_transport_control import MigrationCommandService
                    reply=MigrationCommandService(self.migration,transfer_index=self.transfer_index).command(kind)
                    reply['next_layer']=self.frontier
                    self.report.setdefault('migration_command_acks',[]).append(reply)
                    self.send('rpc',reply)
                    if self.smoke_lifecycle is not None:
                        self.smoke_lifecycle.after_control_ack_sent(kind,reply)
                elif kind == "probe": self.send("rpc", {"kind": "probe_result", **self.gpu.snapshot()})
                elif kind == "finish":
                    if self.spec["action"] == "holder" and len(self.report["holder_token_ids"]) != self.spec["generated"]:
                        raise RuntimeError("Holder request finished without all requested generated tokens")
                    if self.migration is not None: self.migration.stop()
                    if self.migration_thread is not None:
                        self.migration_thread.join(timeout=max(0.0, self.deadline - time.monotonic()))
                        if self.migration_thread.is_alive() or self.migration_error:
                            raise RuntimeError("Migration did not drain cleanly before finish")
                        if not self.migration.snapshot()["receiver_drained_ack"]:
                            raise RuntimeError("Receiver must acknowledge the actual migration terminal before finish")
                    self.report["attention_state"] = self.attention.close()
                    if "token_ids" in message:
                        _token_ids(message["token_ids"], "token_ids", self.engine.model.config.vocab_size)
                        if len(message["token_ids"]) != self.spec["generated"]: raise ValueError("Finish token count differs")
                        self.report["destination_reported_token_ids"] = message["token_ids"]
                    self.send("rpc", {"kind": "finish_ack", "remote_calls": self.report["remote_calls"],
                        "next_layer": self.frontier, "owned_delta_references_released": True})
                    finished = True
                    break
                else: raise ValueError("Unknown request RPC: " + str(kind))
                tensors = None
        except BaseException as error:
            self.report["exception"] = f"{type(error).__name__}: {error}"
            self.report["traceback"] = traceback.format_exc()
            self.abort(self.report["exception"])
        finally:
            if self.migration is not None: self.migration.stop()
            if self.migration_thread is not None: self.migration_thread.join(10)
            for channel in self.channels.values(): channel.close()
            for listener in self.listeners.values(): listener.close()
            migration_alive = self.migration_thread is not None and self.migration_thread.is_alive()
            if self.attention is not None:
                try: self.report["attention_state"] = self.attention.close()
                except BaseException as error: self.report["errors"].append("delta_cleanup: " + str(error))
            transport = {name: wrapped.snapshot() for name, wrapped in self.wrapped.items()}
            self.report["transport"] = transport
            self.report["traffic"] = {name: None for name in ("rpc", "bulk")}
            accounting_pass = True
            for name, counters in transport.items():
                frames = self.report["frames"].get(name, {})
                sent, received = frames.get("sent", {}), frames.get("received", {})
                self.report["traffic"][name] = {"sent": counters["sent_bytes"], "received": counters["received_bytes"],
                    "payload_sent": sent.get("application_payload_bytes", 0),
                    "payload_received": received.get("application_payload_bytes", 0),
                    "completed_frame_sent": sent.get("application_frame_bytes", 0),
                    "completed_frame_received": received.get("application_frame_bytes", 0)}
                accounting_pass = accounting_pass and (counters["sent_bytes"] == sent.get("application_frame_bytes", 0)
                    and counters["received_bytes"] == received.get("application_frame_bytes", 0))
            self.report["completed_frame_accounting_matches_actual_socket_bytes"] = accounting_pass
            self.report["cleanup"] = {"migration_thread_alive": migration_alive,
                "sockets_closed": all(channel.fileno() == -1 for channel in self.channels.values()),
                "listeners_closed": all(listener.fileno() == -1 for listener in self.listeners.values()),
                "delta_references_released": self.report.get("attention_state", {}).get("owned_delta_references_released", False)}
            self.report["passed"] = (finished and accounting_pass and not self.report["errors"] and not migration_alive
                and all(self.report["cleanup"][key] for key in ("sockets_closed", "listeners_closed", "delta_references_released")))
            self.report["finished_monotonic_s"] = time.monotonic()
            self.report['smoke_lifecycle']=None if self.smoke_lifecycle is None else self.smoke_lifecycle.snapshot()
            self.report["acceptance_scope"] = "Holder protocol, actual transfers, and owned-delta cleanup; destination numerical acceptance is separate"


def _batch(engine, gpu, prefixes, control, base_port, command, coordinator_override=None):
    specs = command.get("requests")
    if not isinstance(specs, list) or not 1 <= len(specs) <= 32:
        raise ValueError("Batch must contain 1..32 requests")
    timeout = command.get("timeout_s", 600)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 1 <= timeout <= 1200:
        raise ValueError("timeout_s must be finite in [1,1200]")
    deadline = time.monotonic() + timeout
    ids = set()
    for spec in specs:
        request_id = spec.get("request_id")
        if not isinstance(request_id, str) or not request_id or request_id in ids: raise ValueError("Unique request IDs required")
        ids.add(request_id)
        if spec.get("prefix_key") not in prefixes: raise ValueError("Requested prefix was not prepared")
        _token_ids(spec.get("suffix_ids"), "suffix_ids", engine.model.config.vocab_size)
        _integer(spec.get("generated"), "generated", 1)
        if not isinstance(spec.get("action"), str) or not spec["action"]: raise ValueError("Action is required")
        if not 0 <= _integer(spec.get("initial_frontier", 0), "initial_frontier") <= engine.layers: raise ValueError("Invalid initial frontier")
    integrity = command.get("integrity_mode", "none")
    if integrity not in ("none", "sha256"): raise ValueError("Invalid integrity mode")
    epoch = command.get("rate_epoch_monotonic", time.monotonic())
    if coordinator_override is not None:
        if command.get("rate_plan") is not None:
            raise ValueError("Full E1 uses one constant preconfigured trial budget")
        coordinator = coordinator_override
    elif command.get("rate_plan") is not None:
        raw = command["rate_plan"]
        plan = RatePlan(raw["steps"], plan_id=raw["plan_id"], seed=raw["seed"])
        limiter = VariableRateLimiter(plan, epoch_monotonic=epoch, chunk_bytes=16 * 1024)
        coordinator = SharedBandwidthCoordinator(limiter=limiter, priority=command.get("priority", "rpc_first"))
    else:
        coordinator = SharedBandwidthCoordinator(command["rate_bps"], priority=command.get("priority", "rpc_first"))
    requests, ports, watchdog = [], {}, None
    settled = threading.Event()
    try:
        for index, spec in enumerate(specs):
            endpoint = {"rpc": base_port + 10 + 2 * index}
            if spec["action"] != "holder": endpoint["bulk"] = endpoint["rpc"] + 1
            ports[spec["request_id"]] = endpoint
            requests.append(_Request(engine, gpu, prefixes[spec["prefix_key"]], spec, endpoint,
                                     coordinator, timeout, deadline, integrity))
        control.send_message({"kind": "batch_ready", "ports": ports, "integrity_mode": integrity,
            "rate_epoch_monotonic": epoch, "epoch_scope": "Holder-local; no cross-host synchronization inferred",
            "rate_plan_sha256": None if command.get("rate_plan") is None else plan.sha256})
        def guard():
            if not settled.wait(max(0.0, deadline - time.monotonic())):
                for request in requests: request.abort("batch_hard_deadline_expired")
                if coordinator_override is None:
                    coordinator.close()
        watchdog = threading.Thread(target=guard, daemon=True)
        watchdog.start()
        for request in requests:
            request.thread = threading.Thread(target=request.run, daemon=True)
            request.thread.start()
        for request in requests:
            request.thread.join(max(0.0, deadline + 10 - time.monotonic()))
        settled.set()
        watchdog.join(10)
        if coordinator_override is None:
            coordinator.close()
        alive = [request.request_id for request in requests if request.thread.is_alive()]
        passed = not alive and all(request.report["passed"] for request in requests)
        response = {"kind": "batch_done", "passed": passed, "requests": [request.report for request in requests],
            "coordinator": coordinator.snapshot(), "gpu_admission": gpu.snapshot(), "alive_request_threads": alive,
            "cleanup_deadline_s": 10, "GPU_performance_acceptance": False}
        control.send_message(response)
        if not passed: raise RuntimeError("Holder batch failed; inspect batch_done reports")
        return response
    finally:
        settled.set()
        if coordinator_override is None:
            coordinator.close()
        for request in requests:
            if request.thread is None or request.thread.is_alive(): request.abort("batch_scope_closed")
        if watchdog is not None: watchdog.join(10)


def serve_holder(engine, control: FramedConnection, base_port: int):
    """Blocking holder control loop; failure is reported and then re-raised.

    prepare {prefix_ids,prefix_key}; probe; batch as documented in the module;
    stop. Request-level probe remains available while control waits for a batch.
    No cache is evicted while requests or migration borrow its immutable tensors.
    """
    if not isinstance(control, FramedConnection): raise TypeError("control must be FramedConnection")
    _integer(base_port, "base_port", 1024)
    if base_port + 74 > 65535: raise ValueError("Insufficient TCP port range")
    prefixes, gpu = {}, _SharedGPU(engine)
    from full_server import FullServer
    full = FullServer(engine, control)
    try:
        while True:
            message, tensors, _ = control.recv_message(tensor_format="bytes")
            if tensors: raise ValueError("Holder control uses metadata only")
            kind = message.get("kind")
            if kind.startswith("full_"):
                if not full.handle(message):
                    raise ValueError("Unknown full E3 command: " + kind)
                continue
            if kind=='e3_epoch_start':
                full.start_epoch(message)
                continue
            if kind == "prepare":
                key = message.get("prefix_key")
                if not isinstance(key, str) or not key: raise ValueError("prefix_key is required")
                ids = _token_ids(message.get("prefix_ids"), "prefix_ids", engine.model.config.vocab_size)
                token_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
                if key in prefixes:
                    if prefixes[key]["token_ids_sha256"] != token_hash: raise ValueError("Immutable prefix key reused for different tokens")
                else:
                    if len(prefixes) >= 8: raise ValueError("Eight prefix keys retained; explicitly retire an unused prefix first")
                    pairs = engine.prefix(engine.torch.tensor([ids], dtype=engine.torch.int64, device=engine.device))
                    cpu_pairs = cache_to_cpu_pairs(pairs)
                    prefix_hash = cache_sha256(cpu_pairs)
                    cpu_pairs = None
                    prefixes[key] = {"pairs": pairs, "length": len(ids), "token_ids_sha256": token_hash,
                        "cache_sha256": prefix_hash,
                        "cache_bytes": sum(t.numel() * t.element_size() for pair in pairs for t in pair)}
                    pairs = None
                control.send_message({"kind": "prepared", "prefix_key": key, "layers": engine.layers,
                    "cache_bytes": prefixes[key]["cache_bytes"], "cache_sha256": prefixes[key]["cache_sha256"],
                    "prefix_length": len(ids), "token_ids_sha256": token_hash})
            elif kind == "probe":
                control.send_message({"kind": "probe_result", "prefix_keys": sorted(prefixes), **gpu.snapshot()})
            elif kind == "batch":
                if full.budget is None:
                    raise ValueError("Full E3 batch requires trial environment and byte budget")
                if full.epoch_started is None:
                    raise ValueError('E3 batch preceded the actual local epoch marker')
                if message.get("local_request") is True:
                    full.local_batch(message)
                else:
                    _batch(engine, gpu, prefixes, control, base_port, message, coordinator_override=full.budget)
            elif kind == "retire":
                key = message.get("prefix_key")
                refs = [weakref.ref(t) for pair in prefixes[key]["pairs"] for t in pair]
                del prefixes[key]
                gc.collect()
                released = all(ref() is None for ref in refs)
                if not released: raise RuntimeError("Retired prefix still has live tensor references")
                control.send_message({"kind": "retired", "prefix_key": key, "references_released": released})
            elif kind == "stop":
                if full.budget is not None:
                    raise ValueError("Cannot stop while a full trial is armed")
                full.close()
                refs = [weakref.ref(t) for item in prefixes.values() for pair in item["pairs"] for t in pair]
                prefixes.clear()
                gc.collect()
                released = all(ref() is None for ref in refs)
                if not released: raise RuntimeError("Holder shared prefix references remain at stop")
                control.send_message({"kind": "stopped", "shared_prefix_references_released": released})
                return {"passed": True, "shared_prefix_references_released": released}
            else: raise ValueError("Unknown holder control command: " + str(kind))
    except BaseException as error:
        try: control.send_message({"kind": "holder_error", "error": f"{type(error).__name__}: {error}"})
        except BaseException: pass
        raise
    finally:
        try:
            full.close()
        finally:
            prefixes.clear()
            control.close()
