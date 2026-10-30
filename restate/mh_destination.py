"""E6/E7 target (destination) orchestration over several replica holders.

Generalizes E4Destination to N holder control channels. Each holder runs the
unmodified E4 arm/epoch/batch/settle/disarm protocol on its own sub-trial, with
its own outgoing application budget (the holder->target path). Requests bound
to the same holder therefore share that path and that holder's GPU queue.
Target-local recompute jobs use only the target GPU. Requests may arrive after
the common acceptance instant (arrival_offset_s); every request metric is taken
relative to its own arrival. Clocks are target-native monotonic only.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import json
import math
import socket
import threading
import time
import traceback

from e4_destination import OwnedResources, DestinationError, digest, require, _totals
from e4_scheduler import E4BandwidthCoordinator, policy_config
from mh_group_execution import OfferedJob, execute_group

HOLDER_ACTIONS = ("remote", "repair", "holder")
LOCAL_ACTIONS = ("recompute",)
OWNED_ACTIONS = ("remote", "repair")


def holder_subtrial(trial, holder):
    """The exact E4-format sub-trial that one holder arms (its own requests only)."""
    spec = trial["holders"][str(holder)]
    requests = []
    for item in trial["requests"]:
        if item["action"] in HOLDER_ACTIONS and item["holder"] == holder:
            requests.append({k: copy.deepcopy(item[k]) for k in (
                "role", "input_id", "input_family_id", "seed", "prefix_tokens", "suffix_tokens",
                "prefix_key", "action", "public_output_cap", "executor_only")})
    return {"trial_id": spec["subtrial_id"], "requests": requests, "transfer_jobs": [],
            "scheduler": copy.deepcopy(spec["scheduler"]), "load_location": spec["load_location"],
            "load_seed": spec["load_seed"], "added_RTT_ms": spec.get("added_RTT_ms", 0)}


def production_dependencies():
    from types import SimpleNamespace
    from campaign_client import execute
    from e3_background_load import CachedLocalBackground
    from full_control import reconcile_control
    return SimpleNamespace(execute=execute, Background=CachedLocalBackground, reconcile_control=reconcile_control)


class MHDestination:
    def __init__(self, engine, holders, *, trials, dependencies=None):
        require(isinstance(holders, list) and holders and all(set(h) == {"index", "control", "host"} for h in holders),
                "Explicit holder index/control/host rows required")
        require([h["index"] for h in holders] == list(range(len(holders))), "Holders must be ordered 0..N-1")
        require(isinstance(trials, dict) and trials and all(k == v.get("trial_id") for k, v in trials.items()),
                "Frozen trial registry required")
        self.engine, self.holders = engine, holders
        self.trials = copy.deepcopy(trials)
        self.deps = dependencies or production_dependencies()
        self.queue = engine.compute_queue
        require(self.queue is not None and self.queue.backend.kind == "cuda_stream_completion", "Actual GPU queue required")
        require(engine.layers == 48 and engine.kv_heads == 8 and engine.head_dim == 128
                and str(engine.dtype) == "torch.bfloat16", "Frozen main Qwen14B BF16 geometry required")
        self.background = None
        self.prepared = [OrderedDict() for _ in holders]
        self.closed = self.active = self.poisoned = False
        self.sequence = 0
        self.resources = []

    def _idle(self):
        state = self.queue.state()
        require(state["active_gpu_operations"] == state["queued_gpu_operations"] == 0, "Destination queue not idle")
        return state

    def _recv(self, holder, kind, **identity):
        value, tensors, _ = self.holders[holder]["control"].recv_message(tensor_format="bytes")
        require(not tensors and value.get("kind") == kind and all(value.get(k) == v for k, v in identity.items()),
                "Holder %d control reply identity differs: %s" % (holder, kind))
        return value

    def _send(self, holder, message):
        self.holders[holder]["control"].send_message(message)

    def initialize(self, background_prefix_ids, background_suffix_ids):
        require(not self.closed and self.background is None and not self.active, "Destination already initialized")
        self._idle()
        self.background = self.deps.Background(self.engine, self.queue, background_prefix_ids, background_suffix_ids, generated=32)
        self.background.prepare()
        calibration = self.background.calibrate()
        require(calibration["warmup_jobs"] == 2 and calibration["measured_jobs"] == 10
                and math.isfinite(calibration["mean_service_s"]) and calibration["mean_service_s"] > 0
                and calibration.get("GPU_service_evidence") is True, "Isolated real GPU calibration required")
        sources = []
        for h in range(len(self.holders)):
            self._send(h, dict(kind="full_initialize", background_prefix_ids=background_prefix_ids,
                               background_suffix_ids=background_suffix_ids, generated=32))
            source = self._recv(h, "full_initialized")
            require(source.get("CPU_component_only") is False, "Holder execution scope differs")
            sources.append(source)
        self.queue.clear_completed_events()
        return dict(destination_calibration=calibration, holders=sources)

    def _prepare_holder_prefix(self, holder, key, ids):
        token_sha = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        cache = self.prepared[holder]
        if key in cache:
            require(cache[key]["token_ids_sha256"] == token_sha, "Immutable prefix key reused for other tokens")
            cache.move_to_end(key)
            return copy.deepcopy(cache[key])
        while len(cache) >= 8:
            old = next(iter(cache))
            self._send(holder, dict(kind="retire", prefix_key=old))
            retired = self._recv(holder, "retired", prefix_key=old)
            require(retired.get("references_released") is True, "Holder retired prefix remains referenced")
            cache.pop(old)
        self._send(holder, dict(kind="prepare", prefix_key=key, prefix_ids=ids))
        prepared = self._recv(holder, "prepared", prefix_key=key)
        require(prepared.get("token_ids_sha256") == token_sha and prepared.get("prefix_length") == len(ids)
                and prepared.get("layers") == self.engine.layers, "Holder prepared a different prefix")
        cache[key] = prepared
        return copy.deepcopy(prepared)

    def _validate(self, trial, values):
        require(isinstance(values, dict) and set(values) == {r["role"] for r in trial["requests"]},
                "Exact request roles and token arrays required")
        vocab = self.engine.model.config.vocab_size
        holders = set()
        for item in trial["requests"]:
            tokens = values[item["role"]]
            require(item["action"] in HOLDER_ACTIONS + LOCAL_ACTIONS and item["suffix_tokens"] == 16
                    and item["executor_only"]["realized_stop_tokens"] == 64 and item["prefix_tokens"] in (1024, 4096),
                    "Frozen MH request geometry/action changed")
            off = item["arrival_offset_s"]
            require(type(off) in (int, float) and math.isfinite(off) and 0 <= off <= 600, "Finite arrival offset required")
            if item["action"] in HOLDER_ACTIONS:
                require(type(item["holder"]) is int and 0 <= item["holder"] < len(self.holders), "Bound holder required")
                holders.add(item["holder"])
            else:
                require(item["holder"] is None, "Local recompute cannot bind a holder")
            for field, size in (("prefix_ids", item["prefix_tokens"]), ("suffix_ids", item["suffix_tokens"])):
                require(isinstance(tokens[field], list) and len(tokens[field]) == size
                        and all(type(v) is int and 0 <= v < vocab for v in tokens[field]), "Actual token geometry differs")
        require(set(int(h) for h in trial["holders"]) == holders, "Armed holder set differs from request bindings")
        for h in holders:
            sub = holder_subtrial(trial, h)
            require(1 <= len(sub["requests"]) <= 4, "Each armed holder serves one to four requests")
            require(sub["scheduler"] == policy_config(sub["scheduler"]["policy"], sub["scheduler"]["rate_bps"], sub["scheduler"]["chunk_bytes"]),
                    "Holder scheduler config differs")
        return sorted(holders)

    def run_trial(self, trial_id, role_inputs, *, diagnostic=False, replay_by_role=None, timeout_s=1200.):
        require(not self.closed and not self.poisoned and not self.active and self.background is not None,
                "Initialize an idle healthy MH session first")
        trial = copy.deepcopy(self.trials[trial_id])
        holders = self._validate(trial, role_inputs)
        if diagnostic:
            require(isinstance(replay_by_role, dict), "Diagnostic must bind main routes")
            for item in trial["requests"]:
                if item["action"] in OWNED_ACTIONS:
                    route = replay_by_role.get(item["role"])
                    require(isinstance(route, list) and len(route) == self.engine.layers, "Invalid main route")
        else:
            require(replay_by_role is None, "Main execution cannot replay")
        self._idle()
        prepared = {}
        for item in trial["requests"]:
            if item["action"] in HOLDER_ACTIONS:
                prepared[item["role"]] = self._prepare_holder_prefix(item["holder"], item["prefix_key"],
                                                                     role_inputs[item["role"]]["prefix_ids"])
        self.engine.e1_netem.set_delay_ms(0)
        self.queue.clear_completed_events()
        self.sequence += 1
        self.resources = []
        self.active = True
        subtrials = {h: holder_subtrial(trial, h) for h in holders}
        report = dict(trial_id=trial_id, trial_sha256=digest(trial), diagnostic=diagnostic, passed=False,
                      prepared_prefixes=prepared, errors=[], requests={}, holders={str(h): {} for h in holders},
                      clock_scope="target native monotonic only")
        rows = {}
        lock = threading.Lock()
        state = {"armed": [], "batched": []}
        budgets = {h: E4BandwidthCoordinator(subtrials[h]["scheduler"]["rate_bps"], subtrials[h]["scheduler"]["policy"],
                                      chunk_bytes=subtrials[h]["scheduler"]["chunk_bytes"])
                   for h in holders}
        jobs = []
        specs = {h: [] for h in holders}
        for item in trial["requests"]:
            role = item["role"]
            request = "%s.%d.%s" % (trial_id, self.sequence, role)
            tokens = role_inputs[role]
            owned = OwnedResources()
            self.resources.append(owned)
            case = dict(prefix=item["prefix_tokens"], suffix=item["suffix_tokens"],
                        generated=item["executor_only"]["realized_stop_tokens"], action=item["action"],
                        prefix_key=item["prefix_key"], initial_frontier=0)
            if item["action"] in HOLDER_ACTIONS:
                specs[item["holder"]].append(dict(request_id=request, role=role, prefix_key=item["prefix_key"],
                    suffix_ids=tokens["suffix_ids"], generated=case["generated"], action=item["action"], initial_frontier=0))

            def run(accepted, deadline, setup, cancelled, *, item=item, case=case, request=request, tokens=tokens,
                    role=role, owned=owned):
                owned.deadline = deadline
                arrival = accepted + float(item["arrival_offset_s"])
                try:
                    while True:
                        left = arrival - time.monotonic()
                        if left <= 0:
                            break
                        if cancelled.wait(min(left, .05)):
                            raise RuntimeError("Group cancelled before request arrival")
                    self.engine.torch.cuda.set_device(self.engine.device)
                    holder = item["holder"]
                    host = None if holder is None else self.holders[holder]["host"]
                    ports = None if holder is None else setup["ports"][request]
                    budget = None if holder is None else budgets[holder]
                    replay = None
                    if diagnostic and item["action"] in OWNED_ACTIONS:
                        replay = replay_by_role[role]
                    with self.queue.bind(request, deadline=deadline, cancel_event=cancelled):
                        value, logits, retained = self.deps.execute(
                            self.engine, tokens["prefix_ids"] + tokens["suffix_ids"], case, request, host, ports, budget,
                            diagnostic=diagnostic, replay=replay, accepted_s=arrival,
                            e4_owned_resources=owned if item["action"] in OWNED_ACTIONS else None)
                    require(retained is None, "Request retained a prefix")
                    if item["action"] in OWNED_ACTIONS:
                        require(value.get("own_prefix_references_released") is True and value.get("own_settled_s") is not None,
                                "Owned prefix/receiver not released")
                        settled = value["own_settled_s"]
                    else:
                        settled = value["settled_s"]
                    value.update(arrival_s=arrival, request_settled_s=settled, holder=holder, mh_action=item["action"])
                    with lock:
                        report["requests"][role] = value
                        rows[role] = logits
                    return dict(passed=value.get("passed") is True, accepted_s=accepted, settled_s=settled,
                                request_id=request, role=role, arrival_s=arrival)
                except BaseException as error:
                    value = copy.deepcopy(owned.partial_report or {})
                    value.update(passed=False, error=repr(error), traceback=traceback.format_exc(), arrival_s=arrival)
                    with lock:
                        report["requests"][role] = value
                    raise

            kind = "local" if item["action"] in LOCAL_ACTIONS else "decode"
            jobs.append(OfferedJob(request, kind, run, owned.abort))

        def setup(accepted, deadline):
            report["accepted_s"] = accepted
            if trial.get("destination_load"):
                report["background_start"] = self.background.start(seed=trial["load_seed"], rho=.5,
                                                                   lead_service_multiples=0, acceptance_monotonic_s=accepted)
            ports = {}
            for h in holders:
                control = self.holders[h]["control"]
                control.raw_socket.settimeout(max(.001, deadline - time.monotonic()))
                sub = subtrials[h]
                self._send(h, dict(kind="e4_epoch_start", trial_id=sub["trial_id"], trial_sha256=digest(sub)))
                report["holders"][str(h)]["epoch_ack"] = self._recv(h, "e4_epoch_started", trial_id=sub["trial_id"],
                                                                     trial_sha256=digest(sub))
                self._send(h, dict(kind="batch", trial_id=sub["trial_id"], requests=specs[h], transfers=[]))
                state["batched"].append(h)
                ready = self._recv(h, "batch_ready", trial_id=sub["trial_id"])
                require(set(ready["ports"]) == {s["request_id"] for s in specs[h]}, "Holder omitted/added jobs")
                report["holders"][str(h)]["batch_ready"] = ready
                ports.update(ready["ports"])
            return dict(ports=ports)

        def finalize(deadline, failed):
            done = {}
            for h in state["batched"]:
                self.holders[h]["control"].raw_socket.settimeout(max(.001, deadline - time.monotonic()))
                done[h] = self._recv(h, "batch_done", trial_id=subtrials[h]["trial_id"])
                report["holders"][str(h)]["batch_done"] = done[h]
            report["background_report"] = (self.background.stop_and_drain(timeout=max(.001, deadline - time.monotonic()))
                                           if self.background.snapshot()["running"] else None)
            self._idle()
            settled = {}
            for h in state["batched"]:
                self._send(h, dict(kind="e4_trial_settle", trial_id=subtrials[h]["trial_id"]))
                settled[h] = self._recv(h, "e4_trial_settled", trial_id=subtrials[h]["trial_id"])
            service_settled = time.monotonic()
            passed = not failed and (report["background_report"] is None or report["background_report"].get("passed") is True)
            for h in list(state["armed"]):
                sub_id = subtrials[h]["trial_id"]
                self._send(h, dict(kind="full_trial_disarm", trial_id=sub_id))
                accounting = self.holders[h]["control"].disarm()
                state["armed"].remove(h)
                source = self._recv(h, "full_trial_disarmed", trial_id=sub_id)
                coordinator = budgets[h].snapshot()
                left, right = _totals(coordinator), _totals(source["final_coordinator"])
                socket_ok = left["sent_bytes"] == right["received_bytes"] and left["received_bytes"] == right["sent_bytes"]
                reconciliation = self.deps.reconcile_control(accounting, source["control_accounting"])
                report["holders"][str(h)].update(settled=settled.get(h), source=source, control_accounting=accounting,
                    control_reconciliation=reconciliation, final_coordinator=coordinator,
                    socket_totals=dict(target=left, holder=right, passed=socket_ok))
                passed = passed and done.get(h, {}).get("passed") is True and settled.get(h, {}).get("passed") is True \
                    and source.get("passed") is True and socket_ok
            return dict(passed=passed, service_settled_s=service_settled, audit_received_s=time.monotonic())

        try:
            for h in holders:
                sub = subtrials[h]
                self._send(h, dict(kind="full_trial_arm", trial_id=sub["trial_id"], trial_sha256=digest(sub),
                                   timeout_s=timeout_s, diagnostic=diagnostic))
                arm = self._recv(h, "full_trial_armed", trial_id=sub["trial_id"], trial_sha256=digest(sub))
                require(arm["scheduler"] == sub["scheduler"], "Holder scheduler differs")
                report["holders"][str(h)]["arm"] = arm
                self.holders[h]["control"].arm(budgets[h])
                state["armed"].append(h)
            report["group"] = execute_group(jobs, setup, finalize, timeout_s=timeout_s, cleanup_timeout_s=30.)
            report["passed"] = report["group"]["passed"]
            if not report["passed"]:
                raise RuntimeError("One or more MH offered jobs/finalization failed")
        except BaseException as error:
            report["errors"].append(dict(error=repr(error), traceback=traceback.format_exc()))
            self.poisoned = True
            for owned in self.resources:
                owned.abort("MH trial failed")
            for h in self.holders:
                h["control"].close()
            raise DestinationError("MH trial failed; raw evidence retained", report) from error
        finally:
            cleanup = []
            try:
                if self.background.snapshot()["running"]:
                    report["exception_background_cleanup"] = self.background.stop_and_drain(timeout=30.)
            except BaseException as error:
                cleanup.append(repr(error))
            for h, budget in budgets.items():
                try:
                    budget.close()
                except BaseException as error:
                    cleanup.append(repr(error))
            report["cleanup"] = dict(errors=cleanup)
            if cleanup:
                report["passed"] = False
                self.poisoned = True
            report["ownership_finished_s"] = time.monotonic()
            self.active = False
        if not report["passed"]:
            raise DestinationError("MH cleanup failed; raw evidence retained", report)
        return report, rows

    def close(self):
        require(not self.active, "Join the active group before session close")
        if self.closed:
            return self.close_report
        errors, sources, background = [], [], None
        try:
            if not self.poisoned:
                for h in range(len(self.holders)):
                    self._send(h, dict(kind="stop"))
                    source = self._recv(h, "stopped")
                    require(source.get("passed") is True, "Holder session cleanup failed")
                    sources.append(source)
        except BaseException as error:
            errors.append(repr(error))
        finally:
            for h in self.holders:
                h["control"].close()
            if self.background is not None:
                try:
                    if self.background.snapshot()["running"]:
                        self.background.stop_and_drain(timeout=30.)
                    background = self.background.close()
                except BaseException as error:
                    errors.append(repr(error))
            self.closed = True
        self.close_report = dict(passed=not errors and not self.poisoned, errors=errors, holders=sources,
                                 destination_background_cleanup=background, original_trial_failed=self.poisoned)
        return self.close_report
