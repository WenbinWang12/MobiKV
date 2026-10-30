"""E2 continuation state, causal policy, and frame-boundary migration control.

These are executable controller components, not a GPU experiment or a claimed
optimal algorithm. The caller owns tensors and must capture state only after a
complete token/layer pass and publish KV only after GPU readiness. Numerical
correctness, measured counterfactuals, and GPU restart integration remain gates.
No predicted cost returned here is an experimental latency observation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import threading
import time


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ProgressState:
    request_id: str
    prefix_identity: str
    layer_payload_bytes: tuple[int, ...]
    committed_layers: tuple[int, ...]
    completed_frame_payload_bytes: int
    in_flight_layer: int | None
    generated_tokens: int
    processed_delta_tokens: int
    destination_delta_lengths: tuple[int, ...]
    holder_delta_lengths: tuple[int, ...]
    migration_state: str
    at_token_boundary: bool
    holder_prefix_available: bool
    remote_available: bool
    token_history_available: bool
    queued_work_items: int = 0
    sunk_elapsed_seconds: float = 0.0

    def __post_init__(self):
        if not self.request_id or not self.prefix_identity or not self.layer_payload_bytes:
            raise ValueError("Request, immutable-prefix identity, and layers are required")
        if any(type(value) is not tuple for value in (self.layer_payload_bytes, self.committed_layers,
                                                     self.destination_delta_lengths, self.holder_delta_lengths)):
            raise TypeError("State sequences must be immutable tuples")
        for size in self.layer_payload_bytes:
            _integer(size, "layer_payload_bytes", 1)
        for name in ("completed_frame_payload_bytes", "generated_tokens", "processed_delta_tokens", "queued_work_items"):
            _integer(getattr(self, name), name)
        for name in ("at_token_boundary", "holder_prefix_available", "remote_available", "token_history_available"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for layer in self.committed_layers:
            _integer(layer, "committed layer")
        _finite(self.sunk_elapsed_seconds, "sunk_elapsed_seconds")
        if self.committed_layers != tuple(range(len(self.committed_layers))) or len(self.committed_layers) > self.layers:
            raise ValueError("This implementation supports only ascending, GPU-ready prefix publication")
        if not self.committed_bytes <= self.completed_frame_payload_bytes <= sum(self.layer_payload_bytes):
            raise ValueError("Committed/sent prefix payload byte accounting is inconsistent")
        if self.in_flight_layer is not None:
            _integer(self.in_flight_layer, "in_flight_layer")
            if not len(self.committed_layers) <= self.in_flight_layer < self.layers:
                raise ValueError("In-flight layer is already published or outside the model")
        for lengths in (self.destination_delta_lengths, self.holder_delta_lengths):
            if len(lengths) != self.layers:
                raise ValueError("One delta length is required per layer")
            for value in lengths:
                _integer(value, "delta length")
                if value > self.processed_delta_tokens:
                    raise ValueError("Delta cannot include unprocessed future tokens")
        if self.at_token_boundary and self.destination_delta_lengths != (self.processed_delta_tokens,) * self.layers:
            raise ValueError("A token-boundary snapshot must have every destination delta aligned")
        if self.migration_state not in ("running", "paused", "stop_requested", "stopped", "complete", "failed"):
            raise ValueError("Unknown migration state")
        if self.migration_state == "complete" and self.committed_layers != tuple(range(self.layers)):
            raise ValueError("Complete migration requires every layer GPU-ready")

    @property
    def layers(self):
        return len(self.layer_payload_bytes)

    @property
    def committed_bytes(self):
        return sum(self.layer_payload_bytes[:len(self.committed_layers)])

    @property
    def remaining_payload_bytes(self):
        # Receiver-uncommitted bytes include any transmitted but not GPU-ready data.
        return sum(self.layer_payload_bytes) - self.committed_bytes

    def checkpoint_identity(self):
        if (not self.at_token_boundary or self.in_flight_layer is not None or self.queued_work_items
                or self.completed_frame_payload_bytes != self.committed_bytes):
            raise ValueError("Matched-state checkpoint requires a token/frame boundary and drained queues")
        return identity(asdict(self))


@dataclass(frozen=True)
class ExecutionCapabilities:
    frame_pause_resume: bool = False
    graceful_stop_and_drain: bool = False
    block_foreground_until_fetch: bool = False
    full_history_recompute_and_delta_replace: bool = False
    full_prefix_recompute_keep_delta: bool = False

    def __post_init__(self):
        if any(type(value) is not bool for value in asdict(self).values()):
            raise ValueError("Capabilities must be explicit booleans from the executor")


def legal_continuations(state: ProgressState, capabilities: ExecutionCapabilities):
    """No independent historical layer recomputation or unsupported holder handoff."""
    if not state.at_token_boundary:
        return ()
    if len(state.committed_layers) == state.layers:
        return ("continue_local",)
    missing = range(len(state.committed_layers), state.layers)
    remote_usable = state.remote_available and all(
        state.holder_delta_lengths[layer] == state.processed_delta_tokens for layer in missing)
    migration_usable = state.holder_prefix_available and state.migration_state in ("running", "paused")
    actions = []
    if migration_usable and remote_usable:
        actions.append("continue_repair")
    if remote_usable and capabilities.frame_pause_resume and state.migration_state in ("running", "paused"):
        actions.append("pause_migration_and_remote")
    if remote_usable and capabilities.graceful_stop_and_drain and state.migration_state not in ("failed", "complete"):
        actions.append("stop_migration_and_remote")
    if migration_usable and capabilities.block_foreground_until_fetch:
        actions.append("complete_fetch_then_local")
    if (capabilities.full_history_recompute_and_delta_replace and state.token_history_available
            and state.in_flight_layer is None and state.queued_work_items == 0):
        actions.append("recompute_full_history_then_local")
    if (capabilities.full_prefix_recompute_keep_delta and state.token_history_available
            and state.in_flight_layer is None and state.queued_work_items == 0):
        actions.append("recompute_full_prefix_keep_delta")
    return tuple(actions)


@dataclass(frozen=True)
class RemainingEstimate:
    generated_tokens: int
    mean_remaining_tokens: float
    training_identity: str
    survivor_samples: int
    fallback_used: bool

    def __post_init__(self):
        _integer(self.generated_tokens, "generated_tokens")
        _integer(self.survivor_samples, "survivor_samples")
        _finite(self.mean_remaining_tokens, "mean_remaining_tokens", 1)
        if not self.training_identity or type(self.fallback_used) is not bool:
            raise ValueError("Estimate must carry frozen population provenance")


class EmpiricalRemainingEstimator:
    """Conditional mean from a frozen historical length population, never true future.

    Fit only on training requests. The caller records that split; a content hash
    makes the population auditable but cannot certify its provenance by itself.
    """
    def __init__(self, training_total_lengths, *, training_split_id, fallback_remaining=16.0):
        self.lengths = tuple(training_total_lengths)
        if not self.lengths or not isinstance(training_split_id, str) or not training_split_id:
            raise ValueError("A nonempty frozen training population and split identity are required")
        for value in self.lengths:
            _integer(value, "training length", 1)
        _finite(fallback_remaining, "fallback_remaining", 1)
        self.fallback = float(fallback_remaining)
        self.training_identity = identity({"split": training_split_id, "lengths": self.lengths,
                                           "fallback_remaining": self.fallback})

    def estimate(self, generated_tokens):
        _integer(generated_tokens, "generated_tokens")
        survivors = [length - generated_tokens for length in self.lengths if length > generated_tokens]
        return RemainingEstimate(generated_tokens, sum(survivors) / len(survivors) if survivors else self.fallback,
                                 self.training_identity, len(survivors), not survivors)


@dataclass(frozen=True)
class CostObservation:
    sampled_at: float
    available_at: float
    local_step_seconds: float
    all_remote_step_seconds: float
    effective_prefix_payload_bytes_per_second: float
    full_history_recompute_seconds: float | None = None
    switching_seconds: float = 0.0
    source: str = "measured calibration"
    full_prefix_recompute_seconds: float | None = None

    def __post_init__(self):
        for name in ("sampled_at", "available_at", "local_step_seconds", "all_remote_step_seconds",
                     "effective_prefix_payload_bytes_per_second", "switching_seconds"):
            _finite(getattr(self, name), name)
        if self.available_at < self.sampled_at or self.local_step_seconds <= 0 or self.all_remote_step_seconds <= 0:
            raise ValueError("Observation time or measured step duration is invalid")
        if self.full_history_recompute_seconds is not None:
            _finite(self.full_history_recompute_seconds, "full_history_recompute_seconds")
        if self.full_prefix_recompute_seconds is not None:
            _finite(self.full_prefix_recompute_seconds, "full_prefix_recompute_seconds")
        if not self.source:
            raise ValueError("A calibration/measurement source identity is required")


class ProgressPolicy:
    """Small prospective-cost baseline with stale fallback and switch hysteresis.

    Linear decay of remote-layer overhead during repair is a prediction model,
    not a simulator result or validated cost law. Calibrate and test it against
    actual held-out continuation branches before making a benefit claim.
    """
    def __init__(self, *, max_observation_age=1.0, minimum_dwell_seconds=0.1,
                 relative_improvement=0.05, max_switches=8):
        for value in (max_observation_age, minimum_dwell_seconds, relative_improvement):
            _finite(value, "policy parameter")
        _integer(max_switches, "max_switches")
        self.max_age, self.dwell, self.improvement = max_observation_age, minimum_dwell_seconds, relative_improvement
        self.max_switches, self.current, self.last_switch, self.switches = max_switches, None, None, 0
        self._pending = None

    def decide(self, state, capabilities, estimate, observation, *, now):
        _finite(now, "now")
        if self._pending is not None:
            raise RuntimeError("A prior proposed decision requires executor acknowledgement or rejection")
        if self.last_switch is not None and now < self.last_switch:
            raise ValueError("Policy clock cannot move backwards")
        if estimate.generated_tokens != state.generated_tokens:
            raise ValueError("Remaining estimate belongs to a different progress point")
        _finite(estimate.mean_remaining_tokens, "mean_remaining_tokens", 1)
        legal = legal_continuations(state, capabilities)
        if not legal:
            raise RuntimeError("No implemented safe continuation is available")
        stale = observation is None or observation.available_at > now or now - observation.sampled_at > self.max_age
        costs = {}
        if not stale:
            horizon = estimate.mean_remaining_tokens
            local = horizon * observation.local_step_seconds
            fraction = (state.layers - len(state.committed_layers)) / state.layers
            remote_step = observation.local_step_seconds + fraction * max(
                0.0, observation.all_remote_step_seconds - observation.local_step_seconds)
            remote = horizon * remote_step
            rate = observation.effective_prefix_payload_bytes_per_second
            transfer = state.remaining_payload_bytes / rate if rate > 0 else math.inf
            costs.update(continue_local=local, pause_migration_and_remote=remote,
                         stop_migration_and_remote=remote, complete_fetch_then_local=transfer + local)
            # Integrate a declared per-step linear-decay approximation; cap loop work.
            elapsed, remaining = 0.0, horizon
            if horizon > 1_000_000:
                raise ValueError("Estimated horizon exceeds this bounded predictor")
            while remaining > 0:
                weight = min(1.0, remaining)
                decay = 1.0 if not math.isfinite(transfer) else max(0.0, 1.0 - elapsed / transfer) if transfer else 0.0
                elapsed += weight * (observation.local_step_seconds + decay * (remote_step - observation.local_step_seconds))
                remaining -= weight
            costs["continue_repair"] = elapsed
            if observation.full_history_recompute_seconds is not None:
                costs["recompute_full_history_then_local"] = observation.full_history_recompute_seconds + local
            if observation.full_prefix_recompute_seconds is not None:
                costs["recompute_full_prefix_keep_delta"] = observation.full_prefix_recompute_seconds + local
            costs = {action: cost + (observation.switching_seconds if self.current not in (None, action) else 0)
                     for action, cost in costs.items() if action in legal and math.isfinite(cost)}
        if stale or not costs:
            candidate = next(action for action in ("continue_local", "pause_migration_and_remote",
                                                   "stop_migration_and_remote", "continue_repair",
                                                   "complete_fetch_then_local", "recompute_full_prefix_keep_delta",
                                                   "recompute_full_history_then_local") if action in legal)
            reason = "stale_or_unavailable_observation_fallback"
            if (self.current in legal and (self.switches >= self.max_switches
                    or now - self.last_switch < self.dwell)):
                candidate, reason = self.current, "bounded_switch_retains_legal_action_during_staleness"
        else:
            candidate = min(costs, key=lambda action: (costs[action], action))
            reason = "prospective_model_minimum"
            if self.current in costs and candidate != self.current:
                benefit = costs[self.current] - costs[candidate]
                if (self.switches >= self.max_switches or now - self.last_switch < self.dwell
                        or benefit <= self.improvement * costs[self.current]):
                    candidate, reason = self.current, "bounded_switch_hysteresis"
        result = {"action": candidate, "legal_actions": list(legal), "reason": reason,
                "predicted_remaining_seconds_not_measurements": costs,
                "sunk_elapsed_seconds_excluded_from_costs": state.sunk_elapsed_seconds,
                "remaining_estimate": asdict(estimate), "switch_count": self.switches,
                "requires_executor_application_ack": True}
        self._pending = identity(result)
        return result

    def acknowledge(self, decision, *, now, applied=True):
        """Advance switching state only after a real executor action ACK."""
        _finite(now, "now")
        if type(applied) is not bool or self._pending is None or identity(decision) != self._pending:
            raise ValueError("Decision acknowledgement does not match the pending proposal")
        if self.last_switch is not None and now < self.last_switch:
            raise ValueError("Policy clock cannot move backwards")
        self._pending = None
        if applied and self.current != decision["action"]:
            if self.current is not None:
                self.switches += 1
            self.current, self.last_switch = decision["action"], now


class MigrationControl:
    """One sender thread; commands may arrive from a separate control/RPC thread.

    begin_frame waits while paused without holding the condition lock. Once it
    succeeds, the executor finishes that entire frame before finish_frame, even
    if pause/stop arrives. Stop is irreversible and terminal metadata is emitted
    only after all begun frames finish. Receiver drain ACK is a separate step.
    Transport errors call fail; they cannot be disguised as graceful stop.
    """
    def __init__(self, request_id, layers, *, initially_paused=False):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id is required")
        _integer(layers, "layers", 1)
        self.request_id, self.layers = request_id, layers
        self._condition = threading.Condition()
        self._mode = "paused" if initially_paused else "running"
        self._next_layer, self._inflight, self._payload_bytes = 0, None, 0
        self._terminal_sent, self._terminal_ack, self._error = False, False, None

    def pause(self):
        with self._condition:
            if self._mode in ("stopping", "stopped", "complete", "failed"):
                raise RuntimeError("Cannot pause a terminal migration")
            self._mode = "paused"
            self._condition.notify_all()

    def resume(self):
        with self._condition:
            if self._mode in ("stopping", "stopped", "complete", "failed"):
                raise RuntimeError("Stop is irreversible; start a new transfer explicitly if needed")
            self._mode = "running"
            self._condition.notify_all()

    def stop(self):
        with self._condition:
            if self._mode not in ("complete", "failed", "stopped"):
                self._mode = "stopping"
            self._condition.notify_all()

    def begin_frame(self, layer, *, timeout=None):
        _integer(layer, "layer")
        with self._condition:
            if self._inflight is not None or layer != self._next_layer or layer >= self.layers:
                raise ValueError("Only one sender and ascending complete layer frames are supported")
            if not self._condition.wait_for(lambda: self._mode != "paused", timeout):
                raise TimeoutError("Migration remains paused")
            if self._inflight is not None or layer != self._next_layer:
                raise ValueError("A second sender attempted to start a frame while paused")
            if self._mode == "failed":
                raise RuntimeError(self._error)
            if self._mode in ("stopping", "stopped", "complete"):
                return False
            self._inflight = layer
            return True

    def finish_frame(self, layer, payload_bytes):
        _integer(layer, "layer")
        _integer(payload_bytes, "payload_bytes", 1)
        with self._condition:
            if self._inflight != layer:
                raise ValueError("Finished frame does not match the active frame")
            self._inflight = None
            self._next_layer += 1
            self._payload_bytes += payload_bytes
            if self._next_layer == self.layers and self._mode != "failed":
                self._mode = "complete"
            self._condition.notify_all()

    def wait_quiescent(self, timeout=None):
        with self._condition:
            return self._condition.wait_for(lambda: self._inflight is None and self._mode != "running", timeout)

    def terminal_message(self):
        with self._condition:
            if self._inflight is not None or self._mode not in ("stopping", "stopped", "complete") or self._terminal_sent:
                raise RuntimeError("Terminal must follow complete frames exactly once")
            self._terminal_sent = True
            if self._mode != "complete":
                self._mode = "stopped"
            return {"kind": "migration_complete" if self._mode == "complete" else "migration_cancelled",
                    "request_id": self.request_id, "complete_layers": self._next_layer,
                    "payload_bytes": self._payload_bytes}

    def acknowledge_drained(self, message):
        with self._condition:
            expected = {"kind": "migration_drained", "request_id": self.request_id,
                        "complete_layers": self._next_layer, "payload_bytes": self._payload_bytes}
            if not self._terminal_sent or self._terminal_ack or message != expected:
                raise ValueError("Receiver terminal/drain acknowledgement mismatch")
            self._terminal_ack = True

    def fail(self, error):
        with self._condition:
            self._mode, self._error = "failed", str(error)
            self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            return {"request_id": self.request_id, "state": self._mode, "complete_layers": self._next_layer,
                    "in_flight_layer": self._inflight, "payload_bytes": self._payload_bytes,
                    "terminal_sent": self._terminal_sent, "receiver_drained_ack": self._terminal_ack,
                    "error": self._error, "thread_and_socket_cleanup_checked_by_executor": False}
