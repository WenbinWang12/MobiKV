"""Replayable application-byte capacity and causally delayed observations for E3.

This controls real PacedSocket writes when installed as its shared limiter. It
does not emulate physical NIC/RDMA bandwidth, propagation RTT, loss, or GPU load.
The complete rate plan is actuator-only; policies receive released observations,
not future trace entries. A RatePlan uses elapsed monotonic seconds from an
executor-established epoch. Both hosts record their epochs and synchronization
uncertainty; synchronized clocks must never be inferred from equal plan hashes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import heapq
import json
import math
import threading
import time

from e1_transport import RateLimiter, Reservation


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class RateStep:
    at_seconds: float
    rate_bits_per_second: float

    def __post_init__(self):
        _nonnegative(self.at_seconds, "at_seconds")
        _nonnegative(self.rate_bits_per_second, "rate_bits_per_second")


@dataclass(frozen=True, init=False)
class RatePlan:
    """Piecewise constant cap; zero means unpaced, NOT link disconnection."""
    steps: tuple[RateStep, ...]
    plan_id: str
    seed: int
    sha256: str

    def __init__(self, steps, *, plan_id, seed):
        object.__setattr__(self, "steps", tuple(step if isinstance(step, RateStep) else RateStep(**step) for step in steps))
        if (not self.steps or self.steps[0].at_seconds != 0 or
                any(left.at_seconds >= right.at_seconds for left, right in zip(self.steps, self.steps[1:]))):
            raise ValueError("Rate steps start at zero and have strictly increasing change times")
        if not isinstance(plan_id, str) or not plan_id or type(seed) is not int:
            raise ValueError("Plan identity and fixed integer seed are required")
        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "sha256", hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                           separators=(",", ":"), allow_nan=False).encode()).hexdigest())

    def to_dict(self):
        return {"plan_id": self.plan_id, "seed": self.seed, "steps": [asdict(step) for step in self.steps],
                "scope": "application_socket_bytes_per_direction", "zero_rate": "unpaced_not_disconnected"}

    def rate_at(self, elapsed_seconds):
        _nonnegative(elapsed_seconds, "elapsed_seconds")
        return self.steps[self._index(elapsed_seconds)].rate_bits_per_second

    def _index(self, seconds):
        index = 0
        while index + 1 < len(self.steps) and self.steps[index + 1].at_seconds <= seconds:
            index += 1
        return index

    def reserve_until(self, elapsed_start, application_bytes):
        """Integrate one reservation across changes; do not reset prior budget."""
        _nonnegative(elapsed_start, "elapsed_start")
        if type(application_bytes) is not int or application_bytes <= 0:
            raise ValueError("application_bytes must be a positive integer")
        remaining_bits, cursor = application_bytes * 8, float(elapsed_start)
        index = self._index(cursor)
        while True:
            rate = self.steps[index].rate_bits_per_second
            if rate == 0:
                return cursor
            boundary = self.steps[index + 1].at_seconds if index + 1 < len(self.steps) else math.inf
            available_bits = (boundary - cursor) * rate
            if remaining_bits <= available_bits:
                return cursor + remaining_bits / rate
            remaining_bits -= available_bits
            cursor, index = boundary, index + 1


class VariableRateLimiter(RateLimiter):
    """Drop-in RateLimiter for real shared RPC/bulk PacedSockets.

    rate_bits_per_second remains the initial rate for stable endpoint handshake
    compatibility. snapshot reports plan and instantaneous planned rate separately.
    The caller must validate identical plan hashes at both peers; the old E1
    handshake alone only checks initial rate. Reservations are nonrefundable;
    rate changes integrate into future chunks without creating a new budget.
    Existing wait/cancellation/statistics APIs are inherited unchanged.
    """
    def __init__(self, plan, *, epoch_monotonic, chunk_bytes=64 * 1024, clock=time.monotonic):
        if not isinstance(plan, RatePlan):
            raise TypeError("plan must be RatePlan")
        _nonnegative(epoch_monotonic, "epoch_monotonic")
        self.plan, self.epoch_monotonic = plan, float(epoch_monotonic)
        super().__init__(plan.steps[0].rate_bits_per_second, chunk_bytes, clock=clock)

    def reserve(self, wire_bytes, connection_id):
        if type(wire_bytes) is not int or wire_bytes <= 0:
            raise ValueError("wire_bytes must be a positive integer")
        if not isinstance(connection_id, str) or not connection_id:
            raise ValueError("connection_id must be a nonempty string")
        with self._lock:
            now = float(self._clock())
            if now < self.epoch_monotonic:
                raise ValueError("Pacing cannot start before the declared experiment epoch")
            start = max(now, self._next_ready)
            ready = self.epoch_monotonic + self.plan.reserve_until(start - self.epoch_monotonic, wire_bytes)
            self._next_ready = ready
            # _register/_add take this same lock; update while locked directly.
            if connection_id not in self._connections:
                from e1_transport import _empty_stats
                self._connections[connection_id] = _empty_stats()
            for stats in (self._connections[connection_id], self._total):
                stats["reserved_bytes"] += wire_bytes
                stats["reservations"] += 1
            return Reservation(connection_id, wire_bytes, start, ready, self._owner)

    def snapshot(self):
        result = super().snapshot()
        elapsed = max(0.0, self._clock() - self.epoch_monotonic)
        result.update(rate_plan=self.plan.to_dict(), rate_plan_sha256=self.plan.sha256,
                      epoch_monotonic_seconds=self.epoch_monotonic,
                      instantaneous_planned_rate_bits_per_second=self.plan.rate_at(elapsed),
                      rate_bits_per_second_field_meaning="initial_rate_handshake_compatibility",
                      physical_link_measurement=False)
        return result


class DelayedObservations:
    """Causal queue; future/unreleased measurements never reach policy callers.

    Measurement production and delay values are executor inputs, not synthetic
    performance results. Ties preserve insertion order; released records include
    measured_at and available_at for age/fallback checks. Payloads are copied.
    """
    def __init__(self):
        self._heap, self._sequence, self._last_now = [], 0, -math.inf
        self._lock = threading.Lock()

    def enqueue(self, payload, *, measured_at, delay_seconds):
        _nonnegative(measured_at, "measured_at")
        _nonnegative(delay_seconds, "delay_seconds")
        json.dumps(payload, allow_nan=False)
        with self._lock:
            sequence = self._sequence
            self._sequence += 1
            record = {"sequence": sequence, "measured_at": measured_at,
                      "available_at": measured_at + delay_seconds, "payload": copy.deepcopy(payload)}
            heapq.heappush(self._heap, (record["available_at"], sequence, record))
            return sequence

    def release(self, *, now):
        _nonnegative(now, "now")
        with self._lock:
            if now < self._last_now:
                raise ValueError("Observation delivery clock cannot move backwards")
            self._last_now = now
            records = []
            while self._heap and self._heap[0][0] <= now:
                records.append(heapq.heappop(self._heap)[2])
            return records

    @property
    def pending_count(self):
        with self._lock:
            return len(self._heap)
