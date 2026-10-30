"""E4 application-byte schedulers; CPU components, not GPU-stage evidence.

All wrapped sockets in one direction share one RateLimiter and one active
nonpreemptive chunk. A grant covers pacing AND the actual socket write.
Reverse-direction traffic requires its own coordinator. No NIC-byte, deadline,
starvation-freedom or performance claim follows from these counters.
"""
from collections import deque
from contextlib import contextmanager
import copy
import math
import socket
import threading
import time

try:
    from e1_transport import RateLimiter, PacedSocket, SendCancelledError
except ModuleNotFoundError as error:
    if error.name != "e1_transport":
        raise
    from restate_direct.e1_full.runtime.e1_transport import RateLimiter, PacedSocket, SendCancelledError

Q = 16384
POLICIES = ("coarse_fifo", "fine_fifo", "fixed_weighted_3to1", "strict_rpc_first", "age_bounded_rpc_first")
SOURCE_DEPENDENCIES = ("e1_transport.py",)


SCALABLE = ("fine_fifo", "strict_rpc_first", "age_bounded_rpc_first")


def policy_config(policy, rate_bps, chunk_bytes=None):
    """chunk_bytes=None (or the default) reproduces the E4 configuration exactly.

    Experiment 3 scales the chunk with the cap so each chunk keeps the E4 service
    time (16 KiB at 100 Mbps = 1.31 ms); only RPC-first/FIFO policies may scale.
    """
    if policy not in POLICIES:
        raise ValueError("Unknown E4 scheduler")
    if type(rate_bps) not in (int, float) or not math.isfinite(rate_bps) or rate_bps <= 0:
        raise ValueError("A finite positive shared application cap is required")
    default = 1048576 if policy == "coarse_fifo" else Q
    if chunk_bytes is None or chunk_bytes == default:
        q = default
    elif policy in SCALABLE and type(chunk_bytes) is int and Q <= chunk_bytes <= 64 * 1048576:
        q = chunk_bytes
    else:
        raise ValueError("Scaled chunks only for fine_fifo/strict/age_bounded, 16 KiB..64 MiB")
    return {"policy": policy, "rate_bps": rate_bps,
            "chunk_bytes": q,
            "class_weights": {"rpc": 3, "bulk": 1} if policy == "fixed_weighted_3to1" else None,
            "bulk_age_threshold_s": max(.005, 4 * q * 8 / rate_bps) if policy == "age_bounded_rpc_first" else None}


class E4BandwidthCoordinator:
    """Finite-chunk scheduling with actual-byte DRR and interruptible shutdown.

    trace_grants is for excluded CPU checks/profiles. Default main accounting
    keeps bounded per-stream counters and 100ms bins, not full chunk histories.
    One sender per socket is enforced by the accepted PacedSocket adapter.
    """
    def __init__(self, rate_bps, policy="fine_fifo", *, chunk_bytes=None, clock=time.monotonic, trace_grants=False):
        self.config = policy_config(policy, rate_bps, chunk_bytes)
        self.policy, self.clock = policy, clock
        self.priority = policy
        self.limiter = RateLimiter(rate_bps, self.config["chunk_bytes"], clock=clock)
        self._condition = threading.Condition()
        self._streams, self._pending = {}, []
        self._active = self._selected = None
        self._ticket, self._closed = 0, False
        self._deficit = {"rpc": 0, "bulk": 0}
        self._class_index, self._visit_open = 0, False
        self._epoch = clock()
        self._bins, self._grants = {}, []
        self._trace = bool(trace_grants)
        self._recent = deque(maxlen=256)
        self._total_grants = self._aging_overrides = 0
        self._coeligible_grants = self._successful_coeligible_grants = 0

    def wrap(self, sock, stream_id, kind):
        if kind not in ("rpc", "bulk") or not isinstance(stream_id, str) or not stream_id:
            raise ValueError("Need a nonempty stream ID and rpc/bulk class")
        timeout = sock.gettimeout()
        if timeout is None or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Finite positive socket timeout required")
        with self._condition:
            if self._closed or stream_id in self._streams:
                raise ValueError("Coordinator closed or stream ID reused")
            wrapped = _ScheduledSocket(sock, self, stream_id)
            self._streams[stream_id] = {"kind": kind, "paused": False, "paused_since": None,
                "cancelled": False, "grants": 0, "queue_wait_seconds": 0.,
                "maximum_eligible_queue_age_s": 0., "sent_bytes_in_grants": 0,
                "socket": wrapped}
            return wrapped

    def _eligible(self):
        return [p for p in self._pending if not self._streams[p["stream_id"]]["paused"]
                and not self._streams[p["stream_id"]]["cancelled"]]

    def _pick(self, now):
        """Caller holds condition; DRR state advances once per selected grant."""
        eligible = self._eligible()
        if not eligible:
            return None
        # A previously selected ticket can be paused before its thread claims
        # the grant. Do not retain an aging-trigger label from that old choice.
        for pending in eligible:
            pending["aging_override"] = False
        by_class = {kind: [p for p in eligible if self._streams[p["stream_id"]]["kind"] == kind]
                    for kind in ("rpc", "bulk")}
        if self.policy in ("coarse_fifo", "fine_fifo"):
            return min(eligible, key=lambda p: p["ticket"])
        if self.policy in ("strict_rpc_first", "age_bounded_rpc_first"):
            if self.policy == "age_bounded_rpc_first" and by_class["bulk"]:
                oldest = min(by_class["bulk"], key=lambda p: (p["eligible_since"], p["ticket"]))
                if now - oldest["eligible_since"] >= self.config["bulk_age_threshold_s"]:
                    oldest["aging_override"] = bool(by_class["rpc"])
                    return oldest
            return min(by_class["rpc"] or by_class["bulk"], key=lambda p: p["ticket"])
        # Deficits are byte quantities. A partial/failed socket write refunds
        # unsent bytes to this scheduler, never to the no-credit rate limiter.
        while True:
            kind = ("rpc", "bulk")[self._class_index]
            if not by_class[kind]:
                self._deficit[kind] = 0
                self._class_index = 1 - self._class_index
                self._visit_open = False
                continue
            if not self._visit_open:
                self._deficit[kind] += self.config["class_weights"][kind] * Q
                self._visit_open = True
            first = min(by_class[kind], key=lambda p: p["ticket"])
            if first["bytes"] <= self._deficit[kind]:
                return first
            self._class_index = 1 - self._class_index
            self._visit_open = False

    @contextmanager
    def _turn(self, stream_id, byte_count):
        if type(byte_count) is not int or not 0 < byte_count <= self.config["chunk_bytes"]:
            raise ValueError("A grant must be one finite chunk")
        started = self.clock()
        lease = {"sent_bytes": 0}
        granted, pending = False, None
        try:
            with self._condition:
                state = self._streams[stream_id]
                pending = {"ticket": self._ticket, "stream_id": stream_id, "bytes": byte_count,
                           "enqueued_s": started, "eligible_since": started, "aging_override": False}
                self._ticket += 1
                self._pending.append(pending)
                while True:
                    if self._closed or state["cancelled"]:
                        raise SendCancelledError("E4 stream cancelled while waiting")
                    if self._active is None:
                        if self._selected is None:
                            self._selected = self._pick(self.clock())
                            self._condition.notify_all()
                        if self._selected is pending:
                            pending["coeligible_at_grant"] = {
                                self._streams[item["stream_id"]]["kind"] for item in self._eligible()
                            } == {"rpc", "bulk"}
                            self._pending.remove(pending)
                            self._active, self._selected = pending, None
                            granted = True
                            grant_at = self.clock()
                            pending["granted_s"] = grant_at
                            state["grants"] += 1
                            state["queue_wait_seconds"] += max(0., grant_at - started)
                            state["maximum_eligible_queue_age_s"] = max(state["maximum_eligible_queue_age_s"], grant_at - pending["eligible_since"])
                            if self.policy == "fixed_weighted_3to1":
                                self._deficit[state["kind"]] -= byte_count
                            break
                    self._condition.wait(.05)
            yield lease
        finally:
            with self._condition:
                if granted:
                    if self._active is not pending:
                        raise RuntimeError("Active chunk ownership corrupted")
                    sent = lease["sent_bytes"]
                    state["sent_bytes_in_grants"] += sent
                    if self.policy == "fixed_weighted_3to1":
                        self._deficit[state["kind"]] += byte_count - sent
                    ended = self.clock()
                    event = dict(pending, kind=state["kind"], finished_s=ended, actual_sent_bytes=sent)
                    self._recent.append(event)
                    if self._trace:
                        self._grants.append(event)
                    self._total_grants += 1
                    self._aging_overrides += int(pending["aging_override"])
                    self._coeligible_grants += int(pending["coeligible_at_grant"])
                    self._successful_coeligible_grants += int(pending["coeligible_at_grant"] and sent > 0)
                    self._active = None
                elif pending is not None:
                    if pending in self._pending:
                        self._pending.remove(pending)
                    if self._selected is pending:
                        self._selected = None
                    state["queue_wait_seconds"] += max(0., self.clock() - started)
                self._condition.notify_all()

    def _sent(self, stream_id, count):
        # Called immediately after each successful OS write, not at reservation
        # or whole-frame completion. Bins are local-clock application bytes.
        with self._condition:
            elapsed = max(0., self.clock() - self._epoch)
            bucket = int(elapsed / .1)
            item = self._bins.setdefault(bucket, {"start_offset_s": bucket / 10., "rpc_sent_bytes": 0,
                "bulk_sent_bytes": 0, "maximum_pending_rpc": 0, "maximum_pending_bulk": 0})
            kind = self._streams[stream_id]["kind"]
            item[kind + "_sent_bytes"] += count
            eligible = self._eligible()
            for candidate in ("rpc", "bulk"):
                item["maximum_pending_" + candidate] = max(item["maximum_pending_" + candidate],
                    sum(self._streams[p["stream_id"]]["kind"] == candidate for p in eligible))

    def pause(self, stream_id):
        with self._condition:
            state = self._streams[stream_id]
            if state["kind"] != "bulk" or state["cancelled"]:
                raise ValueError("Only live bulk sends can pause")
            if not state["paused"]:
                state.update(paused=True, paused_since=self.clock())
            if self._selected is not None and self._selected["stream_id"] == stream_id:
                self._selected = None
            self._condition.notify_all()

    def resume(self, stream_id):
        with self._condition:
            state = self._streams[stream_id]
            if state["kind"] != "bulk" or state["cancelled"]:
                raise ValueError("Only live bulk sends can resume")
            if state["paused"]:
                now = self.clock()
                for pending in self._pending:
                    if pending["stream_id"] == stream_id:
                        pending["eligible_since"] += now - max(pending["eligible_since"], state["paused_since"])
                state.update(paused=False, paused_since=None)
            self._condition.notify_all()

    def cancel(self, stream_id):
        with self._condition:
            state = self._streams[stream_id]
            state["cancelled"] = True
            if self._selected is not None and self._selected["stream_id"] == stream_id:
                self._selected = None
            self._condition.notify_all()
            wrapped = state["socket"]
        wrapped._cancel_from_coordinator()

    def snapshot(self):
        with self._condition:
            streams = {name: {k: v for k, v in item.items() if k != "socket"}
                       for name, item in self._streams.items()}
            result = {"configuration": copy.deepcopy(self.config), "closed": self._closed,
                "active": copy.deepcopy(self._active), "pending": copy.deepcopy(self._pending),
                "streams": streams, "class_deficits_bytes": dict(self._deficit),
                "total_grants": self._total_grants, "aging_overrides": self._aging_overrides,
                "coeligible_grant_count": self._coeligible_grants,
                "successful_coeligible_grant_count": self._successful_coeligible_grants,
                "coeligible_scope": "Both actual eligible classes observed under the same lock immediately before this grant; not inferred from separate bin maxima",
                "bins_100ms": copy.deepcopy([self._bins[k] for k in sorted(self._bins)]),
                "recent_grants": copy.deepcopy(list(self._recent)), "recent_grants_truncated": self._total_grants > 256,
                "full_grant_trace_enabled": self._trace,
                "grant_trace": copy.deepcopy(self._grants) if self._trace else None}
        limiter = self.limiter.snapshot()
        for name, item in streams.items():
            item["transport"] = limiter["connections"][name]
            item["application_sent_bytes"] = item["transport"]["sent_bytes"]
            item["application_received_bytes"] = item["transport"]["received_bytes"]
        return dict(result, limiter=limiter, priority=self.policy, physical_wire_bytes=None, wire_measurement_available=False,
                    scope="One directional application socket budget; no clock synchronization or GPU evidence")

    def close(self):
        with self._condition:
            self._closed = True
            sockets = [item["socket"] for item in self._streams.values()]
            self._condition.notify_all()
        for wrapped in sockets:
            wrapped.close()


class _ScheduledSocket(PacedSocket):
    def __init__(self, sock, coordinator, stream_id):
        super().__init__(sock, coordinator.limiter, stream_id)
        self.coordinator = coordinator

    def settimeout(self, value):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Finite positive I/O timeout required")
        self._socket.settimeout(value)

    def setblocking(self, value):
        raise ValueError("Use a finite positive settimeout")

    def _write_granted(self, data, flags, lease):
        count = self._write(data, flags)
        lease["sent_bytes"] += count
        self.coordinator._sent(self.connection_id, count)
        return count

    def send(self, data, flags=0):
        with self._single_sender():
            chunk = memoryview(data).cast("B")[:self.limiter.chunk_bytes]
            if not chunk:
                return 0
            with self.coordinator._turn(self.connection_id, len(chunk)) as lease:
                self._pace(len(chunk))
                return self._write_granted(chunk, flags, lease)

    def sendall(self, data, flags=0):
        with self._single_sender():
            view = memoryview(data).cast("B")
            for offset in range(0, len(view), self.limiter.chunk_bytes):
                chunk = view[offset:offset + self.limiter.chunk_bytes]
                with self.coordinator._turn(self.connection_id, len(chunk)) as lease:
                    self._pace(len(chunk))
                    sent = 0
                    while sent < len(chunk):
                        sent += self._write_granted(chunk[sent:], flags, lease)

    def _cancel_from_coordinator(self):
        super().cancel_send()

    def cancel_send(self):
        self.coordinator.cancel(self.connection_id)

    def shutdown(self, how):
        if how in (socket.SHUT_WR, socket.SHUT_RDWR):
            self.coordinator.cancel(self.connection_id)
        super().shutdown(how)

    def close(self):
        self.coordinator.cancel(self.connection_id)
        super().close()
