"""CPU transport/control components for later E4/E6 GPU integration.

One coordinator owns one outgoing application-byte budget for all its sockets.
RPC is prioritized only at finite chunk boundaries, not inside a socket write.
A blocked OS write uses the required per-operation timeout; this is no whole-
chunk or RPC latency deadline, particularly under backpressure or low pacing.
Pause completes the current chunk; cancellation destroys that connection's send
side, including any partial frame. Never resume/reuse a cancelled frame stream.
These counters do not measure NIC bytes, delivery, GPU overlap, or throughput.

ReplicaDirectory stores complete, caller-validated replica identities. It does
not transfer tensors or infer readiness, merge shards, or silently fail over an
already bound request. Retiring metadata is not evidence of GPU memory release.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import re
import socket
import threading
import time

from e1_transport import PacedSocket, RateLimiter, SendCancelledError


class SharedBandwidthCoordinator:
    """RPC-first, FIFO-within-class serialization shared by outgoing streams.

    wrap() returns a socket-compatible object usable by e1_wire. Each connection
    has one sender, and all RPC/bulk connections in one direction share this
    coordinator. The peer needs its own coordinator for the reverse direction.
    Strict priority can starve bulk under sustained RPC load; no fairness or
    speedup guarantee is implied. Chunk grants remain held through actual send.
    """

    def __init__(self, rate_bits_per_second=None, chunk_bytes=16 * 1024, *, limiter=None, priority="rpc_first"):
        if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= 1024 * 1024:
            raise ValueError("chunk_bytes must be in [1, 1048576]")
        if priority not in ("rpc_first", "fifo"):
            raise ValueError("priority must be rpc_first or fifo")
        if limiter is not None:
            if not isinstance(limiter, RateLimiter) or rate_bits_per_second is not None:
                raise ValueError("Supply either a compatible RateLimiter or a rate, not both")
            if type(limiter.chunk_bytes) is not int or not 1 <= limiter.chunk_bytes <= 1024 * 1024:
                raise ValueError("Injected limiter needs finite chunks no larger than 1 MiB")
            self.limiter = limiter
        else:
            if rate_bits_per_second is None:
                raise ValueError("rate_bits_per_second or limiter is required")
            self.limiter = RateLimiter(rate_bits_per_second, chunk_bytes)
        self.priority = priority
        self._condition = threading.Condition()
        self._streams = {}
        self._pending = []
        self._active = None
        self._next_ticket = 0
        self._closed = False
        self._recent_grants = deque(maxlen=256)

    def wrap(self, sock, stream_id, kind):
        if kind not in ("rpc", "bulk") or not isinstance(stream_id, str) or not stream_id:
            raise ValueError("Need a unique stream_id and kind rpc/bulk")
        timeout = sock.gettimeout()
        if timeout is None or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Underlying socket needs a finite positive I/O timeout")
        with self._condition:
            if self._closed or stream_id in self._streams:
                raise ValueError("Coordinator closed or stream_id already used")
            wrapped = _PrioritySocket(sock, self, stream_id)
            self._streams[stream_id] = {"kind": kind, "paused": False, "cancelled": False,
                                        "grants": 0, "queue_wait_seconds": 0.0,
                                        "queue_waits": 0, "socket": wrapped}
            return wrapped

    def pause(self, stream_id):
        """Pause a bulk stream after its current chunk; does not wait for drain."""
        with self._condition:
            state = self._streams[stream_id]
            if state["kind"] != "bulk" or state["cancelled"]:
                raise ValueError("Only a live bulk stream can be paused")
            state["paused"] = True
            self._condition.notify_all()

    def resume(self, stream_id):
        with self._condition:
            state = self._streams[stream_id]
            if state["kind"] != "bulk" or state["cancelled"]:
                raise ValueError("Only a live bulk stream can be resumed")
            state["paused"] = False
            self._condition.notify_all()

    def cancel(self, stream_id):
        """Permanently cancel sends and unblock a pending OS send via shutdown."""
        with self._condition:
            state = self._streams[stream_id]
            state["cancelled"] = True
            wrapped = state["socket"]
            self._condition.notify_all()
        wrapped._cancel_from_coordinator()

    @contextmanager
    def _turn(self, stream_id):
        ticket = None
        granted = False
        started = time.monotonic()
        queue_recorded = False
        waited = False
        try:
            with self._condition:
                ticket = self._next_ticket
                self._next_ticket += 1
                self._pending.append((ticket, stream_id))
                while True:
                    state = self._streams[stream_id]
                    if self._closed or state["cancelled"]:
                        raise SendCancelledError("Shared stream was cancelled")
                    eligible = [(0 if self.priority == "fifo" or self._streams[s]["kind"] == "rpc" else 1, t, s)
                                for t, s in self._pending
                                if not self._streams[s]["paused"] and not self._streams[s]["cancelled"]]
                    winner = min(eligible) if eligible else None
                    if self._active is None and winner is not None and winner[1] == ticket:
                        self._pending.remove((ticket, stream_id))
                        self._active = (ticket, stream_id)
                        state["grants"] += 1
                        state["queue_wait_seconds"] += max(0.0, time.monotonic() - started)
                        state["queue_waits"] += int(waited)
                        queue_recorded = True
                        self._recent_grants.append({"ticket": ticket, "stream_id": stream_id,
                                                    "kind": state["kind"]})
                        granted = True
                        break
                    waited = True
                    self._condition.wait()
            yield
        finally:
            with self._condition:
                if not queue_recorded:
                    self._streams[stream_id]["queue_wait_seconds"] += max(0.0, time.monotonic() - started)
                    self._streams[stream_id]["queue_waits"] += int(waited)
                if granted:
                    if self._active != (ticket, stream_id):
                        raise RuntimeError("Corrupt shared-bandwidth grant ownership")
                    self._active = None
                elif ticket is not None and (ticket, stream_id) in self._pending:
                    self._pending.remove((ticket, stream_id))
                self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            state = {name: {key: value for key, value in item.items() if key != "socket"}
                     for name, item in self._streams.items()}
            scheduling = {"closed": self._closed, "active": self._active,
                          "pending": list(self._pending), "streams": state,
                          "recent_grants": list(self._recent_grants)}
        limiter = self.limiter.snapshot()
        for name, item in state.items():
            item["transport"] = limiter["connections"][name]
            item["application_sent_bytes"] = item["transport"]["sent_bytes"]
            item["application_received_bytes"] = item["transport"]["received_bytes"]
        return {**scheduling, "limiter": limiter, "priority": self.priority,
                "physical_wire_bytes": None, "wire_measurement_available": False,
                "scope": "Outgoing application socket bytes; RPC priority at chunk boundaries only"}

    def close(self):
        with self._condition:
            self._closed = True
            streams = [item["socket"] for item in self._streams.values()]
            self._condition.notify_all()
        for stream in streams:
            stream.close()


class _PrioritySocket(PacedSocket):
    def __init__(self, sock, coordinator, stream_id):
        super().__init__(sock, coordinator.limiter, stream_id)
        self.coordinator = coordinator

    def settimeout(self, value):
        if value is None or not math.isfinite(value) or value <= 0:
            raise ValueError("Shared streams require finite positive I/O timeouts")
        self._socket.settimeout(value)

    def setblocking(self, value):
        raise ValueError("Use settimeout with a finite positive timeout")

    def send(self, data, flags=0):
        with self._single_sender():
            chunk = memoryview(data).cast("B")[:self.limiter.chunk_bytes]
            if not chunk:
                return 0
            with self.coordinator._turn(self.connection_id):
                self._pace(len(chunk))
                return self._write(chunk, flags)

    def sendall(self, data, flags=0):
        with self._single_sender():
            view = memoryview(data).cast("B")
            for offset in range(0, len(view), self.limiter.chunk_bytes):
                chunk = view[offset:offset + self.limiter.chunk_bytes]
                with self.coordinator._turn(self.connection_id):
                    self._pace(len(chunk))
                    sent = 0
                    while sent < len(chunk):
                        sent += self._write(chunk[sent:], flags)

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


@dataclass(frozen=True)
class ReplicaBinding:
    prefix_id: str
    version: int
    holder: str
    replica_id: str
    cache_sha256: str


class ReplicaDirectory:
    """Thread-safe complete-replica directory, with explicit request ownership.

    publish() is called only after caller readiness/integrity validation. The
    caller supplies an incarnation replica_id so a restarted holder cannot take
    over an old binding. A new version affects new requests only. lookup() never
    silently reselects a holder. retire() refuses replicas with active requests.
    The caller must also drain migration or other non-request owners before
    retiring; this directory owns no GPU tensors and performs no deallocation.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._replicas = {}
        self._content = {}
        self._latest = {}
        self._requests = {}

    def publish(self, prefix_id, version, holder, cache_sha256, *, replica_id):
        if any(not isinstance(x, str) or not x for x in (prefix_id, holder, replica_id)):
            raise ValueError("Nonempty prefix, holder, and replica incarnation required")
        if type(version) is not int or version < 0 or not isinstance(cache_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", cache_sha256) is None:
            raise ValueError("Need a nonnegative integer version and lowercase SHA256")
        binding = ReplicaBinding(prefix_id, version, holder, replica_id, cache_sha256)
        key = (prefix_id, version, holder)
        with self._lock:
            content_key = (prefix_id, version)
            if content_key in self._content and self._content[content_key] != cache_sha256:
                raise ValueError("Replicas of one prefix/version must have identical complete content")
            if version < self._latest.get(prefix_id, version) and content_key not in self._content:
                raise ValueError("Cannot publish an unknown stale version")
            if key in self._replicas and self._replicas[key] != binding:
                raise ValueError("Retire the old replica incarnation before replacement")
            self._content[content_key] = cache_sha256
            self._latest[prefix_id] = max(version, self._latest.get(prefix_id, version))
            self._replicas[key] = binding
        return binding

    def bind(self, request_id, prefix_id, *, version=None, holder=None):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("Nonempty request_id required")
        if version is not None and (type(version) is not int or version < 0):
            raise ValueError("Explicit version must be a nonnegative integer")
        with self._lock:
            if request_id in self._requests:
                old = self._requests[request_id]
                if old.prefix_id != prefix_id or (version is not None and old.version != version) or (holder is not None and old.holder != holder):
                    raise ValueError("An active request cannot change its prefix/version/holder")
                return old
            selected_version = self._latest[prefix_id] if version is None else version
            candidates = [item for (prefix, v, h), item in self._replicas.items()
                          if prefix == prefix_id and v == selected_version and (holder is None or h == holder)]
            if not candidates:
                raise LookupError("No complete replica for the requested identity")
            binding = min(candidates, key=lambda item: item.holder)
            self._requests[request_id] = binding
            return binding

    def lookup(self, request_id):
        with self._lock:
            return self._requests[request_id]

    def release(self, request_id):
        with self._lock:
            return self._requests.pop(request_id)

    def retire(self, prefix_id, version, holder, *, replica_id):
        key = (prefix_id, version, holder)
        with self._lock:
            binding = self._replicas[key]
            if binding.replica_id != replica_id:
                raise ValueError("Replica incarnation mismatch")
            if binding in self._requests.values():
                raise RuntimeError("Cannot retire a replica with active request bindings")
            return self._replicas.pop(key)

    def snapshot(self):
        with self._lock:
            return {"replicas": [asdict(item) for _, item in sorted(self._replicas.items())],
                    "requests": {key: asdict(value) for key, value in sorted(self._requests.items())},
                    "latest_versions": dict(self._latest), "shard_merge_supported": False,
                    "scope": "Metadata/request ownership only; readiness and tensor release belong to caller"}
