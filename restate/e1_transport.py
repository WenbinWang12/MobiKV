"""Shared application-layer pacing for E1 TCP streams (standard library only).

Create ONE RateLimiter per process/outgoing direction and share it across every
RPC and migration PacedSocket.  The other host uses its own limiter at the same
rate: the link is full duplex, with one budget in EACH direction.  Rates are in
bits/second, not MiB/second.  A zero rate disables pacing.

Every byte supplied to send/sendall is charged, including frame headers and
payloads.  This is an application socket-byte cap, NOT NIC/RDMA shaping: TCP/IP
headers, retransmissions, kernel queues, and packet timing are outside its scope.
The first chunk waits for its serialization budget; there is no initial token
credit or idle-credit accumulation.  Writes are chunk-granular, and scheduling
delays/backpressure can change their actual timing.  Failed/cancelled reservation
budget is conservatively retained.  Stats distinguish reservations from bytes
accepted by successful socket.send calls; accepted bytes do not imply delivery.

Use one sender at a time per connection.  Reading may run concurrently.  Explicit
cancel_send()/close() wakes pacing waits and shuts down blocked writes.  A caller
cancel_event is polled during pacing and between writes; to interrupt a blocked
OS write immediately, also call cancel_send() or close().  Socket timeouts remain
unchanged and apply to each underlying I/O operation, not to pacing waits or the
whole emulated sendall call.  Use an external deadline/cancellation policy when
an end-to-end operation timeout is required.

Alternate output paths (makefile, sendfile, sendmsg, sendto, dup, detach) are
disabled: all framed writes must pass through send/sendall.  Direct raw-socket or
file-descriptor writes are outside this wrapper's contract.

Example::

    outgoing = RateLimiter(rate_bits_per_second=100_000_000)
    rpc = PacedSocket(rpc_socket, outgoing, "rpc")
    migration = PacedSocket(migration_socket, outgoing, "migration")

Run CPU-only checks with: python -B e1_transport.py --self-test
"""

from __future__ import annotations

import errno
import math
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterable


class SendCancelledError(OSError):
    """The caller cancelled sending or closed/shut down the write side."""

    def __init__(self, message: str = "paced socket sending cancelled") -> None:
        super().__init__(errno.ECANCELED, message)


@dataclass(frozen=True)
class Reservation:
    """A chunk's serialization slot in a limiter's shared outgoing budget."""

    connection_id: str
    wire_bytes: int
    start_at: float
    ready_at: float
    _owner: object = field(repr=False, compare=False)


def _empty_stats() -> dict:
    return {
        "reserved_bytes": 0,
        "reservations": 0,
        "sent_bytes": 0,
        "received_bytes": 0,
        "pacing_wait_seconds": 0.0,
        "pacing_waits": 0,
        "cancelled_waits": 0,
        "cancelled_send_calls": 0,
        "failed_send_calls": 0,
    }


class RateLimiter:
    """Thread-safe, shared, no-credit serializer of outgoing byte reservations.

    reserve() only schedules budget; wait() performs interruptible waiting with
    no limiter lock held.  Normally PacedSocket invokes both.  The optional clock
    is a monotonic-seconds callable, principally for deterministic budget tests.
    Connection IDs identify stats; use a unique ID for each socket lifetime.
    """

    _POLL_SECONDS = 0.05

    def __init__(
        self,
        rate_bits_per_second: float,
        chunk_bytes: int = 64 * 1024,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        rate = float(rate_bits_per_second)
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("rate_bits_per_second must be finite and nonnegative")
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be a positive integer")
        self.rate_bits_per_second = rate
        self.chunk_bytes = chunk_bytes
        self._clock = clock
        self._lock = threading.Lock()
        self._owner = object()
        self._next_ready = float(clock())
        self._total = _empty_stats()
        self._connections: dict[str, dict] = {}

    def _register(self, connection_id: str) -> None:
        if not isinstance(connection_id, str) or not connection_id:
            raise ValueError("connection_id must be a nonempty string")
        with self._lock:
            self._connections.setdefault(connection_id, _empty_stats())

    def _add(self, connection_id: str, **increments) -> None:
        with self._lock:
            stats = self._connections.setdefault(connection_id, _empty_stats())
            for name, amount in increments.items():
                stats[name] += amount
                self._total[name] += amount

    def reserve(self, wire_bytes: int, connection_id: str) -> Reservation:
        """Reserve one chunk; cancellation does not refund an allocated slot."""
        if isinstance(wire_bytes, bool) or not isinstance(wire_bytes, int) or wire_bytes <= 0:
            raise ValueError("wire_bytes must be a positive integer")
        if not isinstance(connection_id, str) or not connection_id:
            raise ValueError("connection_id must be a nonempty string")
        with self._lock:
            now = float(self._clock())
            if self.rate_bits_per_second:
                start = max(now, self._next_ready)
                ready = start + wire_bytes * 8 / self.rate_bits_per_second
                self._next_ready = ready
            else:
                start = ready = now
                self._next_ready = now
            stats = self._connections.setdefault(connection_id, _empty_stats())
            for target in (stats, self._total):
                target["reserved_bytes"] += wire_bytes
                target["reservations"] += 1
            return Reservation(connection_id, wire_bytes, start, ready, self._owner)

    def wait(
        self,
        reservation: Reservation,
        *,
        cancel_events: Iterable[threading.Event] = (),
    ) -> float:
        """Wait outside the budget lock; return measured pacing wait seconds.

        The first event wakes the wait directly; all events are checked at least
        once per polling interval.  Raising cancellation never records sent bytes.
        """
        if reservation._owner is not self._owner:
            raise ValueError("reservation belongs to another RateLimiter")
        events = tuple(cancel_events)
        started = float(self._clock())
        slept = False
        cancelled = False
        try:
            while True:
                if any(event.is_set() for event in events):
                    cancelled = True
                    raise SendCancelledError()
                remaining = reservation.ready_at - float(self._clock())
                if remaining <= 0:
                    break
                slept = True
                delay = min(remaining, self._POLL_SECONDS)
                if events:
                    events[0].wait(delay)
                else:
                    time.sleep(delay)
        finally:
            elapsed = max(0.0, float(self._clock()) - started) if slept else 0.0
            self._add(
                reservation.connection_id,
                pacing_wait_seconds=elapsed,
                pacing_waits=int(slept),
                cancelled_waits=int(cancelled),
            )
        return elapsed

    def snapshot(self) -> dict:
        """Return a JSON-serializable copy; concurrent wait durations may overlap."""
        with self._lock:
            return {
                "rate_bits_per_second": self.rate_bits_per_second,
                "chunk_bytes": self.chunk_bytes,
                "pacing_enabled": self.rate_bits_per_second > 0,
                "scope": "outgoing application socket bytes; one full-duplex direction",
                "next_ready_monotonic_seconds": self._next_ready,
                "total": dict(self._total),
                "connections": {name: dict(stats) for name, stats in self._connections.items()},
            }


class PacedSocket:
    """A connected stream socket whose send paths share a RateLimiter.

    Ordinary socket attributes/methods delegate to the wrapped socket.  send()
    sends at most one chunk and can return a partial count.  sendall() loops on
    raw send(), counting each successful partial write immediately.  Concurrent
    sends on the same wrapper raise RuntimeError rather than interleave frames.
    """

    _UNCAPPED_METHODS = frozenset(
        {"makefile", "sendfile", "sendmsg", "sendmsg_afalg", "sendto", "dup", "detach"}
    )

    def __init__(
        self,
        sock: socket.socket,
        limiter: RateLimiter,
        connection_id: str,
        *,
        cancel_event: threading.Event | None = None,
    ) -> None:
        if not isinstance(limiter, RateLimiter):
            raise TypeError("limiter must be a shared RateLimiter")
        limiter._register(connection_id)
        self._socket = sock
        self.limiter = limiter
        self.connection_id = connection_id
        self._external_cancel = cancel_event
        self._cancelled = threading.Event()
        self._send_lock = threading.Lock()

    def __getattr__(self, name):
        if name in self._UNCAPPED_METHODS:
            raise AttributeError(f"{name} bypasses pacing; use PacedSocket.send/sendall")
        return getattr(self._socket, name)

    def _is_cancelled(self) -> bool:
        return self._cancelled.is_set() or (
            self._external_cancel is not None and self._external_cancel.is_set()
        )

    def _check_cancelled(self) -> None:
        if self._is_cancelled():
            raise SendCancelledError()

    @contextmanager
    def _single_sender(self):
        if not self._send_lock.acquire(blocking=False):
            raise RuntimeError("PacedSocket requires one sender at a time per connection")
        try:
            self._check_cancelled()
            yield
        except SendCancelledError:
            self.limiter._add(self.connection_id, cancelled_send_calls=1)
            raise
        except Exception:
            self.limiter._add(self.connection_id, failed_send_calls=1)
            raise
        finally:
            self._send_lock.release()

    def _pace(self, byte_count: int) -> None:
        self._check_cancelled()
        reservation = self.limiter.reserve(byte_count, self.connection_id)
        events = [self._cancelled]
        if self._external_cancel is not None:
            events.append(self._external_cancel)
        self.limiter.wait(reservation, cancel_events=events)
        self._check_cancelled()

    def _write(self, data: memoryview, flags: int) -> int:
        self._check_cancelled()
        try:
            count = self._socket.send(data, flags)
        except OSError as exc:
            if self._is_cancelled():
                raise SendCancelledError() from exc
            raise
        if count <= 0:
            raise ConnectionError("socket.send returned zero with nonempty data")
        self.limiter._add(self.connection_id, sent_bytes=count)
        return count

    def send(self, data, flags: int = 0) -> int:
        """Reserve at most one chunk, then perform one possibly partial send."""
        with self._single_sender():
            view = memoryview(data).cast("B")
            if not view:
                return 0
            chunk = view[: self.limiter.chunk_bytes]
            self._pace(len(chunk))
            return self._write(chunk, flags)

    def sendall(self, data, flags: int = 0) -> None:
        """Send all bytes; failed/partial writes never inflate sent-byte stats."""
        with self._single_sender():
            view = memoryview(data).cast("B")
            for offset in range(0, len(view), self.limiter.chunk_bytes):
                chunk = view[offset : offset + self.limiter.chunk_bytes]
                self._pace(len(chunk))
                sent = 0
                while sent < len(chunk):
                    sent += self._write(chunk[sent:], flags)

    def recv(self, bufsize: int, flags: int = 0) -> bytes:
        data = self._socket.recv(bufsize, flags)
        self.limiter._add(self.connection_id, received_bytes=len(data))
        return data

    def recv_into(self, buffer, nbytes: int = 0, flags: int = 0) -> int:
        count = self._socket.recv_into(buffer, nbytes, flags)
        self.limiter._add(self.connection_id, received_bytes=count)
        return count

    def snapshot(self) -> dict:
        """Return this connection's counters, copied from the shared limiter."""
        return self.limiter.snapshot()["connections"][self.connection_id]

    def cancel_send(self) -> None:
        """Permanently stop this socket's sends; leave its read side available."""
        self._cancelled.set()
        try:
            self._socket.shutdown(socket.SHUT_WR)
        except OSError:
            pass

    def shutdown(self, how: int) -> None:
        if how in (socket.SHUT_WR, socket.SHUT_RDWR):
            self._cancelled.set()
        self._socket.shutdown(how)

    def close(self) -> None:
        """Cancel pacing and unblock I/O without acquiring the sender/budget lock."""
        self._cancelled.set()
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        finally:
            self._socket.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def _self_test() -> None:
    """CPU socket tests; join deadlines detect hangs, never assert throughput."""
    import struct
    import unittest

    class TransportTests(unittest.TestCase):
        def test_two_connections_share_atomic_reservation_budget(self):
            now = [100.0]
            limiter = RateLimiter(8000, clock=lambda: now[0])
            barrier = threading.Barrier(3)
            slots = []
            slots_lock = threading.Lock()

            def reserve_many(name):
                barrier.wait()
                local = [limiter.reserve(100, name) for _ in range(20)]
                with slots_lock:
                    slots.extend(local)

            workers = [threading.Thread(target=reserve_many, args=(name,)) for name in ("rpc", "migration")]
            for worker in workers:
                worker.start()
            barrier.wait()
            for worker in workers:
                worker.join(5)
                self.assertFalse(worker.is_alive(), "reservation deadlock")
            slots.sort(key=lambda slot: slot.start_at)
            self.assertEqual(len(slots), 40)
            self.assertAlmostEqual(slots[0].start_at, 100.0)
            self.assertAlmostEqual(slots[0].ready_at, 100.1)  # no initial burst
            for previous, current in zip(slots, slots[1:]):
                self.assertAlmostEqual(previous.ready_at, current.start_at)
            self.assertAlmostEqual(slots[-1].ready_at, 104.0)
            stats = limiter.snapshot()
            self.assertEqual(stats["total"]["reserved_bytes"], 4000)
            self.assertEqual(stats["total"]["sent_bytes"], 0)
            for name in ("rpc", "migration"):
                self.assertEqual(stats["connections"][name]["reserved_bytes"], 2000)
            now[0] = 200.0
            after_idle = limiter.reserve(100, "rpc")
            self.assertAlmostEqual(after_idle.ready_at, 200.1)  # no saved idle credit

        def test_socketpair_integrity_shared_actual_writes_include_headers(self):
            limiter = RateLimiter(80_000_000, chunk_bytes=257)
            payloads = {"rpc": bytes(range(256)) * 17, "migration": b"migration\x00\xff" * 1024}
            received = {}
            errors = []
            wrappers = []
            peers = []
            workers = []

            def exchange(name, sender, peer, payload):
                try:
                    sender.sendall(struct.pack("!Q", len(payload)))
                    sender.sendall(payload)
                    sender.shutdown(socket.SHUT_WR)
                except Exception as exc:
                    errors.append(exc)

            def drain(name, peer):
                try:
                    chunks = []
                    while True:
                        chunk = peer.recv(123)
                        if not chunk:
                            break
                        chunks.append(chunk)
                    received[name] = b"".join(chunks)
                except Exception as exc:
                    errors.append(exc)

            try:
                for name, payload in payloads.items():
                    raw, peer = socket.socketpair()
                    raw.settimeout(5)
                    peer.settimeout(5)
                    sender = PacedSocket(raw, limiter, name)
                    wrappers.append(sender)
                    peers.append(peer)
                    workers.extend([
                        threading.Thread(target=drain, args=(name, peer)),
                        threading.Thread(target=exchange, args=(name, sender, peer, payload)),
                    ])
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(5)
                    self.assertFalse(worker.is_alive(), "socket transfer deadlock")
                self.assertEqual(errors, [])
                for name, payload in payloads.items():
                    expected = struct.pack("!Q", len(payload)) + payload
                    self.assertEqual(received[name], expected)
                    self.assertEqual(limiter.snapshot()["connections"][name]["sent_bytes"], len(expected))
                total = sum(8 + len(payload) for payload in payloads.values())
                self.assertEqual(limiter.snapshot()["total"]["sent_bytes"], total)
                self.assertEqual(limiter.snapshot()["total"]["reserved_bytes"], total)
            finally:
                for wrapped in wrappers:
                    wrapped.close()
                for peer in peers:
                    peer.close()
                for worker in workers:
                    if worker.ident is not None:
                        worker.join(5)

        def test_zero_rate_recv_into_and_timeout_delegation(self):
            raw, peer = socket.socketpair()
            outgoing = RateLimiter(0, chunk_bytes=3)
            incoming = RateLimiter(0)
            with PacedSocket(raw, outgoing, "left") as left, PacedSocket(peer, incoming, "right") as right:
                left.settimeout(2)
                self.assertEqual(left.gettimeout(), 2)
                left.sendall(b"abcdef")
                buffer = bytearray(6)
                filled = 0
                while filled < len(buffer):
                    filled += right.recv_into(memoryview(buffer)[filled:])
                self.assertEqual(buffer, b"abcdef")
                self.assertEqual(right.snapshot()["received_bytes"], 6)
                self.assertEqual(left.snapshot()["sent_bytes"], 6)
                self.assertEqual(left.snapshot()["pacing_wait_seconds"], 0)
                self.assertFalse(outgoing.snapshot()["pacing_enabled"])
                with self.assertRaises(AttributeError):
                    left.makefile("wb")

        def test_cancellation_wakes_pacing_without_sending_reserved_bytes(self):
            reserved = threading.Event()

            class SignallingLimiter(RateLimiter):
                def reserve(self, wire_bytes, connection_id):
                    slot = super().reserve(wire_bytes, connection_id)
                    reserved.set()
                    return slot

            limiter = SignallingLimiter(8, chunk_bytes=64)
            raw, peer = socket.socketpair()
            wrapped = PacedSocket(raw, limiter, "cancelled")
            errors = []

            def attempt():
                try:
                    wrapped.sendall(b"x" * 64)
                except Exception as exc:
                    errors.append(exc)

            worker = threading.Thread(target=attempt)
            try:
                worker.start()
                self.assertTrue(reserved.wait(5), "sender never reserved")
                # This second reservation must be possible during another wait.
                second = limiter.reserve(1, "another-connection")
                self.assertGreater(second.ready_at, second.start_at)
                wrapped.cancel_send()
                worker.join(5)
                self.assertFalse(worker.is_alive(), "cancellation did not release sender")
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], SendCancelledError)
                stats = wrapped.snapshot()
                self.assertEqual(stats["reserved_bytes"], 64)
                self.assertEqual(stats["sent_bytes"], 0)
                self.assertEqual(stats["cancelled_send_calls"], 1)
                self.assertEqual(stats["cancelled_waits"], 1)
                peer.settimeout(2)
                self.assertEqual(peer.recv(1), b"")
            finally:
                wrapped.close()
                peer.close()
                worker.join(5)

        def test_partial_write_error_counts_only_successful_bytes(self):
            class PartialSocket:
                calls = 0

                def send(self, data, flags):
                    self.calls += 1
                    if self.calls == 1:
                        return 3
                    raise OSError(errno.EPIPE, "injected partial-write failure")

            limiter = RateLimiter(0, chunk_bytes=8)
            wrapped = PacedSocket(PartialSocket(), limiter, "partial")
            with self.assertRaises(OSError):
                wrapped.sendall(b"12345678")
            self.assertEqual(wrapped.snapshot()["reserved_bytes"], 8)
            self.assertEqual(wrapped.snapshot()["sent_bytes"], 3)
            self.assertEqual(wrapped.snapshot()["failed_send_calls"], 1)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TransportTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run CPU socketpair and accounting checks")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        parser.print_help()
