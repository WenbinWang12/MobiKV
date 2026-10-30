"""Request-owned E1 channels with complete application-frame accounting.

One Endpoint owns ONE outgoing RateLimiter shared by up to two connections
(normally 'rpc' then 'bulk'). The peer owns the opposite full-duplex budget.
Connect/accept handshakes use the same framed, paced send/receive path as request
traffic and are included in counters. No socket writes bypass PacedSocket.

Take snapshots only after both application protocols have fully settled. This
module checks idle operations, but cannot infer that an application has sent its
last message. Exchange snapshots using a separate outside-request mechanism;
do not send accounting snapshots over the measured channels themselves.

All byte counts are application frames accepted/read by sockets, NOT physical
wire bytes. Partial failed frames remain visible in transport byte counters but
are never counted as successfully completed frames or accepted by reconciliation.

Caller owns its listening socket. accept() sets its finite timeout; closing the
listener is the caller's way to interrupt a pending accept. Endpoint.close()
cancels its owned established channels without taking send/receive frame locks.
Connect/accept can remain in the OS until their finite timeout when no owned
connected socket exists yet. Peer identity is an explicit protocol check, not
cryptographic authentication or independent proof of physical-host placement.

Run CPU-only loopback tests: python -B e1_channels.py --self-test
"""

from __future__ import annotations

import math
import socket
import threading

from e1_transport import PacedSocket, RateLimiter
from e1_wire import FramedConnection


PROTOCOL = "restate-e1-channels-v1"
COUNTER_FIELDS = ("frames", "length_prefix_bytes", "application_header_bytes",
                  "application_payload_bytes", "application_frame_bytes")


class ChannelIdentityError(ValueError):
    """The peer's request/channel/host/configuration identity is unexpected."""


class AccountingMismatch(ValueError):
    """Settled endpoint snapshots do not reconcile exactly."""


def _empty_counts():
    return {field: 0 for field in COUNTER_FIELDS}


def _sum_counts(items):
    return {field: sum(item[field] for item in items) for field in COUNTER_FIELDS}


class Channel:
    """One framed channel; send/recv include handshake frames in their counters."""

    def __init__(self, endpoint, name, request_id, paced, role):
        self.endpoint = endpoint
        self.name = name
        self.request_id = request_id
        self.role = role
        self._paced = paced
        self._framed = FramedConnection(paced, integrity_mode=endpoint.integrity_mode)
        self._lock = threading.Lock()
        self._sent, self._received = _empty_counts(), _empty_counts()
        self._active = {"send": 0, "recv": 0}
        self.peer_hostname = None
        self.handshake_complete = False

    def _begin(self, direction):
        with self._lock:
            self._active[direction] += 1

    def _finish(self, direction, stats=None):
        with self._lock:
            if stats is not None:
                target = self._sent if direction == "send" else self._received
                target["frames"] += 1
                for field in COUNTER_FIELDS[1:]:
                    target[field] += stats[field]
            self._active[direction] -= 1

    def send(self, metadata, tensors=()):
        self._begin("send")
        stats = None
        try:
            stats = self._framed.send_message(metadata, tensors)
            return stats
        except BaseException as exc:
            self.endpoint._fail("send", self.name, exc)
            raise
        finally:
            self._finish("send", stats)

    def recv(self, *, tensor_format="torch"):
        """Return (metadata, tensors, stats); bytes format is available for CPU tests."""
        self._begin("recv")
        stats = None
        try:
            metadata, tensors, stats = self._framed.recv_message(tensor_format=tensor_format)
            return metadata, tensors, stats
        except BaseException as exc:
            self.endpoint._fail("recv", self.name, exc)
            raise
        finally:
            self._finish("recv", stats)

    def snapshot(self):
        with self._lock:
            result = {"name": self.name, "request_id": self.request_id, "role": self.role,
                      "peer_hostname": self.peer_hostname, "handshake_complete": self.handshake_complete,
                      "active_operations": dict(self._active), "sent": dict(self._sent),
                      "received": dict(self._received)}
        result["transport"] = self._paced.snapshot()
        result["socket_closed"] = self._paced.fileno() == -1
        return result

    def close(self):
        self._framed.close()


class Endpoint:
    """Request endpoint with one shared outgoing cap, including hello/ack.

    One endpoint binds one request_id and accepts at most two distinct channel
    names. Create a fresh endpoint for another request. allow_same_host is an
    explicit CPU-test escape hatch and is recorded in every endpoint snapshot.
    """

    def __init__(self, rate_bits_per_second, integrity_mode="none", timeout=120,
                 *, allow_same_host=False):
        if integrity_mode not in ("none", "sha256"):
            raise ValueError("integrity_mode must be 'none' or 'sha256'")
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.limiter = RateLimiter(rate_bits_per_second)
        self.integrity_mode = integrity_mode
        self.timeout = timeout
        self.allow_same_host = bool(allow_same_host)
        self.hostname = socket.gethostname()
        self.request_id = None
        self.peer_hostname = None
        self.no_interhost_request = False
        self._lock = threading.Lock()
        self._channels = {}
        self._pending = set()
        self._closed = threading.Event()
        self._failures = []
        self._cleanup_errors = []

    def declare_local_request(self, request_id, peer_hostname):
        """Bind an already-agreed local action without opening a request socket."""
        if not all(isinstance(value, str) and value for value in (request_id, peer_hostname)):
            raise ValueError("A local action requires request and peer identities")
        if peer_hostname == self.hostname and not self.allow_same_host:
            raise ValueError("Expected a distinct peer from the outside-request identity exchange")
        with self._lock:
            if self.request_id is not None or self._channels or self._pending or self._closed.is_set():
                raise ValueError("Only an unused endpoint can declare a local request")
            self.request_id, self.peer_hostname = request_id, peer_hostname
            self.no_interhost_request = True

    def _reserve_name(self, name, request_id):
        if not isinstance(name, str) or not name or not isinstance(request_id, str) or not request_id:
            raise ValueError("name and request_id must be nonempty strings")
        with self._lock:
            if self._closed.is_set():
                raise ConnectionError("Endpoint is closed")
            if self.no_interhost_request:
                raise ValueError("A declared local action cannot open an inter-host request channel")
            if self.request_id is not None and self.request_id != request_id:
                raise ValueError("One Endpoint cannot mix request IDs")
            if name in self._channels or name in self._pending:
                raise ValueError("Channel name is already owned by this Endpoint")
            if len(set(self._channels) | self._pending) >= 2:
                raise ValueError("One Endpoint supports at most two channels")
            self.request_id = request_id
            self._pending.add(name)

    def _adopt(self, raw, name, request_id, role):
        try:
            raw.settimeout(self.timeout)
            if raw.family in (socket.AF_INET, socket.AF_INET6):
                raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                if self._closed.is_set():
                    raise ConnectionError("Endpoint closed while establishing channel")
                paced = PacedSocket(raw, self.limiter, name)
                channel = Channel(self, name, request_id, paced, role)
                self._channels[name] = channel
            return channel
        except BaseException:
            raw.close()
            raise

    def _identity(self, kind, name, request_id, peer_hostname=None):
        identity = {"protocol": PROTOCOL, "kind": kind, "request_id": request_id,
                    "name": name, "hostname": self.hostname,
                    "integrity_mode": self.integrity_mode,
                    "rate_bits_per_second": self.limiter.rate_bits_per_second}
        if peer_hostname is not None:
            identity["peer_hostname"] = peer_hostname
        return identity

    def _validate_identity(self, message, tensors, name, request_id, kind):
        expected = self._identity(kind, name, request_id)
        for field in ("protocol", "kind", "request_id", "name", "integrity_mode", "rate_bits_per_second"):
            if message.get(field) != expected[field]:
                raise ChannelIdentityError(f"Channel handshake mismatch: {field}")
        if tensors:
            raise ChannelIdentityError("Handshake must not contain tensor payloads")
        peer = message.get("hostname")
        if not isinstance(peer, str) or not peer:
            raise ChannelIdentityError("Missing peer hostname")
        if not self.allow_same_host and peer == self.hostname:
            raise ChannelIdentityError("Peer hostname must differ from this host")
        if kind == "ack" and message.get("peer_hostname") != self.hostname:
            raise ChannelIdentityError("Acknowledgement does not identify its connecting peer")
        with self._lock:
            if self.peer_hostname is not None and self.peer_hostname != peer:
                raise ChannelIdentityError("Channels belong to different peer hosts")
            self.peer_hostname = peer
        return peer

    def connect(self, host, port, name, request_id):
        self._reserve_name(name, request_id)
        try:
            raw = socket.create_connection((host, port), timeout=self.timeout)
            channel = self._adopt(raw, name, request_id, "connector")
            channel.send(self._identity("hello", name, request_id))
            message, tensors, _ = channel.recv(tensor_format="bytes")
            peer = self._validate_identity(message, tensors, name, request_id, "ack")
            with channel._lock:
                channel.peer_hostname = peer
                channel.handshake_complete = True
            return channel
        except BaseException as exc:
            self._fail("connect", name, exc)
            raise
        finally:
            with self._lock:
                self._pending.discard(name)

    def accept(self, listener, expected_name, request_id):
        self._reserve_name(expected_name, request_id)
        try:
            listener.settimeout(self.timeout)
            raw, _ = listener.accept()
            channel = self._adopt(raw, expected_name, request_id, "acceptor")
            message, tensors, _ = channel.recv(tensor_format="bytes")
            peer = self._validate_identity(message, tensors, expected_name, request_id, "hello")
            channel.send(self._identity("ack", expected_name, request_id, peer_hostname=peer))
            with channel._lock:
                channel.peer_hostname = peer
                channel.handshake_complete = True
            return channel
        except BaseException as exc:
            self._fail("accept", expected_name, exc)
            raise
        finally:
            with self._lock:
                self._pending.discard(expected_name)

    def _fail(self, operation, name, exc):
        with self._lock:
            self._failures.append({"operation": operation, "channel": name,
                                   "type": type(exc).__name__, "message": str(exc)})
        self.close()

    def snapshot(self):
        """Snapshot idle channels; caller must already have settled its protocol."""
        with self._lock:
            channels = dict(self._channels)
            pending = sorted(self._pending)
            failures = list(self._failures)
            cleanup_errors = list(self._cleanup_errors)
        snapshots = {name: channel.snapshot() for name, channel in sorted(channels.items())}
        if pending or any(any(item["active_operations"].values()) for item in snapshots.values()):
            raise RuntimeError("Endpoint snapshot requires no pending connection/send/receive operation")
        return {
            "protocol": PROTOCOL, "hostname": self.hostname, "peer_hostname": self.peer_hostname,
            "request_id": self.request_id, "integrity_mode": self.integrity_mode,
            "no_interhost_request": self.no_interhost_request,
            "rate_bits_per_second": self.limiter.rate_bits_per_second,
            "allow_same_host_test_only": self.allow_same_host, "closed": self._closed.is_set(),
            "failures": failures, "cleanup_errors": cleanup_errors, "channels": snapshots,
            "totals": {direction: _sum_counts([item[direction] for item in snapshots.values()])
                       for direction in ("sent", "received")},
            "limiter": self.limiter.snapshot(),
            "physical_wire_bytes": None, "wire_measurement_available": False,
            "snapshot_scope": "Caller-settled request, including identity handshakes; no in-band stats exchange",
        }

    def close(self):
        self._closed.set()
        with self._lock:
            channels = list(self._channels.values())
        for channel in channels:
            try:
                channel.close()
            except OSError as exc:
                with self._lock:
                    self._cleanup_errors.append(f"{channel.name}: {type(exc).__name__}: {exc}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def _require(condition, message):
    if not condition:
        raise AccountingMismatch(message)


def _validate_accounting(snapshot, label):
    _require(snapshot.get("protocol") == PROTOCOL, f"{label}: unrecognized snapshot protocol")
    _require(not snapshot["failures"] and not snapshot["cleanup_errors"], f"{label}: endpoint recorded failures")
    channels = snapshot["channels"]
    local = snapshot.get("no_interhost_request", False)
    _require(type(local) is bool and (bool(channels) or local), f"{label}: no completed channels or declared local action")
    if local:
        _require(not channels and snapshot.get("closed") is True, f"{label}: a local action must settle without request sockets")
    _require(set(channels) == set(snapshot["limiter"]["connections"]), f"{label}: limiter/channel coverage differs")
    _require(snapshot["rate_bits_per_second"] == snapshot["limiter"]["rate_bits_per_second"], f"{label}: cap configuration differs")
    for name, channel in channels.items():
        path = f"{label}/{name}"
        _require(channel["handshake_complete"], f"{path}: incomplete handshake")
        _require(channel["request_id"] == snapshot["request_id"], f"{path}: request identity differs")
        _require(channel["peer_hostname"] == snapshot["peer_hostname"], f"{path}: peer host differs")
        _require(not any(channel["active_operations"].values()), f"{path}: operations still active")
        _require(channel["transport"] == snapshot["limiter"]["connections"][name], f"{path}: transport snapshot differs")
        for direction, transport_field in (("sent", "sent_bytes"), ("received", "received_bytes")):
            count = channel[direction]
            _require(all(type(count[field]) is int and count[field] >= 0 for field in COUNTER_FIELDS), f"{path}/{direction}: invalid count")
            _require(count["length_prefix_bytes"] == 4 * count["frames"], f"{path}/{direction}: frame prefix count differs")
            _require(count["application_frame_bytes"] == count["length_prefix_bytes"] + count["application_header_bytes"] + count["application_payload_bytes"], f"{path}/{direction}: frame byte decomposition differs")
            _require(count["application_frame_bytes"] == channel["transport"][transport_field], f"{path}/{direction}: completed frames differ from actual socket bytes")
    for direction, transport_field in (("sent", "sent_bytes"), ("received", "received_bytes")):
        actual_total = _sum_counts([channel[direction] for channel in channels.values()])
        _require(snapshot["totals"][direction] == actual_total, f"{label}/{direction}: aggregate counters differ")
        _require(actual_total["application_frame_bytes"] == snapshot["limiter"]["total"][transport_field], f"{label}/{direction}: aggregate socket bytes differ")
    _require(snapshot["physical_wire_bytes"] is None and snapshot["wire_measurement_available"] is False,
             f"{label}: application counters cannot certify physical wire bytes")


def reconcile_pair(destination_snapshot, holder_snapshot):
    """Return a successful audit, or raise AccountingMismatch on any mismatch."""
    destination, holder = destination_snapshot, holder_snapshot
    try:
        _validate_accounting(destination, "destination")
        _validate_accounting(holder, "holder")
        for field in ("request_id", "integrity_mode", "rate_bits_per_second"):
            _require(destination[field] == holder[field], f"Peer {field} differs")
        _require(destination.get("no_interhost_request", False) == holder.get("no_interhost_request", False),
                 "Peer local-request declarations differ")
        same_host_test = destination["allow_same_host_test_only"] and holder["allow_same_host_test_only"]
        _require(destination["hostname"] != holder["hostname"] or same_host_test, "Physical-host identity gate failed")
        _require(destination["peer_hostname"] == holder["hostname"] and holder["peer_hostname"] == destination["hostname"], "Peer hostname identities do not cross-match")
        _require(set(destination["channels"]) == set(holder["channels"]), "Peer channel names differ")
        channels = {}
        for name in destination["channels"]:
            local, remote = destination["channels"][name], holder["channels"][name]
            _require({local["role"], remote["role"]} == {"connector", "acceptor"}, f"{name}: channel side roles differ")
            _require(local["sent"] == remote["received"], f"{name}: destination sent differs from holder received")
            _require(local["received"] == remote["sent"], f"{name}: holder sent differs from destination received")
            channels[name] = {"destination_to_holder": dict(local["sent"]),
                              "holder_to_destination": dict(remote["sent"])}
        _require(destination["totals"]["sent"] == holder["totals"]["received"], "Aggregate destination-to-holder counts differ")
        _require(holder["totals"]["sent"] == destination["totals"]["received"], "Aggregate holder-to-destination counts differ")
        return {"passed": True, "request_id": destination["request_id"],
                "hosts": {"destination": destination["hostname"], "holder": holder["hostname"]},
                "same_host_cpu_test": same_host_test, "channels": channels,
                "totals": {"destination_to_holder": dict(destination["totals"]["sent"]),
                           "holder_to_destination": dict(holder["totals"]["sent"])},
                "physical_wire_bytes": None, "wire_measurement_available": False}
    except (KeyError, TypeError) as exc:
        raise AccountingMismatch(f"Malformed endpoint accounting snapshot: {exc}") from exc


def _self_test():
    import copy
    import unittest
    from e1_wire import TensorBuffer

    class ChannelTests(unittest.TestCase):
        def test_declared_local_action_has_zero_bytes_and_cannot_connect(self):
            destination, holder = Endpoint(100_000_000), Endpoint(100_000_000)
            destination.hostname, holder.hostname = "destination-fixture", "holder-fixture"
            for endpoint, peer in ((destination, holder), (holder, destination)):
                endpoint.declare_local_request("local-fixture", peer.hostname)
                with self.assertRaises(ValueError):
                    endpoint.connect("unused", 1, "rpc", "local-fixture")
                endpoint.close()
            snapshots = [endpoint.snapshot() for endpoint in (destination, holder)]
            result = reconcile_pair(*snapshots)
            self.assertTrue(result["passed"])
            self.assertEqual(result["channels"], {})
            self.assertEqual(result["totals"]["destination_to_holder"]["application_frame_bytes"], 0)
            forged = copy.deepcopy(snapshots[0])
            forged["limiter"]["total"]["sent_bytes"] = 1
            with self.assertRaises(AccountingMismatch):
                reconcile_pair(forged, snapshots[1])

        def listener(self):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", 0))
            listener.listen(2)
            listener.settimeout(2)
            self.addCleanup(listener.close)
            return listener

        def endpoints(self, allow_same_host=True):
            endpoints = [Endpoint(100_000_000, timeout=2, allow_same_host=allow_same_host) for _ in range(2)]
            for endpoint in endpoints:
                self.addCleanup(endpoint.close)
            return endpoints

        def exchange(self):
            listener = self.listener()
            destination, holder = self.endpoints()
            result, errors = {}, []
            payloads = {"rpc": b"abc", "bulk": bytes(range(251))}

            def serve():
                try:
                    channels = {name: holder.accept(listener, name, "cpu-request") for name in ("rpc", "bulk")}
                    for name, channel in channels.items():
                        metadata, tensors, _ = channel.recv(tensor_format="bytes")
                        if metadata != {"kind": "request", "name": name} or tensors[0].data != payloads[name]:
                            raise AssertionError("Request data changed")
                        channel.send({"kind": "reply", "name": name}, tensors)
                    result["holder"] = holder.snapshot()
                except BaseException as exc:
                    errors.append(exc)

            worker = threading.Thread(target=serve)
            worker.start()
            try:
                channels = {name: destination.connect(*listener.getsockname(), name, "cpu-request") for name in ("rpc", "bulk")}
                self.assertIs(channels["rpc"]._paced.limiter, channels["bulk"]._paced.limiter)
                self.assertIs(channels["rpc"]._paced.limiter, destination.limiter)
                for name, channel in channels.items():
                    channel.send({"kind": "request", "name": name}, [TensorBuffer("uint8", (len(payloads[name]),), payloads[name])])
                    metadata, tensors, _ = channel.recv(tensor_format="bytes")
                    self.assertEqual(metadata, {"kind": "reply", "name": name})
                    self.assertEqual(tensors[0].data, payloads[name])
                worker.join(5)
                self.assertFalse(worker.is_alive(), "Loopback exchange did not settle")
                self.assertEqual(errors, [])
                result["destination"] = destination.snapshot()
                return destination, holder, result
            finally:
                if worker.is_alive():
                    destination.close()
                    holder.close()
                    listener.close()
                    worker.join(5)

        def test_two_channels_shared_budget_handshake_and_reconciliation(self):
            destination, holder, result = self.exchange()
            audit = reconcile_pair(result["destination"], result["holder"])
            self.assertTrue(audit["passed"])
            self.assertIsNone(audit["physical_wire_bytes"])
            for snapshot in result.values():
                self.assertEqual(snapshot["totals"]["sent"]["frames"], 4)
                self.assertEqual(snapshot["totals"]["received"]["frames"], 4)
                self.assertEqual(snapshot["totals"]["sent"]["application_payload_bytes"], 254)
                self.assertEqual(snapshot["limiter"]["total"]["sent_bytes"], snapshot["totals"]["sent"]["application_frame_bytes"])
                self.assertEqual(snapshot["limiter"]["total"]["reserved_bytes"], snapshot["limiter"]["total"]["sent_bytes"])
            destination.close()
            holder.close()
            for endpoint in (destination, holder):
                self.assertTrue(all(channel["socket_closed"] for channel in endpoint.snapshot()["channels"].values()))

        def test_mismatched_accounting_fails(self):
            _, _, result = self.exchange()
            bad = copy.deepcopy(result["holder"])
            bad["channels"]["rpc"]["received"]["application_payload_bytes"] += 1
            with self.assertRaises(AccountingMismatch):
                reconcile_pair(result["destination"], bad)
            bad = copy.deepcopy(result["holder"])
            bad["limiter"]["total"]["received_bytes"] += 1
            with self.assertRaises(AccountingMismatch):
                reconcile_pair(result["destination"], bad)

        def test_wrong_identity_and_same_host_gate_close_owned_sockets(self):
            for expected_name, expected_request, allow_same_host in (
                    ("bulk", "cpu-request", True), ("rpc", "another-request", True),
                    ("rpc", "cpu-request", False)):
                with self.subTest(name=expected_name, request=expected_request, allow_same_host=allow_same_host):
                    listener = self.listener()
                    destination, holder = self.endpoints(allow_same_host)
                    errors = []

                    def reject():
                        try:
                            holder.accept(listener, expected_name, expected_request)
                        except BaseException as exc:
                            errors.append(exc)

                    worker = threading.Thread(target=reject)
                    worker.start()
                    try:
                        with self.assertRaises((OSError, EOFError, ChannelIdentityError)):
                            destination.connect(*listener.getsockname(), "rpc", "cpu-request")
                        worker.join(5)
                        self.assertFalse(worker.is_alive(), "Identity rejection did not unblock peer")
                        self.assertEqual(len(errors), 1)
                        self.assertIsInstance(errors[0], ChannelIdentityError)
                        for endpoint in (destination, holder):
                            snapshot = endpoint.snapshot()
                            self.assertTrue(snapshot["failures"])
                            self.assertTrue(all(channel["socket_closed"] for channel in snapshot["channels"].values()))
                    finally:
                        destination.close()
                        holder.close()
                        listener.close()
                        worker.join(5)

        def test_local_send_error_closes_both_channels(self):
            destination, _, _ = self.exchange()
            with self.assertRaises(ValueError):
                destination._channels["rpc"].send({}, [TensorBuffer("uint8", (4,), b"bad")])
            snapshot = destination.snapshot()
            self.assertTrue(snapshot["failures"])
            self.assertTrue(snapshot["closed"])
            self.assertTrue(all(channel["socket_closed"] for channel in snapshot["channels"].values()))

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ChannelTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run bounded CPU loopback tests")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        parser.print_help()
