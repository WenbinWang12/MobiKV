"""Typed, bounded E1 application frames; no pickle, GPU access, or rate sleeps.

Wrap a supplied stream socket (normally e1_transport.PacedSocket) in exactly one
FramedConnection.  Its lock covers a COMPLETE outgoing frame, including the
length prefix, JSON header, and all payloads.  All writes use that supplied
socket's sendall; there is no raw-socket or Gloo bypass.  Concurrent readers also
serialize at complete-frame boundaries.

The explicit integrity_mode must agree at both peers.  'none', the performance
mode, neither computes nor carries tensor hashes.  'sha256' is a separate
diagnostic mode.  Both modes validate the entire bounded header before allocating
any payload buffers.  Protocol/receive/write errors invalidate and close the
connection; malformed input is not followed by an attempted stream resync.

Wire layout: network-order uint32 JSON length, UTF-8 JSON header, contiguous
native-endian tensor bytes in descriptor order.  Counters are APPLICATION bytes,
not physical wire/TCP/IP/RDMA measurements.  Serialization copies CPU tensor
bytes into an owned snapshot; include this work in the E1 request interval.

Public API:
    connection = FramedConnection(paced_socket, integrity_mode='none')
    stats = connection.send_message(metadata, [cpu_tensor, ...])
    metadata, tensors, stats = connection.recv_message()

TensorBuffer and tensor_format='bytes' offer the same typed framing without
importing torch, for control-plane/CPU tests.  Torch is otherwise imported lazily;
the runner must finish its normal engine/import warmup before request timing.
Run: python -B e1_wire.py --self-test
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import socket
import struct
import sys
import threading
import time


PROTOCOL = "restate-e1-wire-v1"
MAX_HEADER_BYTES = 32 * 1024 * 1024
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
MAX_TENSOR_DIMS = 8
MAX_TENSORS = 256
INTEGRITY_MODES = frozenset({"none", "sha256"})
DTYPE_BYTES = {
    "float32": 4, "float64": 8, "float16": 2, "bfloat16": 2,
    "int64": 8, "int32": 4, "int16": 2, "int8": 1, "uint8": 1, "bool": 1,
}


class FrameProtocolError(ValueError):
    """A received frame violates the frozen framing/integrity contract."""


@dataclass(frozen=True)
class TensorBuffer:
    """Native-endian typed CPU bytes; received data owns a writable bytearray.

    Sending snapshots data, including caller-owned bytearrays/memoryviews.  Shape
    and dtype are validated by the codec.  This is not a pickled tensor object.
    """

    dtype: str
    shape: tuple[int, ...]
    data: bytes | bytearray | memoryview


def _mode(mode: str) -> str:
    if not isinstance(mode, str) or mode not in INTEGRITY_MODES:
        raise ValueError("integrity_mode must be 'none' or 'sha256'")
    return mode


def _shape_bytes(shape, dtype) -> int:
    if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
        raise FrameProtocolError("Unsupported tensor dtype")
    if not isinstance(shape, list) or len(shape) > MAX_TENSOR_DIMS:
        raise FrameProtocolError("Tensor shape must be a list with at most eight dimensions")
    if any(type(dim) is not int or not 0 <= dim <= MAX_PAYLOAD_BYTES for dim in shape):
        raise FrameProtocolError("Invalid tensor shape dimension")
    if math.prod(max(dim, 1) for dim in shape) > 2**63 - 1:
        raise FrameProtocolError("Tensor shape has unrepresentable strides")
    size = math.prod(shape) * DTYPE_BYTES[dtype]
    if size > MAX_PAYLOAD_BYTES:
        raise FrameProtocolError("Tensor exceeds payload bound")
    return size


def _digest(data) -> str:
    # This function is never called in performance ('none') mode.
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _reject_constant(value):
    raise FrameProtocolError(f"Non-finite JSON constant: {value}")


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise FrameProtocolError(f"Duplicate JSON key: {name}")
        result[name] = value
    return result


def _validate_header(raw: bytes | bytearray, expected_mode: str):
    try:
        header = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant,
                            object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FrameProtocolError("Invalid UTF-8 JSON header") from exc
    expected_keys = {"protocol", "byteorder", "integrity_mode", "metadata", "tensors"}
    if not isinstance(header, dict) or set(header) != expected_keys or header["protocol"] != PROTOCOL:
        raise FrameProtocolError("Unrecognized E1 frame header/protocol")
    if header["integrity_mode"] != expected_mode:
        raise FrameProtocolError("Peer integrity mode mismatch")
    if header["byteorder"] != sys.byteorder:
        raise FrameProtocolError("Peer tensor byte order differs")
    metadata, descriptors = header["metadata"], header["tensors"]
    if not isinstance(metadata, dict) or not isinstance(descriptors, list) or len(descriptors) > MAX_TENSORS:
        raise FrameProtocolError("Invalid metadata or tensor descriptor list")
    total = 0
    descriptor_keys = {"dtype", "shape", "nbytes"}
    if expected_mode == "sha256":
        descriptor_keys.add("sha256")
    for descriptor in descriptors:
        if not isinstance(descriptor, dict) or set(descriptor) != descriptor_keys:
            raise FrameProtocolError("Unexpected tensor descriptor fields for integrity mode")
        size = _shape_bytes(descriptor["shape"], descriptor["dtype"])
        if type(descriptor["nbytes"]) is not int or descriptor["nbytes"] != size:
            raise FrameProtocolError("Tensor byte count disagrees with shape and dtype")
        if expected_mode == "sha256":
            digest = descriptor["sha256"]
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)):
                raise FrameProtocolError("Invalid SHA256 descriptor")
        total += size
        if total > MAX_PAYLOAD_BYTES:
            raise FrameProtocolError("Combined tensor payload exceeds frame bound")
    return metadata, descriptors, total


def _snapshot(tensor, remaining_bytes: int) -> tuple[str, list[int], bytes]:
    if isinstance(tensor, TensorBuffer):
        dtype, shape = tensor.dtype, list(tensor.shape)
        expected = _shape_bytes(shape, dtype)
        if expected > remaining_bytes:
            raise ValueError("Combined tensor payload exceeds frame bound")
        view = memoryview(tensor.data).cast("B")
        if len(view) != expected:
            raise ValueError("TensorBuffer data length disagrees with shape and dtype")
        return dtype, shape, view.tobytes()
    import torch
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Payload entries must be dense CPU torch.Tensor or TensorBuffer")
    if tensor.device.type != "cpu" or tensor.layout != torch.strided:
        raise TypeError("Only dense CPU tensors may be serialized")
    dtype, shape = str(tensor.dtype).removeprefix("torch."), list(tensor.shape)
    expected = _shape_bytes(shape, dtype)
    if expected > remaining_bytes:
        raise ValueError("Combined tensor payload exceeds frame bound")
    if tensor.element_size() != DTYPE_BYTES[dtype]:
        raise ValueError("Torch dtype size disagrees with the wire contract")
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    if len(raw) != expected:
        raise ValueError("Tensor serialization returned an unexpected byte count")
    return dtype, shape, raw


def _encode(metadata, tensors, integrity_mode):
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")
    if not isinstance(tensors, (list, tuple)) or len(tensors) > MAX_TENSORS:
        raise TypeError("tensors must be a bounded list or tuple")
    descriptors, payloads, total = [], [], 0
    for tensor in tensors:
        dtype, shape, raw = _snapshot(tensor, MAX_PAYLOAD_BYTES - total)
        total += len(raw)
        if total > MAX_PAYLOAD_BYTES:
            raise ValueError("Combined tensor payload exceeds frame bound")
        descriptor = {"dtype": dtype, "shape": shape, "nbytes": len(raw)}
        if integrity_mode == "sha256":
            descriptor["sha256"] = _digest(raw)
        descriptors.append(descriptor)
        payloads.append(raw)
    header = json.dumps({"protocol": PROTOCOL, "byteorder": sys.byteorder,
                         "integrity_mode": integrity_mode, "metadata": metadata,
                         "tensors": descriptors}, allow_nan=False,
                        separators=(",", ":")).encode("utf-8")
    if not 0 < len(header) <= MAX_HEADER_BYTES:
        raise ValueError("JSON header exceeds frame bound")
    return header, payloads, total


def _recv_exact(sock, count: int) -> bytearray:
    data = bytearray(count)
    view = memoryview(data)
    received = 0
    while received < count:
        amount = sock.recv_into(view[received:])
        if amount == 0:
            raise EOFError(f"Socket closed with {count - received} application frame bytes missing")
        received += amount
    return data


def _stats(header_bytes: int, payload_bytes: int, integrity_mode: str, elapsed: float) -> dict:
    return {
        "length_prefix_bytes": 4,
        "application_header_bytes": header_bytes,
        "application_payload_bytes": payload_bytes,
        "application_frame_bytes": 4 + header_bytes + payload_bytes,
        "integrity_mode": integrity_mode,
        "elapsed_seconds": elapsed,
        "byte_scope": "application frame only; excludes TCP/IP and physical-link overhead",
    }


class FramedConnection:
    """Own one framed stream; send/receive each have a whole-frame lock.

    send_message statistics return only after every frame byte was accepted by
    the supplied socket.  They do not imply peer delivery.  If a send fails, use
    the supplied PacedSocket's counters for actual successful partial writes;
    never count the intended full frame as sent.  Timings include serialization,
    lock/queue wait and I/O (and receive tensor wrapping), not only network time.
    """

    def __init__(self, sock, *, integrity_mode: str = "none"):
        self._socket = sock
        self.integrity_mode = _mode(integrity_mode)
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()
        self._closed = threading.Event()

    def _ensure_open(self):
        if self._closed.is_set():
            raise ConnectionError("Framed connection is closed or invalidated")

    def send_message(self, metadata: dict, tensors=()) -> dict:
        self._ensure_open()
        started = time.monotonic()
        # Validate and snapshot before writing any bytes; local invalid inputs
        # do not corrupt the stream and therefore need not close it.
        header, payloads, total = _encode(metadata, tensors, self.integrity_mode)
        with self._send_lock:
            self._ensure_open()
            try:
                self._socket.sendall(struct.pack("!I", len(header)))
                self._socket.sendall(header)
                for payload in payloads:
                    self._socket.sendall(payload)
            except BaseException:
                self.close()
                raise
        return _stats(len(header), total, self.integrity_mode, time.monotonic() - started)

    def recv_message(self, *, tensor_format: str = "torch"):
        """Return metadata, tensor list, and application-byte stats.

        tensor_format='bytes' returns TensorBuffer instances backed by separate
        owned writable bytearrays.  The default 'torch' wraps those buffers as
        CPU tensors, preserving dtype/shape/native bits without a pickle loader.
        """
        if tensor_format not in ("torch", "bytes"):
            raise ValueError("tensor_format must be 'torch' or 'bytes'")
        if tensor_format == "torch":
            import torch
        self._ensure_open()
        started = time.monotonic()
        with self._recv_lock:
            self._ensure_open()
            try:
                header_length = struct.unpack("!I", _recv_exact(self._socket, 4))[0]
                if not 0 < header_length <= MAX_HEADER_BYTES:
                    raise FrameProtocolError("JSON header length is outside bounds")
                metadata, descriptors, total = _validate_header(
                    _recv_exact(self._socket, header_length), self.integrity_mode)
                # No payload allocation occurs until ALL descriptors validate.
                tensors = []
                for descriptor in descriptors:
                    raw = _recv_exact(self._socket, descriptor["nbytes"])
                    if self.integrity_mode == "sha256" and _digest(raw) != descriptor["sha256"]:
                        raise FrameProtocolError("Tensor SHA256 mismatch")
                    if tensor_format == "bytes":
                        tensor = TensorBuffer(descriptor["dtype"], tuple(descriptor["shape"]), raw)
                    else:
                        dtype = getattr(torch, descriptor["dtype"])
                        if descriptor["nbytes"]:
                            tensor = torch.frombuffer(raw, dtype=dtype).reshape(descriptor["shape"]).detach()
                        else:
                            tensor = torch.empty(descriptor["shape"], dtype=dtype, device="cpu")
                    tensors.append(tensor)
            except BaseException:
                self.close()
                raise
        return metadata, tensors, _stats(header_length, total, self.integrity_mode, time.monotonic() - started)

    def settimeout(self, value):
        return self._socket.settimeout(value)

    def gettimeout(self):
        return self._socket.gettimeout()

    def fileno(self):
        return self._socket.fileno()

    def shutdown(self, how):
        return self._socket.shutdown(how)

    def close(self):
        # Do not take frame locks: closing must interrupt blocked readers/writers.
        self._closed.set()
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


def send_message(connection: FramedConnection, metadata: dict, tensors=()) -> dict:
    """Convenience API; require the wrapper so complete-frame locking is retained."""
    if not isinstance(connection, FramedConnection):
        raise TypeError("Pass one shared FramedConnection, not a raw socket")
    return connection.send_message(metadata, tensors)


def recv_message(connection: FramedConnection, *, tensor_format: str = "torch"):
    if not isinstance(connection, FramedConnection):
        raise TypeError("Pass one shared FramedConnection, not a raw socket")
    return connection.recv_message(tensor_format=tensor_format)


def _self_test():
    import importlib.util
    import unittest
    from unittest.mock import patch

    class FragmentingSocket:
        def __init__(self, sock):
            self.sock = sock

        def sendall(self, data):
            view = memoryview(data)
            for offset in range(0, len(view), 7):
                self.sock.sendall(view[offset:offset + 7])
                time.sleep(0)  # Exercise scheduling, not a measured delay/cap.

        def recv_into(self, view):
            return self.sock.recv_into(view[:3])

        def __getattr__(self, name):
            return getattr(self.sock, name)

    def raw_frame(metadata, descriptors, mode="none", payload=b""):
        header = json.dumps({"protocol": PROTOCOL, "byteorder": sys.byteorder,
                             "integrity_mode": mode, "metadata": metadata,
                             "tensors": descriptors}, separators=(",", ":")).encode()
        return struct.pack("!I", len(header)) + header + payload, len(header)

    class CodecTests(unittest.TestCase):
        def pair(self, mode="none", fragmented=False):
            left, right = socket.socketpair()
            left.settimeout(5)
            right.settimeout(5)
            sender = FramedConnection(FragmentingSocket(left) if fragmented else left, integrity_mode=mode)
            receiver = FramedConnection(FragmentingSocket(right) if fragmented else right, integrity_mode=mode)
            self.addCleanup(sender.close)
            self.addCleanup(receiver.close)
            return sender, receiver

        def test_typed_buffers_partial_receive_scalar_empty_and_native_bits(self):
            for mode in ("none", "sha256"):
                with self.subTest(mode=mode):
                    sender, receiver = self.pair(mode, fragmented=True)
                    inputs = [TensorBuffer("int64", (), struct.pack("=q", -17)),
                              TensorBuffer("bfloat16", (2,), struct.pack("=HH", 0x3F80, 0xBF80)),
                              TensorBuffer("bool", (0, 3), b""),
                              TensorBuffer("uint8", (2, 2), bytearray(b"\x00\xff\x01\x80"))]
                    sent, errors = {}, []

                    def send_fragmented():
                        try:
                            sent["stats"] = sender.send_message({"kind": "typed", "label": "鍓嶇紑"}, inputs)
                        except BaseException as exc:
                            errors.append(exc)

                    # Tiny writes can exhaust Linux socketpair packet buffers
                    # even when the frame contains few bytes. Drain concurrently
                    # as a real peer would; do not rely on socket buffer capacity.
                    worker = threading.Thread(target=send_fragmented)
                    try:
                        worker.start()
                        metadata, outputs, got_stats = receiver.recv_message(tensor_format="bytes")
                        worker.join(5)
                        self.assertFalse(worker.is_alive(), "fragmented transfer deadlock")
                        self.assertEqual(errors, [])
                        stats = sent["stats"]
                    finally:
                        sender.close()
                        receiver.close()
                        worker.join(5)
                    self.assertEqual(metadata, {"kind": "typed", "label": "鍓嶇紑"})
                    self.assertEqual(len(outputs), len(inputs))
                    for original, restored in zip(inputs, outputs):
                        self.assertEqual((restored.dtype, restored.shape), (original.dtype, original.shape))
                        self.assertEqual(restored.data, original.data)
                        self.assertIsInstance(restored.data, bytearray)
                    self.assertEqual(len({id(item.data) for item in outputs}), len(outputs))
                    self.assertEqual(stats["application_payload_bytes"], 16)
                    self.assertEqual(stats["application_frame_bytes"], got_stats["application_frame_bytes"])
                    self.assertEqual(stats["application_frame_bytes"], 4 + stats["application_header_bytes"] + 16)

        def test_performance_mode_never_hashes_and_local_validation_keeps_stream(self):
            sender, receiver = self.pair()
            with patch.object(sys.modules[__name__], "_digest", side_effect=AssertionError("unexpected tensor hash")):
                with self.assertRaises(ValueError):
                    sender.send_message({}, [TensorBuffer("float32", (2,), b"tiny")])
                sender.send_message({"kind": "valid"}, [TensorBuffer("uint8", (3,), b"abc")])
                metadata, tensors, _ = receiver.recv_message(tensor_format="bytes")
                self.assertEqual(metadata["kind"], "valid")
                self.assertEqual(tensors[0].data, b"abc")
                encoded, _, _ = _encode({}, [TensorBuffer("uint8", (1,), b"x")], "none")
                self.assertNotIn("sha256", json.loads(encoded)["tensors"][0])

        def test_concurrent_whole_frame_writes(self):
            sender, receiver = self.pair(fragmented=True)
            barrier = threading.Barrier(3)
            errors, received = [], []

            def write_many(writer):
                try:
                    barrier.wait()
                    for index in range(4):
                        sender.send_message({"writer": writer, "index": index},
                            [TensorBuffer("uint8", (257,), bytes([writer * 10 + index]) * 257)])
                except BaseException as exc:
                    errors.append(exc)

            def read_many():
                try:
                    for _ in range(8):
                        received.append(receiver.recv_message(tensor_format="bytes"))
                except BaseException as exc:
                    errors.append(exc)

            writers = [threading.Thread(target=write_many, args=(writer,)) for writer in (1, 2)]
            reader = threading.Thread(target=read_many)
            workers = writers + [reader]
            try:
                for worker in workers:
                    worker.start()
                barrier.wait()
                for worker in workers:
                    worker.join(10)  # Deadlock safeguard, not a performance assertion.
                    self.assertFalse(worker.is_alive(), "whole-frame transfer deadlock")
                self.assertEqual(errors, [])
                self.assertEqual(len(received), 8)
                self.assertEqual({(m["writer"], m["index"]) for m, _, _ in received},
                                 {(w, i) for w in (1, 2) for i in range(4)})
                for metadata, tensors, _ in received:
                    self.assertEqual(tensors[0].data, bytes([metadata["writer"] * 10 + metadata["index"]]) * 257)
            finally:
                sender.close()
                receiver.close()
                for worker in workers:
                    worker.join(5)

        def test_invalid_shapes_and_combined_bound_reject_before_payload_read(self):
            invalid_lists = [
                [{"dtype": "uint8", "shape": [-1], "nbytes": 0}],
                [{"dtype": "uint8", "shape": [True], "nbytes": 1}],
                [{"dtype": "uint8", "shape": [0, MAX_PAYLOAD_BYTES, MAX_PAYLOAD_BYTES, 256], "nbytes": 0}],
                [{"dtype": "unknown", "shape": [1], "nbytes": 1}],
                [{"dtype": "float32", "shape": [1], "nbytes": 2}],
                [{"dtype": "uint8", "shape": [MAX_PAYLOAD_BYTES], "nbytes": MAX_PAYLOAD_BYTES}] * 2,
            ]
            for descriptors in invalid_lists:
                with self.subTest(descriptors=descriptors):
                    sender, receiver = self.pair()
                    packet, header_length = raw_frame({}, descriptors)
                    sender._socket.sendall(packet)
                    with patch.object(sys.modules[__name__], "_recv_exact", wraps=_recv_exact) as reads:
                        with self.assertRaises(FrameProtocolError):
                            receiver.recv_message(tensor_format="bytes")
                        self.assertEqual([call.args[1] for call in reads.call_args_list], [4, header_length])
                    self.assertEqual(receiver.fileno(), -1)

        def test_truncation_at_length_header_and_payload_closes_connection(self):
            descriptor = {"dtype": "uint8", "shape": [4], "nbytes": 4}
            complete, header_length = raw_frame({}, [descriptor], payload=b"abcd")
            for truncated in (complete[:2], complete[:4 + header_length // 2], complete[:-2]):
                with self.subTest(length=len(truncated)):
                    sender, receiver = self.pair()
                    sender._socket.sendall(truncated)
                    sender.shutdown(socket.SHUT_WR)
                    with self.assertRaises(EOFError):
                        receiver.recv_message(tensor_format="bytes")
                    self.assertEqual(receiver.fileno(), -1)

        def test_integrity_mode_mismatch_precedes_payload_allocation(self):
            raw, peer = socket.socketpair()
            raw.settimeout(5)
            peer.settimeout(5)
            receiver = FramedConnection(raw, integrity_mode="sha256")
            self.addCleanup(receiver.close)
            self.addCleanup(peer.close)
            packet, header_length = raw_frame({}, [{"dtype": "uint8", "shape": [4], "nbytes": 4}])
            peer.sendall(packet)
            with patch.object(sys.modules[__name__], "_recv_exact", wraps=_recv_exact) as reads:
                with self.assertRaisesRegex(FrameProtocolError, "mode mismatch"):
                    receiver.recv_message(tensor_format="bytes")
                self.assertEqual([call.args[1] for call in reads.call_args_list], [4, header_length])

        def test_diagnostic_checksum_detects_corruption(self):
            sender, receiver = self.pair("sha256")
            descriptor = {"dtype": "uint8", "shape": [4], "nbytes": 4, "sha256": _digest(b"good")}
            packet, _ = raw_frame({}, [descriptor], mode="sha256", payload=b"evil")
            sender._socket.sendall(packet)
            with self.assertRaisesRegex(FrameProtocolError, "SHA256 mismatch"):
                receiver.recv_message(tensor_format="bytes")

        def test_supplied_paced_socket_counts_every_frame_byte(self):
            from e1_transport import PacedSocket, RateLimiter
            raw, peer = socket.socketpair()
            raw.settimeout(5)
            peer.settimeout(5)
            limiter = RateLimiter(0, chunk_bytes=7)
            paced = PacedSocket(raw, limiter, "framed")
            sender, receiver = FramedConnection(paced), FramedConnection(peer)
            self.addCleanup(sender.close)
            self.addCleanup(receiver.close)
            stats = sender.send_message({"kind": "request"}, [TensorBuffer("uint8", (3,), b"abc")])
            metadata, tensors, received = receiver.recv_message(tensor_format="bytes")
            self.assertEqual(metadata["kind"], "request")
            self.assertEqual(tensors[0].data, b"abc")
            self.assertEqual(paced.snapshot()["sent_bytes"], stats["application_frame_bytes"])
            self.assertEqual(limiter.snapshot()["total"]["reserved_bytes"], stats["application_frame_bytes"])
            self.assertEqual(received["application_frame_bytes"], stats["application_frame_bytes"])

        def test_header_bounds_and_nonstandard_json_reject(self):
            invalid_headers = [b'{"protocol":"a","protocol":"b"}',
                               b'{"value":NaN}', b'not-json']
            for header in invalid_headers:
                sender, receiver = self.pair()
                sender._socket.sendall(struct.pack("!I", len(header)) + header)
                with self.assertRaises(FrameProtocolError):
                    receiver.recv_message(tensor_format="bytes")
            for length in (0, MAX_HEADER_BYTES + 1):
                sender, receiver = self.pair()
                sender._socket.sendall(struct.pack("!I", length))
                with self.assertRaises(FrameProtocolError):
                    receiver.recv_message(tensor_format="bytes")

        @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "local torch is unavailable; no dependency installed")
        def test_native_cpu_torch_all_supported_dtypes(self):
            import torch
            for mode in ("none", "sha256"):
                sender, receiver = self.pair(mode)
                inputs = [torch.arange(6, dtype=torch.int64).reshape(2, 3).to(getattr(torch, dtype))
                          for dtype in DTYPE_BYTES]
                inputs.extend([torch.tensor(7, dtype=torch.int64), torch.empty((0, 3), dtype=torch.bfloat16)])
                sender.send_message({"kind": "torch-cpu"}, inputs)
                _, outputs, _ = receiver.recv_message()
                for original, restored in zip(inputs, outputs):
                    self.assertEqual(restored.device.type, "cpu")
                    self.assertEqual(restored.dtype, original.dtype)
                    self.assertEqual(restored.shape, original.shape)
                    self.assertTrue(torch.equal(restored, original))
                    self.assertEqual(restored.reshape(-1).view(torch.uint8).numpy().tobytes(),
                                     original.reshape(-1).view(torch.uint8).numpy().tobytes())
                sparse = torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([2.]), (2, 2))
                with self.assertRaises(TypeError):
                    sender.send_message({}, [sparse])

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CodecTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run CPU socketpair tests; torch tests skip if unavailable")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        parser.print_help()
