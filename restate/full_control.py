"""Trial-scoped control frames sharing the request application-byte budget.

Arm/disarm only between complete frames, with both peers handshaken outside the
request. No socket is closed on disarm: its owning FramedConnection retains it.
Actual socket counts and complete-frame counts are separately reconciled.
"""
import socket
import hashlib
import json
from e1_wire import FramedConnection, TensorBuffer


class _BorrowedSocket:
    """The outer session owns this socket; a trial budget cannot close it."""
    def __init__(self, sock):
        self.sock = sock

    def __getattr__(self, name):
        return getattr(self.sock, name)

    def shutdown(self, how):
        pass

    def close(self):
        pass


class TrialControl(FramedConnection):
    def __init__(self, sock):
        super().__init__(sock, integrity_mode="none")
        self.raw_socket = sock
        self.paced = None
        self.frames = None

    def arm(self, coordinator, stream_id="outer-control"):
        if self.paced is not None:
            raise ValueError("Control already armed")
        self.paced = coordinator.wrap(_BorrowedSocket(self.raw_socket), stream_id, "rpc")
        self._socket = self.paced
        self.frames = {"sent": 0, "received": 0, "sent_frames": 0, "received_frames": 0}

    def send_message(self, metadata, tensors=()):
        if metadata.get("kind") in ("full_initialized", "full_trial_disarmed"):
            if self.paced is not None or tensors:
                raise ValueError("Large audit reports must be outside the armed request")
            raw = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode()
            if len(raw) > 64 * 1024 * 1024:
                raise ValueError("Untimed audit report exceeds 64 MiB bound")
            stats = super().send_message({"kind": "e1_untimed_audit_report", "report_kind": metadata["kind"],
                "json_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
                [TensorBuffer("uint8", (len(raw),), raw)])
        else:
            stats = super().send_message(metadata, tensors)
        if self.paced is not None:
            self.frames["sent"] += stats["application_frame_bytes"]
            self.frames["sent_frames"] += 1
        return stats

    def recv_message(self, **kwargs):
        # This channel carries metadata only, apart from explicitly bounded
        # untimed audit blobs. Avoid importing torch just to decode those blobs.
        kwargs["tensor_format"] = "bytes"
        message, tensors, stats = super().recv_message(**kwargs)
        if message.get("kind") == "e1_untimed_audit_report":
            if self.paced is not None or len(tensors) != 1 or message.get("report_kind") not in ("full_initialized", "full_trial_disarmed"):
                raise ValueError("Unexpected/armed untimed audit report")
            tensor = tensors[0]
            raw = bytes(tensor.data)
            if tensor.dtype != "uint8" or tuple(tensor.shape) != (len(raw),) or not 0 < len(raw) <= 64 * 1024 * 1024:
                raise ValueError("Malformed audit payload")
            if len(raw) != message["json_bytes"] or hashlib.sha256(raw).hexdigest() != message["sha256"]:
                raise ValueError("Audit payload identity changed")
            decoded = json.loads(raw)
            if decoded.get("kind") != message["report_kind"]:
                raise ValueError("Audit report kind changed")
            message, tensors = decoded, []
        if self.paced is not None:
            self.frames["received"] += stats["application_frame_bytes"]
            self.frames["received_frames"] += 1
        return message, tensors, stats

    def disarm(self):
        if self.paced is None:
            raise ValueError("Control is not armed")
        actual = self.paced.snapshot()
        result = {"frames": dict(self.frames), "socket": actual,
                  "passed": actual["sent_bytes"] == self.frames["sent"]
                            and actual["received_bytes"] == self.frames["received"],
                  "physical_wire_bytes": None}
        self._socket = self.raw_socket
        self.paced = self.frames = None
        if not result["passed"]:
            raise RuntimeError("Trial control frame/socket accounting mismatch")
        return result

    def close(self):
        try:
            super().close()
        finally:
            try:
                self.raw_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.raw_socket.close()


def reconcile_control(destination, holder):
    if not destination["passed"] or not holder["passed"]:
        raise ValueError("Control did not settle")
    left, right = destination["frames"], holder["frames"]
    for field, other in (("sent", "received"), ("received", "sent"),
                         ("sent_frames", "received_frames"), ("received_frames", "sent_frames")):
        if left[field] != right[other]:
            raise ValueError("Control peer accounting mismatch: " + field)
    return {"passed": True, "scope": "all armed control frames and actual socket bytes",
            "destination": left, "holder": right, "physical_wire_bytes": None}
