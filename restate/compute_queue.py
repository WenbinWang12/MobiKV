"""FIFO admission of ready CUDA computation, with completion before release.

One queue belongs to one selected GPU and one resident model. Whole HF prefill
and local-token forwards are nonpreemptive operations. Remote-token forwards
release admission at each blocking RPC, then rejoin FIFO; transport copies may
run outside this compute queue. This is application scheduling, not CUDA kernel
preemption. The CPU backend exists solely for orchestration tests.
"""
from contextlib import contextmanager, nullcontext
from collections import deque
import copy
import threading
import time


class ComputeCancelled(RuntimeError):
    pass


class CudaBackend:
    kind = "cuda_stream_completion"

    def __init__(self, torch, device):
        self.torch, self.device = torch, device

    @contextmanager
    def execution(self):
        torch = self.torch
        with torch.cuda.device(self.device), torch.cuda.stream(torch.cuda.default_stream(self.device)), torch.inference_mode():
            yield

    def complete(self):
        # Only the compute stream: migration intentionally owns a separate stream.
        self.torch.cuda.default_stream(self.device).synchronize()


class CPUFixtureBackend:
    kind = "cpu_fixture_not_gpu_evidence"

    def execution(self):
        return nullcontext()

    def complete(self):
        pass


class ComputeQueue:
    def __init__(self, backend):
        if not callable(getattr(backend, "execution", None)) or not callable(getattr(backend, "complete", None)):
            raise TypeError("A completion-aware execution backend is required")
        self.backend = backend
        self._condition, self._waiting = threading.Condition(), deque()
        self._active, self._next, self._closed = None, 0, False
        self._local = threading.local()
        self._records, self._peak = [], 0

    @classmethod
    def for_engine(cls, engine):
        return cls(CudaBackend(engine.torch, engine.device))

    @contextmanager
    def bind(self, owner, report=None, deadline=None, cancel_event=None):
        if not isinstance(owner, str) or not owner:
            raise ValueError("A nonempty request/phase owner is required")
        previous = getattr(self._local, "binding", None)
        self._local.binding = (owner, report, deadline, cancel_event)
        try:
            yield
        finally:
            self._local.binding = previous

    @contextmanager
    def operation(self, kind):
        if not isinstance(kind, str) or not kind:
            raise ValueError("An operation kind is required")
        existing = getattr(self._local, "lease", None)
        if existing is not None:
            # Setup wrappers and HolderAttention may share one outer admission.
            if not existing.active:
                raise RuntimeError("Cannot nest compute inside an inactive lease")
            yield existing
            return
        binding = getattr(self._local, "binding", None) or (threading.current_thread().name, None, None, None)
        lease = _Lease(self, kind, binding)
        self._local.lease = lease
        try:
            lease.acquire()
            yield lease
        except BaseException as error:
            lease.release(error)
            raise
        else:
            lease.release()
        finally:
            self._local.lease = None

    def call_unqueued(self, callback, *args, **kwargs):
        """Finish compute, yield admission for I/O, then rejoin the FIFO tail.

        callback may perform transport copies, but MUST NOT execute model math.
        No queue lock or compute-stream context is held while callback blocks.
        """
        lease = getattr(self._local, "lease", None)
        if lease is None:
            return callback(*args, **kwargs)
        lease.release(status="yielded_for_io")
        self._local.lease = None
        try:
            result = callback(*args, **kwargs)
        except BaseException:
            raise
        else:
            lease.segment += 1
            lease.acquire()
            return result
        finally:
            self._local.lease = lease

    def cursor(self):
        with self._condition:
            return self._next

    def state(self):
        """Small control-plane counters; never copy or transmit operation history."""
        with self._condition:
            return {"backend": self.backend.kind, "closed": self._closed,
                "next_sequence": self._next,
                "active_gpu_operations": int(self._active is not None),
                "queued_gpu_operations": len(self._waiting),
                "peak_queued_gpu_operations": self._peak,
                "model_execution_serialized": True, "fifo": True,
                "transport_copies_outside_compute_queue": True,
                "service_scope": "CPU wall interval ending after CUDA-stream completion; not CUDA-event/kernel time"}

    def snapshot(self, since_sequence=0):
        if type(since_sequence) is not int or since_sequence < 0:
            raise ValueError("since_sequence must be a nonnegative integer")
        with self._condition:
            return {"backend": self.backend.kind, "closed": self._closed,
                "next_sequence": self._next, "since_sequence": since_sequence,
                "active_gpu_operations": int(self._active is not None),
                "queued_gpu_operations": len(self._waiting), "peak_queued_gpu_operations": self._peak,
                "model_execution_serialized": True, "fifo": True,
                "service_scope": "monotonic admission-to-CUDA-stream-completion wall time; not kernel time",
                "granularity": "HF whole-prefix; local whole-token; remote compute segments separated by RPC; greedy selection",
                "transport_copies_outside_compute_queue": True,
                "operations": copy.deepcopy([r for r in self._records if r["ticket"] >= since_sequence])}

    def clear_completed_events(self):
        """Caller must save reports first; stable ticket numbering is never reset."""
        with self._condition:
            if self._active is not None or self._waiting:
                raise RuntimeError("Cannot clear events while compute is active or queued")
            removed = len(self._records)
            self._records.clear()
            return {"removed_events": removed, "next_sequence": self._next}

    def close(self, timeout=30):
        """Reject/cancel waiting work; an already admitted operation must finish."""
        deadline = time.monotonic() + timeout
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            while self._active is not None or self._waiting:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError("Compute queue did not drain")
                self._condition.wait(min(left, 0.05))
        return self.snapshot()


class _Lease:
    def __init__(self, queue, kind, binding):
        self.queue, self.kind, self.binding = queue, kind, binding
        self.active, self.segment, self.context, self.record = False, 0, None, None

    def acquire(self):
        q = self.queue
        owner, report, deadline, cancelled = self.binding
        with q._condition:
            ticket = q._next
            q._next += 1
            record = {"ticket": ticket, "owner": owner, "kind": self.kind, "segment": self.segment,
                "arrival_monotonic_s": time.monotonic(), "start_monotonic_s": None,
                "end_monotonic_s": None, "queue_wait_s": None, "service_s": None,
                "service_start_perf_counter_s": None, "service_end_perf_counter_s": None,
                "status": "pending", "error": None, "completion_backend": q.backend.kind}
            q._records.append(record)
            q._waiting.append(ticket)
            q._peak = max(q._peak, len(q._waiting))
            try:
                while True:
                    now = time.monotonic()
                    if q._closed or (cancelled is not None and cancelled.is_set()):
                        raise ComputeCancelled("Compute admission cancelled")
                    if deadline is not None and now >= deadline:
                        raise TimeoutError("Compute admission exceeded deadline")
                    if q._active is None and q._waiting[0] == ticket:
                        q._waiting.popleft()
                        q._active = ticket
                        self.active = True
                        self._service_started = time.perf_counter()
                        record.update(start_monotonic_s=now, queue_wait_s=now - record["arrival_monotonic_s"],
                                      service_start_perf_counter_s=self._service_started, status="running")
                        break
                    q._condition.wait(0.05 if deadline is None else min(0.05, max(0, deadline - now)))
            except BaseException as error:
                if ticket in q._waiting:
                    q._waiting.remove(ticket)
                record.update(status="cancelled" if isinstance(error, ComputeCancelled) else "failed",
                              error=f"{type(error).__name__}: {error}", end_monotonic_s=time.monotonic())
                q._condition.notify_all()
                raise
        self.record = record
        self.context = q.backend.execution()
        try:
            self.context.__enter__()
        except BaseException as error:
            self.context = None
            self.release(error)
            raise
        if report is not None:
            report["gpu_queue_wait_s"] = report.get("gpu_queue_wait_s", 0.0) + record["queue_wait_s"]
            report["gpu_admissions"] = report.get("gpu_admissions", 0) + 1

    def release(self, error=None, status="completed"):
        if not self.active:
            return
        completion_error = None
        try:
            self.queue.backend.complete()
        except BaseException as caught:
            completion_error = caught
        finally:
            if self.context is not None:
                try:
                    self.context.__exit__(None, None, None)
                except BaseException as caught:
                    completion_error = completion_error or caught
            end = time.monotonic()
            effective = error or completion_error
            with self.queue._condition:
                service_end = time.perf_counter()
                self.record.update(end_monotonic_s=end, service_s=service_end - self._service_started,
                    service_end_perf_counter_s=service_end,
                    service_duration_clock="perf_counter",
                    status="failed" if effective else status,
                    error=None if effective is None else f"{type(effective).__name__}: {effective}")
                report = self.binding[1]
                if report is not None:
                    report["gpu_service_s"] = report.get("gpu_service_s", 0.0) + self.record["service_s"]
                self.queue._active = None
                self.active, self.context = False, None
                self.queue._condition.notify_all()
        if completion_error is not None and error is None:
            raise completion_error
