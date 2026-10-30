# E3-specific timed-arrival adapter copied from frozen E1 background runtime.
# Same actual cached local model/queue execution; only zero lead-in and the
# earlier actual admission anchor differ. Not used by E1/E2.
"""Real cached-local arrivals on a resident model; no synthetic service times.

Poisson arrival schedules are seeded and open-loop. One background worker owns
each request's delta and competes with foreground at ComputeQueue operations.
Calibration is isolated: 2 warmups, 10 measured jobs; rate is 0.5 / mean actual
CUDA-completion-bounded service. CPU fixture jobs cannot certify GPU service.
"""
import copy
import hashlib
import json
import math
import random
import threading
import time
import weakref


class BackgroundLoad:
    """Arrival/drain controller. job(report) must execute actual queued work."""
    def __init__(self, queue, job, workload):
        self.queue, self.job, self.workload = queue, job, dict(workload)
        self.calibration = None
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._records, self._pending, self._threads = [], [], []
        self._running, self._producer_done, self._error = False, True, None
        self._trial = None

    def _execute(self, record):
        record["start_monotonic_s"] = time.monotonic()
        record["request_queue_wait_s"] = record["start_monotonic_s"] - record["arrival_monotonic_s"]
        record["status"] = "running"
        try:
            with self.queue.bind(record["job_id"], record):
                result = self.job(record)
            if record.get("gpu_admissions", 0) <= 0 or record.get("gpu_service_s", 0) <= 0:
                raise RuntimeError("Background job executed no completion-bounded queued work")
            record["result"] = result
            record["status"] = "completed"
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["end_monotonic_s"] = time.monotonic()
            record["request_wall_s"] = record["end_monotonic_s"] - record["start_monotonic_s"]

    @staticmethod
    def _record(job_id, scheduled, actual, phase):
        return {"job_id": job_id, "phase": phase, "scheduled_monotonic_s": scheduled,
            "arrival_monotonic_s": actual, "arrival_lateness_s": actual - scheduled,
            "start_monotonic_s": None, "end_monotonic_s": None, "request_queue_wait_s": None,
            "request_wall_s": None, "gpu_queue_wait_s": 0.0, "gpu_service_s": 0.0,
            "gpu_admissions": 0, "status": "pending", "error": None}

    def calibrate(self):
        if self._running:
            raise RuntimeError("Cannot calibrate during a trial")
        initial = self.queue.snapshot()
        if initial["active_gpu_operations"] or initial["queued_gpu_operations"]:
            raise RuntimeError("Calibration requires an idle isolated compute queue")
        records = []
        first_ticket = initial["next_sequence"]
        for index in range(12):
            phase = "warmup" if index < 2 else "calibration"
            now = time.monotonic()
            record = self._record(f"background-{phase}-{index}", now, now, phase)
            records.append(record)
            self._execute(record)
        operations = self.queue.snapshot(since_sequence=first_ticket)["operations"]
        owners = {record["job_id"] for record in records}
        if any(operation["owner"] not in owners for operation in operations):
            raise RuntimeError("Calibration was not isolated from other compute")
        measured = records[2:]
        mean = sum(record["gpu_service_s"] for record in measured) / 10
        if not math.isfinite(mean) or mean <= 0:
            raise RuntimeError("Invalid measured background service")
        self.calibration = {"warmup_jobs": 2, "measured_jobs": 10, "mean_service_s": mean,
            "mean_request_wall_s": sum(record["request_wall_s"] for record in measured) / 10,
            "service_scope": "sum of admitted compute wall intervals ending after stream synchronization",
            "backend": self.queue.backend.kind, "GPU_service_evidence": self.queue.backend.kind == "cuda_stream_completion",
            "records": records, "operations": operations, "workload": self.workload}
        return copy.deepcopy(self.calibration)

    def start(self, seed=2026110701, rho=0.5, lead_service_multiples=0, acceptance_monotonic_s=None):
        if self._running or self.calibration is None:
            raise RuntimeError("Calibrate first and drain the previous trial")
        if type(seed) is not int or rho != 0.5 or lead_service_multiples != 0:
            raise ValueError("E3 requires integer seed, rho=0.5 and zero lead-in")
        self._stop.clear()
        self._records, self._pending, self._error = [], [], None
        self._producer_done, self._running = False, True
        began = time.monotonic()
        if type(acceptance_monotonic_s) not in (int,float) or not math.isfinite(acceptance_monotonic_s) or not 0 <= acceptance_monotonic_s <= began:
            self._running = False
            raise ValueError("E3 actual local acceptance epoch required before load arrival start")
        mean = self.calibration["mean_service_s"]
        self._trial = {"seed": seed, "rho": rho, "lambda_jobs_per_s": rho / mean,
            "started_monotonic_s": began, "lead_service_multiples": 0, "lead_in_s": 0.0,
            "workload": self.workload, "backend": self.queue.backend.kind,
            "queue_first_sequence": self.queue.cursor(), "acceptance_monotonic_s": acceptance_monotonic_s}

        def produce():
            rng, scheduled = random.Random(seed), began
            index = 0
            try:
                while not self._stop.is_set():
                    scheduled += rng.expovariate(rho / mean)
                    if self._stop.wait(max(0, scheduled - time.monotonic())):
                        break
                    with self._condition:
                        record = self._record(f"background-job-{index}", scheduled, time.monotonic(), "trial")
                        self._records.append(record)
                        self._pending.append(record)
                        index += 1
                        self._condition.notify_all()
            except BaseException as error:
                self._error = f"{type(error).__name__}: {error}"
                self._stop.set()
            finally:
                with self._condition:
                    self._producer_done = True
                    self._condition.notify_all()

        def consume():
            while True:
                with self._condition:
                    while not self._pending and not self._producer_done:
                        self._condition.wait(0.05)
                    if not self._pending:
                        return
                    record = self._pending.pop(0)
                try:
                    self._execute(record)
                except BaseException as error:
                    self._error = f"{type(error).__name__}: {error}"
                    self._stop.set()
                    # Retain every not-started arrival explicitly, without inventing service.
                    return

        self._threads = [threading.Thread(target=produce, name="background-arrivals", daemon=True),
                         threading.Thread(target=consume, name="background-worker", daemon=True)]
        for thread in self._threads:
            thread.start()
        if self._error:
            self.stop_and_drain()
            raise RuntimeError("Background lead-in failed: " + self._error)
        self._trial["lead_in_finished_monotonic_s"] = time.monotonic()
        return copy.deepcopy(self._trial)

    def snapshot(self):
        with self._condition:
            records = copy.deepcopy(self._records)
            return {**(self._trial or {}), "running": self._running, "records": records,
                "arrivals": len(records), "completed": sum(r["status"] == "completed" for r in records),
                "failed": sum(r["status"] == "failed" for r in records),
                "pending": sum(r["status"] in ("pending", "running") for r in records),
                "error": self._error, "threads_alive": [t.name for t in self._threads if t.is_alive()]}

    def mark_acceptance(self, monotonic_s=None):
        """A local-clock observation, never a claim of synchronized peer clocks."""
        if not self._running or self._trial["acceptance_monotonic_s"] is not None:
            raise RuntimeError("Acceptance anchor needs one running, unmarked trial")
        now = time.monotonic() if monotonic_s is None else monotonic_s
        if (isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now)
            or not self._trial["started_monotonic_s"] <= now <= time.monotonic()):
            raise ValueError("Acceptance anchor must be an observed local monotonic timestamp")
        self._trial["acceptance_monotonic_s"] = now
        return {"acceptance_monotonic_s": now, "scope": "this process local monotonic clock only"}

    def stop_and_drain(self, timeout=120):
        if not self._running:
            raise RuntimeError("No running background trial")
        stopped, deadline = time.monotonic(), time.monotonic() + timeout
        self._stop.set()
        for thread in self._threads:
            thread.join(max(0, deadline - time.monotonic()))
        self._running = any(thread.is_alive() for thread in self._threads)
        self._trial.update(stop_requested_monotonic_s=stopped, drained_monotonic_s=time.monotonic())
        result = self.snapshot()
        operations = self.queue.snapshot(since_sequence=self._trial["queue_first_sequence"])["operations"]
        result["operations"] = [item for item in operations if item["owner"].startswith("background-job-")]
        result["cleanup"] = {"producer_exited": not self._threads[0].is_alive(),
            "worker_exited": not self._threads[1].is_alive(), "all_accepted_jobs_completed": result["pending"] == 0 and result["failed"] == 0}
        result["passed"] = not self._running and not self._error and all(result["cleanup"].values())
        return result


class CachedLocalBackground(BackgroundLoad):
    """GPU workload P1024/S16/G32 with separate immutable prefix and per-job delta."""
    def __init__(self, engine, queue, prefix_ids, suffix_ids, generated=32):
        if engine.compute_queue is not queue:
            raise ValueError("Foreground engine and background must share exactly one ComputeQueue")
        vocab = engine.model.config.vocab_size
        if (len(prefix_ids) != 1024 or len(suffix_ids) != 16 or generated != 32 or
            any(type(token) is not int or not 0 <= token < vocab for token in list(prefix_ids) + list(suffix_ids))):
            raise ValueError("Background requires valid P1024/S16/G32 token IDs")
        self.engine, self.prefix_ids, self.suffix_ids = engine, list(prefix_ids), list(suffix_ids)
        self.pairs = self.store = self.suffix = None
        self.closed = False
        recipe = {"prefix_ids": self.prefix_ids, "suffix_ids": self.suffix_ids, "generated": 32}
        digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        super().__init__(queue, self._job, {"prefix": 1024, "suffix": 16, "generated": 32,
            "input_sha256": digest, "model_shared": True, "prefix_shared_with_foreground": False,
            "delta_per_request": True, "request_worker_count": 1})

    def prepare(self):
        if self.closed or self.pairs is not None:
            raise RuntimeError("Background already prepared or closed")
        torch = self.engine.torch
        with self.queue.bind("background-prepare"):
            prefix = torch.tensor([self.prefix_ids], dtype=torch.int64, device=self.engine.device)
            self.pairs = self.engine.prefix(prefix)
            self.store = self.engine.store_from_prefix(self.pairs)
            self.suffix = torch.tensor([self.suffix_ids], dtype=torch.int64, device=self.engine.device)
        return dict(self.workload)

    def _job(self, report):
        if self.closed or self.store is None:
            raise RuntimeError("Background prefix is not prepared")
        result = self.engine.decode(self.store, 1024, self.suffix, 32, capture_logits=False)
        if (result["state_report"]["owned_delta_references_released"] is not True or
            result["state_report"]["mode_counts"] != {"local": self.engine.layers * 47, "remote": 0}):
            raise RuntimeError("Background did not execute the full cached-local workload cleanly")
        return {"token_ids": result["token_ids"], "state_report": result["state_report"]}

    def close(self):
        if self._running:
            raise RuntimeError("Stop and drain background before clearing its prefix")
        refs = [] if self.pairs is None else [weakref.ref(tensor) for pair in self.pairs for tensor in pair]
        if self.store is not None:
            self.store.values.clear()
        if self.pairs is not None:
            self.pairs.clear()
        self.pairs = self.store = self.suffix = None
        self.closed = True
        released = all(reference() is None for reference in refs)
        if not released:
            raise RuntimeError("Background prefix tensors remain retained")
        return {"prefix_references_released": released, "background_threads_exited": not self._running}
