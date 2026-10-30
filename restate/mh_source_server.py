"""E6/E7 holder: the reviewed E4 holder server plus holder-side decode.

Only two things differ from E4SourceServer: decode specs may request the
"holder" action (holder-side greedy decode streaming tokens to the target), and
such a job opens only its RPC port because holder decode never uses the bulk
channel. Arm/epoch/settle/disarm, budgets, background load and prefix ownership
are inherited unchanged.
"""
from __future__ import annotations

import copy
import threading
import time
import traceback

from e4_source_server import E4SourceServer, _TransferJob, _tokens, require, transfer_only

ACTIONS = ("remote", "repair", "holder")


class MHSourceServer(E4SourceServer):
    def _specs(self, command):
        require(command.get("trial_id") == self.trial["trial_id"], "Batch trial identity differs")
        requests, transfers = command.get("requests"), command.get("transfers")
        require(isinstance(requests, list) and isinstance(transfers, list), "Explicit decode/transfer lists required")
        require(len(requests) == len(self.trial["requests"]) and len(transfers) == len(self.trial["transfer_jobs"]),
                "Every frozen offered job must remain in denominator")
        jobs, ids = [], set()
        for spec, expected in zip(requests, self.trial["requests"]):
            allowed = {"request_id", "role", "prefix_key", "suffix_ids", "generated", "action", "initial_frontier"}
            require(set(spec) == allowed, "Unexpected decode fields/controller injection")
            require(spec["role"] == expected["role"] and spec["prefix_key"] == expected["prefix_key"]
                    and spec["action"] == expected["action"] and spec["action"] in ACTIONS, "Decode role/action/prefix differs")
            suffix = _tokens(spec["suffix_ids"], self.engine.model.config.vocab_size, "suffix")
            require(len(suffix) == expected["suffix_tokens"] == 16
                    and spec["generated"] == expected["executor_only"]["realized_stop_tokens"] == 64
                    and spec["initial_frontier"] == 0, "Frozen suffix/stop/frontier changed")
            jobs.append(("decode", spec, expected))
        for spec, expected in zip(transfers, self.trial["transfer_jobs"]):
            transfer_only.validate_spec(spec)
            require(spec["prefix_key"] == expected["prefix_key"] and spec["prefix_tokens"] == expected["prefix_tokens"]
                    and spec["diagnostic"] is self.diagnostic and spec["cpu_component_only"] is self.cpu_component,
                    "Transfer prefix/scope differs")
            jobs.append(("transfer_only", spec, expected))
        for kind, spec, expected in jobs:
            key = spec["request_id"]
            require(isinstance(key, str) and key and len(key) <= 1024 and key not in ids, "Unique actual request IDs required")
            ids.add(key)
        return jobs

    def batch(self, command):
        require(self.budget is not None and self.epoch is not None and not self.batch_started,
                "Exactly one batch after native epoch required")
        self.batch_started = True
        report = self.current_report
        threads = []
        barrier = None
        report["batch_received_local_s"] = time.monotonic()
        ports = {}
        outcomes = report["outcomes"]
        try:
            specs = self._specs(command)
            for i, (kind, spec, expected) in enumerate(specs):
                endpoint = {"rpc": self.base_port + 10 + 2 * i, "bulk": self.base_port + 11 + 2 * i}
                if kind == "decode" and spec["action"] == "holder":
                    endpoint = {"rpc": endpoint["rpc"]}
                ports[spec["request_id"]] = endpoint
                outcome = outcomes[i]
                outcome["request_id"] = spec["request_id"]
                if kind == "decode":
                    job = self.deps.Request(self.engine, self.gpu, self.prefixes[spec["prefix_key"]], copy.deepcopy(spec),
                                            endpoint, self.budget, self.remaining(), self.deadline, "none")
                else:
                    job = _TransferJob(self, copy.deepcopy(spec), endpoint)
                self.active.append(job)
            barrier = threading.Barrier(len(self.active) + 1)

            def worker(job, outcome):
                outcome["thread_arrived_local_s"] = time.monotonic()
                try:
                    barrier.wait(timeout=self.remaining())
                    outcome["released_local_s"] = time.monotonic()
                    outcome["status"] = "running"
                    job.run()
                    outcome["raw"] = copy.deepcopy(job.report)
                    outcome["passed"] = job.report.get("passed") is True
                    outcome["status"] = "passed" if outcome["passed"] else "failed"
                except BaseException as error:
                    outcome.update(passed=False, status="failed", error=repr(error), traceback=traceback.format_exc(),
                                   raw=copy.deepcopy(job.report))
                finally:
                    outcome["thread_finished_local_s"] = time.monotonic()

            for job, outcome in zip(self.active, outcomes):
                thread = threading.Thread(target=worker, args=(job, outcome), name="MH-source-" + job.request_id, daemon=True)
                job.thread = thread
                threads.append(thread)
                thread.start()
            report["ready_send_started_local_s"] = time.monotonic()
            self.control.send_message(dict(kind="batch_ready", trial_id=self.trial["trial_id"], ports=ports,
                                           decoded_integrity_mode="none",
                                           transfer_integrity_mode="sha256" if self.diagnostic else "none",
                                           source_epoch_s=self.epoch))
            report["ready_sent_local_s"] = time.monotonic()
            barrier.wait(timeout=self.remaining())
            report["barrier_returned_local_s"] = time.monotonic()
            while any(t.is_alive() for t in threads):
                if any(x["status"] == "failed" for x in outcomes):
                    raise RuntimeError("Source job failed")
                for thread in threads:
                    thread.join(min(.02, self.remaining()))
        except BaseException as error:
            report["errors"].append(dict(scope="batch", error=repr(error), traceback=traceback.format_exc()))
            if barrier is not None:
                barrier.abort()
            self.abort_active("MH source batch failure/deadline")
        finally:
            deadline = time.monotonic() + 30
            for thread in threads:
                thread.join(max(0., deadline - time.monotonic()))
            report["alive_request_threads"] = [t.name for t in self._live_threads()]
            for outcome in outcomes:
                if outcome["status"] == "offered":
                    outcome["status"] = "not_started"
            report["passed"] = not report["errors"] and not report["alive_request_threads"] and all(x["passed"] for x in outcomes)
            report["source_jobs_finished_local_s"] = time.monotonic()
            report["gpu_admission"] = self.gpu.snapshot()
            if not report["alive_request_threads"]:
                self.active.clear()
        compact = [dict(request_id=x["request_id"], role=x["role"], kind=x["kind"], passed=x["passed"], status=x["status"],
                        source_finished_local_s=x.get("thread_finished_local_s"), cleanup=x.get("raw", {}).get("cleanup"),
                        remote_calls=x.get("raw", {}).get("remote_calls")) for x in outcomes]
        self.control.send_message(dict(kind="batch_done", passed=report["passed"], trial_id=self.trial["trial_id"],
                                       requests=compact, offered_count=len(outcomes),
                                       alive_request_threads=report["alive_request_threads"], GPU_performance_acceptance=False,
                                       full_reports_location="full_trial_disarmed.trial_report.outcomes[].raw"))
        report["batch_done_sent_local_s"] = time.monotonic()
        return report
