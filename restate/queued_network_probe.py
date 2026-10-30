"""Excluded CPU path probe; call only after the experiment queue's idle gate.

The controller may run on any node with SSH access and the shared package path.
No GPU, qdisc, sysctl, or immutable package files are changed. Workers receive a
copy of this module's worker function over SSH's command argument, so this file
does not need to be installed on the other nodes. Their stdin remains open as
an ownership lease: closing it stops that worker, with an independent hard
deadline as a fallback. Each repetition releases all connected senders together.
"""
from __future__ import annotations

import restate_config
import argparse
import hashlib
import inspect
import ipaddress
import json
import math
import shlex
import subprocess
import threading
import time
import uuid
from pathlib import Path

PYTHON = restate_config.PYTHON
PORT = 35998
SSH_CONNECT_TIMEOUT = 10
READY_TIMEOUT = 30.0
BLOCK_BYTES = 16 << 20


def _remote_entry():
    """Self-contained remote worker, deliberately restricted to one process."""
    import importlib
    import json
    import math
    import os
    import socket
    import sys
    import threading
    import time
    import traceback

    cfg = json.loads(sys.argv[1])
    started = threading.Event()
    done = threading.Event()
    print_lock = threading.Lock()

    def emit(event, **values):
        with print_lock:
            print(json.dumps(dict(event=event, token=cfg["token"], **values), allow_nan=False), flush=True)

    def exit_owned(reason, code):
        if not done.is_set():
            # Do not wait for a blocked socket, Python reader thread, or scheduler.
            try:
                emit("aborted", error=reason)
            finally:
                os._exit(code)

    def stdin_lease():
        try:
            for line in sys.stdin:
                if line.strip() == "START":
                    started.set()
        finally:
            exit_owned("controller_stdin_closed", 125)

    def deadline():
        if not done.wait(cfg["timeout_s"]):
            exit_owned("worker_hard_timeout", 124)

    threading.Thread(target=stdin_lease, daemon=True).start()
    threading.Thread(target=deadline, daemon=True).start()

    def measurement(total, elapsed):
        if total <= 0 or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Empty transfer or nonpositive/nonfinite duration")
        return dict(bytes=total, seconds=elapsed, gbps=total * 8 / elapsed / 1e9)

    def receiver():
        rows, errors, threads = [], [], []
        result_lock = threading.Lock()

        def reader(conn, peer):
            total, first = 0, None
            try:
                conn.settimeout(30)
                buf = bytearray(4 << 20)
                while True:
                    n = conn.recv_into(buf)
                    now = time.monotonic()
                    if not n:
                        elapsed = now - first if first is not None else 0.0
                        break
                    if first is None:
                        first = now
                    total += n
                row = dict(peer=peer, **measurement(total, elapsed))
                with result_lock:
                    rows.append(row)
                emit("connection_result", **row)
            except BaseException as exc:
                with result_lock:
                    errors.append(dict(peer=peer, bytes=total, error=repr(exc)))
                traceback.print_exc()
            finally:
                conn.close()

        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((cfg["target"], cfg["port"]))
            listener.listen(len(cfg["holders"]))
            listener.settimeout(30)
            emit("ready", role="receiver", pid=os.getpid())
            accepted = set()
            try:
                for _ in cfg["holders"]:
                    conn, addr = listener.accept()
                    peer = addr[0]
                    if peer not in cfg["holders"] or peer in accepted:
                        conn.close()
                        raise ValueError("Unexpected or duplicate source IP: " + peer)
                    accepted.add(peer)
                    thread = threading.Thread(target=reader, args=(conn, peer), daemon=True)
                    thread.start()
                    threads.append(thread)
                for thread in threads:
                    thread.join()
            except BaseException as exc:
                errors.append(dict(error=repr(exc)))
                traceback.print_exc()
            result = dict(rows=rows, errors=errors, expected=len(cfg["holders"]))
            emit("result", role="receiver", data=result)
            return 0 if not errors and len(rows) == len(cfg["holders"]) else 1

    def sender():
        sys.path.insert(0, cfg["pkg"])
        coordinator_class = importlib.import_module("e4_scheduler").E4BandwidthCoordinator
        coordinator = coordinator_class(cfg["rate"], "fine_fifo", chunk_bytes=cfg["chunk"])
        try:
            with socket.socket() as sock:
                sock.settimeout(30)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                # Force the IP subsequently used to pair receiver results.
                sock.bind((cfg["holder"], 0))
                sock.connect((cfg["target"], cfg["port"]))
                stream = coordinator.wrap(sock, "probe-bulk", "bulk")
                block = memoryview(bytearray(cfg["block_bytes"]))
                emit("ready", role="sender", holder=cfg["holder"], pid=os.getpid())
                if not started.wait(30):
                    raise TimeoutError("Controller did not release sender barrier")
                begin, sent = time.monotonic(), 0
                emit("started", role="sender", holder=cfg["holder"])
                while time.monotonic() - begin < cfg["seconds"]:
                    stream.sendall(block)
                    sent += len(block)
                elapsed = time.monotonic() - begin
                sock.shutdown(socket.SHUT_WR)
                result = dict(holder=cfg["holder"], target=cfg["target"], rate=cfg["rate"],
                              chunk=cfg["chunk"], grants=coordinator.snapshot()["total_grants"],
                              **measurement(sent, elapsed))
                emit("result", role="sender", data=result)
                return 0
        finally:
            coordinator.close()

    try:
        status = receiver() if cfg["role"] == "receiver" else sender()
    except BaseException as exc:
        emit("error", error=repr(exc))
        traceback.print_exc()
        status = 1
    finally:
        done.set()
    raise SystemExit(status)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _ssh_command(host, config):
    source = inspect.getsource(_remote_entry) + "\n_remote_entry()\n"
    remote = shlex.join([PYTHON, "-u", "-B", "-c", source, json.dumps(config, allow_nan=False)])
    return ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
            "-o", "ConnectionAttempts=1", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
            "-o", "LogLevel=ERROR", "root@" + host, remote]


class _Worker:
    def __init__(self, host, config, directory, name):
        self.host, self.config, self.name = host, config, name
        self.events, self.errors, self.threads = [], [], []
        self.timed_out = False
        self.stopped_owned_process = False
        self.condition = threading.Condition()
        self.paths = {stream: Path(directory) / (name + "." + stream + ".txt") for stream in ("stdout", "stderr")}
        self.proc = subprocess.Popen(_ssh_command(host, config), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1)
        for stream in ("stdout", "stderr"):
            thread = threading.Thread(target=self._read, args=(stream,), daemon=True)
            thread.start()
            self.threads.append(thread)

    def _read(self, stream):
        try:
            with self.paths[stream].open("w", encoding="utf-8", newline="") as raw:
                for line in getattr(self.proc, stream):
                    raw.write(line)
                    raw.flush()
                    if stream == "stdout":
                        try:
                            event = json.loads(line)
                            if not isinstance(event, dict) or event.get("token") != self.config["token"]:
                                raise ValueError("Unexpected event or ownership token")
                            with self.condition:
                                self.events.append(event)
                                self.condition.notify_all()
                        except (ValueError, TypeError) as exc:
                            self.errors.append("invalid stdout protocol: " + str(exc))
        except Exception as exc:
            self.errors.append(stream + " capture failed: " + repr(exc))
        finally:
            with self.condition:
                self.condition.notify_all()

    def ready(self, timeout):
        end = time.monotonic() + timeout
        with self.condition:
            while not any(event.get("event") == "ready" for event in self.events):
                if self.proc.poll() is not None:
                    raise RuntimeError(self.name + " exited before ready: " + str(self.proc.returncode))
                remaining = end - time.monotonic()
                if remaining <= 0:
                    self.timed_out = True
                    raise TimeoutError(self.name + " ready timeout")
                self.condition.wait(min(remaining, 0.2))

    def release(self):
        self.proc.stdin.write("START\n")
        self.proc.stdin.flush()

    def finish(self, timeout):
        try:
            self.proc.wait(timeout=max(0.01, timeout))
        except subprocess.TimeoutExpired:
            self.timed_out = True
            raise TimeoutError(self.name + " completion timeout")

    def stop(self):
        # Only close/terminate Popen objects created by this repetition. The
        # remote worker exits on stdin EOF; its hard deadline outlives neither
        # the normal controller completion timeout nor a lost SSH connection.
        if self.proc.poll() is None:
            self.stopped_owned_process = True
            try:
                self.proc.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=3)
        else:
            self.proc.stdin.close()
        for thread in self.threads:
            thread.join(timeout=3)
        if any(thread.is_alive() for thread in self.threads):
            self.errors.append("Output capture thread did not finish")

    def report(self):
        return dict(host=self.host, role=self.config["role"], returncode=self.proc.returncode,
                    timed_out=self.timed_out, stopped_owned_process=self.stopped_owned_process,
                    errors=list(self.errors), events=list(self.events),
                    raw={stream: dict(path=str(path.resolve()), sha256=_sha(path))
                         for stream, path in self.paths.items() if path.exists()})


def _positive_number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return value


def _measurement(row, name):
    if not isinstance(row, dict) or type(row.get("bytes")) is not int or row["bytes"] <= 0:
        raise ValueError(name + ": bytes must be a positive integer")
    duration = _positive_number(row.get("seconds"), name + " seconds")
    gbps = _positive_number(row.get("gbps"), name + " gbps")
    if not math.isclose(gbps, row["bytes"] * 8 / duration / 1e9, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(name + ": gbps does not agree with bytes/duration")
    return gbps * 1e9


def _result(report, name):
    if type(report.get("returncode")) is not int or report["returncode"] != 0 or report.get("timed_out") or report.get("errors"):
        raise ValueError(name + ": process failed, timed out, or output capture failed")
    events = report.get("events", [])
    if any(event.get("event") in ("error", "aborted") for event in events):
        raise ValueError(name + ": remote error/abort event")
    results = [event.get("data") for event in events if event.get("event") == "result"]
    if len(results) != 1 or not isinstance(results[0], dict):
        raise ValueError(name + ": expected exactly one result")
    return results[0]


def _validate_rep(workers, holders, target, rate, chunk, released):
    """Never treat an rc=0 receiver with a missing/crashed reader as success."""
    if not released:
        raise ValueError("All senders must reach the ready barrier before release")
    if set(workers) != {"receiver", *("sender-" + host for host in holders)}:
        raise ValueError("Missing/extra worker report")
    received = _result(workers["receiver"], "receiver")
    rows = received.get("rows")
    if received.get("errors") or received.get("expected") != len(holders) or not isinstance(rows, list) or len(rows) != len(holders):
        raise ValueError("Receiver reported errors or wrong connection count")
    peers = [row.get("peer") for row in rows if isinstance(row, dict)]
    if len(peers) != len(holders) or len(set(peers)) != len(holders) or set(peers) != set(holders):
        raise ValueError("Receiver source IPs must match holders exactly once")
    by_peer = {row["peer"]: row for row in rows}
    achieved = {}
    for host in holders:
        sent = _result(workers["sender-" + host], "sender " + host)
        if (sent.get("holder") != host or sent.get("target") != target or
                sent.get("rate") != rate or sent.get("chunk") != chunk):
            raise ValueError("Sender configuration does not match request: " + host)
        if type(sent.get("grants")) is not int or sent["grants"] <= 0:
            raise ValueError("No positive coordinator grant count: " + host)
        sender_bps = _measurement(sent, "sender " + host)
        receiver_bps = _measurement(by_peer[host], "receiver " + host)
        if sent["bytes"] != by_peer[host]["bytes"]:
            raise ValueError("Sender/receiver bytes differ: " + host)
        achieved[host] = dict(sender_bps=sender_bps, receiver_bps=receiver_bps,
                              achieved_bps=min(sender_bps, receiver_bps), bytes=sent["bytes"])
    return achieved


def _run_rep(pkg, directory, rate, chunk, holders, target, seconds, token):
    directory.mkdir(parents=True, exist_ok=False)
    # A full sendall may run beyond the requested interval by one 16 MiB block.
    transfer_budget = seconds + min(BLOCK_BYTES * 8 / rate, 30.0) + 30.0
    worker_timeout = 2 * READY_TIMEOUT + transfer_budget
    common = dict(pkg=str(pkg.resolve()), rate=rate, chunk=chunk, holders=holders, target=target,
                  seconds=seconds, block_bytes=BLOCK_BYTES, port=PORT, token=token,
                  timeout_s=worker_timeout)
    workers, errors, released = {}, [], False
    try:
        workers["receiver"] = _Worker(target, dict(common, role="receiver"), directory, "receiver")
        workers["receiver"].ready(READY_TIMEOUT)
        for host in holders:
            name = "sender-" + host
            workers[name] = _Worker(host, dict(common, role="sender", holder=host), directory, name)
        ready_deadline = time.monotonic() + READY_TIMEOUT
        for host in holders:
            workers["sender-" + host].ready(max(0.01, ready_deadline - time.monotonic()))
        for host in holders:
            workers["sender-" + host].release()
        released = True
        # Every remote hard timer began before its ready event. This completion
        # deadline therefore falls after all worker hard deadlines, irrespective
        # of remote clock offsets or the order in which SSH sessions connected.
        deadline = time.monotonic() + worker_timeout + 5.0
        for name in [*("sender-" + host for host in holders), "receiver"]:
            workers[name].finish(deadline - time.monotonic())
    except Exception as exc:
        errors.append(repr(exc))
    finally:
        # Concurrent EOF cleanup prevents another sender continuing while we
        # wait for a failed sender's SSH process to exit.
        for worker in workers.values():
            if worker.proc.poll() is None:
                try:
                    worker.proc.stdin.close()
                except (OSError, ValueError):
                    pass
        for worker in workers.values():
            try:
                worker.stop()
            except Exception as exc:
                errors.append("owned worker cleanup failed: " + repr(exc))
    reports = {name: worker.report() for name, worker in workers.items()}
    achieved = {}
    try:
        achieved = _validate_rep(reports, holders, target, rate, chunk, released)
    except (ValueError, TypeError, KeyError) as exc:
        errors.append(str(exc))
    row = dict(passed=not errors, errors=errors, token=token, all_senders_ready_before_release=released,
               achieved=achieved, workers=reports, timeout_s=worker_timeout)
    _write_json(directory / "result.json", row)
    row["result_path"] = str((directory / "result.json").resolve())
    row["result_sha256"] = _sha(directory / "result.json")
    return row


def probe(pkg: Path, out: Path, rate: int, chunk: int, holders: list[str], target: str,
          reps: int = 3, seconds: float = 6) -> dict:
    """Run all repetitions and retain failures; never freeze a rate on partial success.

    `out` must be new or empty. Return `passed`, `errors`, `holder_min_bps`, and
    `global_min_bps`; both minimum fields are null for an incomplete/failed run.
    `successful_holder_min_bps` retains diagnostic minima, never a passing gate.
    The caller owns the GPU/experiment idle gate; this function cannot infer it.
    """
    pkg, out = Path(pkg), Path(out)
    if type(rate) is not int or rate <= 0:
        raise ValueError("rate must be a positive integer bps")
    if type(chunk) is not int or not 16384 <= chunk <= 64 * 1048576:
        raise ValueError("chunk must be an integer in [16384, 64 MiB]")
    if type(reps) is not int or reps <= 0:
        raise ValueError("reps must be a positive integer")
    _positive_number(seconds, "seconds")
    if seconds > 3600:
        raise ValueError("An excluded preflight probe is limited to one hour per repetition")
    if not isinstance(holders, list) or not holders or len(set(holders)) != len(holders):
        raise ValueError("holders must be a nonempty list of unique IPv4 strings")
    for host in [target, *holders]:
        if not isinstance(host, str) or str(ipaddress.IPv4Address(host)) != host:
            raise ValueError("Only canonical IPv4 addresses are accepted")
    if target in holders:
        raise ValueError("Target must differ from each holder")
    source_hashes = {name: _sha(pkg / name) for name in ("e4_scheduler.py", "e1_transport.py")}
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Refusing to overwrite existing probe evidence: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    run_id = "queued-network-" + uuid.uuid4().hex
    request = dict(run_id=run_id, pkg=str(pkg.resolve()), rate_bps=rate, chunk_bytes=chunk,
                   holders=holders, target=target, reps=reps, seconds=seconds, port=PORT,
                   source_sha256=source_hashes, probe_sha256=_sha(__file__),
                   scope="excluded CPU simultaneous application-path probe; no GPU evidence")
    _write_json(out / "request.json", request)
    repetitions, errors = [], []
    for index in range(reps):
        directory = out / ("rep-%02d" % (index + 1))
        try:
            row = _run_rep(pkg, directory, rate, chunk, holders, target, seconds, run_id + "-" + str(index + 1))
        except Exception as exc:
            # Preserve even orchestration errors instead of dropping the rep.
            row = dict(passed=False, errors=["orchestration error: " + repr(exc)], achieved={}, workers={})
            directory.mkdir(parents=True, exist_ok=True)
            _write_json(directory / "orchestration-error.json", row)
        repetitions.append(row)
        errors.extend("rep %d: %s" % (index + 1, error) for error in row.get("errors", []))
        if not row.get("passed") and not row.get("errors"):
            errors.append("rep %d: failed without a diagnostic" % (index + 1))
    passed = not errors and all(row.get("passed") for row in repetitions)
    diagnostic = {}
    for host in holders:
        values = [row["achieved"][host]["achieved_bps"] for row in repetitions
                  if row.get("passed") and host in row.get("achieved", {})]
        diagnostic[host] = min(values) if values else None
    minima = diagnostic if passed else {host: None for host in holders}
    result = dict(**request, passed=passed, error="; ".join(errors) if errors else None, errors=errors,
                  holder_min_bps=minima, global_min_bps=min(minima.values()) if passed else None,
                  successful_holder_min_bps=diagnostic, repetitions=repetitions)
    path = out / "summary.json"
    _write_json(path, result)
    result["summary_path"], result["summary_sha256"] = str(path.resolve()), _sha(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkg", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rate", type=int, required=True)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--holders", nargs="+", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=6.0)
    result = probe(**vars(parser.parse_args()))
    print(json.dumps(result, allow_nan=False), flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
