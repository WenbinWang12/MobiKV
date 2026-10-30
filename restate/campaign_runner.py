"""Bounded real-GPU campaign; smoke acceptance is separate from paper completion.

Explicit direct context, rank one is the destination. Each measured
candidate has a separate full-vocabulary diagnostic with its exact layer routes.
No diagnostic latency, constructed counterfactual cost, or A6000 timing is paper
performance evidence. Failed raw candidates remain in the result directory.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
import signal
from pathlib import Path
import socket
import statistics
import subprocess
import threading
import time
import traceback

from e1_runner import canonical_hash, connect_raw, digest, listener, save
from e1_wire import FramedConnection
from direct_context import load_context, selected_cuda_identity, peer_hello, validate_peer


LOCAL = {"recompute", "local_reuse", "native_hf_control"}


def materialize(case, tokenizer):
    """Freeze actual IDs before any timed request; no outcome-dependent inputs."""
    recipe = case.get("input_recipe")
    if recipe is not None and recipe.get("kind") == "e7":
        prefix_text, suffix_text = recipe["prefix_family"], recipe["suffix_text"]
    else:
        prefix_text = ("Distributed inference reuses immutable document keys and values. "
                       + case["input_id"] + ". ")
        suffix_text = "Continue the technical explanation for " + case["input_id"] + ". "
    def tokens(text, length):
        unit = tokenizer(text, add_special_tokens=False).input_ids
        if not unit:
            raise ValueError("Empty tokenizer recipe")
        return (unit * (length // len(unit) + 1))[:length]
    return tokens(prefix_text, case["prefix"]) + tokens(suffix_text, case["suffix"])


def metadata(control, message, expected):
    control.send_message(message)
    reply, tensors, _ = control.recv_message()
    if tensors or reply.get("kind") != expected:
        raise ValueError("Unexpected control reply: " + repr(reply))
    return reply


def reconcile(destination, source):
    if source.get("passed") is not True or not destination["background_drained"]:
        raise ValueError("Unsettled source/destination request")
    for channel in ("rpc", "bulk"):
        left, right = destination["traffic"][channel], source["traffic"][channel]
        if (left is None) != (right is None):
            raise ValueError("Request channel presence differs")
        if left is None:
            continue
        for field, peer_field in (("sent", "received"), ("received", "sent"),
                                  ("payload_sent", "payload_received"), ("payload_received", "payload_sent")):
            if left[field] != right[peer_field]:
                raise ValueError(f"Actual socket accounting mismatch {channel}/{field}: {left[field]} != {right[peer_field]}")
    if len(destination["remote_calls"]) != source["remote_calls"]:
        raise ValueError("Remote attention invocation counts differ")
    if destination["remote_calls"] != [[row["layer"], row["position"]] for row in source["remote_audit"]]:
        raise ValueError("Source attention positions differ from destination")
    return {"passed": True, "scope": "complete frames versus actual socket application bytes", "physical_wire_bytes": None}


class Campaign:
    def __init__(self, engine, tokenizer, controls, hosts, plan, directory, report, transport_hosts=None):
        from e4_e6_mechanisms import ReplicaDirectory
        self.engine, self.tokenizer = engine, tokenizer
        self.controls, self.hosts, self.plan = controls, hosts, plan
        self.transport_hosts = hosts if transport_hosts is None else transport_hosts
        self.directory, self.report = directory, report
        self.directory.mkdir(parents=True, exist_ok=True)
        self.replicas = ReplicaDirectory()
        self.prepared = [{} for _ in controls]
        self.resident = {}
        self.calibration = None
        self.counter = 0
        self.matched = {}

    def case(self, specification):
        case = dict(specification)
        if "initial_frontier_fraction" in case:
            case["initial_frontier"] = int(case["initial_frontier_fraction"] * self.engine.layers)
        case.setdefault("holder", 0)
        case.setdefault("prefix_key", f'{case["input_id"]}-p{case["prefix"]}')
        return case

    def prepare(self, case, ids, holder):
        cached = self.prepared[holder]
        key = case["prefix_key"]
        if key in cached:
            if cached[key]["token_ids_sha256"] != canonical_hash(ids[:case["prefix"]]):
                raise ValueError("Prefix identity was reused for different actual tokens")
            return cached[key]
        if len(cached) >= 8:
            old = next(iter(cached))
            metadata(self.controls[holder], {"kind": "retire", "prefix_key": old}, "retired")
            self.replicas.retire(old, 1, self.hosts[holder], replica_id=self.hosts[holder] + ":" + old)
            del cached[old]
        reply = metadata(self.controls[holder], {"kind": "prepare", "prefix_key": key,
                         "prefix_ids": ids[:case["prefix"]]}, "prepared")
        if reply["layers"] != self.engine.layers:
            raise ValueError("Holder model layer count differs")
        cached[key] = reply
        self.replicas.publish(key, 1, self.hosts[holder], reply["cache_sha256"],
                              replica_id=self.hosts[holder] + ":" + key)
        return reply

    def run_batch(self, experiment, *, diagnostic=False, measured=None, keep_prefix=False):
        from campaign_client import coordinator, execute
        cases = [self.case(case) for case in experiment["cases"]]
        self.counter += 1
        name = f'{self.counter:04d}-{experiment["id"]}-' + ("diagnostic" if diagnostic else "measured")
        path = self.directory / (name + ".json")
        result = {"name": name, "experiment": experiment, "diagnostic": diagnostic,
                  "source_run_id": self.report["run_id"], "scheduler": "direct", "topology": self.report["topology"],
                  "context_sha256": self.report["context_sha256"], "passed": False, "requests": [], "holders": [],
                  "batch_ready": [], "calibration_used": self.calibration}
        ids_list = [materialize(case, self.tokenizer) for case in cases]
        request_ids = [f'{self.report["run_id"]}-{name}-{index}' for index in range(len(cases))]
        grouped, endpoints, cached_inputs = {}, {}, {}
        for index, (case, ids) in enumerate(zip(cases, ids_list)):
            if case["action"] == "local_reuse":
                if case["prefix_key"] not in self.resident:
                    raise ValueError("A target cache hit requires real retained tensors")
                cached_inputs[index] = self.resident.pop(case["prefix_key"])
                self.resident[case["prefix_key"]] = cached_inputs[index]
            if case["action"] in LOCAL:
                continue
            holder = case["holder"]
            if not 0 <= holder < len(self.controls):
                raise ValueError("Selected holder has no allocated physical host")
            # E6 actually establishes every complete replica and checks KV identity.
            for replica in (range(len(self.controls)) if self.plan["stage"] == "E6" else [holder]):
                self.prepare(case, ids, replica)
            binding = self.replicas.bind(request_ids[index], case["prefix_key"], version=1, holder=self.hosts[holder])
            grouped.setdefault(holder, []).append({"request_id": request_ids[index], "prefix_key": case["prefix_key"],
                "suffix_ids": ids[case["prefix"]:], "generated": case["generated"], "action": case["action"],
                "initial_frontier": case.get("initial_frontier", 0)})
        budget = coordinator(experiment["rate_bps"], experiment.get("priority", "rpc_first"), experiment.get("rate_plan"))
        outputs = {}
        try:
            for holder, specs in grouped.items():
                reply = metadata(self.controls[holder], {"kind": "batch", "requests": specs,
                    "rate_bps": experiment["rate_bps"], "priority": experiment.get("priority", "rpc_first"),
                    "rate_plan": experiment.get("rate_plan"), "timeout_s": 600, "integrity_mode": "none"}, "batch_ready")
                if experiment.get("rate_plan"):
                    from e3_perturbation import RatePlan
                    raw = experiment["rate_plan"]
                    expected_plan = RatePlan(raw["steps"], plan_id=raw["plan_id"], seed=raw["seed"])
                    if reply.get("rate_plan_sha256") != expected_plan.sha256:
                        raise ValueError("Holder uses a different application rate trace")
                result["batch_ready"].append({"holder": self.hosts[holder], "reply": reply,
                                              "destination_received_at_s": time.monotonic()})
                endpoints.update(reply["ports"])
            def run(index):
                self.engine.torch.cuda.set_device(0)
                case = cases[index]
                output = execute(self.engine, ids_list[index], case, request_ids[index], self.transport_hosts[case["holder"]],
                    endpoints.get(request_ids[index], {}), budget, diagnostic=diagnostic,
                    replay=measured["requests"][index].get("cutovers") if measured is not None else None,
                    calibration=self.calibration, cached_pairs=cached_inputs.get(index), keep_prefix=keep_prefix)
                outputs[index] = output
                return output
            if len(cases) == 1:
                run(0)
            else:
                with ThreadPoolExecutor(max_workers=len(cases)) as pool:
                    futures = [pool.submit(run, index) for index in range(len(cases))]
                    for future in futures:
                        future.result()
            result["requests"] = [outputs[index][0] for index in range(len(cases))]
            source_requests = {}
            for holder in grouped:
                reply, tensors, _ = self.controls[holder].recv_message()
                result["holders"].append({"holder": self.hosts[holder], "report": reply})
                if tensors or reply.get("kind") != "batch_done" or reply.get("passed") is not True:
                    raise ValueError("Holder batch failed: " + repr(reply))
                source_requests.update({row["request_id"]: row for row in reply["requests"]})
            for index, (case, row, ids) in enumerate(zip(cases, result["requests"], ids_list)):
                row.update(case=case, input_ids=ids, input_ids_sha256=canonical_hash(ids),
                           source_run_id=self.report["run_id"], scheduler="direct", topology=self.report["topology"],
                           correctness_passed=False,
                           measurement_kind="gpu_request", sample_role="diagnostic" if diagnostic else "measured")
                if row["request_id"] in source_requests:
                    row["source"] = source_requests[row["request_id"]]
                    row["traffic_acceptance"] = reconcile(row, row["source"])
                    self.replicas.release(row["request_id"])
                elif any(row["traffic"].values()):
                    raise ValueError("Local action unexpectedly performed request traffic")
                if keep_prefix and outputs[index][2] is not None:
                    self.resident[case["prefix_key"]] = outputs[index][2]
                    while len(self.resident) > 8:
                        del self.resident[next(iter(self.resident))]
            result.update(passed=True, coordinator=budget.snapshot(), replica_directory=self.replicas.snapshot(),
                          artifact=str(path), source_sha256=self.report["source_sha256"])
            save(path, result)
            return result, [outputs[index][1] for index in range(len(cases))]
        except BaseException as exc:
            result.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(),
                          requests=[value[0] for _, value in sorted(outputs.items())])
            save(path, result)
            raise
        finally:
            budget.close()

    def validate_batch(self, experiment, measured):
        from campaign_client import verify_outputs
        diagnostic, logits = self.run_batch(experiment, diagnostic=True, measured=measured)
        torch = self.engine.torch
        for index, row in enumerate(diagnostic["requests"]):
            case, ids = row["case"], row["input_ids"]
            prefix = torch.tensor([ids[:case["prefix"]]], device=self.engine.device)
            suffix = torch.tensor([ids[case["prefix"]:]], device=self.engine.device)
            reference = self.engine.HF_reference(prefix, suffix, case["generated"])
            check = verify_outputs(torch, reference, row, logits[index], measured["requests"][index])
            measured["requests"][index].update(correctness_passed=True, numerical_check=check,
                diagnostic_artifact=diagnostic["artifact"])
            row.update(correctness_passed=True, numerical_check=check)
            del reference
        measured["correctness_passed"] = diagnostic["correctness_passed"] = True
        save(Path(diagnostic["artifact"]), diagnostic)
        save(Path(measured["artifact"]), measured)
        logits.clear()
        return measured

    def audit_checkpoint(self, measured):
        from campaign_decode import compare_checkpoint_snapshots
        for row in measured["requests"]:
            if row.get("checkpoint") is None:
                continue
            case, checkpoint = row["case"], row["checkpoint"]
            # Source state is re-executed from the same empty delta; compare every
            # actual attention position through the checkpoint and complete KV hash.
            source = row.get("source", {})
            before = [[item["layer"], item["position"]] for item in source.get("remote_audit", [])
                      if item["position"] < checkpoint["position"]]
            external = {"holder_prefix_sha256": source.get("prefix_cache_sha256"), "remote_history": before,
                        "initial_frontier": case.get("initial_frontier", 0),
                        "initial_transfer": [{key: segment.get(key) for key in ("start", "end", "next_layer", "payload_bytes", "stopped")}
                            for segment in row["transfer_segments"][:1]]}
            key = canonical_hash({"ids": row["input_ids"], "generated_checkpoint": case["checkpoint_generated"],
                                  "frontier": case.get("initial_frontier", 0)})
            if key in self.matched:
                previous, old_external = self.matched[key]
                compare_checkpoint_snapshots(previous, checkpoint)
                if old_external != external:
                    raise ValueError("Holder/transport checkpoint differs across continuation branches")
            else:
                self.matched[key] = checkpoint, external
            row["matched_checkpoint_audit"] = {"passed": True, "group_sha256": key,
                "scope": "actual destination KV/history and holder prefix/history; drained initial migration"}
        save(Path(measured["artifact"]), measured)

    def calibrate(self):
        rows = {}
        for action in ("recompute", "remote", "native_fetch", "local_reuse"):
            case = {"id": "calibration-" + action, "input_id": "excluded-warmup-calibration", "seed": 990001,
                    "prefix": 64 if self.plan["mode"] == "smoke" else 128, "suffix": 4,
                    "generated": 8, "action": action, "prefix_key": "excluded-calibration", "holder": 0}
            experiment = {"id": "calibration-" + action, "cases": [case], "rate_bps": 1_000_000_000, "priority": "rpc_first"}
            measured, _ = self.run_batch(experiment, keep_prefix=True)
            self.validate_batch(experiment, measured)
            measured["sample_role"] = "calibration_excluded_from_main_evaluation"
            save(Path(measured["artifact"]), measured)
            rows[action] = measured["requests"][0]
        payload = rows["native_fetch"]["traffic"]["bulk"]["payload_received"]
        fetch = rows["native_fetch"]["ttft_s"]
        self.calibration = {"local_step_seconds": statistics.mean(rows["local_reuse"]["itl_s"]),
            "all_remote_step_seconds": statistics.mean(rows["remote"]["itl_s"]),
            "effective_prefix_payload_bytes_per_second": payload / fetch,
            "full_prefix_recompute_seconds": rows["recompute"]["prefill_duration_s"],
            "training_lengths": [8, 16, 32, 64], "training_split_id": "predeclared-synthetic-training-population",
            "source": "earlier accepted actual GPU calibration in direct run " + self.report["run_id"],
            "rate_estimator_scope": "full payload divided by fetch TTFT, includes setup and first token",
            "calibration_geometry": {"prefix_tokens": case["prefix"], "suffix_tokens": case["suffix"],
                                     "generation_cap": 8, "application_rate_bps": 1_000_000_000},
            "prediction_limitation": "Single small-prefix calibration is an intentionally unscaled baseline; prediction accuracy for different prefix sizes is unestablished",
            "length_estimator_scope": "predeclared synthetic population, never realized future request length"}
        save(self.directory / "calibration.json", self.calibration)
        self.resident.clear()

    def run(self):
        if self.plan["stage"] not in {"E0", "E1", "E2", "E3", "E4"}:
            raise ValueError("This direct runtime supports E0--E4 only")
        self.calibrate()
        measurements = []
        for experiment in self.plan["experiments"]:
            measured, _ = self.run_batch(experiment)
            self.validate_batch(experiment, measured)
            if self.plan["stage"] == "E2":
                self.audit_checkpoint(measured)
            measurements.append(measured)
            self.report.update(completed_experiments=len(measurements), latest_artifact=measured["artifact"])
            save(self.directory / "progress.json", self.report)
        self.report["measurements"] = [{"artifact": value["artifact"], "sha256": digest(value["artifact"])} for value in measurements]
        self.report["matched_checkpoint_groups"] = len(self.matched)
        if self.plan["stage"] == "E2" and len(self.matched) != 4:
            raise ValueError("The paired continuation campaign did not validate all four actual checkpoints")
        if self.plan["stage"] == "E3":
            control_rows = [row for batch in measurements for row in batch["requests"] if row["case"].get("control_script")]
            self.report["actual_migration_control_events"] = [event for row in control_rows for event in row.get("control_events", [])]
            applied = {event.get("command") for event in self.report["actual_migration_control_events"] if event.get("applied") is True}
            if not {"resume", "stop"}.issubset(applied):
                raise ValueError("E3 smoke did not exercise an actual resumed and stopped migration; a completed-transfer no-op is insufficient")
        if self.plan["stage"] == "E6":
            used = {source["holder"] for batch in measurements for source in batch["holders"]}
            if used != set(self.hosts):
                raise ValueError("E6 did not execute requests on both physical replica holders")
        return measurements


def analyze(args, plan, report):
    raise ValueError("E5--E7 are outside the direct E0--E4 runtime")


def main():
    parser = argparse.ArgumentParser()
    for name in ("plan", "result-dir"):
        parser.add_argument("--" + name, required=True)
    for name in ("model", "model-id", "revision", "source-result"):
        parser.add_argument("--" + name)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16"])
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    from e4_protocol import validate_wrapper, trials
    validate_wrapper(plan)
    rank = int(os.environ.get("RANK", "-1"))
    directory = Path(args.result_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report = {"stage": plan["stage"], "mode": plan["mode"], "passed": False,
              "scheduler": "direct", "rank": rank, "hostname": socket.gethostname(),
              "started_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": digest(args.plan),
              "paper_stage_complete": False, "performance_evidence": False,
              "prior_E3_acceptance": plan["e3_dependency"], "inherited_Babel_crosshost_acceptance": False,
              "scope": "Main14B E4 on the attested two-host A100 20261014 cohort; simultaneous RPC and migration under one actual outgoing budget; full acceptance additionally requires raw validation and sixteen separate profiles"}
    controls, raw_sockets, sock = [], [], None
    network = network_thread = None
    network_stop, network_errors = threading.Event(), []
    try:
        context, context_hash, rank, physical = load_context(args, plan, Path(__file__).parent)
        hosts = context["hosts"]
        report.update(run_id=context["run_id"], cluster_id=context["cluster_id"], topology=context["topology"],
                      context_sha256=context_hash, execution_context=context, physical_identity=physical,
                      source_sha256=context["source_sha256"], rank=rank,
                      network_scope="Real TCP; one application-byte budget per direction including request setup; ten service ports with scoped ingress IFB netem at zero added RTT",
                      added_propagation_rtt_s=0.0)
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from e0_holder_decode import verify_manifest
        from e1_runtime import Engine
        selected = selected_cuda_identity(torch, context, rank)
        report.update(selected)
        gpu = selected["gpu"]
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.set_float32_matmul_precision("highest")
        manifest, manifest_sha = verify_manifest(Path(args.model))
        if manifest["revision"] != args.revision or manifest_sha != context["model_manifest_sha256"]:
            raise ValueError("Model revision differs from frozen submission")
        if plan["mode"] != "smoke" and manifest_sha != "0ef3c81db38c9d2fe14959dc6e8abb5780bbf826b264dbacbdbf80d586cc5e99":
            raise ValueError("Main campaign requires the previously attested Qwen2.5-14B model manifest")
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation="eager").to("cuda:0").eval()
        engine = Engine(model)
        engine.e4_execution_mode=plan['mode']
        engine.model_identity = {"model_id": args.model_id, "model_revision": args.revision,
                                 "model_manifest_sha256": manifest_sha}
        from compute_queue import ComputeQueue
        engine.compute_queue = ComputeQueue.for_engine(engine)
        # Equal untimed warmup on every physical GPU; neither warmup output nor
        # diagnostic timing enters an action-cost table.
        warm_case = {"input_id": "kernel-warmup-excluded", "prefix": 64, "suffix": 4}
        warm_ids = materialize(warm_case, tokenizer)
        warm_prefix = torch.tensor([warm_ids[:64]], device=engine.device)
        warm_suffix = torch.tensor([warm_ids[64:]], device=engine.device)
        warm_pairs = engine.prefix(warm_prefix)
        warm_store = engine.store_from_prefix(warm_pairs)
        engine.decode(warm_store, 64, warm_suffix, 2, capture_logits=False)
        engine.HF_reference(warm_prefix, warm_suffix, 2, capture_logits=False)
        warm_store.values.clear()
        warm_pairs.clear()
        del warm_store, warm_pairs, warm_prefix, warm_suffix
        gc.collect()
        torch.cuda.synchronize()
        from scoped_netem import ScopedNetem, make_plan
        network = engine.e1_netem = ScopedNetem(make_plan(
            context["ranks"][rank]["internal_ip"], context["ranks"][1-rank]["internal_ip"],
            context["run_id"], lease_seconds=180), directory / f"netem-rank-{rank}.json")
        report["network_setup"] = network.setup()
        def heartbeat():
            while not network_stop.wait(30):
                try:
                    network.renew()
                except BaseException as error:
                    network_errors.append(str(error))
                    return
        network_thread = threading.Thread(target=heartbeat, name="owned-netem-heartbeat", daemon=True)
        network_thread.start()
        identity = {"plan_sha256": report["plan_sha256"], "source_sha256": report["source_sha256"],
                    "model_revision": args.revision, "model_manifest_sha256": manifest_sha, "hosts": hosts,
                    "torch": torch.__version__, "transformers": transformers.__version__, "dtype": args.dtype,
                    "layers": engine.layers, "gpu": gpu, "TP": 1,
                    "transport_version": "full-E4-v1-shared-directional-budget-common-barrier-owned-settlement",
                    "scheduler": "direct", "run_id": context["run_id"], "cluster_id": context["cluster_id"],
                    "topology": context["topology"], "context_sha256": context_hash,
                    "rank_allocations": context["ranks"],
                    "backend_flags": {"attention": "eager", "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                        "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                        "float32_matmul_precision": torch.get_float32_matmul_precision()}}
        report.update(identity=identity, gpu=gpu, completed_experiments=0,
                      model_revision=args.revision, model_manifest_sha256=manifest_sha)
        base = context["master_port"]
        hello = peer_hello(context, rank, identity, physical, selected["gpu_uuid"])
        if rank < len(hosts) - 1:
            sock = listener(base + 100 * rank)
            raw, _ = sock.accept()
            raw_sockets.append(raw)
            raw.settimeout(1200)
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            from full_control import TrialControl
            control = TrialControl(raw)
            controls.append(control)
            control.send_message(hello)
            peer, tensors, _ = control.recv_message()
            report["peer_identity_acceptance"] = validate_peer(peer, tensors, context, rank, identity)
            from e4_source_server import E4SourceServer
            server = E4SourceServer(engine, control, base + 100 * rank,
                                    trials={trial['trial_id']: trial for trial in trials(plan)})
            report["server"] = server.serve()
            if report["server"].get("passed") is not True:
                raise RuntimeError("E4 holder session shutdown failed")
        else:
            for holder, host in enumerate(hosts[:-1]):
                raw = connect_raw(context["ranks"][holder]["internal_ip"], base + 100 * holder)
                raw_sockets.append(raw)
                raw.settimeout(1200)
                from full_control import TrialControl
                control = TrialControl(raw)
                controls.append(control)
                control.send_message(hello)
                peer, tensors, _ = control.recv_message()
                report["peer_identity_acceptance"] = validate_peer(peer, tensors, context, rank, identity)
            from e4_campaign import FullCampaign
            campaign = FullCampaign(engine, tokenizer, controls, hosts[:-1], plan, directory, report,
                                transport_hosts=[row["internal_ip"] for row in context["ranks"][:-1]])
            campaign.run()
            campaign.resident.clear()
            gc.collect()
            report["holder_shutdown"] = campaign.report["full_background_cleanup"]
        if network_errors:
            raise RuntimeError("Network lease renewal failed: " + repr(network_errors))
        report["passed"] = True
        return 0
    except BaseException as exc:
        report.update(passed=False,error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        cleanup_errors = []
        network_stop.set()
        if network_thread is not None:
            network_thread.join(10)
            if network_thread.is_alive():
                cleanup_errors.append("network heartbeat did not stop")
        if network is not None:
            try:
                restored = network.cleanup()
                report["network_cleanup"] = restored
                if restored.get("restored") is not True or restored.get("eth0_root_unchanged") is not True:
                    cleanup_errors.append("owned network rules did not restore initial state")
            except BaseException as error:
                cleanup_errors.append("network cleanup: " + str(error))
        cleanup_errors.extend("network heartbeat: " + value for value in network_errors)
        for control in controls:
            try:
                control.close()
            except Exception as error:
                cleanup_errors.append("control: " + repr(error))
        for raw in raw_sockets:
            try:
                raw.close()
            except Exception as error:
                cleanup_errors.append("raw control: " + repr(error))
        if sock:
            try:
                sock.close()
            except Exception as error:
                cleanup_errors.append("listener: " + repr(error))
        report["cleanup"] = {"all_controls_closed": all(control.fileno() == -1 for control in controls)
                                                     and all(raw.fileno() == -1 for raw in raw_sockets),
                             "listener_closed": sock is None or sock.fileno() == -1, "errors": cleanup_errors,
                             "scope": "Actual finally-closed outer control sockets; request cleanup is separately audited per request"}
        if cleanup_errors or not report["cleanup"]["all_controls_closed"] or not report["cleanup"]["listener_closed"]:
            report["passed"] = False
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save(directory / f"rank-{rank}.json", report)
        if cleanup_errors or not report["cleanup"]["all_controls_closed"] or not report["cleanup"]["listener_closed"]:
            raise RuntimeError("Direct campaign outer socket cleanup failed")


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise RuntimeError("Full E4 runtime interrupted by signal " + str(signum))
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    raise SystemExit(main())
