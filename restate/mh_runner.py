"""E6/E7 worker: ranks 0..N-2 are replica holders, the last rank is the target.

Environment: MH_CONTEXT=<absolute context JSON>, RANK=<int>, CUDA_VISIBLE_DEVICES=<this rank's GPU UUID>.
No added propagation delay is applied (NullNetem); path heterogeneity is the
per-holder application budget, exactly the E4 directional coordinator.
"""
from __future__ import annotations

import gc
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time
import traceback
from datetime import datetime, timezone


class NullNetem:
    """Zero added delay and no IFB redirection; satisfies the E4 holder/target zero-delay checks."""
    def set_delay_ms(self, value):
        if value != 0:
            raise ValueError("MH runs never add propagation delay")

    def snapshot(self):
        return {"requested_added_rtt_ms": 0, "observed_one_way_delay_ms": 0, "ifb": None,
                "scope": "no IFB/netem; zero added delay; application budget only"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    rank = int(os.environ["RANK"])
    context_path = Path(os.environ["MH_CONTEXT"])
    context = json.loads(context_path.read_text())
    plan_path = Path(context["plan_path"])
    require(sha(plan_path) == context["plan_sha256"], "Plan differs from context")
    plan = json.loads(plan_path.read_text())
    ranks = context["ranks"]
    me = ranks[rank]
    result_dir = Path(context["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    session = context["session_id"]
    report = dict(stage=plan["stage"], mode=plan["mode"], run_id=context["run_id"], session_id=session, rank=rank,
                  role=me["role"], hostname=socket.gethostname(), started_utc=datetime.now(timezone.utc).isoformat(),
                  passed=False, context_sha256=sha(context_path), plan_sha256=context["plan_sha256"])
    controls, raw_sockets, sock = [], [], None
    netem = None
    gpu_lock = None
    netem_stop = threading.Event()
    netem_thread = None
    netem_errors = []
    exit_code = 1
    try:
        for name, expected in context['package_sha256'].items():
            require(sha(Path(__file__).resolve().parent / name) == expected, 'Frozen package changed: ' + name)
        require(socket.gethostname() == me["hostname"], "Worker hostname differs from rank identity")
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == me["gpu_uuid"], "Visibility must name this rank's GPU UUID")
        gpu_lock = open('/tmp/restate-mh-' + me['gpu_uuid'] + '.lock', 'a')
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from e0_holder_decode import verify_manifest
        from e1_runtime import Engine
        from compute_queue import ComputeQueue
        from e1_runner import connect_raw, listener
        from full_control import TrialControl
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one visible GPU")
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        report.update(gpu=props.name, gpu_memory_bytes=props.total_memory, gpu_uuid=me["gpu_uuid"])
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.set_float32_matmul_precision("highest")
        manifest, manifest_sha = verify_manifest(Path(context["model_path"]))
        require(manifest["revision"] == context["model_revision"] and manifest_sha == context["model_manifest_sha256"],
                "Model revision/manifest differs")
        tokenizer = AutoTokenizer.from_pretrained(context["model_path"], local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(context["model_path"], local_files_only=True, trust_remote_code=False,
                                                     dtype=torch.bfloat16, attn_implementation="eager").to("cuda:0").eval()
        engine = Engine(model)
        engine.e4_execution_mode = "e4_full"
        engine.model_identity = {"model_id": context["model_id"], "model_revision": context["model_revision"],
                                 "model_manifest_sha256": manifest_sha}
        engine.compute_queue = ComputeQueue.for_engine(engine)
        engine.e1_netem = NullNetem()
        if me["role"] == "holder" and plan.get("holder_added_rtt_ms_grid"):
            # Experiment 5: holder-side ingress delay on TCP from the target only.
            from mh_netem import HolderIngressDelay
            netem = HolderIngressDelay(me["internal_ip"], ranks[-1]["internal_ip"], context["run_id"],
                plan["holder_added_rtt_ms_grid"], base_port=context['base_port'] + 100 * rank,
                journal=result_dir / ('netem-rank-%d-%s.json' % (rank, session)), lease_seconds=180)
            report["netem_setup"] = netem.setup()
            engine.e1_netem = netem
            def renew_netem():
                while not netem_stop.wait(30):
                    try:
                        netem.renew()
                    except BaseException as error:
                        netem_errors.append(repr(error))
                        if not netem_stop.is_set():
                            os.kill(os.getpid(), signal.SIGTERM)
                        return
            netem_thread = threading.Thread(target=renew_netem, name='x5-netem-lease', daemon=True)
            netem_thread.start()
        from campaign_runner import materialize
        warm_ids = materialize({"input_id": "kernel-warmup-excluded", "prefix": 64, "suffix": 4}, tokenizer)
        wp = torch.tensor([warm_ids[:64]], device=engine.device)
        ws = torch.tensor([warm_ids[64:]], device=engine.device)
        pairs = engine.prefix(wp)
        store = engine.store_from_prefix(pairs)
        engine.decode(store, 64, ws, 2, capture_logits=False)
        engine.HF_reference(wp, ws, 2, capture_logits=False)
        store.values.clear()
        pairs.clear()
        del store, pairs, wp, ws
        gc.collect()
        torch.cuda.synchronize()
        identity = dict(plan_sha256=context["plan_sha256"], run_id=context["run_id"], session_id=session,
                        model_revision=context["model_revision"], model_manifest_sha256=manifest_sha,
                        torch=torch.__version__, transformers=transformers.__version__, layers=engine.layers)
        base = context["base_port"]
        hello = dict(kind="mh_identity", identity=identity, rank=rank, hostname=socket.gethostname(), gpu_uuid=me["gpu_uuid"])
        if me["role"] == "holder":
            sock = listener(base + 100 * rank)
            raw, _ = sock.accept()
            raw_sockets.append(raw)
            raw.settimeout(1200)
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            control = TrialControl(raw)
            controls.append(control)
            control.send_message(hello)
            peer, tensors, _ = control.recv_message()
            require(not tensors and peer.get("identity") == identity and peer.get("rank") == len(ranks) - 1,
                    "Target identity differs")
            from mh_source_server import MHSourceServer
            registry = plan["holder_registries"][str(rank)]
            server = MHSourceServer(engine, control, base + 100 * rank, trials=registry)
            report["server"] = server.serve()
            require(report["server"].get("passed") is True, "Holder session shutdown failed")
        else:
            holders = []
            for index, row in enumerate(ranks[:-1]):
                deadline = time.monotonic() + 600
                while True:
                    try:
                        raw = connect_raw(row["internal_ip"], base + 100 * index)
                        break
                    except OSError:
                        require(time.monotonic() < deadline, "Holder %d never listened" % index)
                        time.sleep(2)
                raw_sockets.append(raw)
                raw.settimeout(1200)
                control = TrialControl(raw)
                controls.append(control)
                control.send_message(hello)
                peer, tensors, _ = control.recv_message()
                require(not tensors and peer.get("identity") == identity and peer.get("rank") == index
                        and peer.get("hostname") == row["hostname"] and peer.get("gpu_uuid") == row["gpu_uuid"],
                        "Holder identity differs")
                holders.append(dict(index=index, control=control, host=row["internal_ip"]))
            from mh_campaign import MHCampaign
            campaign = MHCampaign(engine, tokenizer, holders, plan, result_dir, report)
            campaign.run()
        require(not netem_errors, 'X5 network lease failed: ' + repr(netem_errors))
        report["passed"] = True
        exit_code = 0
    except BaseException as exc:
        report.update(passed=False, error="%s: %s" % (type(exc).__name__, exc), traceback=traceback.format_exc())
        traceback.print_exc()
        exit_code = 1
    finally:
        netem_stop.set()
        if netem_thread is not None:
            netem_thread.join(timeout=20)
            if netem_thread.is_alive():
                netem_errors.append('Netem renewal thread did not stop within 20 seconds')
        if netem_errors:
            report.update(passed=False, netem_lease_errors=list(netem_errors))
            exit_code = 1
        if netem is not None:
            try:
                report["netem_cleanup"] = netem.cleanup()
                if not report['netem_cleanup'].get('restored') or report['netem_cleanup'].get('errors'):
                    report['passed'] = False
                    exit_code = 1
            except Exception as error:
                report["netem_cleanup_error"] = repr(error)
                report['passed'] = False
                exit_code = 1
        for c in controls:
            try:
                c.close()
            except Exception:
                pass
        for r in raw_sockets:
            try:
                r.close()
            except Exception:
                pass
        if sock is not None:
            sock.close()
        if gpu_lock is not None:
            gpu_lock.close()
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        from mh_campaign import save
        save(result_dir / ("rank-%d-%s.json" % (rank, session)), report)
    return exit_code


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise RuntimeError("MH runtime interrupted by signal " + str(signum))
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, interrupted)
    sys.exit(main())
