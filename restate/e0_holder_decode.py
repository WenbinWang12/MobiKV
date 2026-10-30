"""E0 holder-side decode correctness with a genuine pre-existing GPU KV cache.

Two distinct GPU hosts exchange a suffix request and stream generated token IDs.
Full-vocabulary logits are also sent solely for correctness diagnostics. No KV
tensor is transmitted. This functional check is not performance evidence.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

from e0_native_fetch import recv_json, send_json, sha_json


CASES = [(64, 16, 16), (128, 7, 16), (257, 17, 32)]
ATOL = RTOL = 1e-4
PROMPT = (
    "A distributed inference server stores the attention keys and values of a document. "
    "A later request reuses the document and adds a new question. "
    "The system transfers the cache to another server before continuing generation. "
) * 20


def verify_manifest(model_path):
    manifest_path = model_path / "e0_model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        path = model_path / entry["name"]
        if path.stat().st_size != entry["size_bytes"]:
            raise RuntimeError(f"Model file size mismatch: {entry['name']}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Model file SHA256 mismatch: {entry['name']}")
    return manifest, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def compare(torch, reference, candidate, reference_tokens, candidate_tokens):
    difference = (reference - candidate).abs()
    ratio = difference / (ATOL + RTOL * reference.abs())
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    top2 = torch.topk(reference, 2, dim=-1).values
    return {
        "logits_pass": finite and bool((ratio <= 1).all()),
        "tokens_match": reference_tokens == candidate_tokens,
        "finite": finite,
        "max_abs_error": float(difference.max()),
        "max_tolerance_ratio": float(ratio.max()),
        "per_step_max_abs_error": difference.max(dim=-1).values.tolist(),
        "min_reference_top2_margin": float((top2[:, 0] - top2[:, 1]).min()),
    }


def main(args, report, event):
    import torch
    import torch.distributed as dist
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from e0_cache_utils import cache_to_cpu_pairs, cache_sha256, cache_bytes, cache_metadata

    rank = report["rank"]
    if int(os.environ.get("SLURM_NTASKS", "0")) != 2 or rank not in (0, 1):
        raise RuntimeError("Exactly two Slurm tasks, one per distinct host, are required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one allocated CUDA GPU per process is required")
    torch.cuda.set_device(0)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    if dtype == torch.bfloat16:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Allocated device does not support BF16")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    backend_flags = {
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
    }
    protocol_path = Path(__file__).with_name("e0_holder_precision_protocol.json")
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if protocol["cases"] != [list(case) for case in CASES] or protocol["atol"] != ATOL or protocol["rtol"] != RTOL:
        raise RuntimeError("Precision preregistration disagrees with fixed cases or tolerances")
    precision_rule = protocol["precision_rules"][args.dtype]
    expected_gating = (["local_vs_holder"] if args.dtype == "bfloat16" else
                       ["full_vs_local", "full_vs_holder", "local_vs_holder"])
    if precision_rule["gating_checks"] != expected_gating:
        raise RuntimeError("Precision preregistration has an unexpected acceptance rule")
    dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=240))

    model_path = Path(args.model).resolve()
    manifest, manifest_sha = verify_manifest(model_path)
    local_identity = {"hostname": socket.gethostname(), "manifest_sha256": manifest_sha,
                      "source_sha256": report["source_sha256"], "dtype": args.dtype,
                      "backend_flags": backend_flags, "precision_protocol_sha256": protocol_sha}
    if rank == 0:
        send_json(dist, torch, local_identity, 1)
        peer_identity = recv_json(dist, torch, 1)
    else:
        peer_identity = recv_json(dist, torch, 0)
        send_json(dist, torch, local_identity, 0)
    if peer_identity["hostname"] == local_identity["hostname"]:
        raise RuntimeError("Holder and destination must be on distinct hosts")
    for field in ("manifest_sha256", "source_sha256", "dtype", "backend_flags", "precision_protocol_sha256"):
        if peer_identity[field] != local_identity[field]:
            raise RuntimeError(f"Peer disagrees on {field}")
    report.update(
        holder_hostname=local_identity["hostname"] if rank == 0 else peer_identity["hostname"],
        destination_hostname=local_identity["hostname"] if rank == 1 else peer_identity["hostname"],
        distinct_hostnames_pass=True, peer_identity=peer_identity,
        model_repo=manifest["repo_id"], model_revision=manifest["revision"],
        model_manifest_sha256=manifest_sha, torch_version=torch.__version__,
        transformers_version=transformers.__version__, torch_cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), precision=args.dtype, attention="eager",
        backend_flags=backend_flags, precision_protocol_id=protocol["protocol_id"],
        precision_protocol_sha256=protocol_sha, precision_rule=precision_rule,
        diagnostic_logits_wire_dtype="torch.float32", token_wire_dtype="torch.int64",
        tf32=False, atol=ATOL, rtol=RTOL, crossnode=True,
        transport="gloo_tcp_cpu_tensors", timing_is_performance_evidence=False,
        generation_policy="fixed_length_greedy_no_eos_early_stop", cases=[])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        dtype=dtype, attn_implementation="eager").to("cuda:0").eval()
    if model.config.model_type != "qwen2" or getattr(model.config, "use_sliding_window", False):
        raise RuntimeError("Only full-attention Qwen2 is supported")
    parameter_dtypes = sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()})
    report["model_parameter_dtypes"] = parameter_dtypes
    report["model_buffer_dtypes"] = sorted({str(b.dtype) for b in model.buffers()})
    if parameter_dtypes != [str(dtype)]:
        raise RuntimeError("Actual floating model parameters do not match requested precision")
    event("model_loaded", parameter_count=sum(p.numel() for p in model.parameters()))

    def forward(ids, cache, audit, phase):
        previous = 0 if cache is None else int(cache.get_seq_length())
        count = int(ids.shape[-1])
        positions = torch.arange(previous, previous + count, device="cuda:0").unsqueeze(0)
        with torch.inference_mode():
            output = model(input_ids=ids, past_key_values=cache, position_ids=positions,
                           attention_mask=torch.ones((1, previous + count), device="cuda:0",
                                                     dtype=torch.long), use_cache=True)
        after = int(output.past_key_values.get_seq_length())
        if after != previous + count:
            raise RuntimeError(f"Invalid cache progression: {previous}+{count}!={after}")
        kv_dtypes = sorted({str(t.dtype) for layer in output.past_key_values.layers
                            for t in (layer.keys, layer.values)})
        if kv_dtypes != [str(dtype)]:
            raise RuntimeError("Actual KV dtype does not match requested precision")
        audit.append({"phase": phase, "input_tokens": count, "cache_before": previous,
                      "cache_after": after, "first_position": previous,
                      "last_position": previous + count - 1, "kv_dtypes": kv_dtypes,
                      "native_logits_dtype": str(output.logits.dtype)})
        return output

    def generate(ids, cache, count, audit, phase, stream=False):
        rows, tokens = [], []
        output = forward(ids, cache, audit, phase)
        for step in range(count):
            scores = output.logits[:, -1, :].detach().float()
            token = scores.argmax(dim=-1, keepdim=True)
            cpu_scores = scores.to("cpu").contiguous().reshape(-1)
            token_id = int(token.item())
            rows.append(cpu_scores)
            tokens.append(token_id)
            if stream:
                # Ordinary CPU tensors: Gloo's worker thread may write outside inference_mode.
                dist.send(torch.tensor([token_id], dtype=torch.int64), 1)
                dist.send(cpu_scores, 1)
                event("holder_token_sent", case_id=case_id, step=step, token_id=token_id)
            if step + 1 < count:
                output = forward(token, output.past_key_values, audit, phase)
        return torch.stack(rows), tokens

    all_ids = tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt").input_ids
    if all_ids.shape[-1] < max(p + s for p, s, _ in CASES):
        raise RuntimeError("Fixed prompt produces too few tokens")
    for prefix_len, suffix_len, generated_len in CASES:
        case_id = f"p{prefix_len}-s{suffix_len}-g{generated_len}"
        ids_cpu = all_ids[:, :prefix_len + suffix_len].contiguous()
        prefix = ids_cpu[:, :prefix_len].to("cuda:0")
        suffix_cpu = ids_cpu[:, prefix_len:].contiguous()
        identity = {
            "case_id": case_id, "model_manifest_sha256": manifest_sha,
            "model_revision": manifest["revision"], "prefix_tokens": prefix_len,
            "suffix_tokens": suffix_len, "generated_tokens": generated_len,
            "input_ids_sha256": sha_json(ids_cpu.tolist()),
            "prefix_ids_sha256": sha_json(ids_cpu[:, :prefix_len].tolist()),
            "suffix_ids_sha256": sha_json(suffix_cpu.tolist()),
            "dtype": args.dtype, "backend_flags": backend_flags,
            "precision_protocol_sha256": protocol_sha,
        }
        counts = {"request_token_bytes": suffix_len * 8, "output_token_bytes": generated_len * 8,
                  "diagnostic_logit_bytes": generated_len * model.config.vocab_size * 4,
                  "kv_bytes_transferred": 0,
                  "byte_count_scope": "application_tensor_payload_excluding_json_and_gloo_framing"}
        counts["output_bytes_including_diagnostics"] = counts["output_token_bytes"] + counts["diagnostic_logit_bytes"]
        case = {**identity, **counts, "audit": [], "checks": {}, "passed": False,
                "gating_checks": precision_rule["gating_checks"],
                "diagnostic_only_checks": precision_rule["diagnostic_only_checks"]}
        report["cases"].append(case)
        if rank == 0:
            cached = forward(prefix, None, case["audit"], "holder_prefix_precompute")
            retained_cache = cached.past_key_values
            del cached
            if any(layer.keys.device.type != "cuda" or layer.values.device.type != "cuda"
                   for layer in retained_cache.layers):
                raise RuntimeError("Holder prefix cache is not resident on GPU")
            pairs = cache_to_cpu_pairs(retained_cache)
            cache_audit = {"prefix_cache_sha256": cache_sha256(pairs), "prefix_cache_bytes": cache_bytes(pairs),
                           "prefix_cache_metadata": cache_metadata(pairs), "prefix_cache_device": "cuda:0",
                           "prefix_cache_dtype": str(dtype),
                           "ready_before_request_barrier": True}
            del pairs
            event("holder_cache_ready", case_id=case_id, **cache_audit)
        else:
            reference, reference_tokens = generate(ids_cpu.to("cuda:0"), None, generated_len,
                                                    case["audit"], "full_recompute_reference")
            local_prefix = forward(prefix, None, case["audit"], "local_control_prefix")
            control, control_tokens = generate(suffix_cpu.to("cuda:0"), local_prefix.past_key_values,
                                                generated_len, case["audit"], "local_control_decode")
            del local_prefix
        torch.cuda.synchronize()
        dist.barrier()  # Holder KV exists, and destination references finish, before this request opens.
        event("cached_hit_request_open", case_id=case_id)
        start = time.monotonic()
        if rank == 1:
            send_json(dist, torch, identity, 0)
            dist.send(suffix_cpu, 0)
            header = recv_json(dist, torch, 0)
            if header["identity"] != identity or header["counts"] != counts:
                raise RuntimeError("Holder response identity or tensor byte counts differ")
            if header["cache_audit"]["prefix_cache_dtype"] != str(dtype):
                raise RuntimeError("Holder prefix cache has an unexpected actual dtype")
            received_rows, received_tokens = [], []
            for step in range(generated_len):
                token = torch.empty(1, dtype=torch.int64)
                logits = torch.empty(model.config.vocab_size, dtype=torch.float32)
                dist.recv(token, 0)
                dist.recv(logits, 0)
                received_tokens.append(int(token.item()))
                received_rows.append(logits)
                if int(logits.argmax().item()) != received_tokens[-1]:
                    raise RuntimeError("Streamed token does not match its diagnostic logits")
                event("destination_token_received", case_id=case_id, step=step,
                      token_id=received_tokens[-1])
            completion = recv_json(dist, torch, 0)
            if completion["identity"] != identity or completion["counts"] != counts:
                raise RuntimeError("Holder completion disagrees on request identity or payload counts")
            candidate = torch.stack(received_rows)
            checks = {"full_vs_local": compare(torch, reference, control, reference_tokens, control_tokens),
                      "full_vs_holder": compare(torch, reference, candidate, reference_tokens, received_tokens),
                      "local_vs_holder": compare(torch, control, candidate, control_tokens, received_tokens)}
            no_recompute = completion["no_prefix_recompute_pass"]
            passed = bool(no_recompute and all(c["finite"] for c in checks.values()) and all(
                checks[name]["logits_pass"] and checks[name]["tokens_match"]
                for name in precision_rule["gating_checks"]))
            case.update(role="destination", checks=checks, passed=passed,
                        no_prefix_recompute_pass=no_recompute, holder_cache_audit=header["cache_audit"],
                        holder_execution_audit=completion["execution_audit"],
                        reference_token_ids=reference_tokens, local_token_ids=control_tokens,
                        holder_token_ids=received_tokens,
                        diagnostic_request_seconds=time.monotonic() - start)
            send_json(dist, torch, {"identity": identity, "passed": passed, "checks": checks}, 0)
            del reference, control, candidate, received_rows
        else:
            request = recv_json(dist, torch, 1)
            if request != identity:
                raise RuntimeError("Incoming request does not identify the precomputed prefix")
            received_suffix = torch.empty((1, suffix_len), dtype=torch.int64)
            dist.recv(received_suffix, 1)
            if not torch.equal(received_suffix, suffix_cpu) or sha_json(received_suffix.tolist()) != identity["suffix_ids_sha256"]:
                raise RuntimeError("Incoming suffix token payload is corrupt")
            if int(retained_cache.get_seq_length()) != prefix_len:
                raise RuntimeError("Pre-existing holder cache changed before request execution")
            send_json(dist, torch, {"identity": identity, "counts": counts, "cache_audit": cache_audit}, 1)
            execution_audit = []
            holder_logits, holder_tokens = generate(received_suffix.to("cuda:0"), retained_cache,
                                                     generated_len, execution_audit,
                                                     "holder_cached_hit_decode", stream=True)
            no_recompute = (
                [entry["input_tokens"] for entry in execution_audit] == [suffix_len] + [1] * (generated_len - 1)
                and execution_audit[0]["cache_before"] == prefix_len
                and execution_audit[-1]["cache_after"] == prefix_len + suffix_len + generated_len - 1)
            send_json(dist, torch, {"identity": identity, "counts": counts,
                                   "execution_audit": execution_audit,
                                   "no_prefix_recompute_pass": no_recompute}, 1)
            acknowledgement = recv_json(dist, torch, 1)
            if acknowledgement["identity"] != identity:
                raise RuntimeError("Destination acknowledgement identity mismatch")
            case["audit"].extend(execution_audit)
            case.update(role="holder", holder_cache_audit=cache_audit,
                        no_prefix_recompute_pass=no_recompute, holder_token_ids=holder_tokens,
                        destination_checks=acknowledgement["checks"],
                        passed=bool(no_recompute and acknowledgement["passed"]))
            del retained_cache, holder_logits
        event("case_checked", case_id=case_id, passed=case["passed"])
        if not case["passed"]:
            raise RuntimeError(f"Holder-side decode failed {case_id}; tolerances were not relaxed")
        dist.barrier()
    report.update(passed=True, scope="cross-node holder-side cached decode correctness only")
    dist.destroy_process_group()


def entrypoint():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Use a scheduled GPU allocation; never run on a login node")
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("SLURM_PROCID", "-1"))
    sources = [Path(__file__), Path(__file__).with_name("e0_cache_utils.py"),
               Path(__file__).with_name("e0_native_fetch.py"),
               Path(__file__).with_name("e0_holder_precision_protocol.json")]
    report = {"stage": "E0_holder_decode", "passed": False, "rank": rank,
              "hostname": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "python_executable": sys.executable, "started_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    with (result_dir / f"rank-{rank}-events.jsonl").open("w", encoding="utf-8") as stream:
        def event(name, **fields):
            stream.write(json.dumps({"event": name, "rank": rank,
                                     "monotonic_s": time.monotonic(), **fields}) + "\n")
            stream.flush()
        try:
            main(args, report, event)
        except Exception as exc:
            report.update(passed=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            event("failed", error=report["error"])
        finally:
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            (result_dir / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rank": rank, "passed": report["passed"], "error": report.get("error"),
                      "result": str(result_dir / f"rank-{rank}.json")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(entrypoint())
