"""E0 native KV fetch correctness, using real Qwen2 weights and isolated caches.

This harness measures semantics, not serving performance. Cross-node transport is
Gloo TCP with CPU staging. No selector, repair, or overlap is implemented here.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def send_json(dist, torch, value, peer):
    raw = json.dumps(value).encode()
    dist.send(torch.tensor([len(raw)], dtype=torch.int64), peer)
    dist.send(torch.tensor(list(raw), dtype=torch.uint8), peer)


def recv_json(dist, torch, peer):
    size = torch.zeros(1, dtype=torch.int64)
    dist.recv(size, peer)
    length = int(size.item())
    if not 0 < length < 1024 * 1024:
        raise RuntimeError(f"Invalid metadata length: {length}")
    data = torch.empty(length, dtype=torch.uint8)
    dist.recv(data, peer)
    return json.loads(bytes(data.tolist()))


def main(args, report, event):
    import torch
    import torch.distributed as dist
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from e0_cache_utils import (cache_to_cpu_pairs, pairs_to_cache,
                                cache_sha256, cache_bytes, cache_metadata)

    cross = args.mode == "crossnode"
    rank = int(os.environ.get("SLURM_PROCID", "0")) if cross else 0
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one allocated CUDA GPU per process is required.")
    torch.cuda.set_device(0)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    if args.dtype == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Allocated GPU does not support BF16.")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    numerical_settings = {
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_bf16_reduced_precision_reduction":
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "matmul_allow_fp16_reduced_precision_reduction":
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
    }
    protocol_path = Path(__file__).with_name("e0_native_precision_protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not (protocol["logit_atol"] == args.atol == 1e-4
            and protocol["logit_rtol"] == args.rtol == 1e-4):
        raise RuntimeError("Native precision protocol requires the preregistered strict 1e-4 tolerances.")
    required_checks = (["full_vs_local", "full_vs_fetch", "local_vs_fetch"]
                       if args.dtype == "float32" else ["local_vs_fetch"])
    if protocol["precision_policies"][args.dtype]["required_checks"] != required_checks:
        raise RuntimeError("Native precision protocol disagrees with implemented acceptance.")
    if cross:
        world = int(os.environ.get("SLURM_NTASKS", "0"))
        if world != 2:
            raise RuntimeError("Cross-node E0 needs exactly two Slurm tasks.")
        dist.init_process_group("gloo", rank=rank, world_size=world,
                                timeout=timedelta(seconds=240))

    model_path = Path(args.model).resolve()
    manifest_file = model_path / "e0_model_manifest.json"
    manifest = json.loads(manifest_file.read_text())
    for entry in manifest["files"]:
        file_path = model_path / entry["name"]
        if file_path.stat().st_size != entry["size_bytes"]:
            raise RuntimeError(f"Staged model file size changed: {entry['name']}")
        digest = hashlib.sha256()
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Staged model file hash changed: {entry['name']}")
    report.update(model_repo=manifest["repo_id"], model_revision=manifest["revision"],
                  model_manifest_sha256=hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
                  torch_version=torch.__version__, transformers_version=transformers.__version__,
                  torch_cuda_version=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0), precision=args.dtype, attention="eager",
                  numerical_settings=numerical_settings, required_checks=required_checks,
                  precision_protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                  atol=args.atol, rtol=args.rtol, crossnode=cross,
                  transport="gloo_tcp_cpu_staging" if cross else "isolated_cpu_clone",
                  timing_is_performance_evidence=False, cases=[])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                               trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        dtype=model_dtype, attn_implementation="eager").to("cuda:0").eval()
    if model.dtype != model_dtype or any(parameter.dtype != model_dtype
            for parameter in model.parameters() if parameter.is_floating_point()):
        raise RuntimeError("Model floating parameters do not match requested precision.")
    if model.config.model_type != "qwen2" or getattr(model.config, "use_sliding_window", False):
        raise RuntimeError("This first harness supports full-attention Qwen2 only.")
    peer_identity = {
        "model_revision": manifest["revision"],
        "model_manifest_sha256": report["model_manifest_sha256"],
        "source_sha256": report["source_sha256"],
        "dtype": args.dtype, "model_dtype": str(model.dtype), "wire_dtype": str(model_dtype),
        "numerical_settings": numerical_settings,
        "precision_protocol_sha256": report["precision_protocol_sha256"],
    }
    report["peer_identity"] = peer_identity
    if cross:
        local_hello = {"hostname": socket.gethostname(), "identity": peer_identity}
        if rank == 0:
            send_json(dist, torch, local_hello, 1)
            peer = recv_json(dist, torch, 1)
        else:
            peer = recv_json(dist, torch, 0)
            send_json(dist, torch, local_hello, 0)
        if peer.get("hostname") == socket.gethostname() or peer.get("identity") != peer_identity:
            raise RuntimeError("Peer host/model/source/precision identity mismatch.")
        report["peer_hostname"] = peer["hostname"]
        report["holder_hostname"] = socket.gethostname() if rank == 0 else peer["hostname"]
    event("model_loaded", parameter_count=sum(p.numel() for p in model.parameters()))

    def forward(ids, cache, audit, phase):
        previous = 0 if cache is None else int(cache.get_seq_length())
        count = int(ids.shape[-1])
        positions = torch.arange(previous, previous + count, device="cuda:0").unsqueeze(0)
        output = model(input_ids=ids, past_key_values=cache, position_ids=positions,
                       attention_mask=torch.ones((1, previous + count), device="cuda:0",
                                                 dtype=torch.long), use_cache=True)
        after = int(output.past_key_values.get_seq_length())
        if after != previous + count:
            raise RuntimeError(f"Unexpected cache length: {previous}+{count}!={after}")
        audit.append({"phase": phase, "input_tokens": count,
                      "cache_before": previous, "cache_after": after,
                      "first_position": previous, "last_position": previous + count - 1})
        return output

    def sequence(ids, cache, n, audit, phase):
        logits, tokens = [], []
        output = forward(ids, cache, audit, phase)
        for index in range(n):
            scores = output.logits[:, -1, :].float()
            logits.append(scores.cpu().clone())
            token = scores.argmax(dim=-1, keepdim=True)
            tokens.append(int(token.item()))
            if index + 1 < n:
                output = forward(token, output.past_key_values, audit, phase)
        return torch.cat(logits, dim=0), tokens, int(output.past_key_values.get_seq_length())

    def compare(reference, candidate, ref_tokens, candidate_tokens):
        diff = (candidate - reference).abs()
        allowed = args.atol + args.rtol * reference.abs()
        finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
        normalized = diff / allowed
        top2 = torch.topk(reference, k=2, dim=-1).values
        return {"logits_pass": finite and bool((normalized <= 1).all()),
                "tokens_match": ref_tokens == candidate_tokens,
                "finite": finite, "max_abs_error": float(diff.max()),
                "max_tolerance_ratio": float(normalized.max()),
                "min_reference_top2_margin": float((top2[:, 0] - top2[:, 1]).min()),
                "per_step_max_abs_error": diff.max(dim=-1).values.tolist()}

    cases = [(64, 16, 16), (128, 7, 16), (257, 17, 32)]
    if protocol["cases"] != [list(case) for case in cases]:
        raise RuntimeError("Native precision protocol must preserve all three original cases.")
    text = ("A distributed inference server stores the attention keys and values of a document. "
            "A later request reuses the document and adds a new question. "
            "The system transfers the cache to another server before continuing generation. ") * 20
    all_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    if all_ids.shape[-1] < max(p + s for p, s, _ in cases):
        raise RuntimeError("Fixed prompt tokenizes to too few tokens.")
    with torch.inference_mode():
        for case_index, (prefix_len, suffix_len, generated_len) in enumerate(cases):
            case_id = f"p{prefix_len}-s{suffix_len}-g{generated_len}"
            ids = all_ids[:, :prefix_len + suffix_len].to("cuda:0")
            prefix, suffix = ids[:, :prefix_len], ids[:, prefix_len:]
            identity = {**peer_identity,
                        "input_ids_sha256": sha_json(ids.cpu().tolist()),
                        "prefix_ids_sha256": sha_json(prefix.cpu().tolist()),
                        "prefix_tokens": prefix_len, "suffix_tokens": suffix_len,
                        "generated_tokens": generated_len, "case_id": case_id}
            case = {**identity, "audit": [], "checks": {}}
            holder_pairs = None
            if rank == 0:
                cached = forward(prefix, None, case["audit"], "holder_prefix_precompute")
                holder_pairs = cache_to_cpu_pairs(cached.past_key_values)
                if any(t.dtype != model_dtype for pair in holder_pairs for t in pair):
                    raise RuntimeError("Holder cache/wire dtype differs from the model precision.")
                del cached
                torch.cuda.synchronize()
                event("holder_cache_ready", case_id=case_id, bytes=cache_bytes(holder_pairs))
            destination = not cross or rank == 1
            if destination:
                ref_logits, ref_tokens, reference_length = sequence(ids, None, generated_len,
                                                  case["audit"], "full_recompute_reference")
                local_prefix = forward(prefix, None, case["audit"], "local_control_prefix")
                local_prefix_pairs = cache_to_cpu_pairs(local_prefix.past_key_values)
                local_prefix_sha256 = cache_sha256(local_prefix_pairs)
                del local_prefix_pairs
                control_logits, control_tokens, control_length = sequence(
                    suffix, local_prefix.past_key_values, generated_len,
                    case["audit"], "local_control_decode")
                del local_prefix
            if cross:
                dist.barrier()  # Both real prefix KV and reference computations precede the request.
            event("cached_hit_request_open", case_id=case_id)
            request_start = time.monotonic()
            if cross and rank == 0:
                # The receiver independently derives the expected shapes from this model and input.
                metadata = {"identity": identity, "sha256": cache_sha256(holder_pairs),
                            "bytes": cache_bytes(holder_pairs)}
                send_json(dist, torch, metadata, 1)
                for key, value in holder_pairs:
                    dist.send(key, 1)
                    dist.send(value, 1)
                case.update(sent_sha256=metadata["sha256"], bytes_sent=metadata["bytes"],
                            cache_metadata=cache_metadata(holder_pairs),
                            role="holder", passed=True)
            elif destination:
                if cross:
                    metadata = recv_json(dist, torch, 0)
                    if metadata["identity"] != identity:
                        raise RuntimeError("Holder and destination disagree on model or input identity.")
                    pairs = []
                    head_dim = getattr(model.config, "head_dim", None) or (
                        model.config.hidden_size // model.config.num_attention_heads)
                    shape = (1, model.config.num_key_value_heads, prefix_len, head_dim)
                    for _ in range(model.config.num_hidden_layers):
                        key = torch.empty(shape, dtype=model_dtype)
                        value = torch.empty(shape, dtype=model_dtype)
                        dist.recv(key, 0)
                        dist.recv(value, 0)
                        pairs.append((key, value))
                else:
                    metadata = {"sha256": cache_sha256(holder_pairs),
                                "bytes": cache_bytes(holder_pairs)}
                    pairs = [(key.clone(), value.clone()) for key, value in holder_pairs]
                if any(t.dtype != model_dtype for pair in pairs for t in pair):
                    raise RuntimeError("Received cache/wire dtype differs from model precision.")
                digest = cache_sha256(pairs)
                integrity = digest == metadata["sha256"] and cache_bytes(pairs) == metadata["bytes"]
                if not integrity:
                    raise RuntimeError("KV payload integrity check failed.")
                restored = pairs_to_cache(pairs, model.config, "cuda:0")
                if int(restored.get_seq_length()) != prefix_len:
                    raise RuntimeError("Restored cache has the wrong length.")
                restored_pairs = cache_to_cpu_pairs(restored)
                restored_sha256 = cache_sha256(restored_pairs)
                restored_integrity = (restored_sha256 == digest
                    and cache_bytes(restored_pairs) == metadata["bytes"]
                    and all(t.dtype == model_dtype for pair in restored_pairs for t in pair))
                del restored_pairs
                if not restored_integrity:
                    raise RuntimeError("Restored GPU cache does not preserve exact prefix bytes and dtype.")
                event("cache_received_and_restored", case_id=case_id,
                      sha256=digest, bytes=cache_bytes(pairs))
                hit_audit = []
                hit_logits, hit_tokens, fetched_length = sequence(suffix, restored, generated_len,
                                                  hit_audit, "fetched_cache_decode")
                torch.cuda.synchronize()
                case["audit"].extend(hit_audit)
                checks = {
                    "full_vs_local": compare(ref_logits, control_logits, ref_tokens, control_tokens),
                    "full_vs_fetch": compare(ref_logits, hit_logits, ref_tokens, hit_tokens),
                    "local_vs_fetch": compare(control_logits, hit_logits, control_tokens, hit_tokens),
                }
                for name, check in checks.items():
                    check["required_for_acceptance"] = name in required_checks
                numerical_pass = all(checks[name]["logits_pass"] and checks[name]["tokens_match"]
                                     for name in required_checks)
                expected_length = prefix_len + suffix_len + generated_len - 1
                lengths_pass = all(length == expected_length
                    for length in (reference_length, control_length, fetched_length))
                no_recompute = ([entry["input_tokens"] for entry in hit_audit]
                                == [suffix_len] + [1] * (generated_len - 1)
                                and hit_audit[0]["cache_before"] == prefix_len
                                and hit_audit[-1]["cache_after"] == prefix_len + suffix_len + generated_len - 1)
                case.update(role="destination", checks=checks, integrity_pass=integrity,
                            restored_integrity_pass=restored_integrity,
                            required_checks=required_checks, numerical_pass=numerical_pass,
                            cache_lengths_pass=lengths_pass, expected_final_cache_length=expected_length,
                            final_cache_lengths={"full": reference_length, "local": control_length,
                                                 "fetched": fetched_length},
                            no_prefix_recompute_pass=no_recompute,
                            received_sha256=digest, bytes_received=cache_bytes(pairs),
                            restored_sha256=restored_sha256, local_prefix_sha256=local_prefix_sha256,
                            local_prefix_matches_received=local_prefix_sha256 == digest,
                            cache_metadata=cache_metadata(pairs),
                            reference_token_ids=ref_tokens, local_token_ids=control_tokens,
                            fetched_token_ids=hit_tokens,
                            diagnostic_request_seconds=time.monotonic() - request_start,
                            passed=integrity and restored_integrity and local_prefix_sha256 == digest and lengths_pass
                                and no_recompute and numerical_pass)
                event("case_checked", case_id=case_id, passed=case["passed"])
                del restored, pairs, ref_logits, control_logits, hit_logits
            report["cases"].append(case)
            if cross:
                passed = torch.tensor([int(case["passed"])], dtype=torch.int64)
                dist.all_reduce(passed, op=dist.ReduceOp.MIN)
                if not bool(passed.item()):
                    raise RuntimeError(f"Case {case_id} failed; inspect destination diagnostics.")
            elif not case["passed"]:
                raise RuntimeError(f"Case {case_id} failed; tolerances were not relaxed.")
            del holder_pairs
    report["passed"] = True
    report["scope"] = ("cross-node native-fetch correctness only" if cross
                       else "single-GPU split-cache and cache reconstruction correctness only")
    if cross:
        dist.destroy_process_group()


def entrypoint():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["local", "crossnode"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Use a scheduled GPU allocation; never run on a login node.")
    if not all(math.isfinite(v) and v > 0 for v in [args.atol, args.rtol]):
        raise SystemExit("Tolerances must be finite and positive.")
    if args.atol != 1e-4 or args.rtol != 1e-4:
        raise SystemExit("This preregistered native-fetch protocol requires atol=rtol=1e-4.")
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    report = {"stage": "E0_native_fetch", "passed": False,
              "requested_dtype": args.dtype,
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "rank": rank,
              "hostname": socket.gethostname(),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "python_executable": sys.executable,
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), Path(__file__).with_name("e0_cache_utils.py"),
                                          Path(__file__).with_name("e0_native_precision_protocol.json")]}}
    events_path = result_dir / f"rank-{rank}-events.jsonl"
    with events_path.open("w", encoding="utf-8") as stream:
        def event(name, **fields):
            stream.write(json.dumps({"event": name, "rank": rank,
                                     "monotonic_s": time.monotonic(), **fields}) + "\n")
            stream.flush()
        try:
            main(args, report, event)
        except Exception as exc:
            report.update(passed=False, error=f"{type(exc).__name__}: {exc}",
                          traceback=traceback.format_exc())
            event("failed", error=report["error"])
        finally:
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            (result_dir / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rank": rank, "passed": report["passed"],
                      "result": str(result_dir / f"rank-{rank}.json"),
                      "error": report.get("error")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(entrypoint())
