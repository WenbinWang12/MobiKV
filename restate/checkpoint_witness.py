"""Source-side E2 checkpoint evidence, not a GPU executor or an RPC handler.

Call from the owning _Request RPC thread at a completed token boundary. Content
capture requires the initial transfer to have terminated and its exact receiver
drain ACK to have been processed. It synchronizes/copies real tensors and belongs
outside prospective timing. Metadata mode performs neither operation and does
not establish content equality. Destination publication/delta/pending-token and
external-state evidence must still be checked by the joint matched-state gate.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import threading
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, "Invalid " + name)
    return value


def _sha(value, name):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid " + name)
    return value


class TorchTensors:
    """Real torch metadata and the existing E0 byte-exact CPU hash format."""
    evidence_kind = "real_torch_tensors"

    def __init__(self, engine):
        self.engine, self.torch = engine, engine.torch

    def metadata(self, tensor):
        torch = self.torch
        require(isinstance(tensor, torch.Tensor), "Expected a real torch.Tensor")
        require(tensor.layout == torch.strided and tensor.device.type != "meta", "Unsupported tensor layout/device")
        require(tensor.device == self.engine.device and tensor.dtype == self.engine.dtype, "Foreign cache device/dtype")
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "nbytes": tensor.numel() * tensor.element_size()}

    def synchronize(self):
        if self.engine.device.type == "cuda":
            self.torch.cuda.synchronize(self.engine.device)

    def hash_pairs(self, pairs):
        from e0_cache_utils import cache_to_cpu_pairs, cache_sha256
        snapshots = cache_to_cpu_pairs(pairs)
        try:
            return cache_sha256(snapshots)
        finally:
            snapshots.clear()


def _pair(pair, backend, engine, length, name):
    require(isinstance(pair, (tuple, list)) and len(pair) == 2, name + " is not a K/V pair")
    entries = [backend.metadata(tensor) for tensor in pair]
    expected_shape = [1, engine.kv_heads, length, engine.head_dim]
    for entry in entries:
        require(set(entry) == {"shape", "dtype", "nbytes"}, "Unexpected tensor metadata schema")
        require(entry["shape"] == expected_shape, name + " shape differs")
        require(entry["dtype"] == str(engine.dtype), name + " dtype differs")
        dtype_bytes = {"torch.float32": 4, "torch.bfloat16": 2, "torch.float16": 2}
        require(entry["dtype"] in dtype_bytes, "Unsupported E2 cache dtype")
        require(type(entry["nbytes"]) is int and entry["nbytes"] == math.prod(expected_shape) * dtype_bytes[entry["dtype"]],
                name + " payload-byte metadata differs")
    return {"key": entries[0], "value": entries[1],
            "payload_bytes": entries[0]["nbytes"] + entries[1]["nbytes"]}


def _idle(snapshot):
    return (snapshot.get("active_gpu_operations") == 0 and
            snapshot.get("queued_gpu_operations") == 0 and
            snapshot.get("model_execution_serialized") is True)


def _transfer_state(request, layer_bytes, committed_layers, strict):
    """Snapshot immutable completed segments; an active mode is observational."""
    migration = None if request.migration is None else request.migration.snapshot()
    transfers = copy.deepcopy(request.report.get("transfers", []))
    thread_alive = request.migration_thread is not None and request.migration_thread.is_alive()
    result = {"sender_frontier": request.frontier, "transfer_index": request.transfer_index,
              "migration": migration, "segments": transfers, "migration_thread_alive": thread_alive,
              "migration_error": request.migration_error, "receiver_GPU_publication_independently_verified": False}
    if not strict:
        return result
    require(not thread_alive and request.migration_error is None, "Migration is active or failed")
    require(request.frontier == committed_layers, "Source frontier differs from declared receiver commitment")
    require(len(transfers) == request.transfer_index, "Missing transfer terminal record")
    frontier = total = 0
    for index, segment in enumerate(transfers, 1):
        start, end, next_layer = (integer(segment.get(k), k) for k in ("start", "end", "next_layer"))
        require(segment.get("transfer_index") == index and start == frontier and start < end <= len(layer_bytes),
                "Transfer segments are not contiguous/complete")
        require(start <= next_layer <= end, "Invalid terminal frontier")
        control = segment.get("control", {})
        expected_payload = sum(layer_bytes[start:next_layer])
        require(control.get("request_id") == request.request_id and
                control.get("state") in ("complete", "stopped") and
                control.get("complete_layers") == next_layer-start and
                control.get("payload_bytes") == expected_payload and
                control.get("terminal_sent") is True and control.get("receiver_drained_ack") is True and
                control.get("in_flight_layer") is None and control.get("error") is None,
                "Transfer lacks an exact completed-frame receiver-drain proof")
        require(control["state"] != "complete" or next_layer == end, "Complete transfer has missing frames")
        total += expected_payload
        frontier = next_layer
    require(frontier == committed_layers, "Terminal history does not reach declared frontier")
    if not transfers:
        require(migration is None and request.migration_thread is None, "Unrecorded migration state")
    else:
        require(migration == transfers[-1]["control"], "Live migration differs from retained terminal/drain state")
    result.update(complete_frame_payload_bytes=total, quiescent_transfer_verified=True)
    return result


def capture_holder_witness(request, *, mode, model_identity, processed_delta_tokens,
                           committed_layers, _tensor_backend=None):
    """Return a JSON-only source witness for an actual campaign _Request.

    model_identity requires model_id/model_revision/model_manifest_sha256 from the
    verified execution context, not the network caller's unchecked claims.
    processed_delta_tokens is S+D-1, not the total requested output length.
    _tensor_backend exists for labeled CPU fixtures only. It cannot produce a
    production-qualified content witness. No references to tensors are returned.
    """
    require(mode in ("metadata", "content"), "Mode must explicitly be metadata or content")
    require(isinstance(model_identity, dict) and set(model_identity) ==
            {"model_id", "model_revision", "model_manifest_sha256"}, "Model identity schema differs")
    require(all(isinstance(v, str) and v for v in model_identity.values()), "Missing model identity")
    _sha(model_identity["model_manifest_sha256"], "model manifest SHA")
    processed_delta_tokens = integer(processed_delta_tokens, "processed_delta_tokens")
    committed_layers = integer(committed_layers, "committed_layers")
    engine = request.engine
    layers = integer(engine.layers, "model layers", 1)
    require(committed_layers <= layers, "Committed frontier exceeds model")
    owner = request.thread
    require(owner is not None and owner.ident == threading.get_ident(), "Witness must run in owning serialized request RPC thread")
    attention = request.attention
    require(attention is not None, "Holder attention has not been initialized")
    backend = TorchTensors(engine) if _tensor_backend is None else _tensor_backend
    # External fixture injection can never grant production content acceptance.
    real_backend = _tensor_backend is None
    started = time.monotonic()
    strict = mode == "content"
    with attention._lock, request.lock:
        require(not attention.closed, "Holder request was already closed")
        prefix = request.prefix
        p = integer(prefix.get("length"), "prefix length", 1)
        require(attention.prefix_length == p and len(prefix["pairs"]) == layers and
                len(attention.prefix_pairs) == layers and len(attention.replicas) == layers and
                len(attention.positions) == layers and len(attention.calls_per_layer) == layers,
                "Holder prefix/replica structure differs from model")
        token_hash = _sha(prefix.get("token_ids_sha256"), "prefix token SHA")
        declared_prefix_hash = _sha(prefix.get("cache_sha256"), "prepared prefix content SHA")
        prefix_rows, delta_rows, delta_pairs = [], [], []
        for layer in range(layers):
            pair = prefix["pairs"][layer]
            require(all(x is y for x, y in zip(pair, attention.prefix_pairs[layer])), "Holder no longer borrows actual immutable prefix tensors")
            row = _pair(pair, backend, engine, p, "prefix")
            prefix_rows.append({"layer": layer, **row})
            calls = integer(attention.calls_per_layer[layer], "calls per layer")
            position = integer(attention.positions[layer], "holder position", p)
            require(position == p+calls and calls <= processed_delta_tokens, "Holder position/calls exceed processed history")
            pair = attention.replicas[layer]
            require((pair is None) == (calls == 0), "Replica presence does not match calls")
            if layer >= committed_layers:
                require(calls == processed_delta_tokens, "Missing-layer holder delta is not current")
            metadata = None if pair is None else _pair(pair, backend, engine, calls, "delta")
            delta_rows.append({"layer": layer, "relevant_for_future_remote": layer >= committed_layers,
                "next_position": position, "delta_length": calls, "calls": calls,
                "tensors": metadata, "content_sha256": None})
            delta_pairs.append(pair)
        total_prefix_bytes = sum(x["payload_bytes"] for x in prefix_rows)
        require(prefix.get("cache_bytes") == total_prefix_bytes, "Prepared prefix byte count differs")
        audits = request.report.get("remote_audit", [])
        history = []
        next_positions = [p]*layers
        for row in audits:
            layer = integer(row.get("layer"), "RPC layer")
            position = integer(row.get("position"), "RPC position", p)
            require(layer < layers and position == next_positions[layer], "RPC history has holes/duplicates")
            if history:
                require((position, layer) > (history[-1]["position"], history[-1]["layer"]), "RPC history is not token/layer ordered")
            next_positions[layer] += 1
            history.append({"layer": layer, "position": position})
        require(next_positions == list(attention.positions) and request.report.get("remote_calls") == len(history),
                "RPC history does not establish actual replica positions")
        queue_before = copy.deepcopy(request.gpu.snapshot())
        transfer = _transfer_state(request, [x["payload_bytes"] for x in prefix_rows], committed_layers, strict)
        if strict:
            require(not request.report.get("errors") and not request.report.get("exception"),
                    "Failed source request cannot establish a conditional checkpoint")
            require(_idle(queue_before), "Conditional content checkpoint has pending GPU work")
            backend.synchronize()
            prefix_hash = backend.hash_pairs(prefix["pairs"])
            require(prefix_hash == declared_prefix_hash, "Immutable prefix actual content changed")
            for pair, row in zip(delta_pairs, delta_rows):
                if pair is not None:
                    row["content_sha256"] = backend.hash_pairs([pair])
                    _sha(row["content_sha256"], "delta content SHA")
            queue_after = copy.deepcopy(request.gpu.snapshot())
            require(_idle(queue_after), "GPU work appeared during conditional checkpoint")
            require(queue_before.get("next_sequence") == queue_after.get("next_sequence"), "GPU queue advanced during checkpoint")
            require(transfer == _transfer_state(request, [x["payload_bytes"] for x in prefix_rows], committed_layers, True),
                    "Migration changed during conditional checkpoint")
        else:
            prefix_hash = None
            queue_after = None
        scientific = {"model": copy.deepcopy(model_identity), "prefix_length": p,
            "prefix_token_ids_sha256": token_hash, "prefix_content_sha256": prefix_hash,
            "prefix_layers": prefix_rows, "processed_delta_tokens": processed_delta_tokens,
            "expected_destination_next_position": p+processed_delta_tokens,
            "committed_layers": committed_layers, "holder_delta_layers": delta_rows,
            "actual_RPC_history": history,
            "transfer_boundary": None if not strict else {"sender_frontier": transfer["sender_frontier"],
                "complete_frame_payload_bytes": transfer["complete_frame_payload_bytes"],
                "terminal_and_receiver_drain_verified": True, "migration_state": None if transfer["migration"] is None else transfer["migration"]["state"]}}
        result = {"schema_version": 1, "kind": "E2_holder_checkpoint_witness", "mode": mode,
            "tensor_evidence_kind": backend.evidence_kind,
            "source_content_validated": strict and real_backend,
            "matched_state_passed": False,
            "holder_equivalence_sha256": digest(scientific) if strict else None,
            "scientific_state": scientific,
            "provenance": {"request_id": request.request_id, "prefix_key": request.spec.get("prefix_key"),
                "prepared_prefix_cache_sha256": declared_prefix_hash,
                "request_owner_thread_ident": owner.ident, "transfer": transfer,
                "gpu_before": queue_before, "gpu_after": queue_after},
            "capture": {"started_monotonic_s": started, "finished_monotonic_s": time.monotonic(),
                "tensor_payload_bytes_copied": 0 if not strict else total_prefix_bytes + sum(
                    0 if r["tensors"] is None else r["tensors"]["payload_bytes"] for r in delta_rows),
                "cuda_completion_synchronized": strict and real_backend and str(engine.device).startswith("cuda"),
                "main_path_eligible": not strict},
            "limits": ["Holder-only evidence cannot certify destination GPU publication, delta content, pending token or global matched state.",
                       "Receiver drain is the source's retained protocol ACK, not a second independent receiver observation.",
                       "Metadata capture is not an atomic cross-thread tensor/content or transfer snapshot.",
                       "Content capture overhead must precede prospective timing; model/context provenance is independently bound by the runner."]}
        result["capture"]["duration_s"] = result["capture"]["finished_monotonic_s"]-started
        result["witness_sha256"] = digest(result)
        return result


def validated_holder_state(witness):
    """Return canonical content only for a hash-bound real-tensor witness.

    This is one input to the external destination+holder matching procedure. It
    does not itself set matched_state_passed or accept an experimental record.
    """
    require(isinstance(witness, dict), "Missing holder witness")
    raw = {k: v for k, v in witness.items() if k != "witness_sha256"}
    require(witness.get("witness_sha256") == digest(raw), "Holder witness artifact was changed")
    require(witness.get("kind") == "E2_holder_checkpoint_witness" and witness.get("mode") == "content" and
            witness.get("source_content_validated") is True and witness.get("tensor_evidence_kind") == "real_torch_tensors",
            "Metadata or CPU fixture cannot establish real content equivalence")
    require(witness.get("holder_equivalence_sha256") == digest(witness["scientific_state"]), "Canonical holder content hash differs")
    return copy.deepcopy(witness["scientific_state"])
