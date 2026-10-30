"""Real token-step Qwen continuation with destination-state checkpoints.

Caller owns immutable prefix KV, transport, migration, clocks, and policy. This
adapter performs no prefill: it reuses LayeredDecoder and permits a callback to
publish full-prefix recomputation/fetch results while keeping request delta KV.
GPU numerical acceptance and source-side matched-state checks remain external.
"""
from __future__ import annotations

import hashlib
import json
import time
import weakref

from e0_layered_runtime import LayeredDecoder, PrefixStore
from e1_runtime import NullTimeline


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _lengths(decoder):
    return [[int(tensor.shape[-2]) for tensor in pair] for pair in decoder.delta]


def _token_list(tokens):
    return [int(value) for value in tokens.detach().to(device="cpu", copy=True).tolist()[0]]


def checkpoint_snapshot(decoder, prefix_tokens, suffix_tokens, token_ids, *, capture_checkpoint_hashes=False,
                        recorded_committed_frontier=None, checkpoint_input_ids=None):
    """Capture destination state before the continuation decision starts.

    Foreground must be at a complete-token boundary. Caller must quiesce prefix
    publication if matching an exact frontier across branches. Published tensors
    are borrowed briefly under the store lock, then copied/hash-tested without
    holding it. Entries must remain immutable; publication must imply GPU-ready.
    The latest emitted token is pending and has NOT been appended to delta yet.
    A recorded frontier is a diagnostic borrow/hash VIEW of an actually received
    prefix subset, not a mutation of PrefixStore or a reconstruction of in-flight
    migration. Physical publication may be ahead of the recorded logical view.
    """
    if type(capture_checkpoint_hashes) is not bool:
        raise ValueError("capture_checkpoint_hashes must be an explicit boolean")
    if not token_ids or any(type(token) is not int or token < 0 for token in token_ids):
        raise ValueError("Checkpoint requires at least one emitted token")
    if checkpoint_input_ids is not None:
        if not isinstance(checkpoint_input_ids, dict) or set(checkpoint_input_ids) != {"prefix", "suffix"}:
            raise ValueError("Checkpoint host inputs require prefix and suffix only")
        for name, tensor in (("prefix", prefix_tokens), ("suffix", suffix_tokens)):
            ids = checkpoint_input_ids[name]
            if (not isinstance(ids, list) or len(ids) != tensor.shape[1] or
                    any(type(value) is not int or value < 0 for value in ids)):
                raise ValueError("Checkpoint host token input shape/type differs")
        prefix_ids, suffix_ids = (list(checkpoint_input_ids[name]) for name in ("prefix", "suffix"))
        if capture_checkpoint_hashes and (prefix_ids != _token_list(prefix_tokens) or suffix_ids != _token_list(suffix_tokens)):
            raise ValueError("Actual device tokens differ from checkpoint host input identity")
    else:
        prefix_ids, suffix_ids = _token_list(prefix_tokens), _token_list(suffix_tokens)
    expected = len(suffix_ids) + len(token_ids) - 1
    lengths = _lengths(decoder)
    if decoder.position != len(prefix_ids) + expected or any(pair != [expected, expected] for pair in lengths):
        raise RuntimeError("Checkpoint delta position/length is not a token boundary")
    store = decoder.prefix_store
    with store.lock:
        physical_frontier = store.frontier
        physical_layers = [index for index, pair in enumerate(store.values) if pair is not None]
        if recorded_committed_frontier is not None:
            if (type(recorded_committed_frontier) is not int or
                    not 0 <= recorded_committed_frontier <= physical_frontier):
                raise ValueError("Recorded checkpoint frontier requires actually published KV for every borrowed layer")
        frontier = physical_frontier if recorded_committed_frontier is None else recorded_committed_frontier
        published = [(index, store.values[index]) for index in range(frontier)]
    if physical_layers != list(range(physical_frontier)) or any(pair is None for _, pair in published):
        raise RuntimeError("Checkpoint requires ascending prefix publication")
    if recorded_committed_frontier is not None and any(
            mode == "local" and layer >= frontier for layer, mode in enumerate(decoder.last_modes)):
        raise ValueError("Recorded frontier contradicts a layer already used locally")
    state = {
        "position": decoder.position, "prefix_length": len(prefix_ids), "delta_lengths": lengths,
        "mode_counts": dict(decoder.mode_counts), "last_modes": list(decoder.last_modes),
        "transitions": [{"layer": item["layer"], "position": item["position"]} for item in decoder.transitions],
        "prefix_frontier": frontier, "committed_layers": [index for index, _ in published],
        "physical_prefix_frontier": physical_frontier, "physical_committed_layers": physical_layers,
        "logical_committed_frontier": frontier,
        "recorded_frontier_view": recorded_committed_frontier is not None,
        "frontier_view_scope": ("recorded diagnostic borrow/hash subset of actual GPU-ready tensors; not a physical migration snapshot"
                               if recorded_committed_frontier is not None else "actual prefix publication at capture"),
        "prefix_token_ids": prefix_ids, "suffix_token_ids": suffix_ids,
        "emitted_token_ids": list(token_ids), "pending_token_id": token_ids[-1],
        "processed_generated_tokens": len(token_ids) - 1,
        "generated_count": len(token_ids),
        "processed_history_token_ids": prefix_ids + suffix_ids + list(token_ids[:-1]),
        "content_hashes_captured": bool(capture_checkpoint_hashes),
        "delta_sha256": None, "committed_prefix_sha256": None,
        "scope": "destination state only; holder replicas, queues, in-flight transfer require external matching",
    }
    if capture_checkpoint_hashes:
        from e0_cache_utils import cache_sha256, cache_to_cpu_pairs
        state["delta_sha256"] = cache_sha256(cache_to_cpu_pairs(decoder.delta))
        pairs_digest = cache_sha256(cache_to_cpu_pairs([pair for _, pair in published])) if published else None
        state["committed_prefix_sha256"] = _canonical_hash({"layers": state["committed_layers"], "pairs_sha256": pairs_digest})
    state["state_sha256"] = _canonical_hash(state)
    return state


_EVENT_FIELDS = {"state_sha256", "history_elapsed_seconds", "history_compute_seconds_before_snapshot",
                 "checkpoint_capture_seconds", "callback_seconds", "callback_result", "checkpoint_index",
                 "checkpoint_capture_started_monotonic_s", "checkpoint_capture_finished_monotonic_s",
                 "callback_started_monotonic_s", "callback_finished_monotonic_s"}
_PHYSICAL_VIEW_FIELDS = {"physical_prefix_frontier", "physical_committed_layers",
                         "recorded_frontier_view", "frontier_view_scope"}


def compare_checkpoint_snapshots(left, right, *, require_content_hashes=True,
                                 allow_diagnostic_logical_view=False):
    """Compare canonical destination state; timers/callback results are excluded.

    Default requires actual K/V content hashes. A metadata-only comparison must
    be explicitly requested and cannot certify matched tensor contents.
    Logical-view comparison additionally requires an explicit opt-in. It checks
    only the recorded destination tensor/route view, never equal physical
    migration progress, queues, frame ownership or holder state.
    """
    if type(require_content_hashes) is not bool or type(allow_diagnostic_logical_view) is not bool:
        raise ValueError("Checkpoint comparison flags must be booleans")
    if require_content_hashes and not (left.get("content_hashes_captured") is True and right.get("content_hashes_captured") is True):
        raise ValueError("Matched tensor-state audit requires captured content hashes")
    if require_content_hashes:
        for snapshot in (left, right):
            for field in ("delta_sha256", "committed_prefix_sha256"):
                digest = snapshot.get(field)
                if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                    raise ValueError("Missing or malformed tensor-content hash")
    payloads = [{key: value for key, value in item.items() if key not in _EVENT_FIELDS} for item in (left, right)]
    hashes = [_canonical_hash(payload) for payload in payloads]
    if any(item.get("state_sha256") != digest for item, digest in zip((left, right), hashes)):
        raise ValueError("Checkpoint canonical state SHA differs from its data")
    if allow_diagnostic_logical_view:
        payloads = [{key: value for key, value in item.items() if key not in _PHYSICAL_VIEW_FIELDS} for item in payloads]
    if payloads[0] != payloads[1]:
        raise ValueError("Destination checkpoint states differ")
    return {"passed": True, "state_sha256": _canonical_hash(payloads[0]),
            "left_snapshot_sha256": hashes[0], "right_snapshot_sha256": hashes[1],
            "content_hashes_checked": require_content_hashes,
            "physical_publication_compared": not allow_diagnostic_logical_view,
            "scope": ("diagnostic logical destination tensor/route view only; physical/in-flight migration, holder and queues not matched"
                      if allow_diagnostic_logical_view else "destination checkpoint only; external holder/transport state not audited")}


def _checkpoint_schedule(generated, checkpoint_generated, checkpoints_generated, on_checkpoint,
                         capture_checkpoint_hashes, recorded_checkpoint_frontiers, capture_logits):
    if checkpoint_generated is not None and checkpoints_generated is not None:
        raise ValueError("checkpoint_generated and checkpoints_generated are mutually exclusive")
    if checkpoints_generated is not None:
        if not isinstance(checkpoints_generated, (list, tuple)) or not checkpoints_generated:
            raise ValueError("checkpoints_generated must be a nonempty ordered list/tuple")
        points = tuple(checkpoints_generated)
    else:
        points = () if checkpoint_generated is None else (checkpoint_generated,)
    if any(type(point) is not int or not 1 <= point < generated for point in points):
        raise ValueError("Every checkpoint must be >=1 and strictly before the last output token")
    if tuple(sorted(set(points))) != points:
        raise ValueError("Checkpoints must be unique and strictly increasing")
    if points and not callable(on_checkpoint):
        raise TypeError("Checkpoint selection requires an on_checkpoint callback")
    if not points and (on_checkpoint is not None or capture_checkpoint_hashes):
        raise ValueError("Checkpoint callback/hashes require a checkpoint schedule")
    views = {} if recorded_checkpoint_frontiers is None else recorded_checkpoint_frontiers
    if not isinstance(views, dict) or any(type(d) is not int or d not in points or type(f) is not int or f < 0
                                          for d, f in views.items()):
        raise ValueError("Recorded frontiers must map scheduled generated counts to nonnegative integers")
    if recorded_checkpoint_frontiers is not None and (not points or not capture_logits):
        raise ValueError("Recorded frontier views are only allowed in checkpointed diagnostic logits replay")
    return points, dict(views)


def decode_request(engine, prefix_store, prefix_tokens, suffix_tokens, generated,
                   remote_attention=None, on_token=None, on_checkpoint=None,
                   checkpoint_generated=None, capture_logits=False, *, capture_checkpoint_hashes=False,
                   checkpoints_generated=None, recorded_checkpoint_frontiers=None, checkpoint_input_ids=None):
    """Decode G greedy tokens, with callbacks after selected tokens are emitted.

    Each D is one-based and satisfies 1 <= D < G. New checkpoints_generated and
    legacy checkpoint_generated are mutually exclusive. Callback receives step,
    the live decoder, and an immutable tuple of emitted token IDs. Its snapshot is
    available at decoder.campaign_checkpoint BEFORE callback entry. Hash capture
    and GPU-to-CPU input copies finish before entry, outside prospective timing.
    Callback may publish ready prefix, but cannot replace/mutate delta or position.
    It returns optional JSON diagnostics. Actual value preservation is validated
    by numerical checks; pointer/length checks additionally reject replacement.

    Diagnostic logits are cloned on GPU per step, copied to CPU only after all
    tokens. Performance mode capture_logits=False retains no full-vocabulary rows.
    All caller prefix references remain owned by caller; request delta is cleared.
    checkpoints contains all captures; legacy checkpoint is its first entry.
    recorded_checkpoint_frontiers={D:F} is restricted to diagnostic logits runs.
    checkpoint_input_ids may hold the same host IDs used to create the input
    tensors. Metadata capture then performs no per-checkpoint device-to-host
    token copy. Content capture still reads actual device IDs and checks equality.
    """
    if type(generated) is not int or generated <= 0:
        raise ValueError("generated must be a positive integer")
    if any(callback is not None and not callable(callback) for callback in (on_token, remote_attention)):
        raise TypeError("Callbacks must be callable")
    if type(capture_logits) is not bool or type(capture_checkpoint_hashes) is not bool:
        raise ValueError("Capture flags must be booleans")
    points, recorded_views = _checkpoint_schedule(generated, checkpoint_generated, checkpoints_generated,
        on_checkpoint, capture_checkpoint_hashes, recorded_checkpoint_frontiers, capture_logits)
    torch = engine.torch
    engine._tokens(prefix_tokens, "prefix")
    engine._tokens(suffix_tokens, "suffix")
    prefix_length = int(prefix_tokens.shape[1])
    if not isinstance(prefix_store, PrefixStore) or len(prefix_store.values) != engine.layers:
        raise ValueError("PrefixStore must match this model's layer count")
    with prefix_store.lock:
        borrowed = [pair for pair in prefix_store.values if pair is not None]
    for pair in borrowed:
        for tensor in pair:
            if (tensor.device != engine.device or tensor.dtype != engine.dtype or
                    tuple(tensor.shape) != (1, engine.kv_heads, prefix_length, engine.head_dim)):
                raise ValueError("Published prefix shape/dtype/device differs")
    del borrowed
    decoder = LayeredDecoder(engine.model, prefix_length, prefix_store, remote_attention, NullTimeline(), engine.compute_queue)
    tokens, gpu_rows, rows, state, checkpoints = [], [], [], {}, []
    checkpoint_refs = []
    scores = token = checkpoint = None
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for index in range(suffix_tokens.shape[1]):
                scores = decoder.forward(suffix_tokens[:, index:index + 1])
            for step in range(generated):
                with engine.operation("greedy_selection"):
                    token = scores.argmax(dim=-1, keepdim=True)
                    token_id = int(token.item())
                    tokens.append(token_id)
                    if capture_logits:
                        gpu_rows.append(scores.detach().clone())
                if on_token is not None:
                    on_token(step, token_id)
                if len(tokens) in points:
                    capture_start = time.perf_counter()
                    capture_started_monotonic = time.monotonic()
                    checkpoint = checkpoint_snapshot(decoder, prefix_tokens, suffix_tokens, tokens,
                        capture_checkpoint_hashes=capture_checkpoint_hashes,
                        recorded_committed_frontier=recorded_views.get(len(tokens)),
                        checkpoint_input_ids=checkpoint_input_ids)
                    capture_end = time.perf_counter()
                    checkpoint.update(history_compute_seconds_before_snapshot=capture_start - started,
                                      checkpoint_capture_seconds=capture_end - capture_start,
                                      history_elapsed_seconds=capture_end - started,
                                      checkpoint_index=len(checkpoints),
                                      checkpoint_capture_started_monotonic_s=capture_started_monotonic,
                                      checkpoint_capture_finished_monotonic_s=time.monotonic())
                    checkpoints.append(checkpoint)
                    decoder.campaign_checkpoint = checkpoint
                    before_position = decoder.position
                    checkpoint_refs.extend(weakref.ref(tensor) for pair in decoder.delta for tensor in pair)
                    before_delta = [(id(tensor), tensor.data_ptr(), tuple(tensor.shape))
                                    for pair in decoder.delta for tensor in pair]
                    callback_started = time.perf_counter()
                    checkpoint["callback_started_monotonic_s"] = time.monotonic()
                    callback_result = on_checkpoint(step, decoder, tuple(tokens))
                    checkpoint["callback_seconds"] = time.perf_counter() - callback_started
                    checkpoint["callback_finished_monotonic_s"] = time.monotonic()
                    if callback_result is not None:
                        checkpoint["callback_result"] = json.loads(json.dumps(callback_result, allow_nan=False))
                    after_delta = [(id(tensor), tensor.data_ptr(), tuple(tensor.shape))
                                   for pair in decoder.delta for tensor in pair]
                    if (decoder.position != before_position or before_delta != after_delta or
                            decoder.prefix_store is not prefix_store):
                        raise RuntimeError("Continuation callback replaced delta or changed decoder position")
                if step + 1 < generated:
                    scores = decoder.forward(token)
            expected = int(suffix_tokens.shape[1]) + generated - 1
            lengths = _lengths(decoder)
            if lengths != [[expected, expected] for _ in range(engine.layers)] or decoder.position != prefix_length + expected:
                raise RuntimeError("Destination final delta length/position mismatch")
            state.update(mode_counts=dict(decoder.mode_counts), transitions=[dict(item) for item in decoder.transitions],
                         delta_lengths=lengths, final_position=decoder.position, prefix_length=prefix_length,
                         suffix_forward_calls=int(suffix_tokens.shape[1]), generation_forward_calls=generated - 1,
                         decode_audit=[dict(item) for item in decoder.audit], prefix_store_borrowed=True,
                         logits_captured=capture_logits, checkpoint_generated=points[0] if points else None,
                         checkpoints_generated=list(points))
            if capture_logits:
                rows = [row.to(device="cpu", copy=True).float() for row in gpu_rows]
        gpu_rows.clear()
    finally:
        refs = checkpoint_refs + [weakref.ref(tensor) for pair in decoder.delta if pair is not None for tensor in pair]
        decoder.delta.clear()
        scores = token = None
        gpu_rows.clear()
        del decoder
        state["owned_delta_entries_cleared"] = True
        state["owned_delta_references_released"] = all(reference() is None for reference in refs)
    if not state["owned_delta_references_released"]:
        raise RuntimeError("A callback retained request-owned delta tensors")
    return {"token_ids": tokens, "logits": rows, "state_report": state,
            "checkpoint": checkpoints[0] if checkpoints else None, "checkpoints": checkpoints}
