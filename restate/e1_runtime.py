"""Qwen2 adapters for E1; no model loading, transport, timers, or CUDA profiling.

Prefix tensors and PrefixStore belong to the caller. Decode owns only its growing
delta; HolderAttention owns one request's replica. GPU correctness must be checked
against the E0 controls before results from this adapter are used as evidence.
"""

from contextlib import nullcontext
import inspect
import threading
import weakref

from e0_layered_runtime import LayeredDecoder, PrefixStore, full_attention


class NullTimeline:
    """Must be truthy: LayeredDecoder otherwise creates a CUDA event timeline."""
    def __bool__(self):
        return True

    def measure(self, *args, **kwargs):
        return nullcontext()


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class Engine:
    def __init__(self, model, compute_queue=None):
        import torch

        self.torch, self.model, self.compute_queue = torch, model, compute_queue
        config = model.config
        if config.model_type != "qwen2" or getattr(config, "use_sliding_window", False):
            raise ValueError("Only full-attention Qwen2 is supported")
        if model.training:
            raise ValueError("The caller must put the model in eval mode")
        parameter = next(model.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        if self.device.type != "cuda" or self.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Expected a CUDA FP32 or BF16 model")
        if any(p.device != self.device or (p.is_floating_point() and p.dtype != self.dtype)
               for p in model.parameters()):
            raise ValueError("Parameters must use one GPU and one floating dtype")
        self.layers = config.num_hidden_layers
        self.query_heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.query_heads
        if len(model.model.layers) != self.layers or config.hidden_size != self.query_heads * self.head_dim:
            raise ValueError("Unsupported Qwen2 head dimensions")
        for layer in model.model.layers:
            attention = layer.self_attn
            if (attention.head_dim != self.head_dim or attention.q_proj.out_features != self.query_heads * self.head_dim
                or attention.k_proj.out_features != self.kv_heads * self.head_dim
                or attention.v_proj.out_features != self.kv_heads * self.head_dim):
                raise ValueError("Q/K/V projection dimensions disagree with the model config")
        parameter = inspect.signature(model.forward).parameters.get("logits_to_keep")
        self.logits_to_keep_enabled = parameter is not None and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        if not self.logits_to_keep_enabled:
            raise ValueError("This E1 adapter requires explicit HF logits_to_keep support")

    def _tokens(self, tokens, name):
        torch = self.torch
        if (not isinstance(tokens, torch.Tensor) or tokens.device != self.device
            or tokens.dtype != torch.int64 or tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1):
            raise ValueError(f"{name} must be nonempty int64 token IDs shaped [1, T] on {self.device}")

    def _prefix_pairs(self, pairs, expected_length=None):
        torch = self.torch
        if not isinstance(pairs, (list, tuple)) or len(pairs) != self.layers:
            raise ValueError("Expected exactly one K/V pair for every layer")
        length = None
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Each prefix entry must be a K/V pair")
            for tensor in pair:
                if (not isinstance(tensor, torch.Tensor) or tensor.device != self.device or tensor.dtype != self.dtype
                    or tensor.layout != torch.strided or tensor.ndim != 4):
                    raise ValueError("Prefix tensor layout, device, or dtype mismatch")
                if length is None:
                    length = int(tensor.shape[-2])
                if length < 1 or tuple(tensor.shape) != (1, self.kv_heads, length, self.head_dim):
                    raise ValueError("Prefix tensors must all have shape [1, KV heads, P, head dim]")
        if expected_length is not None and length != expected_length:
            raise ValueError("Prefix length mismatch")
        return length

    def operation(self, kind):
        return self.compute_queue.operation(kind) if self.compute_queue is not None else nullcontext()

    def _hf_forward(self, tokens, cache=None):
        with self.operation("hf_prefix" if cache is None else "hf_token"):
            return self._hf_forward_unqueued(tokens, cache)

    def _hf_forward_unqueued(self, tokens, cache=None):
        torch = self.torch
        before = 0 if cache is None else int(cache.get_seq_length())
        count = int(tokens.shape[1])
        output = self.model(input_ids=tokens, past_key_values=cache, use_cache=True, logits_to_keep=1,
            position_ids=torch.arange(before, before + count, device=self.device).unsqueeze(0),
            attention_mask=torch.ones((1, before + count), dtype=torch.long, device=self.device))
        if tuple(output.logits.shape[:2]) != (1, 1) or int(output.past_key_values.get_seq_length()) != before + count:
            raise RuntimeError("HF cache length or logits_to_keep contract failed")
        return output

    def prefix(self, tokens):
        """One HF full-prefix prefill; return ready, caller-owned GPU K/V pairs."""
        self._tokens(tokens, "prefix")
        with self.torch.inference_mode():
            output = self._hf_forward(tokens)
            pairs = [(layer.keys, layer.values) for layer in output.past_key_values.layers]
            self._prefix_pairs(pairs, int(tokens.shape[1]))
            del output
        self.torch.cuda.synchronize(self.device)
        return pairs

    def store_from_prefix(self, pairs):
        """Borrow complete GPU prefix tensors; readiness precedes ordered publication.

        No tensor copy is made. The caller must keep the prefix immutable until
        every decode using the returned store has completed.
        """
        self._prefix_pairs(pairs)
        self.torch.cuda.synchronize(self.device)
        store = PrefixStore(self.layers)
        for index, (key, value) in enumerate(pairs):
            store.publish(index, key, value, key.numel() * key.element_size() + value.numel() * value.element_size())
        return store

    def decode(self, store, prefix_length, suffix_tokens, generation, remote_attention=None,
               on_token=None, capture_logits=False):
        """Tokenwise suffix/decode. Callback receives (zero-based step, token ID).

        Background publication is allowed; its producer must validate shape/dtype
        and finish GPU copies before PrefixStore.publish. This method neither
        clears the borrowed store nor closes the caller's remote endpoint.
        """
        torch = self.torch
        _positive_integer(prefix_length, "prefix_length")
        _positive_integer(generation, "generation")
        self._tokens(suffix_tokens, "suffix")
        if not isinstance(store, PrefixStore) or len(store.values) != self.layers:
            raise ValueError("Expected a PrefixStore with this model's layer count")
        if remote_attention is not None and not callable(remote_attention):
            raise TypeError("remote_attention must be callable")
        if on_token is not None and not callable(on_token):
            raise TypeError("on_token must be callable")
        with store.lock:
            ready_pairs = [(index, pair) for index, pair in enumerate(store.values) if pair is not None]
        for _, pair in ready_pairs:
            # Validate each present layer without retaining a second prefix cache.
            for tensor in pair:
                if (tensor.device != self.device or tensor.dtype != self.dtype
                    or tuple(tensor.shape) != (1, self.kv_heads, prefix_length, self.head_dim)):
                    raise ValueError("Published prefix shape, device, or dtype mismatch")
        del ready_pairs
        decoder = LayeredDecoder(self.model, prefix_length, store, remote_attention, NullTimeline(), self.compute_queue)
        tokens, rows = [], []
        scores = token = None
        state = {}
        try:
            with torch.inference_mode():
                for index in range(suffix_tokens.shape[1]):
                    scores = decoder.forward(suffix_tokens[:, index:index+1])
                for step in range(generation):
                    with self.operation("greedy_selection"):
                        token = scores.argmax(dim=-1, keepdim=True)
                        token_id = int(token.item())
                        tokens.append(token_id)
                        if capture_logits:
                            rows.append(scores.detach().to(device="cpu", copy=True).float())
                    if on_token is not None:
                        on_token(step, token_id)
                    if step + 1 < generation:
                        scores = decoder.forward(token)
            expected = int(suffix_tokens.shape[1]) + generation - 1
            lengths = [[int(t.shape[-2]) for t in pair] for pair in decoder.delta]
            if lengths != [[expected, expected] for _ in range(self.layers)] or decoder.position != prefix_length + expected:
                raise RuntimeError("Destination delta length or final position mismatch")
            state.update(mode_counts=dict(decoder.mode_counts), transitions=[dict(item) for item in decoder.transitions],
                delta_lengths=lengths, final_position=decoder.position, prefix_length=prefix_length,
                suffix_forward_calls=int(suffix_tokens.shape[1]), generation_forward_calls=generation - 1,
                decode_audit=[dict(item) for item in decoder.audit], prefix_store_borrowed=True,
                logits_captured=bool(capture_logits))
        finally:
            refs = [weakref.ref(t) for pair in decoder.delta if pair is not None for t in pair]
            decoder.delta.clear()
            scores = token = None
            del decoder
            state["owned_delta_entries_cleared"] = True
            state["owned_delta_references_released"] = all(ref() is None for ref in refs)
        if not state["owned_delta_references_released"]:
            raise RuntimeError("A decode delta tensor is still retained, possibly by the remote callback")
        return {"token_ids": tokens, "logits": rows, "state_report": state}

    def HF_reference(self, prefix, suffix, generation, *, on_token=None, capture_logits=True):
        """HF full-prefix prefill then tokenwise suffix and greedy decode.

        Can also serve as a native execution control with capture_logits=False.
        The callback and output format match decode; all HF cache state is owned
        here and released before returning. No generation beyond the first G
        requested greedy token IDs is performed.
        """
        torch = self.torch
        _positive_integer(generation, "generation")
        self._tokens(prefix, "prefix")
        self._tokens(suffix, "suffix")
        if on_token is not None and not callable(on_token):
            raise TypeError("on_token must be callable")
        output = cache = scores = token = None
        tokens, rows, state = [], [], {}
        try:
            with torch.inference_mode():
                output = self._hf_forward(prefix)
                cache = output.past_key_values
                for index in range(suffix.shape[1]):
                    output = self._hf_forward(suffix[:, index:index+1], cache)
                    cache = output.past_key_values
                for step in range(generation):
                    with self.operation("greedy_selection"):
                        scores = output.logits[:, -1, :].float()
                        token = scores.argmax(dim=-1, keepdim=True)
                        token_id = int(token.item())
                        tokens.append(token_id)
                        if capture_logits:
                            rows.append(scores.detach().to(device="cpu", copy=True))
                    if on_token is not None:
                        on_token(step, token_id)
                    if step + 1 < generation:
                        output = self._hf_forward(token, cache)
                        cache = output.past_key_values
            expected = int(prefix.shape[1] + suffix.shape[1]) + generation - 1
            lengths = [int(cache.get_seq_length(index)) for index in range(self.layers)]
            if lengths != [expected] * self.layers:
                raise RuntimeError("HF reference final cache lengths disagree")
            state.update(final_cache_lengths=lengths, final_position=expected,
                prefix_prefill_tokens=int(prefix.shape[1]), suffix_forward_calls=int(suffix.shape[1]),
                generation_forward_calls=generation - 1, logits_captured=bool(capture_logits))
        finally:
            refs = ([] if cache is None else [weakref.ref(t) for layer in cache.layers for t in (layer.keys, layer.values)])
            output = cache = scores = token = None
            state["owned_cache_references_released"] = all(ref() is None for ref in refs)
        if not state["owned_cache_references_released"]:
            raise RuntimeError("HF reference cache tensors are still retained")
        return {"token_ids": tokens, "logits": rows, "state_report": state}


class HolderAttention:
    """One request's source replica; prefix is borrowed and never mutated.

    Q and K must already include RoPE at the supplied absolute position. Calls
    may skip layers after destination cutover, but each called layer's positions
    must be contiguous. Calls on a holder use one CUDA stream, protected by a
    Python lock; inputs must be ready on that stream. No GPU-parallelism claim.
    """
    def __init__(self, model, prefix_pairs, prefix_length, compute_queue=None):
        _positive_integer(prefix_length, "prefix_length")
        self.engine = Engine(model, compute_queue=compute_queue)
        self.engine._prefix_pairs(prefix_pairs, prefix_length)
        self.engine.torch.cuda.synchronize(self.engine.device)
        self.prefix_pairs = [tuple(pair) for pair in prefix_pairs]
        self.prefix_length = prefix_length
        self.replicas = [None] * self.engine.layers
        self.positions = [prefix_length] * self.engine.layers
        self.calls_per_layer = [0] * self.engine.layers
        self._lock, self._stream_id = threading.RLock(), None
        self.closed, self._closed_summary = False, None

    def call(self, layer, position, query, key, value):
        with self.engine.operation("holder_attention"):
            return self._call_unqueued(layer, position, query, key, value)

    def _call_unqueued(self, layer, position, query, key, value):
        engine, torch = self.engine, self.engine.torch
        with self._lock:
            if self.closed:
                raise RuntimeError("HolderAttention is closed")
            if type(layer) is not int or not 0 <= layer < engine.layers or type(position) is not int:
                raise ValueError("Invalid layer or absolute position")
            if position != self.positions[layer]:
                raise RuntimeError("Source replica position is not contiguous for this layer")
            stream_id = torch.cuda.current_stream(engine.device).cuda_stream
            if self._stream_id is None:
                self._stream_id = stream_id
            elif self._stream_id != stream_id:
                raise RuntimeError("A request replica cannot change CUDA streams without an explicit dependency")
            for tensor, heads in ((query, engine.query_heads), (key, engine.kv_heads), (value, engine.kv_heads)):
                if (not isinstance(tensor, torch.Tensor) or tensor.device != engine.device or tensor.dtype != engine.dtype
                    or tuple(tensor.shape) != (1, heads, 1, engine.head_dim)):
                    raise ValueError("Remote attention Q/K/V shape, device, or dtype mismatch")
            with torch.inference_mode():
                previous = self.replicas[layer]
                # Own the first K/V storage; caller input buffers may be reused.
                pair = ((key.detach().clone(), value.detach().clone()) if previous is None else
                        (torch.cat((previous[0], key), dim=-2), torch.cat((previous[1], value), dim=-2)))
                expected = position - self.prefix_length + 1
                if any(t.shape[-2] != expected for t in pair):
                    raise RuntimeError("Source replica length mismatch")
                prefix = self.prefix_pairs[layer]
                output = full_attention(engine.model.model.layers[layer].self_attn, query,
                    torch.cat((prefix[0], pair[0]), dim=-2), torch.cat((prefix[1], pair[1]), dim=-2))
                if output.dtype != engine.dtype or tuple(output.shape) != (1, 1, engine.query_heads, engine.head_dim):
                    raise RuntimeError("Remote attention output shape or dtype mismatch")
                self.replicas[layer] = pair
                self.positions[layer] += 1
                self.calls_per_layer[layer] += 1
                return output

    def summarize(self):
        with self._lock:
            if self.closed:
                return {key: list(value) if isinstance(value, list) else value for key, value in self._closed_summary.items()}
            return {"prefix_length": self.prefix_length, "next_positions": list(self.positions),
                "calls_per_layer": list(self.calls_per_layer), "total_calls": sum(self.calls_per_layer),
                "delta_lengths": [0 if pair is None else int(pair[0].shape[-2]) for pair in self.replicas],
                "prefix_borrowed": True, "closed": False}

    def close(self):
        """Release this request's replicas and borrowed references, preserving caller prefix."""
        with self._lock:
            if self.closed:
                return self.summarize()
            summary = self.summarize()
            refs = [weakref.ref(t) for pair in self.replicas if pair is not None for t in pair]
            self.replicas.clear()
            self.prefix_pairs.clear()
            self.closed = True
            summary.update(closed=True, owned_delta_references_released=all(ref() is None for ref in refs),
                           borrowed_prefix_references_dropped=True)
            self._closed_summary = summary
            if not summary["owned_delta_references_released"]:
                raise RuntimeError("Source replica tensors are still retained")
            return self.summarize()
