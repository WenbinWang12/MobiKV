"""Qwen2 layer execution for E0 repair validation; not a serving engine.

The destination alone owns the growing KV delta. A layer's immutable prefix may
arrive in the background; switching attention location never appends KV twice.
"""

from contextlib import contextmanager, nullcontext
import threading
import time


class PrefixStore:
    def __init__(self, layers):
        self.values = [None] * layers
        self.lock = threading.Lock()
        self.frontier = 0
        self.commits = []

    def publish(self, layer, key, value, payload_bytes):
        # The caller MUST finish the H2D event before publishing either pointer.
        with self.lock:
            if layer != self.frontier or self.values[layer] is not None:
                raise RuntimeError("Prefix commits must be complete, unique, and in layer order")
            self.values[layer] = (key, value)
            self.frontier += 1
            self.commits.append({"layer": layer, "frontier": self.frontier,
                                 "monotonic_s": time.monotonic(), "payload_bytes": payload_bytes})

    def get(self, layer):
        with self.lock:
            return self.values[layer]

    def snapshot(self):
        with self.lock:
            return self.frontier, list(self.commits)


class CudaTimeline:
    """Same-device CUDA event intervals; not a substitute for kernel profiling."""
    def __init__(self, nvtx=False):
        import torch
        self.torch = torch
        self.origin = torch.cuda.Event(enable_timing=True)
        self.origin.record()
        self.origin.synchronize()
        self.records = []
        self.lock = threading.Lock()
        self.nvtx = nvtx

    @contextmanager
    def measure(self, category, **metadata):
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        start.record()
        if self.nvtx:
            label = category + "".join(f"|{key}={value}" for key, value in metadata.items())
            self.torch.cuda.nvtx.range_push(label)
        try:
            yield
        finally:
            if self.nvtx:
                self.torch.cuda.nvtx.range_pop()
            end.record()
            with self.lock:
                self.records.append((category, start, end, metadata))

    def finish(self):
        self.torch.cuda.synchronize()
        return [{"category": category, "start_ms": self.origin.elapsed_time(start),
                 "end_ms": self.origin.elapsed_time(end), **metadata}
                for category, start, end, metadata in self.records]


def full_attention(attention_module, query, keys, values):
    """One query token; all supplied keys are at or before its absolute position."""
    from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward
    if query.shape[-2] != 1:
        raise ValueError("First repair harness supports one query token at a time")
    output, _ = eager_attention_forward(attention_module, query, keys, values, None,
                                        scaling=attention_module.scaling, dropout=0.0)
    return output


class LayeredDecoder:
    def __init__(self, model, prefix_length, prefix_store, remote_attention=None, timeline=None, compute_queue=None):
        import torch
        if model.config.model_type != "qwen2" or getattr(model.config, "use_sliding_window", False):
            raise ValueError("Only full-attention Qwen2 is supported")
        self.torch = torch
        self.model = model
        self.prefix_length = prefix_length
        self.position = prefix_length
        self.prefix_store = prefix_store
        self.remote_attention = remote_attention
        self.compute_queue = compute_queue
        self.timeline = timeline or CudaTimeline()
        self.delta = [None] * model.config.num_hidden_layers
        self.audit = []
        self.mode_counts = {"local": 0, "remote": 0}
        self.last_modes = [None] * model.config.num_hidden_layers
        self.transitions = []

    def forward(self, token):
        operation = self.compute_queue.operation("layered_token") if self.compute_queue is not None else nullcontext()
        with operation:
            return self._forward(token)

    def _forward(self, token):
        torch = self.torch
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        if tuple(token.shape) != (1, 1):
            raise ValueError("Expected exactly one token, batch size one")
        position_ids = torch.tensor([[self.position]], device=token.device, dtype=torch.long)
        hidden = self.model.model.embed_tokens(token)
        cosine, sine = self.model.model.rotary_emb(hidden, position_ids)
        for layer_index, layer in enumerate(self.model.model.layers):
            residual = hidden
            attn = layer.self_attn
            with self.timeline.measure("compute_projection", layer=layer_index, position=self.position):
                normalized = layer.input_layernorm(hidden)
                shape = (*normalized.shape[:-1], -1, attn.head_dim)
                query = attn.q_proj(normalized).view(shape).transpose(1, 2)
                key = attn.k_proj(normalized).view(shape).transpose(1, 2)
                value = attn.v_proj(normalized).view(shape).transpose(1, 2)
                query, key = apply_rotary_pos_emb(query, key, cosine, sine)
                previous = self.delta[layer_index]
                delta = ((key, value) if previous is None else
                         (torch.cat((previous[0], key), dim=-2),
                          torch.cat((previous[1], value), dim=-2)))
                self.delta[layer_index] = delta
            prefix = self.prefix_store.get(layer_index)
            mode = "local" if prefix is not None else "remote"
            if mode == "local":
                if any(t.shape[-2] != self.prefix_length for t in prefix):
                    raise RuntimeError("Published prefix includes missing or extra tokens")
                if self.last_modes[layer_index] == "remote":
                    self.transitions.append({"layer": layer_index, "position": self.position,
                                             "monotonic_s": time.monotonic()})
                with self.timeline.measure("compute_attention", layer=layer_index, position=self.position):
                    output = full_attention(attn, query, torch.cat((prefix[0], delta[0]), dim=-2),
                                             torch.cat((prefix[1], delta[1]), dim=-2))
            else:
                if self.last_modes[layer_index] == "local":
                    raise RuntimeError("A committed layer cannot revert to remote attention")
                if self.remote_attention is None:
                    raise RuntimeError("Missing local prefix and no remote attention endpoint")
                if self.compute_queue is None:
                    output = self.remote_attention(layer_index, self.position, query, key, value)
                else:
                    # Never reserve the GPU while waiting for network service.
                    output = self.compute_queue.call_unqueued(self.remote_attention,
                        layer_index, self.position, query, key, value)
            self.last_modes[layer_index] = mode
            self.mode_counts[mode] += 1
            with self.timeline.measure("compute_output_mlp", layer=layer_index, position=self.position):
                output = output.reshape(*hidden.shape[:-1], -1).contiguous()
                hidden = residual + attn.o_proj(output)
                residual = hidden
                hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
            expected_delta = self.position - self.prefix_length + 1
            if delta[0].shape[-2] != expected_delta or delta[1].shape[-2] != expected_delta:
                raise RuntimeError("Destination KV delta was appended incorrectly")
        with self.timeline.measure("compute_final", position=self.position):
            logits = self.model.lm_head(self.model.model.norm(hidden)).float()
        self.audit.append({"input_tokens": 1, "absolute_position": self.position,
                           "delta_length": self.position - self.prefix_length + 1,
                           "frontier": self.prefix_store.snapshot()[0],
                           "monotonic_s": time.monotonic()})
        self.position += 1
        return logits[:, -1, :]
