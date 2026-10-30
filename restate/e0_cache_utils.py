"""Lossless E0 cache snapshots for decoder-only, full-attention DynamicCache.

Pairs use layer order and tensor layout [batch, KV heads, sequence, head dim].
This module does not serialize network messages or use pickle. Heavy libraries
are imported only when a helper is called. Sliding/static/quantized caches are
deliberately unsupported because K/V tensors alone do not capture their state.
"""

import hashlib
import json
import struct


def _validate_pairs(pairs, require_cpu=False):
    import torch

    if not isinstance(pairs, (list, tuple)) or not pairs:
        raise ValueError("Expected a nonempty ordered list/tuple of (K, V) pairs")
    reference_shape = reference_dtype = None
    for index, pair in enumerate(pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Layer {index}: expected exactly (K, V)")
        for label, tensor in zip(("key", "value"), pair):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Layer {index} {label}: expected torch.Tensor")
            if tensor.layout != torch.strided or tensor.device.type == "meta":
                raise ValueError(f"Layer {index} {label}: unsupported tensor layout/device")
            if tensor.ndim != 4 or any(size <= 0 for size in tensor.shape):
                raise ValueError(f"Layer {index} {label}: expected nonempty [B,H,S,D]")
            if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                raise ValueError(f"Layer {index} {label}: unsupported dtype {tensor.dtype}")
            if require_cpu and tensor.device.type != "cpu":
                raise ValueError(f"Layer {index} {label}: expected a CPU snapshot")
            if reference_shape is None:
                reference_shape, reference_dtype = tuple(tensor.shape), tensor.dtype
            if tuple(tensor.shape) != reference_shape or tensor.dtype != reference_dtype:
                raise ValueError("All full-attention layer K/V shapes and dtypes must match")
    return reference_shape


def cache_to_cpu_pairs(cache):
    """Return independent contiguous CPU snapshots in ascending layer order.

    Accepts an ordinary full-attention DynamicCache or legacy tuple/list pairs.
    Call only while the source cache is quiescent; this is not an atomic snapshot
    of a concurrently updated cache.
    """
    if isinstance(cache, (tuple, list)):
        pairs = cache
    else:
        from transformers.cache_utils import DynamicCache, DynamicLayer

        if type(cache) is not DynamicCache or getattr(cache, "offloading", False):
            raise TypeError("Only non-offloaded DynamicCache or legacy K/V pairs are supported")
        pairs = []
        for index, layer in enumerate(cache.layers):
            if type(layer) is not DynamicLayer or not layer.is_initialized:
                raise TypeError(f"Layer {index}: requires an initialized full-attention DynamicLayer")
            pairs.append((layer.keys, layer.values))
    _validate_pairs(pairs)
    return [
        tuple(tensor.detach().to(device="cpu", copy=True).contiguous() for tensor in pair)
        for pair in pairs
    ]


def pairs_to_cache(pairs, model_config, device):
    """Restore a fresh DynamicCache; update each layer once with the full prefix.

    CPU pairs remain independent of the restored cache, including for a CPU
    destination. The caller supplies the next token's position_ids to the model.
    """
    import torch
    from transformers.cache_utils import DynamicCache, DynamicLayer

    shape = _validate_pairs(pairs, require_cpu=True)
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("E0 cache restoration supports CPU or CUDA only")
    config = model_config.get_text_config(decoder=True) if hasattr(model_config, "get_text_config") else model_config
    if getattr(model_config, "is_encoder_decoder", False):
        raise ValueError("Encoder-decoder caches are not supported")
    if len(pairs) != config.num_hidden_layers:
        raise ValueError("Cache layer count does not match the model config")
    expected_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    expected_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    if shape[1] != expected_heads or shape[3] != expected_dim:
        raise ValueError("Cache KV-head count/head dimension does not match model config")
    cache = DynamicCache(config=model_config)
    if len(cache.layers) != len(pairs) or any(type(layer) is not DynamicLayer for layer in cache.layers):
        raise TypeError("Model config must construct only full-attention DynamicLayers")
    for index, (key, value) in enumerate(pairs):
        cache.update(
            key.detach().to(device=target, copy=True).contiguous(),
            value.detach().to(device=target, copy=True).contiguous(),
            index,
        )
        if cache.get_seq_length(index) != shape[2]:
            raise RuntimeError(f"Layer {index}: restored sequence length mismatch")
    return cache


def cache_bytes(pairs):
    """Total tensor payload bytes, excluding protocol/metadata overhead."""
    _validate_pairs(pairs)
    return sum(tensor.numel() * tensor.element_size() for pair in pairs for tensor in pair)


def cache_metadata(pairs):
    """JSON-serializable, device-independent description of an ordered snapshot."""
    shape = _validate_pairs(pairs)
    layers = []
    for index, pair in enumerate(pairs):
        entry = {"layer_index": index}
        for label, tensor in zip(("key", "value"), pair):
            entry[label] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "nbytes": tensor.numel() * tensor.element_size(),
            }
        layers.append(entry)
    return {
        "format": "restate-e0-kv-v1",
        "layout": "batch,kv_heads,sequence,head_dim",
        "num_layers": len(pairs),
        "sequence_length": shape[2],
        "total_bytes": sum(t.numel() * t.element_size() for pair in pairs for t in pair),
        "layers": layers,
    }


def cache_sha256(pairs):
    """Hash canonical metadata followed by exact K/V bytes in layer order.

    The uint8 view supports bf16 without numerical conversion or a bf16 NumPy
    dtype. Snapshots must be CPU tensors and must not be mutated during hashing.
    """
    import torch

    _validate_pairs(pairs, require_cpu=True)
    metadata = json.dumps(cache_metadata(pairs), sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"restate-e0-kv-v1\x00")
    digest.update(struct.pack(">Q", len(metadata)))
    digest.update(metadata)
    for pair in pairs:
        for tensor in pair:
            raw = tensor.detach().contiguous().view(torch.uint8).numpy()
            digest.update(memoryview(raw).cast("B"))
    return digest.hexdigest()
