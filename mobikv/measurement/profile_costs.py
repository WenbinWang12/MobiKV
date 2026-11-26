#!/usr/bin/env python3
"""CT-C / CT-D grid for one (model, compute share) on one GPU.

Cost-profile procedure used in the final A800 campaign (full prefill, cached-prefix prefill, batched
decode; deterministic pseudo-random token ids; medians taken later by the fit script), with the dense
cached-prefix grid of run_ct_delta163.sh merged into the same pass.  Additions:
  * token ids are bounded by the model's vocabulary (Mistral has 32768 entries);
  * cells longer than --max_ctx are skipped (Gemma-2 has an 8k context);
  * the size of the returned cache is measured at every full-prefill length (all tensors in the cache
    object, split into K/V tensors and other state), instead of being derived from the config only;
  * --no_crop: models whose cache cannot be cropped (state that is not per-token) get a fresh copy of
    the prefix cache for every cached-prefix repetition, and batched decode builds its cache by a
    batched prefill.
The compute share is set from outside through CUDA_MPS_ACTIVE_THREAD_PERCENTAGE (MPS client) before
CUDA initialises; this script records it together with the MPS pipe directory in use.
Every row is appended to the JSONL file immediately.
"""
import argparse, copy, json, os, socket, time, statistics

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--full', default='1024,2048,4096,8192,16384,32768')
ap.add_argument('--prefix', default='1024,4096,8192,16384,24576')
ap.add_argument('--delta', default='32,128,256,512,1024,2048,4096,8192')
ap.add_argument('--dec_ctx', default='4096,16384')
ap.add_argument('--dec_b', default='1,2,4,8,16')
ap.add_argument('--dec_steps', type=int, default=32)
ap.add_argument('--skip_decode', action='store_true')
ap.add_argument('--no_crop', action='store_true')
ap.add_argument('--max_ctx', type=int, default=0, help='skip cells whose total length exceeds this (0 = no limit)')
a = ap.parse_args()

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


def ints(s):
    return [int(x) for x in s.split(',') if x]


share = os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', '100')
mps_pipe = os.environ.get('CUDA_MPS_PIPE_DIRECTORY', '')
host = socket.gethostname()
dev = torch.device('cuda:0')
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev)
model.eval()
load_s = time.time() - t0
cfg = model.config
head_dim = getattr(cfg, 'head_dim', None) or (cfg.hidden_size // cfg.num_attention_heads)
kv_heads = getattr(cfg, 'num_key_value_heads', None) or cfg.num_attention_heads
kv_cfg = 2 * cfg.num_hidden_layers * kv_heads * head_dim * 2  # K and V, bf16
vocab = int(cfg.vocab_size)
gen = torch.Generator().manual_seed(20261103)
MAXL = 49152
ids = torch.randint(1000, min(100000, vocab - 1), (1, MAXL), generator=gen).to(dev)
out = open(a.out, 'a')


def emit(**kw):
    kw.update(host=host, tag=a.tag, share=share, gpu=torch.cuda.get_device_name(0), ts=time.time())
    out.write(json.dumps(kw) + '\n'); out.flush()


def sync():
    torch.cuda.synchronize(dev)


def ok(n):
    return a.max_ctx <= 0 or n <= a.max_ctx


def cache_bytes(cache):
    """(bytes in tensors named keys/values, bytes in all other tensors) reachable from the cache object."""
    seen = set(); tot = {'kv': 0, 'other': 0}

    def walk(o, name, depth):
        if depth > 8 or id(o) in seen: return
        seen.add(id(o))
        if torch.is_tensor(o):
            tot['kv' if name in ('keys', 'values') else 'other'] += o.numel() * o.element_size(); return
        if isinstance(o, dict):
            for k, v in o.items(): walk(v, str(k), depth + 1)
        elif isinstance(o, (list, tuple)):
            for v in o: walk(v, name, depth + 1)
        elif hasattr(o, '__dict__'):
            for k, v in vars(o).items(): walk(v, k, depth + 1)

    walk(cache, '', 0)
    return tot['kv'], tot['other']


def standard(cache):
    ls = getattr(cache, 'layers', None)
    return bool(ls) and all(torch.is_tensor(getattr(l, 'keys', None)) and torch.is_tensor(getattr(l, 'values', None)) for l in ls)


with torch.inference_mode():
    for _ in range(3):  # warm-up
        o = model(ids[:, :1024], use_cache=True, logits_to_keep=1); sync()
    kvb, othb = cache_bytes(o.past_key_values)
    std = standard(o.past_key_values)
    emit(kind='meta', load_s=load_s, kv_bytes_per_token=kv_cfg, kv_bytes_per_token_measured=kvb / 1024,
         cache_other_bytes=othb, cache_class=type(o.past_key_values).__name__, cache_standard=std,
         layer_classes=sorted({type(l).__name__ for l in getattr(o.past_key_values, 'layers', [])}),
         layers=cfg.num_hidden_layers, kv_heads=kv_heads, head_dim=head_dim, hidden=cfg.hidden_size, vocab=vocab,
         model_type=cfg.model_type, max_pos=getattr(cfg, 'max_position_embeddings', None), mps_pipe=mps_pipe,
         model_path=a.model, attn_impl=str(getattr(cfg, '_attn_implementation', None)), torch=torch.__version__,
         no_crop=bool(a.no_crop))
    o = None; torch.cuda.empty_cache()
    use_copy = a.no_crop or not std

    # CT-C full recompute (and the size of the state it leaves)
    for L in ints(a.full):
        if not ok(L): continue
        for r in range(a.reps):
            try:
                sync(); t = time.perf_counter()
                o = model(ids[:, :L], use_cache=True, logits_to_keep=1); sync()
                ms = (time.perf_counter() - t) * 1e3
                kvb, othb = cache_bytes(o.past_key_values) if r == 0 else (None, None)
                emit(kind='full', L=L, rep=r, ms=ms, cache_kv_bytes=kvb, cache_other_bytes=othb)
            except torch.cuda.OutOfMemoryError:
                emit(kind='full', L=L, rep=r, ms=None, error='oom')
            o = None; torch.cuda.empty_cache()

    # CT-C delta recompute on top of a cached prefix
    for P in ints(a.prefix):
        if not ok(P + 1): continue
        try:
            o = model(ids[:, :P], use_cache=True, logits_to_keep=1); sync()
        except torch.cuda.OutOfMemoryError:
            emit(kind='delta', P=P, D=None, rep=0, ms=None, error='oom_prefix'); o = None; torch.cuda.empty_cache(); continue
        cache = o.past_key_values; o = None
        for D in ints(a.delta):
            if not ok(P + D): continue
            for r in range(a.reps):
                try:
                    c2 = copy.deepcopy(cache) if use_copy else cache
                    sync(); t = time.perf_counter()
                    model(ids[:, P:P + D], past_key_values=c2, use_cache=True, logits_to_keep=1); sync()
                    ms = (time.perf_counter() - t) * 1e3
                    emit(kind='delta', P=P, D=D, rep=r, ms=ms)
                except torch.cuda.OutOfMemoryError:
                    emit(kind='delta', P=P, D=D, rep=r, ms=None, error='oom')
                if use_copy: c2 = None
                else: cache.crop(P)
                torch.cuda.empty_cache()
        cache = None; torch.cuda.empty_cache()

    # CT-D batched decode (continuous-batching proxy): b sequences decode in lockstep
    if not a.skip_decode:
        for C in ints(a.dec_ctx):
            if not ok(C + a.dec_steps + 2): continue
            o = model(ids[:, :C], use_cache=True, logits_to_keep=1); sync()
            base = o.past_key_values; o = None
            kb, ob = cache_bytes(base); state_bytes = kb + ob
            for b in ints(a.dec_b):
                need = b * state_bytes
                free = torch.cuda.mem_get_info(dev)[0]
                if need * 1.3 > free:
                    emit(kind='decode', C=C, b=b, itl_ms=None, error='skip_mem', need_gb=need / 2**30, free_gb=free / 2**30)
                    continue
                try:
                    if use_copy:
                        o = model(ids[:, :C].repeat(b, 1), use_cache=True, logits_to_keep=1); sync()
                        cache = o.past_key_values; o = None
                    else:
                        cache = DynamicCache()
                        for li, layer in enumerate(base.layers):
                            cache.update(layer.keys.repeat(b, 1, 1, 1), layer.values.repeat(b, 1, 1, 1), li)
                    tok = ids[:, C:C + 1].repeat(b, 1)
                    steps = []
                    for s in range(a.dec_steps + 2):
                        sync(); t = time.perf_counter()
                        lo = model(tok, past_key_values=cache, use_cache=True).logits
                        tok = lo[:, -1:].argmax(-1); sync()
                        steps.append((time.perf_counter() - t) * 1e3)
                    st = steps[2:]
                    emit(kind='decode', C=C, b=b, itl_ms=statistics.median(st), itl_p90=sorted(st)[int(0.9 * (len(st) - 1))], steps=len(st),
                         cache_build='batched_prefill' if use_copy else 'repeat')
                except torch.cuda.OutOfMemoryError:
                    emit(kind='decode', C=C, b=b, itl_ms=None, error='oom')
                except RuntimeError as ex:  # e.g. the conv1d 32-bit index limit of a recurrent layer at a large batch
                    emit(kind='decode', C=C, b=b, itl_ms=None, error='runtime', msg=str(ex)[:160])
                cache = None; tok = None; torch.cuda.empty_cache()
            base = None; torch.cuda.empty_cache()

emit(kind='done', wall_s=time.time() - t0)
print('done', a.tag, share, time.time() - t0)
