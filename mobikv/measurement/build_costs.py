#!/usr/bin/env python3
"""Build a simulator cost file from the measured grids of profile_costs.py (no proxy scaling).

For every tag "<model>-g<share>" that finished (has a 'done' row):
  full   : median over repetitions of the full-prefill time at each length
  delta  : median cached-prefix prefill time at each (prefix, new tokens)
  itl    : batched-decode inter-token latency at each (context, batch)
  state  : bytes of the cache returned by a full prefill at each length (K/V tensors + other state)
The g50 ITL table, which is not measured, is the mean of the g25 and g100 tables.
Transfer parameters and any model listed in --keep are copied from the base cost file.

usage: python -m mobikv.measurement.build_costs --base configs/costs_a800.json --out costs.json ct_compute.jsonl [...]
"""
import argparse, collections, json, statistics as S
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('files', nargs='+'); ap.add_argument('--base', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--keep', default='', help='Comma-separated base models to retain; default: none'); ap.add_argument('--ctx_cap', type=int, default=32768)
a = ap.parse_args()

rows = collections.defaultdict(list)
for f in a.files:
    for line in open(f):
        r = json.loads(line); r['_file'] = f
        rows[(f, r['tag'])].append(r)
done = {k: v for k, v in rows.items() if any(r['kind'] == 'done' for r in v)}
# a tag measured in several files: the last file given wins
by_tag = {}
for (f, tag), v in done.items():
    by_tag[tag] = v

base = json.load(open(a.base))
out = {'model_contexts': {}, 'kv_bytes_per_token': {}, 'compute': {}, 'state_bytes': {}, 'meta': {}, 'transfer': base['transfer']}
for m in [x for x in a.keep.split(',') if x]:
    out['model_contexts'][m] = base['model_contexts'][m]; out['kv_bytes_per_token'][m] = base['kv_bytes_per_token'][m]
    for k, v in base['compute'].items():
        if k.rsplit('-g', 1)[0] == m: out['compute'][k] = v

report = []
for tag, rs in sorted(by_tag.items()):
    model, share = tag.rsplit('-g', 1)
    meta = next(r for r in rs if r['kind'] == 'meta')
    full = collections.defaultdict(list); delta = collections.defaultdict(list); dec = collections.defaultdict(list); state = {}
    for r in rs:
        if r['kind'] == 'full' and r.get('ms'):
            full[r['L']].append(r['ms'])
            if r.get('cache_kv_bytes') is not None: state[r['L']] = [r['cache_kv_bytes'], r['cache_other_bytes']]
        if r['kind'] == 'delta' and r.get('ms'): delta[(r['P'], r['D'])].append(r['ms'])
        if r['kind'] == 'decode' and r.get('itl_ms'): dec[r['C']].append([r['b'], r['itl_ms']])
    if str(meta['share']) != share:
        raise SystemExit(f'{tag}: recorded share {meta["share"]} differs from the tag')
    if not meta.get('mps_pipe'):
        raise SystemExit(f'{tag}: no MPS pipe directory recorded; the share label cannot be trusted')
    rec = {'full': sorted([L, S.median(v)] for L, v in full.items()),
           'delta': {str(P): sorted([D, S.median(v)] for (P2, D), v in delta.items() if P2 == P) for P in sorted({p for p, _ in delta})},
           'itl': {str(C): sorted(v) for C, v in sorted(dec.items())},
           'src': f'profile_costs {meta["gpu"]} share {share} ({Path(meta["_file"]).name})',
           'spread_full_max_rel': max((max(v) - min(v)) / S.median(v) for v in full.values()),
           'spread_delta_max_rel': max(((max(v) - min(v)) / S.median(v) for v in delta.values()), default=0.0)}
    out['compute'][tag] = rec
    out['state_bytes'].setdefault(model, {})[share] = sorted([L, kv, ot] for L, (kv, ot) in state.items())
    out['meta'][tag] = {k: meta.get(k) for k in ('gpu', 'host', 'layers', 'kv_heads', 'head_dim', 'hidden', 'vocab', 'model_type', 'max_pos',
                                                 'cache_class', 'cache_standard', 'layer_classes', 'kv_bytes_per_token',
                                                 'kv_bytes_per_token_measured', 'attn_impl', 'torch', 'model_path', 'no_crop', 'mps_pipe')}
    st = out['state_bytes'][model][share]
    Lmax = st[-1][0]; per_tok = (st[-1][1] + st[-1][2]) / Lmax
    # state model from the measured cache sizes: bytes(n) = kappa*n + kappa_win*min(n, win) + const
    L1, kv1, ot1 = st[0]; Ln, kvn, otn = st[-1]; k_tot = kv1 / L1; sm = None
    if otn > 2**20:                                   # fixed-size state of recurrent (linear-attention) layers
        sm = {'const': float(otn), 'kappa_win': 0.0, 'win': 0}; kappa = kvn / Ln
    elif abs(kvn / Ln - k_tot) > 1 and len(st) >= 2:   # sliding-window layers keep K/V for the last `win` tokens only
        La, kva, _ = st[-2]
        k_g = (kvn - kva) / (Ln - La); k_w = k_tot - k_g
        sm = {'const': 0.0, 'kappa_win': k_w, 'win': int(round((kvn - k_g * Ln) / k_w))}; kappa = k_g
    else:
        kappa = meta['kv_bytes_per_token'] if abs(k_tot - meta['kv_bytes_per_token']) <= 1 else k_tot
    out['kv_bytes_per_token'][model] = kappa
    if sm: out.setdefault('state_model', {})[model] = sm
    out['model_contexts'][model] = min(a.ctx_cap, int(meta.get('max_pos') or a.ctx_cap))
    f8 = dict(rec['full']).get(8192)
    report.append(f"{tag:22s} full8k={f8:8.1f}ms  full_pts={len(rec['full'])} delta_prefixes={list(rec['delta'])} itl_ctx={list(rec['itl'])} "
                  f"state/token@{Lmax}={per_tok:9.0f}B (cfg {meta['kv_bytes_per_token']}) spread full {rec['spread_full_max_rel']:.3f} delta {rec['spread_delta_max_rel']:.3f}")

# g50 ITL = mean of g25 and g100 where it was not measured
for model in sorted({k.rsplit('-g', 1)[0] for k in out['compute']}):
    k25, k50, k100 = (f'{model}-g{g}' for g in (25, 50, 100))
    if k50 in out['compute'] and not out['compute'][k50].get('itl') and k25 in out['compute'] and k100 in out['compute']:
        it = {}
        for C, tab in out['compute'][k25]['itl'].items():
            a25 = dict(map(tuple, tab)); a100 = dict(map(tuple, out['compute'][k100]['itl'].get(C, [])))
            it[C] = [[b, 0.5 * (a25[b] + a100[b])] for b in sorted(a25) if b in a100]
        out['compute'][k50]['itl'] = it; out['compute'][k50]['itl_note'] = 'interpolated mean of g25 and g100 (not measured)'

json.dump(out, open(a.out, 'w'), indent=1)
print('\n'.join(report))
eff = out['transfer']['eff']
print('\nrho_8k at 930 Mb/s and at 10 Gb/s (state bytes per token x 8192 / link rate / full-prefill time):')
for tag in sorted(out['compute']):
    model, share = tag.rsplit('-g', 1)
    C = dict(map(tuple, out['compute'][tag]['full'])).get(8192)
    if C is None: continue
    k = out['kv_bytes_per_token'][model]; sm = out.get('state_model', {}).get(model, {})
    b8 = 8192 * k + sm.get('const', 0.0) + sm.get('kappa_win', 0.0) * min(8192, sm.get('win', 0))
    r = lambda cap: b8 * 8 / (cap * 1e6 * eff) / (C / 1e3)
    print(f'  {tag:22s} C8k={C / 1e3:6.3f}s state@8k={b8 / 2**20:7.1f}MiB ({b8 / 8192 / 1024:6.1f}KiB/token; {json.dumps(sm) if sm else "per-token only"})  rho@0.93G={r(930):6.2f}  rho@10G={r(10000):5.2f}')
