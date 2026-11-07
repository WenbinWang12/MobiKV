#!/usr/bin/env python3
"""Fit the simulator cost model from CT-C/CT-D/CT-T measurements.

Recompute of n tokens on top of a prefix s:  T(s,n) = a + b*n + c*n*(s + n/2)   (least squares,
relative error, medians of 3 reps).  Held-out check: fit on L in {1k,4k,16k} + deltas, report
the error at L in {2k,8k,32k}.  ITL table from the batched-decode runs.  Transfer: goodput
efficiency = median(measured Mb/s / cap) for capped runs; fixed cost from the small payloads.
"""
import json, sys, statistics as S, collections
import numpy as np

B = sys.argv[1]; OUT = sys.argv[2]
rows = []
for sub in ['ct_c134', 'ct_c163', 'ct_c147', 'ct_c163x']:
    try:
        rows += [dict(json.loads(l), src=sub) for l in open(f'{B}/{sub}/ct_compute.jsonl')]
    except FileNotFoundError:
        pass
groups = collections.defaultdict(list)
for r in rows:
    if r.get('ms') is None and r.get('itl_ms') is None: continue
    key = (r['src'], r['tag'])
    groups[key].append(r)


def fit(pts):
    A = np.array([[1, n, n * (s + n / 2)] for s, n, _ in pts], float)
    y = np.array([t for _, _, t in pts], float)
    w = 1.0 / y
    coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    return coef


out = {'kv_bytes_per_token': {'7B': 57344, '14B': 196608}, 'compute': {}, 'fits': {}}
report = []
tables = {}
for (src, tag), rs in sorted(groups.items()):
    model = tag.split('-')[0][1:]; share = tag.split('-g')[1]
    full = collections.defaultdict(list); delta = collections.defaultdict(list); dec = collections.defaultdict(list)
    for r in rs:
        if r['kind'] == 'full' and r['ms']: full[r['L']].append(r['ms'])
        if r['kind'] == 'delta' and r['ms']: delta[(r['P'], r['D'])].append(r['ms'])
        if r['kind'] == 'decode' and r.get('itl_ms'): dec[r['C']].append((r['b'], r['itl_ms']))
    pts = [(0, L, S.median(v)) for L, v in full.items()] + [(P, D, S.median(v)) for (P, D), v in delta.items()]
    if len(pts) < 4: continue
    coef = fit(pts)
    train = [p for p in pts if not (p[0] == 0 and p[1] in (2048, 8192, 32768))]
    test = [p for p in pts if p[0] == 0 and p[1] in (2048, 8192, 32768)]
    c2 = fit(train)
    err = [abs((c2[0] + c2[1] * n + c2[2] * n * (s + n / 2)) - t) / t for s, n, t in test]
    allerr = [abs((coef[0] + coef[1] * n + coef[2] * n * (s + n / 2)) - t) / t for s, n, t in pts]
    key = f'{model}-g{share}'
    rec = {'a_ms': float(coef[0]), 'b_ms': float(coef[1]), 'c_ms': float(coef[2]),
           'itl': {str(C): sorted(v) for C, v in dec.items()}, 'src': src,
           'points': [[s, n, t] for s, n, t in sorted(pts)],
           'heldout_max_rel_err': max(err) if err else None, 'fit_max_rel_err': max(allerr)}
    out['fits'][f'{src}:{key}'] = rec
    tables[(src, key)] = dict(full=sorted([n, S.median(v)] for n, v in full.items()),
                              delta={P: sorted([D, S.median(v)] for (P2, D), v in delta.items() if P2 == P) for P in sorted({p for p, _ in delta})},
                              itl={str(C): sorted(v) for C, v in dec.items()})
    report.append(f"{src:8s} {key:9s} a={coef[0]:7.2f} b={coef[1]*1e3:7.3f}us/tok c={coef[2]*1e6:8.4f}ns/tok2 "
                  f"heldout_err={max(err) if err else float('nan'):.3f} fit_err={max(allerr):.3f} full16k={S.median(full.get(16384,[float('nan')])):.0f}ms itl4k={dict(dec.get(4096,[]))}")
# Assemble interpolation tables: full-recompute curve and ITL from 134 (MPS emulation host); dense delta grid
# from 163 (deviation D1) when present, else the sparse 134 delta grid.
for (src, key), tb in tables.items():
    if src != 'ct_c134': continue
    rec = dict(full=tb['full'], itl=tb['itl'], delta={str(P): v for P, v in tb['delta'].items()}, delta_src='ct_c134')
    dx = tables.get(('ct_c163x', key))
    if dx and dx['delta']:
        rec['delta'] = {str(P): v for P, v in dx['delta'].items()}; rec['delta_src'] = 'ct_c163x'
        f134 = dict(map(tuple, tb['full'])); f163 = dict(map(tuple, dx['full']))
        rec['cross_host_full_ratio_163x_over_134'] = {str(n): f163[n] / f134[n] for n in f163 if n in f134}
        # cross-check of the two delta grids on the shared prefixes (134 sparse vs 163x dense)
        chk = {}
        for P, v in tb['delta'].items():
            if P in dx['delta']:
                xs = [p[0] for p in dx['delta'][P]]; ys = [p[1] for p in dx['delta'][P]]
                for D, ms in v:
                    if D in xs: chk[f'{P}:{D}'] = ys[xs.index(D)] / ms
                    else: chk[f'{P}:{D}'] = float(np.exp(np.interp(np.log(D), np.log(xs), np.log(ys)))) / ms
        rec['cross_host_delta_ratio_163x_over_134'] = chk
    out['compute'][key] = rec
# ITL for g=50 was not measured (protocol): use g=25 and g=100 interpolation in 1/share
for model in ['7B', '14B']:
    k50 = f'{model}-g50'; k25 = f'{model}-g25'; k100 = f'{model}-g100'
    if k50 in out['compute'] and k25 in out['compute'] and k100 in out['compute']:
        it = {}
        for C in out['compute'][k25]['itl']:
            a25 = dict(out['compute'][k25]['itl'][C]); a100 = dict(out['compute'][k100]['itl'].get(C, []))
            it[C] = [[b, 0.5 * (a25[b] + a100[b])] for b in sorted(a25) if b in a100]
        out['compute'][k50]['itl'] = it; out['compute'][k50]['itl_note'] = 'interpolated mean of g25 and g100 (not measured)'
# transfer
tr = []
try:
    tr = [json.loads(l) for l in open(f'{B}/ct_t/transfer.jsonl')]
except FileNotFoundError:
    pass
eff = {}; probes = []
for r in tr:
    if r['kind'] == 'transfer' and r['cap_mbps']:
        eff.setdefault(r['cap_mbps'], []).append(r['mbps'] / r['cap_mbps'])
    if r['kind'] == 'probe': probes.append(r['mbps'])
    if r['kind'] == 'transfer' and not r['cap_mbps']:
        eff.setdefault('uncapped_mbps', []).append(r['mbps'])
small = [r['ms'] - r['nbytes'] * 8 / (r['cap_mbps'] * 1e3) for r in tr if r['kind'] == 'transfer' and r['cap_mbps'] and r['ntok'] == 256]
out['transfer'] = {'eff': S.median([x for k, v in eff.items() if k != 'uncapped_mbps' for x in v]) if any(k != 'uncapped_mbps' for k in eff) else 0.95,
                   'a_ms': max(0.0, S.median(small)) if small else 5.0,
                   'eff_by_cap': {str(k): S.median(v) for k, v in eff.items()},
                   'probe_mbps': {'min': min(probes), 'median': S.median(probes), 'max': max(probes)} if probes else None,
                   'n_rows': len(tr)}
json.dump(out, open(OUT, 'w'), indent=1)
print('\n'.join(report)); print(json.dumps(out['transfer']))
