#!/usr/bin/env python3
"""A7 / E1 (SV-3) analysis.  Requests executed at the three replayed edges in [2400, 3300) s, seeds 21 + 22 pooled.
Measured vs simulated TTFT (continuation = turn >= 2) and TPOT; ratios with a paired session bootstrap (1000 resamples,
stratified by seed).  Claims E1-1 (rpp/mobikv measured continuation p95, CI upper < 1); E1-2..E1-4 reported."""
import glob, json, os, sys
import numpy as np
O = sys.argv[1]; OUT = sys.argv[2]; NB = 1000; rng = np.random.default_rng(20261103)
POLS = ['rpp', 'mobikv', 'ouyang_jsac18']
def load(p, s):
    fs = sorted(glob.glob(f'{O}/replay_{p}_s{s}_n*.json'))
    if len(fs) < 3: return None
    sim = json.load(open(f'{O}/sim_{p}_s{s}.json')); tp = {x[0]: (x[10], x[11]) for x in sim['per_request']}
    rows = []; info = []
    for f in fs:
        r = json.load(open(f)); info.append(dict(edge=r['edge'], gpu=r['gpu'], jobs=r['n_done'], fg=r['n_fg'], dec=r['dec'], offset=r['offset_s']))
        for q in r['reqs']:
            b, s_ = tp.get(q['rid'], (None, None))
            rows.append([q['sess'], q['turn'], q['ttft_meas'], q['ttft_sim'], q['tpot_meas'] if q['tpot_meas'] is not None else np.nan,
                         b if b is not None else np.nan, s_ if s_ is not None else np.nan, q['rpp'], q['boundary'], q['ntok'], q['out']])
    a = np.array(rows, float); a = a[np.argsort(a[:, 0], kind='stable')]
    return dict(a=a, info=info)
D = {p: {s: load(p, s) for s in (21, 22)} for p in POLS}
D = {p: {s: v for s, v in d.items() if v is not None} for p, d in D.items()}
COL = dict(sess=0, turn=1, meas=2, sim=3, tpot=4, tpot_b=5, tpot_s=6, rpp=7, boundary=8, ntok=9, out=10)
def sel(a, kind):
    if kind == 'cont': return a[a[:, 1] >= 2]
    if kind == 'dec': return a[(a[:, 9] >= 10)]
    return a
def p95(x): x = x[~np.isnan(x)]; return float(np.quantile(x, .95)) if len(x) else float('nan')
def boot(pA, pB, kind, ca, cb=None):
    cb = ca if cb is None else cb; P = []
    for s in sorted(set(D[pA]) & set(D[pB])):
        A = sel(D[pA][s]['a'], kind); B = sel(D[pB][s]['a'], kind); U = np.union1d(A[:, 0], B[:, 0])
        P.append((U, A, B))
    pt = p95(np.concatenate([x[1][:, ca] for x in P])) / p95(np.concatenate([x[2][:, cb] for x in P])); rs = []
    for _ in range(NB):
        va = []; vb = []
        for U, A, B in P:
            pick = rng.choice(U, len(U)); cnt = dict(zip(*np.unique(pick, return_counts=True)))
            wa = np.array([cnt.get(v, 0) for v in A[:, 0]]); wb = np.array([cnt.get(v, 0) for v in B[:, 0]])
            va.append(np.repeat(A[:, ca], wa)); vb.append(np.repeat(B[:, cb], wb))
        rs.append(p95(np.concatenate(va)) / p95(np.concatenate(vb)))
    lo, hi = np.nanquantile(rs, [.025, .975]); return dict(point=pt, lo=float(lo), hi=float(hi))
res = dict(per_policy={}, cmp={})
for p in POLS:
    if not D[p]: continue
    a = np.concatenate([v['a'] for v in D[p].values()]); c = sel(a, 'cont'); dd = sel(a, 'dec')
    err = np.abs(a[:, 2] - a[:, 3]) / np.maximum(a[:, 3], 1e-3)
    rp = a[a[:, 7] == 1]
    res['per_policy'][p] = dict(seeds=sorted(D[p]), n=len(a), n_cont=len(c), n_rpp=int(len(rp)), n_boundary=int(a[:, 8].sum()),
        cont_p95_meas=p95(c[:, 2]), cont_p95_sim=p95(c[:, 3]), cont_mean_meas=float(c[:, 2].mean()), cont_mean_sim=float(c[:, 3].mean()),
        all_p95_meas=p95(a[:, 2]), all_p95_sim=p95(a[:, 3]), med_abs_rel_err=float(np.median(err)),
        rpp_ttft_meas_med=float(np.median(rp[:, 2])) if len(rp) else None, rpp_ttft_sim_med=float(np.median(rp[:, 3])) if len(rp) else None,
        tpot_p95_meas=p95(dd[:, 4]), tpot_p50_meas=float(np.nanmedian(dd[:, 4])), tpot_b_p95_sim=p95(dd[:, 5]), tpot_s_p95_sim=p95(dd[:, 6]),
        nodes={s: v['info'] for s, v in D[p].items()})
for q in ('mobikv', 'ouyang_jsac18'):
    if D['rpp'] and D[q]:
        res['cmp'][f'rpp/{q}'] = dict(cont_p95_meas=boot('rpp', q, 'cont', 2), cont_p95_sim=boot('rpp', q, 'cont', 3),
                                      tpot_p95_meas=boot('rpp', q, 'dec', 4))
# ---- A7.1 paired ablation: the same rpp job streams with RPP switched off (requests matched by id)
def reqs_of(tag, s):
    fs = sorted(glob.glob(f'{O}/replay_{tag}_s{s}_n*.json'))
    if len(fs) < 3: return None
    out = {}
    for f in fs:
        for q in json.load(open(f))['reqs']: out[q['rid']] = q
    return out
def pboot(X, ia, ib):
    us = np.unique(X[:, 0]); idx = {u: np.where(X[:, 0] == u)[0] for u in us}; rs = []
    pt = p95(X[:, ia]) / p95(X[:, ib])
    for _ in range(NB):
        pick = np.concatenate([idx[u] for u in rng.choice(us, len(us))]); rs.append(p95(X[pick, ia]) / p95(X[pick, ib]))
    lo, hi = np.nanquantile(rs, [.025, .975]); return dict(point=pt, lo=float(lo), hi=float(hi))
def paired(tag_on, tag_off, seeds):
    """Requests matched by id between two replays of the same job streams (A = tag_on, B = tag_off)."""
    pairs = []
    for s in seeds:
        on = reqs_of(tag_on.format(s=s), s) if '{s}' in tag_on else reqs_of(tag_on, s)
        off = reqs_of(tag_off, s)
        if not on or not off: continue
        for rid in sorted(set(on) & set(off)):
            a_, b_ = on[rid], off[rid]
            pairs.append([s * 100000 + a_['sess'], a_['turn'], a_['ttft_meas'], b_['ttft_meas'], a_['tpot_meas'] or np.nan, b_['tpot_meas'] or np.nan, a_['rpp'], a_['ntok']])
    if not pairs: return None
    P = np.array(pairs, float); sp = P[P[:, 6] == 1]; cont = P[P[:, 1] >= 2]; dec = P[P[:, 7] >= 10]
    return dict(n=len(P), n_cont=int(len(cont)), n_split=int(len(sp)), split_faster_frac=float((sp[:, 2] < sp[:, 3]).mean()) if len(sp) else None,
                split_ratio_median=float(np.median(sp[:, 2] / sp[:, 3])) if len(sp) else None,
                cont_p95_on=p95(cont[:, 2]), cont_p95_off=p95(cont[:, 3]), cont_p95_ratio=pboot(cont, 2, 3),
                tpot_p95_on=p95(dec[:, 4]), tpot_p95_off=p95(dec[:, 5]), tpot_p95_ratio=pboot(dec, 4, 5),
                seeds=sorted({int(x // 100000) for x in P[:, 0]}))
res['paired_sets'] = {}
for name, a_, b_, seeds in [('A7.1 RPP on/off (seeds 21-22)', 'rpp', 'rppoff', (21, 22)), ('A7.2 RPP on/off (seeds 21-24)', 'rpp', 'rppoff', (21, 22, 23, 24)),
                            ('A8 R2 decode-first / original scheduler (rpp streams)', 'rppdfbsched', 'rpp', (21, 22)),
                            ('A8 R1 RPP on/off under decode-first (rpp+dfb streams)', 'rppdfb', 'rppdfboff', (21, 22)),
                            ('A8.1 RPP on/off under decode-first (seeds 21-24)', 'rppdfb', 'rppdfboff', (21, 22, 23, 24)),
                            ('A8.1 RPP on/off under decode-first (seeds 23-24 only)', 'rppdfb', 'rppdfboff', (23, 24))]:
    v = paired(a_, b_, seeds)
    if v: res['paired_sets'][name] = v
if 'A7.1 RPP on/off (seeds 21-22)' in res['paired_sets']: res['paired'] = res['paired_sets']['A7.1 RPP on/off (seeds 21-22)']
# A8: measured TPOT/TTFT of the decode-first system against the independent Lyapunov replays
def pooled_reqs(tag, seeds):
    rows = []
    for s in seeds:
        q = reqs_of(tag, s)
        if q: rows += [[s * 100000 + x['sess'], x['turn'], x['ttft_meas'], x['tpot_meas'] or np.nan, x['ntok']] for x in q.values()]
    return np.array(rows, float) if rows else None
X = pooled_reqs('rppdfb', (21, 22)); Y = pooled_reqs('ouyang_jsac18', (21, 22))
if X is not None and Y is not None:
    res['a8_vs_lyap'] = dict(cont_p95_dfb=p95(X[X[:, 1] >= 2][:, 2]), cont_p95_lyap=p95(Y[Y[:, 1] >= 2][:, 2]),
                             tpot_p95_dfb=p95(X[X[:, 4] >= 10][:, 3]), tpot_p95_lyap=p95(Y[Y[:, 4] >= 10][:, 3]),
                             tpot_p50_dfb=float(np.nanmedian(X[X[:, 4] >= 10][:, 3])), tpot_p50_lyap=float(np.nanmedian(Y[Y[:, 4] >= 10][:, 3])))
json.dump(res, open(OUT + '.json', 'w'), indent=1)
ci = lambda x: f"{x['point']:.2f} [{x['lo']:.2f}, {x['hi']:.2f}]"
L = ['# A7 / E1 (SV-3) results\n', '| policy | n (cont) | RPP / boundary | cont p95 meas | cont p95 sim | median abs rel err | TPOT p50 / p95 meas ms | TPOT_b / TPOT_s p95 sim ms |', '|---|---|---|---|---|---|---|---|']
for p, v in res['per_policy'].items():
    L.append(f"| {p} | {v['n']} ({v['n_cont']}) | {v['n_rpp']} / {v['n_boundary']} | {v['cont_p95_meas']:.2f} | {v['cont_p95_sim']:.2f} | {v['med_abs_rel_err']:.3f} | "
             f"{v['tpot_p50_meas']*1e3:.0f} / {v['tpot_p95_meas']*1e3:.0f} | {v['tpot_b_p95_sim']*1e3:.0f} / {v['tpot_s_p95_sim']*1e3:.0f} |")
L.append('')
for k, v in res['cmp'].items():
    L.append(f"- {k}: cont p95 measured {ci(v['cont_p95_meas'])}, simulated {ci(v['cont_p95_sim'])}; TPOT p95 measured {ci(v['tpot_p95_meas'])}")
for name, v in res.get('paired_sets', {}).items():
    L.append(f"\n## {name} (seeds {v['seeds']})\n")
    if v['n_split']:
        L.append(f"- requests paired: {v['n']} ({v['n_cont']} continuations); split by RPP: {v['n_split']}; faster in A: {v['split_faster_frac']:.2f}; median TTFT ratio of split requests: {v['split_ratio_median']:.2f}")
    L.append(f"- continuation p95: A {v['cont_p95_on']:.2f} s, B {v['cont_p95_off']:.2f} s, ratio {ci(v['cont_p95_ratio'])}")
    L.append(f"- TPOT p95: A {v['tpot_p95_on']*1e3:.0f} ms, B {v['tpot_p95_off']*1e3:.0f} ms, ratio {ci(v['tpot_p95_ratio'])}")
if 'a8_vs_lyap' in res:
    v = res['a8_vs_lyap']
    L.append(f"\n## A8: decode-first system vs Lyapunov (independent replays, seeds 21-22)\n- continuation p95: {v['cont_p95_dfb']:.2f} vs {v['cont_p95_lyap']:.2f} s; TPOT p50/p95: {v['tpot_p50_dfb']*1e3:.0f}/{v['tpot_p95_dfb']*1e3:.0f} vs {v['tpot_p50_lyap']*1e3:.0f}/{v['tpot_p95_lyap']*1e3:.0f} ms")
open(OUT + '.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
