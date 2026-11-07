#!/usr/bin/env python3
"""A7 analysis (E3-E6): per scenario, pooled metrics of the 8 policies and rpp's ratio to each of them (paired session
bootstrap, 300 resamples, seeds pooled); E3 decision times."""
import json, glob, os, sys
import numpy as np
B = sys.argv[1]; OUT = sys.argv[2]; NB = 300
rng = np.random.default_rng(20261103)
POLS = ['rpp', 'mobikv', 'follow_me_cloud', 'wang_ton19_mdp', 'ouyang_jsac18', 'preble_iclr25', 'mooncake_fast25', 'loongserve_sosp24']
PUB = POLS[2:]

def load(f):
    r = json.load(open(f)); pr = [x for x in r['per_request'] if x[3] == 1]
    a = np.array([[x[1], x[2], x[4], x[10] if x[10] is not None else np.nan, x[11] if x[11] is not None else np.nan, x[12]] for x in pr], float)
    a = a[np.argsort(a[:, 0], kind='stable')]; st = r['stats']
    return dict(sid=a[:, 0].astype(int), turn=a[:, 1], ttft=a[:, 2], tpot_b=a[:, 3], tpot_s=a[:, 4], out=a[:, 5],
                cost=dict(pp_TB=st.get('pp_bytes', 0) / 1e12, splits=st.get('pp_split', 0), eligible=st.get('pp_eligible', 0),
                          bg_gpu_s=r['gpu_bg_s'], bg_TB=st.get('bg_bytes', 0) / 1e12, util=r['gpu_util_fg']['mean']),
                dec=r.get('dec_time'), n_cont=r['cont']['n'])
def runs(pattern):
    d = {}
    for f in glob.glob(pattern):
        s = int(os.path.basename(f)[:-5].rsplit('_s', 1)[1]); d[s] = load(f)
    return d
SEL = {'cont': lambda d: d['turn'] >= 2, 'first': lambda d: d['turn'] == 1, 'dec': lambda d: d['out'] >= 2}
MET = {'cont_p95': ('cont', 'ttft', 'p95'), 'cont_mean': ('cont', 'ttft', 'mean'), 'first_p95': ('first', 'ttft', 'p95'),
       'tpot_s_p95': ('dec', 'tpot_s', 'p95'), 'tpot_b_p95': ('dec', 'tpot_b', 'p95')}
def stat(x, s):
    x = x[~np.isnan(x)]; return float(np.quantile(x, .95)) if s == 'p95' else float(x.mean())
def pooled(dd):
    r = {m: stat(np.concatenate([d[f][SEL[sel](d)] for d in dd.values()]), s) for m, (sel, f, s) in MET.items()}
    for k in next(iter(dd.values()))['cost']: r[k] = float(np.mean([d['cost'][k] for d in dd.values()]))
    return r
def prep(d, sel, f):
    m = SEL[sel](d); sid = d['sid'][m]; v = d[f][m]; u, st, c = np.unique(sid, return_index=True, return_counts=True); return u, st, c, v
def gather(st, c, v, pick):
    s = st[pick]; cc = c[pick]; off = np.repeat(np.cumsum(cc) - cc, cc); return v[np.repeat(s, cc) + np.arange(cc.sum()) - off]
def boot(dA, dB, metric):
    sel, f, s = MET[metric]; P = []
    for sd in sorted(set(dA) & set(dB)):
        uA, sA, cA, vA = prep(dA[sd], sel, f); uB, sB, cB, vB = prep(dB[sd], sel, f); U = np.union1d(uA, uB)
        def al(u, st, c):
            i = np.minimum(np.searchsorted(u, U), len(u) - 1); ok = u[i] == U; return np.where(ok, st[i], 0), np.where(ok, c[i], 0)
        sa, ca = al(uA, sA, cA); sb, cb = al(uB, sB, cB); P.append((len(U), sa, ca, vA, sb, cb, vB))
    pt = stat(np.concatenate([p[3] for p in P]), s) / stat(np.concatenate([p[6] for p in P]), s); rs = []
    for _ in range(NB):
        A = []; Bv = []
        for n, sa, ca, vA, sb, cb, vB in P:
            pick = rng.integers(0, n, n); A.append(gather(sa, ca, vA, pick)); Bv.append(gather(sb, cb, vB, pick))
        rs.append(stat(np.concatenate(A), s) / stat(np.concatenate(Bv), s))
    lo, hi = np.quantile(rs, [.025, .975]); return dict(point=pt, lo=float(lo), hi=float(hi))

res = {'scen': {}, 'dec': {}}
for sc in sorted(os.listdir(f'{B}/a7_gate')):
    p = f'{B}/a7_gate/{sc}'
    if not os.path.isdir(p): continue
    pols = {q: runs(f'{p}/{q}_s*.json') for q in POLS}; pols = {q: d for q, d in pols.items() if d}
    if sc.startswith('e3_'):
        res['dec'][sc] = {q: dict(dec=[d['dec'] for d in dd.values()], cont_p95=pooled(dd)['cont_p95']) for q, dd in pols.items()}
        if sc != 'e3_default': continue
    if 'rpp' not in pols: continue
    res['scen'][sc] = dict(table={q: pooled(d) for q, d in pols.items()},
                           cmp={q: {m: boot(pols['rpp'], pols[q], m) for m in ('cont_p95', 'first_p95', 'tpot_s_p95', 'tpot_b_p95')}
                                for q in pols if q != 'rpp'})
    print('done', sc, flush=True)
json.dump(res, open(OUT + '.json', 'w'), indent=1)
ci = lambda x: f"{x['point']:.2f} [{x['lo']:.2f},{x['hi']:.2f}]"
L = ['# A7 simulation results (E3-E6)\n', '## Continuation p95 TTFT (s) and rpp ratios (95% CI)\n']
for sc, v in res['scen'].items():
    t = v['table']; c = v['cmp']
    L.append(f'### {sc}\n')
    L.append('| policy | cont p95 | first p95 | TPOT_s p95 ms | TPOT_b p95 ms | util | RPP splits/eligible | rpp/x cont p95 | rpp/x first p95 | rpp/x TPOT_s | rpp/x TPOT_b |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for q in POLS:
        if q not in t: continue
        r = t[q]; cc = c.get(q)
        L.append(f"| {q} | {r['cont_p95']:.2f} | {r['first_p95']:.2f} | {r['tpot_s_p95']*1e3:.0f} | {r['tpot_b_p95']*1e3:.0f} | {r['util']:.2f} | {r['splits']:.0f}/{r['eligible']:.0f} | "
                 + (f"{ci(cc['cont_p95'])} | {ci(cc['first_p95'])} | {ci(cc['tpot_s_p95'])} | {ci(cc['tpot_b_p95'])} |" if cc else '– | – | – | – |'))
    sig = [q for q in c if c[q]['cont_p95']['hi'] < 1]; L.append(f"\nrpp significantly lower cont p95 than: {', '.join(sig)} ({len(sig)}/{len(c)})\n")
L.append('## E3 decision time per request (ms, Python)\n')
for sc, v in res['dec'].items():
    for q, w in v.items():
        ds = [x for x in w['dec'] if x]
        if ds: L.append(f"- {sc} {q}: mean {np.mean([x['mean_ms'] for x in ds]):.3f}, p99 {np.mean([x['p99_ms'] for x in ds]):.3f}, max {max(x['max_ms'] for x in ds):.1f}, "
                        f"pp search {np.mean([x['pp_search_mean_ms'] for x in ds]):.3f}, edges {ds[0]['edges']}, requests {ds[0]['requests']}")
open(OUT + '.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
