#!/usr/bin/env python3
"""A9.1 / A9.2 analysis (AMENDMENT A9).  Uses the metrics and the paired session bootstrap of analyze_a7.py (300 resamples,
seeds pooled; same sessions across policies within a seed).
  A9.1  per scenario: rppdfb (MobiKV-DF with RPP) against every policy run before in that scenario (a3/a5/a6/a7 gates) and
        against mkdfb (MobiKV-DF without RPP); rpp's own ratios recomputed with the same code define the claim sets.
  A9.2  fresh seeds 31-35, default scenario.
  ident identity of the non-DF reruns with the earlier outputs.
Usage: analyze_a9.py <B> <out prefix> [analyze_a7.py]"""
import glob, json, os, sys
import numpy as np
B, OUT = sys.argv[1], sys.argv[2]
A7SRC = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), 'analyze_a7.py')
src = open(A7SRC).read().split("res = {'scen': {}, 'dec': {}}")[0]
src = src.replace("B = sys.argv[1]; OUT = sys.argv[2]; NB = 300", "NB = 300")
exec(src)
PUB = ['follow_me_cloud', 'wang_ton19_mdp', 'ouyang_jsac18', 'preble_iclr25', 'mooncake_fast25', 'loongserve_sosp24']
G = f'{B}/a9_gate'
def scen_runs(sc):
    """Runs of every policy in scenario sc: earlier gates (file prefix -> label) plus the A9 DF runs."""
    D = {}
    for gate in ('a3_gate', 'a5_gate', 'a6_gate', 'a7_gate'):
        p = f'{B}/{gate}/{sc}'
        if not os.path.isdir(p): continue
        for f in glob.glob(f'{p}/*_s*.json'):
            pre = os.path.basename(f)[:-5].rsplit('_s', 1)[0]; lab = 'follow_me_cloud' if pre == 'follow_tx' else pre
            D.setdefault(lab, {}).setdefault('gate', gate)
            if D[lab]['gate'] == gate:
                s = int(os.path.basename(f)[:-5].rsplit('_s', 1)[1]); D[lab][s] = load(f)
    D = {k: {s: v for s, v in d.items() if s != 'gate'} for k, d in D.items()}
    for q in ('rppdfb', 'mkdfb'):
        r = runs(f'{G}/{sc}/{q}_s*.json')
        if r: D[q] = r
    return D
MS = ('cont_p95', 'tpot_s_p95', 'tpot_b_p95')
res = dict(scen={}, param={}, fresh={}, ident={}, claims={})
SC91 = sorted(d for d in os.listdir(G) if os.path.isdir(f'{G}/{d}') and d not in ('param', 'fresh', 'ident'))
for sc in SC91:
    D = scen_runs(sc)
    if 'rppdfb' not in D: continue
    r = dict(table={k: pooled(d) for k, d in D.items()}, seeds={k: sorted(d) for k, d in D.items()},
             rppdfb={k: {m: boot(D['rppdfb'], D[k], m) for m in MS} for k in D if k != 'rppdfb'},
             rpp={k: {'cont_p95': boot(D['rpp'], D[k], 'cont_p95')} for k in D if k in PUB + ['mobikv']} if 'rpp' in D else {})
    res['scen'][sc] = r
# parameter variants of rppdfb against rppdfb's default configuration (a8_gate/default/rpp_dfb, seeds 21-25)
REF = runs(f'{B}/a8_gate/default/rpp_dfb_s*.json')
for f in sorted({os.path.basename(x)[:-5].rsplit('_s', 1)[0] for x in glob.glob(f'{G}/param/*_s*.json')}):
    V = runs(f'{G}/param/{f}_s*.json')
    res['param'][f] = dict(table=pooled(V), vs_default={m: boot(V, REF, m) for m in ('cont_p95', 'tpot_s_p95')})
# A9.2 fresh seeds
FR = {os.path.basename(x)[:-5].rsplit('_s', 1)[0] for x in glob.glob(f'{G}/fresh/*_s*.json')}
F = {k: runs(f'{G}/fresh/{k}_s*.json') for k in sorted(FR)}
if F:
    res['fresh'] = dict(table={k: pooled(d) for k, d in F.items()}, seeds={k: sorted(d) for k, d in F.items()},
                        rppdfb={k: {m: boot(F['rppdfb'], F[k], m) for m in MS + ('first_p95',)} for k in F if k != 'rppdfb'},
                        rpp={k: {m: boot(F['rpp'], F[k], m) for m in MS} for k in PUB + ['mobikv']},
                        mkdfb={k: {m: boot(F['mkdfb'], F[k], m) for m in MS} for k in ('mobikv',) + tuple(PUB)})
# identity checks: per-request rows must be equal
for f in sorted(glob.glob(f'{G}/ident/*_s21.json')):
    tag = os.path.basename(f)[:-9]
    ref = dict(rpp_err25='a7_gate/err25/rpp', rpp_porto='a7_gate/porto/rpp', rpp_mem16='a6_gate/mem16/rpp',
               rpp_cap300='a3_gate/cap300/rpp', ouyang_g50='a5_gate/g50/ouyang_jsac18', follow_tx_default='a3_gate/default/follow_tx')[tag]
    a = json.load(open(f))['per_request']; b = json.load(open(f'{B}/{ref}_s21.json'))['per_request']
    res['ident'][tag] = bool(a == b)
# ---- claims
hi = lambda x: x['hi']
pairs1 = [(sc, k) for sc, r in res['scen'].items() for k in PUB if k in r['rpp'] and hi(r['rpp'][k]['cont_p95']) < 1]
ok1 = [p for p in pairs1 if hi(res['scen'][p[0]]['rppdfb'][p[1]]['cont_p95']) < 1]
pairs2 = [(sc, k) for sc, r in res['scen'].items() for k in PUB if k in r['rppdfb']]
ok2 = [p for p in pairs2 if hi(res['scen'][p[0]]['rppdfb'][p[1]]['tpot_s_p95']) < 1]
sc3 = [sc for sc, r in res['scen'].items() if 'mobikv' in r['rpp'] and hi(r['rpp']['mobikv']['cont_p95']) < 1 and 'mkdfb' in r['rppdfb']]
ok3 = [sc for sc in sc3 if hi(res['scen'][sc]['rppdfb']['mkdfb']['cont_p95']) < 1]
res['claims']['A9.1-1'] = dict(n=len(pairs1), ok=len(ok1), frac=len(ok1) / max(1, len(pairs1)), fail=[p for p in pairs1 if p not in ok1], pass_=len(ok1) >= 0.9 * len(pairs1))
res['claims']['A9.1-2'] = dict(n=len(pairs2), ok=len(ok2), frac=len(ok2) / max(1, len(pairs2)), fail=[p for p in pairs2 if p not in ok2], pass_=len(ok2) >= 0.8 * len(pairs2))
res['claims']['A9.1-3'] = dict(n=len(sc3), ok=len(ok3), frac=len(ok3) / max(1, len(sc3)), fail=[s for s in sc3 if s not in ok3], pass_=len(ok3) >= 0.9 * len(sc3))
if F:
    fr = res['fresh']
    res['claims']['A9.2-1'] = dict(ratios={k: fr['rppdfb'][k]['cont_p95'] for k in PUB}, pass_=all(hi(fr['rppdfb'][k]['cont_p95']) < 1 for k in PUB))
    res['claims']['A9.2-2'] = dict(ratios={k: fr['rppdfb'][k]['tpot_s_p95'] for k in PUB}, pass_=all(hi(fr['rppdfb'][k]['tpot_s_p95']) < 1 for k in PUB))
    res['claims']['A9.2-3'] = dict(ratio=fr['rppdfb']['mkdfb']['cont_p95'], pass_=hi(fr['rppdfb']['mkdfb']['cont_p95']) < 1)
    res['claims']['A9.2-4'] = dict(ratio=fr['rpp']['mobikv']['cont_p95'], pass_=hi(fr['rpp']['mobikv']['cont_p95']) < 1)
json.dump(res, open(OUT + '.json', 'w'), indent=1, default=str)
ci = lambda x: f"{x['point']:.2f} [{x['lo']:.2f}, {x['hi']:.2f}]"
L = ['# A9 results\n', '## Claims']
for k, v in res['claims'].items():
    L.append(f"- {k}: {'PASS' if v['pass_'] else 'FAIL'} " + json.dumps({a: b for a, b in v.items() if a not in ('pass_',)}, default=str)[:900])
L.append('\n## Identity checks\n' + ', '.join(f"{k}: {v}" for k, v in res['ident'].items()))
if F:
    L += ['\n## A9.2 fresh seeds 31-35 (default)', '| policy | cont p95 | first p95 | TPOT_s p95 | TPOT_b p95 | bg GPU s | rppdfb/x cont | rppdfb/x TPOT_s | rppdfb/x TPOT_b |', '|---|---|---|---|---|---|---|---|---|']
    for k, t in res['fresh']['table'].items():
        c = res['fresh']['rppdfb'].get(k)
        L.append(f"| {k} | {t['cont_p95']:.2f} | {t['first_p95']:.2f} | {t['tpot_s_p95']*1e3:.0f} | {t['tpot_b_p95']*1e3:.0f} | {t['bg_gpu_s']:.0f} | "
                 + (f"{ci(c['cont_p95'])} | {ci(c['tpot_s_p95'])} | {ci(c['tpot_b_p95'])} |" if c else '– | – | – |'))
    L.append('\n' + '\n'.join(f"- rpp/{k}: cont {ci(v['cont_p95'])}, TPOT_s {ci(v['tpot_s_p95'])}" for k, v in res['fresh']['rpp'].items()))
    L.append('\n'.join(f"- mkdfb/{k}: cont {ci(v['cont_p95'])}, TPOT_s {ci(v['tpot_s_p95'])}" for k, v in res['fresh']['mkdfb'].items()))
L += ['\n## A9.1 per scenario (rppdfb / x: cont p95 ; TPOT_s p95)', '| scenario | rppdfb cont p95 | rpp cont p95 | rppdfb/rpp | rppdfb/mkdfb | rpp/mobikv | rppdfb/published (cont) | rppdfb/published (TPOT_s) |', '|---|---|---|---|---|---|---|---|']
for sc, r in res['scen'].items():
    t = r['table']; rd = r['rppdfb']
    pubc = '; '.join(f"{k[:6]} {ci(rd[k]['cont_p95'])}" for k in PUB if k in rd)
    pubt = '; '.join(f"{k[:6]} {ci(rd[k]['tpot_s_p95'])}" for k in PUB if k in rd)
    L.append(f"| {sc} | {t['rppdfb']['cont_p95']:.2f} | {t['rpp']['cont_p95']:.2f} | " if 'rpp' in t else f"| {sc} | {t['rppdfb']['cont_p95']:.2f} | – | ")
    L[-1] += (f"{ci(rd['rpp']['cont_p95'])} | " if 'rpp' in rd else '– | ') + (f"{ci(rd['mkdfb']['cont_p95'])} | " if 'mkdfb' in rd else '– | ') \
        + (f"{ci(r['rpp']['mobikv']['cont_p95'])} | " if 'mobikv' in r['rpp'] else '– | ') + f"{pubc or '–'} | {pubt or '–'} |"
L += ['\n## A9.1 parameter variants of rppdfb (vs rppdfb default, seeds 21-25)', '| variant | cont p95 | ratio cont | ratio TPOT_s |', '|---|---|---|---|']
for k, v in res['param'].items():
    L.append(f"| {k} | {v['table']['cont_p95']:.2f} | {ci(v['vs_default']['cont_p95'])} | {ci(v['vs_default']['tpot_s_p95'])} |")
open(OUT + '.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[:40]))
