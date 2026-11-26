#!/usr/bin/env python3
"""Two-GPU RPP measurements (ct_pp.py logs of the quiet pass): one row per cell for the paper's figure and text,
and an exact account of the correctness checks.

Link rate.  The bytes of a front arrive layer by layer; the layers after the first arrive within (last - first), so
the link delivered at least  nbytes*(27/28)*8 / (last - first)  in that trial (27/28: every model here has at least 28
layers).  A trial is link-bound when the nominal transfer time of its front exceeds the helper's compute time by 20%.
  * cap with link-bound trials in the run: delivered rate = cap * median(delivered/nominal over those trials); the
    cell's rho uses it ("verified").
  * cap without link-bound trials (the helper's computation limited the arrival): the delivered rate is only known to
    lie between the largest rate seen in any trial at that cap in the run and the cap.  The cell gets
    rho_min = rho at the cap and rho_max = rho at that largest seen rate ("unverified").

Correctness.  ct_pp.py compares the split's logits with chunked prefill on the executor in the first repetition of
every (cell, front fraction) only; greedy tokens are compared with monolithic prefill in every trial.

usage: python -m mobikv.analysis.rpp <rpp_root> --csv cells.csv --json checks.json
"""
import argparse, collections, glob, json, os, statistics as S

ap = argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('--csv', required=True); ap.add_argument('--json', required=True)
a = ap.parse_args()
NAMES = {'qwen25_7b': 'Qwen2.5-7B', 'qwen25_14b': 'Qwen2.5-14B', 'qwen3_8b': 'Qwen3-8B', 'llama31_8b': 'Llama-3.1-8B', 'mistral_7b': 'Mistral-7B-v0.3'}
rows_out = []; checks = {}
tot = collections.Counter()
for d in sorted(glob.glob(os.path.join(a.root, '*_g*'))):
    f = os.path.join(d, 'rpp.jsonl')
    if not os.path.exists(f): continue
    rows = [json.loads(l) for l in open(f) if l.strip()]
    pp = [r for r in rows if r['kind'] == 'pp']
    if not pp: continue
    slug = os.path.basename(d); model = NAMES[slug.rsplit('_g', 1)[0]]; share = int(slug.rsplit('_g', 1)[1])
    single = collections.defaultdict(list)
    for r in rows:
        if r['kind'] == 'single': single[r['cell']].append(r['single_ms'])
    per_tok = pp[0]['nbytes'] / (pp[0]['m'] - pp[0]['s'])
    bound_ratio = collections.defaultdict(list); seen = collections.defaultdict(float)
    for r in pp:
        span = r['last_layer_arrival_ms'] - r['first_layer_arrival_ms']
        if span <= 0: continue
        rate = r['nbytes'] * (27 / 28) * 8 / (span / 1e3) / 1e6
        seen[r['cap_mbps']] = max(seen[r['cap_mbps']], rate)
        if r['nbytes'] * 8 / (r['cap_mbps'] * 1e6) * 1e3 > 1.2 * r['helper_comp_ms']: bound_ratio[r['cap_mbps']].append(rate / r['cap_mbps'])
    # correctness account
    cmp_ = [r for r in pp if r['logit_diff_vs_local_chunked'] is not None]
    c = dict(trials=len(pp), compared_with_chunked=len(cmp_), equal_to_chunked=sum(r['logit_diff_vs_local_chunked'] == 0 for r in cmp_),
             max_abs_diff_to_chunked=max(r['logit_diff_vs_local_chunked'] for r in cmp_),
             chunked_differs_from_monolithic=sum((r.get('local_chunked_vs_single') or 0) != 0 for r in cmp_),
             greedy_equal_monolithic=sum(r['agree'] == r['n_dec'] for r in pp),
             compared_reps=sorted({r['rep'] for r in cmp_}))
    # repetitions of one (cell, front fraction): do they give the same distance to the monolithic reference?
    grp = collections.defaultdict(set)
    for r in pp: grp[(r['cell'], r['x'])].add((r['logit_maxabs_diff'], r['agree']))
    c['configs'] = len(grp); c['configs_with_identical_repetitions'] = sum(len(v) == 1 for v in grp.values())
    checks[slug] = c
    for k in ('trials', 'compared_with_chunked', 'equal_to_chunked', 'chunked_differs_from_monolithic', 'greedy_equal_monolithic', 'configs', 'configs_with_identical_repetitions'): tot[k] += c[k]
    cells = collections.OrderedDict()
    for r in pp: cells.setdefault(r['cell'], []).append(r)
    for cell, rs in cells.items():
        C = S.median(single[cell]); cap = rs[0]['cap_mbps']; s, e = rs[0]['s'], rs[0]['e']
        rho_nom = (e - s) * per_tok * 8 / (cap * 1e6) * 1e3 / C
        byx = collections.defaultdict(list)
        for r in rs: byx[r['x']].append(r['split_ms'])
        bx, bt = min(((x, S.median(v)) for x, v in byx.items()), key=lambda t: t[1])
        reps_faster = sum(t < C for t in byx[bx]); nrep = len(byx[bx])
        if bound_ratio.get(cap):
            fac = min(1.0, S.median(bound_ratio[cap])); rho = rho_nom / fac; lo = hi = rho; ver = 1
        else:
            fac = min(1.0, seen[cap] / cap); rho = rho_nom; lo = rho_nom; hi = rho_nom / fac; ver = 0
        rows_out.append(dict(run=slug, model=model, share=share, cell=cell, new_tokens=e - s, prefix=s, cap_mbps=cap, C_ms=C, rho=rho, rho_min=lo, rho_max=hi,
                             verified=ver, best_front_fraction=bx, speedup=C / bt, bound_at_rho_min=min(2.0, 1 + 1 / lo), bound_at_rho_max=min(2.0, 1 + 1 / hi),
                             best_split_reps_faster=reps_faster, reps=nrep, link_bound_trials=len(bound_ratio.get(cap, [])),
                             delivered_over_cap=fac))
checks['total'] = dict(tot)
cols = list(rows_out[0].keys())
with open(a.csv, 'w') as fh:
    fh.write(','.join(cols) + '\n')
    for r in rows_out: fh.write(','.join((f'{v:.4f}' if isinstance(v, float) else str(v)) for v in r.values()) + '\n')
json.dump(checks, open(a.json, 'w'), indent=1)
for k, v in checks.items(): print(k, v)
sure = [r for r in rows_out if r['rho_max'] < 1]; n8 = [r for r in sure if r['new_tokens'] >= 4096]
print(f"\ncells: {len(rows_out)} ({sum(r['verified'] for r in rows_out)} with a verified link rate)")
print(f"cells with rho certainly below 1: {len(sure)}; speedup {min(r['speedup'] for r in sure):.2f}-{max(r['speedup'] for r in sure):.2f}; "
      f"of them with at least 4096 new tokens: {len(n8)}, speedup {min(r['speedup'] for r in n8):.2f}-{max(r['speedup'] for r in n8):.2f}; models {sorted({r['model'] for r in sure})}")
v = [r for r in rows_out if r['verified']]
print(f"verified cells: rho {min(r['rho'] for r in v):.2f}-{max(r['rho'] for r in v):.2f}; speedup / bound {min(r['speedup'] / r['bound_at_rho_min'] for r in v):.2f}-{max(r['speedup'] / r['bound_at_rho_min'] for r in v):.2f}; "
      f"best split faster than single-GPU prefill in every repetition: {sum(r['best_split_reps_faster'] == r['reps'] for r in v)} of {len(v)} cells")
u = [r for r in rows_out if not r['verified']]
print(f"unverified cells: {len(u)}; rho_min {min(r['rho_min'] for r in u):.2f}-{max(r['rho_min'] for r in u):.2f}; rho_max {min(r['rho_max'] for r in u):.2f}-{max(r['rho_max'] for r in u):.2f}; "
      f"speedup {min(r['speedup'] for r in u):.2f}-{max(r['speedup'] for r in u):.2f}; speedup / bound at rho_min {min(r['speedup'] / r['bound_at_rho_min'] for r in u):.2f}-{max(r['speedup'] / r['bound_at_rho_min'] for r in u):.2f}")
for lo_, hi_ in ((0, 1), (1, 2), (2, 4), (4, 20)):
    w = [r for r in v if lo_ <= r['rho'] < hi_]
    if w: print(f"verified, rho in [{lo_},{hi_}): {len(w)} cells, speedup {min(r['speedup'] for r in w):.2f}-{max(r['speedup'] for r in w):.2f}, of bound {min(r['speedup'] / r['bound_at_rho_min'] for r in w):.2f}-{max(r['speedup'] / r['bound_at_rho_min'] for r in w):.2f}")
allr = rows_out
print(f"all cells: best split faster than single-GPU prefill in every repetition in {sum(r['best_split_reps_faster'] == r['reps'] for r in allr)} of {len(allr)}; "
      f"not faster: {[(r['run'], r['cell'], round(r['speedup'], 3)) for r in allr if r['best_split_reps_faster'] != r['reps']]}")
