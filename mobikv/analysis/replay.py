#!/usr/bin/env python3
"""Pool the three-server real-GPU replays over seeds (protocol: PROTOCOL_replay_20261123.md, extension).

Reported run per policy and seed:
  seed 41: the quiet-node run, i.e. replicate 2 (replay_slow_rep2) for mobikv and loongserve_sosp24 and replicate 1
           (replay_slow) for the other five policies;  seeds 42, 43: replay_slow_seed42 / replay_slow_seed43.
The other seed-41 replicate of every policy is summarised as a repeatability check.

For follow-up requests of evaluation-half sessions executed at the three replayed servers:
  * pooled request-level percentiles of measured TTFT (s) and TPOT (ms), with a 95% bootstrap interval that resamples
    requests within each seed (2000 resamples, fixed seed);
  * the simulator's percentiles for the same requests and the median relative error per request;
  * MobiKV relative to every baseline: ratio of the pooled percentiles with a 95% interval from the same stratified
    bootstrap (the two policies are resampled independently; requests are not paired across policies);
  * per-seed percentiles.
Also: acceptance and execution checks per run (jobs finished, layer-wait timeouts, send errors, measured / planned
duration of prefill jobs by kind).

usage: python -m mobikv.analysis.replay <work_dir> [--json out.json] [--tex rows.tex] [--csv out.csv]
"""
import argparse, json, math, os, random, statistics as S

ap = argparse.ArgumentParser()
ap.add_argument('work'); ap.add_argument('--json', default=''); ap.add_argument('--tex', default=''); ap.add_argument('--csv', default='')
ap.add_argument('--boot', type=int, default=2000)
a = ap.parse_args()
ORDER = [('follow_me_cloud', 'Follow-Me'), ('wang_ton19_mdp', 'MDP'), ('ouyang_jsac18', 'Lyapunov'), ('preble_iclr25', 'Preble'),
         ('mooncake_fast25', 'Mooncake'), ('loongserve_sosp24', 'ESP'), ('mobikv', 'MobiKV')]
QUIET41 = {'mobikv': 'replay_slow_rep2', 'loongserve_sosp24': 'replay_slow_rep2'}
OTHER41 = {'mobikv': 'replay_slow', 'loongserve_sosp24': 'replay_slow'}
PCT = (.5, .9, .99)


def root(policy, seed, other=False):
    if seed == 41:
        if other: return OTHER41.get(policy, 'replay_slow_rep2')
        return QUIET41.get(policy, 'replay_slow')
    return f'replay_slow_seed{seed}'


def quant(v, p):
    if not v: return float('nan')
    v = sorted(v); pos = (len(v) - 1) * p; lo = math.floor(pos); hi = math.ceil(pos)
    return v[lo] if lo == hi else v[lo] * (1 - (pos - lo)) + v[hi] * (pos - lo)


def load(policy, seed, other=False):
    d = os.path.join(a.work, root(policy, seed, other), policy)
    reps = [json.load(open(os.path.join(d, f'replay_me{i}.json'))) for i in range(3)]
    jl = json.load(open(os.path.join(d, 'joblog.json')))
    ratio = {'plain': [], 'plain_all': [], 'ppf': [], 'ppb': []}
    for r in reps:
        plan = {}
        for j in jl['jobs'].get(r['edge'], []):
            if j[0] == 'fg': plan[(j[7], j[8] if len(j) > 8 else '', round(j[3], 6))] = j
        for m in r['jobs']:
            j = plan.get((m['rid'], m['role'], round(m['st'], 6)))
            if not j: continue
            pd = j[4] - j[3]; md = m['me'] - m['ms']
            if pd <= 0: continue
            if m['role'] == '':
                ratio['plain_all'].append((pd, md))
                if pd > 0.3: ratio['plain'].append(md / pd)
            else: ratio['ppf' if m['role'] == 'ppf' else 'ppb'].append(md / pd)
    acc = dict(root=root(policy, seed, other), jobs_done=sum(r['n_done'] for r in reps), jobs=sum(r['n_fg'] for r in reps),
               send_err=sum(r['send_err'] for r in reps), wait_timeouts=sum(r['wait_timeouts'] for r in reps),
               share=sorted({r['share'] for r in reps}), cap=sorted({r.get('cap_mbps') for r in reps}), mps=sorted({str(r.get('mps_pipe')) for r in reps}),
               plain=(S.median(ratio['plain']) if ratio['plain'] else None, len(ratio['plain'])),
               plain_time_weighted=(sum(m for _, m in ratio['plain_all']) / sum(p for p, _ in ratio['plain_all'])) if ratio['plain_all'] else None,
               ppf=(S.median(ratio['ppf']) if ratio['ppf'] else None, len(ratio['ppf'])), ppb=(S.median(ratio['ppb']) if ratio['ppb'] else None, len(ratio['ppb'])))
    reqs = [q for r in reps for q in r['reqs'] if q['eval']]
    return acc, reqs


def stats(reqs):
    tt = [q['ttft_meas'] for q in reqs]; ts = [q['ttft_sim'] for q in reqs]; tp = [q['tpot_meas'] * 1e3 for q in reqs if q['tpot_meas'] is not None]
    err = [abs(q['ttft_meas'] - q['ttft_sim']) / q['ttft_sim'] for q in reqs if q['ttft_sim'] > 0]
    return dict(n=len(reqs), ttft=[quant(tt, p) for p in PCT], tpot=[quant(tp, p) for p in PCT], ttft_sim=[quant(ts, p) for p in PCT],
                med_rel_err=S.median(err) if err else float('nan'), split=sum(q['rpp'] for q in reqs), n_tpot=len(tp))


SEEDS = (41, 42, 43)
data = {}; acc = {}; other = {}
for key, _ in ORDER:
    data[key] = {}; acc[key] = {}
    for s in SEEDS:
        ac, reqs = load(key, s); acc[key][s] = ac; data[key][s] = [q for q in reqs if q['turn'] >= 2]
    ac, reqs = load(key, 41, other=True); other[key] = (ac, [q for q in reqs if q['turn'] >= 2])

rng = random.Random(20261124)
def resample(per_seed):
    out = []
    for s in SEEDS:
        v = per_seed[s]
        out += [v[rng.randrange(len(v))] for _ in v]
    return out
def col(reqs, kind, p):
    v = [q['ttft_meas'] for q in reqs] if kind == 'ttft' else [q['tpot_meas'] * 1e3 for q in reqs if q['tpot_meas'] is not None]
    return quant(v, p)
boot = {key: [resample(data[key]) for _ in range(a.boot)] for key, _ in ORDER}
def ci(xs):
    xs = sorted(xs); n = len(xs); return xs[int(0.025 * n)], xs[int(0.975 * n) - 1]

res = {}
for key, label in ORDER:
    pooled = [q for s in SEEDS for q in data[key][s]]
    st = stats(pooled)
    st['ttft_ci'] = [ci([col(b, 'ttft', p) for b in boot[key]]) for p in PCT]
    st['tpot_ci'] = [ci([col(b, 'tpot', p) for b in boot[key]]) for p in PCT]
    st['per_seed'] = {s: stats(data[key][s]) for s in SEEDS}
    st['other41'] = stats(other[key][1]); st['other41_root'] = other[key][0]['root']
    res[key] = dict(label=label, pooled=st, acc={s: acc[key][s] for s in SEEDS}, acc_other41=other[key][0])
for key, label in ORDER:
    if key == 'mobikv': continue
    r = {}
    for kind in ('ttft', 'tpot'):
        for i, p in enumerate(PCT):
            point = res['mobikv']['pooled'][kind][i] / res[key]['pooled'][kind][i]
            lo, hi = ci([col(bm, kind, p) / col(bb, kind, p) for bm, bb in zip(boot['mobikv'], boot[key])])
            r[f'{kind}_p{int(p * 100)}'] = (point, lo, hi)
    res[key]['mobikv_over'] = r

print('##### acceptance and execution checks (reported runs)')
print('policy     seed root                 jobs       share cap  | meas/plan: plain>0.3s median (n), all plain time-weighted, front (n), back (n) | send_err timeouts')
for key, label in ORDER:
    for s in SEEDS:
        c = acc[key][s]; f = lambda t: ('%.2f (%d)' % t) if t[0] is not None else '-'
        print(f"{label:10s} {s} {c['root']:20s} {c['jobs_done']:4d}/{c['jobs']:<4d} {'/'.join(map(str, c['share'])):>4s} {'/'.join(str(int(x)) for x in c['cap']):>4s} |"
              f" {f(c['plain']):>11s}  {c['plain_time_weighted']:.2f}  {f(c['ppf']):>9s} {f(c['ppb']):>9s} | {c['send_err']} {c['wait_timeouts']}")
print('\n##### follow-up requests, seeds 41-43 pooled; TTFT s, TPOT ms; [95% bootstrap interval, stratified by seed]')
print('policy       n   TTFT P50             P90               P99               | TPOT P50         P90             P99              | sim TTFT P50/P90/P99  med.err split')
for key, label in ORDER:
    b = res[key]['pooled']
    tt = '  '.join(f"{b['ttft'][i]:6.2f}[{b['ttft_ci'][i][0]:.2f},{b['ttft_ci'][i][1]:.2f}]" for i in range(3))
    tp = '  '.join(f"{b['tpot'][i]:6.1f}[{b['tpot_ci'][i][0]:.0f},{b['tpot_ci'][i][1]:.0f}]" for i in range(3))
    print(f"{label:10s} {b['n']:4d}  {tt} | {tp} | {b['ttft_sim'][0]:5.2f} {b['ttft_sim'][1]:5.2f} {b['ttft_sim'][2]:5.2f}  {b['med_rel_err'] * 100:5.1f}%  {b['split']:3d}")
print('\n##### MobiKV / baseline, ratio of pooled percentiles [95% bootstrap interval]')
print('baseline     TTFT P50            P90                P99               | TPOT P50            P90                P99')
for key, label in ORDER:
    if key == 'mobikv': continue
    r = res[key]['mobikv_over']
    print(f"{label:10s} " + '  '.join(f"{r[k][0]:.2f}[{r[k][1]:.2f},{r[k][2]:.2f}]" for k in ('ttft_p50', 'ttft_p90', 'ttft_p99')) + ' | '
          + '  '.join(f"{r[k][0]:.2f}[{r[k][1]:.2f},{r[k][2]:.2f}]" for k in ('tpot_p50', 'tpot_p90', 'tpot_p99')))
print('\n##### per seed (reported runs) and the other seed-41 replicate: n, TTFT P50/P90/P99 s, TPOT P50/P90/P99 ms, median error')
for key, label in ORDER:
    for s in SEEDS:
        b = res[key]['pooled']['per_seed'][s]
        print(f"{label:10s} seed {s}       n={b['n']:3d}  {b['ttft'][0]:6.2f} {b['ttft'][1]:6.2f} {b['ttft'][2]:6.2f} | {b['tpot'][0]:6.1f} {b['tpot'][1]:6.1f} {b['tpot'][2]:7.1f} | {b['med_rel_err'] * 100:5.1f}%")
    b = res[key]['pooled']['other41']
    print(f"{label:10s} seed 41 other n={b['n']:3d}  {b['ttft'][0]:6.2f} {b['ttft'][1]:6.2f} {b['ttft'][2]:6.2f} | {b['tpot'][0]:6.1f} {b['tpot'][1]:6.1f} {b['tpot'][2]:7.1f} | {b['med_rel_err'] * 100:5.1f}%   ({res[key]['pooled']['other41_root']})")
if a.json: json.dump(res, open(a.json, 'w'), indent=1)
if a.csv:
    with open(a.csv, 'w') as f:
        f.write('policy,scope,n,ttft_p50_s,ttft_p90_s,ttft_p99_s,tpot_p50_ms,tpot_p90_ms,tpot_p99_ms,sim_ttft_p50_s,sim_ttft_p90_s,sim_ttft_p99_s,median_rel_err,split_requests\n')
        for key, label in ORDER:
            rows = [('pooled_41_43', res[key]['pooled'])] + [(f'seed_{s}', res[key]['pooled']['per_seed'][s]) for s in SEEDS] + [('seed_41_other_replicate', res[key]['pooled']['other41'])]
            for scope, b in rows:
                f.write(','.join([key, scope, str(b['n'])] + [f'{x:.4f}' for x in b['ttft']] + [f'{x:.2f}' for x in b['tpot']] + [f'{x:.4f}' for x in b['ttft_sim']]
                                 + [f"{b['med_rel_err']:.4f}", str(b['split'])]) + '\n')
if a.tex:
    BS = chr(92); L = []
    best = {(k, j): min(res[key]['pooled'][k][j] for key, _ in ORDER) for k in ('ttft', 'tpot') for j in range(3)}
    for key, label in ORDER:
        b = res[key]['pooled']
        def c(kind, j, fmt):
            v = b[kind][j]; s = (fmt % v) + f" [{b[kind + '_ci'][j][0]:{fmt[1:]}}, {b[kind + '_ci'][j][1]:{fmt[1:]}}]"
            return (BS + 'textbf{' + s + '}') if abs(v - best[(kind, j)]) < 1e-12 else s
        name = (BS + 'textbf{MobiKV (prop.)}') if key == 'mobikv' else label
        L.append(f"{name} & {b['n']} & " + ' & '.join(c('ttft', j, '%.2f') for j in range(3)) + ' & ' + ' & '.join(c('tpot', j, '%.1f') for j in range(3))
                 + f" & {b['med_rel_err'] * 100:.1f} " + BS * 2)
    open(a.tex, 'w').write('\n'.join(L) + '\n')
