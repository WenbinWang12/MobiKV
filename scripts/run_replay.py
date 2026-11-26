#!/usr/bin/env python3
"""Replay simulator job logs on three real GPUs using the final paper harness."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mobikv.measurement._launch import Run

POLICIES = ['follow_me_cloud', 'wang_ton19_mdp', 'ouyang_jsac18', 'preble_iclr25',
            'mooncake_fast25', 'loongserve_sosp24', 'mobikv']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--joblog-root', type=Path, required=True, help='Input containing <policy>/joblog.json')
    ap.add_argument('--out', type=Path, required=True, help='Output containing one new directory per policy')
    ap.add_argument('--model', required=True)
    ap.add_argument('--gpus', required=True, help='Three GPU identifiers, e.g. 0,1,2')
    ap.add_argument('--share', type=int, choices=(25, 100), default=25)
    ap.add_argument('--cap-mbps', type=float, default=930)
    ap.add_argument('--policies', nargs='+', choices=POLICIES, default=POLICIES)
    ap.add_argument('--edges', default='(1, 2);(2, 2);(1, 3)')
    ap.add_argument('--port', type=int, default=38110)
    ap.add_argument('--bmax', type=int, default=8)
    ap.add_argument('--lmax', type=int, default=32768)
    ap.add_argument('--timeout', type=float, default=3000)
    ap.add_argument('--bg-dfb', type=int, choices=(0, 1), help='Override only for job logs generated with this setting')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    gpus = a.gpus.split(',')
    if len(gpus) != 3 or len(set(gpus)) != 3 or len(a.edges.split(';')) != 3:
        ap.error('Three different GPUs and three edges are required')
    if a.cap_mbps <= 0 or a.bmax < 1 or a.lmax < 1:
        ap.error('The transport cap and decode buffer dimensions must be positive')
    inputs = {p: (a.joblog_root / p / 'joblog.json').resolve() for p in a.policies}
    for policy, source in inputs.items():
        if not source.is_file():
            ap.error(f'Missing job log: {source}')
        jl = json.loads(source.read_text())
        if any(k not in jl for k in ('window', 'NL', 'jobs', 'reqs')):
            ap.error(f'Expected final simulator job-log schema: {source}')
        if any(e not in jl['jobs'] for e in a.edges.split(';')):
            ap.error(f'Replayed edge absent in {source}')
        destination = a.out / policy
        if not a.dry_run and destination.exists() and any(destination.iterdir()):
            ap.error(f'Use a new output directory; existing results found in {destination}')
    for k, policy in enumerate(a.policies):
        run = Run(a.out / policy, a.share, a.dry_run)
        source = inputs[policy]
        run.receipt['joblog_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        run.receipt['policy'] = policy
        dfb = a.bg_dfb if a.bg_dfb is not None else int(policy == 'mobikv')
        joblog = run.out / 'joblog.json'
        if not a.dry_run:
            shutil.copy2(source, joblog)
        error = None
        try:
            procs = []
            for me, gpu in enumerate(gpus):
                procs.append(run.spawn('mobikv.measurement.replay',
                    ['--joblog', joblog, '--edges', a.edges, '--hosts', '127.0.0.1,127.0.0.1,127.0.0.1',
                     '--me', me, '--port', a.port + 10 * k, '--same_host', '--model', a.model,
                     '--out', run.out / f'replay_me{me}.json', '--bg_dfb', dfb, '--bmax', a.bmax,
                     '--lmax', a.lmax, '--cap_mbps', a.cap_mbps, '--null_mbps', a.cap_mbps],
                    gpu, f'replay_me{me}.log'))
            run.wait(procs, a.timeout)
            if not a.dry_run:
                results = [json.loads((run.out / f'replay_me{i}.json').read_text()) for i in range(3)]
                checks = [{'n_done': r['n_done'], 'n_fg': r['n_fg'], 'send_err': r['send_err'],
                           'wait_timeouts': r['wait_timeouts']} for r in results]
                run.receipt['workers'] = checks
                if any(r['n_done'] != r['n_fg'] or r['send_err'] or r['wait_timeouts'] for r in checks):
                    raise RuntimeError('Replay failed its completion/transport acceptance checks')
        except BaseException as exc:
            error = exc
            raise
        finally:
            run.finish(error)


if __name__ == '__main__':
    main()
