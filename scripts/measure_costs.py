#!/usr/bin/env python3
"""Measure one model/share grid, verifying that the worker attached to MPS."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mobikv.measurement._launch import Run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--tag', required=True, help='Canonical model name, without the -g<share> suffix')
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--share', type=int, choices=(25, 100), required=True)
    ap.add_argument('--out', required=True, help='New or empty output directory')
    ap.add_argument('--timeout', type=float, default=7200)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('profile_args', nargs=argparse.REMAINDER, help='Arguments after -- are passed to profile_costs')
    a = ap.parse_args()
    extra = a.profile_args[1:] if a.profile_args[:1] == ['--'] else a.profile_args
    if any(x.split('=')[0] in ('--out', '--tag', '--model') for x in extra):
        ap.error('Set --out, --tag, and --model on the launcher, not in profile_args')
    run = Run(a.out, a.share, a.dry_run)
    error = None
    try:
        proc = run.spawn('mobikv.measurement.profile_costs',
                         ['--model', a.model, '--tag', f'{a.tag}-g{a.share}',
                          '--out', run.out / 'ct_compute.jsonl', '--reps', '3', *extra], a.gpu, 'profile.log')
        run.wait([proc], a.timeout)
        if not a.dry_run:
            import json
            rows = [json.loads(s) for s in (run.out / 'ct_compute.jsonl').read_text().splitlines()]
            if not rows or rows[-1].get('kind') != 'done':
                raise RuntimeError('Profile did not emit its final done row')
    except BaseException as exc:
        error = exc
        raise
    finally:
        run.finish(error)


if __name__ == '__main__':
    main()
