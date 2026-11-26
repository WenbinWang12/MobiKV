#!/usr/bin/env python3
"""Run the final two-GPU RPP cells with measured layer transport and split checks."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mobikv.measurement._launch import Run, cpu_stat


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True, help='New or empty directory, conventionally <slug>_g<share>')
    ap.add_argument('--gpus', required=True, help='helper,executor GPU identifiers')
    ap.add_argument('--share', type=int, choices=(25, 100), required=True)
    ap.add_argument('--caps', default='930,2500,5000,10000')
    ap.add_argument('--fracs', default='0.0625,0.125,0.1875,0.25,0.375,0.5,0.625')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--port', type=int, default=29540)
    ap.add_argument('--timeout', type=float, default=7200)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    gpus = a.gpus.split(',')
    if len(gpus) != 2 or len(set(gpus)) != 2:
        ap.error('--gpus needs two different GPU identifiers')
    caps = [int(s) for s in a.caps.split(',')]
    fracs = [float(s) for s in a.fracs.split(',')]
    if not caps or min(caps) <= 0 or not fracs or any(not 0 < f < 1 for f in fracs) or a.reps < 1:
        ap.error('Caps/repetitions must be positive; split fractions must be inside (0, 1)')
    cells = ';'.join(f'{label}-c{cap}:{s}:{e}:{cap}' for cap in caps
                     for label, s, e in [('4k', 2048, 4096), ('8k', 4096, 8192), ('16k', 8192, 16384)])
    run = Run(a.out, a.share, a.dry_run)
    error = None
    try:
        helper = run.spawn('mobikv.measurement.ct_pp',
                           ['--role', 'helper', '--model', a.model, '--port', a.port], gpus[0], 'helper.log')
        run.ready(helper, 'helper.log', 480)
        run.receipt['cpu_measurement_start'] = cpu_stat()
        server = run.spawn('mobikv.measurement.ct_pp',
                           ['--role', 'server', '--model', a.model, '--peer', '127.0.0.1', '--port', a.port,
                            '--cells', cells, '--reps', a.reps, '--fracs', a.fracs, '--out', run.out / 'rpp.jsonl', '--quit'],
                           gpus[1], 'server.log')
        run.wait([server, helper], a.timeout)
        if not a.dry_run:
            rows = [json.loads(s) for s in (run.out / 'rpp.jsonl').read_text().splitlines()]
            if not rows or rows[-1].get('kind') != 'done':
                raise RuntimeError('RPP server did not emit its final done row')
    except BaseException as exc:
        error = exc
        raise
    finally:
        run.finish(error)


if __name__ == '__main__':
    main()
