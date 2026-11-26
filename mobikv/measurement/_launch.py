"""Shared process management for the paper's GPU measurement launchers."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def cpu_stat():
    path = Path('/sys/fs/cgroup/cpu.stat')
    if not path.is_file():
        return None
    return {k: int(v) for k, v in (line.split() for line in path.read_text().splitlines())}


class Run:
    """Own only the child processes started by one measurement run."""

    def __init__(self, out, share, dry_run=False):
        self.out = Path(out).resolve()
        self.share = share
        self.dry_run = dry_run
        self.children = []
        self.receipt = {'share': share, 'commands': [], 'cpu_before': cpu_stat()}
        self.control = None
        self.offset = 0
        if dry_run:
            return
        if not sys.platform.startswith('linux'):
            raise RuntimeError('GPU measurements require Linux, CUDA, and an existing NVIDIA MPS daemon')
        pipe = os.environ.get('CUDA_MPS_PIPE_DIRECTORY')
        log = os.environ.get('CUDA_MPS_LOG_DIRECTORY')
        if not pipe or not log:
            raise RuntimeError('Set CUDA_MPS_PIPE_DIRECTORY and CUDA_MPS_LOG_DIRECTORY for the running MPS daemon')
        self.control = Path(log) / 'control.log'
        if not self.control.is_file():
            raise RuntimeError(f'MPS control log is missing: {self.control}')
        self.offset = self.control.stat().st_size
        if self.out.exists() and any(self.out.iterdir()):
            raise RuntimeError(f'Output directory must be new or empty: {self.out}')
        self.out.mkdir(parents=True, exist_ok=True)

    def spawn(self, module, args, gpu, log):
        cmd = [sys.executable, '-u', '-m', module, *map(str, args)]
        self.receipt['commands'].append({'gpu': str(gpu), 'argv': cmd})
        if self.dry_run:
            print(json.dumps(self.receipt['commands'][-1]))
            return None
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(self.share),
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        stream = (self.out / log).open('w', encoding='utf-8')
        try:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        finally:
            stream.close()
        self.children.append(proc)
        return proc

    def ready(self, proc, logfile, timeout):
        if self.dry_run:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if 'listening' in (self.out / logfile).read_text(errors='replace'):
                return
            if proc.poll() is not None:
                raise RuntimeError(f'Helper exited before listening; inspect {self.out / logfile}')
            time.sleep(0.2)
        raise RuntimeError('Timed out waiting for the helper to load the model')

    def wait(self, procs, timeout):
        if self.dry_run:
            return
        deadline = time.monotonic() + timeout
        while any(p.poll() is None for p in procs):
            if any(p.poll() not in (None, 0) for p in procs):
                raise RuntimeError('A GPU worker failed; inspect its log')
            if time.monotonic() >= deadline:
                raise RuntimeError('GPU run exceeded its timeout')
            time.sleep(0.2)
        if any(p.returncode != 0 for p in procs):
            raise RuntimeError('A GPU worker failed; inspect its log')

    def finish(self, error=None):
        if self.dry_run:
            return
        self.receipt['cpu_after'] = cpu_stat()
        with self.control.open('rb') as stream:
            stream.seek(self.offset)
            lines = stream.read().decode(errors='replace')
        seen = {str(p.pid): bool(re.search(r'NEW CLIENT\s+' + str(p.pid) + r'\s', lines)) for p in self.children}
        self.receipt['mps_client_seen'] = seen
        self.receipt['exit_codes'] = {str(p.pid): p.poll() for p in self.children}
        self.receipt['error'] = str(error) if error else None
        self.receipt['accepted'] = error is None and bool(seen) and all(seen.values())
        for p in self.children:
            if p.poll() is None:
                p.terminate()
        for p in self.children:
            if p.poll() is None:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        (self.out / 'run.json').write_text(json.dumps(self.receipt, indent=2) + '\n', encoding='utf-8')
        if error is None and not self.receipt['accepted']:
            raise RuntimeError('MPS attachment was not verified for every worker; run is not accepted')
