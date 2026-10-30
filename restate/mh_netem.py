"""Owned X5 RTT treatment on holder ingress and experiment TCP ports only.

The entire requested RTT increment is placed on target->holder packets. Both
holders use this same rule; the target adds no delay. Exact rules, IFB alias,
interface root, durable mutation journal and cleanup are inherited from the
reviewed ScopedNetem. A separate process restores the owned treatment when the
owner exits or its renewable lease expires, including SIGKILL.
"""
from __future__ import annotations

import restate_config
import argparse
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from scoped_netem import ScopedNetem, Commands, TOOL_ROOT, sha


def process_start_ticks(pid):
    try:
        text = Path('/proc/%d/stat' % pid).read_text()
        return text.rsplit(') ', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def owner_alive(state):
    return (state.get('owner_start_ticks') is not None and
            process_start_ticks(state['owner_pid']) == state['owner_start_ticks'])


class HolderIngressDelay(ScopedNetem):
    def __init__(self, local_ip, peer_ip, run_id, allowed_rtt_ms, *, base_port=33080,
                 journal=None, lease_seconds=180, commands=None):
        for value in (local_ip, peer_ip):
            addr = ipaddress.ip_address(value)
            if addr.version != 4 or addr.is_loopback or addr.is_unspecified:
                raise ValueError('Exact non-loopback IPv4 endpoints required')
        if local_ip == peer_ip or type(base_port) is not int or not 1024 <= base_port <= 65518:
            raise ValueError('Distinct endpoints and a bounded experiment port range required')
        if not isinstance(run_id, str) or not run_id or len(run_id) > 180:
            raise ValueError('Explicit bounded run identity required')
        if not allowed_rtt_ms or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
                                     for x in allowed_rtt_ms):
            raise ValueError('Explicit finite nonnegative RTT grid required')
        if type(lease_seconds) not in (int, float) or not 30 <= lease_seconds <= 7200:
            raise ValueError('Lease must be 30..7200 seconds')
        self.allowed_rtt_ms = tuple(float(x) for x in allowed_rtt_ms)
        self.plan = dict(schema_version=1, local_ip=local_ip, peer_ip=peer_ip,
            run_id=run_id, interface='eth0', base_port=base_port,
            ports=[base_port, *range(base_port + 10, base_port + 18)],
            allowed_rtt_ms=list(self.allowed_rtt_ms), tool_root=TOOL_ROOT,
            lease_seconds=lease_seconds, full_rtt_on_holder_ingress=True,
            extension_sha256=sha(__file__))
        self.plan_sha256 = hashlib.sha256(json.dumps(self.plan, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        self.ifb = 'rstx5' + self.plan_sha256[:8]
        self.alias = 'restate-x5:' + self.plan_sha256
        self.journal = (Path(journal) if journal is not None else
            Path(restate_config.RESULTS) / ('x5-netem-%s-%d.json' % (self.plan_sha256[:12], os.getpid()))).absolute()
        self.commands = commands or Commands()
        self.state = None
        self._launch_watchdog = False

    def _rules(self):
        return [(52000 + i * 2 + side, i * 2 + side + 1, direction, port)
                for i, port in enumerate(self.plan['ports'])
                for side, direction in enumerate(('dst_port', 'src_port'))]

    def _save(self):
        if self.state is not None and 'owner_start_ticks' not in self.state:
            self.state['owner_start_ticks'] = process_start_ticks(self.state['owner_pid'])
        super()._save()
        # Start before the first mutation. The watchdog waits on the interface
        # lock while setup is active, then recovers any partial setup.
        if self._launch_watchdog and self.state['status'] == 'setting_up' and 'watchdog_pid' not in self.state:
            with self.journal.with_suffix('.watchdog.log').open('ab') as output:
                child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                    '--watchdog', str(self.journal)], stdin=subprocess.DEVNULL,
                    stdout=output, stderr=output, start_new_session=True, close_fds=True)
            self.state['watchdog_pid'] = child.pid
            super()._save()

    def setup(self, start_watchdog=True):
        self._launch_watchdog = bool(start_watchdog)
        return super().setup(start_watchdog=False)

    def _snapshot_unlocked(self, require_active=False):
        result = super()._snapshot_unlocked(require_active=require_active)
        # ScopedNetem stores a two-ended RTT and checks half on this interface.
        # Internally store 2*rtt, then expose the actual one-ended increment.
        requested = self.state['delay_ms'] / 2
        result.update(requested_added_rtt_ms=requested,
            observed_added_rtt_ms=result['observed_one_way_delay_ms'],
            direction='target->holder TCP, experiment port allowlist, full RTT increment once',
            ports=list(self.plan['ports']), journal=str(self.journal),
            owner_pid=self.state['owner_pid'], owner_start_ticks=self.state['owner_start_ticks'],
            lease_deadline=self.state['lease_deadline'])
        qdisc = result['ifb_qdisc']
        result['ifb_packets'] = sum(row.get('packets', 0) for row in qdisc)
        result['ifb_drops'] = sum(row.get('drops', 0) for row in qdisc)
        return result

    def set_delay_ms(self, value):
        if type(value) not in (int, float) or value not in self.allowed_rtt_ms:
            raise ValueError('RTT is not in the frozen grid')
        with self._lock():
            self._load()
            if self.state['status'] != 'active' or time.time() >= self.state['lease_deadline']:
                raise RuntimeError('X5 treatment lease expired or inactive')
            self._snapshot_unlocked(require_active=True)
            self._mutate('tc', ['qdisc', 'change', 'dev', self.ifb, 'root', 'handle', '1:',
                'netem', 'delay', '%dus' % round(value * 1000), 'limit', '10000'])
            self.state['delay_ms'] = 2 * float(value)
            self._save()
            return self._snapshot_unlocked(require_active=True)


def from_journal(path):
    value = json.loads(Path(path).read_text())
    plan = value['plan']
    instance = HolderIngressDelay(plan['local_ip'], plan['peer_ip'], plan['run_id'],
        plan['allowed_rtt_ms'], base_port=plan['base_port'], journal=path,
        lease_seconds=plan['lease_seconds'])
    if instance.plan != plan:
        raise RuntimeError('X5 journal plan/source identity changed')
    return instance


def watchdog(path):
    instance = from_journal(path)
    while True:
        value = json.loads(Path(path).read_text())
        if value['status'] in ('cleaned', 'cleanup_incomplete'):
            return 0 if value['status'] == 'cleaned' else 1
        if not owner_alive(value) or time.time() >= value['lease_deadline']:
            # Renewal or ordinary cleanup may race the unlocked observation.
            # Re-evaluate liveness and the lease while holding the same lock.
            with instance._lock():
                instance._load()
                current = instance.state
                if current['status'] in ('cleaned', 'cleanup_incomplete'):
                    return 0 if current['status'] == 'cleaned' else 1
                if not owner_alive(current) or time.time() >= current['lease_deadline']:
                    result = instance._cleanup_unlocked()
                    print(json.dumps(result), flush=True)
                    return 0 if result['restored'] and not result['errors'] else 1
        time.sleep(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watchdog', type=Path, required=True)
    sys.exit(watchdog(parser.parse_args().watchdog))
