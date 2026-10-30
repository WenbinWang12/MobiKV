"""Owned, narrowly matched ingress IFB delay; no GPU imports or root-qdisc edits.

Each endpoint applies half the requested added RTT. Both endpoints MUST be
configured before a trial. Zero delay retains precisely the same IFB path.
The owner must call cleanup in its outer finally; a detached lease watchdog
also attempts exact-owned cleanup after expiry. This is not a bandwidth cap.

Primary syntax references (iproute2 upstream): man/man8/tc-flower.8,
tc-mirred.8, tc-netem.8. Physical bytes and wire delay are not inferred.
"""
import restate_config
import argparse
import contextlib
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

PORTS = (32080, 32090, 32091, 32092, 32093, 32094, 32095, 32096, 32097, 32099)
DELAYS_MS = (0.0,)
TOOL_ROOT = restate_config.TOOL_ROOT
PROJECT_ROOT = Path(restate_config.ROOT)


def make_plan(local_ip, peer_ip, run_id, lease_seconds=900):
    return validate_plan({'schema_version': 1, 'local_ip': local_ip, 'peer_ip': peer_ip,
                          'run_id': run_id, 'interface': 'eth0', 'ports': list(PORTS),
                          'tool_root': TOOL_ROOT, 'lease_seconds': lease_seconds})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(str(tmp), str(path))


def validate_plan(plan):
    if not isinstance(plan, dict) or plan.get('schema_version') != 1:
        raise ValueError('Unsupported plan schema')
    for key in ('local_ip', 'peer_ip'):
        value = ipaddress.ip_address(plan[key])
        if value.version != 4 or value.is_loopback or value.is_unspecified:
            raise ValueError('Exact non-loopback IPv4 required')
    if plan['local_ip'] == plan['peer_ip']:
        raise ValueError('Peer must differ')
    if plan.get('interface') != 'eth0' or tuple(plan.get('ports', ())) != PORTS:
        raise ValueError('Only frozen eth0/TCP port allowlist is supported')
    if plan.get('tool_root', TOOL_ROOT) != TOOL_ROOT:
        raise ValueError('Unexpected iproute2 installation')
    run_id = plan.get('run_id', '')
    if not run_id or len(run_id) > 100 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id):
        raise ValueError('Invalid run_id')
    ttl = plan.get('lease_seconds', 900)
    if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not math.isfinite(ttl) or not 30 <= ttl <= 7200:
        raise ValueError('Lease must be finite, 30..7200 seconds')
    return plan


class Commands:
    def __init__(self):
        self.env = dict(os.environ)
        self.env['LD_LIBRARY_PATH'] = TOOL_ROOT + '/usr/lib/x86_64-linux-gnu:' + TOOL_ROOT + '/lib/x86_64-linux-gnu'

    def __call__(self, tool, args, allow_missing=False):
        executable = TOOL_ROOT + ('/bin/ip' if tool == 'ip' else '/sbin/tc')
        result = subprocess.run([executable] + list(args), env=self.env,
                                text=True, capture_output=True, timeout=15)
        if result.returncode:
            if allow_missing and ('does not exist' in result.stderr or 'Cannot find device' in result.stderr):
                return []
            raise RuntimeError('%s failed rc=%s: %s' % (tool + ' ' + ' '.join(args), result.returncode, result.stderr.strip()))
        if '-j' in args:
            return json.loads(result.stdout or '[]')
        return result.stdout.strip()


def config_only(value):
    """Discard changing counters only; retain all classifier/action identity."""
    if isinstance(value, list):
        return [config_only(x) for x in value]
    if isinstance(value, dict):
        omit = {'stats', 'stats2', 'used', 'lastused', 'installed', 'ref', 'bind',
                'refcnt', 'in_hw_count'}
        return {k: config_only(v) for k, v in value.items() if k not in omit}
    return value


def detailed_filters(filters):
    return [f for f in filters if f.get('options')]


class ScopedNetem:
    def __init__(self, plan, journal, commands=None):
        self.plan = validate_plan(dict(plan))
        self.journal = Path(journal).absolute()
        digest = hashlib.sha256(json.dumps(self.plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.plan_sha256 = digest
        self.ifb = 'rste1' + digest[:8]
        self.alias = 'restate-netem:' + self.plan['run_id'] + ':' + digest
        self.commands = commands or Commands()
        self.state = None

    @contextlib.contextmanager
    def _lock(self):
        import fcntl
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        # Interface-wide lock serializes setup/cleanup across different journals.
        lock = PROJECT_ROOT / 'results' / '.restate-eth0-netem.lock'
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def _save(self):
        write_json(self.journal, self.state)

    def _load(self):
        self.state = json.loads(self.journal.read_text())
        if self.state['plan_sha256'] != self.plan_sha256 or self.state['plan'] != self.plan or self.state['ifb'] != self.ifb:
            raise RuntimeError('Journal ownership mismatch')
        if self.state['source_sha256'] != sha(__file__):
            raise RuntimeError('Journal implementation changed')

    def _qdisc(self):
        return self.commands('tc', ['-j', 'qdisc', 'show', 'dev', 'eth0'])

    def _filters(self):
        return self.commands('tc', ['-s', '-j', 'filter', 'show', 'dev', 'eth0', 'ingress'])

    def _link(self):
        return self.commands('ip', ['-j', '-d', 'link', 'show', 'dev', self.ifb], allow_missing=True)

    def _owned_link(self, links):
        return len(links) == 1 and links[0].get('ifname') == self.ifb and links[0].get('ifalias') == self.alias and links[0].get('linkinfo', {}).get('info_kind') == 'ifb'

    def _rules(self):
        return [(52000 + i * 2 + side, i * 2 + side + 1, direction, port)
                for i, port in enumerate(PORTS) for side, direction in enumerate(('dst_port', 'src_port'))]

    def _owned_filter(self, f):
        if f.get('kind') != 'flower' or f.get('protocol') != 'ip':
            return False
        opts = f.get('options', {})
        for pref, handle, direction, port in self._rules():
            try:
                actual_handle = int(str(opts.get('handle', '')), 0)
            except ValueError:
                continue
            if f.get('pref') != pref or actual_handle != handle:
                continue
            keys = opts.get('keys', {})
            if str(keys.get('src_ip', '')).removesuffix('/32') != self.plan['peer_ip'] or str(keys.get('dst_ip', '')).removesuffix('/32') != self.plan['local_ip']:
                return False
            if keys.get('ip_proto') != 'tcp' or int(keys.get(direction, -1)) != port:
                return False
            other = 'src_port' if direction == 'dst_port' else 'dst_port'
            if other in keys:
                return False
            actions = opts.get('actions', [])
            return len(actions) == 1 and actions[0].get('kind') == 'mirred' and actions[0].get('to_dev') == self.ifb and actions[0].get('direction') == 'egress' and actions[0].get('mirred_action') == 'redirect'
        return False

    def _root_unchanged(self, qdisc):
        before = [q for q in self.state['before']['qdisc'] if q.get('root')]
        after = [q for q in qdisc if q.get('root')]
        return config_only(before) == config_only(after)

    def setup(self, start_watchdog=True):
        with self._lock():
            if self.journal.exists():
                self._load()
                if self.state.get('status') == 'active':
                    return self._snapshot_unlocked(require_active=True)
                raise RuntimeError('Journal already used; use a fresh run/journal')
            qdisc, filters, link = self._qdisc(), self._filters(), self._link()
            if filters or any(q.get('kind') in ('ingress', 'clsact') for q in qdisc) or link:
                raise RuntimeError('Preexisting ingress, filters, or IFB; refusing changes')
            if len(qdisc) != 1 or qdisc[0].get('kind') != 'noqueue' or not qdisc[0].get('root'):
                raise RuntimeError('Expected untouched noqueue root qdisc')
            addresses = self.commands('ip', ['-j', 'address', 'show', 'dev', 'eth0'])
            if not any(a.get('local') == self.plan['local_ip'] for row in addresses for a in row.get('addr_info', [])):
                raise RuntimeError('Plan local IP not on eth0')
            self.state = dict(schema_version=1, plan=self.plan, plan_sha256=self.plan_sha256,
                              source_sha256=sha(__file__), ifb=self.ifb, ifalias=self.alias,
                              status='setting_up', owner_pid=os.getpid(), hostname=socket.gethostname(),
                              created_at=time.time(), lease_deadline=time.time() + self.plan.get('lease_seconds', 900),
                              before={'qdisc': qdisc, 'filters': filters}, operations=[], delay_ms=0.0)
            self._save()
            if start_watchdog:
                log = self.journal.with_suffix('.watchdog.log')
                with log.open('ab') as output:
                    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--watchdog', str(self.journal)],
                                               stdin=subprocess.DEVNULL, stdout=output, stderr=output, start_new_session=True, close_fds=True)
                self.state['watchdog_pid'] = process.pid
                self._save()
            try:
                self._mutate('ip', ['link', 'add', 'name', self.ifb, 'type', 'ifb'])
                self._mutate('ip', ['link', 'set', 'dev', self.ifb, 'alias', self.alias])
                self._mutate('ip', ['link', 'set', 'dev', self.ifb, 'up'])
                self._mutate('tc', ['qdisc', 'add', 'dev', self.ifb, 'root', 'handle', '1:', 'netem', 'delay', '0us', 'limit', '10000'])
                self._mutate('tc', ['qdisc', 'add', 'dev', 'eth0', 'handle', 'ffff:', 'ingress'])
                for pref, handle, direction, port in self._rules():
                    self._mutate('tc', ['filter', 'add', 'dev', 'eth0', 'parent', 'ffff:', 'protocol', 'ip', 'pref', str(pref),
                                       'handle', str(handle), 'flower', 'skip_hw', 'ip_proto', 'tcp',
                                       'src_ip', self.plan['peer_ip'] + '/32', 'dst_ip', self.plan['local_ip'] + '/32',
                                       direction, str(port), 'action', 'mirred', 'egress', 'redirect', 'dev', self.ifb])
                self.state['status'] = 'active'
                self._save()
                result = self._snapshot_unlocked(require_active=True)
                return result
            except BaseException as error:
                self.state['setup_error'] = repr(error)
                self._save()
                self._cleanup_unlocked()
                raise

    def _mutate(self, tool, args):
        # Persist intent before the kernel operation, including any partial failure.
        entry = {'tool': tool, 'args': args, 'started': time.time(), 'completed': False}
        self.state['operations'].append(entry)
        self._save()
        value = self.commands(tool, args)
        entry.update(completed=True, ended=time.time())
        self._save()
        return value

    def _snapshot_unlocked(self, require_active=False):
        qdisc, filters, links = self._qdisc(), self._filters(), self._link()
        details = detailed_filters(filters)
        owned = [f for f in details if self._owned_filter(f)]
        if not self._root_unchanged(qdisc):
            raise RuntimeError('eth0 root qdisc changed unexpectedly')
        expected_rules = {(pref, handle) for pref, handle, _, _ in self._rules()}
        observed_rules = {(f['pref'], int(str(f['options']['handle']), 0)) for f in owned}
        if require_active and (len(owned) != len(expected_rules) or len(details) != len(expected_rules)
                               or observed_rules != expected_rules or not self._owned_link(links)):
            raise RuntimeError('Live scoped-rule ownership/integrity check failed')
        ifb_qdisc = self.commands('tc', ['-s', '-j', 'qdisc', 'show', 'dev', self.ifb]) if links else []
        observed_delay = None
        if require_active:
            if len(ifb_qdisc) != 1 or ifb_qdisc[0].get('kind') != 'netem' or not ifb_qdisc[0].get('root') or ifb_qdisc[0].get('handle') != '1:':
                raise RuntimeError('Live IFB root is not the owned netem qdisc')
            options = ifb_qdisc[0].get('options', {})
            if set(options) - {'limit', 'delay', 'ecn', 'gap'} or options.get('limit') != 10000 or options.get('ecn', False) or options.get('gap', 0):
                raise RuntimeError('Unexpected live netem options')
            delay = options.get('delay', {})
            if not isinstance(delay, dict) or set(delay) - {'delay', 'jitter', 'correlation'} or delay.get('jitter', 0) or delay.get('correlation', 0):
                raise RuntimeError('Unexpected live delay distribution')
            observed_delay = float(delay.get('delay', 0)) * 1000
            if not math.isfinite(observed_delay) or not math.isclose(observed_delay, self.state['delay_ms'] / 2, rel_tol=0, abs_tol=0.000001):
                raise RuntimeError('Live IFB delay differs from requested cell')
        return {'timestamp': time.time(), 'hostname': socket.gethostname(), 'ifb': self.ifb,
                'plan_sha256': self.plan_sha256, 'requested_added_rtt_ms': self.state['delay_ms'],
                'applied_one_way_delay_ms': self.state['delay_ms'] / 2,
                'eth0_root_unchanged': True, 'qdisc': qdisc, 'filters': filters,
                'ifb_link': links, 'ifb_qdisc': ifb_qdisc, 'observed_one_way_delay_ms': observed_delay,
                'owned_filter_count': len(owned), 'physical_wire_bytes': None}

    def snapshot(self):
        with self._lock():
            self._load()
            return self._snapshot_unlocked(require_active=self.state['status'] == 'active')

    def set_delay_ms(self, total_added_rtt_ms):
        if isinstance(total_added_rtt_ms, bool) or total_added_rtt_ms not in DELAYS_MS:
            raise ValueError('Only frozen 0, 2, 10 ms RTT cells supported')
        with self._lock():
            self._load()
            if self.state['status'] != 'active' or time.time() >= self.state['lease_deadline']:
                raise RuntimeError('Netem lease is not active')
            self._snapshot_unlocked(require_active=True)
            self._mutate('tc', ['qdisc', 'change', 'dev', self.ifb, 'root', 'handle', '1:', 'netem',
                                'delay', str(int(total_added_rtt_ms * 500)) + 'us', 'limit', '10000'])
            self.state['delay_ms'] = float(total_added_rtt_ms)
            self._save()
            return self._snapshot_unlocked(require_active=True)

    def renew(self):
        with self._lock():
            self._load()
            if self.state['status'] != 'active' or time.time() >= self.state['lease_deadline']:
                raise RuntimeError('Expired/inactive lease cannot be renewed')
            self.state['lease_deadline'] = time.time() + self.plan.get('lease_seconds', 900)
            self._save()
            return self.state['lease_deadline']

    def _cleanup_unlocked(self):
        errors = []
        try:
            for f in detailed_filters(self._filters()):
                if self._owned_filter(f):
                    self._mutate('tc', ['filter', 'del', 'dev', 'eth0', 'parent', 'ffff:', 'protocol', 'ip',
                                       'pref', str(f['pref']), 'handle', str(f['options']['handle']), 'flower'])
            remaining = self._filters()
            ingress = [q for q in self._qdisc() if q.get('kind') == 'ingress']
            ours_created = any(op.get('completed') and op['tool'] == 'tc' and op['args'] == ['qdisc', 'add', 'dev', 'eth0', 'handle', 'ffff:', 'ingress'] for op in self.state['operations'])
            if not remaining and len(ingress) == 1 and ingress[0].get('handle') == 'ffff:' and ours_created:
                self._mutate('tc', ['qdisc', 'del', 'dev', 'eth0', 'handle', 'ffff:', 'ingress'])
            links = self._link()
            if links:
                if not self._owned_link(links):
                    raise RuntimeError('IFB ownership lost; not deleting device')
                if any(self.ifb in json.dumps(f) for f in remaining):
                    raise RuntimeError('Foreign live filter references IFB; not deleting device')
                self._mutate('ip', ['link', 'del', 'dev', self.ifb])
        except BaseException as error:
            errors.append(repr(error))
        after = {'qdisc': self._qdisc(), 'filters': self._filters()}
        restored = config_only(after) == config_only(self.state['before']) and not self._link()
        result = {'restored': restored, 'eth0_root_unchanged': self._root_unchanged(after['qdisc']),
                  'errors': errors, 'after': after, 'timestamp': time.time(),
                  'cleanup_scope': 'owned exact filters, owned IFB, empty owned ingress only'}
        self.state['status'] = 'cleaned' if restored and not errors else 'cleanup_incomplete'
        self.state['cleanup'] = result
        self._save()
        return result

    def cleanup(self):
        with self._lock():
            self._load()
            return self._cleanup_unlocked()


def from_journal(path):
    return ScopedNetem(json.loads(Path(path).read_text())['plan'], path)


def watchdog(path):
    while True:
        value = json.loads(Path(path).read_text())
        if value['status'] in ('cleaned', 'cleanup_incomplete'):
            return 0 if value['status'] == 'cleaned' else 1
        remaining = value['lease_deadline'] - time.time()
        if remaining <= 0:
            result = from_journal(path).cleanup()
            print(json.dumps({'watchdog_cleanup': result}), flush=True)
            return 0 if result['restored'] and not result['errors'] else 1
        time.sleep(min(2, remaining))


def recv_exact(connection, length):
    chunks = bytearray()
    while len(chunks) < length:
        part = connection.recv(length - len(chunks))
        if not part:
            raise EOFError('Probe socket closed prematurely')
        chunks.extend(part)
    return bytes(chunks)


def probe_server(local_ip, peer_ip, port=32099):
    if port != 32099:
        raise ValueError('Probe port must be frozen 32099')
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((local_ip, port))
        listener.listen(1)
        listener.settimeout(120)
        connection, address = listener.accept()
        with connection:
            if address[0] != peer_ip:
                raise RuntimeError('Unexpected probe peer')
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connection.settimeout(30)
            for _ in range(110):
                connection.sendall(recv_exact(connection, 256))


def probe_client(local_ip, peer_ip, port=32099):
    if port != 32099:
        raise ValueError('Probe port must be frozen 32099')
    rows = []
    payload = bytes(range(256))
    with socket.socket() as connection:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection.settimeout(30)
        connection.bind((local_ip, 0))
        connection.connect((peer_ip, port))
        for i in range(110):
            start = time.perf_counter_ns()
            connection.sendall(payload)
            returned = recv_exact(connection, len(payload))
            end = time.perf_counter_ns()
            if returned != payload:
                raise RuntimeError('Probe payload corruption')
            if i >= 10:
                rows.append((end - start) / 1e6)
    return {'warmup_count': 10, 'sample_count': 100, 'request_bytes': 256, 'response_bytes': 256,
            'tcp_nodelay': True, 'median_rtt_ms': statistics.median(rows),
            'min_rtt_ms': min(rows), 'max_rtt_ms': max(rows), 'rtt_ms': rows}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--watchdog')
    parser.add_argument('--journal')
    parser.add_argument('--plan')
    parser.add_argument('--action', choices=['setup', 'delay', 'snapshot', 'cleanup', 'renew', 'probe-server', 'probe-client'])
    parser.add_argument('--delay-ms', type=float)
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    if args.watchdog:
        return watchdog(args.watchdog)
    if not args.journal or not args.action:
        parser.error('--journal and --action are required')
    if args.action == 'setup':
        if not args.plan:
            parser.error('setup requires --plan')
        instance = ScopedNetem(json.loads(Path(args.plan).read_text()), args.journal)
        value = instance.setup()
    else:
        instance = from_journal(args.journal)
        if args.action == 'delay':
            value = instance.set_delay_ms(args.delay_ms)
        elif args.action == 'probe-server':
            value = probe_server(instance.plan['local_ip'], instance.plan['peer_ip'])
        elif args.action == 'probe-client':
            value = probe_client(instance.plan['local_ip'], instance.plan['peer_ip'])
        else:
            value = getattr(instance, args.action)()
    if args.output:
        write_json(args.output, value)
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0 if args.action != 'cleanup' or (value['restored'] and not value['errors']) else 1


if __name__ == '__main__':
    sys.exit(main())
