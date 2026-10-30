"""E4 actual destination orchestration, separate from scientific acceptance.

One trial owns one outgoing scheduler and all offered socket jobs. The common
barrier precedes epoch/batch setup. Each request settles after its own cache
release; group settlement additionally drains both backgrounds. Diagnostic
logits are returned separately and are never part of measured JSON evidence.
"""
from __future__ import annotations
from collections import OrderedDict
import copy
import hashlib
import json
import math
import socket
import threading
import time
import traceback
from types import SimpleNamespace

SOURCE_DEPENDENCIES = ('campaign_client.py','e3_background_load.py','full_control.py',
    'e4_scheduler.py','e4_group_execution.py','e4_transfer_only.py','compute_queue.py')

if __package__:
    from .scheduler import E4BandwidthCoordinator, policy_config
    from .group_execution import OfferedJob, execute_group
    from . import transfer_only
else:
    from e4_scheduler import E4BandwidthCoordinator, policy_config
    from e4_group_execution import OfferedJob, execute_group
    import e4_transfer_only as transfer_only


def require(value, message):
    if not value: raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class DestinationError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


class OwnedResources:
    """Per-request interruptible socket ownership, never a global-budget close."""
    def __init__(self):
        self.lock = threading.Lock()
        self.items = []
        self.cancelled = threading.Event()
        self.deadline = None
        self.partial_report = None
        self.abort_reason = None

    def remaining(self):
        if self.cancelled.is_set(): raise RuntimeError('E4 owned request cancelled')
        left = self.deadline - time.monotonic()
        if left <= 0: raise TimeoutError('E4 request deadline expired')
        return left

    @staticmethod
    def _close(item, reason):
        if isinstance(item, transfer_only.TransferEndpoint):
            item.cancel(reason)
        else:
            try: item.shutdown(socket.SHUT_RDWR)
            except OSError: pass
            item.close()

    def register(self, item):
        with self.lock:
            if not self.cancelled.is_set():
                self.items.append(item)
                return item
        self._close(item, self.abort_reason)
        raise RuntimeError('E4 resource created after owned cancellation')

    def abort(self, reason):
        with self.lock:
            self.abort_reason = str(reason)
            self.cancelled.set()
            items = list(self.items)
        for item in items: self._close(item, self.abort_reason)

    def record_report(self, report):
        with self.lock: self.partial_report = copy.deepcopy(report)


def production_dependencies():
    from campaign_client import execute
    from e3_background_load import CachedLocalBackground
    from full_control import reconcile_control
    return SimpleNamespace(execute=execute, Background=CachedLocalBackground,
        reconcile_control=reconcile_control,
        transfer_backend=lambda engine: transfer_only.CudaKVBackend(engine.torch, engine.device))


def _totals(snapshot):
    connections = snapshot['limiter']['connections']
    return {name: sum(v[name] for v in connections.values()) for name in ('sent_bytes', 'received_bytes')}


class E4Destination:
    def __init__(self, engine, control, holder_host, *, trials, dependencies=None, cpu_component=False):
        require(type(cpu_component) is bool and (dependencies is None or cpu_component),
                'Dependency injection is only an excluded CPU component')
        require(isinstance(trials, dict) and trials and all(k == v.get('trial_id') for k, v in trials.items()),
                'Frozen trial registry required')
        self.engine, self.control, self.host = engine, control, holder_host
        self.trials = copy.deepcopy(trials)
        self.cpu_component = cpu_component
        self.deps = dependencies or production_dependencies()
        self.queue = engine.compute_queue
        require(self.queue is not None, 'Foreground and background must share a compute queue')
        if not cpu_component:
            require(self.queue.backend.kind == 'cuda_stream_completion', 'Actual GPU queue required')
            require(engine.layers == 48 and engine.kv_heads == 8 and engine.head_dim == 128 and
                    str(engine.dtype) == 'torch.bfloat16', 'Frozen main Qwen14B BF16 geometry required')
        self.background = None
        self.prepared = OrderedDict()
        self.budget = None
        self.closed = self.active = False
        self.poisoned = False
        self.sequence = 0
        self.resources = []

    def _idle(self):
        state = self.queue.state()
        require(state['active_gpu_operations'] == state['queued_gpu_operations'] == 0, 'Destination queue not idle')
        return state

    def _recv(self, kind, **identity):
        value, tensors, _ = self.control.recv_message(tensor_format='bytes')
        require(not tensors and value.get('kind') == kind and
                all(value.get(k) == v for k, v in identity.items()), 'E4 control reply identity differs: ' + kind)
        return value

    def initialize(self, background_prefix_ids, background_suffix_ids):
        require(not self.closed and self.background is None and not self.active, 'Destination already initialized')
        self._idle()
        self.background = self.deps.Background(self.engine, self.queue, background_prefix_ids, background_suffix_ids, generated=32)
        self.background.prepare()
        calibration = self.background.calibrate()
        require(calibration['warmup_jobs'] == 2 and calibration['measured_jobs'] == 10 and
                math.isfinite(calibration['mean_service_s']) and calibration['mean_service_s'] > 0,
                'Isolated real calibration required')
        if not self.cpu_component: require(calibration.get('GPU_service_evidence') is True, 'GPU calibration missing')
        self.control.send_message(dict(kind='full_initialize', background_prefix_ids=background_prefix_ids,
            background_suffix_ids=background_suffix_ids, generated=32))
        source = self._recv('full_initialized')
        require(source.get('CPU_component_only') is self.cpu_component, 'Peer execution scope differs')
        self.queue.clear_completed_events()
        return dict(destination_calibration=calibration, source=source, CPU_component_only=self.cpu_component)

    def _inputs(self, trial, values):
        offered = trial['requests'] + trial['transfer_jobs']
        require(isinstance(values, dict) and set(values) == {v['role'] for v in offered}, 'Exact offered roles and actual token arrays required')
        vocab = self.engine.model.config.vocab_size
        result = {}
        for item in offered:
            tokens = values[item['role']]
            require(isinstance(tokens, dict) and set(tokens) == {'prefix_ids', 'suffix_ids'}, 'Explicit prefix/suffix arrays required')
            for field, size in (('prefix_ids', item['prefix_tokens']), ('suffix_ids', item['suffix_tokens'])):
                require(isinstance(tokens[field], list) and len(tokens[field]) == size and
                        all(type(v) is int and 0 <= v < vocab for v in tokens[field]), 'Actual token geometry/range differs')
            result[item['role']] = copy.deepcopy(tokens)
        return result

    def _prepare(self, trial, inputs):
        reports = []
        for item in trial['requests'] + trial['transfer_jobs']:
            key = item['prefix_key']; ids = inputs[item['role']]['prefix_ids']
            token_sha = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
            if key in self.prepared:
                require(self.prepared[key]['token_ids_sha256'] == token_sha, 'Immutable prefix key reused for other tokens')
                self.prepared.move_to_end(key)
            else:
                while len(self.prepared) >= 8:
                    old = next(iter(self.prepared))
                    self.control.send_message(dict(kind='retire', prefix_key=old))
                    retired = self._recv('retired', prefix_key=old)
                    require(retired.get('references_released') is True, 'Source retired prefix remains referenced')
                    self.prepared.pop(old)
                self.control.send_message(dict(kind='prepare', prefix_key=key, prefix_ids=ids))
                prepared = self._recv('prepared', prefix_key=key)
                require(prepared.get('token_ids_sha256') == token_sha and prepared.get('prefix_length') == len(ids)
                        and prepared.get('layers') == self.engine.layers, 'Source prepared a different prefix')
                self.prepared[key] = prepared
            reports.append(copy.deepcopy(self.prepared[key]))
        return reports

    def run_trial(self, trial_id, role_inputs, *, diagnostic=False, replay_by_role=None, timeout_s=1200.):
        """Return (JSON raw report, {role: CPU logits rows}); no HF/acceptance here.

        Diagnostic requires actual main cutovers for every decode role and uses
        the same concurrent offered set. All setup after the barrier is timed.
        A fatal protocol error poisons this session; caller must close it.
        """
        require(not self.closed and not self.poisoned and not self.active and self.background is not None,
                'Initialize an idle healthy E4 session first')
        require(type(diagnostic) is bool and type(timeout_s) in (int, float) and math.isfinite(timeout_s)
                and 0 < timeout_s <= 1200, 'Explicit diagnostic and finite <=1200s bound required')
        trial = copy.deepcopy(self.trials[trial_id]); inputs = self._inputs(trial, role_inputs)
        offered=trial['requests']+trial['transfer_jobs']
        require(1<=len(offered)<=4 and len({v['role'] for v in offered})==len(offered), 'One to four unique offered jobs required')
        require(all(v['action'] in ('remote','repair') and v['suffix_tokens']==16
                    and v['executor_only']['realized_stop_tokens']==64 for v in trial['requests']),
                'E4 decode actions/geometry changed')
        require(all(v['prefix_tokens'] in (1024,4096) for v in offered) and
                all(v['action']=='transfer_only' and v['suffix_tokens']==0 for v in trial['transfer_jobs']),
                'Frozen E4 prefix/transfer geometry changed')
        require(trial['scheduler'] == policy_config(trial['scheduler']['policy'], trial['scheduler']['rate_bps'])
                and trial['added_RTT_ms'] == 0 and trial['load_location'] in ('none', 'holder', 'destination'),
                'Frozen E4 scheduler/network/load differs')
        roles = {v['role'] for v in trial['requests']}
        require((diagnostic and isinstance(replay_by_role, dict) and set(replay_by_role) == roles) or
                (not diagnostic and replay_by_role is None), 'Diagnostic must bind every main actual layer route')
        if diagnostic:
            for item in trial['requests']:
                route = replay_by_role[item['role']]
                end = item['prefix_tokens'] + item['suffix_tokens'] + item['executor_only']['realized_stop_tokens'] - 1
                require(isinstance(route, list) and len(route) == self.engine.layers and all(type(p) is int and
                        item['prefix_tokens'] <= p <= end for p in route), 'Invalid main route')
        self._idle(); prepared = self._prepare(trial, inputs)
        self.engine.e1_netem.set_delay_ms(0)
        network = self.engine.e1_netem.snapshot()
        require(network.get('requested_added_rtt_ms') == network.get('observed_one_way_delay_ms') == 0,
                'Actual destination zero-delay treatment differs')
        self.queue.clear_completed_events(); self.sequence += 1
        self.resources = []; self.active = True
        report = dict(trial_id=trial_id, trial_sha256=digest(trial), diagnostic=diagnostic, passed=False,
            CPU_component_only=self.cpu_component, GPU_performance_acceptance=False, E4_complete=False,
            physical_wire_bytes=None, prepared_prefixes=prepared, network_state=network,
            inputs=inputs, errors=[], requests={}, transfers={}, clock_scope='destination native monotonic only')
        report['offered_jobs']=[dict(role=v['role'],kind='transfer_only' if v['action']=='transfer_only' else 'decode') for v in offered]
        rows = {}; lock = threading.Lock(); setup_state = {}
        budget = E4BandwidthCoordinator(trial['scheduler']['rate_bps'], trial['scheduler']['policy'])
        self.budget = budget
        requests = []; transfers = []; jobs = []
        for item in trial['requests'] + trial['transfer_jobs']:
            role = item['role']; request = f'{trial_id}.{self.sequence}.{role}'
            owned = OwnedResources(); self.resources.append(owned)
            if item['action'] == 'transfer_only':
                kwargs = {} if not self.cpu_component else dict(layers=self.engine.layers, kv_heads=self.engine.kv_heads,
                    head_dim=self.engine.head_dim, dtype='uint8')
                spec = transfer_only.make_spec(request, item['prefix_key'], item['prefix_tokens'], diagnostic=diagnostic,
                    cpu_component=self.cpu_component, **kwargs)
                transfers.append(spec)
                def run_transfer(accepted, deadline, setup, cancelled, *, spec=spec, owned=owned, role=role):
                    owned.deadline = deadline; raw = {}; endpoint = None
                    try:
                        for name in ('rpc', 'bulk'):
                            sock = socket.create_connection((self.host, setup['ports'][spec['request_id']][name]), timeout=owned.remaining())
                            sock.settimeout(owned.remaining()); sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                            raw[name] = owned.register(sock)
                        endpoint = owned.register(transfer_only.TransferEndpoint('destination', raw['rpc'], raw['bulk'], budget,
                            spec, self.deps.transfer_backend(self.engine), accepted_local_s=accepted, deadline_local_s=deadline))
                        value = endpoint.run(); value['accepted_s'] = accepted; value['settled_s'] = value['settled_local_s']
                        with lock: report['transfers'][role] = value
                        return value
                    except BaseException as error:
                        owned.abort('transfer exception')
                        with lock: report['transfers'][role] = copy.deepcopy(getattr(error, 'report', dict(passed=False, error=repr(error))))
                        raise
                jobs.append(OfferedJob(request, 'transfer_only', run_transfer, owned.abort))
            else:
                require(item['action'] in ('remote', 'repair'), 'E4 fixed remote/repair actions only')
                tokens = inputs[role]
                spec = dict(request_id=request, role=role, prefix_key=item['prefix_key'], suffix_ids=tokens['suffix_ids'],
                    generated=item['executor_only']['realized_stop_tokens'], action=item['action'], initial_frontier=0)
                requests.append(spec)
                case = dict(prefix=item['prefix_tokens'], suffix=item['suffix_tokens'], generated=spec['generated'],
                    action=item['action'], prefix_key=item['prefix_key'], initial_frontier=0)
                def run_decode(accepted, deadline, setup, cancelled, *, case=case, spec=spec, tokens=tokens, role=role, owned=owned):
                    owned.deadline = deadline
                    try:
                        if not self.cpu_component: self.engine.torch.cuda.set_device(self.engine.device)
                        with self.queue.bind(spec['request_id'], deadline=deadline, cancel_event=cancelled):
                            value, logits, retained = self.deps.execute(self.engine, tokens['prefix_ids'] + tokens['suffix_ids'],
                                case, spec['request_id'], self.host, setup['ports'][spec['request_id']], budget,
                                diagnostic=diagnostic, replay=None if not diagnostic else replay_by_role[role],
                                accepted_s=accepted, e4_owned_resources=owned)
                        require(retained is None and value.get('own_prefix_references_released') is True and
                                value.get('own_receiver_thread_exited') is True, 'Decode retained owned prefix/receiver')
                        require(value.get('own_settled_s') is not None and value['own_settled_s'] >= value['settled_s'],
                                'Own settlement must follow protocol settlement and prefix cleanup')
                        if diagnostic: require(value['cutovers'] == replay_by_role[role], 'Diagnostic actual route changed')
                        with lock: report['requests'][role] = value; rows[role] = logits
                        return dict(passed=value.get('passed') is True, accepted_s=accepted, settled_s=value['own_settled_s'],
                            request_id=spec['request_id'], role=role, protocol_settled_s=value['settled_s'])
                    except BaseException as error:
                        value = copy.deepcopy(owned.partial_report or {})
                        value.update(passed=False, error=repr(error), traceback=traceback.format_exc())
                        with lock: report['requests'][role] = value
                        raise
                jobs.append(OfferedJob(request, 'decode', run_decode, owned.abort))

        def setup(accepted, deadline):
            self.control.raw_socket.settimeout(max(.001, deadline - time.monotonic()))
            report['accepted_s'] = accepted
            if trial['load_location'] == 'destination':
                report['background_start'] = self.background.start(seed=trial['load_seed'], rho=.5,
                    lead_service_multiples=0, acceptance_monotonic_s=accepted)
            report['epoch_send_started_s'] = time.monotonic()
            self.control.send_message(dict(kind='e4_epoch_start', trial_id=trial_id, trial_sha256=digest(trial)))
            report['source_epoch_ack'] = self._recv('e4_epoch_started', trial_id=trial_id, trial_sha256=digest(trial))
            report['epoch_ack_received_s'] = time.monotonic()
            setup_state['batch_sent'] = True
            self.control.send_message(dict(kind='batch', trial_id=trial_id, requests=requests, transfers=transfers))
            ready = self._recv('batch_ready', trial_id=trial_id)
            require(set(ready['ports']) == {j.job_id for j in jobs}, 'Source omitted/added offered jobs')
            require(ready['decoded_integrity_mode'] == 'none' and
                    ready['transfer_integrity_mode'] == ('sha256' if diagnostic else 'none'),
                    'Source decode/transfer integrity treatment differs')
            report['batch_ready'] = ready
            return ready

        def finalize(deadline, failed):
            require(setup_state.get('batch_sent'), 'No completed source batch to settle')
            self.control.raw_socket.settimeout(max(.001, deadline - time.monotonic()))
            done = self._recv('batch_done', trial_id=trial_id)
            report['batch_done_received_s'] = time.monotonic(); report['batch_done'] = done
            report['background_report'] = (self.background.stop_and_drain(timeout=max(.001, deadline-time.monotonic()))
                if self.background.snapshot()['running'] else None)
            self._idle()
            self.control.send_message(dict(kind='e4_trial_settle', trial_id=trial_id))
            settled = self._recv('e4_trial_settled', trial_id=trial_id)
            service_settled = time.monotonic()
            report['settlement_coordinator'] = budget.snapshot()
            report['source_settlement'] = settled
            self.control.send_message(dict(kind='full_trial_disarm', trial_id=trial_id))
            accounting = self.control.disarm(); setup_state['armed'] = False
            source = self._recv('full_trial_disarmed', trial_id=trial_id)
            report.update(source=source, control_accounting=accounting, final_coordinator=budget.snapshot(),
                destination_queue_events=self.queue.snapshot())
            report.update(coordinator=report['final_coordinator'],source_final_coordinator=source['final_coordinator'],
                source_queue_events=source['source_queue_events'],background_holder=source['background_report'],
                background_destination=report['background_report'],
                environment=dict(destination_network=network,source_network=source['network_state'],
                    destination_acceptance_s=report['accepted_s'],source_epoch=report['source_epoch_ack'],
                    clock_scope='Independent native clocks, no cross-host subtraction'))
            report['control_reconciliation'] = self.deps.reconcile_control(accounting, source['control_accounting'])
            left, right = _totals(report['final_coordinator']), _totals(source['final_coordinator'])
            report['actual_socket_peer_totals'] = dict(destination=left, source=right,
                passed=left['sent_bytes'] == right['received_bytes'] and left['received_bytes'] == right['sent_bytes'],
                scope='actual complete directional application socket bytes; physical wire bytes unknown')
            passed = (not failed and done.get('passed') is True and settled.get('passed') is True and
                source.get('passed') is True and report['actual_socket_peer_totals']['passed'] and
                (report['background_report'] is None or report['background_report'].get('passed') is True))
            return dict(passed=passed, service_settled_s=service_settled,
                source_settlement=settled, audit_received_s=time.monotonic())

        try:
            self.control.send_message(dict(kind='full_trial_arm', trial_id=trial_id, trial_sha256=digest(trial),
                timeout_s=timeout_s, diagnostic=diagnostic))
            report['arm'] = self._recv('full_trial_armed', trial_id=trial_id, trial_sha256=digest(trial))
            require(report['arm']['scheduler'] == trial['scheduler'], 'Source scheduler differs')
            self.control.arm(budget); setup_state['armed'] = True
            report['group'] = execute_group(jobs, setup, finalize, timeout_s=timeout_s, cleanup_timeout_s=30.)
            report['passed'] = report['group']['passed']
            if not report['passed']: raise RuntimeError('One or more actual E4 offered jobs/finalization failed')
        except BaseException as error:
            report['errors'].append(dict(error=repr(error), traceback=traceback.format_exc()))
            self.poisoned = True
            for owned in self.resources: owned.abort('E4 trial failed')
            self.control.close()
            raise DestinationError('E4 trial failed; raw evidence retained', report) from error
        finally:
            cleanup_errors=[]
            try:
                if self.background.snapshot()['running']:
                    report['exception_background_cleanup'] = self.background.stop_and_drain(timeout=30.)
                    require(report['exception_background_cleanup'].get('passed') is True,'Background exception cleanup failed')
            except BaseException as error: cleanup_errors.append(repr(error))
            try:
                report['coordinator_before_close'] = budget.snapshot(); budget.close()
                report['coordinator_after_close'] = budget.snapshot()
            except BaseException as error: cleanup_errors.append(repr(error))
            report['cleanup']=dict(errors=cleanup_errors,
                destination_job_threads_exited=not report.get('group',{}).get('alive_threads',[]),
                trial_coordinator_closed=report.get('coordinator_after_close',{}).get('closed') is True,
                destination_background_running=self.background.snapshot()['running'],
                shared_source_prefix_retained_between_trials=True,
                source_prefix_release_scope='only retire of unborrowed LRU or final source close',
                other_host_stage_resources_closed=False)
            if cleanup_errors:
                report['passed']=False;self.poisoned=True
            report['ownership_finished_s'] = time.monotonic()
            self.budget = None; self.active = False
        if not report['passed']:
            raise DestinationError('E4 cleanup failed; raw evidence retained',report)
        return report, rows

    def close(self):
        require(not self.active, 'Join the active group before session close')
        if self.closed: return self.close_report
        errors = []; source = None; background = None
        try:
            if not self.poisoned:
                self.control.send_message(dict(kind='stop'))
                source = self._recv('stopped')
                require(source.get('passed') is True, 'Source session cleanup failed')
        except BaseException as error: errors.append(repr(error))
        finally:
            self.control.close()
            if self.background is not None:
                try:
                    if self.background.snapshot()['running']: self.background.stop_and_drain(timeout=30.)
                    background = self.background.close()
                except BaseException as error: errors.append(repr(error))
            self.closed = True
        self.close_report = dict(passed=not errors and not self.poisoned, errors=errors, source=source,
            destination_background_cleanup=background, original_trial_failed=self.poisoned,
            CPU_component_only=self.cpu_component, E4_complete=False)
        return self.close_report
