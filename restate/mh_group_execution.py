"""E6/E7 copy of the E4 concurrent dispatcher: up to eight jobs, including target-local jobs.

Actual concurrent destination dispatch; GPU and protocol work is supplied by jobs.

All offered threads wait at one barrier before the destination acceptance instant.
Setup after acceptance stays inside each job's measured interval. A job must
return its own protocol settlement; group completion cannot replace that time.
This module alone is neither a numerical gate nor GPU experiment evidence.
"""
from dataclasses import dataclass
import math
import threading
import time
import traceback


def require(value, message):
    if not value:
        raise ValueError(message)


@dataclass(frozen=True)
class OfferedJob:
    job_id: str
    kind: str
    # run(accepted_local_s, deadline_local_s, setup_result, cancellation_event)
    run: object
    # Interrupt this job's owned sockets only; never the shared coordinator.
    abort: object


def execute_group(jobs, setup, finalize, *, timeout_s=1200., cleanup_timeout_s=30.):
    """Return every offered outcome, including failures and incomplete cleanup.

    setup(accepted, deadline) starts the actual source batch and returns its
    endpoint mapping. finalize(deadline, failed) drains group/background state,
    records service_settled_s after the actual compact peer settlement ACK, then
    collects any separate large audit report. Audit export is ownership cost,
    not the group's service makespan.
    Neither callback is run before the accepted timestamp. A callback that can
    block must enforce its provided deadline and own only this attempt's state.
    Jobs return passed/accepted_s/settled_s and their raw protocol evidence.
    """
    jobs = tuple(jobs)
    require(jobs and len(jobs) <= 8, 'MH offers one to eight jobs')
    require(len({j.job_id for j in jobs}) == len(jobs), 'Duplicate offered job')
    for j in jobs:
        require(isinstance(j.job_id, str) and j.job_id and j.kind in ('decode', 'transfer_only', 'local'),
                'Invalid offered job identity/kind')
        require(callable(j.run) and callable(j.abort), 'Each job needs run and owned abort')
    require(all(type(v) in (float, int) and math.isfinite(v) and v > 0
                for v in (timeout_s, cleanup_timeout_s)), 'Finite positive deadlines required')
    require(callable(setup) and callable(finalize), 'Actual setup/finalize required')
    outcomes = {}; errors = []; lock = threading.Lock()
    cancelled = threading.Event(); configured = threading.Event()
    shared = {}; threads = []

    def accept():
        # Called by the last arrived participant while every job is at the barrier.
        shared['accepted_s'] = time.monotonic()
        shared['deadline_s'] = shared['accepted_s'] + timeout_s

    barrier = threading.Barrier(len(jobs) + 1, action=accept)

    def worker(job):
        record = dict(job_id=job.job_id, kind=job.kind, passed=False,
                      thread_arrived_s=time.monotonic(), status='offered')
        with lock: outcomes[job.job_id] = record
        try:
            barrier.wait(timeout=cleanup_timeout_s)
            record['dispatch_released_s'] = time.monotonic()
            record['accepted_s'] = shared['accepted_s']
            if not configured.wait(max(0., shared['deadline_s'] - time.monotonic())):
                raise TimeoutError('Group setup exceeded attempt deadline')
            if cancelled.is_set():
                raise RuntimeError('Group cancelled before protocol entry')
            record['executor_entered_s'] = time.monotonic()
            raw = job.run(shared['accepted_s'], shared['deadline_s'], shared['setup'], cancelled)
            returned = time.monotonic()
            if isinstance(raw, dict): record['raw'] = raw
            require(isinstance(raw, dict) and raw.get('accepted_s') == shared['accepted_s'],
                    'Job lost the common destination acceptance timestamp')
            settled = raw.get('settled_s')
            require(type(settled) in (int, float) and math.isfinite(settled)
                    and record['executor_entered_s'] <= settled <= returned,
                    'Job must return its own actual protocol settlement')
            record.update(raw=raw, settled_s=settled, settled_latency_s=settled-shared['accepted_s'],
                          protocol_returned_s=returned, passed=raw.get('passed') is True,
                          status='passed' if raw.get('passed') is True else 'failed')
            if not record['passed']: cancelled.set()
        except BaseException as error:
            record.update(status='failed', error=repr(error), traceback=traceback.format_exc())
            cancelled.set()
        finally:
            record['thread_finished_s'] = time.monotonic()

    def abort_owned():
        for job in jobs:
            try: job.abort('E4 group failed or deadline expired')
            except BaseException as error: errors.append(dict(scope='owned_abort', job_id=job.job_id, error=repr(error)))

    try:
        for job in jobs:
            thread = threading.Thread(target=worker, args=(job,), name='E4-'+job.job_id, daemon=True)
            threads.append(thread); thread.start()
        barrier.wait(timeout=cleanup_timeout_s)
        shared['setup_started_s'] = time.monotonic()
        shared['setup'] = setup(shared['accepted_s'], shared['deadline_s'])
        shared['setup_finished_s'] = time.monotonic()
        require(shared['setup_finished_s'] <= shared['deadline_s'], 'Setup exceeded attempt deadline')
        configured.set()
        while any(t.is_alive() for t in threads):
            if cancelled.is_set() or time.monotonic() >= shared['deadline_s']:
                cancelled.set(); abort_owned(); break
            for thread in threads:
                thread.join(min(.05, max(0., shared['deadline_s'] - time.monotonic())))
    except BaseException as error:
        errors.append(dict(scope='group_setup_or_dispatch', error=repr(error), traceback=traceback.format_exc()))
        cancelled.set(); barrier.abort(); configured.set(); abort_owned()
    finally:
        cleanup_deadline = time.monotonic() + cleanup_timeout_s
        for thread in threads: thread.join(max(0., cleanup_deadline-time.monotonic()))
        alive = [thread.name for thread in threads if thread.is_alive()]
        if alive:
            cancelled.set(); errors.append(dict(scope='owned_threads', alive=alive))
        try:
            final = finalize(cleanup_deadline, cancelled.is_set())
            require(isinstance(final, dict) and final.get('passed') is True, 'Group finalization failed')
            if not cancelled.is_set():
                stamp = final.get('service_settled_s')
                require(type(stamp) in (int, float) and math.isfinite(stamp)
                        and max(r['settled_s'] for r in outcomes.values()) <= stamp <= time.monotonic(),
                        'Actual post-drain group service settlement required')
        except BaseException as error:
            errors.append(dict(scope='group_finalize', error=repr(error))); final = dict(passed=False)
    finished = time.monotonic()
    # Preserve every offered ID even if Python could not start a worker.
    for job in jobs:
        outcomes.setdefault(job.job_id, dict(job_id=job.job_id, kind=job.kind, passed=False, status='not_started'))
    passed = not errors and not alive and all(r['passed'] for r in outcomes.values())
    return dict(passed=passed, offered_count=len(jobs), outcomes=[outcomes[j.job_id] for j in jobs],
        accepted_s=shared.get('accepted_s'), setup_started_s=shared.get('setup_started_s'),
        setup_finished_s=shared.get('setup_finished_s'), group_finished_s=finished,
        group_service_settled_s=final.get('service_settled_s'),
        group_service_settled_latency_s=None if not passed else final['service_settled_s']-shared['accepted_s'],
        group_ownership_latency_s=None if 'accepted_s' not in shared else finished-shared['accepted_s'],
        finalization=final, errors=errors, alive_threads=alive,
        clock_scope='destination native monotonic only', E4_complete=False,
        scope='Actual dispatch/own protocol timing; numerical, shared-budget and GPU evidence require independent raw audit')
