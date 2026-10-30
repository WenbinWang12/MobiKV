"""Observe actual shared-limiter reservations only in excluded E3 f05 smoke.

The original reserve/snapshot implementations remain the sole pacing engine.
This records native local slot boundaries; it creates neither bytes nor delays.
Actual sends and peer receives must still be independently reconciled.
"""
import copy
import threading


def instrument_smoke_budget(budget, *, mode, fixture_id):
    if mode != 'e3_smoke' or fixture_id != 'f05':
        raise ValueError('Reservation instrumentation belongs only to excluded E3 f05')
    limiter=budget.limiter
    if getattr(limiter,'_e3_smoke_reservations',None) is not None:
        raise ValueError('Cannot instrument a shared limiter twice')
    original_reserve,original_snapshot=limiter.reserve,limiter.snapshot
    events=[];lock=threading.Lock()
    limiter._e3_smoke_reservations=events

    def reserve(byte_count,connection_id):
        slot=original_reserve(byte_count,connection_id)
        with lock:
            events.append(dict(connection_id=slot.connection_id,bytes=slot.wire_bytes,
                start_at_s=slot.start_at,ready_at_s=slot.ready_at,
                recorded_local_s=limiter._clock(),native_epoch_s=limiter.epoch_monotonic))
        return slot

    def snapshot():
        result=original_snapshot()
        with lock:
            result['smoke_reservation_trace']=dict(fixture_id='f05',mode=mode,
                events=copy.deepcopy(events),scope='actual pacing reservations; successful send counters required separately',
                performance_evidence=False,physical_wire_bytes=None)
        return result

    limiter.reserve=reserve
    limiter.snapshot=snapshot
    return limiter
