"""Bounded source-frame lifecycle injections for excluded E3 smoke only.

Hooks observe the real MigrationControl and are called at actual execution
boundaries. They do not execute GPU copies, send frames, fabricate ACKs, or prove
GPU correctness. A main/performance executor must never instantiate this class.
All times use this endpoint's native monotonic clock; no cross-host subtraction.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time


def _require(condition,message):
    if not condition: raise ValueError(message)


def _number(value,name):
    _require(not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value) and value>=0,
             name+" must be finite and nonnegative")
    return float(value)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _json(value):
    return json.loads(json.dumps(value,allow_nan=False))


def fixture_spec(fixture_id):
    """Exact frozen lifecycle subset; f04/f05/f06 are separate telemetry tests."""
    release={"f01":"pause","f02":"stop","f03":None,"f07":"pause","f08":"failure_cleanup"}
    _require(fixture_id in release,"Unknown source-frame lifecycle fixture")
    result={"schema_version":1,"fixture_id":fixture_id,"required_mode":"e3_smoke",
        "spec_kind":"source_frame_lifecycle_subset_not_full_protocol_fixture",
        "sample_role":"functional_excluded","hold_first_begun_frame":fixture_id!="f03",
        "held_relative_layer":0,"release_after_actual_ACK_sent":release[fixture_id],
        "maximum_latch_wait_seconds":120,
        "scope":"deterministic test injection; no performance or GPU-proof claim"}
    return dict(result,spec_sha256=_digest(result))


def validate_fixture_spec(mode,spec):
    _require(mode=="e3_smoke","Frame lifecycle injections are forbidden outside e3_smoke")
    _require(isinstance(spec,dict) and spec==fixture_spec(spec.get("fixture_id")),"Changed/unfrozen lifecycle specification")
    return copy.deepcopy(spec)


class FixtureAborted(RuntimeError):
    """The real worker must preserve failure and perform its owned cleanup."""


class FrameLifecycleFixture:
    """A latch after begin_frame and before any GPU copy/prefix-frame send.

    Call after_begin_frame(relative_layer) immediately after a successful real
    MigrationControl.begin_frame. It waits only for a qualifying *sent* control
    ACK or failure cleanup. after_finish_frame follows the real finish_frame.
    after_control_ack_sent must run AFTER complete ACK send returns, never when
    the command merely arrived. Complete-frame pausing remains MigrationControl's
    responsibility. Root executor owns threads/sockets and GPU evidence.
    """
    def __init__(self,*,mode,spec,request_id,migration_control,deadline_monotonic,
                 transfer_index=1,transfer_start=0,clock=time.monotonic):
        self.spec=validate_fixture_spec(mode,spec)
        _require(isinstance(request_id,str) and request_id,"Real request identity required")
        _require(type(transfer_index) is int and transfer_index>0 and type(transfer_start) is int and transfer_start>=0,
                 "Invalid transfer identity")
        _require(callable(clock) and all(callable(getattr(migration_control,name,None)) for name in ("snapshot","stop")),
                 "Actual MigrationControl and local clock required")
        self.request_id,self.control,self.clock=request_id,migration_control,clock
        self.transfer_index,self.transfer_start=transfer_index,transfer_start
        self.deadline=_number(deadline_monotonic,"trial deadline")
        _require(self.deadline>_number(clock(),"source clock"),"Trial deadline already expired")
        first=self.control.snapshot()
        _require(first.get("request_id")==request_id and first.get("complete_layers")==0 and
                 first.get("in_flight_layer") is None and first.get("state")=="running",
                 "Fixture must bind a new actual running migration")
        self._lock=threading.RLock(); self._release=threading.Event(); self._entered=threading.Event()
        self._begun=[]; self._finished=[]; self._held=False; self._waiting=False; self._released_by=None
        self._abort_reason=None; self._latch_waiter_exited=True; self._terminal=None; self._drained=False
        self.frame_events=[]; self.control_receipts=[]; self.latch_events=[]; self.terminal_events=[]; self.drain_events=[]; self.failure_events=[]
        self._created=self.clock()

    def after_begin_frame(self,layer):
        _require(type(layer) is int and layer>=0,"Relative layer must be nonnegative integer")
        raw=self.control.snapshot(); at=self.clock()
        with self._lock:
            _require(layer==len(self._begun) and len(self._begun)==len(self._finished),"Begin hook missing, repeated, or overlapping")
            _require(raw.get("in_flight_layer")==layer and raw.get("complete_layers")==layer,
                     "Begin hook lacks actual active MigrationControl frame")
            self._begun.append(layer)
            self.frame_events.append({"event":"begin","source_local_s":at,"relative_layer":layer,
                "model_layer":self.transfer_start+layer,"payload_bytes":None,"migration":_json(raw),
                "hook_boundary":"after real begin_frame, before copy/send"})
            hold=self.spec["hold_first_begun_frame"] and layer==self.spec["held_relative_layer"]
            if not hold:
                if self._abort_reason is not None: raise FixtureAborted(self._abort_reason)
                return
            self._held=True; self._waiting=True; self._latch_waiter_exited=False
            self.latch_events.append({"event":"entered","source_local_s":at,"relative_layer":layer})
            self._entered.set()
            timeout=min(self.spec["maximum_latch_wait_seconds"],max(0.,self.deadline-at))
        try:
            if timeout<=0 or not self._release.wait(timeout):
                self.abort("bounded frame latch deadline expired")
                raise TimeoutError("E3 smoke frame latch deadline expired")
            with self._lock:
                if self._abort_reason is not None: raise FixtureAborted(self._abort_reason)
                _require(self._released_by in ("pause","stop","injected_fault_stop"),"Latch release lacks a qualifying actual sent ACK or recorded injected fault stop")
        finally:
            with self._lock:
                self._held=False; self._waiting=False; self._latch_waiter_exited=True
                self.latch_events.append({"event":"waiter_exited","source_local_s":self.clock(),"relative_layer":layer,
                                          "aborted":self._abort_reason is not None})

    def after_finish_frame(self,layer,payload_bytes):
        _require(type(layer) is int and type(payload_bytes) is int and payload_bytes>0,"Actual positive payload byte count required")
        raw=self.control.snapshot()
        with self._lock:
            _require(self._abort_reason is None,"Cannot turn an aborted frame into successful completion")
            _require(layer==len(self._finished) and self._begun==list(range(layer+1)),"Finish hook missing, repeated, or out of order")
            _require(raw.get("in_flight_layer") is None and raw.get("complete_layers")==layer+1,
                     "Finish hook requires actual completed MigrationControl frame")
            total=sum(row["payload_bytes"] for row in self.frame_events if row["event"]=="finish")+payload_bytes
            _require(raw.get("payload_bytes")==total,"Actual completed payload counter differs from hooks")
            self._finished.append(layer)
            self.frame_events.append({"event":"finish","source_local_s":self.clock(),"relative_layer":layer,
                "model_layer":self.transfer_start+layer,"payload_bytes":payload_bytes,"migration":_json(raw),
                "hook_boundary":"after actual full frame send and real finish_frame"})

    def after_control_ack_sent(self,command,ack):
        """Receipt of an ACK at destination is bracketed by the outer executor."""
        _require(command in ("pause","resume","stop"),"Unknown migration control")
        ack=_json(ack)
        _require(ack.get("kind")==command+"_ack" and type(ack.get("applied")) is bool and
                 ack.get("transfer_index")==self.transfer_index and ack.get("ack_scope")=="command_receipt_not_frame_drain",
                 "Control hook needs the actual complete, identity-bound ACK")
        _require(ack["applied"] or bool(ack.get("noop_reason")),"Non-applied ACK needs truthful noop reason")
        raw=ack.get("migration",{})
        allowed={"pause":("paused","complete"),"resume":("running","complete"),"stop":("stopping","stopped","complete")}
        if ack["applied"]: _require(raw.get("state") in allowed[command],"Applied control contradicts actual ACK state")
        with self._lock:
            current=self.control.snapshot(); at=self.clock()
            if self.spec["fixture_id"]=="f03":
                _require(command in ("pause","resume","stop") and not ack["applied"] and raw.get("state")=="complete"
                         and current["state"]=="complete","f03 requires actual already-complete no-op controls")
            self.control_receipts.append({"command":command,"ACK_send_finished_source_local_s":at,
                "acknowledgement":ack,"current_migration_at_hook":_json(current),
                "receiver_drained_claim":False,"hook_boundary":"after complete actual ACK send returned"})
            target=self.spec["release_after_actual_ACK_sent"]
            if command==target and ack["applied"] and self._released_by is None:
                _require(self._held and self._begun==[0] and not self._finished and
                         raw.get("in_flight_layer")==0 and current.get("in_flight_layer")==0,
                         "Qualifying ACK did not occur while the injected first frame was actually held")
                self._released_by=command
                self.latch_events.append({"event":"released_after_ACK_send","source_local_s":at,"command":command,
                                          "control_receipt_index":len(self.control_receipts)-1})
                self._release.set()

    def _terminal_identity(self,message,raw):
        if message.get("request_id") is not None: _require(message["request_id"]==self.request_id,"Cross-request terminal/drain")
        _require(type(message.get("payload_bytes")) is int and message["payload_bytes"]==raw["payload_bytes"],"Terminal/drain payload differs")
        if "complete_layers" in message:
            _require(type(message["complete_layers"]) is int and message["complete_layers"]==raw["complete_layers"],"Terminal/drain complete-layer count differs")
        else:
            _require(all(type(message.get(k)) is int for k in ("transfer_index","start","next_layer")) and
                     message["transfer_index"]==self.transfer_index and message["start"]==self.transfer_start and
                     message.get("next_layer")==self.transfer_start+raw["complete_layers"],"Terminal/drain transfer identity differs")

    def after_terminal_sent(self,message):
        message=_json(message); raw=self.control.snapshot()
        with self._lock:
            _require(self._terminal is None and raw["terminal_sent"] and raw["in_flight_layer"] is None,
                     "Terminal hook requires the unique real complete-frame terminal")
            _require(message.get("kind") in ("migration_terminal","migration_complete","migration_cancelled"),"Unknown terminal")
            if message["kind"]=="migration_terminal":
                _require(type(message.get("stopped")) is bool and message["stopped"]==(raw["state"]=="stopped"),"Terminal cancellation label differs from actual source state")
            else:
                _require(message["kind"]==("migration_complete" if raw["state"]=="complete" else "migration_cancelled") and
                         raw["state"] in ("complete","stopped"),"Terminal kind differs from actual source state")
            self._terminal_identity(message,raw)
            _require(len(self._finished)==raw["complete_layers"],"Terminal follows an unrecorded completed frame")
            self._terminal=message
            self.terminal_events.append({"source_local_s":self.clock(),"message":message,"migration":_json(raw),
                                         "hook_boundary":"after actual terminal frame send"})

    def after_receiver_drained(self,message):
        message=_json(message); raw=self.control.snapshot()
        with self._lock:
            _require(self._terminal is not None and not self._drained and raw["receiver_drained_ack"] and
                     raw["in_flight_layer"] is None,"Drain hook requires actual accepted receiver acknowledgement")
            _require(message.get("kind") in ("migration_drained","migration_drained_ack"),"Unknown drain acknowledgement")
            self._terminal_identity(message,raw)
            self._drained=True
            self.drain_events.append({"source_local_s":self.clock(),"message":message,"migration":_json(raw),
                                      "hook_boundary":"after real receiver drain acknowledgement accepted"})

    def release_after_injected_fault(self,reason):
        """Expected f08 application fault: stop, then finish the already begun frame.

        Unlike abort(), this preserves the normal complete-frame/terminal/drain
        path. The outer request must retain its original application failure.
        """
        _require(isinstance(reason,str) and reason,"Injected fault reason required")
        with self._lock:
            _require(self.spec["fixture_id"]=="f08" and self._held and self._begun==[0] and not self._finished
                     and self._abort_reason is None and self._released_by is None,"Expected f08 fault requires its actual held first frame")
            before=self.control.snapshot()
            _require(before["state"]=="running" and before["in_flight_layer"]==0,"f08 fault was not injected into a live begun frame")
            self.control.stop(); after=self.control.snapshot()
            _require(after["state"]=="stopping" and after["in_flight_layer"]==0,"Actual f08 stop did not preserve its begun frame")
            at=self.clock(); self._released_by="injected_fault_stop"
            self.failure_events.append({"source_local_s":at,"reason":reason,"expected_application_fault":True,
                "migration_before":_json(before),"migration_after":_json(after),"stop_error":None,
                "actual_stop_requested":True,"allow_begun_frame_to_finish":True,"original_request_must_remain_failed":True})
            self.latch_events.append({"event":"released_after_injected_fault_stop","source_local_s":at})
            self._release.set()
            return self.snapshot()

    def abort(self,reason):
        """Always unblock a waiter; caller still owns actual socket/thread cleanup.

        A held frame has not copied/sent yet and is NOT falsely marked finished.
        The worker receives FixtureAborted and must preserve its failure state.
        """
        _require(isinstance(reason,str) and reason,"Failure reason required")
        with self._lock:
            if self._abort_reason is not None: return self.snapshot()
            self._abort_reason=reason; before=self.control.snapshot(); error=None
            try: self.control.stop()
            except BaseException as exc: error=f"{type(exc).__name__}: {exc}"
            finally:
                self._released_by="failure_cleanup"
                self.failure_events.append({"source_local_s":self.clock(),"reason":reason,"migration_before":_json(before),
                    "migration_after":_json(self.control.snapshot()),"stop_error":error,"actual_stop_requested":error is None})
                self._release.set()
            return self.snapshot()

    def wait_until_held(self,timeout):
        """CPU synchronization helper; remote smoke should query snapshot by RPC."""
        timeout=min(_number(timeout,"wait timeout"),max(0.,self.deadline-self.clock()))
        return self._entered.wait(timeout) and self.snapshot()["frame_latch_held"]

    def snapshot(self):
        with self._lock:
            report={"schema_version":1,"fixture_id":self.spec["fixture_id"],"frozen_spec":copy.deepcopy(self.spec),
                "request_id":self.request_id,"transfer_index":self.transfer_index,"transfer_start":self.transfer_start,
                "mode":"e3_smoke","sample_role":"functional_excluded","source_created_local_s":self._created,
                "source_snapshot_local_s":self.clock(),"trial_deadline_source_local_s":self.deadline,
                "clock_scope":"native source monotonic only; no cross-host subtraction",
                "frame_latch_held":self._held,"latch_waiter_running":self._waiting,
                "latch_waiter_exited":self._latch_waiter_exited,"released_by":self._released_by,
                "abort_reason":self._abort_reason,"terminal_recorded":self._terminal is not None,
                "receiver_drain_recorded":self._drained,"migration":_json(self.control.snapshot()),
                "frame_events":copy.deepcopy(self.frame_events),"control_receipts":copy.deepcopy(self.control_receipts),
                "latch_events":copy.deepcopy(self.latch_events),"terminal_events":copy.deepcopy(self.terminal_events),
                "drain_events":copy.deepcopy(self.drain_events),"failure_events":copy.deepcopy(self.failure_events),
                "GPU_numerical_acceptance":False,"external_thread_socket_cleanup_checked":False,
                "scope":"actual lifecycle hook evidence; executor must prove real GPU execution, peer bytes and owned cleanup"}
            return report
