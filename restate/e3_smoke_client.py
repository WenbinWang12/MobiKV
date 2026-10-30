"""Real RPC/frame lifecycle orchestration for excluded E3 functional smoke.

These scripted probes are not guarded-policy performance runs. Timing waits and
failure injection are forbidden in formal E3. Diagnostic runs inject nothing:
they execute recorded tensor routes and bind the original main evidence instead.
The executor supplies actual GPU-ready publication, drain, and control hooks.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time
try:
    from .e3_protocol import FUNCTIONAL
except ImportError:
    from e3_protocol import FUNCTIONAL


SCRIPTED=frozenset(("f01","f02","f03","f05","f07","f08"))
LIFECYCLE=frozenset(("f01","f02","f03","f07","f08"))


def require(value,message):
    if not value: raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def finite(value,name):
    require(type(value) in (int,float) and math.isfinite(value) and value>=0,name+" must be finite and nonnegative")
    return float(value)


def protocol_fixture(fixture_id):
    names={entry[0]:entry[1] for entry in FUNCTIONAL}
    require(fixture_id in names,"Unknown frozen E3 functional fixture ID")
    result=dict(id=fixture_id,name=names[fixture_id],real_executor_hook_required=True,
        test_injection_excluded_from_performance=True,CPU_fixture_cannot_pass=True)
    if fixture_id=="f08": result.update(
        failure_point="after_complete_8_token_decode_while_migration_still_live",
        original_request_must_remain_failed=True,fixture_acceptance_separate=True,
        expected_error="InjectedE3OwnedCleanupFault")
    return result


def configure_smoke_case(case,fixture,*,mode):
    """Only explicit smoke envelopes may turn a policy-label fixture into script."""
    require(mode=="e3_smoke","Functional adapter forbidden outside e3_smoke")
    require(isinstance(fixture,dict) and fixture==protocol_fixture(fixture.get("id")),"Full protocol fixture changed or incomplete")
    result=copy.deepcopy(case); key=fixture["id"]
    require(result.get("e3_execution_mode",mode)==mode,"Case/outer execution mode differs")
    expected_g={entry[0]:entry[3] for entry in FUNCTIONAL}[key]
    require(result.get("generated")==expected_g and result.get("suffix")==16,
            "Functional output/suffix geometry differs from frozen fixture")
    require(not result.get("control_script") and result.get("initial_frontier",0)==0,
            "Functional lifecycle requires its single real initial full-prefix transfer")
    result.update(e3_execution_mode=mode,e3_smoke_fixture=copy.deepcopy(fixture),
        functional_excluded=True,fixture_is_policy_performance=False)
    if key in SCRIPTED:
        result.update(action="repair",e3_online=False,fixture_execution_kind="scripted_lifecycle_not_guarded_policy")
        result.pop("checkpoints_generated",None); result.pop("checkpoint_generated",None)
    else:
        require(result.get("e3_online") is True,"Observation fixture needs actual online policy execution")
        result["fixture_execution_kind"]="online_policy_with_additional_causal_acceptance_required"
    return result


class InjectedE3OwnedCleanupFault(RuntimeError):
    pass


class ClientSmokeFixture:
    def __init__(self,case,rpc,*,control,drain,snapshot_destination,wait_frontier,
                 deadline_monotonic,epoch_monotonic,diagnostic=False,copied_evidence=None,
                 clock=time.monotonic,wait=None):
        spec=case.get("e3_smoke_fixture")
        require(case.get("e3_execution_mode")=="e3_smoke" and isinstance(spec,dict) and
                spec==protocol_fixture(spec.get("id")),"Fixture must bind an explicit frozen smoke case")
        self.key=spec["id"]; self.spec=copy.deepcopy(spec); self.case=copy.deepcopy(case)
        if self.key in SCRIPTED:
            require(case.get("e3_online") is False and case.get("action")=="repair" and
                    not case.get("checkpoints_generated") and case.get("checkpoint_generated") is None,
                    "Scripted smoke cannot pretend to execute the guarded policy")
        require(all(callable(f) for f in (control,drain,snapshot_destination,wait_frontier,clock)),"Actual lifecycle hooks required")
        self.rpc,self.control,self.drain,self.snapshot_destination,self.wait_frontier=rpc,control,drain,snapshot_destination,wait_frontier
        self.clock,self._wait=clock,wait or threading.Event().wait
        self.deadline=finite(deadline_monotonic,"trial deadline"); self.epoch=finite(epoch_monotonic,"actual local epoch")
        require(self.epoch<=clock()<self.deadline,"Functional interval must use a live actual local epoch/deadline")
        self.diagnostic=diagnostic
        require(type(diagnostic) is bool and diagnostic==(copied_evidence is not None),"Diagnostic requires explicit copied main fixture evidence")
        if diagnostic:
            require(isinstance(copied_evidence,dict) and copied_evidence.get("fixture_id")==self.key and
                    copied_evidence.get("actual_injection") is True,"Missing actual main fixture identity")
        self.copied=copy.deepcopy(copied_evidence); self.events=[]; self._started=False
        self._fault=None; self._recovered=False; self._finished=False

    def _remaining(self):
        amount=self.deadline-self.clock()
        if amount<=0: raise TimeoutError("E3 functional fixture exceeded its frozen trial deadline")
        return amount

    def _event(self,event_kind,**fields):
        row=dict(event=event_kind,destination_local_s=self.clock(),**copy.deepcopy(fields)); self.events.append(row)
        return row

    def _bounded_RPC_operation(self,operation):
        remaining=self._remaining(); sock=getattr(self.rpc,"socket",None)
        previous=sock.gettimeout() if sock is not None else None
        if sock is not None: sock.settimeout(min(remaining,previous) if previous is not None else remaining)
        try: return operation()
        finally:
            if sock is not None:
                try: sock.settimeout(previous)
                except OSError: pass  # Preserve the original transport failure.

    def _rpc(self,kind,expected,**fields):
        sent=self.clock()
        response=self._bounded_RPC_operation(lambda:self.rpc.command(kind,expected,**fields))
        received=self.clock()
        self._event("actual_RPC",kind=kind,sent_destination_s=sent,ACK_received_destination_s=received,response=response)
        return response

    def _source(self):
        reply=self._rpc("e3_smoke_snapshot","e3_smoke_snapshot_ack")
        source=reply.get("fixture")
        require(isinstance(source,dict) and source.get("fixture_id")==self.key,"Source fixture snapshot identity differs")
        return source

    def _held(self):
        while True:
            self._remaining(); source=self._source()
            if source["frame_latch_held"]:
                require(source["migration"]["state"]=="running" and source["migration"]["in_flight_layer"]==0 and
                        source["migration"]["complete_layers"]==0,"Fixture did not hold the real first begun frame")
                return source
            require(source.get("abort_reason") is None and source["migration"]["complete_layers"]==0,
                    "Required held frame was missed or aborted")

    def _control(self,command,*,applied):
        self._remaining(); sent=self.clock(); ack=self._bounded_RPC_operation(lambda:self.control(command)); received=self.clock()
        require(ack.get("kind")==command+"_ack" and ack.get("applied") is applied,
                "Functional command did not produce required actual applied/noop ACK")
        if not applied: require(bool(ack.get("noop_reason")),"Noop ACK lacks a reason")
        self._event("actual_control",command=command,sent_destination_s=sent,ACK_received_destination_s=received,
                    acknowledgement=ack,receiver_drain_inferred=False)
        return ack

    def _drain(self):
        self._remaining(); started=self.clock(); proof=self._bounded_RPC_operation(self.drain); snapshot=self.snapshot_destination()
        require(snapshot["drained"] is True and not snapshot["receiver_thread_alive"],"Actual destination migration did not drain")
        self._event("actual_receiver_drain",started_destination_s=started,evidence=proof,destination=snapshot)
        return snapshot

    def _frontier(self,expected):
        self._remaining(); started=self.clock(); self.wait_frontier(expected,self.deadline)
        snapshot=self.snapshot_destination()
        require(snapshot["physical_frontier"]==expected,"Functional GPU-ready frontier differs")
        self._event("actual_GPU_ready_frontier_hook",expected_frontier=expected,started_destination_s=started,destination=snapshot)
        return snapshot

    def _wait_until(self,elapsed):
        target=self.epoch+elapsed; self._remaining()
        require(target<self.deadline,"Rate-boundary fixture exceeds its frozen deadline")
        started=self.clock(); before=self.snapshot_destination()
        while self.clock()<target: self._wait(min(target-self.clock(),self._remaining()))
        self._event("excluded_real_clock_boundary_wait",target_elapsed_s=elapsed,target_destination_s=target,
            started_destination_s=started,finished_destination_s=self.clock(),before=before,
            after=self.snapshot_destination(),scope="smoke-only actual wall wait; not a measured latency or proof of a straddling reservation")

    def after_transfer_started(self,info):
        if self.diagnostic: return
        require(not self._started and info["start"]==0 and info["end"]==48,"Functional expects exactly one initial full-prefix transfer")
        self._started=True; self._event("actual_transfer_started",transfer=info)
        if self.key in ("f01","f07"):
            self._held(); self._control("pause",applied=True)
        elif self.key=="f02":
            self._held(); self._control("stop",applied=True)
            state=self._drain()
            require(state["physical_frontier"]==1 and state["terminal"]["next_layer"]==1 and state["terminal"]["stopped"] is True,
                    "In-flight stop did not retain exactly its one begun complete frame")
        elif self.key=="f03":
            state=self._drain(); require(state["physical_frontier"]==48,"Noop test requires the actual complete prefix")
            for command in ("pause","resume","stop"):
                ack=self._control(command,applied=False)
                require(ack["migration"]["state"]=="complete","Terminal no-op was not actually complete")
            self._event("actual_post_terminal_noops",source=self._source())
        elif self.key=="f05":
            self._wait_until(.5); self._wait_until(1.5)
        elif self.key=="f08": self._held()

    def on_token(self,step,token):
        if self.diagnostic or self.key not in ("f01","f07"): return
        d=step+1
        if d==4:
            state=self._frontier(1)
            # GPU publication may precede the sender's finish_frame bookkeeping.
            # Obtain a later real observation, never assume an atomic two-host state.
            while True:
                source=self._source(); migration=source["migration"]
                require(migration["state"]=="paused" and migration["complete_layers"] in (0,1),
                        "Pause allowed a later frame before actual resume")
                if migration["complete_layers"]==1 and migration["in_flight_layer"] is None: break
                require(migration["complete_layers"]==0 and migration["in_flight_layer"]==0,
                        "Source first-frame bookkeeping is inconsistent")
            self._event("D4_single_complete_layer",generated_count=d,token_id=token,destination=state,source=source)
        elif d==8:
            self._control("resume",applied=True)
            if self.key=="f07":
                state=self._drain(); require(state["physical_frontier"]==48,"Multi-cutover fixture lacks actual full GPU-ready prefix")
            self._event("D8_actual_resume",generated_count=d,token_id=token)

    def after_decode(self,decoded):
        if self.diagnostic: return
        require(len(decoded["token_ids"])==self.case["generated"],"Fault/acceptance before the real complete decode")
        self._event("actual_decode_complete",generated=len(decoded["token_ids"]),state_report=decoded["state_report"])
        if self.key=="f08":
            source=self._source(); destination=self.snapshot_destination()
            require(len(decoded["token_ids"])==8 and source["frame_latch_held"] and
                    source["migration"]["state"]=="running" and destination["physical_frontier"]==0,
                    "Owned failure requires eight real tokens and a still-live held unsent frame")
            self._fault="InjectedE3OwnedCleanupFault: after complete G8 decode with live migration"
            self._event("actual_application_fault_raised",error=self._fault,source=source,destination=destination)
            raise InjectedE3OwnedCleanupFault("after complete G8 decode with live migration")

    def recover_injected_fault(self,error):
        require(not self.diagnostic and self.key=="f08" and isinstance(error,InjectedE3OwnedCleanupFault) and
                self._fault is not None and not self._recovered,"Unexpected or repeated failure recovery")
        reply=self._rpc("e3_smoke_fault","e3_smoke_fault_ack",reason=self._fault)
        source=reply.get("fixture",{})
        require(source.get("fixture_id")=="f08" and source.get("released_by")=="injected_fault_stop" and
                source.get("abort_reason") is None,"Source did not release its actual begun frame through the expected fault path")
        state=self._drain()
        require(state["physical_frontier"]==1 and state["terminal"]["stopped"] is True and state["terminal"]["next_layer"]==1,
                "Expected fault failed to drain exactly its begun frame")
        self._recovered=True; self._event("expected_application_fault_recovered",original_error_preserved=self._fault,destination=state)

    def finish(self,row):
        if self.diagnostic:
            self._finished=True
            return dict(fixture_id=self.key,actual_injection=False,diagnostic_copy_only=True,
                main_evidence_sha256=digest(self.copied),main_evidence=copy.deepcopy(self.copied),
                physical_state_match_claim=False,scope="actual route/numerics are checked separately; no repeated fixture commands")
        require(self._started,"Smoke fixture never reached a real transfer")
        if self.key=="f07":
            positions={item["position"] for item in row["state_report"]["transitions"]}
            require(len(positions)>=2,"Fixture did not execute two distinct real layer cutover positions")
        if self.key=="f08": require(self._recovered and row.get("passed") is False and self._fault in row.get("error",""),
                                     "Expected application failure was not preserved in the raw request")
        self._finished=True
        return dict(fixture_id=self.key,frozen_protocol_fixture=copy.deepcopy(self.spec),actual_injection=True,
            fixture_execution_kind=self.case["fixture_execution_kind"],scripted_lifecycle=self.key in SCRIPTED,
            performance_evidence=False,policy_performance_claim=False,events=copy.deepcopy(self.events),
            expected_application_fault=self._fault,controlled_recovery_completed=self._recovered,
            all_output_tokens_completed=len(row["token_ids"])==self.case["generated"],
            additional_causal_acceptance_required=self.key in ("f04","f05","f06"),
            GPU_numerical_acceptance=False,full_fixture_acceptance=False,
            scope="actual executor hooks only; independent numerical, source, traffic, background and cleanup gates still required")
