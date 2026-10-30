"""E3 execution plumbing: local epochs, causal telemetry, real transition hooks.

No model code, launcher, SSH or GPU operation is implemented here. Supplied hooks
must perform and audit the actual GPU publication, drain, tc and load operations.
Future plans belong only to EpochActuator; controller receives released samples.
Source monotonic times are retained as source-local evidence, never subtracted
from destination times. Application byte caps are not physical NIC/RDMA caps.
"""
from __future__ import annotations

import copy
import threading
import time
try:
    from . import controller
    from .plan import digest,require
except ImportError:
    import e3_controller as controller
    from e3_plan import digest,require

# A frozen executor supplies these unchanged modules on its runtime import path.
from e1_transport import RateLimiter
from e3_perturbation import RatePlan,VariableRateLimiter
from e4_e6_mechanisms import SharedBandwidthCoordinator

number=controller.number
OBSERVED_FIELDS=frozenset(("application_cap_bps","load_location","added_RTT_ms",
                           "successful_bytes","traffic_elapsed_s","queue_depth"))


def json_copy(value):
    # Validate that hooks did not accidentally retain tensors, sockets or NaNs.
    import json
    return json.loads(json.dumps(value,allow_nan=False))


class EpochRateLimiter(VariableRateLimiter):
    """Prewrap all channels, but reject any send before the real local epoch.

    Received start-marker bytes are counted while unactivated. The destination
    activates before sending that marker; the source activates after receiving it
    and before its first ACK send. No budget credit accumulates during pre-arm.
    """
    def __init__(self,plan,*,clock=time.monotonic,chunk_bytes=16384):
        super().__init__(plan,epoch_monotonic=0.,clock=clock,chunk_bytes=chunk_bytes)
        self._activated=False

    def activate(self):
        with self._lock:
            require(not self._activated and self._total["reserved_bytes"]==0,"epoch already used")
            self.epoch_monotonic=number(self._clock(),"local activation clock")
            self._next_ready=self.epoch_monotonic; self._activated=True
            return self.epoch_monotonic

    def reserve(self,wire_bytes,connection_id):
        require(self._activated,"application send before actual local epoch")
        return super().reserve(wire_bytes,connection_id)

    def snapshot(self):
        result=super().snapshot() if self._activated else RateLimiter.snapshot(self)
        result.update(epoch_activated=self._activated,epoch_monotonic_seconds=self.epoch_monotonic if self._activated else None,
                      rate_plan_sha256=self.plan.sha256)
        return result


class EpochActuator:
    """One endpoint's actuator-only plan and shared outgoing application budget.

    delay_hook(total_added_RTT_ms) must use the owned per-endpoint half-delay tc
    implementation. start_load_hook() starts actual same-GPU arrivals, without
    E1's 5C lead-in. stop_load_hook() drains them; restore_hook() restores only
    owned environment state. Hooks return JSON evidence, not synthetic costs.
    """
    def __init__(self,executor_plan,*,endpoint_id,load_location,queue_snapshot,
                 delay_hook,restore_hook,start_load_hook=None,stop_load_hook=None,
                 clock=time.monotonic,chunk_bytes=16384):
        require(isinstance(executor_plan,dict) and set(executor_plan)=={"rate_steps","added_RTT_steps_ms","load"},"actuator-only plan schema differs")
        require(isinstance(endpoint_id,str) and endpoint_id and load_location in ("holder","destination"),"endpoint identity/location missing")
        self._plan=json_copy(executor_plan); self.plan_sha256=digest(self._plan)
        self.endpoint_id,self.location,self.clock=endpoint_id,load_location,clock
        self.queue_snapshot=queue_snapshot; self.delay_hook=delay_hook; self.restore_hook=restore_hook
        self.start_load_hook,self.stop_load_hook=start_load_hook,stop_load_hook
        require(all(callable(f) for f in (queue_snapshot,delay_hook,restore_hook)),"actual queue/delay/restore hooks required")
        rate=RatePlan(self._plan["rate_steps"],plan_id=self.plan_sha256,seed=0)
        require(all(s.rate_bits_per_second>0 and float(s.rate_bits_per_second).is_integer() for s in rate.steps),"E3 has positive integer application caps, no zero-rate outage")
        delay=self._plan["added_RTT_steps_ms"]
        require(isinstance(delay,list) and delay and delay[0][0]==0,"delay plan starts at zero")
        require(all(len(row)==2 and number(row[0],"delay event time")>=0 and number(row[1],"delay value")>=0 for row in delay)
                and all(a[0]<b[0] for a,b in zip(delay,delay[1:])),"invalid delay schedule")
        load=self._plan["load"]
        require(load=="none" or isinstance(load,dict) and set(load)=={"location","arrival_start_seconds","rho","until"}
                and load["location"] in ("holder","destination") and load["rho"]==.5 and load["until"]=="trial_stop"
                and number(load["arrival_start_seconds"],"load start")>=0,"invalid real-load plan")
        if load!="none" and load["location"]==self.location:
            require(callable(start_load_hook) and callable(stop_load_hook),"scheduled load requires real start/drain hooks")
        self.limiter=EpochRateLimiter(rate,clock=clock,chunk_bytes=chunk_bytes)
        self.budget=SharedBandwidthCoordinator(limiter=self.limiter,priority="rpc_first")
        self.epoch=None; self.events=[]; self.errors=[]; self._cursor=0; self._thread=None
        self._stop=threading.Event(); self._lock=threading.RLock(); self._pump_lock=threading.Lock()
        self._load_started=False; self._load_attempted=False; self._delay_applied=None; self._traffic_previous=None; self._closed=False
        self._cleanup=None
        events=[(s.at_seconds,"rate_boundary",int(s.rate_bits_per_second)) for s in rate.steps]
        events += [(row[0],"delay",row[1]) for row in delay]
        if load!="none": events.append((load["arrival_start_seconds"],"load",load["location"]))
        self._events=sorted(events,key=lambda item:(item[0],item[1]))

    def activate(self,*,start_worker=True):
        require(self.epoch is None and not self._closed,"endpoint epoch is single-use")
        self.epoch=self.limiter.activate()
        try: self.pump()
        except BaseException as exc:
            self.errors.append(f"initial activation: {type(exc).__name__}: {exc}")
            self._stop.set()
            raise
        if start_worker:
            self._thread=threading.Thread(target=self._worker,name="e3-owned-actuator",daemon=True)
            self._thread.start()
        return {"endpoint_id":self.endpoint_id,"plan_sha256":self.plan_sha256,"local_epoch_s":self.epoch,
                "local_initialization_finished_s":self.clock(),"source_clock_domain_only":True}

    def _worker(self):
        try:
            while not self._stop.is_set():
                self.pump()
                with self._lock:
                    if self._cursor==len(self._events): return
                    due=self.epoch+self._events[self._cursor][0]
                self._stop.wait(max(0.,min(.05,due-self.clock())))
        except BaseException as exc:
            with self._lock: self.errors.append(f"{type(exc).__name__}: {exc}")
            self._stop.set()

    def pump(self):
        """Execute due real hooks; production worker calls independently of decode.

        start_worker=False + a fixture clock is for deterministic CPU testing.
        It is never an experiment timing substitute.
        """
        require(self.epoch is not None,"actuator not activated")
        with self._pump_lock:
            while not self._stop.is_set():
                with self._lock:
                    if self._cursor==len(self._events): return
                    elapsed,kind,value=self._events[self._cursor]
                    if self.clock()<self.epoch+elapsed: return
                started=self.clock(); evidence=None
                if kind=="delay":
                    evidence=json_copy(self.delay_hook(value))
                    with self._lock: self._delay_applied=value
                elif kind=="load" and value==self.location:
                    # A hook may create work and then raise: cleanup still owns it.
                    self._load_attempted=True
                    evidence=json_copy(self.start_load_hook())
                    with self._lock: self._load_started=True
                finished=self.clock()
                with self._lock:
                    self.events.append({"kind":kind,"value":value,"scheduled_local_s":self.epoch+elapsed,
                        "application_started_local_s":started,"application_finished_local_s":finished,
                        "applied_on_this_endpoint":kind!="load" or value==self.location,"hook_evidence":evidence,
                        "scope":"reservation integration, observed boundary bookkeeping" if kind=="rate_boundary" else "actual actuator callback"})
                    self._cursor+=1

    def current_observation(self):
        require(self.epoch is not None and not self.errors and not self._closed,"inactive/failed actuator cannot report telemetry")
        started=self.clock(); total=self.budget.snapshot()["limiter"]["total"]["sent_bytes"]
        queue=json_copy(self.queue_snapshot())
        depth=queue["active_gpu_operations"]+queue["queued_gpu_operations"]
        require(type(depth) is int and depth>=0,"actual queue counters invalid")
        with self._lock:
            current_delay=self._delay_applied
            require(current_delay is not None,"initial owned delay has not actually applied")
            active_load=self.location if self._load_started else "none"
            previous=self._traffic_previous; self._traffic_previous=(started,total)
        interval=None if previous is None or started<=previous[0] else started-previous[0]
        amount=None if interval is None else total-previous[1]
        require(amount is None or amount>=0,"successful application bytes regressed")
        observed={"application_cap_bps":int(self.limiter.plan.rate_at(max(0.,started-self.epoch))),
            "load_location":active_load,"added_RTT_ms":current_delay,"successful_bytes":amount,
            "traffic_elapsed_s":interval,"queue_depth":depth}
        return {"observed":observed,"endpoint_id":self.endpoint_id,"plan_sha256":self.plan_sha256,
            "source_epoch_local_s":self.epoch,"source_sample_started_local_s":started,
            "source_sample_finished_local_s":self.clock(),"actual_successful_sent_bytes":total,
            "source_queue":queue,"clock_scope":"all source timestamps source-local; no cross-host subtraction"}

    def close(self,*,timeout=10,close_budget=True):
        """Call after request settlement/disarming any borrowed control socket."""
        number(timeout,"cleanup timeout")
        if self._cleanup is not None: return json_copy(self._cleanup)
        close_started=self.clock()
        self._stop.set()
        if self._thread is not None: self._thread.join(timeout)
        require(self._thread is None or not self._thread.is_alive(),"actuator callback/thread did not stop; cleanup incomplete")
        cleanup={"actuator_thread_exited":True,"load_drain":None,"environment_restore":None,"errors":[]}
        try:
            if self._load_attempted: cleanup["load_drain"]=json_copy(self.stop_load_hook())
        except BaseException as exc: cleanup["errors"].append("load drain: "+repr(exc))
        try: cleanup["environment_restore"]=json_copy(self.restore_hook())
        except BaseException as exc: cleanup["errors"].append("restore: "+repr(exc))
        budget_closed=False
        try:
            if close_budget:
                self.budget.close(); budget_closed=True
        except BaseException as exc: cleanup["errors"].append("budget close: "+repr(exc))
        self._closed=True
        cleanup.update(budget_closed=budget_closed,actuator_errors=list(self.errors),events=json_copy(self.events))
        cleanup.update(close_started_local_s=close_started,closed_local_s=self.clock(),
            local_epoch_s=self.epoch,plan_sha256=self.plan_sha256,endpoint_id=self.endpoint_id,
            clock_scope="native endpoint monotonic clock; no cross-host subtraction",
            unapplied_events=[{"elapsed_s":elapsed,"kind":kind,"value":value,
                "scheduled_local_s":None if self.epoch is None else self.epoch+elapsed,
                "due_at_close_start":self.epoch is not None and self.epoch+elapsed<=close_started}
                for elapsed,kind,value in self._events[self._cursor:]])
        cleanup["due_events_unapplied_at_close_start"]=sum(row["due_at_close_start"] for row in cleanup["unapplied_events"])
        cleanup["passed"]=not cleanup["errors"] and not self.errors
        self._cleanup=json_copy(cleanup)
        return cleanup


class EpochRPCService:
    """Server dispatch for already identity-bound, framed request channels."""
    def __init__(self,actuator): self.actuator=actuator; self.last_sequence=-1

    def handle(self,message):
        received=self.actuator.clock(); kind=message.get("kind")
        if kind=="e3_epoch_start":
            require(message.get("plan_sha256")==self.actuator.plan_sha256,"peer actuator plan changed")
            started=self.actuator.activate()
            return {"kind":"e3_epoch_started","epoch":started,"source_marker_received_local_s":received}
        require(kind=="e3_observe" and type(message.get("sequence")) is int and message["sequence"]>self.last_sequence,"invalid/repeated telemetry request")
        self.last_sequence=message["sequence"]
        evidence=self.actuator.current_observation()
        return {"kind":"e3_observation","sequence":self.last_sequence,"evidence":evidence,"evidence_sha256":digest(evidence)}


def start_epoch_pair(destination,rpc,*,start_worker=True):
    """Destination starts before its paced marker; source starts on receipt."""
    local=destination.activate(start_worker=start_worker)
    sent=destination.clock()
    reply=rpc.command("e3_epoch_start","e3_epoch_started",plan_sha256=destination.plan_sha256)
    received=destination.clock(); source=reply["epoch"]
    require(source["plan_sha256"]==destination.plan_sha256 and source["endpoint_id"]!=destination.endpoint_id,"foreign epoch reply")
    return {"destination":local,"source":json_copy(source),"marker_sent_destination_s":sent,
            "ACK_received_destination_s":received,"round_trip_bracket_s":received-sent,
            "source_marker_received_local_s":reply["source_marker_received_local_s"],
            "clocks_synchronized":False,"source_event_destination_time_interval_claim":False}


class TelemetryClient:
    def __init__(self,*,accepted_destination_s,withholding_delay,destination_observation=None,clock=time.monotonic):
        self.accepted=number(accepted_destination_s,"acceptance"); require(callable(withholding_delay),"executor withholding recipe required")
        self._delay=withholding_delay; self.clock=clock; self.inbox=controller.ObservationInbox()
        require(destination_observation is None or callable(destination_observation),"destination observation must be actual callback")
        self._destination_observation=destination_observation
        self.sequence=0; self.history=[]
        self._returned_observation=None; self._returned_provenance=None

    def observe(self,rpc):
        """Returns only a currently released allowlisted observation to policy."""
        before=self.inbox.release(self.clock()); sent=self.clock()
        require(sent>=self.accepted,"observation predates acceptance")
        sequence=self.sequence; self.sequence+=1
        reply=rpc.command("e3_observe","e3_observation",sequence=sequence)
        received=self.clock()
        require(reply.get("sequence")==sequence,"cross-observation reply")
        evidence=reply["evidence"]
        require(reply["evidence_sha256"]==digest(evidence) and set(evidence["observed"])==OBSERVED_FIELDS,"telemetry evidence/payload changed")
        observed=json_copy(evidence["observed"]); local=None
        if self._destination_observation is not None:
            local=json_copy(self._destination_observation())
            require(set(local["observed"])==OBSERVED_FIELDS,"destination telemetry schema changed")
            source_load,local_load=observed["load_location"],local["observed"]["load_location"]
            require(source_load in ("none","holder") and local_load in ("none","destination") and
                    (source_load=="none" or local_load=="none"),"unexpected simultaneous/opposite-endpoint load")
            observed["load_location"]=local_load if local_load!="none" else source_load
            observed["queue_depth"]+=local["observed"]["queue_depth"]
            # Each owned endpoint actuator applies half of its requested total.
            # This sums observed settings, not a measured physical RTT.
            observed["added_RTT_ms"]=(observed["added_RTT_ms"]+local["observed"]["added_RTT_ms"])/2
        withholding=number(self._delay(sent-self.accepted),"executor withholding delay")
        availability_base=max(received,self.clock())
        sample={"sequence":sequence,"request_sent_s":sent,"response_received_s":received,
                "available_s":availability_base+withholding,**observed}
        self.inbox.enqueue(sample); released=self.inbox.release(self.clock())
        self.history.append({"sequence":sequence,"request_sent_destination_s":sent,"reply_received_destination_s":received,
            "available_destination_s":sample["available_s"],"withheld_seconds":withholding,"source_evidence":json_copy(evidence),
            "sample_sha256":digest(sample),"sample":json_copy(sample),"availability_base_destination_s":availability_base,
            "source_evidence_sha256":reply["evidence_sha256"],"previously_released_sequence":None if before is None else before["sequence"],
            "selected_sequence":None if released is None else released["sequence"],"no_cross_host_clock_subtraction":True,
            "destination_evidence":local,"aggregation_scope":"asynchronous actual settings; source outgoing cap/traffic; summed queue counts; no atomic state or measured RTT claim"})
        # Freeze this observation opportunity, including the provenance of an
        # older selected sample. Looking up provenance must not poll/release a
        # newly eligible withheld reply after observe() has returned.
        self._returned_observation=json_copy(released)
        self._returned_provenance=(None if released is None else json_copy(next(
            row for row in self.history if row['sequence']==released['sequence'])))
        return released

    def released_evidence(self,sample):
        """Executor-only provenance of an already released selected sample.

        In particular, callers must not construct policy migration metadata from
        the newer raw reply when that observation is still being withheld.
        """
        require(sample==self._returned_observation,"evidence is not for the frozen returned observation")
        if sample is None: return None
        controller.validate_sample(sample)
        record=self._returned_provenance
        require(record is not None and record["sample_sha256"]==digest(sample),"released observation provenance changed")
        return json_copy(record)


class MigrationCommandService:
    """Adapt the existing complete-frame MigrationControl; never pause a socket.

    Pausing SharedBandwidthCoordinator mid-frame instead would block the frame
    required to drain, so migration controls act only on begin_frame().
    """
    def __init__(self,migration_control,*,transfer_index,clock=time.monotonic):
        self.control=migration_control; self.transfer_index=transfer_index; self.clock=clock

    def command(self,command):
        require(command in ("pause","resume","stop"),"unsupported migration command")
        received=self.clock(); before=self.control.snapshot(); mode=before["state"]
        require(mode!="failed","failed migration needs cleanup")
        duplicate=(command=="pause" and mode=="paused") or (command=="resume" and mode=="running")
        terminal=mode in ("complete","stopping","stopped")
        applied=not duplicate and not terminal
        started=self.clock()
        if applied:
            try: getattr(self.control,command)()
            except RuntimeError:
                if self.control.snapshot()["state"] not in ("complete","stopping","stopped"): raise
                applied=False
        finished=self.clock(); after=self.control.snapshot()
        return {"kind":command+"_ack","transfer_index":self.transfer_index,"applied":applied,
            "noop_reason":None if applied else "already_"+mode if duplicate else "migration_"+after["state"],
            "ack_scope":"command_receipt_not_frame_drain","migration":after,"before":before,
            "source_received_local_s":received,"source_application_started_local_s":started,
            "source_application_finished_local_s":finished,"source_snapshot_local_s":self.clock()}


class TransitionAdapter:
    """Actual-operation adapter; hooks own model/transport and return evidence.

    snapshot() returns migration_mode,committed_layers,in_flight_layer,
    receiver_drained,transfer_index,transfer_start; in-flight is source metadata.
    control(command) must send/receive the actual framed command and return its
    ACK. drain() must join complete-frame receiver work and verify terminal+GPU
    publication before returning. fetch_remaining() may start only an explicitly
    allowed live segment. recompute_publish() must execute full P and retain delta.
    delta_signature() returns tensor identity/storage/shape/position only, no KV
    content copying on the main path. No hook may fabricate a completion boolean.
    """
    def __init__(self,*,snapshot,control,drain,fetch_remaining,recompute_publish,delta_signature,clock=time.monotonic):
        hooks=(snapshot,control,drain,fetch_remaining,recompute_publish,delta_signature)
        require(all(callable(h) for h in hooks),"real transition hooks required")
        self.snapshot,self.control,self.drain,self.fetch_remaining,self.recompute_publish,self.delta_signature=hooks
        self.clock=clock; self.evidence=[]; self.stopped_irreversibly=False

    def execute(self,proposal,local_state):
        controller.validate_local(local_state)
        require(proposal.get("proposal_sha256")==digest({k:v for k,v in proposal.items() if k!="proposal_sha256"}) and
                proposal["state_sha256"]==digest(local_state),"proposal/local-state SHA mismatch")
        command=proposal["command"]
        require(command in controller.legal_commands(local_state),"illegal proposed continuation")
        before=json_copy(self.snapshot()); signature=copy.deepcopy(self.delta_signature()); operations=[]
        started=self.clock(); receipt=None
        if self.stopped_irreversibly: require(command not in ("resume","complete_fetch_then_local","pause"),"stop cannot be reversed")
        def control(kind):
            sent=self.clock(); value=json_copy(self.control(kind)); received=self.clock()
            require(value.get("kind")==kind+"_ack" and type(value.get("applied")) is bool and
                    value.get("transfer_index")==before["transfer_index"] and value.get("ack_scope")=="command_receipt_not_frame_drain","foreign migration ACK")
            require(value["applied"] or bool(value.get("noop_reason")),"control noop has no reason")
            operations.append({"kind":"source_control","command":kind,"sent_destination_s":sent,
                "ACK_received_destination_s":received,"receipt":value,"receipt_implies_receiver_drain":False})
            return value
        def drain():
            began=self.clock(); proof=json_copy(self.drain()); ended=self.clock()
            require(self.snapshot()["receiver_drained"] is True,"actual receiver did not drain")
            operations.append({"kind":"receiver_complete_frame_and_GPU_drain","started_destination_s":began,
                "finished_destination_s":ended,"evidence":proof})
        if command in ("pause","resume"):
            receipt=control(command)
        elif command=="complete_fetch_then_local":
            require(before["migration_mode"] not in ("stopping","stopped"),"fetch cannot restart stopped segment")
            if before["migration_mode"]=="paused":
                receipt=control("resume")
                require(receipt["migration"]["state"] in ("running","complete"),"paused fetch did not resume")
            if not self.snapshot()["receiver_drained"]: drain()
            if self.snapshot()["committed_layers"]<48:
                operations.append({"kind":"actual_fetch_remaining","evidence":json_copy(self.fetch_remaining())})
            require(self.snapshot()["receiver_drained"] and self.snapshot()["committed_layers"]==48,"fetch lacks complete GPU-ready prefix/drain")
        elif command in ("stop_migration_and_remote","recompute_full_prefix_keep_delta"):
            if not before["receiver_drained"]:
                receipt=control("stop"); drain()
            self.stopped_irreversibly=True
            if command=="recompute_full_prefix_keep_delta":
                operations.append({"kind":"actual_full_prefix_recompute_keep_delta","evidence":json_copy(self.recompute_publish())})
                require(self.snapshot()["committed_layers"]==48,"full-prefix recompute failed to publish complete prefix")
        elif command=="continue_local": require(before["committed_layers"]==48,"local continuation lacks complete actual prefix")
        elif command!="retain": raise ValueError("unsupported continuation")
        after=json_copy(self.snapshot())
        require(signature==self.delta_signature(),"transition changed request-owned delta identity/shape/position")
        mode=receipt["migration"]["state"] if command in ("pause","resume") else after["migration_mode"]
        inflight=receipt["migration"]["in_flight_layer"] if command in ("pause","resume") else after["in_flight_layer"]
        if inflight is not None and command in ("pause","resume"): inflight+=before["transfer_start"]
        if after["receiver_drained"]:
            require(after["in_flight_layer"] is None,"receiver drain hook retained an active frame")
            # A completed receiver drain supersedes an earlier command snapshot.
            # Retain the original stale source marker unmodified in operations.
            inflight=None
        applied=receipt["applied"] if command in ("pause","resume") else command in ("complete_fetch_then_local","recompute_full_prefix_keep_delta") or (
            command=="stop_migration_and_remote" and receipt is not None and receipt["applied"])
        scope="command_receipt_not_frame_drain" if command in ("pause","resume") else (
            "local_no_control_command" if command in ("retain","continue_local") else "continuation_installed_after_receiver_drain")
        finished=self.clock()
        evidence={"proposal_sha256":proposal["proposal_sha256"],"before":before,"after":after,"operations":operations,
            "transition_started_destination_s":started,"transition_finished_destination_s":finished,
            "delta_signature_before":json_copy(signature),"delta_identity_storage_shape_position_preserved":True,
            "delta_signature_after":json_copy(self.delta_signature()),
            "source_inflight_is_asynchronous_metadata":True,"cross_host_atomic_snapshot_claim":False}
        ack={"proposal_sha256":proposal["proposal_sha256"],"command":command,"applied":bool(applied),
            "continuation_installed":True if command not in ("pause","resume") else receipt["applied"],
            "observed_migration_mode":mode,"committed_layers":after["committed_layers"],"in_flight_layer":inflight,
            "receiver_drained":after["receiver_drained"],"evidence_sha256":digest(evidence),
            "acknowledged_at_s":finished,"noop_reason":None if applied else receipt["noop_reason"] if receipt else "no_source_command_required",
            "ack_scope":scope}
        require(set(ack)==controller.ACK_FIELDS,"adapter ACK schema drift")
        self.evidence.append(evidence)
        return ack,copy.deepcopy(evidence)
