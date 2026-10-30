"""E4 holder controller: real request math, one shared directional budget.

The caller binds a frozen trial registry and owns session identity/network
setup. Full reports leave the measured control stream only after disarm.
CPU dependency injection is explicitly labelled and never certifies GPU work.
"""
from __future__ import annotations
import copy
import gc
import hashlib
import json
import math
import socket
import threading
import time
import traceback
from types import SimpleNamespace
import weakref

if __package__:
    from .scheduler import E4BandwidthCoordinator, policy_config
    from . import transfer_only
else:
    from e4_scheduler import E4BandwidthCoordinator, policy_config
    import e4_transfer_only as transfer_only

SOURCE_DEPENDENCIES = ("campaign_server.py", "e3_background_load.py", "e0_cache_utils.py",
    "e4_scheduler.py", "e4_transfer_only.py", "full_control.py", "compute_queue.py")


def require(condition,message):
    if not condition:raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def _tokens(ids,vocab,name):
    require(isinstance(ids,list) and ids and all(type(x) is int and 0<=x<vocab for x in ids),"Invalid "+name)
    return ids


def production_dependencies():
    # Imported only by the actual frozen flat runtime, not by CPU components.
    from campaign_server import _Request, _SharedGPU, _listener
    from e3_background_load import CachedLocalBackground
    from e0_cache_utils import cache_to_cpu_pairs, cache_sha256
    def prepare(engine,ids):
        with engine.compute_queue.bind("e4-prefix-prepare"):
            pairs=engine.prefix(engine.torch.tensor([ids],dtype=engine.torch.int64,device=engine.device))
        snapshot=cache_to_cpu_pairs(pairs)
        checksum=cache_sha256(snapshot);snapshot=None
        return dict(pairs=pairs,length=len(ids),cache_sha256=checksum,
            cache_bytes=sum(t.numel()*t.element_size() for pair in pairs for t in pair))
    return SimpleNamespace(Request=_Request,SharedGPU=_SharedGPU,listener=_listener,
        Background=CachedLocalBackground,prepare_prefix=prepare,
        transfer_backend=lambda engine:transfer_only.CudaKVBackend(engine.torch,engine.device))


class _TransferJob:
    """Only listener/accept ownership; data protocol is the reviewed component."""
    def __init__(self,server,spec,ports):
        self.server,self.spec,self.request_id=server,spec,spec["request_id"]
        self.listeners={};self.raw={};self.endpoint=None;self.thread=None
        self.lock=threading.Lock();self.cancelled=threading.Event()
        self.report=dict(request_id=self.request_id,passed=False,status="not_started")
        try:
            for name,port in ports.items():self.listeners[name]=server.deps.listener(port,server.remaining())
        except BaseException:self.abort("listener_initialization_failed");raise
    def abort(self,reason):
        self.cancelled.set()
        with self.lock:items=list(self.listeners.values())+list(self.raw.values());endpoint=self.endpoint
        if endpoint is not None:endpoint.cancel(reason)
        for item in items:
            try:item.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            item.close()
    def run(self):
        try:
            for name in ("rpc","bulk"):
                require(not self.cancelled.is_set(),"transfer cancelled before accept")
                sock,peer=self.listeners[name].accept();sock.settimeout(self.server.remaining())
                with self.lock:
                    if self.cancelled.is_set():sock.close();raise RuntimeError("transfer cancelled during accept")
                    self.raw[name]=sock
            endpoint=transfer_only.TransferEndpoint("source",self.raw["rpc"],self.raw["bulk"],
                self.server.budget,self.spec,self.server.deps.transfer_backend(self.server.engine),
                accepted_local_s=self.server.epoch,deadline_local_s=self.server.deadline)
            with self.lock:self.endpoint=endpoint
            if self.cancelled.is_set():endpoint.cancel("transfer cancelled during endpoint setup")
            self.report=endpoint.run(prefix_pairs=self.server.prefixes[self.spec["prefix_key"]]["pairs"])
        except BaseException as error:
            self.report=copy.deepcopy(getattr(error,"report",self.report))
            self.report.update(passed=False,error=repr(error),traceback=traceback.format_exc())
        finally:
            for item in self.listeners.values():item.close()
            for item in self.raw.values():
                if item.fileno()!=-1:item.close()
            self.report["listeners_closed"]=all(s.fileno()==-1 for s in self.listeners.values())
            self.report["finished_monotonic_s"]=time.monotonic()


class E4SourceServer:
    def __init__(self,engine,control,base_port,*,trials,dependencies=None,cpu_component=False):
        require(type(base_port) is int and 1024<=base_port<=65518,"Need base+10..17 data ports")
        require(type(cpu_component) is bool,"Explicit component scope required")
        require(isinstance(trials,dict) and trials and all(k==v.get("trial_id") for k,v in trials.items()),"Frozen E4 trial registry required")
        if dependencies is not None:require(cpu_component,"Custom math dependencies are CPU components only")
        self.engine,self.control,self.base_port=engine,control,base_port
        self.deps=dependencies or production_dependencies();self.cpu_component=cpu_component
        self.registry=copy.deepcopy(trials);self.queue=engine.compute_queue
        require(self.queue is not None,"All source work requires the shared FIFO compute queue")
        if not cpu_component:require(self.queue.backend.kind=="cuda_stream_completion","Actual completion-bounded GPU queue required")
        self.gpu=self.deps.SharedGPU(engine);self.prefixes={};self.background=None
        self.budget=None;self.trial=None;self.epoch=self.deadline=None;self.active=[]
        self.current_report=self.last_trial_report=None;self.closed=False
        self._abort_lock=threading.Lock()
    def remaining(self):
        value=self.deadline-time.monotonic()
        if value<=0:raise TimeoutError("E4 source local deadline expired")
        return value
    def _idle(self):
        value=self.queue.state()
        require(value["active_gpu_operations"]==value["queued_gpu_operations"]==0,"Source FIFO queue not idle")
        return value
    def initialize(self,message):
        require(self.background is None and self.budget is None,"E4 source already initialized/armed")
        p=_tokens(message["background_prefix_ids"],self.engine.model.config.vocab_size,"background prefix")
        s=_tokens(message["background_suffix_ids"],self.engine.model.config.vocab_size,"background suffix")
        require(len(p)==1024 and len(s)==16 and message.get("generated")==32,"Frozen background P1024/S16/G32 required")
        self._idle();self.background=self.deps.Background(self.engine,self.queue,p,s,generated=32)
        self.background.prepare();calibration=self.background.calibrate()
        require(calibration["warmup_jobs"]==2 and calibration["measured_jobs"]==10 and math.isfinite(calibration["mean_service_s"]) and calibration["mean_service_s"]>0,"Actual isolated background calibration failed")
        if not self.cpu_component:require(calibration.get("GPU_service_evidence") is True,"CPU calibration cannot open GPU source")
        self.control.send_message(dict(kind="full_initialized",calibration=calibration,
            background_input_identity=dict(prefix_ids=p,suffix_ids=s,generated=32),source_queue_events=self.queue.snapshot(),CPU_component_only=self.cpu_component))
        self.queue.clear_completed_events()
    def prepare_prefix(self,message):
        require(self.budget is None and not self.active,"Prefix preparation must precede arm")
        key=message.get("prefix_key");require(isinstance(key,str) and key,"Prefix key required")
        ids=_tokens(message.get("prefix_ids"),self.engine.model.config.vocab_size,"prefix")
        require(len(ids) in (1024,4096),"E4 prefix geometry changed")
        token_sha=hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest()
        if key in self.prefixes:require(self.prefixes[key]["token_ids_sha256"]==token_sha,"Immutable prefix reused for different tokens")
        else:
            require(len(self.prefixes)<8,"Explicitly retire unused immutable prefixes first")
            self._idle();record=self.deps.prepare_prefix(self.engine,ids)
            require(record["length"]==len(ids) and len(record["pairs"])==self.engine.layers,"Actual prepared prefix geometry differs")
            record["token_ids_sha256"]=token_sha;self.prefixes[key]=record
        record=self.prefixes[key]
        self.control.send_message(dict(kind="prepared",prefix_key=key,layers=self.engine.layers,
            **{k:record[k] for k in ("cache_bytes","cache_sha256","token_ids_sha256")},prefix_length=record["length"]))
    def retire(self,message):
        require(self.budget is None and not self.active,"Cannot retire a borrowed prefix")
        key=message["prefix_key"];item=self.prefixes.pop(key)
        refs=[weakref.ref(t) for pair in item["pairs"] for t in pair];item=None;gc.collect()
        released=all(r() is None for r in refs);require(released,"Retired prefix still referenced")
        self.control.send_message(dict(kind="retired",prefix_key=key,references_released=True))
    def arm(self,message):
        require(self.background is not None and self.budget is None and not self.active,"Initialize and disarm before arm")
        trial=self.registry[message["trial_id"]]
        require(message.get("trial_sha256")==digest(trial),"Frozen E4 trial SHA differs")
        timeout=message.get("timeout_s");require(type(timeout) in (int,float) and math.isfinite(timeout) and 0<timeout<=1200,"Finite <=1200s trial bound required")
        require(type(message.get("diagnostic")) is bool and trial["added_RTT_ms"] in getattr(self.engine.e1_netem,"allowed_rtt_ms",(0,)),"Explicit diagnostic and frozen RTT required")
        offered=[*(dict(x,job_kind="decode") for x in trial["requests"]),*(dict(x,job_kind="transfer_only") for x in trial["transfer_jobs"])]
        require(1<=len(offered)<=4 and len({x["role"] for x in offered})==len(offered),"One to four unique offered roles required")
        require(trial["load_location"] in ("none","holder","destination") and type(trial["load_seed"]) is int,"Frozen background treatment required")
        for item in offered:
            prefix=self.prefixes.get(item["prefix_key"])
            require(prefix is not None and prefix["length"]==item["prefix_tokens"],"Every offered immutable prefix must be prepared before arm")
        self._idle();self.queue.clear_completed_events()
        net=self.engine.e1_netem;net.set_delay_ms(trial["added_RTT_ms"])
        network=net.snapshot()
        require(network.get("requested_added_rtt_ms")==trial["added_RTT_ms"] and network.get("observed_added_rtt_ms",network.get("observed_one_way_delay_ms"))==trial["added_RTT_ms"],"Actual network treatment differs from the frozen RTT")
        scheduler=trial["scheduler"]
        require(scheduler==policy_config(scheduler["policy"],scheduler["rate_bps"],scheduler["chunk_bytes"]),"Frozen shared scheduler differs")
        self.budget=E4BandwidthCoordinator(scheduler["rate_bps"],scheduler["policy"],chunk_bytes=scheduler["chunk_bytes"])
        self.trial=copy.deepcopy(trial);self.timeout_s=timeout;self.diagnostic=message["diagnostic"]
        self.epoch=self.deadline=None;self.batch_started=False;self.settled=False
        self.current_report=dict(trial_id=trial["trial_id"],trial_sha256=digest(trial),passed=False,
            CPU_component_only=self.cpu_component,GPU_performance_acceptance=False,E4_complete=False,
            offered_count=len(offered),outcomes=[dict(role=x["role"],kind=x["job_kind"],request_id=None,status="offered",passed=False) for x in offered],
            scheduler=copy.deepcopy(scheduler),errors=[],clock_scope="source native monotonic only; never subtract destination clocks")
        self.control.send_message(dict(kind="full_trial_armed",trial_id=trial["trial_id"],trial_sha256=digest(trial),
            scheduler=copy.deepcopy(scheduler),load_location=trial["load_location"],source_ready_monotonic_s=time.monotonic(),network_state=net.snapshot()))
        self.control.arm(self.budget)
    def start_epoch(self,message):
        require(self.budget is not None and self.epoch is None and message.get("trial_id")==self.trial["trial_id"] and message.get("trial_sha256")==digest(self.trial),"One native source epoch per frozen trial required")
        received=time.monotonic();self.epoch=time.monotonic();self.deadline=self.epoch+self.timeout_s
        background=None
        if self.trial["load_location"]=="holder":
            background=self.background.start(seed=self.trial["load_seed"],rho=.5,lead_service_multiples=0,acceptance_monotonic_s=self.epoch)
        evidence=dict(marker_received_local_s=received,source_epoch_s=self.epoch,
            ack_send_started_local_s=time.monotonic(),background_start=background,
            clock_scope="source local epoch and ACK send interval; destination brackets are separate")
        self.current_report["epoch"]=evidence
        self.control.send_message(dict(kind="e4_epoch_started",trial_id=self.trial["trial_id"],trial_sha256=digest(self.trial),**copy.deepcopy(evidence)))
        evidence["ack_sent_local_s"]=time.monotonic()
    def _specs(self,command):
        require(command.get("trial_id")==self.trial["trial_id"],"Batch trial identity differs")
        requests,transfers=command.get("requests"),command.get("transfers")
        require(isinstance(requests,list) and isinstance(transfers,list),"Explicit decode/transfer lists required")
        require(len(requests)==len(self.trial["requests"]) and len(transfers)==len(self.trial["transfer_jobs"]),"Every frozen offered job must remain in denominator")
        jobs=[];ids=set()
        for spec,expected in zip(requests,self.trial["requests"]):
            allowed={"request_id","role","prefix_key","suffix_ids","generated","action","initial_frontier"}
            require(set(spec)==allowed,"Unexpected E4 decode fields/controller injection")
            require(spec["role"]==expected["role"] and spec["prefix_key"]==expected["prefix_key"] and spec["action"]==expected["action"] and spec["action"] in ("remote","repair"),"Decode role/action/prefix differs")
            suffix=_tokens(spec["suffix_ids"],self.engine.model.config.vocab_size,"suffix")
            require(len(suffix)==expected["suffix_tokens"]==16 and spec["generated"]==expected["executor_only"]["realized_stop_tokens"]==64 and spec["initial_frontier"]==0,"Frozen suffix/stop/frontier changed")
            jobs.append(("decode",spec,expected))
        for spec,expected in zip(transfers,self.trial["transfer_jobs"]):
            transfer_only.validate_spec(spec)
            require(spec["prefix_key"]==expected["prefix_key"] and spec["prefix_tokens"]==expected["prefix_tokens"] and spec["diagnostic"] is self.diagnostic and spec["cpu_component_only"] is self.cpu_component,"Transfer prefix/scope differs")
            if not self.cpu_component:require(spec["expected_total_payload_bytes"]==expected["expected_KV_payload_bytes"] and spec["layers"]==expected["expected_complete_layers"],"Actual transfer work changed")
            jobs.append(("transfer_only",spec,expected))
        for kind,spec,expected in jobs:
            key=spec["request_id"];require(isinstance(key,str) and key and len(key)<=1024 and key not in ids,"Unique actual request IDs required");ids.add(key)
        return jobs
    def abort_active(self,reason):
        with self._abort_lock:
            for job in list(self.active):
                try:
                    # close() alone need not interrupt another thread's accept
                    # on Linux. Wake only these owned listeners/channels first.
                    for channel in [*job.listeners.values(),*getattr(job,"channels",{}).values()]:
                        try:channel.shutdown(socket.SHUT_RDWR)
                        except OSError:pass
                    job.abort(reason)
                except BaseException as error:self.current_report["errors"].append(dict(scope="abort",request_id=job.request_id,error=repr(error)))
    def _live_threads(self):
        return [thread for job in self.active for thread in
            (job.thread,getattr(job,"migration_thread",None))
            if thread is not None and thread.is_alive()]
    def batch(self,command):
        require(self.budget is not None and self.epoch is not None and not self.batch_started,"Exactly one batch after native epoch required")
        self.batch_started=True;report=self.current_report;threads=[];barrier=None
        report["batch_received_local_s"]=time.monotonic();ports={};outcomes=report["outcomes"]
        try:
            specs=self._specs(command)
            for i,(kind,spec,expected) in enumerate(specs):
                endpoint={"rpc":self.base_port+10+2*i,"bulk":self.base_port+11+2*i};ports[spec["request_id"]]=endpoint
                outcome=outcomes[i];outcome["request_id"]=spec["request_id"]
                if kind=="decode":job=self.deps.Request(self.engine,self.gpu,self.prefixes[spec["prefix_key"]],copy.deepcopy(spec),endpoint,self.budget,self.remaining(),self.deadline,"none")
                else:job=_TransferJob(self,copy.deepcopy(spec),endpoint)
                self.active.append(job)
            barrier=threading.Barrier(len(self.active)+1)
            def worker(job,outcome):
                outcome["thread_arrived_local_s"]=time.monotonic()
                try:
                    barrier.wait(timeout=self.remaining());outcome["released_local_s"]=time.monotonic();outcome["status"]="running"
                    job.run();outcome["raw"]=copy.deepcopy(job.report)
                    outcome["passed"]=job.report.get("passed") is True;outcome["status"]="passed" if outcome["passed"] else "failed"
                except BaseException as error:
                    outcome.update(passed=False,status="failed",error=repr(error),traceback=traceback.format_exc(),raw=copy.deepcopy(job.report))
                finally:outcome["thread_finished_local_s"]=time.monotonic()
            for job,outcome in zip(self.active,outcomes):
                thread=threading.Thread(target=worker,args=(job,outcome),name="E4-source-"+job.request_id,daemon=True)
                job.thread=thread;threads.append(thread);thread.start()
            report["ready_send_started_local_s"]=time.monotonic()
            self.control.send_message(dict(kind="batch_ready",trial_id=self.trial["trial_id"],ports=ports,
                decoded_integrity_mode="none",transfer_integrity_mode="sha256" if self.diagnostic else "none",source_epoch_s=self.epoch))
            report["ready_sent_local_s"]=time.monotonic()
            barrier.wait(timeout=self.remaining());report["barrier_returned_local_s"]=time.monotonic()
            while any(t.is_alive() for t in threads):
                if any(x["status"]=="failed" for x in outcomes):raise RuntimeError("Source job failed")
                for thread in threads:thread.join(min(.02,self.remaining()))
        except BaseException as error:
            report["errors"].append(dict(scope="batch",error=repr(error),traceback=traceback.format_exc()))
            if barrier is not None:barrier.abort()
            self.abort_active("E4 source batch failure/deadline")
        finally:
            deadline=time.monotonic()+30
            for thread in threads:thread.join(max(0.,deadline-time.monotonic()))
            report["alive_request_threads"]=[t.name for t in self._live_threads()]
            for outcome in outcomes:
                if outcome["status"]=="offered":outcome["status"]="not_started"
            report["passed"]=not report["errors"] and not report["alive_request_threads"] and all(x["passed"] for x in outcomes)
            report["source_jobs_finished_local_s"]=time.monotonic();report["gpu_admission"]=self.gpu.snapshot()
            if not report["alive_request_threads"]:self.active.clear()
        compact=[dict(request_id=x["request_id"],role=x["role"],kind=x["kind"],passed=x["passed"],status=x["status"],
            source_finished_local_s=x.get("thread_finished_local_s"),cleanup=x.get("raw",{}).get("cleanup"),
            remote_calls=x.get("raw",{}).get("remote_calls")) for x in outcomes]
        self.control.send_message(dict(kind="batch_done",passed=report["passed"],trial_id=self.trial["trial_id"],requests=compact,
            offered_count=len(outcomes),alive_request_threads=report["alive_request_threads"],GPU_performance_acceptance=False,
            full_reports_location="full_trial_disarmed.trial_report.outcomes[].raw"))
        report["batch_done_sent_local_s"]=time.monotonic()
        return report
    def disarm(self,message):
        require(self.budget is not None and self.batch_started and message.get("trial_id")==self.trial["trial_id"],"Disarm requires the actual finished batch identity")
        require(self.settled and not self.active,"Explicit compact service settlement must precede disarm/audit")
        accounting=self.control.disarm();background=self.current_report.get("background_report")
        queue=self.queue.snapshot();self._idle();budget=self.budget.snapshot();self.budget.close();closed=self.budget.snapshot()
        report=self.current_report
        report["background_report"]=background;report["source_queue_events"]=queue
        report["passed"]=report["passed"] and accounting["passed"] and (background is None or background.get("passed") is True)
        report["group_disarmed_local_s"]=time.monotonic()
        response=dict(kind="full_trial_disarmed",passed=report["passed"],trial_id=self.trial["trial_id"],
            control_accounting=accounting,final_coordinator=budget,coordinator_after_close=closed,
            trial_report=copy.deepcopy(report),background_report=background,source_queue_events=queue,
            network_state=self.engine.e1_netem.snapshot(),CPU_component_only=self.cpu_component,E4_complete=False)
        self.last_trial_report=copy.deepcopy(report);self.budget=None;self.trial=None;self.epoch=self.deadline=None
        self.queue.clear_completed_events();self.control.send_message(response)
        return response
    def settle(self,message):
        require(self.budget is not None and self.batch_started and not self.settled and message.get("trial_id")==self.trial["trial_id"],"Exactly one matching service-settlement marker required")
        require(not self.active and not self.current_report.get("alive_request_threads"),"Source jobs are not settled")
        report=self.current_report;started=time.monotonic();background=None
        if self.background.snapshot()["running"]:
            background=self.background.stop_and_drain(timeout=self.remaining())
        self._idle();report["background_report"]=background
        report["passed"]=report["passed"] and (background is None or background.get("passed") is True)
        evidence=dict(settle_received_local_s=started,source_work_settled_local_s=time.monotonic(),
            offered_count=report["offered_count"],completed_jobs=sum(x["passed"] for x in report["outcomes"]),
            source_threads_exited=True,background_started=background is not None,
            background_drained=background is None or background.get("passed") is True,
            source_queue_idle=True,clock_scope="holder native monotonic only")
        report["settlement"]=evidence
        self.control.send_message(dict(kind="e4_trial_settled",trial_id=self.trial["trial_id"],passed=report["passed"],**copy.deepcopy(evidence)))
        evidence["ack_sent_local_s"]=time.monotonic()
        report["service_settlement_coordinator"]=self.budget.snapshot();self.settled=True
        return evidence
    def handle(self,message):
        handlers={"prepare":self.prepare_prefix,"retire":self.retire,"full_initialize":self.initialize,
            "full_trial_arm":self.arm,"e4_epoch_start":self.start_epoch,"batch":self.batch,"e4_trial_settle":self.settle,"full_trial_disarm":self.disarm}
        function=handlers.get(message.get("kind"))
        if function is None:return False
        require(not self.closed,"Source server closed");function(message);return True
    def close(self):
        if self.closed:return self.close_report
        errors=[];self.abort_active("source session closed")
        deadline=time.monotonic()+30
        for job in self.active:
            for thread in (job.thread,getattr(job,"migration_thread",None)):
                if thread is not None:thread.join(max(0.,deadline-time.monotonic()))
        alive=[thread.name for thread in self._live_threads()]
        if self.background is not None:
            try:
                if self.background.snapshot()["running"]:
                    report=self.background.stop_and_drain(timeout=30)
                    require(report.get("passed") is True,"Source background cleanup failed")
                self.background.close()
            except BaseException as error:errors.append(repr(error))
        if self.budget is not None:
            try:self.budget.close()
            except BaseException as error:errors.append(repr(error))
        released=False
        if not alive:
            self.active.clear()
            job=None
            refs=[weakref.ref(t) for item in self.prefixes.values() for pair in item["pairs"] for t in pair]
            self.prefixes.clear();gc.collect();released=all(r() is None for r in refs)
        self.closed=True;self.close_report=dict(passed=not errors and not alive and released,
            shared_prefix_references_released=released,alive_request_threads=alive,errors=errors,
            last_trial_report=self.current_report,CPU_component_only=self.cpu_component,E4_complete=False)
        return self.close_report
    def serve(self):
        try:
            while True:
                message,tensors,_=self.control.recv_message(tensor_format="bytes")
                require(not tensors,"E4 holder control is metadata only")
                if message.get("kind")=="stop":
                    require(self.budget is None,"Disarm before stopping source")
                    closed=self.close()
                    compact={key:value for key,value in closed.items() if key!="last_trial_report"}
                    self.control.send_message(dict(kind="stopped",**compact));return closed
                require(self.handle(message),"Unknown E4 source command")
        finally:
            self.close();self.control.close()
