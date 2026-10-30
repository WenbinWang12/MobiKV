"""E3 local-clock actuator and shared control-byte budget on the holder.

Model/cache preparation and the common load/rate treatment precede acceptance.
Actual action-specific batch setup follows acceptance and shares the request's
outgoing budget. The caller owns the control socket and outer failure cleanup.
"""
import math
import time

from e3_background_load import CachedLocalBackground
from e3_transport_control import EpochActuator, EpochRPCService


class FullServer:
    def __init__(self, engine, control):
        self.engine, self.control = engine, control
        self.queue = engine.compute_queue
        self.background = None
        self.budget = None
        self.trial_id = None
        self.load_location = None
        self.actuator = self.epoch_service = None
        self.epoch_started = None

    def initialize(self, message):
        if self.background is not None or self.budget is not None:
            raise ValueError("Full server already initialized")
        if message.get("generated") != 32 or len(message["background_prefix_ids"]) != 1024 or len(message["background_suffix_ids"]) != 16:
            raise ValueError("Background geometry differs from frozen E3 design")
        self.background = CachedLocalBackground(self.engine, self.queue,
            message["background_prefix_ids"], message["background_suffix_ids"], generated=32)
        self.background.prepare()
        calibration = self.background.calibrate()
        self.control.send_message({"kind": "full_initialized", "calibration": calibration,
            "background_input_identity": {"prefix_ids": message["background_prefix_ids"],
                                           "suffix_ids": message["background_suffix_ids"], "generated": 32},
            "source_queue_events": self.queue.snapshot()})
        self.queue.clear_completed_events()

    def arm(self, message):
        if self.background is None or self.budget is not None:
            raise ValueError("Uninitialized server or previous trial not disarmed")
        timeout = message["timeout_s"]
        if type(timeout) not in (int,float) or not math.isfinite(timeout) or not 0 < timeout <= 1200:
            raise ValueError("Finite bounded request deadline required")
        if type(message["load_seed"]) is not int or not isinstance(message["trial_id"], str) or not message["trial_id"]:
            raise ValueError("Frozen load seed and trial identity required")
        self.engine.e1_netem.set_delay_ms(0)
        self.trial_id,self.load_seed=message['trial_id'],message['load_seed']
        plan=message['executor_plan'];load=plan['load']
        self.load_location='none' if load=='none' else load['location']
        self.timeout_s = timeout
        self.queue.clear_completed_events()
        def delay(value):
            self.engine.e1_netem.set_delay_ms(value)
            return self.engine.e1_netem.snapshot()
        def start_load():
            return self.background.start(seed=self.load_seed,rho=.5,lead_service_multiples=0,
                acceptance_monotonic_s=self.actuator.epoch)
        def stop_load():
            if not self.background.snapshot()['running']:
                return {'passed':True,'never_started':True,'snapshot':self.background.snapshot()}
            return self.background.stop_and_drain(timeout=self.timeout_s)
        self.actuator=EpochActuator(plan,endpoint_id='holder',load_location='holder',
            queue_snapshot=self.queue.state,delay_hook=delay,restore_hook=lambda:delay(0),
            start_load_hook=start_load,stop_load_hook=stop_load)
        self.budget=self.actuator.budget
        if message.get('smoke_reservation_trace') is not None:
            from e3_smoke_budget import instrument_smoke_budget
            instrument_smoke_budget(self.budget,mode=getattr(self.engine,'e3_execution_mode',None),
                fixture_id=message['smoke_reservation_trace'])
        self.epoch_service=EpochRPCService(self.actuator)
        self.engine.e3_actuator=self.actuator
        self.epoch_started=None
        self.control.send_message({"kind": "full_trial_armed", "trial_id": self.trial_id,
            "load_location": self.load_location, "background_start": None,
            "actuator_plan_sha256":self.actuator.plan_sha256,"epoch_activated":False,
            "source_ready_monotonic_s": time.monotonic(), "network_state": self.engine.e1_netem.snapshot()})
        self.control.arm(self.budget)

    def start_epoch(self,message):
        if self.actuator is None or self.epoch_started is not None:
            raise ValueError('Exactly one epoch marker per armed E3 trial')
        reply=self.epoch_service.handle(message)
        self.epoch_started=reply
        self.control.send_message(reply)

    def local_batch(self, command):
        if self.budget is None or command.get("requests") != [] or command.get("local_request") is not True:
            raise ValueError("Local batch requires an armed empty holder foreground batch")
        self.control.send_message({"kind": "batch_ready", "ports": {}, "local_request": True})
        message, tensors, _ = self.control.recv_message(tensor_format="bytes")
        if tensors or message.get("kind") != "local_done":
            raise ValueError("Local foreground completion missing")
        self.control.send_message({"kind": "batch_done", "passed": True, "local_request": True,
            "requests": [], "alive_request_threads": [], "GPU_performance_acceptance": False})

    def disarm(self, message):
        if self.budget is None or message.get("trial_id") != self.trial_id:
            raise ValueError("Disarm identity differs from armed trial")
        accounting = self.control.disarm()
        actuator_report=self.actuator.close(timeout=self.timeout_s,close_budget=False)
        if not actuator_report.get('passed'):
            raise RuntimeError('Holder actuator cleanup failed')
        background=actuator_report['load_drain']
        if background is not None and background.get("passed") is not True:
            raise RuntimeError("Holder background did not drain cleanly")
        queue = self.queue.snapshot()
        final_budget = self.budget.snapshot()
        self.budget.close()
        self.budget = None
        self.control.send_message({"kind": "full_trial_disarmed", "passed": True,
            "trial_id": self.trial_id, "control_accounting": accounting,
            "final_coordinator": final_budget,
            "actuator_report":actuator_report,"epoch_started":self.epoch_started,
            "background_report": background, "source_queue_events": queue,
            "network_state": self.engine.e1_netem.snapshot()})
        self.engine.e1_netem.set_delay_ms(0)
        self.queue.clear_completed_events()
        self.trial_id = self.load_location = None
        self.actuator=self.epoch_service=self.epoch_started=None
        self.engine.e3_actuator=None

    def handle(self, message):
        handlers = {"full_initialize": self.initialize, "full_trial_arm": self.arm,
                    "full_trial_disarm": self.disarm}
        handler = handlers.get(message.get("kind"))
        if handler is None:
            return False
        handler(message)
        return True

    def close(self):
        failure = None
        try:
            if self.actuator is not None:
                try:
                    cleaned=self.actuator.close(timeout=getattr(self,'timeout_s',120),close_budget=False)
                    if not cleaned.get('passed'):failure=RuntimeError('E3 actuator failure cleanup incomplete')
                except BaseException as error:failure=error
            if self.background is not None:
                if self.background.snapshot()["running"]:
                    try:
                        drained = self.background.stop_and_drain(timeout=getattr(self, "timeout_s", 120))
                        if not drained.get("passed"):
                            failure = RuntimeError("Background failure cleanup did not drain successfully")
                    except BaseException as error:
                        failure = error
                if not self.background.snapshot()["running"]:
                    try:
                        self.background.close()
                    except BaseException as error:
                        failure = failure or error
        finally:
            if self.budget is not None:
                self.budget.close()
                self.budget = None
        if failure is not None:
            raise failure
