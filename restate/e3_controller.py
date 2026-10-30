"""Causal CPU E3 proposals and truthful ACK state; never executes GPU/network.

Only the actuator may receive future traces. CPU artifact hashes do not replace
the independent raw-evidence and full-stage gates required by an executor.
"""
from __future__ import annotations

import copy
import math
try:
    from .e3_plan import FIXED_ACTIONS, METHODS, ONLINE, digest, require
except ImportError:
    from e3_plan import FIXED_ACTIONS, METHODS, ONLINE, digest, require
try:
    from restate_direct.e2_full import controller as e2
except ImportError:
    import e2_controller as e2

LOCAL_FIELDS=frozenset(e2.STATE_FIELDS-{"application_cap_bps","load_location","legal_actions","current_action"}) | {
    "migration_mode","in_flight_layer","receiver_drained"}
SAMPLE_FIELDS=frozenset(("sequence","request_sent_s","response_received_s","available_s",
    "application_cap_bps","load_location","added_RTT_ms","successful_bytes","traffic_elapsed_s","queue_depth"))
ACK_FIELDS=frozenset(("proposal_sha256","command","applied","continuation_installed","observed_migration_mode",
    "committed_layers","in_flight_layer","receiver_drained","evidence_sha256","acknowledged_at_s","noop_reason","ack_scope"))
MODES=frozenset(("running","paused","stopping","stopped","complete","failed"))


def number(value,name): return e2.finite(value,name)


def _bound(payload): return dict(payload,artifact_sha256=digest(payload))


def build_sticky_table(training_records,e1_plan,e1_acceptance_sha256):
    """Consume the exact E1 TRAINING split, including its control for coverage.

    The aggregate excludes control; every included label is a passed executed
    fixed action. No held-out, warmup, retry or two-model-fit substitution.
    Caller must independently verify the linked raw files and full acceptance.
    """
    import analysis as e1_analysis
    e2.hash_value(e1_acceptance_sha256,"full E1 acceptance")
    evidence=e1_analysis.validate_records(training_records,e1_plan,"training",require_complete=True,require_passed=True)
    groups={}; identities=[]
    for row in evidence["rows"].values():
        if row["method"] not in FIXED_ACTIONS: continue
        features=row["public_features"]
        key=f"P{features['prefix_tokens']}:bps{features['application_cap_bps']}"
        groups.setdefault(key,{a:[] for a in FIXED_ACTIONS})[row["method"]].append(row)
        identities.append({"trial_id":row["trial_id"],"record_sha256":digest(row),
            "measurement_sha256":row["provenance"]["measurement_sha256"]})
    table={}
    for key,by_action in sorted(groups.items()):
        require(all(by_action[a] for a in FIXED_ACTIONS),"missing one of five actually measured training actions")
        paired=[{(r["cell_id"],r["input_family_id"],r["seed"]) for r in by_action[a]} for a in FIXED_ACTIONS]
        require(all(p==paired[0] for p in paired),"training action populations are unpaired")
        means={a:sum(r["metrics"]["completion_s"] for r in by_action[a])/len(by_action[a]) for a in FIXED_ACTIONS}
        table[key]={"mean_last_token_completion_s":means,"rows_per_action":{a:len(by_action[a]) for a in FIXED_ACTIONS},
            "selected_action":min(FIXED_ACTIONS,key=lambda a:(means[a],FIXED_ACTIONS.index(a)))}
    return _bound({"kind":"E3_E1_training_five_action_sticky","e1_plan_sha256":e1_plan["plan_sha256"],
        "e1_acceptance_sha256":e1_acceptance_sha256,"cohort":evidence["cohort"],"table":table,
        "record_identities":sorted(identities,key=lambda r:r["trial_id"]),"training_records_sha256":digest(training_records),
        "fit_split":"training","target":"last_token_completion_s","tie_order":list(FIXED_ACTIONS),
        "control_used_as_label":False,"test_labels_used":False,"raw_files_independently_verified":False,
        "execution_authorized":False,"E3_complete":False})


def admission_action(method,public_admission,sticky_table=None):
    try: from .e3_plan import PUBLIC_FIELDS
    except ImportError: from e3_plan import PUBLIC_FIELDS
    require(method in METHODS and set(public_admission)==PUBLIC_FIELDS,"admission has hidden/future or missing features")
    require(all(type(v) is int for v in public_admission.values()),"public admission geometry must be integer-valued")
    p=public_admission["prefix_tokens"]; cap=public_admission["initial_application_cap_bps"]
    require(type(p) is int and p>0 and type(cap) is int and cap>0 and public_admission["suffix_tokens"]==16 and
            public_admission["public_output_cap"]==128 and public_admission["prefix_KV_bytes"]==p*196608,"invalid public admission")
    if method=="holder_decode_static": return "holder"
    if method!="sticky_training_selected_admission": return "repair"
    e2.verify_artifact(sticky_table,"E3_E1_training_five_action_sticky")
    require(sticky_table["fit_split"]=="training" and sticky_table["test_labels_used"] is False,"sticky labels are not training-only")
    entry=sticky_table["table"].get(f"P{p}:bps{cap}")
    require(entry is not None and set(entry["mean_last_token_completion_s"])==set(FIXED_ACTIONS),"missing exact public training stratum")
    for cost in entry["mean_last_token_completion_s"].values(): number(cost,"training completion")
    selected=min(FIXED_ACTIONS,key=lambda a:(entry["mean_last_token_completion_s"][a],FIXED_ACTIONS.index(a)))
    require(entry["selected_action"]==selected,"sticky frozen choice changed")
    return selected


def build_training_support(training_records,fit):
    """Describe actual frozen E2 TRAINING feature ranges without refitting."""
    e2.verify_artifact(fit,"E2_training_fit")
    expected={r["trial_id"]:r["record_sha256"] for r in fit["training_record_identities"]}
    require(bool(expected) and len(expected)==len(fit["training_record_identities"]),"empty/duplicate frozen training support identities")
    require(isinstance(training_records,list) and len(training_records)==len(expected),"incomplete support training population")
    seen=set(); vectors=[]; geometry=[]
    for row in training_records:
        key=row.get("trial_id")
        require(key in expected and key not in seen and digest(row)==expected[key],"foreign/duplicate/changed support training record")
        require(row.get("phase")=="training" and row.get("sample_role")=="measured" and row.get("status")=="passed","support is training-only")
        seen.add(key); state=row["controller_state"]
        vector,_=e2.feature_vector(state,fit); vectors.append(vector)
        geometry.append({k:state[k] for k in ("prefix_tokens","generated_count","committed_layers","application_cap_bps")})
    ranges={name:{"min":min(v[i] for v in vectors),"max":max(v[i] for v in vectors)} for i,name in enumerate(e2.FEATURE_NAMES)}
    return _bound({"kind":"E3_frozen_E2_training_support","fit_artifact_sha256":fit["artifact_sha256"],
        "training_records_sha256":digest(training_records),"record_count":len(seen),"feature_ranges":ranges,
        "geometry_ranges":{k:{"min":min(r[k] for r in geometry),"max":max(r[k] for r in geometry)} for k in geometry[0]},
        "fit_split":"training","refitted":False,"coverage_guarantee":False,"E3_complete":False})


def validate_sample(sample,now=None):
    require(isinstance(sample,dict) and set(sample)==SAMPLE_FIELDS,"unknown/missing telemetry field; future trace prohibited")
    require(type(sample["sequence"]) is int and sample["sequence"]>=0,"invalid telemetry sequence")
    sent,received,available=[number(sample[k],k) for k in ("request_sent_s","response_received_s","available_s")]
    require(sent<=received<=available,"invalid destination-local observation bracket")
    if now is not None: require(available<=number(now,"now"),"unreleased/future observation")
    require(type(sample["application_cap_bps"]) is int and sample["application_cap_bps"]>0,"zero cap is not an outage")
    require(sample["load_location"] in ("none","holder","destination"),"unknown observed load")
    number(sample["added_RTT_ms"],"observed added RTT")
    require(type(sample["queue_depth"]) is int and sample["queue_depth"]>=0,"invalid observed queue depth")
    b,elapsed=sample["successful_bytes"],sample["traffic_elapsed_s"]
    require((b is None)==(elapsed is None),"traffic bytes and duration must both be missing or observed")
    if b is not None:
        require(type(b) is int and b>=0 and number(elapsed,"traffic duration")>0,"invalid successful traffic interval")
    return True


class ObservationInbox:
    """Executor-owned release queue. Pass only release(now), never this object.

    At an opportunity: release earlier samples; perform the real observation RPC;
    enqueue its reply; release at actual post-reply now; then call the controller.
    The withholding recipe belongs to the executor and never enters a sample.
    """
    def __init__(self):
        self._pending=[]; self._selected=None; self._seen=set(); self._last_now=-1; self.events=[]

    def enqueue(self,sample):
        validate_sample(sample)
        require(sample["sequence"] not in self._seen,"duplicate observation sequence")
        self._seen.add(sample["sequence"]); self._pending.append(copy.deepcopy(sample))

    def release(self,now):
        now=number(now,"release time"); require(now>=self._last_now,"destination release clock went backwards")
        self._last_now=now; ready=[s for s in self._pending if s["available_s"]<=now]
        self._pending=[s for s in self._pending if s["available_s"]>now]
        for sample in sorted(ready,key=lambda s:(s["available_s"],s["sequence"])):
            key=lambda s:(s["request_sent_s"],s["sequence"])
            selected=self._selected is None or key(sample)>key(self._selected)
            self.events.append({"sequence":sample["sequence"],"released_at_s":now,"selected":selected,
                                "late_older_sample_ignored":not selected})
            if selected: self._selected=sample
        return copy.deepcopy(self._selected)


def validate_local(state):
    require(isinstance(state,dict) and set(state)==LOCAL_FIELDS,"unknown local feature; realized G/trace/IDs prohibited")
    for field in e2.STATE_FIELDS & LOCAL_FIELDS:
        require(type(state[field]) is int and state[field]>=0,"invalid local integer geometry")
    p,s,d,f,l=(state[k] for k in ("prefix_tokens","suffix_tokens","generated_count","committed_layers","total_layers"))
    require(p>0 and s==16 and l==48 and 0<d<128 and d%4==0 and 0<=f<=l and state["public_output_cap"]==128,"invalid decision opportunity")
    require(state["prefix_KV_bytes"]==p*196608 and state["committed_KV_bytes"]==p*4096*f and
            state["processed_delta_tokens"]==s+d-1 and state["position"]==p+s+d-1,"delta/pending token or committed bytes differ")
    require(state["migration_mode"] in MODES and type(state["receiver_drained"]) is bool,"invalid migration state")
    flight=state["in_flight_layer"]
    # Source bookkeeping and destination GPU publication are independent
    # observations: a received layer may be published before finish_frame()
    # clears its source-side in-flight marker. Never impose a joint snapshot.
    require(flight is None or type(flight) is int and 0<=flight<l,"invalid in-flight layer")
    require(not state["receiver_drained"] or flight is None,"drained segment still has an in-flight frame")


def legal_commands(state):
    validate_local(state)
    if state["migration_mode"]=="failed": raise ValueError("failed migration requires executor cleanup, not policy continuation")
    if state["committed_layers"]==48: return ["continue_local"]
    mode=state["migration_mode"]
    if mode=="stopping": return ["retain"]
    if mode=="stopped": return ["retain","recompute_full_prefix_keep_delta"]
    commands=["retain","stop_migration_and_remote","complete_fetch_then_local","recompute_full_prefix_keep_delta"]
    if mode=="running": commands.append("pause")
    if mode=="paused": commands.append("resume")
    return commands


class PersistentController:
    """One live request, frozen E2 predictors and explicit control receipts.

    Pause receipt is never drain proof. Complete fetch from paused requires an
    executor resume followed by full receiver drain; source-command noops remain
    separate from an actually installed continuation. All facts are audited later.
    """
    def __init__(self,method,fit,calibration,e2_acceptance_sha256,*,training_support=None):
        require(method in ONLINE,"fixed methods must run without dummy telemetry/controller callbacks")
        e2.verify_artifact(fit,"E2_training_fit")
        require(fit.get("fit_split")=="training" and fit.get("future_length_feature") is False and
                fit.get("test_labels_used") is False,"E2 predictors must be frozen training-only")
        e2.verify_artifact(calibration,"E2_residual_calibration",fit["plan_sha256"])
        require(calibration.get("fit_artifact_sha256")==fit["artifact_sha256"] and
                calibration.get("calibration_split")=="calibration" and calibration.get("test_labels_used") is False,
                "foreign/test-recalibrated residuals")
        require(set(calibration["radii_s"])==set(e2.ACTIONS),"four-action residuals required")
        for radius in calibration["radii_s"].values(): number(radius,"frozen residual")
        e2.hash_value(e2_acceptance_sha256,"accepted full E2")
        self.method,self.fit,self.calibration=method,copy.deepcopy(fit),copy.deepcopy(calibration)
        self.e2_acceptance_sha256=e2_acceptance_sha256
        if training_support is not None:
            e2.verify_artifact(training_support,"E3_frozen_E2_training_support")
            require(training_support["fit_artifact_sha256"]==fit["artifact_sha256"],"foreign training support")
        self.training_support=copy.deepcopy(training_support)
        self.pending=None; self.decisions=[]; self.last_generated=0; self.last_now=-1
        self.last_change_s=None; self.applied_changes=0; self.irreversibly_stopped=False

    def propose(self,state,observation,*,now):
        validate_local(state); now=number(now,"decision time")
        require(self.pending is None and state["generated_count"]>self.last_generated and now>=self.last_now,"pending ACK or reversed decision time/progress")
        if self.irreversibly_stopped:
            require(state["migration_mode"] in ("stopping","stopped","complete") or state["committed_layers"]==48,"stopped segment illegally resumed")
        if observation is not None: validate_sample(observation,now)
        legal=legal_commands(state); mode=state["migration_mode"]
        current="continue_local" if state["committed_layers"]==48 else (
            "stop_migration_and_remote" if mode in ("paused","stopping","stopped") else "continue_repair")
        age=None if observation is None else now-observation["request_sent_s"]
        stale=observation is None or age>.5
        result={"method":self.method,"state_sha256":digest(state),"observation_sha256":None if observation is None else digest(observation),
            "fit_artifact_sha256":self.fit["artifact_sha256"],"calibration_artifact_sha256":self.calibration["artifact_sha256"],
            "e2_acceptance_sha256":self.e2_acceptance_sha256,"decision_index":len(self.decisions),"generated_count":state["generated_count"],
            "proposed_at_s":now,"legal_commands":legal,"observation_age_upper_s":age,"stale":stale,
            "future_information_used":False,"executed":False,"E3_complete":False,
            "dynamic_distribution_coverage_claim":False,"paused_prediction_proxy":mode=="paused",
            "predictions":{},"model_state":None,"feature_details":None,"OOD":None,
            "training_support_artifact_sha256":None if self.training_support is None else self.training_support["artifact_sha256"],
            "training_feature_extrapolation":None,
            "command":"retain","target_continuation":current,"reason":"retain_actual_route",
            "successful_goodput_Bps":None if observation is None or not observation["successful_bytes"] else
                observation["successful_bytes"]/observation["traffic_elapsed_s"]}
        guarded=self.method!="same_information_greedy"
        if current=="continue_local": result.update(command="continue_local",reason="natural_all_local")
        elif mode=="stopping": result["reason"]="await_complete_frame_stop_drain"
        elif observation is None or stale and self.method=="state_aware_guarded":
            result.update(command="pause" if "pause" in legal else "retain",reason="missing_observation_safe_hold" if observation is None else "stale_observation_safe_hold")
        else:
            model={k:state[k] for k in e2.STATE_FIELDS & LOCAL_FIELDS}
            actions=list(e2.ACTIONS) if not self.irreversibly_stopped and mode!="stopped" else ["stop_migration_and_remote","recompute_full_prefix_keep_delta"]
            model.update(application_cap_bps=observation["application_cap_bps"],load_location=observation["load_location"],
                         legal_actions=actions,current_action=current)
            predictions,details=e2.predict_costs(model,self.fit)
            ood={"load":observation["load_location"]!="none","RTT":observation["added_RTT_ms"]>0,
                 "prefix":state["prefix_tokens"] not in (1024,4096),"rate":observation["application_cap_bps"] not in (100000000,1000000000),
                 "frontier_outside_training_range":not 12<=state["committed_layers"]<=47,
                 "progress_outside_training_range":not 4<=state["generated_count"]<=16,
                 "temporal_shift":True}
            result.update(predictions=predictions,model_state=model,feature_details=details,OOD=ood)
            if self.training_support is not None:
                ranges=self.training_support["feature_ranges"]
                result["training_feature_extrapolation"]={name:not ranges[name]["min"]<=value<=ranges[name]["max"]
                    for name,value in zip(e2.FEATURE_NAMES,details["feature_vector"])}
            # Unknown load/RTT/cap physics retain a legal route; frontier/progress
            # extrapolation is disclosed and is the explicit purpose of E3.
            hard_ood=any(ood[k] for k in ("load","RTT","prefix","rate"))
            if guarded and hard_ood:
                result["reason"]="observed_environment_OOD_retain_legal_route"
            else:
                if guarded:
                    radii=self.calibration["radii_s"]
                    alternatives=[a for a in actions if a!=current]
                    candidate=min(alternatives,key=lambda a:(predictions[a]+radii[a],e2.ACTIONS.index(a))) if alternatives else current
                    lower=max(0.,predictions[current]-radii[current]); upper=predictions[candidate]+radii[candidate]
                    chosen=candidate if candidate!=current and upper<.95*lower else current
                    result["guard"]={"candidate":candidate,"candidate_upper_s":upper,"current_lower_s":lower,"margin":.05}
                else: chosen=min(actions,key=lambda a:(predictions[a],e2.ACTIONS.index(a)))
                command=("resume" if mode=="paused" else "retain") if chosen=="continue_repair" else chosen if chosen!=current else "retain"
                result.update(command=command,target_continuation=chosen,reason="guarded_comparison" if guarded else "point_minimum")
        if result["command"] in ("pause","resume"):
            result["target_continuation"]="pause_remote_hold" if result["command"]=="pause" else "continue_repair"
        changing=result["command"] not in ("retain","continue_local")
        if changing and (self.applied_changes>=8 or guarded and self.last_change_s is not None and now-self.last_change_s<.25):
            result.update(command="retain",target_continuation=current,reason="applied_change_limit" if self.applied_changes>=8 else "minimum_applied_dwell")
        require(result["command"] in legal,"controller proposed illegal command")
        result["executor_requires_resume_before_fetch"]=result["command"]=="complete_fetch_then_local" and mode=="paused"
        result["proposal_sha256"]=digest(result)
        self.pending={"proposal":copy.deepcopy(result),"state":copy.deepcopy(state),"observation":copy.deepcopy(observation)}
        return copy.deepcopy(result)

    def acknowledge(self,ack):
        require(self.pending is not None,"no pending control proposal")
        p=self.pending["proposal"]; before=self.pending["state"]
        require(isinstance(ack,dict) and set(ack)==ACK_FIELDS,"distinct E3 command/continuation/drain ACK schema required")
        require(ack["proposal_sha256"]==p["proposal_sha256"] and ack["command"]==p["command"],"ACK does not bind exact proposal")
        require(type(ack["applied"]) is bool and type(ack["continuation_installed"]) is bool and
                type(ack["receiver_drained"]) is bool,"ACK truth fields must be booleans")
        require(ack["observed_migration_mode"] in MODES-{"failed"},"failed ACK cannot advance policy")
        at=number(ack["acknowledged_at_s"],"ACK receive time"); require(at>=p["proposed_at_s"],"ACK predates decision")
        e2.hash_value(ack["evidence_sha256"],"actual command/frame evidence")
        f,flight=ack["committed_layers"],ack["in_flight_layer"]
        require(type(f) is int and before["committed_layers"]<=f<=48,"ACK frontier regressed")
        require(flight is None or type(flight) is int and 0<=flight<48,"invalid ACK in-flight layer")
        require(not ack["receiver_drained"] or flight is None,"command receipt is not drained frame evidence")
        command=p["command"]; mode=ack["observed_migration_mode"]
        if command in ("pause","resume"):
            require(ack["ack_scope"]=="command_receipt_not_frame_drain","pause/resume scope must not imply drain")
            if ack["applied"]:
                require(mode in (("paused","complete") if command=="pause" else ("running","complete")),"applied command contradicts migration state")
            else:
                expected=("paused","stopping","stopped","complete") if command=="pause" else ("running","stopping","stopped","complete")
                require(bool(ack["noop_reason"]) and mode in expected,"no-op requires actual terminal/repeated-command reason")
        elif command in ("complete_fetch_then_local","recompute_full_prefix_keep_delta"):
            require(ack["ack_scope"]=="continuation_installed_after_receiver_drain" and ack["applied"] and
                    ack["continuation_installed"] and ack["receiver_drained"] and f==48,"local continuation lacks actual complete prefix/drain")
        elif command=="stop_migration_and_remote":
            require(ack["ack_scope"]=="continuation_installed_after_receiver_drain" and ack["continuation_installed"] and
                    ack["receiver_drained"] and mode in ("stopped","complete"),"stop continuation not safely drained")
            require(ack["applied"] or bool(ack["noop_reason"]),"source stop noop reason missing")
        else:
            require(ack["ack_scope"]=="local_no_control_command" and not ack["applied"] and ack["continuation_installed"],"retain/local is not a fabricated applied control")
            if command=="continue_local": require(f==48,"continue-local ACK lacks full prefix")
        if ack["applied"]: require(ack["noop_reason"] is None,"applied command cannot also be a noop")
        if command in ("stop_migration_and_remote","recompute_full_prefix_keep_delta") and ack["continuation_installed"]:
            self.irreversibly_stopped=True
        if mode in ("stopping","stopped"): self.irreversibly_stopped=True
        actual_change=ack["applied"] and command not in ("retain","continue_local")
        if actual_change:
            self.applied_changes+=1; self.last_change_s=at
        self.last_generated=before["generated_count"]; self.last_now=at
        row=dict(self.pending,ack=copy.deepcopy(ack),actual_applied_change=actual_change,
                 actual_applied_changes_total=self.applied_changes,matched_physical_state_claim=False)
        self.decisions.append(row); self.pending=None
        return copy.deepcopy(row)
