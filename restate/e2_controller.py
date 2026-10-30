"""Pure causal E2 decisions. This module never executes a continuation.

The strict state allowlist intentionally excludes request/cell IDs, realized G,
future tokens, other methods' outcomes and diagnostics. Artifact hashes bind
provenance; a separate raw-file auditor must verify those linked artifacts.
"""
from __future__ import annotations

import math
import re
try:
    from e2_plan import ACTIONS, POLICIES, CONTINUOUS_METHODS, digest, require
except ImportError:
    from e2_plan import ACTIONS, POLICIES, CONTINUOUS_METHODS, digest, require

STATE_FIELDS = frozenset(("prefix_tokens", "suffix_tokens", "public_output_cap", "generated_count", "committed_layers", "total_layers", "prefix_KV_bytes", "committed_KV_bytes", "application_cap_bps", "load_location", "processed_delta_tokens", "position", "legal_actions", "current_action"))
FEATURE_NAMES = ("intercept", "frozen_E1_R_TTFT", "frozen_E1_F_TTFT", "remaining_transfer_budget_s", "causal_horizon", "horizon_missing_fraction", "horizon_context_kilo_tokens")
HEX = re.compile(r"[0-9a-f]{64}\Z")


def finite(value, name, *, positive=False):
    require(type(value) in (int,float), f"{name} must be numeric")
    value = float(value)
    require(math.isfinite(value) and (value>0 if positive else value>=0), f"invalid {name}")
    return value


def hash_value(value, name):
    require(isinstance(value,str) and HEX.fullmatch(value) is not None, f"invalid {name}")


def bound_artifact(payload):
    require("artifact_sha256" not in payload, "cannot rebind an already hashed artifact")
    return dict(payload, artifact_sha256=digest(payload))


def verify_artifact(artifact, kind, plan_sha256=None):
    require(isinstance(artifact,dict), "artifact missing")
    payload = {k:v for k,v in artifact.items() if k != "artifact_sha256"}
    require(artifact.get("kind")==kind and artifact.get("artifact_sha256")==digest(payload), "artifact kind/hash mismatch")
    if plan_sha256 is not None:
        require(artifact.get("plan_sha256")==plan_sha256, "foreign artifact plan")


def validate_state(state):
    require(isinstance(state,dict) and set(state)==STATE_FIELDS, "unknown/missing controller feature; future information prohibited")
    for name in ("prefix_tokens", "suffix_tokens", "public_output_cap", "generated_count", "committed_layers", "total_layers", "prefix_KV_bytes", "committed_KV_bytes", "application_cap_bps", "processed_delta_tokens", "position"):
        require(type(state[name]) is int and state[name]>=0, f"invalid integer state {name}")
    p,s,d,f,l = (state[x] for x in ("prefix_tokens","suffix_tokens","generated_count","committed_layers","total_layers"))
    require(p>0 and s==16 and l==48 and state["public_output_cap"]==128 and 0<d<128 and 0<=f<=l and state["application_cap_bps"]>0, "invalid model/request state")
    require(state["prefix_KV_bytes"]==p*196608 and state["committed_KV_bytes"]==p*4096*f, "KV bytes do not represent actual complete layers")
    require(state["processed_delta_tokens"]==s+d-1 and state["position"]==p+s+d-1, "pending-token position/delta mismatch")
    require(state["load_location"] in ("none","holder","destination"), "unknown load state")
    legal = state["legal_actions"]
    require(isinstance(legal,list) and legal and len(legal)==len(set(legal)) and set(legal)<=set(ACTIONS)|{"continue_local"}, "invalid legal mask")
    require(state["current_action"] in ACTIONS+("continue_local",), "invalid current action")
    require((legal==["continue_local"]) if f==48 else "continue_local" not in legal, "completed-prefix legality mismatch")


def validate_e1_fit(artifact):
    verify_artifact(artifact,"E1_nominal_training_fit")
    require(artifact.get("fit_split")=="training" and artifact.get("future_length_feature") is False and artifact.get("test_labels_used") is False and not artifact.get("smoke_only",False) and not artifact.get("CPU_fixture_only",False), "E1 fit must be genuine training-only evidence")
    for action,n in (("recompute",3),("native_fetch",2)):
        values = artifact["models"][action]["coefficients"]
        require(isinstance(values,list) and len(values)==n, "invalid E1 coefficient shape")
        for value in values: finite(value,"E1 coefficient")


def e1_prediction(e1_fit, state):
    validate_e1_fit(e1_fit)
    p = state["prefix_tokens"]/1024
    r = e1_fit["models"]["recompute"]["coefficients"]
    f = e1_fit["models"]["native_fetch"]["coefficients"]
    return r[0]+r[1]*p+r[2]*p*p, f[0]+f[1]*8*state["prefix_KV_bytes"]/state["application_cap_bps"]


def horizon(population, generated_count, public_cap=128, *, ablated=False):
    require(isinstance(population,list) and bool(population), "empty training horizon population")
    require(all(type(v) is int and 0<v<=public_cap for v in population), "invalid training horizon population")
    require(type(generated_count) is int and 0<generated_count<public_cap, "invalid observed progress")
    if ablated:
        value = sum(population)/len(population)
        source = "fixed_training_mean_total_used_as_remaining"
    else:
        remaining = [g-generated_count for g in population if g>generated_count]
        value = sum(remaining)/len(remaining) if remaining else 16.0
        source = "conditional_training_survivors" if remaining else "fixed_fallback_no_training_survivors"
    return min(float(public_cap-generated_count),max(1.0,value)), source


def feature_vector(state, fit, *, ablate_progress=False, ablate_horizon=False):
    validate_state(state)
    r,f = e1_prediction(fit["e1_fit"],state)
    h,source = horizon(fit["remaining_length_population"],state["generated_count"],state["public_output_cap"],ablated=ablate_horizon)
    frontier = 0 if ablate_progress else state["committed_layers"]
    remaining = state["prefix_KV_bytes"] if ablate_progress else state["prefix_KV_bytes"]-state["committed_KV_bytes"]
    vector = [1.0,r,f,8*remaining/state["application_cap_bps"],h,h*(1-frontier/48),h*state["position"]/1024]
    require(all(math.isfinite(v) for v in vector), "nonfinite feature vector")
    return vector, {"horizon":h,"horizon_source":source,"progress_feature_ablated":ablate_progress,"horizon_feature_ablated":ablate_horizon}


def predict_costs(state, fit, *, ablate_progress=False, ablate_horizon=False):
    verify_artifact(fit,"E2_training_fit")
    require(tuple(fit.get("feature_names",[]))==FEATURE_NAMES and set(fit["models"])==set(ACTIONS), "E2 model schema mismatch")
    features, details = feature_vector(state,fit,ablate_progress=ablate_progress,ablate_horizon=ablate_horizon)
    predictions = {}
    for action in ACTIONS:
        model=fit["models"][action]
        coefficients,mean,scale=(model[k] for k in ("coefficients","feature_mean","feature_scale"))
        require(len(coefficients)==7 and len(mean)==len(scale)==6, "E2 model dimensions mismatch")
        require(all(type(v) in (int,float) and math.isfinite(v) for v in coefficients+mean), "nonfinite E2 parameters")
        for v in scale: finite(v,"feature scale",positive=True)
        value=coefficients[0]+sum(coefficients[j+1]*(features[j+1]-mean[j])/scale[j] for j in range(6))
        predictions[action]=max(1e-6,float(value))
    return predictions,dict(details,feature_vector=features)


def choose_action(state, fit, calibration, method):
    """Return a proposal, never an executed/ACKed action or a completion gate.

    Sticky baselines should be invoked at the first decision only; later calls
    retain an actually ACKed current continuation. Use ``initial_decision`` in
    a runner's external policy state, not a future-bearing controller feature.
    A fully local state always returns continue_local without a model ranking.
    """
    validate_state(state)
    verify_artifact(fit,"E2_training_fit")
    require(method in ACTIONS+POLICIES+CONTINUOUS_METHODS, "unknown policy")
    legal=state["legal_actions"]
    order=[a for a in ACTIONS if a in legal]
    current=state["current_action"]
    result={"method":method,"legal_actions":list(legal),"current_action":current,"executed":False,"future_information_used":False,"fit_artifact_sha256":fit["artifact_sha256"],"E2_complete":False}
    if legal==["continue_local"]:
        return dict(result,action="continue_local",reason="complete_prefix",predictions={})
    require(bool(order),"no implemented legal continuation")
    actual_OOD=state["load_location"]!="none" or state["prefix_tokens"] not in (1024,4096) or state["application_cap_bps"] not in (100_000_000,1_000_000_000)
    if method in ACTIONS:
        require(method in legal,"requested fixed action illegal")
        return dict(result,action=method,reason="fixed_intervention",predictions={},OOD=actual_OOD)
    base={"sticky_initial_E1_nominal_rule":"initial_E1_nominal_rule","sticky_training_tuned_static":"training_tuned_state_blind_static"}.get(method,method)
    if base=="initial_E1_nominal_rule":
        r,f=e1_prediction(fit["e1_fit"],state)
        proposed="recompute_full_prefix_keep_delta" if r<=f else "complete_fetch_then_local"
        return dict(result,action=proposed if proposed in legal else order[0],reason="initial_E1_rule" if proposed in legal else "initial_rule_illegal_safe_fallback",predictions={"E1_R_TTFT":r,"E1_F_TTFT":f},OOD=actual_OOD)
    if base=="training_tuned_state_blind_static":
        key=f"P{state['prefix_tokens']}:bps{state['application_cap_bps']}"
        table=fit["static_table"].get(key)
        require(table is not None,"missing frozen static stratum")
        ranking=sorted(ACTIONS,key=lambda a:(table["mean_remaining_completion_s"][a],ACTIONS.index(a)))
        return dict(result,action=next(a for a in ranking if a in legal),reason="training_static_legal_ranking",predictions=dict(table["mean_remaining_completion_s"]),OOD=actual_OOD)
    predictions,details=predict_costs(state,fit,ablate_progress=base=="ablate_committed_progress",ablate_horizon=base=="ablate_remaining_horizon_update")
    best=min(order,key=lambda a:(predictions[a],ACTIONS.index(a)))
    result.update(details,predictions=predictions,OOD=actual_OOD)
    if base=="same_information_greedy":
        return dict(result,action=best,reason="legal_point_minimum")
    verify_artifact(calibration,"E2_residual_calibration",fit["plan_sha256"])
    require(calibration.get("fit_artifact_sha256")==fit["artifact_sha256"] and set(calibration.get("radii_s",{}))==set(ACTIONS),"foreign residual calibration")
    radii={a:finite(calibration["radii_s"][a],"residual radius") for a in ACTIONS}
    result["calibration_artifact_sha256"]=calibration["artifact_sha256"]
    # Only the proposed guarded controller claims a safe fallback under actual
    # load OOD; the explicitly labeled ablations retain frozen extrapolation.
    if actual_OOD and base=="state_aware_guarded":
        return dict(result,action=current if current in legal else order[0],reason="actual_state_OOD_safe_current_fallback")
    if current not in legal:
        return dict(result,action=best,reason="current_illegal_choose_legal_point_minimum")
    margin=0.0 if base=="ablate_guard_hysteresis_only" else 0.05
    lower=max(0.0,predictions[current]-radii[current])
    alternatives=[a for a in order if a!=current]
    candidate=min(alternatives,key=lambda a:(predictions[a]+radii[a],ACTIONS.index(a))) if alternatives else current
    upper=predictions[candidate]+radii[candidate]
    switch=candidate!=current and upper<(1-margin)*lower
    return dict(result,action=candidate if switch else current,reason="guarded_improvement" if switch else "tied_or_unresolved_retain_current",guard={"candidate":candidate,"candidate_upper_s":upper,"current_lower_s":lower,"required_fractional_margin":margin,"switch_proposed":switch})


class PersistentPolicy:
    """CPU proposal/ACK state; a runner must bind ACK evidence to actual transport."""
    def __init__(self,method,fit,calibration):
        self.method,self.fit,self.calibration=method,fit,calibration
        self.current=None; self.pending=None; self.last_generated=0; self.decisions=[]

    def propose(self,state):
        validate_state(state)
        require(self.pending is None,"previous proposal has no execution ACK")
        require(state["generated_count"]>self.last_generated,"checkpoint not strictly later")
        if self.current is not None:
            require(state["current_action"]==self.current or state["legal_actions"]==["continue_local"],"observed current differs from last actual ACK")
        if self.current is not None and self.method.startswith("sticky_") and self.current in state["legal_actions"]:
            result={"method":self.method,"action":self.current,"reason":"sticky_actual_continuation","executed":False,"future_information_used":False,"E2_complete":False}
        else:
            result=choose_action(state,self.fit,self.calibration,self.method)
        self.pending={"proposal":result,"state_sha256":digest(state),"generated_count":state["generated_count"]}
        return dict(result,decision_index=len(self.decisions),state_sha256=self.pending["state_sha256"])

    def acknowledge(self,ack):
        require(self.pending is not None,"no pending proposal")
        require(isinstance(ack,dict) and set(ack)=={"applied","action","state_sha256","evidence_sha256"},"ACK schema mismatch")
        require(ack["applied"] is True and ack["action"]==self.pending["proposal"]["action"] and ack["state_sha256"]==self.pending["state_sha256"],"ACK does not prove proposed action applied")
        hash_value(ack["evidence_sha256"],"actual ACK evidence")
        previous=self.current
        self.current=ack["action"]; self.last_generated=self.pending["generated_count"]
        changed=previous is not None and previous!=self.current
        automatic_local=self.pending["proposal"]["reason"]=="complete_prefix"
        self.decisions.append(dict(self.pending,ack=dict(ack),initial_installation=previous is None,actual_action_changed=changed,automatic_local_completion=automatic_local,actual_policy_switch=changed and not automatic_local))
        self.pending=None
