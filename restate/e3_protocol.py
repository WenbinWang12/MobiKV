"""E3 execution envelope: 14 excluded smoke trials before 1296 main trials.

The eight lifecycle fixtures need real executor hooks and independent evidence.
Their declaration does not implement them or grant GPU execution acceptance.
"""
import copy
import math
if __package__:
    from . import plan
else:
    import e3_plan as plan

FUNCTIONAL=(
    ('f01','live_pause_resume','p4096-constant_slow-fresh',64),
    ('f02','inflight_stop','p4096-constant_slow-fresh',64),
    ('f03','terminal_control_noops','p1024-constant_fast-fresh',64),
    ('f04','stale_observation_order','p1024-drop-temporary_feedback_delay',64),
    ('f05','rate_boundary_shared_budget','p4096-drop_recover-fresh',64),
    ('f06','hidden_future_invariance','p1024-drop_recover-temporary_feedback_delay',64),
    ('f07','multi_cutover_route_replay','p4096-alternating-fresh',64),
    ('f08','owned_failure_cleanup','p1024-holder-load-step',8),
)


def smoke_trials(protocol):
    plan.validate_plan(protocol)
    result=[]
    selections=[('integration',f'i{i+1:02d}',method,'p1024-constant_fast-fresh',64,None)
                for i,method in enumerate(plan.METHODS)]
    selections += [('functional',key,'state_aware_guarded',cell,g,name) for key,name,cell,g in FUNCTIONAL]
    for order,(kind,key,method,cell,g,name) in enumerate(selections):
        matches=[t for t in protocol['trials'] if t['cell_id']==cell and t['method']==method and t['sample_role']=='warmup']
        plan.require(len(matches)==1,'Missing unique E3 smoke template')
        trial=copy.deepcopy(matches[0]);identity='E3.smoke.'+key+'.'+method
        trial.update(trial_id=identity,phase='smoke',sample_role='functional_excluded',smoke_only=True,
            smoke_kind=kind,block_id=key,paired_group_id='E3.smoke.'+key,method_order_position=order,
            input_family_id='e3-full-v1-smoke',input_id='e3-full-v1-smoke',seed=2026112101,
            prefix_key='e3-full-v1-smoke:P'+str(trial['public_admission']['prefix_tokens']),
            diagnostic_id=identity+'.route-diagnostic',reference_id=identity+'.HF-reference')
        trial['executor_only']['realized_stop_tokens']=g
        if kind=='functional':
            trial['smoke_fixture']=dict(id=key,name=name,real_executor_hook_required=True,
                test_injection_excluded_from_performance=True,CPU_fixture_cannot_pass=True)
            if key=='f08':trial['smoke_fixture'].update(
                failure_point='after_complete_8_token_decode_while_migration_still_live',
                original_request_must_remain_failed=True,fixture_acceptance_separate=True,
                expected_error='InjectedE3OwnedCleanupFault')
        result.append(trial)
    plan.require(len(result)==14 and sum(t['smoke_kind']=='functional' for t in result)==8,'E3 smoke coverage changed')
    return result


def validate_wrapper(wrapper):
    plan.require(wrapper.get('stage')=='E3' and wrapper.get('mode') in ('e3_smoke','e3_full'),'Foreign E3 envelope')
    plan.validate_plan(wrapper['full_protocol'])
    for name,maximum in (('trial_timeout_s',1200),('stage_timeout_s',259200)):
        value=wrapper['limits'].get(name)
        plan.require(type(value) in (int,float) and math.isfinite(value) and 0<value<=maximum,'Invalid bounded deadline '+name)
    for name in ('e1_dependency','e2_dependency'):
        item=wrapper.get(name,{})
        plan.require(set(item)=={'acceptance_path','acceptance_sha256'} and isinstance(item['acceptance_path'],str)
                     and len(item['acceptance_sha256'])==64,'Bound complete-stage dependency required')
    if wrapper['mode']=='e3_smoke':
        plan.require(wrapper.get('smoke_trials')==smoke_trials(wrapper['full_protocol']),'Prescribed 8+6 E3 smoke coverage differs')
    else:
        plan.require(not wrapper.get('smoke_trials') and not wrapper.get('smoke_fixtures'),'Functional injection entered main matrix')
        accepted=wrapper.get('E3_smoke_acceptance',{})
        plan.require(accepted.get('passed') is True and accepted.get('candidate_count')==14
                     and accepted.get('functional_cases')==8 and accepted.get('integration_cases')==6,
                     'Source-bound E3 functional plus integration smoke required')
    return True
