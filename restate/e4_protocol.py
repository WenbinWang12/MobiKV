"""Source-bound E4 execution envelope; no GPU work or stage completion here."""
import math
if __package__:
    from . import plan
else:
    import e4_plan as plan

require = plan.require


def trials(wrapper):
    validate_wrapper(wrapper)
    return active_protocol(wrapper)['trials']


def active_protocol(wrapper):
    return wrapper[{'e4_smoke':'smoke_protocol','e4_full':'full_protocol','e4_profile':'profile_protocol'}[wrapper['mode']]]


def validate_wrapper(wrapper):
    require(wrapper.get('stage')=='E4' and wrapper.get('mode') in ('e4_smoke','e4_full','e4_profile'), 'Foreign E4 envelope')
    require(wrapper.get('E4_complete') is False and wrapper.get('paper_stage_complete') is False,
            'Execution envelope cannot assert completed evidence')
    require(all(wrapper.get(k) is False for k in ('dependency_evidence_complete','scientific_acceptance','coverage_claim_allowed')),
            'Inherited evidence restriction flags required')
    waiver=wrapper.get('e2_raw_dependency_waiver')
    require(isinstance(waiver,dict) and waiver.get('kind')=='historical_E2_raw_dependency_reuse' and
            all(waiver.get(k) is False for k in ('training_support_records_available','synthetic_records_created','coverage_claim_allowed')),
            'Inherited E2 evidence waiver required')
    full=wrapper['full_protocol'];plan.validate_plan(full)
    require(full['mode']=='e4_full' and full['counts']['attempts']==2340,'Complete main plan required')
    dependency=wrapper.get('e3_dependency',{})
    require(set(dependency)=={'acceptance_path','acceptance_sha256'} and
            isinstance(dependency['acceptance_path'],str) and dependency['acceptance_path'] and
            isinstance(dependency['acceptance_sha256'],str) and len(dependency['acceptance_sha256'])==64 and
            all(c in '0123456789abcdef' for c in dependency['acceptance_sha256']),
            'Explicit consolidated E3 identity required; launcher independently verifies it')
    limits=wrapper['limits']
    require(limits.get('trial_timeout_s')==1200,'Common predeclared E4 group/solo deadline changed')
    limit=limits.get('stage_timeout_s')
    require(type(limit) is int and 0<limit<=14*86400,'Bounded E4 stage deadline required')
    if wrapper['mode']=='e4_smoke':
        smoke=wrapper['smoke_protocol'];plan.validate_plan(smoke)
        require(smoke['mode']=='e4_smoke' and smoke['design_sha256']==full['design_sha256'] and
                smoke['counts']['attempts']==107 and limit==43200,'New independent 107-attempt smoke required')
        require(not wrapper.get('E4_smoke_acceptance'),'Smoke cannot manufacture its own acceptance')
    else:
        require(not wrapper.get('smoke_protocol'),'Smoke trials cannot enter full matrix')
        gate=wrapper.get('E4_smoke_acceptance',{})
        require(gate.get('passed') is True and gate.get('candidate_count')==107 and
                gate.get('decoded_request_executions')==156 and gate.get('transfer_job_executions')==10,
                'Actual complete new E4 smoke required')
        review=wrapper.get('resource_review',{})
        require(review.get('passed') is True and review.get('source_bound_smoke_sha256')==gate.get('sha256') and
                review.get('trial_timeout_s')==1200 and
                review.get('actual_sample_count')==107,'Full needs the same-source smoke resource review')
        if wrapper['mode']=='e4_full':
            require(review.get('stage_timeout_s')==limit and not wrapper.get('profile_protocol'),
                    'Main deadline must match actual smoke resource review')
        else:
            profiles=wrapper['profile_protocol'];plan.validate_plan(profiles)
            require(profiles['mode']=='e4_profiles' and profiles['design_sha256']==full['design_sha256'] and
                    profiles['counts']['attempts']==16 and limit==86400,
                    'Independent sixteen profiles and bounded separate session required')
            parent=wrapper.get('profile_main_acceptance',{})
            require(set(parent)=={'path','sha256','run_id','frozen_sha256'} and all(
                isinstance(parent.get(k),str) and parent[k] for k in parent),'Exact accepted main-matrix identity required')
            require(all(len(parent[k])==64 and all(c in '0123456789abcdef' for c in parent[k])
                for k in ('sha256','frozen_sha256')),'Profile parent hashes malformed')
            require(review.get('stage_timeout_s',0)>0,'Profile must retain the original main resource review')
    return True


def review_resources(smoke_costs, smoke_acceptance_sha256, full_protocol):
    """Conservative elapsed-time envelope using only actual completed smoke.

    Per-kind maximum smoke ownership cost (main+separate diagnostic+HF+
    untimed preparation/reset) is multiplied by declared full kind counts.
    This is an estimate, not a measured full runtime or GPU-active duration.
    A common group deadline is never increased after a losing attempt.
    """
    plan.validate_plan(full_protocol)
    expected=plan.build_plan(full_protocol['design'],mode='smoke')
    index={t['trial_id']:t for t in expected['trials']}
    require(len(smoke_costs)==107 and {r['trial_id'] for r in smoke_costs}==set(index),
            'Resource review requires all 107 actual smoke costs')
    by_kind={k:[] for k in ('group','decoded_request_solo','transfer_solo')}
    for row in smoke_costs:
        trial=index[row['trial_id']]
        require(row.get('passed') is True and row.get('kind')==trial['kind'],'Foreign or failed smoke cost')
        for key in ('ownership_elapsed_s','maximum_actual_attempt_service_s'):
            value=row.get(key)
            require(type(value) in (int,float) and math.isfinite(value) and value>0,'Actual finite smoke duration required')
        require(row['maximum_actual_attempt_service_s']<=row['ownership_elapsed_s'], 'Service exceeds full ownership interval')
        require(row['maximum_actual_attempt_service_s']*2+30<=1200,
                'Smoke lacks headroom for common frozen deadline; needs explicit protocol review')
        by_kind[row['kind']].append(row['ownership_elapsed_s'])
    require(all(by_kind.values()),'All group and solo kinds must be measured')
    counts=full_protocol['counts']['by_kind']
    components={kind:dict(actual_sample_count=len(values),observed_min_s=min(values),
        observed_max_s=max(values),full_attempts=counts[kind],upper_estimate_s=max(values)*counts[kind])
        for kind,values in by_kind.items()}
    upper=sum(c['upper_estimate_s'] for c in components.values())
    deadline=math.ceil(1.5*upper+3600)
    require(deadline<=14*86400,'Estimated full session exceeds fourteen-day safety ceiling; needs resource review')
    return dict(passed=True,source_bound_smoke_sha256=smoke_acceptance_sha256,actual_sample_count=107,
        trial_timeout_s=1200,stage_timeout_s=deadline,components=components,
        estimated_full_ownership_upper_s=upper,safety_multiplier=1.5,extra_session_seconds=3600,
        estimates_are_not_measured_full_runtime=True,GPU_active_hours=None,E4_complete=False,
        scope='Main matrix/session bound only; sixteen separate profiles and independent CPU review have separate deadlines')
