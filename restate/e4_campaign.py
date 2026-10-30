"""Persistent real E4 matrix driver; independent raw checks gate each record.

Request timings come from the destination executor. Separate concurrent route
diagnostics and same-shape HF executions never enter main latency samples.
No retry, stage-complete flag, learned scheduler, or new numerical tolerance.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

if __package__:
    from . import protocol as envelope,plan
else:
    import e4_protocol as envelope
    import e4_plan as plan

require,digest=plan.require,plan.digest


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    # Preserve an invalid observed number explicitly; never manufacture zero.
    if type(value) is float and not math.isfinite(value):return {'nonfinite_float':repr(value)}
    if isinstance(value,dict):return {key:json_safe(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(item) for item in value]
    return value


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temporary.open('x',encoding='utf-8') as f:
        json.dump(json_safe(value),f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    temporary.replace(path)


def default_bindings():
    from e4_destination import E4Destination
    import e4_raw_checks
    from campaign_runner import materialize
    from campaign_client import verify_outputs
    from e0_holder_decode import compare
    return SimpleNamespace(Destination=E4Destination,raw_checks=e4_raw_checks,
        materialize=materialize,verify_outputs=verify_outputs,compare=compare,clock=time.monotonic)


class FullCampaign:
    def __init__(self,engine,tokenizer,controls,hosts,wrapper,directory,report,*,transport_hosts=None,bindings=None):
        self.bindings=bindings or default_bindings();envelope.validate_wrapper(wrapper)
        require(len(controls)==len(hosts)==1,'Exactly one real holder required')
        self.engine,self.tokenizer,self.controls,self.hosts=engine,tokenizer,controls,hosts
        self.plan,self.directory,self.report=wrapper,Path(directory),report
        self.trials=copy.deepcopy(envelope.trials(wrapper))
        self.protocol=envelope.active_protocol(wrapper)
        self.context=report['execution_context'];self.clock=self.bindings.clock
        require(self.context['stage']=='E4' and self.context['mode']==wrapper['mode'] and
                self.context['source_sha256']==report['source_sha256'],'Wrong frozen execution context')
        require(engine.compute_queue.backend.kind=='cuda_stream_completion','Real CUDA completion queue required')
        self.queue=engine.compute_queue
        self.destination=self.bindings.Destination(engine,controls[0],(transport_hosts or hosts)[0],
            trials={trial['trial_id']:trial for trial in self.trials})
        self.inputs={};self.records=[];self.manifest=[];self.resource_costs=[]
        self.background_calibration=None;self.resident={}
        self.paths={k:self.directory/name for k,name in (
            ('records','normalized-records.json'),('manifest','measurements-manifest.json'),
            ('inputs','frozen-input-arrays.json'),('calibration','background-calibration.json'),
            ('resources','actual-attempt-ownership.json'))}
        self.directory.mkdir(parents=True,exist_ok=True)

    @staticmethod
    def input_key(spec):
        return digest({k:spec[k] for k in ('role','prefix_key','prefix_tokens','suffix_tokens','input_id','input_family_id','seed')})

    def initialize(self):
        require(not any(path.exists() for path in self.paths.values()),'Existing E4 outputs; no implicit restart/retry')
        started=self.clock()
        for trial in self.trials:
            for spec in trial['requests']+trial['transfer_jobs']:
                key=self.input_key(spec)
                if key in self.inputs:continue
                case=dict(input_id=spec['input_id'],seed=spec['seed'],prefix=spec['prefix_tokens'],suffix=spec['suffix_tokens'])
                ids=self.bindings.materialize(case,self.tokenizer)
                require(len(ids)==spec['prefix_tokens']+spec['suffix_tokens'],'Frozen input geometry differs')
                self.inputs[key]=dict(token_ids=ids,token_ids_sha256=digest(ids),spec_identity=key)
        bgfamily=self.protocol['design']['inputs_repetition_and_order']['excluded_calibration']['family']
        background=self.bindings.materialize(dict(input_id=bgfamily,prefix=1024,suffix=16),self.tokenizer)
        save(self.paths['inputs'],dict(stage='E4',protocol_sha256=self.protocol['plan_sha256'],arrays=self.inputs,
            background_token_ids=background,background_token_ids_sha256=digest(background)))
        self.background_calibration=self.destination.initialize(background[:1024],background[1024:])
        require(self.background_calibration.get('CPU_component_only') is False,'CPU fixture cannot authorize GPU execution')
        save(self.paths['calibration'],self.background_calibration)
        self.report['initialization_elapsed_s']=self.clock()-started

    def role_inputs(self,trial):
        result={}
        for spec in trial['requests']+trial['transfer_jobs']:
            row=self.inputs[self.input_key(spec)];ids=row['token_ids']
            require(digest(ids)==row['token_ids_sha256'],'Frozen token bank changed')
            result[spec['role']]=dict(prefix_ids=ids[:spec['prefix_tokens']],suffix_ids=ids[spec['prefix_tokens']:])
        return result

    def run_trial(self,trial,*,diagnostic=False,measured=None):
        name='diagnostic' if diagnostic else 'candidate'
        path=self.directory/'trials'/trial['trial_id']/(name+'.json')
        require(not path.exists(),'Existing E4 attempt; no implicit replacement')
        start=self.clock();raw=None
        identity=dict(schema_version=1,stage='E4',trial_id=trial['trial_id'],trial=trial,diagnostic=diagnostic,
            artifact=str(path),source_run_id=self.report['run_id'],source_sha256=self.report['source_sha256'],
            context_sha256=self.report['context_sha256'],protocol_sha256=self.protocol['plan_sha256'],
            model_revision=self.report['model_revision'],model_manifest_sha256=self.report['model_manifest_sha256'],
            E4_complete=False)
        try:
            replay=None if measured is None else {role:copy.deepcopy(row['cutovers']) for role,row in measured['requests'].items()}
            raw,logits=self.destination.run_trial(trial['trial_id'],self.role_inputs(trial),
                diagnostic=diagnostic,replay_by_role=replay,timeout_s=self.plan['limits']['trial_timeout_s'])
            raw.update(identity)
            require(raw.get('passed') is True,'Actual E4 execution failed')
            if not diagnostic:require(all(not value for value in logits.values()),'Main captured full-vocabulary logits')
            return raw,logits
        except BaseException as error:
            raw=copy.deepcopy(getattr(error,'report',raw or {}));raw.update(identity)
            raw.update(driver_error=repr(error),driver_traceback=traceback.format_exc())
            if 'passed' not in raw:raw['passed']=False
            raise
        finally:
            if raw is not None:
                raw['driver_ownership_started_s']=start;raw['driver_ownership_finished_s']=self.clock()
                save(path,raw)

    def logits_witness(self,rows):
        result=[]
        for tensor in rows:
            value=tensor.detach().contiguous().view(self.engine.torch.uint8).cpu().numpy().tobytes()
            result.append(dict(shape=list(tensor.shape),dtype=str(tensor.dtype),bytes=len(value),sha256=hashlib.sha256(value).hexdigest()))
        return result

    def validate_candidate(self,trial,measured):
        diagnostic,logits=self.run_trial(trial,diagnostic=True,measured=measured)
        references={};torch=self.engine.torch;inputs=self.role_inputs(trial)
        try:
            require(set(logits)=={r['role'] for r in trial['requests']},'Exact per-role diagnostic logits required')
            for spec in trial['requests']:
                role=spec['role'];ids=inputs[role];row=diagnostic['requests'][role];main=measured['requests'][role]
                path=Path(measured['artifact']).parent/('reference-'+role+'.json')
                require(not path.exists(),'Existing HF reference; no reuse as new execution')
                ref=dict(stage='E4',trial_id=trial['trial_id'],role=role,passed=False,artifact=str(path),
                    source_run_id=self.report['run_id'],source_sha256=self.report['source_sha256'],
                    context_sha256=self.report['context_sha256'],model_revision=self.report['model_revision'],
                    model_manifest_sha256=self.report['model_manifest_sha256'],
                    protocol_sha256=self.protocol['plan_sha256'],inputs=copy.deepcopy(ids),
                    input_ids_sha256=digest(ids['prefix_ids']+ids['suffix_ids']),untimed_same_shape_reference=True,
                    started_s=self.clock())
                try:
                    require(main['cutovers']==row['cutovers'] and main['remote_calls']==row['remote_calls'] and
                            main['token_ids']==row['token_ids'],'Separate diagnostic changed actual route/tokens')
                    prefix=torch.tensor([ids['prefix_ids']],dtype=torch.int64,device=self.engine.device)
                    suffix=torch.tensor([ids['suffix_ids']],dtype=torch.int64,device=self.engine.device)
                    with self.queue.bind(trial['trial_id']+'.HF.'+role,deadline=self.clock()+self.plan['limits']['trial_timeout_s']):
                        actual=self.engine.HF_reference(prefix,suffix,64)
                    ref.update(token_ids=actual['token_ids'],logits=self.logits_witness(actual['logits']),state_report=actual['state_report'])
                    row['diagnostic_logits']=self.logits_witness(logits[role])
                    check=self.bindings.compare(torch,torch.cat(actual['logits'],dim=0),torch.cat(logits[role],dim=0),actual['token_ids'],row['token_ids'])
                    ref['numerical_check']=row['numerical_check']=main['numerical_check']=check
                    self.bindings.verify_outputs(torch,actual,row,logits[role],main)
                    require(actual['state_report']['owned_cache_references_released'] is True,'HF cache ownership leaked')
                    ref['passed']=True;row['correctness_passed']=main['correctness_passed']=True
                    del actual,prefix,suffix
                except BaseException as error:
                    ref.update(error=repr(error),traceback=traceback.format_exc());raise
                finally:
                    ref['finished_s']=self.clock();save(path,ref);references[role]=ref
            # Transfer content checks consume actual peer diagnostic reports in
            # the independent raw normalizer; no main tensor hash is invented.
            measured['correctness_passed']=diagnostic['correctness_passed']=True
            return diagnostic,references
        finally:
            save(measured['artifact'],measured);save(diagnostic['artifact'],diagnostic)
            for rows in logits.values():rows.clear()
            self.queue.clear_completed_events()

    def raw_artifacts(self,trial,measured,diagnostic,references):
        def item(raw):return None if raw is None else dict(path=raw['artifact'],sha256=sha(raw['artifact']))
        result=dict(candidate=item(measured),diagnostic=item(diagnostic),
            references={s['role']:item(references.get(s['role'])) for s in trial['requests']})
        if diagnostic is not None:
            result['diagnostic']['contents_sha256']={role:digest([layer['destination_hashes'] for layer in raw['layers']])
                for role,raw in diagnostic.get('transfers',{}).items() if raw.get('cache_content_equality_verified') is True}
        return result

    def append(self,trial,measured,diagnostic,references,*,error=None):
        require(not any(r['trial_id']==trial['trial_id'] for r in self.records),'Duplicate authoritative E4 attempt')
        artifacts=self.raw_artifacts(trial,measured,diagnostic,references)
        checker=self.bindings.raw_checks
        arguments=(trial,self.protocol,self.context,measured,diagnostic,references,artifacts,self.background_calibration)
        if error is None:record=checker.normalize(*arguments)
        else:record=checker.failed_record(*arguments,error=error)
        path=Path(measured['artifact']).parent/'normalized-record.json';save(path,record)
        self.records.append(record);self.manifest.append(dict(trial_id=trial['trial_id'],status=record['status'],
            raw_artifacts=artifacts,normalized_record_path=str(path),normalized_record_sha256=sha(path)))
        self.persist();return record

    def persist(self):
        save(self.paths['records'],dict(plan_sha256=self.protocol['plan_sha256'],protocol_sha256=self.protocol['plan_sha256'],records=self.records,E4_complete=False))
        save(self.paths['manifest'],dict(protocol_sha256=self.protocol['plan_sha256'],candidates=self.manifest,E4_complete=False))
        save(self.paths['resources'],dict(costs=self.resource_costs,scope='Actual full attempt ownership including preparation/main/diagnostic/HF; separate from request latency'))
        self.report['full_session']=dict(stage='E4',mode=self.plan['mode'],protocol_sha256=self.protocol['plan_sha256'],
            expected_candidates=len(self.trials),completed_candidates=sum(r['status']=='passed' for r in self.records),
            failed_candidates=sum(r['status']!='passed' for r in self.records),
            records_artifact=str(self.paths['records']),records_sha256=sha(self.paths['records']),
            measurements_manifest_artifact=str(self.paths['manifest']),measurements_manifest_sha256=sha(self.paths['manifest']),
            input_arrays_artifact=str(self.paths['inputs']),input_arrays_sha256=sha(self.paths['inputs']) if self.paths['inputs'].exists() else None,
            calibration_artifact=str(self.paths['calibration']),calibration_sha256=sha(self.paths['calibration']) if self.paths['calibration'].exists() else None,
            resource_costs_artifact=str(self.paths['resources']),resource_costs_sha256=sha(self.paths['resources']),
            pending_profiles=16,E4_complete=False)
        self.report['E4_complete']=self.report['paper_stage_complete']=False
        save(self.directory/'progress.json',self.report)

    def run(self):
        require(not self.paths['records'].exists(),'Existing E4 session; no implicit retry')
        original_error=None
        try:
            self.initialize();self.persist()
            for trial in self.trials:
                started=self.clock();measured=diagnostic=None;references={}
                try:
                    measured,_=self.run_trial(trial)
                    diagnostic,references=self.validate_candidate(trial,measured)
                    record=self.append(trial,measured,diagnostic,references)
                    require(record['status']=='passed','Independent E4 candidate raw gate failed')
                    self.resource_costs.append(dict(trial_id=trial['trial_id'],kind=trial['kind'],passed=True,
                        ownership_elapsed_s=self.clock()-started,maximum_actual_attempt_service_s=max(
                            raw['group']['group_service_settled_latency_s'] for raw in (measured,diagnostic))))
                    self.persist()
                except BaseException as error:
                    folder=self.directory/'trials'/trial['trial_id']
                    for name in ('candidate','diagnostic'):
                        path=folder/(name+'.json')
                        if path.exists():
                            value=json.loads(path.read_text())
                            if name=='candidate':measured=value
                            else:diagnostic=value
                    if measured is None:
                        measured=dict(artifact=str(folder/'candidate.json'),trial_id=trial['trial_id'],trial=trial,
                            passed=False,requests={},transfers={},error=repr(error));save(measured['artifact'],measured)
                    for spec in trial['requests']:
                        path=folder/('reference-'+spec['role']+'.json')
                        if path.exists():references[spec['role']]=json.loads(path.read_text())
                    if not any(r['trial_id']==trial['trial_id'] for r in self.records):
                        self.append(trial,measured,diagnostic,references,error=repr(error))
                    raise
            self.report['candidate_session_passed']=True;return self.manifest
        except BaseException as error:
            original_error=error;self.report['candidate_session_passed']=False;raise
        finally:
            close_error=None
            try:
                self.report['full_background_cleanup']=self.destination.close()
                require(self.report['full_background_cleanup'].get('passed') is True,'Actual E4 session shutdown failed')
            except BaseException as error:
                close_error=error;self.report['candidate_session_passed']=False
                self.report['session_shutdown_error']=repr(error)
            finally:
                if self.paths['records'].exists():self.persist()
            if close_error is not None and original_error is None:raise close_error
