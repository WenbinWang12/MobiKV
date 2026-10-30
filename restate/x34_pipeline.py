"""Immutable X34/X5 pipeline: calibrated costs, verified rank cleanup, analysis."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

PKG=Path(__file__).resolve().parent
sys.path.insert(0,str(PKG))
import mh_pipeline as pipeline
import mh_plan


def freeze_prereg(source, destination):
    source, destination=Path(source),Path(destination)
    raw=source.read_bytes()
    if destination.exists():
        pipeline.require(destination.read_bytes()==raw,'Existing preregistration differs from requested file; refusing silent reuse')
    else:
        with destination.open('xb') as stream: stream.write(raw)
    return pipeline.sha(destination)


def run(args):
    root=args.root.resolve();stages=args.stages.split(',')
    pipeline.require(stages and len(set(stages))==len(stages) and set(stages)<={'cal','x34','x5'},'Invalid stage selection')
    pipeline.require(args.rates and len(set(args.rates))==len(args.rates) and all(r>0 for r in args.rates),'Invalid rates')
    pipeline.require(args.x5_rate in args.rates,'X5 rate must be calibrated')
    with pipeline.pipeline_lock(root):
        prereg=root/'preregistration.md'
        prereg_sha=freeze_prereg(args.prereg,prereg)
        base=pipeline.read(args.base_model);pipeline.validate_hash(base,'model_sha256')
        inputs=dict(preregistration_sha256=prereg_sha,base_model_path=str(args.base_model.resolve()),
            base_model_file_sha256=pipeline.sha(args.base_model),base_model_sha256=base['model_sha256'],
            rates=args.rates,x5_rate=args.x5_rate,x5_cell=args.x5_cell,rtts=args.rtts,
            holders=args.holders,target_gpu=args.target_gpu,source_sha256=pipeline.source_closure(),
            scientific_acceptance=False,coverage_claim_allowed=False)
        pipeline.immutable_json(root/'frozen-input-bindings.json',inputs)
        tag=root.name;model_path=root/'cost-model.json'
        calplan=mh_plan.build_x34_calibration(tag+'-CAL',args.rates)
        if 'cal' in stages:
            records=pipeline.run_stage(root,'CAL',calplan,args.holders,args.target_gpu)
        else:
            pipeline.require(model_path.exists(),'No model exists; calibration stage required')
            pipeline.require(pipeline.read(root/'CAL/plan.json')==calplan,'Existing calibration plan differs')
            records=pipeline.session_acceptance(root/'CAL',calplan)
        # Always reproduce the model from bound CAL records and the frozen base,
        # including on resume. Existing model bytes cannot silently skip CAL.
        model=mh_plan.fit_model_x34(records,base,args.rates)
        pipeline.validate_hash(model,'model_sha256')
        pipeline.require(model['base_model_sha256']==base['model_sha256'],'Derived model base differs')
        pipeline.immutable_json(model_path,model)
        binding=dict(preregistration_sha256=prereg_sha,base_model_file_sha256=inputs['base_model_file_sha256'],
            base_model_sha256=base['model_sha256'],calibration_plan_sha256=calplan['plan_sha256'],
            calibration_records_sha256=pipeline.sha(root/'CAL/normalized-records.json'),
            calibration_acceptance_sha256=pipeline.sha(root/'CAL/stage-acceptance.json'),
            model_sha256=model['model_sha256'],model_file_sha256=pipeline.sha(model_path),source_sha256=pipeline.source_closure())
        pipeline.immutable_json(root/'frozen-model-bindings.json',binding)
        pipeline.log(root,stage='fit',event='model_saved',model_sha256=model['model_sha256'],
            base_model_sha256=base['model_sha256'],calibration_plan_sha256=calplan['plan_sha256'],b_star_bps=model['b_star_bps'])
        for selected,builder,name in (('x34',mh_plan.build_x34,'X34-full'),('x5',mh_plan.build_x5,'X5-full')):
            if selected not in stages: continue
            pipeline.require(pipeline.sha(prereg)==prereg_sha and pipeline.sha(args.base_model)==inputs['base_model_file_sha256'],
                             'Frozen input bytes changed before stage')
            pipeline.require(pipeline.source_closure()==inputs['source_sha256'],'Frozen package changed before stage')
            plan=(builder(tag+'-'+name,model,'full',args.rates,prereg_sha) if selected=='x34'
                  else builder(tag+'-'+name,model,'full',args.x5_rate,args.x5_cell,args.rtts,prereg_sha))
            pipeline.log(root,stage=name+'-plan',event='frozen',plan_sha256=plan['plan_sha256'],trials=len(plan['trials']),
                         preregistration_sha256=prereg_sha,model_binding_sha256=pipeline.sha(root/'frozen-model-bindings.json'))
            pipeline.run_stage(root,name,plan,args.holders,args.target_gpu)
        if any(s in stages for s in ('x34','x5')):
            script=PKG/'x34_analysis.py'
            pipeline.require(script.is_file(),'Required analysis program missing')
            output=subprocess.run([sys.executable,'-B',str(script),'--root',str(root)],cwd=str(PKG),
                                  capture_output=True,text=True,timeout=3600)
            pipeline.log(root,stage='analysis',event='done' if output.returncode==0 else 'failed',
                         exit=output.returncode,stdout=output.stdout[-2000:],stderr=output.stderr[-2000:])
            pipeline.require(output.returncode==0,'Analysis process failed; pipeline cannot be complete')
            summary=pipeline.read(root/'analysis/summary.json')
            for selected,name in (('x34','X34'),('x5','X5')):
                if selected in stages:
                    row=summary.get(name)
                    pipeline.require(isinstance(row,dict) and row.get('missing_or_failed')==[],
                                     'Analysis missing requested results or reports missing/failed trials: '+name)
                    pipeline.require(row.get('plan_sha256')==pipeline.read(root/(name+'-full')/'plan.json')['plan_sha256'],
                                     'Analysis plan binding differs: '+name)
            pipeline.log(root,stage='analysis',event='verified',summary_sha256=pipeline.sha(root/'analysis/summary.json'))
        pipeline.log(root,stage='pipeline',event='complete',stages=stages,scientific_acceptance=False,coverage_claim_allowed=False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--holders',nargs='+',required=True)
    ap.add_argument('--target-gpu',type=int,default=0);ap.add_argument('--base-model',type=Path,required=True)
    ap.add_argument('--rates',type=int,nargs='+',required=True);ap.add_argument('--x5-rate',type=int,required=True)
    ap.add_argument('--x5-cell',default='fast-high');ap.add_argument('--rtts',type=float,nargs='+',default=[0.,1.,2.,5.,10.])
    ap.add_argument('--prereg',type=Path,required=True);ap.add_argument('--stages',default='cal,x34,x5')
    run(ap.parse_args())

if __name__=='__main__': main()
