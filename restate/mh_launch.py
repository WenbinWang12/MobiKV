"""Launch and settle one immutable MH session; signal only verified owned PIDs."""
from __future__ import annotations
import restate_config
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid

PKG = Path(__file__).resolve().parent
PYTHON = restate_config.PYTHON
MODEL = restate_config.MODEL
REVISION = restate_config.MODEL_REVISION
MANIFEST = restate_config.MODEL_MANIFEST_SHA256
PROBE = '''import json,socket,subprocess,pathlib
q=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()
a=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
m=subprocess.run(['pgrep','-f','^nvidia-cuda-mps-control -d'],capture_output=True).returncode==0
print(json.dumps(dict(hostname=socket.gethostname(),boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),dmi_uuid=pathlib.Path('/sys/class/dmi/id/product_uuid').read_text().strip(),gpus=[dict(zip(('index','uuid','name','total','used'),[x.strip() for x in l.split(',')])) for l in q],apps=a,mps=m)))
'''

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value, message):
    if not value: raise RuntimeError(message)
def read(path): return json.loads(Path(path).read_text())
def remote(ip, argv, timeout=60):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "root@" + ip,
                           shlex.join(argv)], capture_output=True, text=True, timeout=timeout)
def run_python(ip, source, timeout=60):
    argv=[PYTHON, "-c", source]
    out=remote(ip,argv,timeout) if ip else subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
    require(out.returncode==0,"Endpoint command failed %s: %s" % (ip,out.stderr[-2000:]))
    return json.loads(out.stdout)

def probe(ip, gpu):
    info=run_python(ip,PROBE)
    rows=[g for g in info['gpus'] if int(g['index'])==gpu]
    require(len(rows)==1,'GPU selector not unique'); row=rows[0]
    apps=[[x.strip() for x in l.split(',')] for l in info['apps'].splitlines() if l.strip()]
    mine=[x for x in apps if x[0]==row['uuid']]
    # Under the MPS cap (C3T) the idle MPS server stays resident on the GPU (~30 MiB) between sessions; it is not a workload.
    idle_mps=info['mps'] and len(mine)==1 and int(mine[0][2])<=64
    require((not mine or idle_mps) and int(row['used'])<=1024,'Selected GPU is not idle')
    require(int(row['total'])>=75000 and any(k in row['name'] for k in ('A100','A800')),'Expected A100/A800 80GB cohort')
    return dict(hostname=info['hostname'],boot_id=info['boot_id'],dmi_uuid=info['dmi_uuid'],
                gpu_uuid=row['uuid'],gpu_name=row['name'],gpu_index=gpu,gpu_memory_total_mib=int(row['total']))

def local_ip():
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
        sock.connect((restate_config.ROUTE_PROBE_IP,9)); return sock.getsockname()[0]

def validate_ranks(ranks, n_holders):
    require(n_holders==2 and len(ranks)==3,'Exactly two holders and one target required')
    for key in ('gpu_uuid','hostname','internal_ip','boot_id','dmi_uuid'):
        values=[r.get(key) for r in ranks]
        require(all(isinstance(v,str) and v for v in values) and len(set(values))==3,
                'Three distinct physical endpoints required: '+key)

def verify_rank_reports(result, context, context_path):
    hashes={}; ctx_sha=sha(context_path)
    for rank in context['ranks']:
        path=Path(result)/('rank-%d-%s.json' % (rank['rank'],context['session_id']))
        require(path.exists(),'Missing final rank report: '+str(path)); value=read(path)
        require(value.get('passed') is True and value.get('finished_utc'),'Rank did not finish successfully')
        for key, expected in [('rank',rank['rank']),('session_id',context['session_id']),('run_id',context['run_id']),
                              ('hostname',rank['hostname']),('gpu_uuid',rank['gpu_uuid']),
                              ('context_sha256',ctx_sha),('plan_sha256',context['plan_sha256'])]:
            require(value.get(key)==expected,'Rank terminal binding differs: '+key)
        require(not value.get('netem_cleanup_error'),'Network cleanup raised an error')
        if 'netem_cleanup' in value:
            cleanup=value['netem_cleanup']
            require(cleanup.get('restored') is True and not cleanup.get('errors'),'Network state not restored')
        if rank['role']=='holder': require(value.get('server',{}).get('passed') is True,'Holder cleanup did not pass')
        else:
            require(value.get('campaign_passed') is True and value.get('destination_close',{}).get('passed') is True,
                    'Target campaign or cleanup did not pass')
        hashes[path.name]=sha(path)
    return hashes

def owned_action(process, context_path, action='inspect'):
    """PID reuse or another MH_CONTEXT never grants permission to signal."""
    payload=dict(pid=process['pid'],start_ticks=process['start_ticks'],context=str(context_path),action=action)
    source='''import json,os,pathlib,signal
p=PAYLOAD
root=pathlib.Path('/proc')/str(p['pid'])
out=dict(pid=p['pid'],alive=False,owned=False,action=p['action'])
try:
 stat=(root/'stat').read_text().rsplit(')',1)[1].split()
 if stat[0]!='Z' and stat[19]==str(p['start_ticks']):
  env=dict(x.split(b'=',1) for x in (root/'environ').read_bytes().split(b'\\0') if b'=' in x)
  out['owned']=env.get(b'MH_CONTEXT')==p['context'].encode()
  out['alive']=out['owned']
  out['ownership_mismatch']=not out['owned']
  if out['owned'] and p['action'] in ('term','kill'):
   os.kill(p['pid'],signal.SIGTERM if p['action']=='term' else signal.SIGKILL)
   out['signal_sent']=True
except FileNotFoundError: pass
print(json.dumps(out))
'''.replace('PAYLOAD',repr(payload))
    return run_python(process.get('ip'),source,timeout=30)

def spawn_holder(row, context_path, log):
    receipt=Path(log).with_suffix('.spawn.json')
    payload=dict(context=str(context_path),rank=row['rank'],gpu_uuid=row['gpu_uuid'],log=str(log),
                 receipt=str(receipt),package=str(PKG),python=PYTHON)
    source='''import json,os,pathlib,subprocess,time
p=PAYLOAD
env=dict(os.environ,MH_CONTEXT=p['context'],RANK=str(p['rank']),CUDA_VISIBLE_DEVICES=p['gpu_uuid'],
 PYTHONPATH=p['package'],OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
with open(p['log'],'xb') as log:
 child=subprocess.Popen([p['python'],'-B',p['package']+'/mh_runner.py'],cwd=p['package'],env=env,
  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
root=pathlib.Path('/proc')/str(child.pid)
ticks=(root/'stat').read_text().rsplit(')',1)[1].split()[19]
receipt=dict(pid=child.pid,start_ticks=ticks)
temp=pathlib.Path(p['receipt']+'.tmp')
temp.write_text(json.dumps(receipt));os.replace(temp,p['receipt'])
for unused in range(100):
 if child.poll() is not None: raise RuntimeError('Holder exited during startup: '+str(child.returncode))
 env=(root/'environ').read_bytes().split(b'\\0')
 if ('MH_CONTEXT='+p['context']).encode() in env: break
 time.sleep(.05)
else: raise RuntimeError('Holder did not enter its frozen MH_CONTEXT')
print(json.dumps(receipt))
'''.replace('PAYLOAD',repr(payload))
    try:
        observed=run_python(row['internal_ip'],source,timeout=30)
    except BaseException as error:
        if not receipt.exists(): raise
        observed=read(receipt)
        observed['startup_error']=repr(error)
    return dict(rank=row['rank'],ip=row['internal_ip'],log=str(log),**observed)

def settle(processes, context_path, seconds, child=None):
    deadline=time.monotonic()+seconds
    while True:
        if child is not None: child.poll()
        states=[owned_action(p,context_path) for p in processes]
        if all(not s['alive'] for s in states): return states
        if time.monotonic()>=deadline: return states
        time.sleep(2)

def launch(args):
    from mh_pipeline import save, source_closure, validate_hash
    plan=read(args.plan); validate_hash(plan,'plan_sha256')
    result=args.result_dir.resolve(); result.mkdir(parents=True,exist_ok=True)
    session=datetime.datetime.now(datetime.timezone.utc).strftime('s%Y%m%dT%H%M%SZ-')+uuid.uuid4().hex[:8]
    status=dict(session=session,passed=False,all_owned_exited=False,cleanup_verified=False,holders=[],started=time.time())
    status_path=result/('launch-%s.json' % session)
    def persist():
        save(status_path,status); save(result/'latest-launch.json',status)
    processes=[]; child=None; ctx_path=None; context=None
    try:
        ranks=[]
        for i,spec in enumerate(args.holders):
            ip,gpu=spec.rsplit(':',1)
            ranks.append(dict(rank=i,role='holder',internal_ip=ip,**probe(ip,int(gpu))))
        ranks.append(dict(rank=len(ranks),role='target',internal_ip=local_ip(),**probe(None,args.target_gpu)))
        validate_ranks(ranks,plan['n_holders'])
        # A finite safety ceiling, not a claim about measured execution cost.
        timeout=args.stage_timeout or min(14*86400,max(3600,3600+len(plan['trials'])*plan['trial_timeout_s']*3))
        require(type(timeout) is int and 60<=timeout<=14*86400,'Invalid stage timeout')
        context=dict(schema_version=1,run_id=plan['run_id'],stage=plan['stage'],mode=plan['mode'],session_id=session,
            ranks=ranks,topology='three_physical_hosts',base_port=args.base_port,model_path=MODEL,model_id=restate_config.MODEL_ID,
            model_revision=REVISION,model_manifest_sha256=MANIFEST,plan_path=str(args.plan.resolve()),plan_sha256=sha(args.plan),
            result_dir=str(result),package=str(PKG),package_sha256=source_closure(),stage_timeout_s=timeout,
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        ctx_path=result/('context-%s.json'%session); save(ctx_path,context)
        status.update(context=str(ctx_path),stage_timeout_s=timeout);persist()
        for row in ranks[:-1]:
            status['startup_pending_rank']=row['rank'];persist()
            proc=spawn_holder(row,ctx_path,result/('worker-%d-%s.log'%(row['rank'],session)))
            processes.append(proc);status['holders'].append(proc);status.pop('startup_pending_rank',None);persist()
            require(not proc.get('startup_error'),'Holder startup command failed: '+str(proc.get('startup_error')))
            require(owned_action(proc,ctx_path)['alive'],'Holder failed verified startup')
        target=ranks[-1]
        env=dict(os.environ,MH_CONTEXT=str(ctx_path),PYTHONPATH=str(PKG),OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
                 RANK=str(target['rank']),CUDA_VISIBLE_DEVICES=target['gpu_uuid'])
        with (result/('worker-%d-%s.log'%(target['rank'],session))).open('xb') as output:
            child=subprocess.Popen([PYTHON,'-B',str(PKG/'mh_runner.py')],cwd=PKG,env=env,
                stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            ticks=Path('/proc/%d/stat'%child.pid).read_text().rsplit(')',1)[1].split()[19]
            processes.append(dict(rank=target['rank'],ip=None,pid=child.pid,start_ticks=ticks))
            status['target_pid']=child.pid;status['owned_processes']=processes;persist()
            status['target_exit']=child.wait(timeout=timeout)
        states=settle(processes,ctx_path,300,child)
        status['all_owned_exited']=all(not s['alive'] for s in states);status['exit_observations']=states
        require(status['target_exit']==0 and status['all_owned_exited'],'Target failed or a rank failed to exit')
        status['rank_report_sha256']=verify_rank_reports(result,context,ctx_path)
        status.update(passed=True,cleanup_verified=True)
    except BaseException as error:
        status.update(error=repr(error),traceback=traceback.format_exc(),passed=False)
        if processes and ctx_path is not None:
            status['failure_cleanup']=[]
            try:
                status['failure_cleanup'].extend(owned_action(p,ctx_path,'term') for p in processes)
                states=settle(processes,ctx_path,45,child)
                for p,s in zip(processes,states):
                    if s['alive']: status['failure_cleanup'].append(owned_action(p,ctx_path,'kill'))
                states=settle(processes,ctx_path,15,child)
                status['all_owned_exited']=all(not s['alive'] for s in states)
                status['exit_observations']=states
            except BaseException as cleanup_error:
                status.update(all_owned_exited=False,cleanup_error=repr(cleanup_error))
        elif not processes: status['all_owned_exited']='startup_pending_rank' not in status
        if 'startup_pending_rank' in status:
            # Without a PID/start-time receipt an SSH failure is ambiguous.
            # Preserve the uncertainty; never declare safe automatic retry.
            status['all_owned_exited']=False
    finally:
        if child is not None:
            status['target_exit']=child.poll()
        status['finished']=time.time();persist()
    print(json.dumps(status))
    return 0 if status['passed'] else 1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--result-dir',type=Path,required=True)
    ap.add_argument('--holders',nargs='+',required=True);ap.add_argument('--target-gpu',type=int,default=0)
    ap.add_argument('--base-port',type=int,default=33080);ap.add_argument('--stage-timeout',type=int)
    args=ap.parse_args()
    import fcntl
    args.result_dir.mkdir(parents=True,exist_ok=True)
    with (args.result_dir/'launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return launch(args)

if __name__=='__main__':
    def interrupted(signum,frame): raise RuntimeError('Launcher interrupted: '+str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP): signal.signal(sig,interrupted)
    sys.exit(main())
