"""E4 real immutable KV transfer, no decode and no experiment launcher.

The caller supplies two connected TCP sockets and the SAME outgoing coordinator
used by all other jobs on this host. This module never creates/closes a budget.
Main transfers do no tensor hashing. Separate diagnostics hash actual source KV
bytes and target GPU readback, reporting the extra DtoH copies separately.

Only CudaKVBackend is GPU execution evidence. CPUByteBackend is an explicitly
excluded socket/ownership component fixture and can never certify GPU work.
All timestamps are native endpoint-local monotonic readings, never subtracted
across hosts. A successful receiver publishes every complete GPU-ready layer,
then releases its owned prefix before settlement; the sender borrows its prefix.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import socket
import threading
import time
import traceback
import weakref

try:
    from e1_wire import FramedConnection, TensorBuffer, DTYPE_BYTES
    from e0_layered_runtime import PrefixStore
except ModuleNotFoundError as error:
    if error.name not in ('e1_wire','e0_layered_runtime'):raise
    from restate_direct.e3_full.runtime.e1_wire import FramedConnection, TensorBuffer, DTYPE_BYTES
    from restate_direct.e3_full.runtime.e0_layered_runtime import PrefixStore


PROTOCOL='restate-e4-transfer-only-v1'


def require(condition,message):
    if not condition:raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def finite(value,name):
    require(type(value) in (int,float) and math.isfinite(value) and value>=0,'Invalid '+name)
    return value


def _reference_counts(references):
    """Object ownership only; CPU unit checks do not certify CUDA allocation."""
    return len(references),sum(reference() is not None for reference in references)


def make_spec(request_id,prefix_key,prefix_tokens,*,diagnostic=False,cpu_component=False,
              layers=48,kv_heads=8,head_dim=128,dtype='bfloat16'):
    require(isinstance(request_id,str) and request_id and len(request_id)<=1024,'Request identity required')
    require(isinstance(prefix_key,str) and prefix_key and len(prefix_key)<=1024,'Immutable prefix identity required')
    require(type(diagnostic) is type(cpu_component) is bool,'Explicit diagnostic/component modes required')
    require(all(type(n) is int and n>0 for n in (prefix_tokens,layers,kv_heads,head_dim)) and layers<=48,
            'Bounded positive cache geometry required')
    require(dtype in ('bfloat16','float16','float32','uint8'),'Unsupported cache dtype')
    if not cpu_component:
        require((layers,kv_heads,head_dim,dtype)==(48,8,128,'bfloat16') and prefix_tokens in (1024,4096),
                'GPU transfer geometry must be the frozen E4 main14B BF16 prefix')
    shape=[1,kv_heads,prefix_tokens,head_dim];layer_bytes=2*math.prod(shape)*DTYPE_BYTES[dtype]
    require(layer_bytes<=256*1024*1024,'One complete KV layer exceeds wire frame limit')
    result={'protocol':PROTOCOL,'request_id':request_id,'prefix_key':prefix_key,'prefix_tokens':prefix_tokens,
        'layers':layers,'shape':shape,'dtype':dtype,'diagnostic':diagnostic,'cpu_component_only':cpu_component,
        'expected_layer_payload_bytes':layer_bytes,'expected_total_payload_bytes':layers*layer_bytes,
        'action':'transfer_only','decode_tokens':0}
    return dict(result,spec_sha256=digest(result))


def validate_spec(spec):
    require(isinstance(spec,dict),'Transfer spec must be an object')
    same=make_spec(spec['request_id'],spec['prefix_key'],spec['prefix_tokens'],diagnostic=spec['diagnostic'],
        cpu_component=spec['cpu_component_only'],layers=spec['layers'],kv_heads=spec['shape'][1],
        head_dim=spec['shape'][3],dtype=spec['dtype'])
    require(spec==same,'Transfer spec or digest changed')
    return copy.deepcopy(spec)


def _buffer_check(tensor,spec):
    require(isinstance(tensor,TensorBuffer) and tensor.dtype==spec['dtype']
            and list(tensor.shape)==spec['shape'] and len(tensor.data)*2==spec['expected_layer_payload_bytes'],
            'Received tensor shape/dtype/bytes differ from the bound prefix')


def _buffer_hash(tensor):
    return hashlib.sha256(tensor.data).hexdigest()


def _tensor_hash(torch,tensor):
    require(tensor.device.type=='cpu','Only real CPU snapshots may be hashed')
    raw=tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


class CPUByteBackend:
    """Owned byte buffers for CPU/socket tests; never masquerades as CUDA."""
    kind='cpu_byte_component_only'
    gpu_evidence=False

    def settle_owned_copies(self):
        return {'pending_copy_stream_settled':True,'GPU_copy_stream':False}

    def validate_pairs(self,pairs,spec):
        require(spec['cpu_component_only'] is True,'CPU payloads cannot replace GPU KV')
        require(len(pairs)==spec['layers'],'Source layer count differs')
        for pair in pairs:
            require(len(pair)==2,'Source layer must contain K and V')
            for item in pair:_buffer_check(item,spec)

    def to_host(self,pair,spec):
        start=time.monotonic()
        tensors=[TensorBuffer(t.dtype,t.shape,bytearray(t.data)) for t in pair]
        hashes=[_buffer_hash(t) for t in tensors] if spec['diagnostic'] else None
        return tensors,hashes,{'started_local_s':start,'ready_local_s':time.monotonic(),
            'dtype':spec['dtype'],'shape':spec['shape'],'copies':0,'bytes':0,
            'backend':self.kind,'scope':'actual CPU byte snapshot, no GPU copy'}

    def to_destination(self,tensors,spec):
        start=time.monotonic()
        for tensor in tensors:_buffer_check(tensor,spec)
        result=[TensorBuffer(t.dtype,t.shape,bytearray(t.data)) for t in tensors]
        hashes=[_buffer_hash(t) for t in result] if spec['diagnostic'] else None
        return result,hashes,{'started_local_s':start,'ready_local_s':time.monotonic(),
            'dtype':spec['dtype'],'shape':spec['shape'],'copies':0,'bytes':0,
            'verification_d2h_copies':0,'verification_d2h_bytes':0,'backend':self.kind,
            'scope':'actual CPU byte ownership, no GPU copy/readiness evidence'}


class CudaKVBackend:
    """Concrete real GPU copies; no injected timing/readiness/hash callbacks."""
    kind='cuda_stream_synchronized'
    gpu_evidence=True

    def __init__(self,torch,device):
        self.torch=torch;self.device=torch.device(device)
        require(self.device.type=='cuda' and torch.cuda.is_available(),'Actual selected CUDA device required')
        if self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
        self.stream=None
        # Includes an allocated target even when its copy/hash fails before
        # return/publication. Weak references never prolong tensor ownership.
        self.destination_tensor_refs=[]

    def _stream(self):
        if self.stream is None:self.stream=self.torch.cuda.Stream(device=self.device)
        return self.stream

    def settle_owned_copies(self):
        # An exception may occur after enqueuing a copy but before the usual
        # readiness sync. Keep pinned/device buffers alive until this finishes.
        start=time.monotonic()
        if self.stream is not None:self.stream.synchronize()
        return {'pending_copy_stream_settled':True,'GPU_copy_stream':self.stream is not None,
                'started_local_s':start,'finished_local_s':time.monotonic()}

    def validate_pairs(self,pairs,spec):
        require(spec['cpu_component_only'] is False,'Do not mix component and GPU evidence')
        require(len(pairs)==spec['layers'],'Source GPU layer count differs')
        for pair in pairs:
            require(len(pair)==2,'Source layer must contain K and V')
            for tensor in pair:
                require(self.torch.is_tensor(tensor) and tensor.device==self.device
                    and list(tensor.shape)==spec['shape'] and str(tensor.dtype)=='torch.'+spec['dtype'],
                    'Source must be real selected-device immutable prefix KV')

    def to_host(self,pair,spec):
        torch=self.torch;start=time.monotonic();tensors=[]
        with torch.cuda.device(self.device),torch.inference_mode():
            stream=self._stream()
            with torch.cuda.stream(stream):
                for tensor in pair:
                    cpu=torch.empty(tensor.shape,dtype=tensor.dtype,device='cpu',pin_memory=True)
                    cpu.copy_(tensor,non_blocking=True);tensors.append(cpu)
            stream.synchronize()
            ready=time.monotonic()
            hashes=[_tensor_hash(torch,t) for t in tensors] if spec['diagnostic'] else None
        return tensors,hashes,{'started_local_s':start,'ready_local_s':ready,
            'dtype':spec['dtype'],'shape':spec['shape'],'copies':2,'bytes':spec['expected_layer_payload_bytes'],
            'backend':self.kind,'scope':'real GPU-to-pinned-CPU copies, native completion-bounded wall interval'}

    def to_destination(self,tensors,spec):
        torch=self.torch;start=time.monotonic();gpu=[];pinned=[];verification=[]
        with torch.cuda.device(self.device),torch.inference_mode():
            stream=self._stream();dtype=getattr(torch,spec['dtype'])
            for tensor in tensors:
                _buffer_check(tensor,spec)
                source=torch.frombuffer(tensor.data,dtype=dtype).reshape(spec['shape'])
                cpu=torch.empty(spec['shape'],dtype=dtype,device='cpu',pin_memory=True)
                cpu.copy_(source);pinned.append(cpu)
            with torch.cuda.stream(stream):
                for cpu in pinned:
                    target=torch.empty(spec['shape'],dtype=dtype,device=self.device)
                    self.destination_tensor_refs.append(weakref.ref(target))
                    target.copy_(cpu,non_blocking=True);gpu.append(target)
            stream.synchronize();ready=time.monotonic()
            if spec['diagnostic']:
                with torch.cuda.stream(stream):
                    for target in gpu:
                        cpu=torch.empty(spec['shape'],dtype=dtype,device='cpu',pin_memory=True)
                        cpu.copy_(target,non_blocking=True);verification.append(cpu)
                stream.synchronize()
                hashes=[_tensor_hash(torch,t) for t in verification]
            else:hashes=None
            checked=time.monotonic()
        return gpu,hashes,{'started_local_s':start,'ready_local_s':ready,'verification_finished_local_s':checked,
            'dtype':spec['dtype'],'shape':spec['shape'],'copies':2,'bytes':spec['expected_layer_payload_bytes'],
            'verification_d2h_copies':2 if spec['diagnostic'] else 0,
            'verification_d2h_bytes':spec['expected_layer_payload_bytes'] if spec['diagnostic'] else 0,
            'backend':self.kind,'scope':'real pinned-CPU-to-GPU readiness; optional separate diagnostic GPU readback'}


class TransferError(RuntimeError):
    def __init__(self,message,report):
        super().__init__(message);self.report=report


def _zero_frames():
    return {'sent':{'frames':0,'application_frame_bytes':0,'application_payload_bytes':0},
            'received':{'frames':0,'application_frame_bytes':0,'application_payload_bytes':0}}


class TransferEndpoint:
    """Single-use synchronous worker; group driver owns its worker thread.

    run() returns the complete small endpoint report, or raises TransferError
    carrying that endpoint's failed/partial report after owned cleanup. The group
    can call cancel(reason) from another thread; it closes ONLY this endpoint's
    two sockets. Group must still join its actual run thread before reuse.
    Construction/wrap and run belong after the actual local acceptance barrier.
    Caller owns outer group control, coordinator, source prefix, network rules.
    """
    def __init__(self,role,rpc_socket,bulk_socket,coordinator,spec,backend,*,
                 accepted_local_s,deadline_local_s):
        require(role in ('source','destination'),'Unknown transfer endpoint role')
        self.spec=validate_spec(spec);self.role=role;self.backend=backend;self.coordinator=coordinator
        require(type(backend) in (CudaKVBackend,CPUByteBackend),'Only reviewed concrete GPU/CPU backends are supported')
        require(backend.gpu_evidence is (not spec['cpu_component_only']),'Backend/scope mismatch')
        require(callable(getattr(coordinator,'wrap',None)) and callable(getattr(coordinator,'snapshot',None)),
                'Caller must supply the actual shared outgoing coordinator')
        now=time.monotonic();self.accepted=finite(accepted_local_s,'local acceptance');self.deadline=finite(deadline_local_s,'local deadline')
        require(self.accepted<=now<self.deadline and self.deadline-self.accepted<=1200,'Finite live <=1200s local transfer deadline required')
        require(rpc_socket is not bulk_socket,'Independent RPC and bulk sockets required')
        self.lock=threading.Lock();self.cancelled=threading.Event();self.finished=threading.Event()
        self.cancel_reason=None;self.started=False;self.channels={};self.wrapped={};self.raw={'rpc':rpc_socket,'bulk':bulk_socket}
        self.store=None;self.watchdog=None
        self.report={'schema_version':1,'protocol':PROTOCOL,'role':role,'spec':copy.deepcopy(self.spec),
            'request_id':spec['request_id'],'passed':False,'status':'not_started','GPU_execution_evidence':backend.gpu_evidence,
            'CPU_component_only':not backend.gpu_evidence,'diagnostic':spec['diagnostic'],'decode_tokens':0,
            'accepted_local_s':self.accepted,'deadline_local_s':self.deadline,'clock_scope':'native endpoint monotonic only; no cross-host subtraction',
            'physical_wire_bytes':None,'layers':[],'frames':{name:_zero_frames() for name in self.raw},
            'source_prefix_borrowed':role=='source','source_prefix_released_by_component':False,
            'owned_destination_cache_released':role=='source','all_layers_ready_before_release':False,
            'cache_content_equality_verified':False,'global_coordinator_owned':False,
            'outgoing_streams':{name:{'stream_id':spec['request_id']+'/transfer-'+name,'kind':name} for name in self.raw},
            'coordinator_local_object_id':id(coordinator),'limiter_local_object_id':id(coordinator.limiter),
            'coordinator_config':copy.deepcopy(getattr(coordinator,'config',None)),
            'coordinator_identity_scope':'native process object identity; compare only within the same host/process group'}
        try:
            for name,raw in self.raw.items():
                raw.settimeout(self.deadline-now);raw.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                # No per-flow limiter is ever allocated here.
                stream_id=spec['request_id']+'/transfer-'+name
                wrapped=coordinator.wrap(raw,stream_id,name);self.wrapped[name]=wrapped
                self.channels[name]=FramedConnection(wrapped,integrity_mode='sha256' if spec['diagnostic'] else 'none')
        except BaseException:
            self._close_owned();raise

    def _close_owned(self):
        # shutdown, unlike close alone, wakes a peer/native blocking recv.
        for raw in self.raw.values():
            try:raw.shutdown(socket.SHUT_RDWR)
            except OSError:pass
        for channel in self.channels.values():
            try:channel.close()
            except OSError:pass
        for raw in self.raw.values():
            try:raw.close()
            except OSError:pass

    def cancel(self,reason='group_cancelled'):
        require(isinstance(reason,str) and reason,'Cancellation reason required')
        with self.lock:
            if self.finished.is_set():return False
            if self.cancel_reason is None:self.cancel_reason=reason
            self.cancelled.set()
        self._close_owned();return True

    def _check(self):
        if self.cancelled.is_set():raise RuntimeError('Transfer cancelled: '+str(self.cancel_reason))
        if time.monotonic()>=self.deadline:
            self.cancel('local_deadline_expired');raise TimeoutError('Transfer local deadline expired')

    def _watch(self):
        if not self.finished.wait(max(0.,self.deadline-time.monotonic())):
            self.cancel('local_deadline_expired')

    def _count(self,name,direction,stats):
        target=self.report['frames'][name][direction];target['frames']+=1
        for key in ('application_frame_bytes','application_payload_bytes'):target[key]+=stats[key]

    def _send(self,name,kind,tensors=(),**fields):
        self._check();started=time.monotonic()
        stats=self.channels[name].send_message(dict(kind=kind,request_id=self.spec['request_id'],
            spec_sha256=self.spec['spec_sha256'],**fields),tensors)
        self._count(name,'sent',stats)
        return {'started_local_s':started,'finished_local_s':time.monotonic(),**stats}

    def _recv(self,name,kind):
        self._check();started=time.monotonic()
        message,tensors,stats=self.channels[name].recv_message(tensor_format='bytes')
        self._count(name,'received',stats)
        require(message.get('kind')==kind and message.get('request_id')==self.spec['request_id']
                and message.get('spec_sha256')==self.spec['spec_sha256'],'Foreign request, spec or transfer protocol message')
        return message,tensors,{'started_local_s':started,'finished_local_s':time.monotonic(),**stats}

    def _control(self,name,kind):
        message,tensors,stats=self._recv(name,kind);require(not tensors,'Control message carried unexpected tensors')
        return message,stats

    def _handshake(self):
        for name in ('rpc','bulk'):
            if self.role=='source':
                message,stats=self._control(name,'transfer_'+name+'_hello')
                require(message['backend']==self.backend.kind,'Peer backend/evidence scope differs')
                self._send(name,'transfer_'+name+'_ready',backend=self.backend.kind)
            else:
                self._send(name,'transfer_'+name+'_hello',backend=self.backend.kind)
                message,stats=self._control(name,'transfer_'+name+'_ready')
                require(message['backend']==self.backend.kind,'Peer backend/evidence scope differs')
        self.report['handshake_complete_local_s']=time.monotonic()

    def _source(self,prefix_pairs):
        self.backend.validate_pairs(prefix_pairs,self.spec)
        self._handshake();hashes=[]
        for layer,pair in enumerate(prefix_pairs):
            self._check();entry={'layer':layer,'status':'copying','source_hashes':None};self.report['layers'].append(entry)
            tensors,hashes_at_layer,copied=self.backend.to_host(pair,self.spec)
            entry.update(copy=copied,status='frame_ready',source_hashes=hashes_at_layer)
            entry['send']=self._send('bulk','transfer_prefix',tensors,layer=layer,payload_bytes=self.spec['expected_layer_payload_bytes'],
                                    source_hashes=hashes_at_layer)
            entry['status']='frame_sent';hashes.append(hashes_at_layer)
            del tensors
        self.report['all_frames_sent_local_s']=time.monotonic()
        terminal=dict(complete_layers=self.spec['layers'],payload_bytes=self.spec['expected_total_payload_bytes'],
                      layer_hashes=hashes if self.spec['diagnostic'] else None)
        self.report['terminal_send']=self._send('bulk','transfer_terminal',**terminal)
        drained,stats=self._control('rpc','transfer_drained');self.report['receiver_drained_ACK']=drained
        require(drained['complete_layers']==terminal['complete_layers'] and drained['payload_bytes']==terminal['payload_bytes'],
                'Receiver did not acknowledge every complete layer/payload')
        require(drained['all_layers_ready'] is True and drained['owned_cache_released'] is True,'Receiver readiness/owned release missing')
        require(drained['destination_hashes']==terminal['layer_hashes'],'Receiver actual cache content hashes differ')
        self.report['cache_content_equality_verified']=self.spec['diagnostic']
        self.report['receiver_drain_received_local_s']=stats['finished_local_s']
        self.report['finish_send']=self._send('rpc','transfer_finished',complete_layers=terminal['complete_layers'],
            payload_bytes=terminal['payload_bytes'],content_equality_verified=self.spec['diagnostic'])

    def _destination(self):
        self.store=PrefixStore(self.spec['layers']);self._handshake();hashes=[]
        for layer in range(self.spec['layers']):
            message,tensors,received=self._recv('bulk','transfer_prefix')
            require(message['layer']==layer and len(tensors)==2 and message['payload_bytes']==self.spec['expected_layer_payload_bytes'],
                    'Missing, repeated, partial or foreign prefix layer')
            for tensor in tensors:_buffer_check(tensor,self.spec)
            entry={'layer':layer,'status':'frame_received','receive':received,'source_hashes':message['source_hashes']}
            self.report['layers'].append(entry);self._check()
            values,destination_hashes,copied=self.backend.to_destination(tensors,self.spec)
            entry.update(copy=copied,destination_hashes=destination_hashes)
            if self.spec['diagnostic']:
                require(message['source_hashes']==[_buffer_hash(t) for t in tensors]==destination_hashes,
                        'Actual source/wire/target cache content differs')
            else:require(message['source_hashes'] is None and destination_hashes is None,'Main path unexpectedly hashed tensors')
            self._check();self.store.publish(layer,*values,self.spec['expected_layer_payload_bytes'])
            entry.update(status='published',published_local_s=time.monotonic())
            require(entry['published_local_s']>=copied['ready_local_s'],'Cache published before actual device readiness')
            hashes.append(destination_hashes);del values,tensors
        self.report['all_layers_ready_local_s']=time.monotonic()
        self.report['all_layers_ready_before_release']=True
        terminal,timing=self._control('bulk','transfer_terminal');self.report['received_terminal']=terminal
        require(terminal['complete_layers']==self.spec['layers'] and terminal['payload_bytes']==self.spec['expected_total_payload_bytes']
                and terminal['layer_hashes']==(hashes if self.spec['diagnostic'] else None),'Source complete terminal/hash manifest differs')
        self.report['cache_content_equality_verified']=self.spec['diagnostic']
        require(self._release_cache(),'Destination tensor references survived owned cache release')
        self.report['drain_send']=self._send('rpc','transfer_drained',complete_layers=self.spec['layers'],
            payload_bytes=self.spec['expected_total_payload_bytes'],all_layers_ready=True,owned_cache_released=True,
            destination_hashes=hashes if self.spec['diagnostic'] else None)
        finished,timing=self._control('rpc','transfer_finished');self.report['source_finish_ACK']=finished
        self.report['source_finish_receive']=timing
        require(finished['complete_layers']==self.spec['layers'] and finished['payload_bytes']==self.spec['expected_total_payload_bytes']
                and finished['content_equality_verified']==self.spec['diagnostic'],'Source final receipt differs')

    def _release_cache(self):
        if self.store is not None:
            frontier,commits=self.store.snapshot()
            self.report['published_frontier_before_release']=frontier
            self.report['publication_records']=commits
            with self.store.lock:self.store.values.clear()
            self.store=None
        if self.role=='destination':
            gpu=self.backend.gpu_evidence
            refs=self.backend.destination_tensor_refs if gpu else []
            count,live=_reference_counts(refs)
            released=live==0
            self.report.update(owned_destination_cache_released=released,
                destination_GPU_weakref_check_performed=gpu,
                destination_GPU_allocations_tracked=count if gpu else None,
                destination_GPU_references_remaining=live if gpu else None,
                destination_release_evidence_scope=('weak references to every actual destination CUDA allocation'
                    if gpu else 'CPU component store clear only; not GPU reference-release evidence'))
            if released:self.report.setdefault('cache_released_local_s',time.monotonic())
            return released
        return True

    def run(self,prefix_pairs=None):
        require(not self.started,'Transfer endpoint is single-use');self.started=True
        self.report.update(status='running',executor_entered_local_s=time.monotonic())
        self.watchdog=threading.Thread(target=self._watch,name='e4-transfer-deadline-'+self.role,daemon=True)
        self.watchdog.start();error=None
        try:
            require((prefix_pairs is not None)==(self.role=='source'),'Only source supplies borrowed immutable GPU prefix')
            self._check()
            if self.role=='source':self._source(prefix_pairs)
            else:self._destination()
            self.report.update(passed=True,status='completed',protocol_completed=True)
        except BaseException as exc:
            error=exc;self.report.update(passed=False,status='cancelled' if self.cancelled.is_set() else 'failed',
                error=type(exc).__name__+': '+str(exc),protocol_completed=False)
        finally:
            self._close_owned()
            copy_cleanup_error=None
            try:self.report['copy_stream_cleanup']=self.backend.settle_owned_copies()
            except BaseException as exc:
                copy_cleanup_error=type(exc).__name__+': '+str(exc)
                self.report['copy_stream_cleanup']={'pending_copy_stream_settled':False,'error':copy_cleanup_error}
            # Clear ended call frames before claiming owned release/settlement;
            # otherwise a failed pending layer could survive in a traceback.
            prefix_pairs=None
            if error is not None:
                traceback.clear_frames(error.__traceback__)
                error.__traceback__=None
            cache_cleanup_error=None
            try:
                if not self._release_cache():cache_cleanup_error='Destination GPU allocations remain referenced'
            except BaseException as exc:
                cache_cleanup_error=type(exc).__name__+': '+str(exc)
                self.report['owned_destination_cache_released']=False
            self.report['cache_cleanup_error']=cache_cleanup_error
            self.finished.set();self.watchdog.join(timeout=1)
            self.report['cancel_reason']=self.cancel_reason
            self.report['socket_accounting']={name:wrapped.snapshot() for name,wrapped in self.wrapped.items()}
            self.report['complete_frame_accounting_matches_actual_socket_bytes']=all(
                self.report['socket_accounting'][name][direction+'_bytes']==self.report['frames'][name][direction]['application_frame_bytes']
                for name in self.wrapped for direction in ('sent','received'))
            self.report['cleanup']={'sockets_closed':all(raw.fileno()==-1 for raw in self.raw.values()),
                'watchdog_exited':not self.watchdog.is_alive(),'owned_destination_cache_released':self.report['owned_destination_cache_released'],
                'pending_copy_stream_settled':copy_cleanup_error is None,
                'failed_call_tensor_references_cleared':True,
                'global_budget_closed_by_component':False,'caller_must_join_run_thread':True}
            self.report['settled_local_s']=time.monotonic()
            self.report['settled_latency_s']=self.report['settled_local_s']-self.accepted
            if self.report['passed'] and (self.cancelled.is_set() or self.report['settled_local_s']>self.deadline):
                self.report.update(passed=False,status='cancelled' if self.cancelled.is_set() else 'failed',
                    error='Transfer cancelled or local deadline exceeded before complete settlement')
                error=RuntimeError(self.report['error'])
            self.report['completed_layers']=sum(e['status']==('frame_sent' if self.role=='source' else 'published') for e in self.report['layers'])
            self.report['unfinished_layers']=self.spec['layers']-self.report['completed_layers']
            self.report['copy_counts']={name:sum(e.get('copy',{}).get(field,0) for e in self.report['layers']) for name,field in
                (('migration_copies','copies'),('migration_bytes','bytes'),('diagnostic_verification_d2h_copies','verification_d2h_copies'),
                 ('diagnostic_verification_d2h_bytes','verification_d2h_bytes'))}
            self.report['copy_counts_complete']=not any('copy' not in entry for entry in self.report['layers'])
            self.report['copy_counter_scope']='completed backend copy calls; failed in-progress copy counts may be unknown'
            self.report['partial_frame_application_bytes']={name:{direction:max(0,
                self.report['socket_accounting'][name][direction+'_bytes']-
                self.report['frames'][name][direction]['application_frame_bytes']) for direction in ('sent','received')}
                for name in self.wrapped}
            if self.report['passed'] and (not self.report['complete_frame_accounting_matches_actual_socket_bytes']
                    or not all(self.report['cleanup'][key] for key in ('sockets_closed','watchdog_exited','owned_destination_cache_released','pending_copy_stream_settled'))):
                self.report.update(passed=False,status='failed',error='Ownership/accounting cleanup failed')
                error=RuntimeError(self.report['error'])
        if error is not None:raise TransferError(self.report['error'],copy.deepcopy(self.report)) from None
        return copy.deepcopy(self.report)


def reconcile(source,destination):
    """Small independent successful-pair gate; never compare remote clocks."""
    require(source['role']=='source' and destination['role']=='destination','Endpoint roles differ')
    require(source['passed'] is destination['passed'] is True,'Failed/cancelled transfer cannot count as completed work')
    require(source['spec']==destination['spec'],'Peer frozen transfer identity differs')
    spec=validate_spec(source['spec'])
    require(source['coordinator_config']==destination['coordinator_config'],'Directional scheduler/cap configurations differ')
    expected_streams={name:{'stream_id':spec['request_id']+'/transfer-'+name,'kind':name} for name in ('rpc','bulk')}
    for report in (source,destination):
        require(report['protocol_completed'] is True and report['completed_layers']==spec['layers'] and report['unfinished_layers']==0,
                'Transfer did not finish every offered layer')
        require(report['complete_frame_accounting_matches_actual_socket_bytes'] is True,'Partial socket/frame mismatch')
        require(report['GPU_execution_evidence'] is (not spec['cpu_component_only']),'Actual backend evidence scope differs')
        require(report['physical_wire_bytes'] is None and report['decode_tokens']==0,'Transfer-only work/byte scope changed')
        require(report['outgoing_streams']==expected_streams,'Transfer streams do not bind the actual request owner')
        require(len(report['layers'])==spec['layers'] and report['copy_counts_complete'] is True,'Missing actual per-layer copy records')
        times=[finite(report[k],k) for k in ('accepted_local_s','executor_entered_local_s','handshake_complete_local_s','settled_local_s')]
        require(all(a<=b for a,b in zip(times,times[1:])) and times[-1]<=report['deadline_local_s'],'Endpoint-native acceptance/deadline order failed')
        for key in ('sockets_closed','watchdog_exited','owned_destination_cache_released','pending_copy_stream_settled'):
            require(report['cleanup'][key] is True,'Transfer ownership cleanup failed')
        require(report['cleanup']['global_budget_closed_by_component'] is False,'Component closed another job shared budget')
        require(report['cache_content_equality_verified'] is spec['diagnostic'],'Main/diagnostic content-check scope differs')
        gpu=not spec['cpu_component_only'];verification=gpu and spec['diagnostic'] and report['role']=='destination'
        require(report['copy_counts']=={'migration_copies':2*spec['layers'] if gpu else 0,
            'migration_bytes':spec['expected_total_payload_bytes'] if gpu else 0,
            'diagnostic_verification_d2h_copies':2*spec['layers'] if verification else 0,
            'diagnostic_verification_d2h_bytes':spec['expected_total_payload_bytes'] if verification else 0},
            'Real migration/diagnostic copy counts differ from completed layer payloads')
        require(all(amount==0 for directions in report['partial_frame_application_bytes'].values() for amount in directions.values()),
                'Partial frames cannot pass completed transfer acceptance')
    for name in ('rpc','bulk'):
        a,b=source['socket_accounting'][name],destination['socket_accounting'][name]
        require(a['sent_bytes']==b['received_bytes'] and a['received_bytes']==b['sent_bytes'],'Actual peer socket bytes differ')
        require(source['frames'][name]['sent']==destination['frames'][name]['received']
                and source['frames'][name]['received']==destination['frames'][name]['sent'],'Peer complete-frame counts/bytes differ')
    require(destination['all_layers_ready_before_release'] is True and destination['published_frontier_before_release']==spec['layers']
            and destination['all_layers_ready_local_s']<=destination['cache_released_local_s']<=destination['settled_local_s'],
            'Every actual ready layer must precede owned release and local settlement')
    require(destination['destination_GPU_weakref_check_performed'] is (not spec['cpu_component_only']),
            'Destination GPU weak-reference evidence scope differs')
    if not spec['cpu_component_only']:
        require(destination['destination_GPU_allocations_tracked']==2*spec['layers']
                and destination['destination_GPU_references_remaining']==0,
                'Every actual destination CUDA allocation must be released')
    require(source['frames']['bulk']['sent']['application_payload_bytes']==spec['expected_total_payload_bytes']
            and destination['frames']['bulk']['received']['application_payload_bytes']==spec['expected_total_payload_bytes'],
            'Actual transmitted prefix payload total differs')
    require(len(destination['publication_records'])==spec['layers'],'Actual per-layer publication records missing')
    for layer,(a,b) in enumerate(zip(source['layers'],destination['layers'])):
        require(a['layer']==b['layer']==layer and a['status']=='frame_sent' and b['status']=='published','Actual contiguous publication differs')
        commit=destination['publication_records'][layer]
        require((commit['layer'],commit['frontier'],commit['payload_bytes'])==(layer,layer+1,spec['expected_layer_payload_bytes']),
                'Actual complete layer cache publication differs')
        require(b['copy']['ready_local_s']<=commit['monotonic_s']<=b['published_local_s']<=destination['all_layers_ready_local_s'],
                'Layer published before device readiness')
        for entry,report in ((a,source),(b,destination)):
            copied=entry['copy'];gpu=not spec['cpu_component_only']
            require(copied['backend']==('cuda_stream_synchronized' if gpu else 'cpu_byte_component_only')
                    and copied['copies']==(2 if gpu else 0) and copied['bytes']==(spec['expected_layer_payload_bytes'] if gpu else 0),
                    'Concrete backend copy evidence differs')
            require(report['handshake_complete_local_s']<=copied['started_local_s']<=copied['ready_local_s']<=report['settled_local_s'],
                    'Copy completion is outside actual endpoint execution')
        require(a['copy']['ready_local_s']<=a['send']['started_local_s']<=a['send']['finished_local_s']<=source['settled_local_s'],
                'Source sent a layer before actual copy readiness')
        if spec['diagnostic']:
            require(a['source_hashes']==b['source_hashes']==b['destination_hashes'],'Actual peer per-layer content hash differs')
            require(isinstance(a['source_hashes'],list) and len(a['source_hashes'])==2 and all(
                isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h) for h in a['source_hashes']),
                'Missing actual K/V SHA256 values')
        else:require(a['source_hashes'] is b['source_hashes'] is b['destination_hashes'] is None,'Main path captured tensor hashes')
    return {'passed':True,'GPU_execution_evidence':not spec['cpu_component_only'],'CPU_component_only':spec['cpu_component_only'],
        'spec_sha256':spec['spec_sha256'],'complete_layers':spec['layers'],'payload_bytes':spec['expected_total_payload_bytes'],
        'content_equality_verified':spec['diagnostic'],'content_scope':'actual target GPU readback' if spec['diagnostic'] and not spec['cpu_component_only']
            else 'CPU component only' if spec['cpu_component_only'] else 'main readiness/bytes only; separate diagnostic required',
        'decode_tokens':0,'physical_wire_bytes':None,'source_report_sha256':digest(source),'destination_report_sha256':digest(destination)}
