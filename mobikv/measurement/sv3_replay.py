#!/usr/bin/env python3
"""A7 / E1 (SV-3): replay the job streams of three edges on three real GPUs (one process per host), with replica-assisted
parallel prefill across hosts and real batched decoding.

Input: a joblog v2 of mobikv_sim_a7 (per edge: fg/bg jobs with the simulator's submit/start/end times; RPP fronts 'ppf'
and backs 'ppb' with their partner edge) and the request records.  Each host replays one edge in real time:
  * prefill-class foreground jobs FCFS, one pass each (no chunking), back to back while any is waiting;
  * background replication in 512-token chunks when no foreground job waits, alternating with decode steps;
  * decode: every request executed at this edge decodes its output length, one token per step, batched over all active
    streams (attention over the batch's mean context, from a preallocated buffer), between prefill passes, between
    background chunks and while an RPP back waits for front layers;
  * RPP front (ppf): one pass on top of the prefix; after each layer its K/V of [s, m) goes to the executor's host over the
    real network (one sender per host, FIFO = egress); if the executor is outside the three edges, the sender only holds
    the egress for the layer's transfer time;
  * RPP back (ppb): before layer l it waits for layer l of the front (from the helper's host, or, if the helper is outside
    the three edges, at the arrival time implied by the simulator's schedule of that helper), then runs the back.
Submit times come from the simulator; consecutive jobs of one request are shifted by the measured delay of the previous
job (same edge; for a front at another host, by a message from the executor).  Output: per job and per request timings.
"""
import argparse, collections, json, math, os, queue, socket, struct, threading, time
import numpy as np
import torch
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from ct_hybrid import send_msg, recv_msg, recv_exact, load_model, ctx_ids

ap = argparse.ArgumentParser()
ap.add_argument('--joblog', required=True); ap.add_argument('--edges', required=True); ap.add_argument('--hosts', required=True)
ap.add_argument('--me', type=int, required=True); ap.add_argument('--port', type=int, default=29571)
ap.add_argument('--model', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--bmax', type=int, default=16); ap.add_argument('--lmax', type=int, default=24576)
ap.add_argument('--bg_chunk', type=int, default=512); ap.add_argument('--drain', type=float, default=300.0)
ap.add_argument('--null_mbps', type=float, default=930.0); ap.add_argument('--lead', type=float, default=20.0)
ap.add_argument('--limit', type=float, default=0.0)  # smoke tests only: replay [T0, T0 + limit)
ap.add_argument('--bg_dfb', type=int, default=0); ap.add_argument('--dfb_chunk', type=int, default=128)  # A8: strict decode priority for background replay
ap.add_argument('--no_rpp', type=int, default=0)  # A7.1 paired ablation: fronts dropped, backs run as single-edge [s, e)
a = ap.parse_args()
EDGES = a.edges.split(';'); HOSTS = a.hosts.split(','); ME = a.me; E = EDGES[ME]; N = len(EDGES)
jl = json.load(open(a.joblog)); T0, T1 = jl['window']; NLs = jl['NL']
if a.limit > 0:
    T1 = T0 + a.limit; jl['jobs'] = {e: [j for j in v if j[1] < T1] for e, v in jl['jobs'].items()}
JOBS = jl['jobs']; REQS = [r for r in jl['reqs'] if T0 <= r['t'] < T1]
dev = torch.device('cuda:0'); torch.cuda.set_device(0)
model = load_model(a.model, dev); cfg = model.config
NL = cfg.num_hidden_layers; KVH = cfg.num_key_value_heads; HD = cfg.hidden_size // cfg.num_attention_heads
assert NL == NLs
PER_TOK = KVH * HD * 2 * 2  # bytes of K and V per token per layer (bf16)
ids = ctx_ids(40000, dev)
LAYERS = model.model.layers
G = dict(in_decode=False)

# ------------------------------------------------------------------ job lists
def J(j):
    d = dict(kind=j[0], t=j[1], d=j[2], st=j[3], en=j[4], a=j[5], b=j[6], rid=j[7])
    if len(j) > 8:
        d['role'] = j[8]; d['partner'] = j[9]
        if j[8] == 'ppf': d.update(Cf=j[10], X=j[11], dd=j[12])
        else: d.update(s=j[10], Cb=j[11], X=j[12], dd=j[13])
    else: d['role'] = ''
    return d
mine = sorted([J(j) for j in JOBS.get(E, [])], key=lambda d: (d['t'], d['st'] if d['st'] is not None else d['t']))
if a.no_rpp:  # same job stream with RPP switched off: no helper work, the executor prefills [s, e) itself
    mine = [d for d in mine if d['role'] != 'ppf']
    for d in mine:
        if d['role'] == 'ppb': d.update(role='', a=d['s'], b=d['b'], was_ppb=1)
fgj = [d for d in mine if d['kind'] == 'fg']; bgj = [d for d in mine if d['kind'] == 'bg']
for k, d in enumerate(fgj): d['k'] = k
by_rid = collections.defaultdict(list)
for d in fgj: by_rid[d['rid']].append(d)
for lst in by_rid.values():
    lst.sort(key=lambda d: d['st'])
    for i, d in enumerate(lst): d['prev'] = lst[i - 1] if i else None
# fronts at this edge whose executor (in the set) has earlier jobs of the same request: wait for its carry message
def jobs_of(edge, rid): return [J(j) for j in JOBS.get(edge, []) if j[0] == 'fg' and j[7] == rid]
for d in fgj:
    d['await_carry'] = False
    if d['role'] == 'ppf' and d['partner'] in EDGES:
        d['await_carry'] = any(x['en'] <= d['t'] + 1e-6 and x['role'] != 'ppb' for x in jobs_of(d['partner'], d['rid']))
# phantom fronts for backs whose helper is outside the set: layer arrival times implied by the simulator's helper schedule
def phantom_arrivals(d):
    f = next((x for x in jobs_of(d['partner'], d['rid']) if x['role'] == 'ppf'), None)
    if f is None:
        tx_end = d['en'] - d['Cb'] / NL; return [d['st'] + (l + 1) / NL * (tx_end - d['st']) for l in range(NL)]
    hs, Cf, X, dd = f['st'], f['Cf'], f['X'], f['dd']
    return [max(hs + (l + 1) * Cf / NL + X / NL, hs + Cf / NL + (l + 1) * X / NL) + dd / 2 for l in range(NL)]
for d in fgj:
    if d['role'] == 'ppb' and (d['partner'] not in EDGES or not any(x['role'] == 'ppf' for x in jobs_of(d['partner'], d['rid']))):
        d['phantom'] = phantom_arrivals(d)  # helper outside the set, or its front falls outside the replayed window
# requests executed here
my_reqs = {r['rid']: r for r in REQS if r['x'] == E}
last_job = {}
for rid in my_reqs:
    lst = by_rid.get(rid)
    if lst: last_job[rid] = lst[-1]
print(E, 'fg', len(fgj), 'bg', len(bgj), 'reqs', len(my_reqs), 'ppf', sum(d['role'] == 'ppf' for d in fgj),
      'ppb', sum(d['role'] == 'ppb' for d in fgj), 'phantom', sum('phantom' in d for d in fgj), flush=True)

# ------------------------------------------------------------------ GPU state
with torch.inference_mode():
    smax = max([0] + [d['a'] for d in fgj] + [d.get('s', 0) for d in fgj] + [d['a'] for d in bgj])
    smax = min(max(smax, 1024), 32768)
    master = model(ids[:, :smax], use_cache=True, logits_to_keep=1).past_key_values
    MK = [(ly.keys, ly.values) for ly in master.layers]; master = None
    DK = [torch.zeros(a.bmax, KVH, a.lmax + 8, HD, dtype=torch.bfloat16, device=dev) for _ in range(NL)]
    DV = [torch.zeros(a.bmax, KVH, a.lmax + 8, HD, dtype=torch.bfloat16, device=dev) for _ in range(NL)]
torch.cuda.synchronize(); torch.cuda.empty_cache()
print('prefix', smax, 'mem GB', round(torch.cuda.memory_allocated() / 2**30, 1), flush=True)


def prefix_cache(s):
    c = DynamicCache()
    if s > 0:
        for li, (K, V) in enumerate(MK): c.update(K[:, :, :s], V[:, :, :s], li)
    return c


class DecLayer(DynamicLayer):
    def __init__(self, li, B, L):
        super().__init__(); self.li, self.B, self.L = li, B, L
        self.dtype, self.device = torch.bfloat16, dev; self.is_initialized = True
        self.keys = DK[li][:B, :, :L]; self.values = DV[li][:B, :, :L]
    def update(self, k, v, *args, **kw):
        n = k.shape[-2]; L = self.L
        DK[self.li][:self.B, :, L:L + n].copy_(k); DV[self.li][:self.B, :, L:L + n].copy_(v)
        return DK[self.li][:self.B, :, :L + n], DV[self.li][:self.B, :, :L + n]


# ------------------------------------------------------------------ network
socks_out = {}; inbox = {}; inbox_lock = threading.Lock(); carry_in = {}
def kv_slot(rid, n):
    with inbox_lock:
        if rid not in inbox:
            inbox[rid] = dict(buf=torch.empty(NL * n * PER_TOK, dtype=torch.uint8).pin_memory(), n=n,
                              ev=[threading.Event() for _ in range(NL)], tarr=[None] * NL)
        return inbox[rid]
def receiver(conn):
    while True:
        try: h = recv_msg(conn)
        except (ConnectionError, struct.error): return
        if h['type'] == 'kv':
            sl = kv_slot(h['rid'], h['n']); per = h['n'] * PER_TOK; li = h['layer']
            mv = memoryview(sl['buf'].numpy())[li * per:(li + 1) * per]; got = 0
            while got < per:
                k = conn.recv_into(mv[got:], per - got)
                if k == 0: return
                got += k
            sl['tarr'][li] = now(); sl['ev'][li].set()
        elif h['type'] == 'carry':
            carry_in[h['rid']] = h['carry']
        elif h['type'] == 'bye':
            return
sendq = queue.Queue(); send_err = [0]; wait_timeouts = [0]
def sender():
    while True:
        it = sendq.get()
        if it is None: return
        dst, hdr, host = it
        if dst is None:  # executor outside the set: hold the egress for the transfer time only
            time.sleep(host.numel() * 8 / (a.null_mbps * 1e6)); continue
        try:
            s_ = socks_out[dst]; send_msg(s_, hdr)
            if host is not None:
                mv = memoryview(host.numpy()); off = 0
                while off < len(mv): off += s_.send(mv[off:off + 4 * 1024 * 1024])
        except OSError:
            send_err[0] += 1

srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); srv.bind(('0.0.0.0', a.port)); srv.listen(8)
incoming = {}
def acceptor():
    for _ in range(N - 1):
        c, _ = srv.accept(); c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); who = recv_msg(c)['me']; incoming[who] = c
th_acc = threading.Thread(target=acceptor); th_acc.start()
for j in range(N):
    if j == ME: continue
    while True:
        try:
            s_ = socket.create_connection((HOSTS[j], a.port), timeout=5); s_.settimeout(None); break
        except OSError: time.sleep(1)
    s_.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); send_msg(s_, {'me': ME}); socks_out[j] = s_
th_acc.join()
# clock sync with node 0 (Cristian), then common start
if ME == 0:
    for j, c in incoming.items():
        for _ in range(30):
            recv_msg(c); send_msg(c, {'t2': time.time()})
    Tstart = time.time() + a.lead; offset = 0.0
    for j in socks_out: send_msg(socks_out[j], {'type': 'start', 'T': Tstart})
else:
    best = None; s0 = socks_out[0]; c0 = incoming[0]
    for _ in range(30):
        t1 = time.time(); send_msg(s0, {'t1': t1}); t2 = recv_msg(s0)['t2']; t3 = time.time()
        if best is None or t3 - t1 < best[0]: best = (t3 - t1, t2 - (t1 + t3) / 2)
    offset = best[1]; Tstart = recv_msg(c0)['T']
for j, c in incoming.items(): threading.Thread(target=receiver, args=(c,), daemon=True).start()
threading.Thread(target=sender, daemon=True).start()
Tloc = Tstart - offset  # local wall time of simulated time T0
def now(): return T0 + (time.time() - Tloc)
print('offset_s', round(offset, 4), 'start in', round(Tloc - time.time(), 1), 's', flush=True)

# ------------------------------------------------------------------ decode streams
streams = []  # dict(rid, ctx, left, toks=[times])
done_streams = {}
dec_stats = dict(steps=0, ms=0.0, tok=0)
def decode_step():
    if not streams: return
    G['in_decode'] = True
    t_ = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(streams), a.bmax):
            grp = streams[i:i + a.bmax]; B = len(grp)
            L = int(min(a.lmax, max(256, round(np.mean([s_['ctx'] for s_ in grp]) / 256) * 256)))
            c = Cache(layers=[DecLayer(li, B, L) for li in range(NL)])
            model(ids[:B, :1].reshape(B, 1) if False else ids[:, 7:8].expand(B, 1), past_key_values=c, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
    tt = now(); dec_stats['steps'] += 1; dec_stats['ms'] += (time.perf_counter() - t_) * 1e3; dec_stats['tok'] += len(streams)
    G['in_decode'] = False
    keep = []
    for s_ in streams:
        s_['toks'].append(tt); s_['ctx'] += 1; s_['left'] -= 1
        if s_['left'] <= 0: done_streams[s_['rid']] = s_
        else: keep.append(s_)
    streams[:] = keep

# ------------------------------------------------------------------ job execution
meas = {}; carry = collections.defaultdict(float)
def finish(d, t_start, t_end, **kw):
    meas[d['k']] = dict(k=d['k'], rid=d['rid'], role=d['role'], t=d['t'], st=d['st'], en=d['en'], ms=t_start, me=t_end, **kw)
    c_ = max(0.0, t_end - d['en']); carry[d['rid']] = c_
    # tell a helper in the set whose front for this request follows this job
    for j_edge in EDGES:
        if j_edge == E: continue
        for x in jobs_of(j_edge, d['rid']):
            if x['role'] == 'ppf' and x['partner'] == E and x['t'] >= d['en'] - 1e-6 and d['role'] != 'ppb':
                sendq.put((EDGES.index(j_edge), {'type': 'carry', 'rid': d['rid'], 'carry': c_}, None))
    if d['rid'] in last_job and last_job[d['rid']] is d:
        r = my_reqs[d['rid']]
        streams.append(dict(rid=d['rid'], ctx=min(a.lmax, r['R'] + r['q']), left=max(1, r['out']), toks=[], first=t_end))

def run_plain(d, s, e_):
    t_ = now()
    with torch.inference_mode():
        c = prefix_cache(s); model(ids[:, s:e_], past_key_values=c, use_cache=True, logits_to_keep=1); torch.cuda.synchronize()
        c = None
    finish(d, t_, now())

def run_front(d):
    s, m = d['a'], d['b']; n = m - s; dst = EDGES.index(d['partner']) if d['partner'] in EDGES else None
    t_ = now(); evq = queue.Queue(); cs = torch.cuda.Stream()
    with torch.inference_mode():
        c = prefix_cache(s)
        def hook(mod, i, o, li=None):
            if G['in_decode']: return
            ev = torch.cuda.Event(); ev.record(); evq.put((li, ev))
        hs = [LAYERS[li].register_forward_hook(lambda mod, i, o, li=li: hook(mod, i, o, li)) for li in range(NL)]
        def copier():
            with torch.inference_mode():
                for _ in range(NL):
                    li, ev = evq.get()
                    with torch.cuda.stream(cs):
                        cs.wait_event(ev); ly = c.layers[li]
                        part = torch.cat([ly.keys[0, :, s:m, :].reshape(-1), ly.values[0, :, s:m, :].reshape(-1)]).view(torch.uint8)
                        host = part.to('cpu', non_blocking=True)
                    cs.synchronize()
                    sendq.put((dst, {'type': 'kv', 'rid': d['rid'], 'layer': li, 'n': n}, host))
        th = threading.Thread(target=copier); th.start()
        model(ids[:, s:m], past_key_values=c, use_cache=True, logits_to_keep=1); torch.cuda.synchronize()
        for h_ in hs: h_.remove()
        th.join(); c = None
    finish(d, t_, now(), front_tok=n)

def run_back(d):
    m, e_, s = d['a'], d['b'], d['s']; n = m - s; t_ = now(); waited = [0.0]
    ph = d.get('phantom'); sl = None if ph else kv_slot(d['rid'], n)
    with torch.inference_mode():
        c = prefix_cache(s)
        def wait_layer(li):
            tw = now()
            if ph:
                while now() < ph[li]:
                    if streams: decode_step()
                    else: time.sleep(min(0.002, max(0.0, ph[li] - now())))
                K = torch.zeros(1, KVH, n, HD, dtype=torch.bfloat16, device=dev); V = K
            else:
                while not sl['ev'][li].is_set():
                    if now() - tw > 120.0: break  # safety: the helper never delivered (counted, reported)
                    if streams: decode_step()
                    else: sl['ev'][li].wait(0.002)
                if not sl['ev'][li].is_set():
                    wait_timeouts[0] += 1; K = torch.zeros(1, KVH, n, HD, dtype=torch.bfloat16, device=dev); V = K
                else:
                    g = sl['buf'][li * n * PER_TOK:(li + 1) * n * PER_TOK].to(dev, non_blocking=True).view(torch.bfloat16).view(2, KVH, n, HD)
                    K, V = g[0].unsqueeze(0), g[1].unsqueeze(0)
            waited[0] += now() - tw
            c.update(K, V, li)
        def pre(mod, inp, li=None):
            if G['in_decode']: return
            wait_layer(li)
        wait_layer(0)
        hs = [LAYERS[li].register_forward_pre_hook(lambda mod, inp, li=li: pre(mod, inp, li)) for li in range(1, NL)]
        model(ids[:, m:e_], past_key_values=c, use_cache=True, logits_to_keep=1); torch.cuda.synchronize()
        for h_ in hs: h_.remove()
        c = None
    if sl is not None:
        with inbox_lock: inbox.pop(d['rid'], None)
    finish(d, t_, now(), wait_s=waited[0], phantom=int(ph is not None))

# ------------------------------------------------------------------ main loop
def eligible(d):
    t_ = d['t']
    if d.get('prev') is not None: t_ += carry.get(d['rid'], 0.0)
    if d['await_carry']:
        if d['rid'] in carry_in: t_ = d['t'] + carry_in[d['rid']]
        elif now() < d['t'] + 60.0: return None
    return t_
pend = list(fgj); fgq = collections.deque(); bgi = 0; bgq = collections.deque(); bgcur = None; bg_log = []
while time.time() < Tloc: time.sleep(0.01)
while True:
    tn = now()
    for d in list(pend):
        el = eligible(d)
        if el is not None and el <= tn: fgq.append(d); pend.remove(d)
        elif d['t'] > tn + 5: break
    while bgi < len(bgj) and bgj[bgi]['t'] <= tn: bgq.append(bgj[bgi]); bgi += 1
    if fgq:
        d = fgq.popleft()
        if d['role'] == 'ppf': run_front(d)
        elif d['role'] == 'ppb': run_back(d)
        else: run_plain(d, d['a'], d['b'])
        continue
    did = False
    bg_ok = not (a.bg_dfb and streams)  # A8: background replay only while no stream decodes here
    if bg_ok and bgcur is None and bgq:
        b = bgq.popleft(); bgcur = dict(d=b, pos=b['a'], c=None, t0=now())
    if bg_ok and bgcur is not None:
        b = bgcur['d']; lo = bgcur['pos']; hi = min(b['b'], lo + (a.dfb_chunk if a.bg_dfb else a.bg_chunk))
        with torch.inference_mode():
            if bgcur['c'] is None: bgcur['c'] = prefix_cache(lo)
            model(ids[:, lo:hi], past_key_values=bgcur['c'], use_cache=True, logits_to_keep=1); torch.cuda.synchronize()
        bgcur['pos'] = hi
        if hi >= b['b']: bg_log.append(dict(t=b['t'], ms=bgcur['t0'], me=now(), n=b['b'] - b['a'])); bgcur = None; torch.cuda.empty_cache()
        did = True
    if streams: decode_step(); did = True
    if not did:
        if not pend and not fgq and bgi >= len(bgj) and bgcur is None and not streams: break
        if tn > T1 + a.drain: break
        time.sleep(0.002)
    if tn > T1 + a.drain and not fgq: break
for s_ in streams: done_streams.setdefault(s_['rid'], s_)
for j in socks_out:
    try: sendq.put((j, {'type': 'bye'}, None))
    except Exception: pass
time.sleep(2)
req_out = []
for rid, r in my_reqs.items():
    lj = last_job.get(rid)
    if lj is None or lj['k'] not in meas: continue
    mj = meas[lj['k']]; ttft_sim = r['first'] - r['t']; ttft_meas = ttft_sim + (mj['me'] - lj['en'])
    st_ = done_streams.get(rid); tp = None; ntok = 0
    if st_ is not None and len(st_['toks']) >= 2:
        ntok = len(st_['toks']); tp = (st_['toks'][-1] - st_['first']) / ntok
    boundary = any(d_['role'] == 'ppb' and 'phantom' in d_ for d_ in by_rid.get(rid, []))
    req_out.append(dict(rid=rid, sess=r['sess'], turn=r['turn'], eval=r['eval'], t=r['t'], ttft_sim=ttft_sim, ttft_meas=ttft_meas,
                        tpot_meas=tp, ntok=ntok, out=r['out'], rpp=int(lj['role'] == 'ppb' or lj.get('was_ppb', 0)), boundary=int(boundary), kind=r['kind']))
res = dict(edge=E, me=ME, host=socket.gethostname(), gpu=torch.cuda.get_device_name(0), share=os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'),
           offset_s=offset, window=[T0, T1], jobs=list(meas.values()), bg=bg_log, reqs=req_out, dec=dec_stats,
           n_fg=len(fgj), n_done=len(meas), joblog=a.joblog, bg_dfb=a.bg_dfb, dfb_chunk=a.dfb_chunk, no_rpp=a.no_rpp, send_err=send_err[0], wait_timeouts=wait_timeouts[0])
json.dump(res, open(a.out, 'w'))
print('done', E, 'jobs', len(meas), '/', len(fgj), 'reqs', len(req_out), 'dec', dec_stats, flush=True)
os._exit(0)
