#!/usr/bin/env python3
"""A7 / E2: parallel prefill of a new turn [s, e) with k = 1 or 2 helpers that hold the prefix [0, s) (Theorem 2).

x  (executor, coordinator): prefills the back [m_k, e) on top of [0, s) and all helper chunks.
h1 (helper 1):  prefills [s, m1) and streams each layer's new K/V to x and, if k = 2, to h2.
h2 (helper 2):  receives h1's chunk layer by layer, prefills [m1, m2) on top of [0, m1) and streams its layers to x.
Every node waits for layer l of every earlier chunk before running layer l of its own chunk (per-layer pipelining),
exactly as ct_pp.py does for k = 1.  Measured: T_k from x's start signal to the end of x's back prefill, and T_single
(x prefills [s, e) alone).  Greedy decode of 32 tokens is compared with the single-edge reference; the logits are compared
with x prefilling the same chunks itself.  Compute shares come from CUDA_MPS_ACTIVE_THREAD_PERCENTAGE.
"""
import argparse, json, os, queue, socket, threading, time
import torch
from ct_hybrid import send_msg, recv_msg, paced_send, load_model, ctx_ids
from ct_pp import prefix_cache, clone_cache


def geom(model):
    c = model.config
    return c.num_hidden_layers, c.num_key_value_heads, c.hidden_size // c.num_attention_heads


class Rx:
    """Receives the K/V of one chunk of n tokens, layer by layer, into pinned memory."""
    def __init__(self, sock, NL, kvh, hd, n, t0):
        self.NL, self.kvh, self.hd, self.n = NL, kvh, hd, n; self.per = n * kvh * hd * 2 * 2
        self.buf = torch.empty(NL * self.per, dtype=torch.uint8).pin_memory()
        self.ev = [threading.Event() for _ in range(NL)]; self.tarr = [0.0] * NL; self.wait = 0.0
        self.th = threading.Thread(target=self.run, args=(sock, t0)); self.th.start()

    def run(self, sock, t0):
        mv = memoryview(self.buf.numpy())
        for li in range(self.NL):
            got = 0; base = li * self.per
            while got < self.per:
                k = sock.recv_into(mv[base + got:base + self.per], self.per - got)
                if k == 0: raise ConnectionError('eof')
                got += k
            self.tarr[li] = time.perf_counter() - t0; self.ev[li].set()

    def get(self, li, dev):
        tw = time.perf_counter(); self.ev[li].wait(); self.wait += time.perf_counter() - tw
        g = self.buf[li * self.per:(li + 1) * self.per].to(dev, non_blocking=True).view(torch.bfloat16).view(2, self.kvh, self.n, self.hd)
        return g[0].unsqueeze(0), g[1].unsqueeze(0)


def prefill_chunk(model, ids, cache, lo, hi, rxs, sinks, dev):
    """Prefill [lo, hi) on cache (which holds [0, lo) minus the chunks in rxs, which arrive layer by layer).  After each
    layer the new K/V of [lo, hi) is sent to every sink (socket, cap_mbps).  Returns (compute_ms, last_send_ms, logits)."""
    layers = model.model.layers; NL = len(layers); t0 = time.perf_counter()
    def put(li):
        for rx in rxs:
            K, V = rx.get(li, dev); cache.update(K, V, li)
    hooks = []; senders = []
    if rxs:
        put(0); hooks += [layers[li].register_forward_pre_hook(lambda mod, inp, li=li: put(li)) for li in range(1, NL)]
    if sinks:
        qs = [queue.Queue() for _ in sinks]; cs = torch.cuda.Stream(); evq = queue.Queue(); done = {}
        def hook(mod, i, o, li=None):
            ev = torch.cuda.Event(); ev.record(); evq.put((li, ev))
        hooks += [layers[li].register_forward_hook(lambda mod, i, o, li=li: hook(mod, i, o, li)) for li in range(NL)]
        def copier():
            with torch.inference_mode():
                for _ in range(NL):
                    li, ev = evq.get()
                    with torch.cuda.stream(cs):
                        cs.wait_event(ev); ly = cache.layers[li]
                        part = torch.cat([ly.keys[0, :, lo:hi, :].reshape(-1), ly.values[0, :, lo:hi, :].reshape(-1)]).view(torch.uint8)
                        host = part.to('cpu', non_blocking=True)
                    cs.synchronize()
                    for q_ in qs: q_.put(host)
        def snd(k, sock, cap):
            st = {}
            for _ in range(NL):
                h = qs[k].get(); paced_send(sock, memoryview(h.numpy()), cap, st)
            done[k] = (time.perf_counter() - t0) * 1e3
        threads = [threading.Thread(target=copier)] + [threading.Thread(target=snd, args=(k, s_, c_)) for k, (s_, c_) in enumerate(sinks)]
        for th in threads: th.start()
    o = model(ids[:, lo:hi], past_key_values=cache, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize(); comp = (time.perf_counter() - t0) * 1e3
    for h_ in hooks: h_.remove()
    send = 0.0
    if sinks:
        for th in threads: th.join()
        send = max(done.values())
    return comp, send, o


def helper(a):
    dev = torch.device('cuda:0'); model = load_model(a.model, dev); NL, kvh, hd = geom(model)
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); srv.bind(('0.0.0.0', a.port)); srv.listen(4)
    print('listening', a.port, flush=True)
    up = None  # h2: data connection from h1
    down = None  # h1: data connection to h2
    if a.role == 'h2':
        while up is None:
            try: up = socket.create_connection((a.up, a.up_port)); up.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError: time.sleep(1)
        send_msg(up, {'who': 'h2'}); print('connected to h1', flush=True)
    pref = None; key = None; blob = None
    while True:
        conn, addr = srv.accept(); conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        who = recv_msg(conn)['who']; print('conn', who, addr, flush=True)
        if who == 'h2': down = conn; continue
        x = conn
        try:
            while True:
                req = recv_msg(x)
                if req['type'] == 'quit': x.close(); return
                if req['type'] == 'bye': break
                if req['type'] == 'blob':
                    n = req['nbytes']
                    if blob is None or blob.numel() < n: blob = torch.zeros(n, dtype=torch.uint8).pin_memory()
                    send_msg(x, {'ok': 1}); paced_send(x, memoryview(blob.numpy())[:n], 0); continue
                if req['type'] == 'prepare':
                    s, e = req['s'], req['e']; ids = ctx_ids(e + 1, dev)
                    with torch.inference_mode():
                        pref = None; torch.cuda.empty_cache(); _, pref = prefix_cache(model, ids, s)
                    torch.cuda.synchronize(); key = (s, e, ids); send_msg(x, {'ok': 1}); continue
                if req['type'] == 'run':
                    s, e, ids = key; k = req['k']; m1 = req['m1']; m2 = req.get('m2'); cap = req['cap_mbps']
                    with torch.inference_mode():
                        cache = clone_cache(pref); torch.cuda.synchronize()
                        if a.role == 'h1':
                            sinks = [(x, cap)] + ([(down, cap)] if k == 2 else [])
                            send_msg(x, {'ok': 1})
                            comp, sent, _ = prefill_chunk(model, ids, cache, s, m1, [], sinks, dev)
                        else:
                            send_msg(x, {'ok': 1})
                            t0 = time.perf_counter(); rx = Rx(up, NL, kvh, hd, m1 - s, t0)
                            comp, sent, _ = prefill_chunk(model, ids, cache, m1, m2, [rx], [(x, cap)], dev)
                            rx.th.join(); comp_extra = {'wait_ms': rx.wait * 1e3}
                    send_msg(x, dict(comp_ms=comp, send_ms=sent, **(comp_extra if a.role == 'h2' else {})))
                    cache = None; torch.cuda.empty_cache(); continue
        except ConnectionError:
            pass
        x.close()


class X:
    def __init__(self, a):
        self.a = a; self.dev = torch.device('cuda:0'); self.out = open(a.out, 'a'); self.host = socket.gethostname()
        self.share = os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', '100')
        self.model = load_model(a.model, self.dev); self.NL, self.kvh, self.hd = geom(self.model)
        self.h = {}
        for name, hp in (('h1', a.h1), ('h2', a.h2)):
            if not hp: continue
            host, port = hp.split(':'); s_ = socket.create_connection((host, int(port))); s_.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            send_msg(s_, {'who': 'x'}); self.h[name] = s_

    def emit(self, **kw):
        kw.update(host=self.host, share=self.share, ts=time.time(), gpu=torch.cuda.get_device_name(0))
        self.out.write(json.dumps(kw) + '\n'); self.out.flush()

    def blob(self, name, n):
        s_ = self.h[name]; buf = bytearray(n); mv = memoryview(buf)
        send_msg(s_, {'type': 'blob', 'nbytes': n}); recv_msg(s_); got = 0
        while got < n:
            k = s_.recv_into(mv[got:], n - got)
            if k == 0: raise ConnectionError
            got += k

    def probe(self, label, n=64 * 1024 * 1024):
        for name in self.h:
            t = time.perf_counter(); self.blob(name, n); ms = (time.perf_counter() - t) * 1e3
            self.emit(kind='probe', label=label, src=name, nbytes=n, ms=ms, mbps=n * 8 / ms / 1e3)
        if len(self.h) == 2:  # both helpers at once: is x's ingress shared?
            t = time.perf_counter(); th = [threading.Thread(target=self.blob, args=(nm, n)) for nm in self.h]
            for t_ in th: t_.start()
            for t_ in th: t_.join()
            ms = (time.perf_counter() - t) * 1e3; self.emit(kind='probe2', label=label, nbytes=2 * n, ms=ms, mbps=2 * n * 8 / ms / 1e3)

    def decode(self, cache, tok, n):
        toks = []
        for _ in range(n):
            tok = self.model(tok, past_key_values=cache, use_cache=True).logits[:, -1:].argmax(-1); toks.append(int(tok))
        return toks

    def local(self, ids, pref, bounds):
        """x prefills the chunks itself (numerical control, same chunk boundaries)."""
        with torch.inference_mode():
            cache = clone_cache(pref); o = None
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                o = self.model(ids[:, lo:hi], past_key_values=cache, use_cache=True, logits_to_keep=1)
            lg = o.logits[0, -1].float().cpu(); cache = None; o = None; torch.cuda.empty_cache()
        return lg

    def single(self, ids, pref, s, e):
        with torch.inference_mode():
            cache = clone_cache(pref); torch.cuda.synchronize(); t = time.perf_counter()
            o = self.model(ids[:, s:e], past_key_values=cache, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
            first = o.logits[:, -1:].argmax(-1); lg = o.logits[0, -1].float().cpu()
            toks = [int(first)] + self.decode(cache, first, 31); cache = None; o = None; torch.cuda.empty_cache()
        return ms, toks, lg

    def trial(self, cell, ids, pref, s, e, k, fr, cap, ref, ref_lg, rep):
        m1 = s + int(round(fr[0] * (e - s))); m2 = m1 + int(round(fr[1] * (e - s))) if k == 2 else None
        mk = m2 if k == 2 else m1
        bounds = [s, m1] + ([m2] if k == 2 else []) + [e]
        lc = self.local(ids, pref, bounds) if rep == 0 else None
        with torch.inference_mode():
            cache = clone_cache(pref); torch.cuda.synchronize()
            if k == 2:
                send_msg(self.h['h2'], {'type': 'run', 'k': 2, 'm1': m1, 'm2': m2, 'cap_mbps': cap}); recv_msg(self.h['h2'])
            send_msg(self.h['h1'], {'type': 'run', 'k': k, 'm1': m1, 'cap_mbps': cap}); recv_msg(self.h['h1'])
            t0 = time.perf_counter()
            rxs = [Rx(self.h['h1'], self.NL, self.kvh, self.hd, m1 - s, t0)]
            if k == 2: rxs.append(Rx(self.h['h2'], self.NL, self.kvh, self.hd, m2 - m1, t0))
            comp, _, o = prefill_chunk(self.model, ids, cache, mk, e, rxs, [], self.dev)
            T = (time.perf_counter() - t0) * 1e3
            for rx in rxs: rx.th.join()
            hr = {nm: recv_msg(self.h[nm]) for nm in (['h1', 'h2'] if k == 2 else ['h1'])}
            assert cache.get_seq_length() == e, (cache.get_seq_length(), e)
            first = o.logits[:, -1:].argmax(-1); lg = o.logits[0, -1].float().cpu()
            toks = [int(first)] + self.decode(cache, first, 31)
            self.emit(kind='pp', cell=cell, s=s, e=e, k=k, m1=m1, m2=m2, fr=fr, cap_mbps=cap, rep=rep, T_ms=T,
                      x_wait_ms=sum(rx.wait for rx in rxs) * 1e3, last_arrival_ms=[rx.tarr[-1] * 1e3 for rx in rxs],
                      helpers=hr, agree=sum(int(p == q) for p, q in zip(toks, ref)),
                      logit_diff_vs_local_chunked=float((lg - lc).abs().max()) if lc is not None else None,
                      logit_diff_vs_single=float((lg - ref_lg).abs().max()))
            cache = None; o = None; torch.cuda.empty_cache()
        return T

    def run_cell(self, cell, s, e, cap, reps, grid1, grid2):
        ids = ctx_ids(e + 1, self.dev); self.probe(f'{cell}-pre')
        for s_ in self.h.values(): send_msg(s_, {'type': 'prepare', 's': s, 'e': e})
        for s_ in self.h.values(): recv_msg(s_)
        with torch.inference_mode(): _, pref = prefix_cache(self.model, ids, s)
        torch.cuda.synchronize(); ref = None; ref_lg = None
        for rep in range(reps):
            ms, toks, lg = self.single(ids, pref, s, e)
            if ref is None: ref, ref_lg = toks, lg
            self.emit(kind='single', cell=cell, s=s, e=e, rep=rep, single_ms=ms, same_as_rep0=int(toks == ref))
        for rep in range(reps):
            for f in grid1: self.trial(cell, ids, pref, s, e, 1, (f,), cap, ref, ref_lg, rep)
            if 'h2' in self.h:
                for f in grid2: self.trial(cell, ids, pref, s, e, 2, (f, f), cap, ref, ref_lg, rep)
        self.probe(f'{cell}-post'); pref = None; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--role', required=True); ap.add_argument('--model', required=True); ap.add_argument('--port', type=int, default=29561)
    ap.add_argument('--up', default=''); ap.add_argument('--up_port', type=int, default=29561)
    ap.add_argument('--h1', default=''); ap.add_argument('--h2', default=''); ap.add_argument('--out', default='')
    ap.add_argument('--cells', default=''); ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--grid1', default='0.3,0.4,0.5'); ap.add_argument('--grid2', default='0.25,0.3,0.333')
    ap.add_argument('--quit', action='store_true')
    a = ap.parse_args()
    if a.role in ('h1', 'h2'): return helper(a)
    x = X(a); g1 = [float(v) for v in a.grid1.split(',')]; g2 = [float(v) for v in a.grid2.split(',')] if a.grid2 else []
    for spec in a.cells.split(';'):
        if not spec: continue
        name, s, e, cap = spec.split(':'); x.run_cell(name, int(s), int(e), int(cap), a.reps, g1, g2)
    for s_ in x.h.values(): send_msg(s_, {'type': 'quit' if a.quit else 'bye'})
    x.emit(kind='done')


if __name__ == '__main__':
    main()
