#!/usr/bin/env python3
"""A3 / G2: replica-assisted parallel prefill (RPP) across two real GPUs on two hosts.

helper (h):   holds the session prefix [0, s) (a replica).  On request it prefills the front [s, m) on top of
              it and, layer by layer, streams the new K/V of that layer to the server through the A1 token
              bucket (64 KiB chunks, 128 KiB bucket).  A forward hook records a CUDA event after each decoder
              layer; a sender thread copies that layer's slice on a separate CUDA stream and sends it, so the
              copy and the network overlap with the remaining layers.
server (x):   holds the same prefix [0, s).  At t0 it asks the helper for [s, m) and prefills the back [m, e)
              on top of [0, s) + front.  Layer l of the back runs only after layer l of the front has arrived
              (pre-forward hook waits, copies the slice to the GPU and appends it to the cache).
Measured:     T_split (t0 -> back prefill done at the server), its parts, and T_single (server prefills
              [s, e) alone).  Greedy decode of 32 tokens after the split is compared with the single-edge
              reference.
Compute share of both GPUs is set from outside with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE.
"""
import argparse, json, os, queue, socket, threading, time
import torch
from ct_hybrid import send_msg, recv_msg, paced_send, load_model, ctx_ids


def prefix_cache(model, ids, s):
    from transformers.cache_utils import DynamicCache
    if s == 0:
        return DynamicCache(), None
    o = model(ids[:, :s], use_cache=True, logits_to_keep=1)
    return o.past_key_values, [(ly.keys, ly.values) for ly in o.past_key_values.layers]


def clone_cache(pref):
    from transformers.cache_utils import DynamicCache
    c = DynamicCache()
    if pref:
        for li, (K, V) in enumerate(pref):
            c.update(K.clone(), V.clone(), li)
    return c


def helper(a):
    dev = torch.device('cuda:0'); model = load_model(a.model, dev)
    layers = model.model.layers; NL = len(layers)
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', a.port)); srv.listen(1); print('listening', a.port, flush=True)
    pref = None; key = None; cs = torch.cuda.Stream(); blob = None
    while True:
        conn, addr = srv.accept(); conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); print('conn', addr, flush=True)
        try:
            while True:
                req = recv_msg(conn)
                if req['type'] in ('bye', 'quit'):
                    if req['type'] == 'quit': conn.close(); return
                    break
                if req['type'] == 'blob':  # capacity probe: raw bytes, unpaced
                    n = req['nbytes']
                    if blob is None or blob.numel() < n:
                        blob = torch.zeros(n, dtype=torch.uint8).pin_memory()
                    send_msg(conn, {'ok': 1}); paced_send(conn, memoryview(blob.numpy())[:n], 0); continue
                if req['type'] == 'prepare':
                    s, e = req['s'], req['e']
                    ids = ctx_ids(e + 1, dev)
                    with torch.inference_mode():
                        pref = None; torch.cuda.empty_cache()
                        _, pref = prefix_cache(model, ids, s)
                    torch.cuda.synchronize(); key = (s, e, ids)
                    send_msg(conn, {'ok': 1}); continue
                if req['type'] == 'pp':
                    s, e, ids = key; m = req['m']; cap = req['cap_mbps']
                    q = queue.Queue(); st = {}
                    with torch.inference_mode():
                        cache = clone_cache(pref); torch.cuda.synchronize()
                        send_msg(conn, {'ok': 1})
                        t0 = time.perf_counter()
                        def hook(mod, inp, out, li=None):
                            ev = torch.cuda.Event(); ev.record(); q.put((li, ev))
                        hs = [layers[li].register_forward_hook(lambda mod, i, o, li=li: hook(mod, i, o, li)) for li in range(NL)]
                        def sender():
                          with torch.inference_mode():
                            for _ in range(NL):
                                li, ev = q.get()
                                with torch.cuda.stream(cs):
                                    cs.wait_event(ev)
                                    ly = cache.layers[li]
                                    part = torch.cat([ly.keys[0, :, s:m, :].reshape(-1), ly.values[0, :, s:m, :].reshape(-1)]).view(torch.uint8)
                                    host = part.to('cpu', non_blocking=True)
                                cs.synchronize()
                                paced_send(conn, memoryview(host.numpy()), cap, st)
                        th = threading.Thread(target=sender); th.start()
                        model(ids[:, s:m], past_key_values=cache, use_cache=True, logits_to_keep=1)
                        torch.cuda.synchronize(); comp_ms = (time.perf_counter() - t0) * 1e3
                        th.join(); send_ms = (time.perf_counter() - t0) * 1e3
                        for h_ in hs: h_.remove()
                    send_msg(conn, {'comp_ms': comp_ms, 'send_ms': send_ms})
                    cache = None; torch.cuda.empty_cache(); continue
        except ConnectionError:
            pass
        conn.close()


class Server:
    def __init__(self, a):
        self.a = a; self.dev = torch.device('cuda:0')
        self.sock = socket.create_connection((a.peer, a.port)); self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.out = open(a.out, 'a'); self.host = socket.gethostname()
        self.share = os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', '100')
        self.model = load_model(a.model, self.dev); c = self.model.config
        self.NL = c.num_hidden_layers; self.kvh = c.num_key_value_heads; self.hd = c.hidden_size // c.num_attention_heads

    def emit(self, **kw):
        kw.update(host=self.host, share=self.share, ts=time.time(), gpu=torch.cuda.get_device_name(0))
        self.out.write(json.dumps(kw) + '\n'); self.out.flush()

    def probe(self, label, n=64 * 1024 * 1024):
        buf = bytearray(n); mv = memoryview(buf); t0 = time.perf_counter()
        send_msg(self.sock, {'type': 'blob', 'nbytes': n}); recv_msg(self.sock); got = 0
        while got < n:
            k = self.sock.recv_into(mv[got:], n - got)
            if k == 0: raise ConnectionError
            got += k
        ms = (time.perf_counter() - t0) * 1e3
        self.emit(kind='probe', label=label, nbytes=n, ms=ms, mbps=n * 8 / ms / 1e3)

    def decode(self, cache, first_tok, n=32):
        toks = []; tok = first_tok
        for _ in range(n):
            lo = self.model(tok, past_key_values=cache, use_cache=True).logits
            tok = lo[:, -1:].argmax(-1); toks.append(int(tok))
        return toks

    def single(self, ids, pref, s, e):
        with torch.inference_mode():
            cache = clone_cache(pref); torch.cuda.synchronize(); t = time.perf_counter()
            o = self.model(ids[:, s:e], past_key_values=cache, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
            first = o.logits[:, -1:].argmax(-1); lg = o.logits[0, -1].float().cpu()
            toks = [int(first)] + self.decode(cache, first, 31)
            cache = None; o = None; torch.cuda.empty_cache()
        return ms, toks, lg

    def chunked(self, ids, pref, s, m, e):
        """Numerical control: the server itself prefills [s, m) then [m, e) (same split, one GPU)."""
        with torch.inference_mode():
            cache = clone_cache(pref)
            self.model(ids[:, s:m], past_key_values=cache, use_cache=True, logits_to_keep=1)
            o = self.model(ids[:, m:e], past_key_values=cache, use_cache=True, logits_to_keep=1)
            lg = o.logits[0, -1].float().cpu(); cache = None; o = None; torch.cuda.empty_cache()
        return lg

    def trial(self, cell, ids, pref, s, e, m, cap, ref, ref_lg, rep, label):
        ch_lg = self.chunked(ids, pref, s, m, e) if rep == 0 else None
        NL = self.NL; per = (m - s) * self.kvh * self.hd * 2 * 2  # K and V, bf16
        layers = self.model.model.layers
        pinned = torch.empty(NL * per, dtype=torch.uint8).pin_memory()
        arrived = [threading.Event() for _ in range(NL)]; tarr = [0.0] * NL
        with torch.inference_mode():
            cache = clone_cache(pref); torch.cuda.synchronize()
            send_msg(self.sock, {'type': 'pp', 'm': m, 'cap_mbps': cap}); recv_msg(self.sock)
            t0 = time.perf_counter()
            def rx():
                mv = memoryview(pinned.numpy())
                for li in range(NL):
                    got = 0; base = li * per
                    while got < per:
                        k = self.sock.recv_into(mv[base + got:base + per], per - got)
                        if k == 0: raise ConnectionError
                        got += k
                    tarr[li] = time.perf_counter() - t0; arrived[li].set()
            th = threading.Thread(target=rx); th.start()
            wait = [0.0]
            def put(li):
                tw = time.perf_counter(); arrived[li].wait(); wait[0] += time.perf_counter() - tw
                g = pinned[li * per:(li + 1) * per].to(self.dev, non_blocking=True).view(torch.bfloat16).view(2, self.kvh, m - s, self.hd)
                cache.update(g[0].unsqueeze(0), g[1].unsqueeze(0), li)
            put(0)
            hs = [layers[li].register_forward_pre_hook(lambda mod, inp, li=li: put(li)) for li in range(1, NL)]
            o = self.model(ids[:, m:e], past_key_values=cache, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize(); split_ms = (time.perf_counter() - t0) * 1e3
            for h_ in hs: h_.remove()
            th.join(); hr = recv_msg(self.sock)
            assert cache.get_seq_length() == e, (cache.get_seq_length(), e)
            first = o.logits[:, -1:].argmax(-1)
            lgs = o.logits[0, -1].float().cpu(); dlg = float((lgs - ref_lg).abs().max())
            dch = float((lgs - ch_lg).abs().max()) if ch_lg is not None else None
            dch_ref = float((ch_lg - ref_lg).abs().max()) if ch_lg is not None else None
            toks = [int(first)] + self.decode(cache, first, 31)
            agree = sum(int(x == y) for x, y in zip(toks, ref))
            prefix_agree = next((i for i, (x, y) in enumerate(zip(toks, ref)) if x != y), len(ref))
            self.emit(kind='pp', cell=cell, label=label, s=s, e=e, m=m, x=(m - s) / (e - s), cap_mbps=cap, rep=rep, split_ms=split_ms,
                      helper_comp_ms=hr['comp_ms'], helper_send_ms=hr['send_ms'], last_layer_arrival_ms=tarr[-1] * 1e3,
                      first_layer_arrival_ms=tarr[0] * 1e3, server_wait_ms=wait[0] * 1e3, nbytes=NL * per,
                      agree=agree, prefix_agree=prefix_agree, n_dec=len(ref), logit_maxabs_diff=dlg, logit_diff_vs_local_chunked=dch, local_chunked_vs_single=dch_ref)
            cache = None; o = None; pinned = None; torch.cuda.empty_cache()
        return split_ms

    def run_cell(self, cell, s, e, cap, reps, fracs):
        ids = ctx_ids(e + 1, self.dev)
        self.probe(f'{cell}-pre')
        send_msg(self.sock, {'type': 'prepare', 's': s, 'e': e}); recv_msg(self.sock)
        with torch.inference_mode():
            _, pref = prefix_cache(self.model, ids, s)
        torch.cuda.synchronize()
        singles = []; ref = None; ref_lg = None
        for rep in range(reps):
            ms, toks, lg = self.single(ids, pref, s, e); singles.append(ms)
            if ref is None: ref = toks; ref_lg = lg
            self.emit(kind='single', cell=cell, s=s, e=e, rep=rep, single_ms=ms, same_as_rep0=int(toks == ref),
                      logit_maxabs_diff_vs_rep0=float((lg - ref_lg).abs().max()))
        self.emit(kind='ref', cell=cell, s=s, e=e, ref=ref)
        for rep in range(reps):
            for x in fracs:
                m = s + int(round(x * (e - s)))
                if m <= s or m >= e: continue
                self.trial(cell, ids, pref, s, e, m, cap, ref, ref_lg, rep, 'grid')
        self.probe(f'{cell}-post')
        pref = None; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--role', required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--port', type=int, default=29540); ap.add_argument('--peer', default='')
    ap.add_argument('--out', default=''); ap.add_argument('--cells', default='')  # name:s:e:cap;...
    ap.add_argument('--reps', type=int, default=3); ap.add_argument('--fracs', default='0.25,0.375,0.5,0.625,0.75')
    ap.add_argument('--quit', action='store_true')
    a = ap.parse_args()
    if a.role == 'helper':
        return helper(a)
    sv = Server(a); fr = [float(x) for x in a.fracs.split(',')]
    for spec in a.cells.split(';'):
        if not spec: continue
        name, s, e, cap = spec.split(':'); sv.run_cell(name, int(s), int(e), int(cap), a.reps, fr)
    send_msg(sv.sock, {'type': 'quit' if a.quit else 'bye'}); sv.emit(kind='done')


if __name__ == '__main__':
    main()
