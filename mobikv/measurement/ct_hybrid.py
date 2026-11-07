#!/usr/bin/env python3
"""CT-T / CT-H: KV transfer and concurrent hybrid (recompute + transfer) repair.

sender  (holder A): keeps the full KV of the context resident on its GPU; on request it packs
                    the KV of tokens [m, L-1) (all layers, K and V) and streams it through an
                    application-level token bucket (64 KiB chunks, 128 KiB bucket).
receiver (edge B):  owns a stale replica [0, s); at t0 it asks A for [m, L-1) and, at the same
                    time, recomputes [s, m) on its own GPU.  Ready = both parts resident in one
                    cache.  It then greedily decodes 32 tokens and compares them with a
                    reference decode from a full local recompute.

Compute share of the receiver is set from outside with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE.
"""
import argparse, json, os, socket, struct, threading, time, statistics
import torch

CHUNK = 64 * 1024
BUCKET = 128 * 1024


def send_msg(sock, obj):
    b = json.dumps(obj).encode()
    sock.sendall(struct.pack('!I', len(b)) + b)


def recv_exact(sock, n):
    buf = bytearray(n); mv = memoryview(buf); got = 0
    while got < n:
        k = sock.recv_into(mv[got:], n - got)
        if k == 0:
            raise ConnectionError('eof')
        got += k
    return bytes(buf)


def recv_msg(sock):
    (n,) = struct.unpack('!I', recv_exact(sock, 4))
    return json.loads(recv_exact(sock, n))


def paced_send(sock, mv, cap_mbps, st=None):
    """Token bucket (64 KiB chunks, 128 KiB bucket).  st carries bucket state across calls of one request."""
    n = len(mv)
    if not cap_mbps:
        off = 0
        while off < n:
            off += sock.send(mv[off:off + 4 * 1024 * 1024])
        return
    rate = cap_mbps * 1e6 / 8.0
    if st is None:
        st = {}
    if 'tokens' not in st:
        st['tokens'] = float(BUCKET); st['last'] = time.perf_counter()
    off = 0
    while off < n:
        now = time.perf_counter(); st['tokens'] = min(BUCKET, st['tokens'] + (now - st['last']) * rate); st['last'] = now
        c = min(CHUNK, n - off)
        if st['tokens'] < c:
            time.sleep(max(0.0, (c - st['tokens']) / rate)); continue
        sock.sendall(mv[off:off + c]); off += c; st['tokens'] -= c


def load_model(path, dev):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(dev)
    m.eval(); return m


def ctx_ids(L, dev, seed=20261103):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1000, 100000, (1, 40000), generator=g)[:, :L].to(dev)


def sender(a):
    dev = torch.device('cuda:0')
    model = load_model(a.model, dev) if a.model else None
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', a.port)); srv.listen(1)
    print('listening', a.port, flush=True)
    kv = {}  # L -> list of (K, V)
    blob = None
    while True:
        conn, addr = srv.accept(); conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print('conn', addr, flush=True)
        try:
            while True:
                req = recv_msg(conn)
                if req['type'] == 'bye':
                    break
                if req['type'] == 'quit':
                    conn.close(); return
                if req['type'] == 'blob':  # CT-T and capacity probe: raw bytes from a pinned host buffer
                    n = req['nbytes']
                    if blob is None or blob.numel() < n:
                        blob = torch.zeros(max(n, 1 << 20), dtype=torch.uint8).pin_memory()
                    send_msg(conn, {'ok': 1, 'nbytes': n})
                    paced_send(conn, memoryview(blob.numpy())[:n], req.get('cap_mbps', 0))
                    continue
                if req['type'] == 'prepare':  # build the holder's resident KV for context length L-1 (untimed)
                    L = req['L']
                    if L not in kv:
                        kv.clear(); torch.cuda.empty_cache()
                        with torch.inference_mode():
                            o = model(ctx_ids(L - 1, dev), use_cache=True, logits_to_keep=1)
                        kv[L] = [(ly.keys, ly.values) for ly in o.past_key_values.layers]; o = None
                        torch.cuda.synchronize()
                    send_msg(conn, {'ok': 1}); continue
                if req['type'] == 'kv':  # pack [m, L-1) for all layers and stream it
                    # v1.1: per-layer pack on a helper thread, overlapped with sending (pipelined holder)
                    import queue
                    L, m = req['L'], req['m']
                    n_tok = (L - 1) - m
                    K0 = kv[L][0][0]
                    nbytes = n_tok * len(kv[L]) * 2 * K0.shape[1] * K0.shape[3] * 2
                    send_msg(conn, {'ok': 1, 'nbytes': nbytes, 'pack_ms': None})
                    q = queue.Queue(maxsize=4)

                    def packer():
                        for (K, V) in kv[L]:
                            q.put(torch.cat([K[0, :, m:L - 1, :].reshape(-1), V[0, :, m:L - 1, :].reshape(-1)]).view(torch.uint8).cpu())
                        q.put(None)
                    th = threading.Thread(target=packer); th.start()
                    cap = req.get('cap_mbps', 0); st = {}
                    while True:
                        part = q.get()
                        if part is None: break
                        paced_send(conn, memoryview(part.numpy()), cap, st)
                    th.join()
                    continue
        except ConnectionError:
            pass
        conn.close()


class Receiver:
    def __init__(self, a):
        self.a = a; self.dev = torch.device('cuda:0')
        self.sock = socket.create_connection((a.peer, a.port)); self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.out = open(a.out, 'a'); self.host = socket.gethostname()
        self.share = os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', '100')
        self.model = load_model(a.model, self.dev) if a.model else None
        if self.model is not None:
            c = self.model.config
            self.L_layers = c.num_hidden_layers; self.kvh = c.num_key_value_heads; self.hd = c.hidden_size // c.num_attention_heads

    def emit(self, **kw):
        kw.update(host=self.host, share=self.share, ts=time.time(), gpu=torch.cuda.get_device_name(0))
        self.out.write(json.dumps(kw) + '\n'); self.out.flush()

    def blob(self, nbytes, cap):
        buf = bytearray(nbytes); mv = memoryview(buf)
        t0 = time.perf_counter(); send_msg(self.sock, {'type': 'blob', 'nbytes': nbytes, 'cap_mbps': cap})
        recv_msg(self.sock); got = 0
        while got < nbytes:
            k = self.sock.recv_into(mv[got:], nbytes - got)
            if k == 0: raise ConnectionError
            got += k
        return (time.perf_counter() - t0) * 1e3

    def probe(self, label):
        n = 64 * 1024 * 1024
        ms = self.blob(n, 0)
        self.emit(kind='probe', label=label, nbytes=n, ms=ms, mbps=n * 8 / ms / 1e3)

    def decode(self, cache, first_tok, n=32):
        toks = []; tok = first_tok
        for _ in range(n):
            lo = self.model(tok, past_key_values=cache, use_cache=True).logits
            tok = lo[:, -1:].argmax(-1); toks.append(int(tok))
        return toks

    def stale(self, ids, s):
        from transformers.cache_utils import DynamicCache
        if s == 0:
            return DynamicCache()
        return self.model(ids[:, :s], use_cache=True, logits_to_keep=1).past_key_values

    def trial(self, cell, ids, L, s, m, cap, pinned, ref, rep, label):
        sync = torch.cuda.synchronize
        with torch.inference_mode():
            cache = self.stale(ids, s); sync()
            n_tx_tok = (L - 1) - m
            nbytes = n_tx_tok * self.L_layers * 2 * self.kvh * self.hd * 2
            res = {}

            def rx():
                t = time.perf_counter()
                if n_tx_tok > 0:
                    send_msg(self.sock, {'type': 'kv', 'L': L, 'm': m, 'cap_mbps': cap})
                    hdr = recv_msg(self.sock); assert hdr['nbytes'] == nbytes, (hdr, nbytes)
                    mv = memoryview(pinned.numpy())[:nbytes]; got = 0
                    while got < nbytes:
                        k = self.sock.recv_into(mv[got:], nbytes - got)
                        if k == 0: raise ConnectionError
                        got += k
                    res['pack_ms'] = hdr['pack_ms']
                res['tx_ms'] = (time.perf_counter() - t) * 1e3

            t0 = time.perf_counter()
            th = threading.Thread(target=rx); th.start()
            if m > s:
                self.model(ids[:, s:m], past_key_values=cache, use_cache=True, logits_to_keep=1)
            sync(); rc_ms = (time.perf_counter() - t0) * 1e3
            th.join(); join_ms = (time.perf_counter() - t0) * 1e3
            if n_tx_tok > 0:
                g = pinned[:nbytes].to(self.dev, non_blocking=True).view(torch.bfloat16).view(self.L_layers, 2, self.kvh, n_tx_tok, self.hd)
                for li in range(self.L_layers):
                    cache.update(g[li, 0].unsqueeze(0), g[li, 1].unsqueeze(0), li)
            sync(); ready_ms = (time.perf_counter() - t0) * 1e3
            assert cache.get_seq_length() == L - 1, (cache.get_seq_length(), L - 1)
            toks = self.decode(cache, ids[:, L - 1:L])
            agree = sum(int(x == y) for x, y in zip(toks, ref)); prefix_agree = next((i for i, (x, y) in enumerate(zip(toks, ref)) if x != y), len(ref))
            x = (m - s) / max(1, (L - 1 - s))
            self.emit(kind='hybrid', cell=cell, label=label, L=L, s=s, m=m, x=x, cap_mbps=cap, rep=rep, rc_ms=rc_ms if m > s else 0.0,
                      tx_ms=res['tx_ms'] if n_tx_tok > 0 else 0.0, pack_ms=res.get('pack_ms'), join_ms=join_ms, ready_ms=ready_ms,
                      nbytes=nbytes, agree=agree, prefix_agree=prefix_agree, n_dec=len(ref))
            cache = None; torch.cuda.empty_cache()
            return ready_ms, (rc_ms if m > s else 0.0), (res['tx_ms'] if n_tx_tok > 0 else 0.0)

    def run_cell(self, cell, L, s, cap, reps, splits):
        ids = ctx_ids(L, self.dev)
        self.probe(f'{cell}-pre')
        send_msg(self.sock, {'type': 'prepare', 'L': L}); recv_msg(self.sock)
        maxbytes = (L - 1 - s) * self.L_layers * 2 * self.kvh * self.hd * 2
        pinned = torch.empty(maxbytes, dtype=torch.uint8).pin_memory()
        with torch.inference_mode():
            torch.cuda.synchronize(); t = time.perf_counter()
            c = self.model(ids[:, :L - 1], use_cache=True, logits_to_keep=1).past_key_values
            torch.cuda.synchronize(); fr_ms = (time.perf_counter() - t) * 1e3
            ref = self.decode(c, ids[:, L - 1:L]); c = None; torch.cuda.empty_cache()
        self.emit(kind='ref', cell=cell, L=L, s=s, full_recompute_ms=fr_ms, ref=ref)
        span = L - 1 - s
        comp = {}  # x -> list of (rc, tx)
        for rep in range(reps):
            for x in splits:
                m = s + int(round(x * span))
                r, rc, tx = self.trial(cell, ids, L, s, m, cap, pinned, ref, rep, 'grid')
                comp.setdefault(x, []).append((rc, tx))
        # out-of-sample optimum: fit component curves on the grid splits, solve rc(x)=tx(x)
        xs = sorted(comp)
        rc_med = [statistics.median(v[0] for v in comp[x]) for x in xs]
        tx_med = [statistics.median(v[1] for v in comp[x]) for x in xs]

        def interp(xq, ys):
            for i in range(len(xs) - 1):
                if xs[i] <= xq <= xs[i + 1]:
                    w = (xq - xs[i]) / (xs[i + 1] - xs[i]); return ys[i] * (1 - w) + ys[i + 1] * w
            return ys[-1]
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if interp(mid, rc_med) < interp(mid, tx_med): lo = mid
            else: hi = mid
        xstar = (lo + hi) / 2; pred = max(interp(xstar, rc_med), interp(xstar, tx_med))
        self.emit(kind='xstar', cell=cell, xstar=xstar, pred_ready_ms=pred, xs=xs, rc_med=rc_med, tx_med=tx_med)
        for rep in range(reps):
            m = s + int(round(xstar * span))
            self.trial(cell, ids, L, s, m, cap, pinned, ref, rep, 'xstar')
        self.probe(f'{cell}-post')
        pinned = None; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--role', required=True)
    ap.add_argument('--model', default='')
    ap.add_argument('--port', type=int, default=29517)
    ap.add_argument('--peer', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--mode', default='hybrid')  # hybrid | transfer
    ap.add_argument('--cells', default='')  # name:L:s:cap;...
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--splits', default='1,0,0.5,0.25,0.75')
    ap.add_argument('--kv_bytes_per_token', type=int, default=57344)
    ap.add_argument('--quit', action='store_true')
    a = ap.parse_args()
    if a.role == 'sender':
        return sender(a)
    rv = Receiver(a)
    if a.mode == 'transfer':
        for rep in range(a.reps):
            rv.probe(f'T-pre-{rep}')
            for cap in [100, 300, 0]:
                for ntok in [256, 1024, 4096, 16384]:
                    n = ntok * a.kv_bytes_per_token
                    ms = rv.blob(n, cap)
                    rv.emit(kind='transfer', ntok=ntok, nbytes=n, cap_mbps=cap, rep=rep, ms=ms, mbps=n * 8 / ms / 1e3)
            rv.probe(f'T-post-{rep}')
    else:
        splits = [float(x) for x in a.splits.split(',')]
        for spec in a.cells.split(';'):
            if not spec: continue
            name, L, s, cap = spec.split(':')
            rv.run_cell(name, int(L), int(s), int(cap), a.reps, splits)
    send_msg(rv.sock, {'type': 'quit' if a.quit else 'bye'})
    rv.emit(kind='done')


if __name__ == '__main__':
    main()
