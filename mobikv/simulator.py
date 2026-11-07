#!/usr/bin/env python3
"""MobiKV trace-driven simulator (final A9 implementation).

Mobility: GeoLife trips (10 s grid, metres), randomly translated into a D x D square and
reflected at its borders; edge = pointy-top hexagonal cell of radius r.
Workload: Mooncake conversation trace; sessions rebuilt from 512-token block-hash prefix
chains; closed loop (next turn = parent completion + think time).
Costs: fitted from the GPU measurements (costs.json).  Every edge has one GPU (prefill-class
work, foreground preempts background) and one egress port for state transfer (same two
classes).  Decode uses the measured batched ITL.  KV store per edge with a byte budget.

Policies include MobiKV, MobiKV-DF (decode-first background replay), RPP, and
the six published-policy adaptations used in the paper.  A9 robustness knobs
cover estimated-cost noise, time-varying and heterogeneous links, helper
failures, KV compression, and delayed background work.  The simulator keeps
the original deterministic event model so a job log can be replayed by
``measurement/sv3_replay.py``.
"""
import argparse, bisect, collections, hashlib, heapq, json, math, random, sys, time
import numpy as np

INF = float('inf')


# ---------------------------------------------------------------- cost model
def _li(x, xs, ys):
    """Piecewise-linear interpolation in log-log space with linear extrapolation in x at the ends."""
    if x <= xs[0]:
        return ys[0] * (x / xs[0]) if len(xs) > 1 and x < xs[0] and False else ys[0]
    if x >= xs[-1]:
        sl = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) if len(xs) > 1 else ys[-1] / xs[-1]
        return ys[-1] + sl * (x - xs[-1])
    i = bisect.bisect_left(xs, x)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    w = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
    return math.exp(math.log(y0) * (1 - w) + math.log(y1) * w)


class Cost:
    def __init__(self, cfg, model, share, cap_mbps):
        c = cfg['compute'][f'{model}-g{share}']
        self.full_n = [p[0] for p in c['full']]; self.full_ms = [p[1] for p in c['full']]
        self.dS = sorted(int(k) for k in c['delta'])
        self.dtab = {s_: ([p[0] for p in c['delta'][str(s_)]], [p[1] for p in c['delta'][str(s_)]]) for s_ in self.dS}
        self.itl_tab = {int(k): v for k, v in c['itl'].items()}  # C -> [[b, ms], ...]
        self.kappa = cfg['kv_bytes_per_token'][model]
        self.kappa_tx = self.kappa  # A7: bytes per token on the wire (KV compression divides this)
        t = cfg['transfer']
        self.rate = cap_mbps * 1e6 / 8.0 * t['eff']
        self.tx_a = t['a_ms'] / 1e3
        self.ideal = False
        self.nc = 1.0; self.nx = 1.0; self.fac = 1.0  # A7: estimate noise (compute, transfer) and link-rate factor
        self.chunk = self.rc(0, 2048)

    def rc(self, s, e):
        return self._rc(s, e) * self.nc if self.nc != 1.0 else self._rc(s, e)

    def _rc(self, s, e):
        """Seconds to (re)compute tokens [s, e) on top of a resident prefix of length s (measured grid)."""
        n = e - s
        if n <= 0: return 0.0
        if s <= 0:
            return _li(n, self.full_n, self.full_ms) / 1e3
        if self.ideal:  # A3 sensitivity: efficient prefix-cache kernel (incremental cost of a cold prefill)
            f = lambda k: _li(k, self.full_n, self.full_ms) / 1e3
            return max(f(e) - f(s), f(n))
        S_ = self.dS
        def row(si):
            xs, ys = self.dtab[si]; return _li(n, xs, ys)
        if s <= S_[0]: return row(S_[0]) / 1e3
        if s >= S_[-1]:
            a_, b_ = row(S_[-2]), row(S_[-1])
            return max(b_, b_ + (b_ - a_) * (s - S_[-1]) / (S_[-1] - S_[-2])) / 1e3
        i = bisect.bisect_left(S_, s); s0, s1 = S_[i - 1], S_[i]; w = (s - s0) / (s1 - s0)
        return (row(s0) * (1 - w) + row(s1) * w) / 1e3

    def tx(self, n, lm=1.0):
        if n <= 0: return 0.0
        if self.nx == 1.0 and self.fac == 1.0 and self.kappa_tx == self.kappa and lm == 1.0:
            return self.tx_a + n * self.kappa / self.rate
        if lm == 1.0:
            return (self.tx_a + n * self.kappa_tx / (self.rate * self.fac)) * self.nx
        return (self.tx_a + n * self.kappa_tx / (self.rate * self.fac * lm)) * self.nx  # A9: slower link

    def itl(self, b, C):
        Cs = sorted(self.itl_tab)
        def at(Ck):
            tab = self.itl_tab[Ck]; bs = [x[0] for x in tab]; ms = [x[1] for x in tab]
            if b <= bs[0]: return ms[0]
            if b >= bs[-1]: return ms[-1] * b / bs[-1]  # beyond the measured batch: linear in b
            return float(np.interp(b, bs, ms))
        if C <= Cs[0]: v = at(Cs[0])
        elif C >= Cs[-1]: v = at(Cs[-1]) * (1 + 0.5 * (C - Cs[-1]) / Cs[-1])
        else:
            i = bisect.bisect_left(Cs, C); c0, c1 = Cs[i - 1], Cs[i]; w = (C - c0) / (c1 - c0)
            v = at(c0) * (1 - w) + at(c1) * w
        return v / 1e3

    def split(self, v0, hi, wg, wt, lm=1.0):
        """Recompute [v0,m) after wait wg, transfer [m,hi) after wait wt, in parallel.
        Returns (m, ready) minimising max of the two."""
        if hi <= v0: return v0, wg
        f = lambda m: (wg + self.rc(v0, m), wt + self.tx(hi - m, lm))
        lo, up = v0, hi
        for _ in range(30):
            mid = (lo + up) // 2
            a, b = f(mid)
            if a < b: lo = mid + 1
            else: up = mid
            if up - lo <= 1: break
        best = min((max(*f(m)), m) for m in {v0, hi, lo, up})
        return best[1], best[0]


# ---------------------------------------------------------------- resources
class Job:
    __slots__ = ('rem', 'elig', 'done', 'dead')
    def __init__(self, d, t):
        self.rem = d; self.elig = t; self.done = None; self.dead = False


class Res:
    """One server; foreground preempts background at chunk granularity: a foreground job that finds a
    background chunk in service waits for that chunk (non-preemptive chunk, preemptive between chunks)."""
    __slots__ = ('fg_free', 'iv', 'bgq', 'bg_t', 'busy_fg', 'busy_bg', 'chunk', 'interf', 'recent', 'vq', 'vq_t', 'log', 'blog', 'bblog', 'du', 'dus')
    CHUNK = 0.1
    def __init__(self, chunk=None):
        self.fg_free = 0.0; self.iv = collections.deque(); self.bgq = collections.deque(); self.bg_t = 0.0
        self.busy_fg = 0.0; self.busy_bg = 0.0; self.chunk = Res.CHUNK if chunk is None else chunk; self.interf = 0.0
        self.recent = collections.deque(); self.vq = 0.0; self.vq_t = 0.0; self.log = None; self.blog = None; self.bblog = None; self.du = None; self.dus = None

    def fg(self, t, d, meta=None):
        st = max(t, self.fg_free)
        if d > 0 and self.bgq and st == t:
            self.advance(t)
            if self.bgq:
                h = next((j for j in self.bgq if not j.dead), None)
                if h is not None:
                    dl = min(self.chunk, h.rem); st = t + dl; self.interf += dl
        en = st + d
        if self.log is not None and meta is not None and d > 0: self.log.append(('fg', t, d, st, en) + tuple(meta))  # A7: joblog v2
        if d > 0:
            self.iv.append((st, en)); self.fg_free = en; self.busy_fg += d; self.recent.append((t, d))
            if self.blog is not None: self.blog.append((st, en))
        return st, en

    def wait(self, t):
        return max(0.0, self.fg_free - t)

    def util(self, t, W=300.0):
        while self.recent and self.recent[0][0] < t - W: self.recent.popleft()
        return min(1.0, sum(d for _, d in self.recent) / W)

    def vq_update(self, t, beta):
        """Virtual budget queue for background work: Q <- max(0, Q - beta*dt)."""
        self.vq = max(0.0, self.vq - beta * (t - self.vq_t)); self.vq_t = t
        return self.vq

    def advance(self, t):
        cur = self.bg_t
        while self.bgq and cur < t:
            j = self.bgq[0]
            if j.dead:
                self.bgq.popleft(); continue
            cur = max(cur, j.elig)
            while self.iv and self.iv[0][1] <= cur: self.iv.popleft()
            if self.iv and self.iv[0][0] <= cur:
                cur = self.iv[0][1]; continue
            nd_ = INF
            if self.du is not None:  # A8 strict decode priority: background replay waits while any stream decodes here
                k = bisect.bisect_right(self.dus, cur) - 1
                if k >= 0 and self.du[k][1] > cur:
                    cur = self.du[k][1]; continue
                if k + 1 < len(self.du): nd_ = self.du[k + 1][0]
            nb = min(self.iv[0][0] if self.iv else INF, nd_)
            end = min(nb, t)
            if end <= cur: break
            av = end - cur
            if j.rem <= av:
                tg = self.bblog if self.bblog is not None else self.blog
                if tg is not None: tg.append((cur, cur + j.rem))
                cur += j.rem; self.busy_bg += j.rem; j.rem = 0.0; j.done = cur; self.bgq.popleft()
            else:
                tg = self.bblog if self.bblog is not None else self.blog
                if tg is not None: tg.append((cur, end))
                j.rem -= av; self.busy_bg += av; cur = end
        if not self.bgq:
            while self.iv and self.iv[0][1] <= t: self.iv.popleft()
        self.bg_t = max(self.bg_t, t)

    def add_dec(self, s, e):
        """A8: register a decode window [s, e) (merged, disjoint, sorted)."""
        if e <= s: return
        i = bisect.bisect_left(self.dus, s)
        if i > 0 and self.du[i - 1][1] >= s:
            i -= 1; s = self.du[i][0]; e = max(e, self.du[i][1]); del self.du[i]; del self.dus[i]
        while i < len(self.du) and self.du[i][0] <= e:
            e = max(e, self.du[i][1]); del self.du[i]; del self.dus[i]
        self.du.insert(i, [s, e]); self.dus.insert(i, s)

    def bg(self, t, d, meta=None):
        if self.log is not None and meta is not None and d > 0: self.log.append(('bg', t, d, None, None) + tuple(meta))
        self.advance(t); j = Job(d, t)
        if d <= 0: j.done = t
        else: self.bgq.append(j)
        return j

    def backlog(self, t):
        self.advance(t); return sum(j.rem for j in self.bgq if not j.dead)


# ---------------------------------------------------------------- geometry
SQ3 = math.sqrt(3.0)
DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


def hex_of(x, y, r):
    q = (SQ3 / 3 * x - y / 3) / r; rr = (2.0 / 3 * y) / r
    cx, cz = q, rr; cy = -cx - cz
    rx, ry, rz = round(cx), round(cy), round(cz)
    dx, dy, dz = abs(rx - cx), abs(ry - cy), abs(rz - cz)
    if dx > dy and dx > dz: rx = -ry - rz
    elif dy > dz: ry = -rx - rz
    else: rz = -rx - ry
    return (int(rx), int(rz))


def hexdist(a, b):
    dq = a[0] - b[0]; dr = a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def reflect(v, D):
    v = v % (2 * D)
    return np.where(v > D, 2 * D - v, v) if isinstance(v, np.ndarray) else (2 * D - v if v > D else v)


# ---------------------------------------------------------------- trace
def load_sessions(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    key = {}; parent = [-1] * len(rows)
    for i, r in enumerate(rows):
        h = r['hash_ids']
        for ln in range(len(h) - 1, 1, -1):
            k = tuple(h[:ln])
            if k in key: parent[i] = key[k]; break
        key[tuple(h)] = i; key[tuple(h[:-1])] = i
    sess = [-1] * len(rows); ns = 0; turn = [0] * len(rows)
    for i in range(len(rows)):
        if parent[i] < 0: sess[i] = ns; ns += 1; turn[i] = 1
        else: sess[i] = sess[parent[i]]; turn[i] = turn[parent[i]] + 1
    reqs = []
    for i, r in enumerate(rows):
        p = parent[i]
        if p >= 0:
            pr = rows[p]; R = min(pr['input_length'] + pr['output_length'], r['input_length'] - 16)
            R = max(R, 0); q = r['input_length'] - R
        else:
            R = 0; q = r['input_length']
        reqs.append(dict(i=i, t=r['timestamp'] / 1e3, inp=r['input_length'], out=max(1, r['output_length']), parent=p,
                         sess=sess[i], turn=turn[i], R=R, q=q))
    # Sessions whose context would exceed the model's native 32k window (outside the measured cost grid)
    # are scaled down by one session-wide factor, which keeps the turn-to-turn growth structure.
    CL = 32768
    bys = collections.defaultdict(list)
    for r in reqs: bys[r['sess']].append(r)
    ver = {}
    for sid, rs in bys.items():
        mx = max(r['R'] + r['q'] + r['out'] for r in rs)
        if mx <= CL:
            for r in rs: ver[r['i']] = r['R'] + r['q'] + r['out']
            continue
        f = CL / mx
        for r in rs:  # index order: parents first
            inp = max(2, int(f * r['inp'])); out = max(1, int(f * r['out']))
            R = min(int(f * r['R']), ver[r['parent']]) if r['parent'] >= 0 else 0
            R = min(R, inp - 1)
            r.update(inp=inp, out=out, R=R, q=inp - R, clipped=1)
            ver[r['i']] = r['R'] + r['q'] + r['out']
    nchild = collections.Counter(p for p in parent if p >= 0)
    branching = sum(1 for p, c in nchild.items() if c > 1)
    return reqs, ns, branching


# ---------------------------------------------------------------- simulator
class Sim:
    def __init__(self, a, cfg, trips):
        self.a = a; self.cost = Cost(cfg, a.model, a.share, a.cap)
        self.rng = random.Random(a.seed); self.nprng = np.random.default_rng(a.seed)
        self.reqs, self.nsess, self.branching = load_sessions(a.trace)
        if a.load != 1.0:
            self.scale_load(a.load)
        self.D = a.side_cells * a.r if a.side_cells else a.side
        self.trips = trips
        self.prepare_mobility()
        R_ = a.r; rng_q = int(self.D / (R_ * SQ3)) + 3; rng_r = int(self.D / (R_ * 1.5)) + 3
        self.area_edges = [(q, r) for r in range(-2, rng_r) for q in range(-rng_r - 2, rng_q + 2)
                           if 0 <= R_ * SQ3 * (q + r / 2.0) <= self.D and 0 <= R_ * 1.5 * r <= self.D]
        def mkgpu():
            r = Res(a.gpu_chunk)
            if a.joblog: r.log = []
            if a.tpot: r.blog = []
            if a.tpot and a.bg_dfb: r.bblog = []
            if a.bg_dfb: r.du = []; r.dus = []
            return r
        self.gpu = collections.defaultdict(mkgpu); self.egr = collections.defaultdict(lambda: Res(a.link_chunk))
        self.ing = collections.defaultdict(lambda: Res(a.link_chunk))  # A3: ingress, used only by parallel prefill
        self.cost.ideal = bool(a.delta_ideal)
        # ---- A7: estimates used by every controller (noise, stale link rate) vs the actual costs; KV compression
        self.cost.kappa_tx = self.cost.kappa / a.kv_comp
        if a.pred_err > 0 or a.bw_var or a.kv_comp != 1.0:
            import copy
            self.ce = copy.copy(self.cost)
        else:
            self.ce = self.cost
        self.err_rng = np.random.default_rng(a.seed + 7919)
        self.emult = {}  # A9: per-edge link-rate multiplier (heterogeneous links)
        self.hf_rng = random.Random(a.seed * 977 + 11)  # A9: helper failures (drawn only when --helper_fail > 0)
        self.bw_t = [0.0]; self.bw_f = [1.0]
        if a.bw_var:
            rb = random.Random(a.seed * 31 + 5); tt = 0.0; slow = False
            while tt < 2e4:
                tt += rb.expovariate(1.0 / (a.bw_dwell_s if slow else a.bw_dwell_n)); slow = not slow
                self.bw_t.append(tt); self.bw_f.append(a.bw_slow if slow else 1.0)
        self.dec_wall = {}; self.pp_search_wall = 0.0; self.pp_evals = 0
        self.reqlog = [] if a.joblog else None
        self.decinfo = {}
        self.Z = collections.defaultdict(float); self.Zt = collections.defaultdict(float)
        self.oyQ = 0.0; self.oyT = 0.0; self.svc = {}
        self.store = collections.defaultdict(dict)  # edge -> sid -> [len, last_use]
        self.used = collections.defaultdict(float)
        self.pinned = collections.defaultdict(collections.Counter)
        self.dec = collections.defaultdict(list)  # edge -> heap of decode end times
        self.primary = {}  # sid -> edge holding the latest version
        self.version = {}  # sid -> latest version length
        self.bgjobs = collections.defaultdict(list)  # (sid, edge) -> list of [ver, gpu_job, tx_job, src]
        self.M = a.mem_gb * 2**30
        self.stats = collections.Counter(); self.out = []
        self.child = collections.defaultdict(list)
        for r in self.reqs:
            if r['parent'] >= 0: self.child[r['parent']].append(r['i'])
        self.think_tune = []
        self.last_complete = {}
        self.turns_done = collections.Counter()
        self.pcont = None
        self.pred_cache = {}

    # ---- load scaling: duplicate sessions with a random time shift (load > 1) or thin them (load < 1)
    def scale_load(self, f):
        bys = collections.defaultdict(list)
        for r in self.reqs: bys[r['sess']].append(r)
        new = []; ns = 0; base = 0
        copies = int(math.floor(f)); frac = f - copies
        rng = random.Random(self.a.seed + 7)
        for sid, rs in bys.items():
            k = copies + (1 if rng.random() < frac else 0)
            for c in range(k):
                shift = 0.0 if c == 0 else rng.uniform(-300, 300)
                idmap = {}
                for r in rs:
                    nr = dict(r); nr['i'] = len(new); idmap[r['i']] = nr['i']
                    nr['sess'] = ns; nr['t'] = min(3599.0, max(0.0, r['t'] + shift))
                    nr['parent'] = idmap.get(r['parent'], -1) if r['parent'] >= 0 else -1
                    new.append(nr)
                ns += 1
        self.reqs = new; self.nsess = ns

    # ---- mobility
    def prepare_mobility(self):
        T = self.trips; a = self.a
        lens = T['trip_len']; test = T['trip_test']
        self.trip_idx_train = [i for i in range(len(lens)) if not test[i] and lens[i] >= 60]
        self.trip_idx_test = [i for i in range(len(lens)) if test[i] and lens[i] >= 60]
        if a.mode_filter:
            modes = set(a.mode_filter.split(','))
            self.trip_idx_test = [i for i in self.trip_idx_test if T['trip_mode'][i] in modes] or self.trip_idx_test
            self.trip_idx_train = [i for i in self.trip_idx_train if T['trip_mode'][i] in modes] or self.trip_idx_train
        first_t = {}
        for r in self.reqs:
            if r['parent'] < 0 and r['sess'] not in first_t: first_t[r['sess']] = r['t']
        self.sess_start = first_t
        self.sess_trip = {}
        for sid, t0 in first_t.items():
            pool = self.trip_idx_test if t0 >= a.eval_start else self.trip_idx_train
            ti = pool[self.rng.randrange(len(pool))]
            L = int(lens[ti]); off = int(T['trip_off'][ti])
            k0 = self.rng.randrange(max(1, L - 30))
            xy0 = T['xy'][off + k0]
            dx = self.rng.uniform(0, self.D) - xy0[0]; dy = self.rng.uniform(0, self.D) - xy0[1]
            self.sess_trip[sid] = (off, L, k0, dx, dy, t0)

    def pos(self, sid, t):
        off, L, k0, dx, dy, t0 = self.sess_trip[sid]
        k = k0 + int(self.a.speedup * max(0.0, t - t0) / 10.0)
        per = 2 * (L - 1)
        k = k % per if per > 0 else 0
        if k >= L: k = per - k
        x, y = self.trips['xy'][off + k]
        return reflect(x + dx, self.D), reflect(y + dy, self.D)

    def edge_at(self, sid, t):
        x, y = self.pos(sid, t); return hex_of(x, y, self.a.r)

    def moves(self, sid, t0, t1):
        """Edge changes strictly inside (t0, t1), at 10 s resolution."""
        res = []; e = self.edge_at(sid, t0); t = t0 + 10.0
        while t < t1:
            e2 = self.edge_at(sid, t)
            if e2 != e: res.append((t, e2)); e = e2
            t += 10.0
        return res

    # ---- KV store
    def touch(self, e, sid, t):
        st = self.store[e]
        if sid in st: st[sid][1] = t

    def put(self, e, sid, ln, t):
        st = self.store[e]; k = self.cost.kappa
        old = st.get(sid)
        if old: self.used[e] -= old[0] * k
        st[sid] = [ln, t]; self.used[e] += ln * k
        if self.used[e] > self.M: self.evict(e, t, keep=sid)

    def drop(self, e, sid):
        st = self.store[e]
        if sid in st:
            self.used[e] -= st[sid][0] * self.cost.kappa; del st[sid]
            if self.primary.get(sid) == e:
                self.primary.pop(sid, None)

    def evict(self, e, t, keep):
        st = self.store[e]
        cands = [s for s in st if s != keep and not self.pinned[e][s]]
        if self.a.evict == 'value':
            cands.sort(key=lambda s: self.value(e, s, t))
        else:
            cands.sort(key=lambda s: st[s][1])
        for s in cands:
            if self.used[e] <= self.M: break
            self.drop(e, s); self.stats['evictions'] += 1
        if self.used[e] > self.M: self.stats['mem_overflow'] += 1

    def value(self, e, sid, t):
        """Expected reconstruction cost avoided per byte (GreedyDual-style)."""
        ln = self.store[e][sid][0]
        if ln <= 0: return 0.0
        k = self.turns_done.get(sid, 1)
        pc = self.pcont_k(k)
        el = t - self.last_complete.get(sid, t)
        S = self.surv(el)
        pnext = pc * S / max(1e-9, (1 - pc) + pc * S)
        ploc = self.pred_cache.get((sid, e), 0.5 if self.primary.get(sid) == e else 0.15)
        return pnext * ploc * self.cost.rc(0, ln) / (ln * self.cost.kappa)

    # ---- empirical think-time model (fitted on the tuning half only)
    def fit_think(self):
        th = sorted(self.think_tune) or [60.0]
        self.th_sorted = np.array(th)
        cnt = collections.Counter(); cont = collections.Counter()
        for r in self.reqs:
            if self.sess_start.get(r['sess'], 0) < self.a.eval_start:
                cnt[r['turn']] += 1
                if self.child.get(r['i']): cont[r['turn']] += 1
        self.pc_tab = {k: cont[k] / cnt[k] for k in cnt if cnt[k] >= 20}

    def pcont_k(self, k):
        if k in self.pc_tab: return self.pc_tab[k]
        ks = [x for x in self.pc_tab if x <= k]
        return self.pc_tab[max(ks)] if ks else 0.3

    def surv(self, el):
        n = len(self.th_sorted)
        return max(0.0, 1.0 - np.searchsorted(self.th_sorted, el) / n)

    def remaining_think_samples(self, el, n):
        th = self.th_sorted; idx = np.searchsorted(th, el)
        if idx >= len(th): return np.full(n, 5.0)
        return th[self.nprng.integers(idx, len(th), n)] - el

    # ---- displacement model (train trips only)
    def fit_displacement(self):
        T = self.trips; lags = [1, 2, 4, 8, 16, 32, 64]  # x10 s
        bins = [0.5, 2.0, 5.0, 10.0, INF]
        tab = {(b, l): [] for b in range(len(bins)) for l in lags}
        rng = random.Random(1)
        for ti in self.trip_idx_train:
            off = int(T['trip_off'][ti]); L = int(T['trip_len'][ti]); xy = T['xy'][off:off + L]
            for k in range(6, L, 3):
                v = (xy[k] - xy[k - 6]) / 60.0; sp = float(np.hypot(*v))
                b = next(i for i, ub in enumerate(bins) if sp < ub)
                h = v / sp if sp > 0.2 else np.array([1.0, 0.0])
                for l in lags:
                    if k + l < L:
                        d = xy[k + l] - xy[k]
                        tab[(b, l)].append((float(d @ h), float(-d[0] * h[1] + d[1] * h[0])))
        self.disp = {k: np.array(v, dtype=np.float32) if v else np.zeros((1, 2), np.float32) for k, v in tab.items()}
        self.lags = lags; self.vbins = bins

    def predict(self, sid, t, el):
        """Distribution over edges of the next request, given the elapsed think time el."""
        x, y = self.pos(sid, t); xp, yp = self.pos(sid, max(self.sess_start[sid], t - 60.0))
        v = np.array([x - xp, y - yp]) / 60.0; sp = float(np.hypot(*v))
        b = next(i for i, ub in enumerate(self.vbins) if sp < ub)
        h = v / sp if sp > 0.2 else np.array([1.0, 0.0])
        n = self.a.pred_samples
        rem = self.remaining_think_samples(el, n)
        lagi = np.clip(np.searchsorted(np.array(self.lags) * 10.0, rem), 0, len(self.lags) - 1)
        P = collections.Counter()
        for li in set(lagi.tolist()):
            m = int((lagi == li).sum()); tab = self.disp[(b, self.lags[li])]
            sel = tab[self.nprng.integers(0, len(tab), m)]
            dx = sel[:, 0] * h[0] - sel[:, 1] * h[1]; dy = sel[:, 0] * h[1] + sel[:, 1] * h[0]
            px = reflect(x + dx, self.D); py = reflect(y + dy, self.D)
            for X, Y in zip(px.tolist(), py.tolist()):
                P[hex_of(X, Y, self.a.r)] += 1
        return {e: c / n for e, c in P.items()}, float(np.median(rem))

    # ---- A7: link-rate regime and estimate noise, set at the start of every event
    def bwf(self, t):
        return self.bw_f[bisect.bisect_right(self.bw_t, t) - 1]

    def set_env(self, t):
        a = self.a
        if a.bw_var:
            self.cost.fac = self.bwf(t)
            self.ce.fac = 1.0 if a.bw_lag < 0 else self.bwf(max(0.0, t - a.bw_lag))
        if a.pred_err > 0:
            z = self.err_rng.standard_normal(2)
            self.ce.nc = float(math.exp(a.pred_err * z[0])); self.ce.nx = float(math.exp(a.pred_err * z[1]))

    # ---- A9: heterogeneous links.  Each edge's network is slow (rate x link_het_mult) with probability link_het, drawn
    # once per edge from (seed, edge); a transfer between two edges runs at the slower endpoint's rate.  lm(): actual;
    # lme(): what the controllers assume (actual if --link_het_known 1, nominal otherwise).
    def em(self, e):
        v = self.emult.get(e)
        if v is None:
            v = self.a.link_het_mult if random.Random(f'het-{self.a.seed}-{e[0]}-{e[1]}').random() < self.a.link_het else 1.0
            self.emult[e] = v
        return v
    def lm(self, a_, b_):
        if not self.a.link_het or a_ is None or b_ is None: return 1.0
        return min(self.em(a_), self.em(b_))
    def lme(self, a_, b_):
        return self.lm(a_, b_) if self.a.link_het_known else 1.0

    # ---- request-time options
    def rtt(self, a_, b_):
        if a_ == b_: return 0.0
        return (self.a.rtt_base_ms + self.a.rtt_hop_ms * hexdist(a_, b_)) / 1e3

    def gpu_factor(self, e, t):
        h = self.dec[e]
        while h and h[0] <= t: heapq.heappop(h)
        nd = len(h)
        if nd == 0: return 1.0, 0
        return 1.0 + self.cost.itl(nd, 8192) / max(1e-3, self.cost.chunk), nd

    def harvest(self, sid, e, t):
        """Turn finished background jobs for (sid, e) into a replica; cancel stale unfinished ones on request."""
        lst = self.bgjobs.get((sid, e))
        if not lst: return
        keep = []
        for rec in lst:
            ver, gj, tj, src = rec
            self.gpu[e].advance(t)
            if src is not None: self.egr[src].advance(t)
            if gj.done is not None and tj.done is not None and max(gj.done, tj.done) <= t:
                cur = self.store[e].get(sid, [0])[0]
                if ver > cur:
                    self.put(e, sid, ver, max(gj.done, tj.done)); self.stats['bg_replicas_ready'] += 1
            else:
                keep.append(rec)
        self.bgjobs[(sid, e)] = keep

    def cancel_bg(self, sid, e):
        for ver, gj, tj, src in self.bgjobs.get((sid, e), []):
            gj.dead = True; tj.dead = True; self.stats['bg_cancelled'] += 1
        self.bgjobs[(sid, e)] = []

    def options(self, r, e, t):
        """Candidate plans for request r arriving at edge e at time t: list of (est_ttft, plan)."""
        sid = r['sess']; R = r['R']; q = r['q']; c = self.ce; pol = self.a.policy  # A7: estimates
        opts = []
        fe, _ = self.gpu_factor(e, t)
        holders = {x: v[0] for x, v in ((x, self.store[x].get(sid)) for x in self.holders_of(sid)) if v}
        ve = holders.get(e, 0)
        src = max(holders.items(), key=lambda kv: kv[1]) if holders else (None, 0)
        # (1) execute at e: repair [ve, R) from the best holder by split, then prefill q
        if R > 0:
            hi = min(R, src[1]) if src[0] is not None and src[0] != e else ve
            wg = self.gpu[e].wait(t); wt = self.egr[src[0]].wait(t) if src[0] is not None and src[0] != e else INF
            if pol in ('follow_rc',) or hi <= ve or wt == INF or not self.a.hybrid_ok:
                m_rc = hi  # recompute-only (or no transfer source)
            else:
                m_rc = None
            modes = []
            if pol == 'follow_tx':
                modes = [('tx', ve)]
            elif pol == 'follow_rc':
                modes = [('rc', hi)]
            elif not self.a.hybrid_ok:
                modes = [('rc', hi), ('tx', ve)]
            else:
                m, _ = c.split(ve, hi, wg, wt if wt < INF else 0.0, self.lme(src[0], e)) if hi > ve else (ve, 0)
                modes = [('hy', m)]
            for name, m in modes:
                if hi <= ve: m = ve
                if wt == INF: m = hi
                rc_t = wg + fe * c.rc(ve, m); tx_t = (wt + c.tx(hi - m, self.lme(src[0], e))) if hi > m else 0.0
                ready = max(rc_t, tx_t); tail = fe * c.rc(hi, R) if hi < R else 0.0
                est = ready + tail + fe * c.rc(R, R + q)
                opts.append((est, ('local', e, ve, m, hi, src[0])))
        else:
            est = self.gpu[e].wait(t) + fe * c.rc(0, q)
            opts.append((est, ('local', e, 0, 0, 0, None)))
            if self.a.route_first == 'p2c':
                for j in [(e[0] + dq, e[1] + dr) for dq, dr in DIRS]:
                    if not self.in_area(j): continue
                    fj, _ = self.gpu_factor(j, t); d = self.rtt(e, j)
                    opts.append((d + self.gpu[j].wait(t + d / 2) + fj * c.rc(0, q), ('hold', j, 0)))
        # (2) HOLD: execute at a holder x != e
        if pol not in ('follow_rc', 'follow_tx') and R > 0:
            for x, vx in holders.items():
                if x == e: continue
                if pol == 'anchor' and x != self.anchor.get(sid): continue
                fx, _ = self.gpu_factor(x, t)
                d = self.rtt(e, x)
                est = d + self.gpu[x].wait(t + d / 2) + fx * (c.rc(vx, R) + c.rc(R, R + q))
                opts.append((est, ('hold', x, vx)))
        if pol == 'anchor' and R > 0:
            an = self.anchor.get(sid)
            hold = [o for o in opts if o[1][0] == 'hold']
            if an is not None and an != e:
                if hold: return hold
                # anchor lost the state: rebuild at the anchor
                fx, _ = self.gpu_factor(an, t); d = self.rtt(e, an)
                return [(d + self.gpu[an].wait(t) + fx * (c.rc(0, R) + c.rc(R, R + q)), ('hold', an, 0))]
            return [o for o in opts if o[1][0] == 'local']
        return opts

    # ---- A5: baselines from published work, adapted to this setting
    def plan_at(self, r, a_, x, t, mode='hy'):
        """Estimated TTFT and plan for executing request r (arriving at access edge a_) at edge x: repair the
        missing range at x from the freshest other holder (hy: rho split [Cake-style]; best: the faster of
        transfer and recompute [Mooncake-style]; tx; rc), then prefill the new tokens."""
        sid = r['sess']; R = r['R']; q = r['q']; c = self.ce  # A7: estimates
        d = self.rtt(a_, x); ta = t + d / 2
        fx, _ = self.gpu_factor(x, t)
        wg = self.gpu[x].wait(ta)
        if R <= 0:
            return d + wg + fx * c.rc(0, q), ('local', x, 0, 0, 0, None)
        holders = {h: v[0] for h, v in ((h, self.store[h].get(sid)) for h in self.holders_of(sid)) if v}
        vx = holders.get(x, 0)
        others = [(h, v) for h, v in holders.items() if h != x]
        src = max(others, key=lambda kv: kv[1]) if others else (None, 0)
        hi = min(R, src[1]) if src[0] is not None else vx
        if hi <= vx:
            hi = vx; m = vx; wt = 0.0
        else:
            wt = self.egr[src[0]].wait(ta)
            if mode == 'hy': m, _ = c.split(vx, hi, wg, wt, self.lme(src[0], x))
            elif mode == 'best': m = hi if wg + fx * c.rc(vx, hi) <= wt + c.tx(hi - vx, self.lme(src[0], x)) else vx
            elif mode == 'tx': m = vx
            else: m = hi
        rc_t = wg + fx * c.rc(vx, m); tx_t = (wt + c.tx(hi - m, self.lme(src[0], x))) if hi > m else 0.0
        ready = max(rc_t, tx_t); tail = fx * c.rc(hi, R) if hi < R else 0.0
        return d + ready + tail + fx * c.rc(R, R + q), ('local', x, vx, m, hi, src[0] if hi > vx else None)

    BASELINES = ('mooncake', 'ouyang', 'mdp_thr', 'preble')

    def choose_baseline(self, r, a_, t):
        pol = self.a.policy; sid = r['sess']; R = r['R']; c = self.ce  # A7: estimates
        if pol == 'mooncake':
            # Mooncake Conductor (FAST'25): pick the instance with the lowest estimated TTFT = queueing + loading
            # the cached prefix from wherever it is (or recomputing it) + prefill, over all instances.
            return min((self.plan_at(r, a_, x, t, 'best') for x in self.area_edges), key=lambda o: o[0])
        if pol == 'ouyang':
            # Follow Me at the Edge (Ouyang et al., JSAC'18): Lyapunov drift-plus-penalty placement minimising
            # V * latency + Q * migration cost under a long-term migration budget (here: GB of KV moved, GB/s).
            Q = max(0.0, self.oyQ - self.a.oy_B * (t - self.oyT)); self.oyT = t
            cands = {j for j in self.area_edges if hexdist(j, a_) <= self.a.oy_hops} | set(self.holders_of(sid))
            best = None
            for x in cands:
                est, plan = self.plan_at(r, a_, x, t, 'hy')
                mig = max(0, plan[4] - plan[3]) * c.kappa_tx / 1e9 if R > 0 else 0.0
                sc = self.a.oy_V * est + Q * mig
                if best is None or sc < best[0]: best = (sc, est, plan, mig)
            self.oyQ = Q + best[3]
            return best[1], best[2]
        if pol == 'mdp_thr':
            # Wang et al. (ToN'19): the optimal MDP migration policy is distance-threshold: migrate the service
            # to the user's cell once the user is >= k hops away, otherwise serve from the current location.
            y = self.svc.get(sid)
            if R <= 0 or y is None or hexdist(a_, y) >= self.a.mdp_k:
                self.svc[sid] = a_; return self.plan_at(r, a_, a_, t, 'hy')
            return self.plan_at(r, a_, y, t, 'hy')
        if pol == 'preble':
            # Preble E2 (ICLR'25): exploit the instance with the longest cached prefix if the cached part exceeds
            # the missed part; otherwise explore: the instance with the lowest load cost (recent prefill-class
            # work in a window plus this request's own cost there).
            holders = {h: v[0] for h, v in ((h, self.store[h].get(sid)) for h in self.holders_of(sid)) if v}
            if holders:
                hstar, vh = max(holders.items(), key=lambda kv: kv[1])
                cached = min(R, vh)
                if cached > R + r['q'] - cached:
                    return self.plan_at(r, a_, hstar, t, 'hy')
            best = None
            for x in self.area_edges:
                est, plan = self.plan_at(r, a_, x, t, 'hy')
                d = self.rtt(a_, x)
                load = self.gpu[x].util(t) * 300.0 + (est - d - self.gpu[x].wait(t + d / 2))
                if best is None or load < best[0]: best = (load, est, plan)
            return best[1], best[2]
        raise ValueError(pol)

    def holders_of(self, sid):
        return [x for x in self.holder_idx.get(sid, ()) if sid in self.store[x]]

    def execute(self, r, e, t, plan):
        sid = r['sess']; R = r['R']; q = r['q']; c = self.cost
        kind = plan[0]
        if kind == 'local':
            _, x, ve, m, hi, srcx = plan
            dx = self.rtt(e, x) if x != e else 0.0; ts = t + dx / 2 if dx else t
            fe, nd = self.gpu_factor(x, t)
            rc_job = self.gpu[x].fg(ts, fe * c.rc(ve, m), (ve, m, r['i'])) if m > ve else (ts, ts)
            tx_job = self.egr[srcx].fg(ts, c.tx(hi - m, self.lm(srcx, x))) if (hi > m and srcx is not None) else (ts, ts)
            ready = max(rc_job[1], tx_job[1])
            tail = self.gpu[x].fg(ready, fe * c.rc(hi, R), (hi, R, r['i'])) if hi < R else (ready, ready)
            first = self.do_prefill(r, x, tail[1], R, R + q, fe) + (dx / 2 if dx else 0.0)
            self.stats['repair_rc_tok'] += max(0, m - ve) + max(0, R - hi)
            self.stats['repair_tx_tok'] += max(0, hi - m)
            self.stats['fg_bytes'] += max(0, hi - m) * c.kappa_tx
            if R > 0 and ve < R: self.stats['miss_local'] += 1
            elif R > 0: self.stats['hit_local'] += 1
            rtt = dx
        else:
            _, x, vx = plan
            d = self.rtt(e, x)
            fx, nd = self.gpu_factor(x, t)
            if vx < R: self.gpu[x].fg(t + d / 2, fx * c.rc(vx, R), (vx, R, r['i']))
            first = self.do_prefill(r, x, t + d / 2, R, R + q, fx) + d / 2; rtt = d
            self.stats['hold'] += 1
        _, nd = self.gpu_factor(x, first)
        itl = c.itl(nd + 1, R + q)
        done = first + r['out'] * itl + rtt / 2
        heapq.heappush(self.dec[x], done)
        if self.a.tpot: self.decinfo[r['i']] = (x, first - rtt / 2, itl, r['out'])
        if self.a.bg_dfb: self.gpu[x].add_dec(first - rtt / 2, first - rtt / 2 + r['out'] * itl)
        if self.reqlog is not None and self.a.jl_t0 - 600 <= t < self.a.jl_t1 + 600:
            self.reqlog.append(dict(rid=r['i'], t=t, e=str(e), x=str(x), R=R, q=q, out=r['out'], first=first, done=done,
                                    kind=kind, rtt=rtt, sess=sid, turn=r['turn'], itl=itl, eval=int(self.sess_start[sid] >= self.a.eval_start)))
        return x, first, done

    # ---- A3: replica-assisted parallel prefill
    NL = 28

    def zget(self, g, t):
        """Lazy drain of GPU g's stall virtual queue: Z <- max(0, Z - eps * dt)."""
        z = max(0.0, self.Z[g] - self.a.pp_eps * (t - self.Zt[g])); self.Z[g] = z; self.Zt[g] = t
        return z

    def do_prefill(self, r, x, t0, s, e_, fx):
        """Prefill tokens [s, e_) for request r at executor x (ready at t0).  Single-edge FCFS job unless
        --pp: a helper h computes the front [s, m) and streams its KV layer by layer to x, which computes the
        back [m, e_) on top of it.  h must hold a prefix >= s of the session (s > 0) or be a neighbour of x
        (s == 0, --pp_cold).  m minimises the predicted finish time.  Returns the finish time at x."""
        c = self.cost; ce = self.ce; a = self.a; meta = (s, e_, r['i'])
        single = fx * c.rc(s, e_)
        single_e = single if ce is c else fx * ce.rc(s, e_)  # A7: the controller decides on estimates
        absz = a.pp and a.pp_ctrl == 'dpp' and a.pp_zmode == 'abs'
        def run_single():
            if absz:  # A4b: Z_x tracks the ACTUAL decode stall caused by all prefill work at x
                nd = self.gpu_factor(x, t0)[1]; self.zget(x, t0); self.Z[x] += nd * single
            return self.gpu[x].fg(t0, single, meta)[1]
        if not a.pp or e_ - s < a.pp_qmin:
            return run_single()
        sid = r['sess']
        pre = {}
        if s > 0:
            cands = [h for h in self.holders_of(sid) if h != x and self.store[h][sid][0] >= s]
            if a.pp_esp:  # LoongServe-style ESP: any neighbour can help after the prefix KV [0, s) is shipped to it
                for h in ((x[0] + dq, x[1] + dr) for dq, dr in DIRS):
                    if self.in_area(h) and h not in cands: cands.append(h); pre[h] = ce.tx(s, self.lme(x, h))
        elif a.pp_cold:
            cands = [h for h in ((x[0] + dq, x[1] + dr) for dq, dr in DIRS) if self.in_area(h)]
        else:
            cands = []
        self.stats['pp_eligible'] += 1
        if not cands:
            self.stats['pp_nohelper'] += 1
            return run_single()
        NL = self.NL
        st_x = max(t0, self.gpu[x].fg_free)
        best = (st_x + single_e, None, None)
        nd_x = self.gpu_factor(x, t0)[1]
        tps = time.perf_counter() if a.time_dec else 0.0
        dpp = a.pp_ctrl == 'dpp'
        if dpp:
            # Drift-plus-penalty: minimise V * (TTFT + marginal congestion cost of the GPU time used) plus
            # sum_g Z_g * (extra decode stall at g).  Z_g is GPU g's virtual queue for the constraint
            # "time-average extra decode stall caused by parallel prefill <= eps" (decoder-seconds per second).
            def price(g):
                if not a.pp_price: return 0.0  # A4: no congestion price, TPOT virtual queues only
                u = min(0.95, self.gpu[g].util(t0)); return u / (1.0 - u) ** 2
            Zx = self.zget(x, t0); px = price(x)
            best_sc = a.pp_V * (st_x + single_e - t0) + (Zx * nd_x * single_e if absz else 0.0)
        for h in cands:
            d = self.rtt(x, h); fh, nd_h = self.gpu_factor(h, t0)
            if a.pp_dec and nd_h > nd_x: continue  # decode-aware: never move prefill onto a GPU with more decodes
            p0 = (max(t0, self.egr[x].fg_free, self.ing[h].fg_free) + pre[h] + d / 2) if h in pre else t0 + d / 2
            hs = max(p0, self.gpu[h].fg_free)
            lf = max(self.egr[h].fg_free, self.ing[x].fg_free)
            if dpp: Zh = self.zget(h, t0); ph = price(h)
            for k in range(1, a.pp_grid):
                m = s + ((e_ - s) * k) // a.pp_grid
                Cf = fh * ce.rc(s, m); X = ce.tx(m - s, self.lme(h, x)); Cb = fx * ce.rc(m, e_)
                txe = max(max(lf, hs + Cf / NL) + X, hs + Cf + X / NL) + d / 2
                f = max(st_x + Cb, txe + Cb / NL)
                if dpp:
                    occ_x = max(Cb, txe + Cb / NL - st_x) - single_e
                    sc = (a.pp_V * (f - t0 + ph * Cf + px * occ_x)
                          + Zh * nd_h * Cf + Zx * nd_x * (Cb if absz else Cb - single_e))
                    if sc < best_sc: best_sc = sc; best = (f, h, m)
                elif f < best[0]: best = (f, h, m)
        if a.time_dec: self.pp_search_wall += time.perf_counter() - tps; self.pp_evals += len(cands) * (a.pp_grid - 1)
        f, h, m = best
        if h is None or f > st_x + single_e - max(a.pp_min_gain, a.pp_min_rel * single_e):
            self.stats['pp_nogain'] += 1
            return run_single()
        if dpp:
            fh_, nd_h_ = self.gpu_factor(h, t0); Cf_ = fh_ * ce.rc(s, m); Cb_ = fx * ce.rc(m, e_)
            self.Z[h] = max(0.0, self.Z[h] + nd_h_ * Cf_)
            self.Z[x] = max(0.0, self.Z[x] + nd_x * (Cb_ if absz else Cb_ - single_e))
        if a.helper_fail > 0 and self.hf_rng.random() < a.helper_fail:
            # A9: the helper becomes unavailable after the split was planned.  The executor notices when the first
            # front layer should have arrived (planned schedule) plus a timeout, then prefills [s, e_) alone.
            self.stats['pp_helper_fail'] += 1
            d = self.rtt(x, h); fh, _ = self.gpu_factor(h, t0)
            first_layer = t0 + d / 2 + pre.get(h, 0.0) + fh * ce.rc(s, m) / NL + ce.tx(m - s, self.lme(h, x)) / NL + d / 2
            return self.gpu[x].fg(first_layer + a.helper_timeout_ms / 1e3, single, meta)[1]
        d = self.rtt(x, h); fh, _ = self.gpu_factor(h, t0)
        Cf = fh * c.rc(s, m); X = c.tx(m - s, self.lm(h, x)); Cb = fx * c.rc(m, e_)  # actual costs
        if self.lm(h, x) < 1.0: self.stats['pp_split_slowlink'] += 1
        h0 = t0 + d / 2
        if h in pre:  # ship the prefix to the helper first (x egress -> h ingress)
            ps = max(t0, self.egr[x].fg_free, self.ing[h].fg_free); pre_a = c.tx(s, self.lm(x, h))
            _, pe1 = self.egr[x].fg(ps, pre_a); _, pe2 = self.ing[h].fg(ps, pre_a); h0 = max(pe1, pe2) + d / 2
            self.stats['pp_prefix_bytes'] += s * c.kappa_tx; self.stats['pp_split_esp'] += 1
        hs, he = self.gpu[h].fg(h0, Cf, (s, m, r['i'], 'ppf', str(x), Cf, X, d))
        ts = max(self.egr[h].fg_free, self.ing[x].fg_free, hs + Cf / NL)
        _, te1 = self.egr[h].fg(ts, X); _, te2 = self.ing[x].fg(ts, X)
        txe = max(max(te1, te2), he + X / NL) + d / 2
        st = max(t0, self.gpu[x].fg_free)
        dur = max(Cb, txe + Cb / NL - st)
        g = self.gpu[x]; bl = g.blog; g.blog = None
        _, en = g.fg(t0, dur, (m, e_, r['i'], 'ppb', str(h), s, Cb, X, d))  # x is reserved for the whole interval (conservative for TTFT) ...
        g.blog = bl
        if bl is not None: bl.append((en - Cb, en))  # ... but only Cb of it is compute that stalls decode
        self.stats['pp_split'] += 1; self.stats['pp_split_cold' if s == 0 else 'pp_split_replica'] += 1
        self.stats['pp_front_tok'] += m - s; self.stats['pp_bytes'] += (m - s) * c.kappa_tx
        self.stats['pp_saved_s_est'] += st_x + single_e - f
        return en

    # ---- background replication
    def bg_replicate(self, sid, j, t, why):
        x = self.primary.get(sid)
        if x is None or x == j: return False
        V = self.version.get(sid, 0)
        if V <= 0 or sid not in self.store[x]: return False
        cur = self.store[j].get(sid, [0])[0]
        if cur >= V: return False
        for rec in self.bgjobs.get((sid, j), []):
            if rec[0] >= V: return False
        c = self.cost; ce = self.ce  # A7: the split is decided on estimates, the jobs take the actual costs
        bg_g = self.gpu[j].backlog(t); bg_t = self.egr[x].backlog(t)
        mode = self.a.bg_mode
        if mode == 'kv':
            m = cur
        elif mode == 'token':
            m = V
        elif mode == 'best_single':
            m = V if bg_g + ce.rc(cur, V) <= bg_t + ce.tx(V - cur, self.lme(x, j)) else cur
        else:
            m, _ = ce.split(cur, V, bg_g, bg_t, self.lme(x, j))
        dg = c.rc(cur, m) * self.a.bg_slow  # A9: actual background cost x bg_slow (estimates unchanged)
        gj = self.gpu[j].bg(t, dg, (cur, m, -1)); tj = self.egr[x].bg(t, c.tx(V - m, self.lm(x, j)) * self.a.bg_slow)
        self.bgjobs[(sid, j)].append([V, gj, tj, x])
        self.holder_idx[sid].add(j)
        self.stats['bg_jobs'] += 1; self.stats['bg_' + why] += 1
        self.stats['bg_rc_tok'] += m - cur; self.stats['bg_tx_tok'] += V - m
        self.stats['bg_bytes'] += (V - m) * c.kappa_tx
        return True

    def mobikv2_epoch(self, sid, t, el, why):
        """MobiKV v2 (amendment A1): keep <= d replicas per session.  Targets are scored by the expected
        backhaul RTT to the predicted next access edge plus a load term; admission is drift-plus-penalty
        against a per-edge virtual budget queue for background work (budget beta of each GPU/egress)."""
        a = self.a; x = self.primary.get(sid)
        if x is None: return
        V = self.version.get(sid, 0)
        if V <= 0 or sid not in self.store[x]: return
        k = self.turns_done.get(sid, 1); pc = self.pcont_k(k); S = self.surv(el)
        pnext = pc * S / max(1e-9, (1 - pc) + pc * S)
        if pnext < a.cont_min: return
        if a.policy == 'mobikv2_nopred':
            P = {self.edge_at(sid, t): 1.0}
        else:
            P, _ = self.predict(sid, t, el)
        for e_, p in P.items(): self.pred_cache[(sid, e_)] = p
        holders = {h for h in self.holders_of(sid) if self.store[h][sid][0] >= V}
        for h in self.holder_idx.get(sid, ()):
            if any(rec[0] >= V for rec in self.bgjobs.get((sid, h), [])): holders.add(h)
        holders.add(x)
        if len(holders) >= a.d and why != 'move': return
        def exp_rtt(j): return sum(p * self.rtt(j, e_) for e_, p in P.items())
        if why == 'move' and min(exp_rtt(h) for h in holders) * 1e3 <= a.move_rtt_ms: return
        cand = set(P) | {(u[0] + dq, u[1] + dr) for u in list(P) + [x] for dq, dr in DIRS}
        cand = [j for j in cand if j not in holders and self.in_area(j)]
        if not cand: return
        score = lambda j: exp_rtt(j) + a.w_load * self.gpu[j].util(t)
        need = max(1, a.d - len(holders)) if why != 'move' else 1
        c = self.cost
        for j in sorted(cand, key=score)[:need]:
            cur = self.store[j].get(sid, [0])[0]
            m, _ = c.split(cur, V, self.gpu[j].backlog(t), self.egr[x].backlog(t), self.lme(x, j))
            wg, wt = c.rc(cur, m), c.tx(V - m, self.lme(x, j))
            qg = self.gpu[j].vq_update(t, a.beta); qt = self.egr[x].vq_update(t, a.beta)
            benefit = pnext * c.rc(0, V)
            if a.V * benefit >= qg * wg + qt * wt:
                if self.bg_replicate(sid, j, t, why):
                    self.gpu[j].vq += wg; self.egr[x].vq += wt
            else:
                self.stats['bg_rejected_budget'] += 1

    def mobikv_final(self, sid, t):
        """MobiKV (A1): keep one secondary replica near the user.  Refresh the existing secondary with the
        delta (sticky) unless it is far (> 1 hop from the user) or clearly busier than the best candidate;
        otherwise place it at the least-loaded edge among the user's edge and its neighbours.  Background
        work is split between token replay on the target GPU and KV transfer by the rho rule (Cost.split).
        Optional per-edge virtual budget queue (drift-plus-penalty) when --beta < 1."""
        a = self.a; x = self.primary.get(sid)
        if x is None: return
        V = self.version.get(sid, 0)
        if V <= 0 or sid not in self.store[x]: return
        k = self.turns_done.get(sid, 1); pc = self.pcont_k(k)
        if pc < a.cont_min: return
        u = self.edge_at(sid, t)
        if a.policy == 'mobikv_pred':
            P, _ = self.predict(sid, t, 0.0)
            near = {e_ for e_, p in P.items() if p >= 0.1} | {u}
            near = near | {(e_[0] + dq, e_[1] + dr) for e_ in list(near) for dq, dr in DIRS}
        else:
            near = {u} | {(u[0] + dq, u[1] + dr) for dq, dr in DIRS}
        near = [j for j in near if j != x and self.in_area(j)]
        if not near: return
        sec = [h for h in self.holders_of(sid) if h != x]
        best = min(near, key=lambda j: self.gpu[j].util(t))
        tgt = best
        if a.policy != 'mobikv_nosticky' and sec:
            j0 = max(sec, key=lambda h: self.store[h][sid][0])
            if hexdist(j0, u) <= 1 and self.gpu[j0].util(t) <= self.gpu[best].util(t) + a.sticky_slack:
                tgt = j0
        if a.beta < 1.0:
            c = self.ce; cur = self.store[tgt].get(sid, [0])[0]
            m, _ = c.split(cur, V, self.gpu[tgt].backlog(t), self.egr[x].backlog(t), self.lme(x, tgt))
            wg, wt = c.rc(cur, m), c.tx(V - m, self.lme(x, tgt))
            qg = self.gpu[tgt].vq_update(t, a.beta); qt = self.egr[x].vq_update(t, a.beta)
            if a.V * pc * c.rc(0, V) < qg * wg + qt * wt:
                self.stats['bg_rejected_budget'] += 1; return
            self.gpu[tgt].vq += wg; self.egr[x].vq += wt
        self.bg_replicate(sid, tgt, t, 'mobikv')

    def in_area(self, j):
        # centre of hex j inside the D x D square (edges exist only where users can be)
        q, r = j; R = self.a.r
        x = R * SQ3 * (q + r / 2.0); y = R * 1.5 * r
        return -R <= x <= self.D + R and -R <= y <= self.D + R

    def mobikv_epoch(self, sid, t, el):
        """Predictive, budget-aware background replication (drift-plus-penalty admission)."""
        x = self.primary.get(sid)
        if x is None: return
        V = self.version.get(sid, 0)
        if V <= 0: return
        c = self.ce; k = self.turns_done.get(sid, 1)  # A7: estimates
        pc = self.pcont_k(k); S = self.surv(el); pnext = pc * S / max(1e-9, (1 - pc) + pc * S)
        if self.a.policy == 'mobikv_nopred':
            u = self.edge_at(sid, t); P = {u: 1.0}; rem_med = 30.0
        else:
            P, rem_med = self.predict(sid, t, el)
        for e_, p in P.items(): self.pred_cache[(sid, e_)] = p
        qbar = 32
        fx, _ = self.gpu_factor(x, t)
        for j, p in sorted(P.items(), key=lambda kv: -kv[1])[:self.a.topk]:
            if j == x or p < self.a.pmin: continue
            cur = self.store[j].get(sid, [0])[0]
            if cur >= V or any(rec[0] >= V for rec in self.bgjobs.get((sid, j), [])): continue
            fj, _ = self.gpu_factor(j, t)
            local = self.gpu[j].wait(t) + fj * c.rc(V, V + qbar)
            d = self.rtt(j, x)
            hold = d + self.gpu[x].wait(t) + fx * c.rc(V, V + qbar)
            wt = self.egr[x].wait(t)
            m, rdy = c.split(cur, V, self.gpu[j].wait(t), wt, self.lme(x, j))
            loc_miss = rdy + fj * c.rc(V, V + qbar)
            react = min(hold, loc_miss)
            gain = max(0.0, react - local)
            bg_g = self.gpu[j].backlog(t); bg_t = self.egr[x].backlog(t)
            m2, rdy2 = c.split(cur, V, bg_g * 1.5, bg_t * 1.5, self.lme(x, j))
            rem = self.remaining_think_samples(el, 64)
            pready = float((rem > rdy2).mean())
            benefit = pnext * p * gain * pready
            use_g = c.rc(cur, m2); use_t = c.tx(V - m2, self.lme(x, j))
            price = (bg_g * use_g + bg_t * use_t) / self.a.V
            if benefit > price + self.a.min_gain:
                self.bg_replicate(sid, j, t, 'pred')

    # ---- main loop
    def run(self):
        a = self.a; ev = []; seq = 0
        self.anchor = {}; self.holder_idx = collections.defaultdict(set)
        for r in self.reqs:
            if r['parent'] < 0:
                heapq.heappush(ev, (r['t'], seq, 'arr', r['i'])); seq += 1
        # think times of the tuning half (trace gaps minus our idle service estimate)
        for r in self.reqs:
            p = r['parent']
            if p >= 0 and self.sess_start[r['sess']] < a.eval_start:
                pr = self.reqs[p]
                svc = self.cost.rc(pr['R'], pr['R'] + pr['q']) + pr['out'] * self.cost.itl(1, pr['inp'])
                self.think_tune.append(max(2.0, r['t'] - pr['t'] - svc))
        self.fit_think()
        if a.policy.startswith('mobikv') and not a.policy.endswith('nopred'):
            self.fit_displacement()
        t_wall = time.time()
        while ev:
            t, _, kind, payload = heapq.heappop(ev)
            self.set_env(t)  # A7: link-rate regime and estimate noise for this event (no-op by default)
            if kind == 'arr':
                r = self.reqs[payload]; sid = r['sess']
                e = self.edge_at(sid, t)
                for x in list(self.holder_idx.get(sid, ())):
                    self.harvest(sid, x, t)
                td0 = time.perf_counter() if a.time_dec else 0.0
                if a.policy in self.BASELINES:
                    est, plan = self.choose_baseline(r, e, t)
                else:
                    opts = self.options(r, e, t)
                    est, plan = min(opts, key=lambda o: o[0])
                if a.time_dec: td1 = time.perf_counter() - td0; ps0 = self.pp_search_wall
                # cancel useless in-flight background jobs for this session (state is about to change)
                for x in list(self.holder_idx.get(sid, ())):
                    if self.bgjobs.get((sid, x)): self.cancel_bg(sid, x)
                x, first, done = self.execute(r, e, t, plan)
                if a.time_dec: self.dec_wall[payload] = (td1, self.pp_search_wall - ps0)
                self.pinned[x][sid] += 1
                if sid not in self.anchor: self.anchor[sid] = x
                newv = r['R'] + r['q'] + r['out']
                old = self.primary.get(sid)
                self.version[sid] = newv; self.primary[sid] = x; self.holder_idx[sid].add(x)
                self.put(x, sid, newv, t)
                handoff = (old is not None and old != e)
                self.out.append((r['i'], sid, r['turn'], self.sess_start[sid] >= a.eval_start, first - t, done - t,
                                 plan[0], handoff, r['R'], e != x))
                heapq.heappush(ev, (done, seq, 'done', (payload, x))); seq += 1
            elif kind == 'done':
                i, x = payload; r = self.reqs[i]; sid = r['sess']
                self.pinned[x][sid] -= 1; self.touch(x, sid, t)
                self.last_complete[sid] = t; self.turns_done[sid] = r['turn']
                kids = self.child.get(i, [])
                for kidx in kids:
                    kr = self.reqs[kidx]
                    svc = self.cost.rc(r['R'], r['R'] + r['q']) + r['out'] * self.cost.itl(1, r['inp'])
                    think = max(2.0, kr['t'] - r['t'] - svc)
                    heapq.heappush(ev, (t + think, seq, 'arr', kidx)); seq += 1
                pol = a.policy
                svc0 = self.cost.rc(r['R'], r['R'] + r['q']) + r['out'] * self.cost.itl(1, r['inp'])
                t_next = (t + max(2.0, min(self.reqs[k]['t'] for k in kids) - r['t'] - svc0)) if kids else None
                if pol == 'oracle' and kids:
                    j = self.edge_at(sid, t_next)
                    self.bg_replicate(sid, j, t, 'oracle')
                elif pol == 'neighbors':
                    u = self.edge_at(sid, t)
                    for j in [u] + [(u[0] + dq, u[1] + dr) for dq, dr in DIRS]:
                        if self.in_area(j): self.bg_replicate(sid, j, t, 'nbr')
                elif pol in ('mobikv', 'mobikv_kvonly', 'mobikv_tokenonly', 'mobikv_nohybrid', 'mobikv_nosticky', 'mobikv_budget', 'mobikv_pred'):
                    self.mobikv_final(sid, t)
                elif pol in ('randany2', 'leastany2'):
                    x0 = self.primary.get(sid)
                    nb = [j for j in self.area_edges if j != x0]
                    j = self.rng.choice(nb) if pol == 'randany2' else min(nb, key=lambda j: self.gpu[j].util(t))
                    self.bg_replicate(sid, j, t, pol)
                elif pol in ('rand2', 'least2'):
                    u = self.edge_at(sid, t); x0 = self.primary.get(sid)
                    nb = [j for j in [u] + [(u[0] + dq, u[1] + dr) for dq, dr in DIRS] if j != x0 and self.in_area(j)]
                    if nb:
                        j = self.rng.choice(nb) if pol == 'rand2' else min(nb, key=lambda j: self.gpu[j].util(t))
                        self.bg_replicate(sid, j, t, pol)
                elif pol in ('mobikv2', 'mobikv2_nopred'):
                    self.mobikv2_epoch(sid, t, 0.0, 'done')
                    horizon = t_next if t_next is not None else t + a.max_watch
                    for (tm, e2) in self.moves(sid, t, min(horizon, t + a.max_watch)):
                        heapq.heappush(ev, (tm, seq, 'move', (sid, e2, t))); seq += 1
                elif pol in ('follow_bg',) or pol.startswith('mobikv'):
                    horizon = t_next if t_next is not None else t + a.max_watch
                    if pol.startswith('mobikv'):
                        heapq.heappush(ev, (t, seq, 'epoch', (sid, t))); seq += 1
                    for (tm, e2) in self.moves(sid, t, min(horizon, t + a.max_watch)):
                        heapq.heappush(ev, (tm, seq, 'move', (sid, e2, t))); seq += 1
            elif kind == 'move':
                sid, e2, tc = payload
                if self.last_complete.get(sid) != tc: continue  # a newer turn started
                if a.policy == 'follow_bg':
                    self.bg_replicate(sid, e2, t, 'move')
                elif a.policy.startswith('mobikv2'):
                    self.mobikv2_epoch(sid, t, t - tc, 'move')
                else:
                    self.mobikv_epoch(sid, t, t - tc)
            elif kind == 'epoch':
                sid, tc = payload
                if self.last_complete.get(sid) != tc: continue
                self.mobikv_epoch(sid, t, t - tc)
        self.wall = time.time() - t_wall

    def summary(self):
        a = self.a
        rows = [o for o in self.out if o[3]]
        tune = [o for o in self.out if not o[3]]
        def q(v, p): return float(np.quantile(v, p)) if len(v) else None
        res = {}
        for name, sel in [('all', rows), ('cont', [o for o in rows if o[2] >= 2]),
                          ('cont_handoff', [o for o in rows if o[2] >= 2 and o[7]]),
                          ('first', [o for o in rows if o[2] == 1]),
                          ('tune_cont', [o for o in tune if o[2] >= 2]),
                          ('tune_cont_handoff', [o for o in tune if o[2] >= 2 and o[7]])]:
            tt = np.array([o[4] for o in sel]); cp = np.array([o[5] for o in sel])
            res[name] = dict(n=len(sel), ttft_mean=float(tt.mean()) if len(tt) else None, ttft_p50=q(tt, .5), ttft_p95=q(tt, .95),
                             ttft_p99=q(tt, .99), slo2=float((tt <= 2.0).mean()) if len(tt) else None,
                             slo5=float((tt <= 5.0).mean()) if len(tt) else None, compl_mean=float(cp.mean()) if len(cp) else None)
        edges = set(self.gpu) | set(self.egr)
        T = max(o[5] + self.reqs[o[0]]['t'] for o in self.out) if self.out else 1
        util = [self.gpu[e].busy_fg / 3600.0 for e in self.gpu]
        res['stats'] = dict(self.stats); res['edges_used'] = len(edges)
        res['gpu_util_fg'] = dict(mean=float(np.mean(util)) if util else 0, p90=q(util, .9), max=float(max(util)) if util else 0)
        res['gpu_bg_s'] = float(sum(self.gpu[e].busy_bg for e in self.gpu)); res['egr_bg_s'] = float(sum(self.egr[e].busy_bg for e in self.egr))
        res['egr_fg_s'] = float(sum(self.egr[e].busy_fg for e in self.egr))
        res['fg_delay_by_bg_s'] = float(sum(r.interf for r in list(self.gpu.values()) + list(self.egr.values())))
        res['wall_s'] = self.wall; res['branching_parents'] = self.branching; res['sessions'] = self.nsess
        res['D_m'] = self.D
        res['args'] = vars(a)
        if a.joblog:
            busy = {str(e): sum(j[2] for j in self.gpu[e].log if j[0] == 'fg' and a.jl_t0 <= j[1] < a.jl_t1) for e in self.gpu}
            json.dump({'version': 2, 'busy_fg_window': busy, 'window': [a.jl_t0, a.jl_t1], 'NL': self.NL,
                       'jobs': {str(e): [list(j) for j in self.gpu[e].log if a.jl_t0 <= j[1] < a.jl_t1] for e in self.gpu},
                       'reqs': self.reqlog, 'args': vars(a)},
                      open(a.joblog, 'w'))
        if a.time_dec:
            d = np.array([v[0] + v[1] for v in self.dec_wall.values()]); ps = np.array([v[1] for v in self.dec_wall.values()])
            res['dec_time'] = dict(n=len(d), mean_ms=float(d.mean() * 1e3), p50_ms=float(np.quantile(d, .5) * 1e3),
                                   p99_ms=float(np.quantile(d, .99) * 1e3), max_ms=float(d.max() * 1e3),
                                   pp_search_mean_ms=float(ps.mean() * 1e3), pp_evals=self.pp_evals, edges=len(self.area_edges),
                                   requests=len(self.reqs))
        res['per_request'] = [[o[0], o[1], o[2], int(o[3]), round(o[4], 4), round(o[5], 3), o[6], int(o[7]), o[8], int(o[9])] for o in self.out]
        if a.tpot:
            self.add_tpot(res)
        return res

    def add_tpot(self, res):
        """A3: TPOT under two decode models.  tpot_b = measured batched ITL (A1 model).  tpot_s = ITL plus a
        full stall of the decode for every second the same GPU spends on prefill-class work (foreground,
        background, parallel-prefill helper work and waits) inside the decode window (prefill-prioritising
        iteration scheduler without chunked prefill).  Post-hoc, fixed point over the window, 3 iterations."""
        for e in self.gpu: self.gpu[e].advance(1e12)
        def merged(lst):
            mg = []
            for s_, e2 in sorted(lst):
                if e2 <= s_: continue
                if mg and s_ <= mg[-1][1]:
                    if e2 > mg[-1][1]: mg[-1][1] = e2
                else: mg.append([s_, e2])
            st = [v[0] for v in mg]; en = [v[1] for v in mg]
            cum = [0.0]
            for v in mg: cum.append(cum[-1] + v[1] - v[0])
            return (st, en, cum)
        ivs = {e: merged(g.blog) for e, g in self.gpu.items()}
        # A8 (--bg_dfb): background replay runs only outside decode windows; any residual overlap (a chunk in service when
        # a stream starts) is counted with weight 0.5
        bivs = {e: merged(g.bblog) for e, g in self.gpu.items() if g.bblog is not None}
        def busy(e, lo, hi):
            v = busy1(ivs[e], lo, hi)
            if e in bivs: v += 0.5 * busy1(bivs[e], lo, hi)
            return v
        def busy1(iv, lo, hi):
            st, en, cum = iv
            if not st or hi <= lo: return 0.0
            i = bisect.bisect_right(en, lo)  # first interval ending after lo
            j = bisect.bisect_left(st, hi)   # intervals starting before hi: [i, j)
            if j <= i: return 0.0
            tot = cum[j] - cum[i]
            tot -= max(0.0, lo - st[i])
            tot -= max(0.0, en[j - 1] - hi)
            return max(0.0, tot)
        tp = {}
        for i, (x, first, itl, out) in self.decinfo.items():
            w = out * itl; stall = 0.0
            for _ in range(3): stall = busy(x, first, first + w + stall)
            tp[i] = (itl, (w + stall) / out, out)
        for row in res['per_request']:
            v = tp.get(row[0])
            row += [round(v[0], 5), round(v[1], 5), v[2]] if v else [None, None, 0]
        def q(v, p): return float(np.quantile(v, p)) if len(v) else None
        rows = [r_ for r_ in res['per_request'] if r_[3] and r_[12] >= 2]
        tune = [r_ for r_ in res['per_request'] if not r_[3] and r_[12] >= 2]
        tt1 = [r_[4] for r_ in res['per_request'] if not r_[3] and r_[2] == 1]
        res['tune_first'] = dict(n=len(tt1), ttft_p95=q(np.array(tt1), .95), ttft_mean=float(np.mean(tt1)) if tt1 else None)
        for name, sel in (('all', rows), ('cont', [r_ for r_ in rows if r_[2] >= 2]), ('first', [r_ for r_ in rows if r_[2] == 1]),
                          ('tune_cont', [r_ for r_ in tune if r_[2] >= 2]), ('tune_first', [r_ for r_ in tune if r_[2] == 1])):
            b = np.array([r_[10] for r_ in sel]); s_ = np.array([r_[11] for r_ in sel])
            res[name].update(tpot_n=len(sel), tpot_b_p50=q(b, .5), tpot_b_p95=q(b, .95), tpot_b_mean=float(b.mean()) if len(b) else None,
                             tpot_s_p50=q(s_, .5), tpot_s_p95=q(s_, .95), tpot_s_mean=float(s_.mean()) if len(s_) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', required=True); ap.add_argument('--trips', required=True); ap.add_argument('--costs', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--policy', default='reactive')
    ap.add_argument('--model', default='7B'); ap.add_argument('--share', type=int, default=25); ap.add_argument('--cap', type=float, default=1000)
    ap.add_argument('--r', type=float, default=500.0); ap.add_argument('--side', type=float, default=3500.0); ap.add_argument('--side_cells', type=float, default=7.0)
    ap.add_argument('--mem_gb', type=float, default=32.0); ap.add_argument('--load', type=float, default=1.0)
    ap.add_argument('--rtt_base_ms', type=float, default=2.0); ap.add_argument('--rtt_hop_ms', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--eval_start', type=float, default=1800.0)
    ap.add_argument('--evict', default='lru'); ap.add_argument('--bg_mode', default='hybrid'); ap.add_argument('--hybrid_ok', type=int, default=1)
    ap.add_argument('--V', type=float, default=4.0); ap.add_argument('--pmin', type=float, default=0.1); ap.add_argument('--topk', type=int, default=3)
    ap.add_argument('--min_gain', type=float, default=0.0); ap.add_argument('--pred_samples', type=int, default=128)
    ap.add_argument('--max_watch', type=float, default=900.0); ap.add_argument('--mode_filter', default='')
    ap.add_argument('--gpu_chunk', type=float, default=0.1); ap.add_argument('--link_chunk', type=float, default=0.05)
    ap.add_argument('--d', type=int, default=2); ap.add_argument('--cont_min', type=float, default=0.2)
    ap.add_argument('--w_load', type=float, default=1.0); ap.add_argument('--beta', type=float, default=0.3)
    ap.add_argument('--move_rtt_ms', type=float, default=4.0)
    ap.add_argument('--route_first', default='local'); ap.add_argument('--sticky_slack', type=float, default=0.2)
    ap.add_argument('--speedup', type=float, default=1.0)
    ap.add_argument('--joblog', default=''); ap.add_argument('--jl_t0', type=float, default=2400.0); ap.add_argument('--jl_t1', type=float, default=2700.0)
    # A3: replica-assisted parallel prefill and TPOT accounting (all off by default = A1 behaviour)
    ap.add_argument('--pp', type=int, default=0); ap.add_argument('--pp_cold', type=int, default=1)
    ap.add_argument('--pp_qmin', type=int, default=1024); ap.add_argument('--pp_min_gain', type=float, default=0.02)
    ap.add_argument('--pp_min_rel', type=float, default=0.0); ap.add_argument('--pp_dec', type=int, default=0)
    ap.add_argument('--pp_grid', type=int, default=32)  # A6: split points searched = pp_grid - 1
    ap.add_argument('--pp_ctrl', default='greedy'); ap.add_argument('--pp_V', type=float, default=10.0)
    ap.add_argument('--pp_eps', type=float, default=0.02); ap.add_argument('--pp_price', type=int, default=1)
    ap.add_argument('--pp_esp', type=int, default=0)
    ap.add_argument('--oy_V', type=float, default=1.0); ap.add_argument('--oy_B', type=float, default=0.5); ap.add_argument('--oy_hops', type=int, default=2)
    ap.add_argument('--mdp_k', type=int, default=2)
    ap.add_argument('--pp_zmode', default='extra')  # extra (A3): constrain extra stall caused by RPP; abs (A4b): actual stall
    ap.add_argument('--tpot', type=int, default=0); ap.add_argument('--delta_ideal', type=int, default=0)
    # A7 (all off by default = A6 behaviour)
    ap.add_argument('--pred_err', type=float, default=0.0)   # sigma of log-normal noise on every controller's cost estimates
    ap.add_argument('--bw_var', type=int, default=0)         # two-state Markov link-rate regime (shared by all links)
    ap.add_argument('--bw_slow', type=float, default=0.4); ap.add_argument('--bw_dwell_n', type=float, default=1200.0)
    ap.add_argument('--bw_dwell_s', type=float, default=600.0)
    ap.add_argument('--bw_lag', type=float, default=30.0)    # controllers see the rate of t - lag; < 0: nominal rate only
    ap.add_argument('--kv_comp', type=float, default=1.0)    # KV compression ratio on the wire (all transfers)
    ap.add_argument('--time_dec', type=int, default=0)       # wall time of every request's decision
    ap.add_argument('--bg_dfb', type=int, default=0)         # A8: strict decode priority for background replay
    # A9 (all off by default = A8 behaviour)
    ap.add_argument('--link_het', type=float, default=0.0)       # probability that an edge's network is slow
    ap.add_argument('--link_het_mult', type=float, default=0.35)  # rate multiplier of a slow edge's links
    ap.add_argument('--link_het_known', type=int, default=1)      # controllers know the per-link rates (1) or assume nominal (0)
    ap.add_argument('--helper_fail', type=float, default=0.0)     # probability that a planned RPP helper becomes unavailable
    ap.add_argument('--helper_timeout_ms', type=float, default=100.0)
    ap.add_argument('--bg_slow', type=float, default=1.0)         # actual background replication cost multiplier
    a = ap.parse_args()
    # policy -> mechanism switches (ablations)
    if a.policy == 'mobikv_kvonly': a.bg_mode = 'kv'
    if a.policy == 'mobikv_tokenonly': a.bg_mode = 'token'
    if a.policy in ('reactive_nohybrid', 'mobikv_nohybrid'): a.hybrid_ok = 0
    if a.policy == 'mobikv_nohybrid': a.bg_mode = 'best_single'
    cfg = json.load(open(a.costs))
    trips = dict(np.load(a.trips, allow_pickle=False))
    s = Sim(a, cfg, trips)
    s.run()
    res = s.summary()
    json.dump(res, open(a.out, 'w'), indent=1)
    c = res['cont']; h = res['cont_handoff']
    print(f"{a.policy} cont n={c['n']} p95={c['ttft_p95']:.3f} mean={c['ttft_mean']:.3f} | handoff n={h['n']} p95={h['ttft_p95'] if h['n'] else float('nan'):.3f} | util={res['gpu_util_fg']['mean']:.2f} wall={res['wall_s']:.0f}s")


if __name__ == '__main__':
    main()
