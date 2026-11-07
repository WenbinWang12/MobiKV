#!/usr/bin/env python3
"""A7 / E6: WildChat-1M (shard 0) -> Mooncake-format request trace for mobikv_sim.

Each conversation is a session.  Token counts come from the Qwen2.5 tokenizer (content only, plus 5 template tokens
per message and a 20-token system prompt).  Turn k's input = all earlier messages + user message k; output = assistant
message k.  Request times keep WildChat's real gaps between consecutive assistant timestamps (the only timestamps in the
dataset); a session ends at its first gap > 30 min.  Session start times are re-drawn uniformly over [0, 3600) s and
sessions are added in a seeded random order until the trace's total prefill volume (new tokens) inside the hour matches
the Mooncake conversation trace (the GPU load is what matters for TTFT).  hash_ids: 16-token blocks, unique per session, so
that the simulator's prefix-chain session reconstruction recovers every session exactly.
Only long sessions are kept (final context >= MIN_CTX tokens, default 4096): the problem concerns long multi-turn
sessions, and Mooncake (Kimi) is a long-context trace; short chats are reported in the stats only.
Usage: wildchat_prep.py <parquet[,parquet...]> <mooncake.jsonl> <tokenizer dir> <out.jsonl> [MIN_CTX]
"""
import json, random, sys, statistics
import pyarrow.parquet as pq
from transformers import AutoTokenizer

PQ, MOON, TOK, OUT = sys.argv[1:5]; MIN_CTX = int(sys.argv[5]) if len(sys.argv) > 5 else 4096
BLK = 16; SYS = 20; TPL = 5; GAP = 1800.0; H = 3600.0
tok = AutoTokenizer.from_pretrained(TOK)
moon = [json.loads(l) for l in open(MOON) if l.strip()]
# Mooncake prefill volume: new tokens per request = input - (parent input + output), reconstructed as in the simulator
key = {}; vol_m = 0
for r in moon:
    h = r['hash_ids']; p = None
    for ln in range(len(h) - 1, 1, -1):
        if tuple(h[:ln]) in key: p = key[tuple(h[:ln])]; break
    key[tuple(h)] = r; key[tuple(h[:-1])] = r
    R = 0 if p is None else max(0, min(p['input_length'] + p['output_length'], r['input_length'] - 16))
    vol_m += r['input_length'] - R
convs = []; n_all = 0; n_short = 0
rows = []
for p_ in PQ.split(','):
    rows += pq.read_table(p_, columns=['conversation_hash', 'conversation']).to_pylist()
for row in rows:
    msgs = row['conversation']; turns = []; ctx = SYS; last_ts = None
    i = 0
    while i + 1 < len(msgs):
        u, a_ = msgs[i], msgs[i + 1]; i += 2
        if u['role'] != 'user' or a_['role'] != 'assistant' or a_['timestamp'] is None: break
        ts = a_['timestamp'].timestamp()
        if last_ts is not None and (ts - last_ts > GAP or ts < last_ts): break
        nu = len(tok(u['content'] or '').input_ids) + TPL; na = len(tok(a_['content'] or '').input_ids) + TPL
        turns.append(dict(gap=0.0 if last_ts is None else ts - last_ts, inp=ctx + nu, out=max(1, na), new=nu if turns else ctx + nu))
        ctx += nu + na; last_ts = ts
    if not turns: continue
    n_all += 1
    if ctx < MIN_CTX: n_short += 1; continue
    convs.append(turns)
rng = random.Random(20261103); rng.shuffle(convs)
reqs = []; vol = 0; nsess = 0; bid = 1
for turns in convs:
    if vol >= vol_m: break
    t0 = rng.uniform(0, H); tt = t0; hashes = []
    for k, tr in enumerate(turns):
        tt += tr['gap']
        if tt >= H: break
        need = (tr['inp'] + BLK - 1) // BLK
        while len(hashes) < need: hashes.append(bid); bid += 1
        reqs.append(dict(timestamp=int(tt * 1e3), input_length=tr['inp'], output_length=tr['out'], hash_ids=hashes[:need]))
        vol += tr['new']
        # the next turn's context includes this turn's output: extend the block chain past input + output
        need2 = (tr['inp'] + tr['out'] + BLK - 1) // BLK
        while len(hashes) < need2: hashes.append(bid); bid += 1
    nsess += 1
reqs.sort(key=lambda r: r['timestamp'])
with open(OUT, 'w') as f:
    for r in reqs: f.write(json.dumps(r) + '\n')
inp = [r['input_length'] for r in reqs]
st = dict(min_ctx=MIN_CTX, conversations_parsed=n_all, conversations_short=n_short, conversations_available=len(convs), volume_matched=vol >= vol_m, sessions=nsess, requests=len(reqs), prefill_tokens=vol, mooncake_prefill_tokens=vol_m,
          mooncake_requests=len(moon), input_p50=statistics.median(inp), input_mean=statistics.mean(inp),
          input_p95=sorted(inp)[int(0.95 * len(inp))], multi_turn_sessions=sum(1 for c in convs[:nsess] if len(c) > 1),
          turns_per_session_mean=len(reqs) / max(1, nsess))
json.dump(st, open(OUT + '.stats.json', 'w'), indent=1); print(json.dumps(st, indent=1))
