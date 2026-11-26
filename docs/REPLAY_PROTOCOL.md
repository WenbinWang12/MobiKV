# Three-server real-GPU replay, protocol (written 2026-11-23 16:40Z, before any run of this campaign)

Smoke tests of 60-120 s with the MobiKV plan were run first (replay_smoke*, replay_smoke_slow); no policy comparison
was made from them.  They showed:
- 10 Gb/s point: fronts between two replayed servers arrive at about 5 Gb/s (TCP loopback and Unix sockets alike, send
  chunk 64 KiB to 4 MiB), so backs with a helper inside the set take about twice their planned time.  The harness cannot
  emulate this link rate on one machine; this point is not replayed.
- 25% share / 0.93 Gb/s point: plain prefill measured/planned duration median 0.92 (17 jobs), median TTFT error 8.8%.

Campaign (code/replay_u.sh slow):
- model Qwen3-8B; plans from simulator MobiKV_v5 with costs_a800_v2.json and the argument strings of the cross-model
  table at the 25% share / 0.93 Gb/s point; MobiKV = final design plus neighbour helpers (--pp_esp 1);
- GeoLife mobility, seed 41 (first evaluation seed), window [2400, 3300) s, servers (1,2), (2,2), (1,3) for every policy;
- one worker per server on GPUs 0-2, attached to the MPS daemon with a 25% share, egress paced at 930 Mb/s,
  decode batches of up to 8 streams, TCP loopback between workers;
- accepted when all three workers finish every job with no layer-wait timeout; CPU-throttling counters are recorded;
- reported: pooled request-level follow-up TTFT and TPOT percentiles of evaluation-half requests (as in the earlier
  replay tables), and the simulator-vs-measured error per policy.  All seven policies are reported whatever the outcome.

## Addendum (2026-11-23 20:08Z, written before any replay result of this campaign was read)

- The runs of the first two policies (mobikv, loongserve_sosp24) overlapped with CPU simulations on the same node
  (24 simulator processes; 0 throttled CPU periods).  The other five ran on a quiet node.
- A second replicate (replay_slow_rep2, identical plans and settings, quiet node) is run for all seven policies, in
  the order mobikv, loongserve_sosp24, then the rest.
- Reported run per policy: the quiet-node run, i.e. replicate 2 for mobikv and loongserve_sosp24 and replicate 1 for
  the other five.  The other replicate of every policy is reported next to it as a repeatability check.
- MPS attachment is verified from the daemon log (new clients during the run) and from the ratio of measured to
  planned duration of plain prefill jobs (about 0.9 at the 25% share in the smoke test; about 0.3 would mean full speed).

## Extension (2026-11-23 20:25Z, written after replicate 1 of seed 41 had been aggregated and read)

- Seed 41 alone gives 64-103 follow-up requests per policy; the request-bootstrap intervals of P90 overlap between
  MobiKV and the best baseline.  To add requests, the same protocol is repeated with seeds 42 and 43 (same window,
  same three servers, all seven policies, quiet node), directories replay_slow_seed42 and replay_slow_seed43.
- Order of GPU work: replicate 2 of mobikv and loongserve_sosp24 (seed 41, quiet node) finishes first; then seeds 42
  and 43; then the remaining five policies of replicate 2.
- Analysis fixed now: the table pools the request-level samples of seeds 41, 42 and 43 per policy (for seed 41 the
  quiet-node run of each policy, as in the addendum) and reports the same percentiles; per-seed rows and replicate 2
  go to the supplementary material.  If seeds 42-43 are not complete when the paper is built, the table reports seed 41
  only and says so.
