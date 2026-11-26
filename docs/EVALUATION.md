# Evaluation protocol

This document describes the final evaluation campaign of 2026-11-24. [Experiment commands](EXPERIMENTS.md) provides the launchers; [GPU reproduction](GPU_REPRODUCTION.md) covers measurements and replay.

## Workload and operating points

We combine GeoLife user mobility with the Mooncake conversation trace. Each session follows a trip through a 3.5 km by 3.5 km area with 33 servers. We reconstruct conversation sessions from shared block-hash prefixes. Sessions starting in the first 30 minutes form the tuning half; later sessions form the evaluation half. Reported latency statistics use follow-up turns of evaluation-half sessions.

The cost profiles come from BF16 measurements on NVIDIA A800 80GB GPUs. The comparison covers Qwen2.5-7B, Qwen2.5-14B, Qwen3-8B, Llama-3.1-8B, Mistral-7B-v0.3, Gemma-2-9B-it, Qwen3.5-9B, and Qwen3.8-27B. The last two use hybrid attention. Gemma-2 uses an 8k-token context window; the other models use 32k. Sessions that exceed a model's window are scaled to fit.

| Launcher option | GPU share per server | Link rate |
| --- | --- | --- |
| `--op full` | One A800 | 10 Gb/s |
| `--op slow` | One quarter of an A800 | 0.93 Gb/s |
| `--op full1g` | One A800 | 0.93 Gb/s |

Each server has a 64 GB KV store. RTT is 2 ms plus 1 ms per hop. When a model's 8k-token prefill exceeds the 2.07 s reference time, the session arrival rate is reduced by the ratio of 2.07 s to that prefill time. The launchers apply this rule from the cost profile.

## Policies and comparisons

The six baselines adapt Follow-Me, MDP migration, Lyapunov placement, Preble, Mooncake, and ESP to the same workload and resources. Their source papers are listed in [Data and citations](DATA_RELEASE.md#baselines). MobiKV considers all servers, keeps one replica, refreshes it through idle links without GPU work, and evaluates at most one prefill helper. It permits cold helpers after charging their prefix transfer. Its settings are `q_min=1024`, 32 split points, `theta=0.25`, `V=30`, `epsilon=0.05`, and `omega=0.2`.

Parameters were selected on the tuning half with seeds 21–25. The main model comparison, link-rate sweep, and ablations use evaluation seeds 41–60. The stress campaign uses ten evaluation seeds. These seeds vary mobility; they do not supply independent conversation traces.

For each seed, we compute the P50, P90, and P99 over follow-up requests. Tables report their means and sample standard deviations over seeds. Policy comparisons use the mean of paired per-seed ratios and a 95% t interval. The best baseline of a cell is selected by its mean latency. A mean paired ratio need not equal the ratio of aggregate means.

Simulation TPOT is stall-aware: it combines batched inter-token latency with the prefill-class work that interrupts decoding, including helper work. This models prefill without chunking and can overestimate stalls in engines that interleave smaller prefill chunks.

The sweep varies Qwen3-8B's link rate through 18 settings. Its horizontal coordinate, `gamma_8k`, compares transfer and recomputation of an 8k-token reference state; the request trace still contains different context lengths. This ratio differs from RPP's `rho`, which compares transfer and computation of the new tokens.

## Real-GPU measurements

The two-GPU RPP study covers 78 configurations across five models and 510 split configurations, with three repetitions per split. Each speedup uses the best tested split and the median of three runs. In the 54 configurations whose delivered link rate was measured, speedup reaches 82–94% of the one-helper bound `min(2, 1 + 1/rho)`. Exactness is checked against chunked prefill at the same split boundary. It is not a claim of bitwise equality with monolithic prefill.

The replay runs Qwen3-8B on three A800 GPUs in one node, one worker per GPU, with a 25% MPS share and application-level pacing at 0.93 Gb/s. All seven policies use the same three simulated servers and the window [2400, 3300) s. Requests assigned to those servers differ across policies. Each request generates its full output.

The table pools seeds 41–43. Seed 41 uses the quiet-node run of each policy; the remaining replicate is a repeatability check. Seeds 42 and 43 were added after the initial seed's P90 intervals overlapped. The [archived replay protocol](REPLAY_PROTOCOL.md) records these decisions and their timing. Replay intervals resample requests within each seed. Ratio intervals resample policies independently; they do not pair requests across policies. Simulator error is `median(abs(measured_TTFT - simulated_TTFT) / simulated_TTFT)` over requests.

## Scope of the results

The 27–42% P90 TTFT reduction concerns the six standard-attention models at the full-speed point. The two hybrid-attention models do not improve at every percentile. The 54% replay reduction concerns Qwen3-8B at the constrained point. Its P99 TTFT and its P90/P99 TPOT advantages over Mooncake and ESP are not statistically significant.

The ideal prefill bound assumes equal GPUs and links, linear costs, helpers that already hold the prefix, idle resources, and negligible pipeline startup. One helper attains the global lower bound when `rho >= 1`; faster links can make additional helpers useful. Stall debt is a scheduling heuristic, not a bound on a server's stall rate.

The repository contains the source and measured costs used by this design. This release does not claim a new local rerun of the GPU campaign. Public traces, local model snapshots, and suitable GPU resources remain necessary for full reproduction.
