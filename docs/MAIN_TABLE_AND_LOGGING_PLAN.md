# Main table and experiment logging plan

This plan records the fixed result schema used by the completed cross-model campaign. Every model and method uses the same 8k-token comparison grid, BF16, A800 80GB, the selected MPS service share, the measured link rate, the same workload version, and five mobility seeds.

## Primary claim and Table I

Table I is the end-to-end latency comparison for every model and every adapted baseline plus MobiKV. Each model block reports `rho_8k = X_8k / C_8k` once. Qwen2.5 uses measured fits. The additional model blocks use A800-anchored proxy cost profiles until full per-model CT grids are available. Each method reports follow-up TTFT and TPOT at P50, P90, and P99. Every percentile is written as `mean +/- sample-std` across seeds.

The aggregation order is fixed. For each seed, compute request-level percentiles. Then compute the mean and sample standard deviation of those seed-level percentiles. Do not pool requests from different seeds before computing a percentile. Store the number of seeds, requests, invalid requests, and retries beside every aggregate.

First-turn latency, TPOT with batched decoding, and TPOT with prefill stalls remain available in the raw records and companion tables. They do not make Table I wider than needed for the primary claim.

## Companion result tables

### Table II records model cost and mechanism evidence

- `C_8k`, `X_8k`, measured link goodput, and `rho_8k`
- BF16 KV state bytes per token and peak memory
- cost-fit held-out error
- repair speedup for recompute, transfer, and hybrid paths
- RPP speedup, split acceptance rate, helper count, overlap, and last-layer slack
- greedy-token agreement and logit difference

Qwen3.5 and Qwen3.8 report the recurrent state and full-attention state separately. Their total state is measured from the implementation instead of being inferred from a GQA KV formula.

The current KV-only CT harness is limited to cache layers exposing K/V tensors. Qwen3.5 and Qwen3.8 use interleaved linear-attention and full-attention layers, so their smoke measurements are retained and their Table-I rows are explicitly marked simulation-only pending a recurrent-state-aware harness.

### Table III covers sensitivity and ablations

- link rate and compute service share `g`
- context length and new-token length
- offered load and mobility trace
- homogeneous and heterogeneous links
- RPP on or off
- decode-first versus immediate replay
- helper count, replica count, cache placement, and repair mode

### Table IV reports overhead and external validation

- foreground and background GPU seconds
- GPU utilization, peak memory, CPU and RAM usage
- ingress and egress bytes and goodput
- KV-store occupancy and replication traffic
- controller decision time, retries, timeouts, and fallback rate
- simulator versus real replay latency error
- OpenAI endpoint model ID, timestamp, token counts, TTFT, total latency, usage, and HTTP errors

The hosted OpenAI endpoint does not expose KV state, repair, or RPP internals, so those fields are not inferred for that row.

## Online replay validation

The planned online replay validates the simulator's timing model on real GPUs. It uses Qwen2.5-7B, Qwen2.5-14B, and Qwen3-8B with MobiKV, ESP, and the best adapted baseline for five seeds. The run pins the checkpoint revision, tokenizer, chat template, block size, KV-cache key/hash, BF16 decoding mode, request order, arrival and think-time gaps, parent dependencies, prefix-sharing relations, mobility decisions, and output-token budget. EOS is disabled until the recorded budget is reached.

If raw prompts are available for a trace window, the replay uses those prompts. Otherwise it materializes deterministic token fixtures that preserve the recorded lengths and shared-prefix IDs. Raw-prompt and fixture runs are reported separately. The manifest records which input mode was used, along with GPU SKU, driver, CUDA, MPS share, thermal state, replayed-host coverage, and the fraction of external helpers. Each seed stores online TTFT and TPOT percentiles, simulator-to-online P99 ratio, direct-versus-RPP exact-match rate and logit tolerance, generated-token counts, and resource counters. These records populate the planned online-replay table and figure in the paper.

## Per-request record

Every request is written to `raw/request_metrics.jsonl` with the following fields.

```text
run_id,campaign_id,model,model_revision,method,seed,trace,
request_id,session_id,turn_id,turn_type,ctx_tokens,new_tokens,
prompt_tokens,output_tokens,arrival_ts,prefill_start_ts,first_token_ts,end_ts,
ttft_s,total_s,tpot_b_ms,tpot_s_ms,queue_wait_s,prefill_s,decode_s,
repair_s,recompute_s,kv_transfer_s,helper_wait_s,executor,holder,helper,
cache_hit,cached_tokens,missing_tokens,split_token,split_fraction,kv_bytes,
bytes_sent,measured_goodput_mbps,rho,fallback,timeout,valid,error_code
```

RPP records also include eligible and accepted split counts, predicted and measured ready time, transfer and compute overlap, last-layer slack, helper count, greedy-token agreement, and logit difference.

## Per-run artifacts

Each run keeps the following files so that every percentile can be recomputed later.

```text
manifest.json
raw/request_metrics.jsonl
summary/percentiles_by_seed.csv
summary/aggregate.csv
cost_fit.json
resource_metrics.csv
checksums.txt
```

`manifest.json` records the repository revision, configuration hash, model revision, tokenizer revision, hardware and software versions, GPU count and SKU, MPS share, link probe, workload and trace version, seed, context and output-length grid, policy parameters, and replay mode. `aggregate.csv` contains `mean`, `std`, `n_seed`, `n_req`, `n_invalid`, and `n_retry` for every metric. Standard deviation uses the sample convention with denominator `n_seed - 1`.

## Campaign order

1. Re-run Qwen2.5-7B and Qwen2.5-14B under this unified schema so their archived pooled summaries do not mix with the new table.
2. Complete Qwen3-8B, Llama-3.1-8B, Mistral-7B, and Gemma-2-9B after their cost fits and CT/RPP checks.
3. Run the text causal-LM path for Qwen3.5-9B and Qwen3.8-27B after fitting their hybrid recurrent and full-attention states.
4. Run the five-seed replay for all six adapted baselines and MobiKV, then generate Table I from `aggregate.csv`.
5. Run the sensitivity, ablation, overhead, and real-replay validation tables.

The completed Table-I campaign has 245 raw summaries across seven additional model blocks, seven methods, and five seeds. Qwen2.5 rows use the measured cost fit. Proxy rows remain clearly identified in the paper and in `configs/costs_cross_model_v1.json` until their full CT grids are collected.

\n