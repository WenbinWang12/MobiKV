# MobiKV experiment protocol

This document records the public protocol for the MobiKV artifact. It is a compact companion to `docs/REPRODUCIBILITY.md` and the paper supplement.

## Workload

The simulator uses GeoLife mobility and the Mooncake conversation schema. Porto taxi and WildChat preprocessing are available for robustness experiments. The trace providers remain the source of the raw data. Each release manifest records the source URL, local source hash, preprocessing revision, and split seeds.

## Models and costs

The main simulator cost model is measured on the NVIDIA A800 80GB PCIe testbed in BF16. Cross-model entries record model revision, tokenizer revision, context and new-token lengths, KV bytes per token, measured or explicitly marked proxy costs, link goodput, and the resulting transfer-to-compute ratio `rho`.

## Policies

The adapted baselines use the same workload, cost model, KV store, repair primitive, and measurement protocol as MobiKV. The final MobiKV configuration uses one secondary replica, decode-first background replay, and replica-assisted parallel prefill when the predicted gain clears the configured threshold.

## Metrics

The public aggregate schema includes P50, P90, and P99 continuation TTFT, mean TTFT, P50, P90, and P99 TPOT, background GPU time, transferred bytes, RPP split counts, model and policy identifiers, ratio `rho`, seed count, request count, and invalid/retry counts. Standard deviation is the sample standard deviation across seeds.

## Real-GPU checks

`ct_hybrid.py` measures repair paths, `ct_pp.py` measures one-helper RPP, `ct_pp2.py` measures two helpers, and `sv3_replay.py` replays planned streams with real decoding. Exact-match checks compare greedy output tokens with the local chunked reference. Raw replay logs are reduced before any public upload.
