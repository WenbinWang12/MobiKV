# Cross-model campaign

The campaign uses one common context grid of 1,024, 2,048, 4,096, and 8,192 tokens. A model enters the main result table only after the following checks complete on the same BF16 stack and A800 node.

1. Load and tokenizer/API smoke, including one cached-prefix prefill and one decode step.
2. Full prefill, cached-prefix prefill, transfer-only repair, recompute-only repair, and hybrid repair.
3. One-helper RPP over the fixed split grid, with greedy-token agreement against a local reference.
4. A model-specific cost file fitted from the measurement rows, with held-out full-prefill error and measured transfer goodput.
5. Trace replay using the same seeds, network rates, MPS shares, baseline policies, and decode-first rule.

Use three or more repetitions for isolated GPU measurements and five paired mobility seeds for system results. Record the GPU SKU, MPS share, context and new-token lengths, peak BF16 memory, KV bytes per token, transfer goodput, rho, p50/p95 TTFT, p95 TPOT, RPP speedup, repair speedup, and greedy-equivalence rate.

The fitter accepts arbitrary model keys instead of assuming 7B and 14B. For example:

```bash
python -m mobikv.preprocess.fit_costs MEASUREMENTS costs-qwen3.json \
  Qwen3-8B=147456 --context=Qwen3-8B=40960
```

The simulator reads `model_contexts` from the cost file. The common comparison can also enforce the 8k grid explicitly with `--context_limit 8192`.

Downloaded ModelScope checkpoints use the immutable linked ETag recorded in `configs/model_matrix.json`. A row with a null revision remains planned until its source checkpoint is fetched and smoke-tested.

The MPS percentage is reported as a controlled compute-service budget. It is paired with the measured 0.93 Gb/s application token-bucket path and the node topology, rather than being described as a cloud GPU SKU.

An OpenAI model can be measured with `scripts/openai_api_replay.py` as an external end-to-end baseline. It reports streaming TTFT, total latency, generated text, output tokens, and API usage. The API run does not populate KV/token, repair, or RPP columns because those internals are unavailable through the hosted endpoint.
