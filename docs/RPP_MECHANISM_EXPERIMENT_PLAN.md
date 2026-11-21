# RPP mechanism evidence plan

## Claim

RPP lowers the ready time of a cached-prefix repair when transfer and helper
recompute overlap. The gain should follow the cut bound as the measured
transfer-to-recompute ratio `rho` changes, and the split must preserve the
single-server output.

## Exact-GPU grid

The g=25 campaign uses two independent A800 helper/executor pairs.

| factor | values |
|---|---|
| models | Qwen3-8B, Mistral-7B-v0.3 |
| MPS share | 25% |
| cached prefix/end | 4096/8192, 8192/16384 |
| link cap | 930, 2500, 5000, 10000 Mb/s |
| split fraction | 0.25, 0.375, 0.50, 0.625, 0.75 |
| repetitions | 3 |
| output check | 32 greedy tokens and logit differences |

This gives 240 split trials. Existing g=100 runs in
`results/rpp_remote/{qwen3,mistral,llama31}_ct.jsonl` add a full-share
comparison with caps 300 and 930 Mb/s.

## Per-trial records

`ct_pp.py` records single-server time, split ready time, helper compute and
send time, first and last layer arrival, executor wait, bytes sent, measured
link probe, and output agreement. Define

```text
speedup = single_ms / split_ms
rho     = transfer_only_ms / single_ms
overlap = helper_send_ms - helper_comp_ms
bound   = 1 + min(rho, 1/rho)
```

Use the median of the three repetitions for a split point. Compute the
out-of-sample best split from the measured grid and report its predicted and
measured ready time separately.

## Paper artifacts

**Table II, mechanism evidence** should report model, MPS share, context,
cap, measured `rho`, best split, single time, hybrid ready time, speedup,
bound ratio, ready-time prediction error, and token agreement.

**Figure 2, speedup versus rho** should plot each split/grid point and the
bound `1 + min(rho, 1/rho)`. Use separate markers for Qwen3 and Mistral and
show the g=25 and g=100 groups.

**Figure 3, pipeline timing** should show one representative run with the
helper compute, layer-wise KV transfer, executor wait, and back prefill on a
shared time axis.

The text should state the token-agreement count and logit difference for all
trials before discussing speedup. Any model with incomplete agreement is
excluded from the correctness claim and retained as a diagnostic.
