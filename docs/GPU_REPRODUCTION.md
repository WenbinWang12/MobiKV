# GPU measurements and replay

The repository contains the measurement and analysis code used by the final paper. Completed local checks were limited to static syntax/source inspection, eight `--help` commands, and three launcher dry runs. No CPU simulation, pytest suite, or GPU measurement was run. A reconstruction of the archived replay summary was interrupted before completion and is not counted as validation. A successful dry run verifies command construction, not timings or CUDA compatibility.

Run commands from the repository root in a Python environment with the `gpu` extra installed. The archived campaign used Linux, NVIDIA A800 GPUs, PyTorch 2.6.0, Transformers 5.16.1, NumPy, BF16 weights, and NVIDIA MPS. Model checkpoints must be obtained separately and supplied through `--model`. The three-worker replay uses three GPUs on one host; RPP measurements use two. Keep unrelated CPU simulation jobs off the measurement host. CPU contention affected the first replay pass despite zero throttled cgroup periods. The new launchers record CPU throttling counters in `run.json`.

The daemon must already be running with the intended GPU visibility. Export `CUDA_MPS_PIPE_DIRECTORY` and `CUDA_MPS_LOG_DIRECTORY` to its directories before launching. The launchers set the per-process GPU and active-thread percentage and check each worker PID in the daemon's `control.log`. They do not start or stop MPS. A share setting alone is insufficient evidence that the limit took effect. Each launcher requires a new or empty output directory and records commands, CPU counters, MPS attachment, and acceptance in `run.json`.

## Measured compute and state costs

[`profile_costs.py`](../mobikv/measurement/profile_costs.py) measures full prefill, cached-prefix prefill, batched decode, and returned cache bytes. It retains the final campaign's support for standard, sliding-window, and hybrid caches. [`build_costs.py`](../mobikv/measurement/build_costs.py) aggregates repetitions using the original fitting rules. The checked-in final profile is [`configs/costs_a800.json`](../configs/costs_a800.json).

```bash
python scripts/measure_costs.py --model "$QWEN3_MODEL" --tag Qwen3-8B \
  --gpu 0 --share 25 --out results/profile/qwen3_g25
python scripts/measure_costs.py --model "$QWEN3_MODEL" --tag Qwen3-8B \
  --gpu 0 --share 100 --out results/profile/qwen3_g100
python -m mobikv.measurement.build_costs --base configs/costs_a800.json \
  --out results/costs_new.json \
  results/profile/qwen3_g25/ct_compute.jsonl results/profile/qwen3_g100/ct_compute.jsonl
```

The builder retains only the base transfer parameters by default (`--keep` is empty). Supply all measured model files to rebuild the complete profile; the example builds the two Qwen3 compute entries. The builder requires a `done` row, matching recorded share, and MPS metadata. Verify the associated launch receipt before using the file. Its interpolated g50 decode values are not direct g50 measurements.

Use the same canonical model names as the final config. Measure each at shares 25 and 100. Pass model-specific grid options after `--`, for example `-- --no_crop` for the hybrid cache models and `-- --no_crop --max_ctx 8192` for Gemma. Batch sizes and context bounds must fit the GPU memory budget; the archived per-model profile rows are the authority for the exact grids. A different grid creates a new profile, not a byte-identical reconstruction of the published one. The model is never inferred from a local checkpoint directory name.

Append `--dry-run` before the final `--` to inspect a launch without loading CUDA. All profile options are listed by:

```bash
python -m mobikv.measurement.profile_costs --help
```

## Two-GPU replica-assisted parallel prefill

[`ct_pp.py`](../mobikv/measurement/ct_pp.py) runs the helper and executor, transfers each newly computed front segment's KV state layer by layer, and makes the executor wait for that layer before processing the back segment. Both workers start with the cached prefix. The launcher records single-GPU references, split timings, token agreement, logit differences, and arrival timestamps.

```bash
python scripts/measure_rpp.py --model "$QWEN3_MODEL" --gpus 0,1 --share 100 \
  --out results/rpp/qwen3_8b_g100
python scripts/measure_rpp.py --model "$QWEN3_MODEL" --gpus 0,1 --share 25 \
  --caps 930,2500,10000 --fracs 0.125,0.25,0.375,0.5,0.625 \
  --out results/rpp/qwen3_8b_g25
python -m mobikv.analysis.rpp results/rpp \
  --csv results/rpp_cells.csv --json results/rpp_checks.json
```

The default full-share grid uses caps 930, 2500, 5000, and 10000 Mb/s, three repetitions, and front fractions 0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, and 0.625. Each cap uses `(prefix, total)` lengths `(2048, 4096)`, `(4096, 8192)`, and `(8192, 16384)`. The final analysis includes five full-share runs (`qwen25_7b`, `qwen25_14b`, `qwen3_8b`, `mistral_7b`, and `llama31_8b`) and two quarter-share runs (`qwen3_8b` and `llama31_8b`). Use these directory slugs followed by `_g100` or `_g25`; the analysis maps them to paper model names. Together those grids contain 78 cells. A launch attempted for Qwen2.5-7B at quarter share is not part of that final set.

The analysis retains the conservative delivered-rate estimate and distinguishes verified rates from intervals when helper compute limits arrival. It also reports the exact number of correctness comparisons: logits against local chunked prefill are checked only in the first repetition; greedy tokens against monolithic prefill are checked in every trial. Do not describe these as identical tests or assume a configured cap was delivered. Run the analysis on the complete final grid: its summary expects both verified and unverified rate cells.

## Three-GPU replay

[`replay.py`](../mobikv/measurement/replay.py) is the final paced transport harness. It replays foreground jobs, background work, RPP dependencies, and batched decoding from simulator job logs. Generate those logs with the final simulator and the matching policy configuration; pass `--joblog PATH --jl_t0 2400 --jl_t1 3300` to `python -m mobikv.simulator` in addition to the simulation options. Arrange the resulting files as `plans/<policy>/joblog.json`.

```bash
python scripts/run_replay.py --joblog-root plans/seed41 \
  --out results/replay/replay_slow --model "$QWEN3_MODEL" --gpus 0,1,2
python scripts/run_replay.py --joblog-root plans/seed41 \
  --out results/replay/replay_slow_rep2 --model "$QWEN3_MODEL" --gpus 0,1,2
python scripts/run_replay.py --joblog-root plans/seed42 \
  --out results/replay/replay_slow_seed42 --model "$QWEN3_MODEL" --gpus 0,1,2
python scripts/run_replay.py --joblog-root plans/seed43 \
  --out results/replay/replay_slow_seed43 --model "$QWEN3_MODEL" --gpus 0,1,2
python -m mobikv.analysis.replay results/replay \
  --json results/replay_pool.json --csv results/replay_pool.csv
```

The defaults match the reported operating point: Qwen3-8B checkpoint supplied by the caller, 25% MPS share, 930 Mb/s paced egress, decode batch limit 8, context buffer 32768, and edges `(1, 2)`, `(2, 2)`, `(1, 3)`. The replay window comes from the job log. All seven policies run by default; use `--policies mobikv loongserve_sosp24` to select a subset. The final MobiKV plans use `bg_dfb=1`; the six baselines use 0. The launcher matches those settings by policy name. `--bg-dfb` overrides this only for plans generated with a different setting. Append `--dry-run` to inspect all commands and validate input paths without loading models.

A run is accepted only if every foreground job finishes, send errors and layer-wait timeouts are zero, all worker processes succeed, and MPS attachment is verified. Review the CPU counters in addition to acceptance; the receipt does not assert that the host was quiet. The launchers preserve outputs and fail if a policy directory already contains results. Use a new directory for each repeat.

The analysis intentionally preserves the published selection rule: for seed 41, MobiKV and ESP use `replay_slow_rep2`; the other five policies use `replay_slow`. The other pass is a repeatability check. Seeds 42 and 43 use their respective directories. This records the archived choice after the initial MobiKV/ESP runs overlapped CPU simulation work; do not silently reinterpret it as a general rule for selecting the best run in a new campaign. All four directory trees, including both complete seed-41 passes, are required by this analysis.

The analysis pools follow-up requests (`turn >= 2`) from evaluation-half sessions at the three replayed servers. It reports P50/P90/P99 with 2000 request bootstrap resamples stratified by seed. Method ratios resample the policies independently and retain asymmetric 95% intervals. Simulator error is the median of `abs(measured TTFT - simulated TTFT) / simulated TTFT` for positive simulated values. A ratio interval crossing one does not establish a difference.

This is a replay of a fixed simulated schedule. TTFT combines simulated request timing with the measured completion shift of the final job. Helpers outside the three replayed servers use schedule-derived arrivals. Decode uses synthetic token IDs and a batch mean context in preallocated buffers. The harness therefore checks the timing model and compares replayed schedules; it is not an online serving implementation. The final paper does not claim a full-share 10 Gb/s replay result: the tested same-host transport did not deliver that rate.

## Supplement S10 mechanisms

The earlier mechanism experiments remain part of the final supplement. Their code is retained:

| Module | Measurement | Worker roles |
| --- | --- | --- |
| `mobikv.measurement.ct_hybrid` | Transfer, recompute, and hybrid prefix handoff | `sender`, `receiver` |
| `mobikv.measurement.ct_pp` | One helper computing a front segment | `helper`, `server` |
| `mobikv.measurement.ct_pp2` | One- and two-helper parallel prefill | `h1`, `h2`, `x` (executor) |

These modules accept explicit checkpoint paths, peers, ports, cells, repetitions, and output paths. Run their `--help` commands in the GPU environment. For cross-host experiments, launch the peer roles on their physical hosts and supply reachable addresses rather than the same-host loopback used by the new wrapper. `ct_hybrid --cells` uses `name:total:cached_prefix:cap`; `ct_pp` and `ct_pp2` use `name:cached_prefix:total:cap`. `ct_pp2` also accepts `--h1`, `--h2`, `--up`, and `--up_port` for the helper dependency chain. The final two-GPU wrapper does not replace that cross-host topology or the two-helper experiment.

## Source provenance and scope

The final sources were recovered from `raw_small_20261124.tgz`. The migration preserves the measurement and statistical algorithms; changes provide package imports, public names, explicit local arguments, and launch checks.

| Archive source | Public module |
| --- | --- |
| `claude_20261122/code/ct_grid2.py` | `mobikv.measurement.profile_costs` |
| `claude_20261122/code/build_costs.py` | `mobikv.measurement.build_costs` |
| `claude_20261123/code/sv3_replay_v2.py` | `mobikv.measurement.replay` |
| `claude_20261123/code/replay_pool.py` | `mobikv.analysis.replay` |
| `claude_20261123/code/rpp_cells.py` | `mobikv.analysis.rpp` |

The three `ct_*` mechanism modules match the archived mechanisms, with formatting and package-import changes. Separate obsolete replay implementations and the old campaign analyzers were removed. GPU datasets, model weights, daemon configuration, and original private-machine paths are not bundled. The launchers are new portable orchestration, so their GPU execution still needs validation on a compatible host.
