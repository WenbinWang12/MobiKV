# Reproducibility

The repository includes the final simulator, measured A800 costs, seven policy configurations, six ablations, GPU source, and analysis tools. Public traces and model weights are not bundled. The GPU programs have not been rerun locally for this release.

## CPU setup

Use Python 3.10 or later. Run these commands from the repository root:

```bash
python -m pip install -e .
python examples/smoke_test.py
```

The smoke test creates small synthetic request and mobility inputs. It checks that the simulator runs; it does not reproduce a paper result.

For development checks:

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```

## Input traces

Obtain the Mooncake and GeoLife inputs from the providers listed in [Data and citations](DATA_RELEASE.md). A request trace is JSONL with one object per request. It contains `timestamp` in milliseconds, `input_length` and `output_length` in tokens, and a `hash_ids` list that identifies shared prefix blocks. The simulator links a request to the latest earlier request whose block-hash list is a proper prefix of its own. No prompt text is needed by the simulator.

Prepare the GeoLife mobility archive:

```bash
python -m mobikv.preprocess.geolife_prep /path/to/Geolife.zip /path/to/geolife.npz
```

The `.npz` contains sampled coordinates `xy` and the arrays `trip_user`, `trip_mode`, `trip_len`, `trip_off`, `trip_test`, `trip_vmed`, and `trip_t0`. The preprocessor resamples positions every 10 seconds and splits users between tuning and evaluation pools. Its companion `.stats.json` records the source checksum and preprocessing counts.

For the Porto stress scenario, prepare the UCI archive with:

```bash
python -m mobikv.preprocess.porto_prep /path/to/porto.zip /path/to/porto.npz
```

Keep input versions and checksums with the run outputs. A different source revision, preprocessing result, or model context window can change which follow-up turns are evaluated.

## One simulator run

```bash
python scripts/run_mobikv.py \
  --method mobikv --model Qwen3-8B --op full \
  --trace /path/to/mooncake.jsonl --trips /path/to/geolife.npz \
  --seed 41 --out runs/qwen3_mobikv_seed41.json
```

The launcher defaults to `configs/costs_a800.json`. `--op full` selects a whole A800 and 10 Gb/s links; `slow` selects a quarter A800 and 0.93 Gb/s; `full1g` selects a whole A800 and 0.93 Gb/s. It applies the paper's load adjustment and policy parameters. Use the named launcher for paper comparisons rather than the simulator module's lower-level defaults.

[Experiment commands](EXPERIMENTS.md) lists the campaign and ablation options. [Evaluation](EVALUATION.md) defines the workload split, metrics, and statistical comparisons.

## Recompute statistics

To summarize a new table campaign:

```bash
python -m mobikv.analysis.simulation \
  --root runs/table_full --out runs/summary
```

The released per-seed summaries also allow CPU-only checks without downloading the traces:

```bash
python -m mobikv.analysis.simulation \
  --seeds docs/data/aggregate/simulation_seeds.csv \
  --campaign table_full --out runs/check
```

The analysis writes `latencies.csv`, `ratios.json`, and a per-seed CSV. Latency rows include the cell, method, metric, percentile, mean, sample standard deviation, and seed count. Comparisons use paired ratios across seeds. The released seed summaries store seconds to six decimal places and milliseconds to four; recomputed values can differ in the final digits from tables made from the original higher-precision outputs.

## GPU reproduction

[GPU reproduction](GPU_REPRODUCTION.md) specifies the environment, model paths, profiling, RPP measurements, and three-server replay. [The archived replay protocol](REPLAY_PROTOCOL.md) records the original run selection and the decision to add seeds 42 and 43. These historical records are preserved separately from the current portable command examples.
