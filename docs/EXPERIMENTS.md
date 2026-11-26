# Experiment commands

Run commands from the repository root after following [Reproducibility](REPRODUCIBILITY.md). `run_campaign.py` expands the final policy settings into simulator commands. `--dry-run` prints those commands without running them; omit it to execute the campaign.

## Model comparison

```bash
python scripts/run_campaign.py --campaign table \
  --trace /path/to/mooncake.jsonl --trips /path/to/geolife.npz \
  --op full --out-root runs --dry-run
```

The default model comparison uses eight models, seven policies, and seeds 41-60: 1,120 runs per operating point. Use `--op slow` for the constrained point or `--op full1g` for full-speed GPUs with slow links. `--models Qwen3-8B` and `--seeds 41,42` select smaller runs while checking an installation.

Outputs are stored under `runs/table_{op}/{model}/{method}/seed_N.json`.

| Method ID | Policy |
| --- | --- |
| `follow_me_cloud` | Follow-Me |
| `wang_ton19_mdp` | MDP migration |
| `ouyang_jsac18` | Lyapunov placement |
| `preble_iclr25` | Preble |
| `mooncake_fast25` | Mooncake |
| `loongserve_sosp24` | ESP |
| `mobikv` | MobiKV |

The baselines adapt their published decision rules to the shared simulator. Their papers are cited in [Data and citations](DATA_RELEASE.md#baselines).

## Link-rate sweep

```bash
python scripts/run_campaign.py --campaign rho \
  --trace /path/to/mooncake.jsonl --trips /path/to/geolife.npz \
  --out-root runs --dry-run
```

This campaign uses Qwen3-8B on full-speed GPUs, 18 link-rate settings, seven policies, and 20 seeds: 2,520 runs. The command retains the historical name `rho`, but the swept coordinate is the paper's `gamma_8k`: the transfer-to-recompute ratio of an 8k-token reference context. It does not fix every request to an 8k-token context.

Outputs are stored under `runs/rho_full_Qwen3-8B/gamma_{g}/{method}/seed_N.json`.

## Stress tests

```bash
python scripts/run_campaign.py --campaign stress \
  --trace /path/to/mooncake.jsonl --trips /path/to/geolife.npz \
  --porto-trips /path/to/porto.npz --out-root runs --dry-run
```

The default stress campaign uses Qwen3-8B on full-speed GPUs, 23 scenarios, seven policies, and ten seeds: 1,610 runs. The Porto scenario requires the separately prepared mobility input. Outputs are stored under `runs/stress_full_Qwen3-8B/{scenario}/{method}/seed_N.json`.

## Ablations

```bash
python scripts/run_campaign.py --campaign ablation \
  --trace /path/to/mooncake.jsonl --trips /path/to/geolife.npz \
  --out-root runs --dry-run
```

The default campaign compares MobiKV with six variants at three model/operating-point cells: Qwen2.5-7B/full, Qwen3-8B/full, and Qwen2.5-7B/slow. With 20 seeds, it contains 420 runs.

| Variant ID | Change from MobiKV |
| --- | --- |
| `no_parallel_prefill` | Disable RPP |
| `no_replica` | Remove the secondary replica |
| `hybrid_refresh` | Allow replica refresh to use GPU recomputation |
| `nearby_execution` | Restrict execution to nearby candidates |
| `no_decode_aware` | Remove the decode-aware routing term |
| `no_cold_helper` | Disable helpers that must first receive the prefix |

Outputs are stored under `runs/ablation_{op}/{model}/{method}/seed_N.json`. To run a single variant, pass its ID to `scripts/run_mobikv.py --method`.

Compare MobiKV with each variant separately when aggregating ablations:

```bash
python -m mobikv.analysis.simulation \
  --root runs/ablation_full --comparisons each --out runs/ablation_summary
```

Each ratio is MobiKV latency divided by the variant's latency, paired over seeds. A ratio below 1 means the removed element reduces latency.

## Analysis and GPU experiments

Use `python -m mobikv.analysis.simulation --root runs/table_full --out runs/summary` to aggregate simulator outputs. [Reproducibility](REPRODUCIBILITY.md#recompute-statistics) also shows how to recompute statistics from the released per-seed summaries.

GPU profiling, the two-GPU RPP grid, and three-server replay have separate hardware requirements. Their entry points and commands are in [GPU reproduction](GPU_REPRODUCTION.md). The [evaluation protocol](EVALUATION.md) and [archived replay protocol](REPLAY_PROTOCOL.md) define what the resulting measurements support.
