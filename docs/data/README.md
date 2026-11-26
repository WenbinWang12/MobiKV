# Released aggregate data

These files are the aggregates of the final evaluation campaign (2026-11-24) behind the paper's tables and figures. They hold latency statistics only: no prompts, generated text, tokens, host names, or model weights. The public input traces are not redistributed; see [data sources and terms](../DATA_RELEASE.md).

All simulations use cost profiles measured on NVIDIA A800 80GB GPUs, the evaluation half of the sessions, and mobility seeds 41-60 unless stated otherwise. TTFT is in seconds and TPOT in milliseconds.

| File | Content |
| --- | --- |
| `aggregate/cross_model_full_speed.csv` | Six baselines and MobiKV for eight models at the full-speed point (one A800 per server, 10 Gb/s links): mean and sample standard deviation over 20 seeds of the P50, P90, and P99 follow-up TTFT and TPOT. |
| `aggregate/cross_model_constrained.csv` | The same at the constrained point (a quarter of an A800 per server, 0.93 Gb/s links). |
| `aggregate/cross_model_full_speed_slow_links.csv` | The same for full-speed GPUs with 0.93 Gb/s links. |
| `aggregate/rho_sweep_qwen3_a800.csv` | Follow-up TTFT against the transfer-to-recompute ratio of an 8k-token context for Qwen3-8B on full-speed servers, with the ratio varied through the link rate (column `rho`): per policy and ratio, the mean over 20 seeds and the mean minus/plus one sample standard deviation. |
| `aggregate/rpp_two_gpu_cells.csv` | One-helper replica-assisted parallel prefill measured on two A800 GPUs: one row per configuration (model, GPU share, context, link rate) with the single-GPU time, the ratio and its range, the best front fraction, and the speedup with its bound. `verified` is 1 when the delivered link rate was measured. |
| `aggregate/replay_three_gpu.csv` | Replay of the job streams of three servers on three A800 GPUs with real decoding (Qwen3-8B, constrained point, seven policies): follow-up TTFT and TPOT percentiles pooled over seeds 41-43 and per seed, the simulated TTFT, and the median relative error of the simulator. |
| `aggregate/replay_three_gpu_pooled.csv` | The pooled rows of the replay in long format with 95% bootstrap intervals (requests resampled within each seed). |
| `manifests/source_hashes.json` | SHA-256 of every file in this directory. |
| `aggregate/simulation_seeds.csv` | 8,050 final per-seed summaries: three operating points, the ratio sweep, 23 stress changes plus the default, and six individual ablations. TTFT is rounded to six decimal places and TPOT to four; rerunning statistics from these summaries can differ in the last digits from the original unrounded results. |

The eight-model list and measured cost profiles are in [`../../configs/model_matrix.json`](../../configs/model_matrix.json) and [`../../configs/costs_a800.json`](../../configs/costs_a800.json).

Policy keys: `follow_me_cloud` (Follow-Me), `wang_ton19_mdp` (MDP), `ouyang_jsac18` (Lyapunov), `preble_iclr25` (Preble), `mooncake_fast25` (Mooncake), `loongserve_sosp24` (ESP), and `mobikv`. The sweep file uses the short names `follow_me`, `mdp`, `lyapunov`, `preble`, `mooncake`, `esp`, and `mobikv`.

The figures of the project page are regenerated from these files by `scripts/plot_rho_sweep.py` and `scripts/plot_real_gpu.py`.

The seed CSV can be aggregated without the original input traces:

```bash
python -m mobikv.analysis.simulation --seeds docs/data/aggregate/simulation_seeds.csv \
  --campaign table_full --out runs/summary
```

Campaign names are `table_full`, `table_slow`, `table_full1g`, `rho_full_Qwen3-8B`, `stress_full_Qwen3-8B`, `ablation_full` and `ablation_slow`. Use `--comparisons each` for the ablations. A sweep cell named `gamma_...` is the transfer-to-recompute ratio for an 8k-token reference context, not a fixed context length for all requests.

The per-request simulation outputs and the raw replay records are not part of the repository. The aggregates released before 2026-11-25 came from an earlier design with proxy cost factors and have been removed; they remain in the Git history.
