# MobiKV

MobiKV is the research artifact for **MobiKV: Replica-Assisted KV Cache Scheduling for Multi-Turn LLM Serving at the Mobile Edge**.

The repository implements the paper's final system, **MobiKV-DF + RPP**:

- decouple the user's access edge from the execution edge;
- keep one secondary KV replica and refresh it during think time;
- repair missing KV with the measured compute/transfer ratio `rho`;
- use replica-assisted parallel prefill (RPP) when a replica can compute the
  front of a long turn while the serving edge computes the back; and
- give decoding strict priority over background replay (decode-first, or DF).

The simulator is trace driven. It uses measured GPU prefill/decode costs and
link goodput, then models foreground prefill, decode, KV transfer, background
replication, mobility, failures, heterogeneous links, and the RPP pipeline.
The real-GPU programs measure the costs used by the simulator and replay a
planned job stream with layer-by-layer RPP and batched decoding.

## Repository layout

| Path | Purpose |
| --- | --- |
| `mobikv/simulator.py` | A9 simulator used for the final MobiKV-DF/RPP evaluation. The `--bg_dfb`, `--pp`, `--link_het`, `--helper_fail`, and `--bg_slow` switches implement the final and robustness variants. |
| `mobikv/measurement/ct_hybrid.py` | Measure full prefill, cached-prefix recompute, and KV transfer/hybrid repair on two hosts. |
| `mobikv/measurement/ct_pp.py` | Measure one-helper RPP with layer-wise KV streaming. |
| `mobikv/measurement/ct_pp2.py` | Measure the two-helper chain used for the k=2 theory check. |
| `mobikv/measurement/sv3_replay.py` | Replay simulator job logs on real GPUs with real decoding, RPP, and decode-first background replay. |
| `mobikv/preprocess/` | Convert GeoLife, Porto taxi, and WildChat data to the simulator formats and fit `costs.json`. |
| `mobikv/analysis/` | Analysis scripts for A9 simulation and SV-3 replay output. |
| `configs/costs_v1.json` | The measured Qwen2.5-7B/14B cost model released with the paper artifact. |
| `scripts/run_mobikv.py` | Preset launcher for MobiKV, MobiKV-DF, MobiKV+RPP, and MobiKV-DF+RPP. |
| `examples/` | Small synthetic inputs and a CPU smoke-test command. |
| `docs/` | Reproduction notes, experiment settings, and the reported final results. |

Raw traces and model weights are intentionally not bundled. The preprocessing
scripts accept local copies of the public GeoLife, Porto, and WildChat data;
the paper's Mooncake trace is supplied in the same JSONL format.

## Installation

The supported environment is Linux, Python 3.10, CUDA 12.4, PyTorch 2.6,
and Transformers 5.16.1. For CPU-only simulation and preprocessing:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The real-GPU measurement and replay programs additionally require three
hosts with one A100/A800-class GPU each, password-less SSH between the hosts,
and a shared model directory. Set `MOBIKV_MODEL` to a local Hugging Face
snapshot. Do not put model weights in this repository.

## Run the simulator

The simulator expects:

1. a JSONL request trace with `timestamp` (ms), `input_length`,
   `output_length`, and `hash_ids`;
2. an `.npz` mobility file produced by `geolife_prep.py` or `porto_prep.py`;
3. a fitted cost model such as `configs/costs_v1.json`.

Use the final system preset:

```bash
python scripts/run_mobikv.py \
  --preset mobikv-df-rpp \
  --trace /path/to/mooncake.jsonl \
  --trips /path/to/geolife_test.npz \
  --costs configs/costs_v1.json \
  --out runs/default.json \
  --seed 31 --tpot 1
```

The preset expands to the frozen paper configuration:

```text
--policy mobikv --route_first p2c --cont_min 0 --beta 1
--pp 1 --pp_cold 1 --pp_ctrl dpp --pp_V 30 --pp_eps 0.05
--pp_min_rel 0.25 --pp_qmin 1024 --pp_grid 32 --bg_dfb 1
```

The other presets are `mobikv`, `mobikv-df`, and `mobikv-rpp`. Any simulator
option can follow the preset and overrides it. For the A9 robustness checks,
the simulator exposes `--link_het`, `--link_het_known`, `--helper_fail`, and
`--bg_slow`; these options are disabled by default.

Run the bundled CPU smoke test with:

```bash
python examples/smoke_test.py
```

It creates synthetic traces in a temporary directory, runs MobiKV-DF+RPP,
and checks that the simulator emits continuation TTFT and TPOT summaries.

## Measure and replay on GPUs

The measurement programs use one process per role. A typical RPP measurement
has a helper and executor process:

```bash
# helper host
python mobikv/measurement/ct_pp.py --role helper --model "$MOBIKV_MODEL" \
  --peer EXECUTOR_IP --cells K1:8192:18432:0 --out helper.jsonl

# executor host
python mobikv/measurement/ct_pp.py --role executor --model "$MOBIKV_MODEL" \
  --peer HELPER_IP --cells K1:8192:18432:0 --out executor.jsonl
```

See `docs/REPRODUCIBILITY.md` for the three-host layout, MPS settings,
measurement grids, and the SV-3 replay command. `fit_costs.py` combines the
measurement JSONL files into the simulator cost-model schema.

## Final reported results

The final evaluation uses fresh seeds 31--35 for the main comparison and
seeds 21--24 for paired real-GPU replays. MobiKV-DF+RPP has continuation p95
TTFT ratios of 0.21--0.65 against the six adapted published baselines and a
0.74 [0.70, 0.78] ratio against MobiKV-DF without RPP. In the paired replay,
RPP reduces continuation p95 from 4.86 s to 4.05 s (0.83 [0.72, 0.98]) while
p95 TPOT remains unchanged (0.98 [0.95, 1.03]). The simulator and replay
limitations, including the selected paper scope and late replicas, are
documented in `docs/REPRODUCIBILITY.md`.

## Citation

```bibtex
@article{wang2026mobikv,
  title   = {MobiKV: Replica-Assisted KV Cache Scheduling for Multi-Turn LLM Serving at the Mobile Edge},
  author  = {Wang, Wenbin and Wang, Runan and Liu, Jiangchuan and Hu, Xiping},
  year    = {2026}
}
```

## License

MIT; see `LICENSE`.
