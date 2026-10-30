# ReState

Code for the paper **"ReState: Contention-Aware Decisions on Remote KV State for Edge–Cloud LLM Serving"**.

When a request's prompt prefix is already cached as KV state on another GPU (a *holder*), the serving system can:

- decode on the holder;
- attend to the remote KV over the network;
- progressively repair the KV onto the target GPU while decoding;
- recompute the prefix locally.

ReState chooses among these actions per request with a calibrated cost model that prices contention on the target and holder resources.

This repository contains the multi-holder serving runtime, the planners and baseline policies, the experiment pipelines and the analysis scripts behind the paper's experiments.

> Status: research code. It was run on three physical hosts, each with one NVIDIA A100/A800 80 GB GPU, using Qwen2.5-14B (and Qwen2.5-7B) in BF16 with TP = 1. Every pipeline writes immutable, hash-bound plans and records, so a run can be resumed and audited.

## What is in the box

| Path | Content |
|---|---|
| `restate/mh_runner.py`, `mh_launch.py`, `mh_destination.py`, `mh_source_server.py`, `mh_group_execution.py`, `mh_campaign.py` | Multi-holder runtime: one target GPU and two holder GPUs on three distinct hosts. Per request the executed action is **holder decode**, **remote attention** (per-layer RPC to the holder), **progressive repair** (remote attention while the prefix KV migrates layer by layer, then local) or **recompute** |
| `restate/e0_*`, `e1_*`, `e3_*`, `e4_*`, `campaign_*`, `full_*`, ... | Layered decode engine, KV transport, migration and bandwidth scheduler, numerical validation, used by the runtime above |
| `restate/mh_plan.py` | Cost-model fitting from calibration, and the policies: `full_controller` (contention-priced selection), `ablate_no_contention`, `least_loaded`, `static_holder`, `static_recompute`, `static_repair`, `threshold_bstar` (INFOCOM B\* rule), plus plan builders for E7/X1/X3–X5 |
| `restate/supp_plan.py` | Plan builders for the supplementary experiments: heterogeneous holders (HET), load-matched compute throttling (LM), second model (M7B), holder-departure deadlines (DEP, `full_controller_dl`) |
| `restate/mh_pipeline.py`, `x1_pipeline.py`, `x34_pipeline.py`, `c3t_pipeline.py`, `supp_pipeline.py` | Experiment pipelines: calibration, frozen plan, run, verified cleanup, analysis |
| `restate/x34_analysis.py`, `x1_analysis.py`, `supp_analysis.py`, `supp_release.py`, `mh_analysis.py`, `supp_predacc.py` | Block-paired analysis (10,000-draw bootstrap, ±5 % labels), action selection, holder-release extraction, cost-model prediction accuracy |
| `restate/mh_netem.py`, `scoped_netem.py` | Scoped holder-ingress netem used for the RTT sweep; needs `NET_ADMIN` and an iproute2 `tc` |
| `restate/queued_network_probe.py` | Application-level dual-holder rate probe, used to select rates against B\* |
| `restate/mps_bench.py` | Prefill benchmark under CUDA MPS SM caps (compute throttling) |
| `restate/restate_config.py` | **All site-specific settings** (paths, IPs, model identity), read from environment variables |
| `patches/qwen2.5-7b.patch` | The only code changes needed for Qwen2.5-7B: geometry check and KV bytes per token |
| `tools/make_model_manifest.py` | Writes the model manifest that every worker verifies at start-up |

## Requirements

- **Hosts**: three hosts on **distinct physical machines**; the launcher checks hostname, boot id and DMI UUID. Each host needs one 80 GB A100/A800 and a shared filesystem mounted at the same path on all three. Password-less SSH from the target host to both holders is required.
- **Software**: Python 3.10, CUDA 12.4, and the packages in `requirements.txt`; tested with torch 2.6.0, transformers 5.16.1 and huggingface_hub 1.29.0.
- **Optional**:
  - iproute2 `tc` with IFB/netem plus `NET_ADMIN`, for the RTT experiment;
  - CUDA MPS (`nvidia-cuda-mps-control`), for the compute-throttling experiments.
  - Note: an MPS cap on only *some* GPUs changes kernel selection, and remote attention then fails the runtime's numerical gate. Throttle either all three GPUs or none.

## Setup

```bash
pip install -r requirements.txt
# model snapshot (any HF mirror works)
hf download Qwen/Qwen2.5-14B --revision 97e1e76335b7017d8f67c08a19d103c0504298c9 --local-dir $MODEL_DIR
export RESTATE_MODEL=$MODEL_DIR
export RESTATE_MODEL_MANIFEST_SHA256=$(python tools/make_model_manifest.py $MODEL_DIR Qwen/Qwen2.5-14B 97e1e76335b7017d8f67c08a19d103c0504298c9)
export RESTATE_PYTHON=$(which python)       # same interpreter path on all hosts
export RESTATE_TARGET_IP=<target internal IP>
export RESTATE_HOLDER_IPS=<holder0 IP>,<holder1 IP>
export RESTATE_ROUTE_PROBE_IP=<any IP routed via the experiment NIC>
export RESTATE_TC_ROOT=<iproute2 prefix>     # RTT experiment only
```

- Put the repository on the shared filesystem and run every pipeline **on the target host**. Holders are addressed as `IP:GPU`, for example `--holders $H0:0 $H1:0`.
- For **Qwen2.5-7B**: apply `patches/qwen2.5-7b.patch`, and set `RESTATE_MODEL`, `RESTATE_MODEL_ID`, `RESTATE_MODEL_REVISION` and `RESTATE_MODEL_MANIFEST_SHA256` for the 7B snapshot.

## Reproducing the experiments

Each command writes a run directory containing `pipeline-status.jsonl`, the frozen plans, the normalized records and an `analysis/` folder. Pipelines refuse to run if a frozen input changes, and they resume in place after a failure. `--prereg` is any text file you want frozen with the run; it is hash-bound into the results.

| Experiment | Command (run on the target host) |
|---|---|
| Base calibration (cost model, B\*) | `python restate/mh_pipeline.py --root RUN/BASE --holders $H0:0 $H1:0 --target-gpu 0 --stages cal` |
| E7 cells with the `least_loaded` baseline (X1) | `python restate/x1_pipeline.py --root RUN/X1 --holders $H0:0 $H1:0 --prereg PREREG.md` |
| Rate sweep with and without holder decode (X3/X4), RTT sweep (X5) | `python restate/x34_pipeline.py --root RUN/SWEEP --holders $H0:0 $H1:0 --base-model RUN/BASE/cost-model.json --rates R1 R2 R3 --x5-rate R1 --stages cal,x34,x5 --prereg PREREG.md` |
| Compute-throttled B\* sweep (C3T) | Start MPS on all three hosts with `set_default_active_thread_percentage P`, then `python restate/c3t_pipeline.py --root RUN/C3T --mps-percentage P --prereg PREREG.md --throttle-selection SEL.json`. `SEL.json` holds at least `{"mps_percentage": P}`; `mps_bench.py` measures prefill slowdown per cap |
| Heterogeneous holders | `python restate/supp_pipeline.py --experiment HET --mode full --root RUN/HET --holders ... --prereg P.md --base-model RUN/BASE/cost-model.json --rate 1000000000 --loaded-holder 1` |
| Load-matched throttling | `... --experiment LM --mode full --cost-model RUN/C3T/SWEEP-A/cost-model.json --native-model RUN/BASE/cost-model.json --rates 400000000 2500000000 --mps <ip>=20 <ip>=20 <ip>=20` |
| Second model | `... --experiment M7B --mode full` (with the 7B patch and model settings) |
| Holder-departure deadlines | `... --experiment DEP --mode full --base-model RUN/BASE/cost-model.json --rate 2500000000` |
| Cost-model prediction accuracy | `python restate/supp_predacc.py OUT.json NAME=<stage_dir>=<cost-model.json> ...` |

- `--mode smoke` runs one calibration repetition and a one-block plan. Use it to check a new setup.
- `x34_analysis.py --root RUN/SWEEP` and `supp_analysis.py --root RUN/X --experiment X --stage X-full` can be re-run offline on finished runs.

## Notes on interpretation

- **Statistics**: all comparisons are block-paired. The relative improvement is (baseline − method) / baseline on mean completion time, with a 10,000-draw bootstrap interval. Labels are `advantage` / `disadvantage` when both the median and the interval clear ±5 %, and `tied_or_unresolved` otherwise. With 8 blocks per cell the results are descriptive; no tail-latency or coverage claims are made.
- **Holder-departure (DEP) limitation**: the runtime does not tear a holder down mid-request. Deadlines are enforced at planning time, and violations are measured from the recorded holder-release times.

## Scope of this release

**Included**: everything needed to rerun the multi-holder experiments reported in the paper:

- calibration;
- the least_loaded baseline;
- rate, RTT and holder-action sweeps;
- compute throttling;
- heterogeneous holders;
- a second model;
- holder-departure deadlines;
- prediction accuracy.

**Not included**:

- the earlier single-holder E1–E4 campaign drivers;
- raw measurement data (per-trial reports and model weights).

Experiment outputs are written to your own run directories.

## Citation

If you use this code, please cite the paper; see also `CITATION.cff`:

```bibtex
@article{wang2026restate,
  title   = {ReState: Contention-Aware Decisions on Remote KV State for Edge--Cloud LLM Serving},
  author  = {Wang, Wenbin and Wang, Runan and Liu, Jiangchuan and Hu, Xiping},
  year    = {2026}
}
```

## License

Released under the MIT License; see `LICENSE`.

## Contact

Please open a GitHub issue for questions or problems with the artifact.
