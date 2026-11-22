# MobiKV

**MobiKV: Replica-Assisted KV Cache Scheduling for Multi-Turn LLM Serving at the Mobile Edge**

[Project page](https://wenbinwang12.github.io/MobiKV/) ? [Source code](https://github.com/WenbinWang12/MobiKV)

MobiKV is a research artifact for scheduling multi-turn LLM sessions across mobile-edge servers. The final system, MobiKV-DF + RPP, combines four ideas:

- separate the user's access edge from the executor;
- keep one secondary KV replica and refresh it during think time;
- repair missing KV using the measured transfer-to-compute ratio `rho`; and
- use replica-assisted parallel prefill (RPP) when a helper can compute the front of a long turn while the executor computes the back.

Decode-first background replay gives foreground decoding priority while a replica is refreshed. The simulator is trace driven and models GPU queues, KV stores, links, foreground prefill, decoding, background replication, mobility, failures, heterogeneous links, and the RPP pipeline. The real-GPU programs measure the cost model and replay planned job streams with layer-wise RPP and batched decoding.

## Snapshot

The revised-design simulator, A800 mechanism rerun, and online-replay aggregates are being finalized and will be added after validation.

The project page contains the overview, ratio sweep, cross-model table, real-GPU validation, and reproducibility links. Aggregate summaries and their manifests are available under [`docs/data`](docs/data/).

## Repository layout

| Path | Purpose |
| --- | --- |
| `mobikv/simulator.py` | Trace-driven simulator for the final MobiKV-DF/RPP configuration |
| `mobikv/measurement/ct_hybrid.py` | Full prefill, cached-prefix recompute, transfer, and hybrid repair |
| `mobikv/measurement/ct_pp.py` | One-helper RPP with layer-wise KV streaming |
| `mobikv/measurement/ct_pp2.py` | Two-helper chain used for the degree bound check |
| `mobikv/measurement/sv3_replay.py` | Real-GPU replay with decoding, RPP, and decode-first background work |
| `mobikv/preprocess/` | GeoLife, Porto, and WildChat preprocessing and cost fitting |
| `mobikv/analysis/` | Aggregation and analysis for simulator and replay outputs |
| `configs/` | Released cost models and the cross-model experiment matrix |
| `scripts/` | Reproduction launchers, aggregators, and model checks |
| `docs/` | Experiment protocol, reproducibility notes, data-release policy, and project page |

Raw traces and model weights are not bundled. The preprocessing scripts accept local copies of the public source data, and the data-release policy documents which derived summaries can be shared.

## Installation

The supported environment is Linux, Python 3.10, CUDA 12.4, PyTorch 2.6, and Transformers 5.16.1. For CPU-only simulation and preprocessing:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Real-GPU measurement and replay use NVIDIA A800 80GB PCIe GPUs, a shared local model snapshot, and password-less host-to-host connectivity. Set `MOBIKV_MODEL` to a local Hugging Face snapshot. Model weights stay outside this repository.

## Run the simulator

The simulator expects a JSONL request trace, an `.npz` mobility file, and a measured cost model.

```bash
python scripts/run_mobikv.py \
  --preset mobikv-df-rpp \
  --trace /path/to/mooncake.jsonl \
  --trips /path/to/geolife_test.npz \
  --costs configs/costs_v1.json \
  --out runs/default.json \
  --seed 31 --tpot 1
```

Run the bundled CPU smoke test with:

```bash
python examples/smoke_test.py
```

The preset expands to the frozen paper configuration. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the full flags, inputs, and measurement protocol.

## Data release

The repository releases code, measured cost models, configuration manifests, figure sources, and compact aggregate summaries. Public traces remain linked to their original providers. We do not redistribute raw prompts, generated text, model weights, private host paths, or network identifiers. See [`docs/DATA_RELEASE.md`](docs/DATA_RELEASE.md) for the Hugging Face dataset layout and sanitization rules.


## License

MIT; see [`LICENSE`](LICENSE).
