# MobiKV

**MobiKV: Replica-Assisted KV Cache Scheduling for Multi-Turn LLM Serving at the Mobile Edge**

[![Website](https://img.shields.io/badge/Website-MobiKV-blue)](https://wenbinwang12.github.io/MobiKV/)
![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

MobiKV is a research artifact for scheduling multi-turn LLM sessions across mobile-edge servers. The final system, MobiKV-DF + RPP, combines four ideas:

- separate the user's access edge from the executor;
- keep one secondary KV replica and refresh it during think time;
- repair missing KV using the measured transfer-to-recompute ratio `gamma`; and
- use replica-assisted parallel prefill (RPP) when a helper can compute the front of a long turn while the executor computes the back.

Decode-first background replay gives foreground decoding priority while a replica is refreshed. The simulator is trace driven and models GPU queues, KV stores, links, foreground prefill, decoding, background replication, mobility, failures, heterogeneous links, and the RPP pipeline. The real-GPU programs measure the cost model and replay planned job streams with layer-wise RPP and batched decoding.

## Snapshot

The numbers below come from the final evaluation campaign (2026-11-24); the aggregates are under [`docs/data`](docs/data/). Simulations use cost profiles measured on NVIDIA A800 80GB GPUs and 20 mobility seeds.

- **Cross-model simulation.** Eight models and seven policies at the full-speed point (one A800 per server, 10 Gb/s links): MobiKV has the lowest mean follow-up TTFT in 21 of the 24 TTFT cells, and its TPOT is within 7% of the best baseline in every cell.
- **Ratio sweep.** Qwen3-8B over 18 values of the transfer-to-recompute ratio: MobiKV's P90 follow-up TTFT is 0.55-0.80 of the best baseline at every point.
- **RPP on two GPUs.** 78 configurations of five models: the speedup reaches 82-94% of the bound in the 54 configurations with a measured link rate, and the final logits are bitwise identical to chunked prefill in all 510 splits.
- **Replay on three GPUs.** Qwen3-8B at the constrained point, seven policies, seeds 41-43: MobiKV's follow-up TTFT is 0.18, 2.24, and 7.61 s at P50, P90, and P99, against 0.30-0.40, 4.83-7.44, and 14.5-17.7 s for the four load-aware baselines.

The simulator, cost files, and protocol documents in this repository are those of the earlier design; the code that produced the final campaign has not been merged yet, so these numbers cannot be reproduced from this revision.

The project page shows the overview video, the ratio sweep, and the real-GPU results.

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
| `tests/` | CPU-only tests run in CI |
| `docs/` | Experiment protocol, reproducibility notes, data-release policy, and project page |

Raw traces and model weights are not bundled. The preprocessing scripts accept local copies of the public source data, and the data-release policy documents which derived summaries can be shared.

## Installation

The supported environment is Linux, Python 3.10, CUDA 12.4, PyTorch 2.6, and Transformers 5.16.1. For CPU-only simulation and preprocessing:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras: `pip install -e ".[gpu]"` for real-GPU measurement and replay, `".[wildchat]"` for WildChat preprocessing, and `".[dev]"` for the test and lint tools.

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

Run the test suite and the linter with:

```bash
pytest
ruff check . && ruff format --check .
```

The preset expands to the frozen paper configuration. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the full flags, inputs, and measurement protocol.

## Data release

The repository releases code, measured cost models, configuration manifests, figure sources, and compact aggregate summaries of our own experiments. MobiKV has no private dataset: every input trace is a public dataset that stays with its original provider. We do not redistribute raw traces, prompts, generated text, model weights, private host paths, or network identifiers. See [`docs/DATA_RELEASE.md`](docs/DATA_RELEASE.md) for the release policy.

## Third-party data and models

Download each dataset from its provider, follow its terms, and cite its authors. The licenses below are the ones stated by the providers at the time of writing; the provider's page is authoritative.

| Dataset | Use in MobiKV | Source | Terms | Reference |
| --- | --- | --- | --- | --- |
| GeoLife GPS Trajectories 1.3 | Main mobility trace | [Microsoft Research](https://www.microsoft.com/en-us/download/details.aspx?id=52367) | Terms in the user guide shipped with the download; the provider asks users to cite [1]-[3] | [1]-[3] |
| Mooncake conversation trace | Main request trace | [`FAST25-release/traces`](https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces) in kvcache-ai/Mooncake | Apache-2.0 (repository license) | [4] |
| Taxi Service Trajectory (ECML PKDD 2015), Porto | Mobility robustness | [UCI Machine Learning Repository, dataset 339](https://archive.ics.uci.edu/dataset/339) | CC BY 4.0 | [5] |
| WildChat-1M | Request-trace robustness | [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | ODC-BY | [6] |

Model weights are not part of this repository. Each model in [`configs/model_matrix.json`](configs/model_matrix.json) is governed by the license on its own model card. Some families, such as Llama and Gemma, require you to accept their terms before download. Obtain the weights from the official publisher and comply with those terms, including any restrictions on use and redistribution.

The project page stylesheet is adapted from an MIT-licensed project page; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Baselines

The six baselines are our adaptations of published policies to KV cache scheduling at the mobile edge. They are reimplemented in `mobikv/simulator.py`; no code from the original systems is included.

| Baseline | Adapted from |
| --- | --- |
| Follow-Me | Follow Me Cloud [7] |
| MDP | MDP-based service migration [8] |
| Lyapunov | Lyapunov-based dynamic service placement [9] |
| Preble | Preble prompt-aware scheduling [10] |
| Mooncake | The Mooncake KVCache-centric scheduler [4] |
| ESP | LoongServe elastic sequence parallelism [11] |

## References

1. Y. Zheng, L. Zhang, X. Xie, and W.-Y. Ma. Mining interesting locations and travel sequences from GPS trajectories. WWW, 2009.
2. Y. Zheng, Q. Li, Y. Chen, X. Xie, and W.-Y. Ma. Understanding mobility based on GPS data. UbiComp, 2008.
3. Y. Zheng, X. Xie, and W.-Y. Ma. GeoLife: A collaborative social networking service among user, location and trajectory. IEEE Data Engineering Bulletin, 33(2), 2010.
4. R. Qin, Z. Li, W. He, J. Cui, F. Ren, M. Zhang, Y. Wu, W. Zheng, and X. Xu. Mooncake: Trading more storage for less computation, a KVCache-centric architecture for serving LLM chatbot. USENIX FAST, 2025.
5. Taxi Service Trajectory, Prediction Challenge, ECML PKDD 2015. UCI Machine Learning Repository, dataset 339.
6. W. Zhao, X. Ren, J. Hessel, C. Cardie, Y. Choi, and Y. Deng. WildChat: 1M ChatGPT interaction logs in the wild. ICLR, 2024.
7. T. Taleb and A. Ksentini. Follow Me Cloud: Interworking federated clouds and distributed mobile networks. IEEE Network, 27(5), 2013.
8. S. Wang, R. Urgaonkar, M. Zafer, T. He, K. Chan, and K. K. Leung. Dynamic service migration in mobile edge computing based on Markov decision process. IEEE/ACM Transactions on Networking, 27(3), 2019.
9. T. Ouyang, Z. Zhou, and X. Chen. Follow me at the edge: Mobility-aware dynamic service placement for mobile edge computing. IEEE Journal on Selected Areas in Communications, 36(10), 2018.
10. V. Srivatsa, Z. He, R. Abhyankar, D. Li, and Y. Zhang. Preble: Efficient distributed prompt scheduling for LLM serving. ICLR, 2025.
11. B. Wu, S. Liu, Y. Zhong, P. Sun, X. Liu, and X. Jin. LoongServe: Efficiently serving long-context large language models with elastic sequence parallelism. ACM SOSP, 2024.

## Citation

The paper is not public yet. Until the arXiv entry is available, please cite the project as follows; the entry will be updated with the arXiv identifier.

```bibtex
@misc{wang2026mobikv,
  title  = {MobiKV: Replica-Assisted KV Cache Scheduling for Multi-Turn LLM Serving at the Mobile Edge},
  author = {Wang, Wenbin and Wang, Runan and Liu, Jiangchuan and Hu, Xiping},
  year   = {2026},
  note   = {Preprint forthcoming},
  url    = {https://github.com/WenbinWang12/MobiKV}
}
```


## License

MIT; see [`LICENSE`](LICENSE).
