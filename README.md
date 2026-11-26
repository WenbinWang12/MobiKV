# MobiKV

**MobiKV: Replica-Assisted KV Cache Scheduling for Multi-Turn LLM Serving at the Mobile Edge**

[Project page](https://wenbinwang12.github.io/MobiKV/) | [Evaluation](docs/EVALUATION.md) | [Released data](docs/data/README.md) | [MIT license](LICENSE)

MobiKV schedules multi-turn LLM sessions across mobile-edge servers. It considers every server as an executor and accounts for cache repair, GPU queues, and the stalls that new work would cause to decoding. It keeps one nearby replica as another place to run the next turn and as a helper for long prefills.

The helper computes the front of the new input and streams its KV layer by layer while the executor computes the back. A cold helper can also participate after receiving the missing prefix. Between turns, the executor refreshes the replica with KV transfers on idle links; refresh uses no GPU. The controller weighs decode stalls with `omega=0.2` and uses stall debt to discourage repeated splits onto the same helper.

## Results

The [final evaluation aggregates](docs/data/README.md) cover eight models, seven policies, a link-rate sweep, and real-GPU measurements.

- Across six standard-attention models at the full-speed point, MobiKV lowers P90 TTFT by **27–42%** relative to the best baseline of each model. This comparison uses one A800 per server, 10 Gb/s links, and mean paired ratios over 20 mobility seeds.
- In the three-GPU Qwen3-8B replay at a 25% GPU share and 0.93 Gb/s, P90 TTFT falls from **4.83 s to 2.24 s**, a 54% reduction. Median TPOT is 33.9 ms versus 33.6 ms for the best baseline.
- One-helper RPP speeds up prefill by **1.65–1.89×** in 21 configurations across five models with at least 4k new tokens and a transfer-to-compute ratio known to be below 1. All 510 tested splits produce the same logits as chunked prefill at the same boundary.

See [Evaluation](docs/EVALUATION.md) for the workload, uncertainty estimates, and limits of these comparisons.

## Run

The repository includes the final simulator, measured A800 cost profiles, campaign launchers, GPU measurement and replay source, and analysis tools. The GPU programs have not been rerun locally during this release. Public input traces and model weights must be obtained from their providers; they are not bundled.

For CPU simulation, use Python 3.10 or later:

```bash
python -m pip install -e .
python examples/smoke_test.py
```

Run one policy with local copies of the request trace and prepared mobility data:

```bash
python scripts/run_mobikv.py \
  --method mobikv --model Qwen3-8B --op full \
  --trace /path/to/mooncake.jsonl --trips /path/to/geolife.npz \
  --seed 41 --out runs/qwen3_mobikv_seed41.json
```

The launcher uses `configs/costs_a800.json` and the final policy settings. The operating points are `full` (whole A800, 10 Gb/s), `slow` (quarter A800, 0.93 Gb/s), and `full1g` (whole A800, 0.93 Gb/s).

| Task | Instructions |
| --- | --- |
| Prepare inputs and run the simulator | [Reproducibility](docs/REPRODUCIBILITY.md) |
| Run the model comparison, sweep, stress tests, or ablations | [Experiment commands](docs/EXPERIMENTS.md) |
| Measure GPU costs, run RPP, or replay three servers | [GPU reproduction](docs/GPU_REPRODUCTION.md) |
| Read the released results and their units | [Data files](docs/data/README.md) |
| Obtain and cite third-party traces, models, and baseline papers | [Data and citations](docs/DATA_RELEASE.md) |

## Citation

The paper is not public yet. Until the preprint is available, cite the project:

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

The code is released under the [MIT license](LICENSE). Third-party data and model weights retain their own terms. The project-page stylesheet attribution is preserved in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
