# Data and citations

## Included in this repository

The release contains the final simulator and GPU source, measured A800 cost profiles, experiment launchers, analysis and plotting code, and the evaluation summaries under [`data/`](data/README.md). The aggregate files record model, policy, operating point, latency statistics, and sample counts. Their units and provenance are listed in the data directory.

The public request and mobility traces remain with their original providers. Download them separately and record the source version and a local checksum when running an experiment. Model weights, raw prompts, generated text, per-request replay logs, host names, network addresses, and machine-specific paths are not distributed with the aggregates. Small synthetic fixtures support CPU checks without these inputs.

The GPU source is included for reproduction on a suitable system. Its presence does not mean that the GPU experiments were rerun locally for this release. See [GPU reproduction](GPU_REPRODUCTION.md) and [Evaluation](EVALUATION.md).

## Third-party data and models

Download each dataset from its provider, follow its terms, and cite its authors. The licenses below are the ones stated by the providers at the time of writing; the provider's page is authoritative.

| Dataset | Use in MobiKV | Source | Terms | Reference |
| --- | --- | --- | --- | --- |
| GeoLife GPS Trajectories 1.3 | Main mobility trace | [Microsoft Research](https://www.microsoft.com/en-us/download/details.aspx?id=52367) | Terms in the user guide shipped with the download; the provider asks users to cite [1]-[3] | [1]-[3] |
| Mooncake conversation trace | Main request trace | [`FAST25-release/traces`](https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces) in kvcache-ai/Mooncake | Apache-2.0 (repository license) | [4] |
| Taxi Service Trajectory (ECML PKDD 2015), Porto | Mobility robustness | [UCI Machine Learning Repository, dataset 339](https://archive.ics.uci.edu/dataset/339) | CC BY 4.0 | [5] |

Model weights are not part of this repository. Each model in [`configs/model_matrix.json`](../configs/model_matrix.json) is governed by the license on its own model card. Some families, such as Llama and Gemma, require you to accept their terms before download. Obtain the weights from the official publisher and comply with those terms, including any restrictions on use and redistribution.

The project page stylesheet is adapted from an MIT-licensed project page; see [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Baselines

The six baselines are our adaptations of published policies to KV cache scheduling at the mobile edge. They are reimplemented in `mobikv/simulator.py`; no code from the original systems is included.

| Baseline | Adapted from |
| --- | --- |
| Follow-Me | Follow Me Cloud [6] |
| MDP | MDP-based service migration [7] |
| Lyapunov | Lyapunov-based dynamic service placement [8] |
| Preble | Preble prompt-aware scheduling [9] |
| Mooncake | The Mooncake KVCache-centric scheduler [4] |
| ESP | LoongServe elastic sequence parallelism [10] |

## References

1. Y. Zheng, L. Zhang, X. Xie, and W.-Y. Ma. Mining interesting locations and travel sequences from GPS trajectories. WWW, 2009.
2. Y. Zheng, Q. Li, Y. Chen, X. Xie, and W.-Y. Ma. Understanding mobility based on GPS data. UbiComp, 2008.
3. Y. Zheng, X. Xie, and W.-Y. Ma. GeoLife: A collaborative social networking service among user, location and trajectory. IEEE Data Engineering Bulletin, 33(2), 2010.
4. R. Qin, Z. Li, W. He, J. Cui, F. Ren, M. Zhang, Y. Wu, W. Zheng, and X. Xu. Mooncake: Trading more storage for less computation, a KVCache-centric architecture for serving LLM chatbot. USENIX FAST, 2025.
5. Taxi Service Trajectory, Prediction Challenge, ECML PKDD 2015. UCI Machine Learning Repository, dataset 339.
6. T. Taleb and A. Ksentini. Follow Me Cloud: Interworking federated clouds and distributed mobile networks. IEEE Network, 27(5), 2013.
7. S. Wang, R. Urgaonkar, M. Zafer, T. He, K. Chan, and K. K. Leung. Dynamic service migration in mobile edge computing based on Markov decision process. IEEE/ACM Transactions on Networking, 27(3), 2019.
8. T. Ouyang, Z. Zhou, and X. Chen. Follow me at the edge: Mobility-aware dynamic service placement for mobile edge computing. IEEE Journal on Selected Areas in Communications, 36(10), 2018.
9. V. Srivatsa, Z. He, R. Abhyankar, D. Li, and Y. Zhang. Preble: Efficient distributed prompt scheduling for LLM serving. ICLR, 2025.
10. B. Wu, S. Liu, Y. Zhong, P. Sun, X. Liu, and X. Jin. LoongServe: Efficiently serving long-context large language models with elastic sequence parallelism. ACM SOSP, 2024.
