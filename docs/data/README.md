# Released aggregate data

This directory contains compact, sanitized summaries that support the paper tables and figures.

- `aggregate/table1_cross_model.csv` contains the simulator cross-model metrics.
- `aggregate/rho_sweep_qwen3_aggregate.csv` contains the ρ sweep used for the sensitivity figure.
- `aggregate/rpp_mechanism_matrix.csv` and `aggregate/rpp_g25_summary.csv` contain RPP mechanism summaries.
- `aggregate/online_replay_qwen3.csv` contains pooled and edge-shard summaries from the completed Qwen3 online replay.
- `manifests/online_replay_qwen3_manifest.json` records hardware, service-rate profile, aggregation, and privacy metadata.
- `schema/online_replay_summary.schema.json` defines the online replay CSV fields.

The raw replay JSONL and trace records remain outside the repository. They contain request-level operational details and are unnecessary for reproducing the reported aggregates. The online replay summary uses seconds for both TTFT and TPOT, with the unit encoded in each column name.
