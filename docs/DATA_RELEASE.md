# MobiKV data-release plan

The artifact separates executable code from data that carries licensing, privacy, or infrastructure information.

## Release in GitHub

- simulator and measurement source code;
- frozen configuration files and cost-model schemas;
- paper PDF, appendix, supplement, figures, and aggregation scripts;
- aggregate tables, edge-shard percentile summaries, and experiment manifests from the completed online replay campaign; and
- small synthetic fixtures used by the CPU smoke test.

## Link to original providers

GeoLife, Porto taxi, WildChat, and the Mooncake conversation trace keep their original provenance and terms. The repository provides preprocessing scripts and the expected schemas. A user downloads each source from its original provider and records a local source hash in the manifest.

## Hugging Face dataset

The recommended dataset release is `WenbinWang12/MobiKV-data`. The current GitHub artifact already contains the sanitized aggregate payload. A Hugging Face upload should be made after the paper revision is frozen so its dataset card, DOI metadata, and exact commit can be aligned. It should contain:

```text
aggregate/
  table1_cross_model.csv
  rho_sweep_qwen3_aggregate.csv
  rpp_mechanism_matrix.csv
  online_replay_qwen3.csv
manifests/
  release.json
  source_hashes.json
schema/
  request_trace.schema.json
  online_replay_summary.schema.json
```

The dataset should contain timing, percentile, seed, model, policy, ratio, hardware, and software metadata. It should exclude raw prompts, generated text, tokens, hostnames, IP addresses, credentials, local paths, and model weights. Raw replay JSON records should be reduced to aggregate rows before upload.

## Long-term archive

After the paper version and data card are final, the GitHub release and Hugging Face dataset can be mirrored to Zenodo for a DOI. The Git tag, paper checksum, dataset revision, and source manifests should be recorded together.
