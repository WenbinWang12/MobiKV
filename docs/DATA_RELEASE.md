# MobiKV data-release plan

The artifact separates executable code from data that carries licensing, privacy, or infrastructure information.

## Release in GitHub

- simulator and measurement source code;
- frozen configuration files and cost-model schemas;
- figure sources and aggregation scripts;
- aggregate tables of the final evaluation campaign (cross-model simulation, ratio sweep, two-GPU RPP, three-GPU replay) with a hash manifest; and
- small synthetic fixtures used by the CPU smoke test.

## Link to original providers

GeoLife, Porto taxi, WildChat, and the Mooncake conversation trace keep their original provenance and terms. Source links, licenses, and the references to cite are listed under "Third-party data and models" in the repository README. The repository provides preprocessing scripts and the expected schemas. A user downloads each source from its original provider and records a local source hash in the manifest.

## Hugging Face dataset

The recommended dataset release is `WenbinWang12/MobiKV-data`. The current GitHub artifact already contains the sanitized aggregate payload. A Hugging Face upload should be made after the paper revision is frozen so its dataset card, DOI metadata, and exact commit can be aligned. It should contain:

```text
aggregate/
  cross_model_full_speed.csv
  cross_model_constrained.csv
  cross_model_full_speed_slow_links.csv
  rho_sweep_qwen3_a800.csv
  rpp_two_gpu_cells.csv
  replay_three_gpu.csv
  replay_three_gpu_pooled.csv
manifests/
  source_hashes.json
```

The dataset should contain timing, percentile, seed, model, policy, ratio, hardware, and software metadata. It should exclude raw prompts, generated text, tokens, hostnames, IP addresses, credentials, local paths, and model weights. Raw replay JSON records should be reduced to aggregate rows before upload.

## Long-term archive

After the paper version and data card are final, the GitHub release and Hugging Face dataset can be mirrored to Zenodo for a DOI. The Git tag, paper checksum, dataset revision, and source manifests should be recorded together.
