"""Generate a deterministic E1 v2 trial PLAN, never run or certify experiments.

Standard library only: no torch/tokenizer/model imports, network, or submission.
The only output file is the explicitly supplied --output path; existing files
are refused.  --self-check generates and validates plans entirely in memory.

Without --token-bank, inputs are frozen tokenization/window recipes.  Their
recipe hashes are NOT claimed to be actual token hashes: token_ids_sha256 stays
null until materialization and verification by a future runner.  An optional
already-tokenized JSON bank can supply exact IDs without loading a tokenizer:

    {"model": "Qwen/Qwen2.5-14B", "revision": "<pinned revision>",
     "add_special_tokens": false, "tokenizer_identity": "<provenance>",
     "token_ids": [<nonnegative integer IDs, at least 1040> ...]}

The planner validates declared model/revision and hashes the bank, but cannot
verify tokenizer provenance or IDs against the actual model vocabulary.  Those
remain explicit unevaluated execution gates.

Examples:
    python -B e1_trial_plan.py --self-check
    python -B e1_trial_plan.py --seed 20261011 --output /explicit/path/plan.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path


DEFAULT_SEED = 20261011
SUPPORTED_PROTOCOL_CANONICAL_SHA256 = "b332f92491ff19acff028b158b18472a6be982c2e74ea9ab988b1a3c5aaf4c6f"
ACTIONS = ("recompute", "native_fetch", "holder", "remote_attention", "fixed_progressive_repair")
NATIVE_CONTROL = "native_hf_same_shape_local_control"
TEXT_UNIT = (
    "A distributed inference server stores the attention keys and values of a document. "
    "A later request reuses the document and adds a new question. "
    "The system transfers the cache to another server before continuing generation. "
)
TEXT_REPEATS = 128
PLANNING_SOURCE_NAMES = ("e1_trial_plan.py", "e1_transport.py", "e1_wire.py",
                         "e0_layered_runtime.py", "e0_cache_utils.py")
E0_PROTOCOL_NAMES = ("e0_precision_protocol.json", "e0_native_precision_protocol.json",
                     "e0_holder_precision_protocol.json")


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def identity_hash(value) -> str:
    return sha256_bytes(canonical_bytes(value))


def validate_protocol(protocol: dict) -> None:
    """Fail closed on any semantic change to the reviewed v2 protocol.

    Formatting-only JSON changes are permitted and get their own raw file hash.
    A semantic revision needs source review and an explicit new supported hash.
    """
    if identity_hash(protocol) != SUPPORTED_PROTOCOL_CANONICAL_SHA256:
        raise ValueError("Unsupported E1 protocol change; review it and update the pinned planner before use")
    if protocol["schema_version"] != 2 or protocol["stage"] != "E1":
        raise ValueError("Only the reviewed E1 v2 protocol is supported")
    grid = protocol["grid"]
    expected = {
        "prefix_tokens": [128, 1024], "application_cap_mbps_decimal": [100, 1000],
        "generated_tokens": [16, 64], "cell_count": 8,
        "warmups_per_action_per_cell": 1, "measured_repetitions_per_action_per_cell": 3,
        "total_warmup_trials": 40, "total_measured_trials": 120,
    }
    if any(grid.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported grid or repetition counts")
    semantics = protocol["common_request_semantics"]
    if (semantics["suffix_tokens"] != 16 or semantics["batch_size"] != 1
            or semantics["concurrent_foreground_requests"] != 1):
        raise ValueError("Unsupported request shape/concurrency")
    if tuple(action["id"] for action in protocol["actions"]) != ACTIONS:
        raise ValueError("The five actions or their canonical identities changed")
    if not semantics.get("native_engine_control"):
        raise ValueError("The separate native-HF control contract is missing")


def validate_token_bank(bank: dict | None, protocol: dict) -> None:
    if bank is None:
        return
    if not isinstance(bank, dict):
        raise ValueError("Token bank must be a JSON object")
    platform = protocol["platform"]
    if bank.get("model") != platform["model"] or bank.get("revision") != platform["revision"]:
        raise ValueError("Token bank model/revision differs from the pinned protocol")
    if bank.get("add_special_tokens") is not False:
        raise ValueError("Token bank must declare add_special_tokens=false")
    if not isinstance(bank.get("tokenizer_identity"), str) or not bank["tokenizer_identity"]:
        raise ValueError("Token bank must declare tokenizer_identity provenance")
    ids = bank.get("token_ids")
    if (not isinstance(ids, list) or len(ids) < 1040
            or any(type(token) is not int or not 0 <= token <= 2**31 - 1 for token in ids)):
        raise ValueError("Token bank needs at least 1040 nonnegative integer IDs fitting signed int32")


def _input_record(protocol, seed, prefix, phase, repetition, bank):
    # Matching inputs across cap and G isolates those grid factors, in addition
    # to mandatory pairing across all five actions within each cell/repetition.
    seed_spec = {"domain": "e1-input-v1", "global_seed": seed, "prefix_tokens": prefix,
                 "phase": phase, "repetition": repetition}
    input_seed = int(identity_hash(seed_spec)[:8], 16)
    recipe = {
        "schema": "e1-input-recipe-v1",
        "input_seed": input_seed,
        "model": protocol["platform"]["model"],
        "revision": protocol["platform"]["revision"],
        "prefix_tokens": prefix,
        "suffix_tokens": 16,
        "window_tokens": prefix + 16,
        "window_offset_rule": "input_seed modulo (bank_token_count - window_tokens + 1)",
        "add_special_tokens": False,
    }
    record = {"input_seed": input_seed, "input_seed_spec": seed_spec,
              "token_ids": None, "token_ids_sha256": None,
              "prefix_ids_sha256": None, "suffix_ids_sha256": None,
              "token_identity_status": "unmaterialized_recipe_only",
              "actual_vocabulary_validation": "not_evaluated"}
    if bank is None:
        text = TEXT_UNIT * TEXT_REPEATS
        recipe.update(kind="tokenize_frozen_text_then_select_window",
                      text_unit=TEXT_UNIT, text_repeats=TEXT_REPEATS,
                      input_text_utf8_sha256=sha256_bytes(text.encode("utf-8")),
                      tokenizer="pinned model revision; runtime must verify tokenizer files and length")
    else:
        recipe.update(kind="select_window_from_supplied_token_bank",
                      token_bank_canonical_sha256=identity_hash(bank),
                      declared_tokenizer_identity=bank["tokenizer_identity"])
        offset = input_seed % (len(bank["token_ids"]) - prefix - 16 + 1)
        tokens = list(bank["token_ids"][offset:offset + prefix + 16])
        record.update(token_ids=tokens, token_ids_sha256=identity_hash(tokens),
                      prefix_ids_sha256=identity_hash(tokens[:prefix]),
                      suffix_ids_sha256=identity_hash(tokens[prefix:]),
                      selected_bank_offset=offset,
                      token_identity_status="exact_ids_from_supplied_bank_provenance_unverified")
    recipe_hash = identity_hash(recipe)
    record.update(input_id="input-" + identity_hash({"seed_spec": seed_spec, "recipe": recipe})[:24],
                  input_recipe=recipe, input_recipe_sha256=recipe_hash,
                  tokens_identity={"kind": ("token_ids_sha256" if bank is not None else "input_recipe_sha256"),
                                   "sha256": record["token_ids_sha256"] or recipe_hash})
    return record


def _randomized_order(seed: int, pair_id: str) -> tuple[list[str], str]:
    order_spec = {"domain": "e1-action-order-v1", "global_seed": seed, "pair_id": pair_id}
    order_seed_sha256 = identity_hash(order_spec)
    # Hash-sort yields a deterministic pseudo-random permutation without relying
    # on version-specific random.shuffle implementations. Action ID breaks ties.
    order = sorted(ACTIONS, key=lambda action: (identity_hash({"order_seed_sha256": order_seed_sha256,
                                                             "action": action}), action))
    return order, order_seed_sha256


def build_plan(protocol: dict, provenance: dict, *, seed: int = DEFAULT_SEED,
               token_bank: dict | None = None) -> dict:
    """Pure manifest construction; never consult progress files or E0 results."""
    validate_protocol(protocol)
    validate_token_bank(token_bank, protocol)
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    grid = protocol["grid"]
    input_records, cells, main_trials, control_trials = {}, [], [], []
    for prefix, cap, generated in itertools.product(grid["prefix_tokens"],
            grid["application_cap_mbps_decimal"], grid["generated_tokens"]):
        cell_id = f"p{prefix}-s16-g{generated}-cap{cap}mbps"
        cell = {"cell_id": cell_id, "prefix_tokens": prefix, "suffix_tokens": 16,
                "generated_tokens": generated, "application_cap_mbps_decimal": cap,
                "rate_bits_per_second_per_direction": cap * 1_000_000,
                "groups": {"warmup": [], "measured": []}}
        for phase, repetitions in (("warmup", 1), ("measured", 3)):
            for repetition in range(repetitions):
                pair_id = f"{cell_id}/{phase}/{repetition}"
                record = _input_record(protocol, seed, prefix, phase, repetition, token_bank)
                input_records.setdefault(record["input_id"], record)
                order, order_seed = _randomized_order(seed, pair_id)
                group = {"pair_id": pair_id, "phase": phase, "repetition": repetition,
                         "input_id": record["input_id"], "input_seed": record["input_seed"],
                         "tokens_identity": record["tokens_identity"],
                         "action_order_seed_sha256": order_seed, "action_order": order,
                         "trial_ids_in_order": []}
                for order_index, action in enumerate(order):
                    trial = {"trial_id": f"{pair_id}/{action}", "main_sequence_index": len(main_trials),
                             "cell_id": cell_id, "pair_id": pair_id, "phase": phase,
                             "repetition": repetition, "action": action,
                             "action_order_index": order_index, "input_id": record["input_id"],
                             "input_seed": record["input_seed"], "tokens_identity": record["tokens_identity"],
                             "prefix_tokens": prefix, "suffix_tokens": 16, "generated_tokens": generated,
                             "rate_bits_per_second_per_direction": cap * 1_000_000,
                             "execution_status": "planning_only", "e0_gate_status": "not_evaluated"}
                    main_trials.append(trial)
                    group["trial_ids_in_order"].append(trial["trial_id"])
                control_id = f"control/{pair_id}/{NATIVE_CONTROL}"
                control_trials.append({
                    "trial_id": control_id, "control_sequence_index": len(control_trials),
                    "control": NATIVE_CONTROL, "cell_id": cell_id, "pair_id": pair_id,
                    "phase": phase, "repetition": repetition, "input_id": record["input_id"],
                    "input_seed": record["input_seed"], "tokens_identity": record["tokens_identity"],
                    "prefix_tokens": prefix, "suffix_tokens": 16, "generated_tokens": generated,
                    "counted_among_five_actions": False,
                    "paired_recompute_trial_id": f"{pair_id}/recompute",
                    "measurement_scope": "Separate same-shape local native-HF engine control; match full-prefix construction and local timing/staging boundaries before comparing custom-engine overhead.",
                    "cap_is_inactive_for_local_compute": True,
                    "execution_status": "planning_only", "e0_gate_status": "not_evaluated",
                })
                group["separate_native_hf_control_trial_id"] = control_id
                cell["groups"][phase].append(group)
        cells.append(cell)
    plan = {
        "schema": "e1-trial-plan-v1",
        "status": "planning_only_not_an_execution_manifest_or_readiness_certificate",
        "execution_authorized_by_this_file": False,
        "protocol_schema_version": 2,
        "protocol_semantic_sha256": identity_hash(protocol),
        "provenance": copy.deepcopy(provenance),
        "platform": copy.deepcopy(protocol["platform"]),
        "randomization": {
            "global_seed": seed,
            "algorithm": "Sort the five canonical action IDs by SHA256(canonical JSON of order_seed_sha256 and action), then action ID for ties.",
            "cell_order": "Cartesian product: prefix_tokens, application_cap_mbps_decimal, generated_tokens, each in protocol order.",
            "group_order": "For each cell: its one warmup group, then its three measured groups; every group contains all five actions exactly once in its recorded randomized order.",
            "input_seed_scope": "Global seed, P, phase and repetition; cap and G are intentionally excluded so bandwidth/generation comparisons also share the same request IDs when P/phase/repetition match.",
            "timestamps": "No generation timestamp is inserted, so unchanged inputs/source hashes regenerate identical JSON bytes.",
        },
        "input_contract": {
            "token_hash_format": "SHA256 of UTF-8 canonical JSON integer list, sorted keys/no spaces/ensure_ascii=false; order is preserved for arrays.",
            "materialization": ("Exact IDs are supplied; tokenizer provenance and actual vocabulary bounds still require execution-time verification."
                                if token_bank is not None else "Recipe only; actual token IDs/hashes are null and must be materialized and bound before execution."),
            "input_recipe_hash_is_not_a_token_hash": True,
            "bank_or_tokenizer_mutation": "Any change requires a new input identity and fresh numerical validation for all paired actions.",
        },
        "gating": {
            "overall": "not_evaluated",
            "evidence_files_read": [],
            "requirements": [{"id": f"protocol-start-gate-{index + 1}", "requirement": requirement,
                              "status": "not_evaluated", "evidence": []}
                             for index, requirement in enumerate(protocol["start_gates"])],
            "additional_requirements": [
                "Materialize and verify exact request IDs and tokenizer identity; validate all paired shapes/actions with the unchanged published E0 numerical gates.",
                "Freeze an implemented common runner, all transitive execution dependencies, real repair schedule, timing boundaries, native-HF control boundaries, and validated transport accounting before submission.",
                "A plan, model/source hash, staged model, or job allocation never implies an E0 pass or authorizes execution.",
            ],
        },
        "counts": {"cells": 8, "actions": 5, "paired_groups": 32,
                   "main_warmup_trials": 40, "main_measured_trials": 120, "main_trials": 160,
                   "separate_native_hf_warmup_trials": 8, "separate_native_hf_measured_trials": 24,
                   "separate_native_hf_trials": 32, "total_planned_entries_including_controls": 192,
                   "unique_input_records": len(input_records)},
        "main_execution_order": [trial["trial_id"] for trial in main_trials],
        "native_hf_controls": {
            "separate_from_five_action_statistics": True,
            "order": "A separately warmed local control block in the listed cell/phase/repetition order; do not insert control work into a five-action measured interval.",
            "trial_ids_in_order": [trial["trial_id"] for trial in control_trials],
            "boundary_status": "Concrete local control implementation/timer must be frozen and validated before execution; this planner does not certify it.",
            "trials": control_trials,
        },
        "cells": cells,
        "inputs": [input_records[key] for key in sorted(input_records)],
        "main_trials": main_trials,
        "measurement_contract": copy.deepcopy(protocol["measurement"]),
        "stop_and_reporting_rules": list(protocol["stop_and_reporting_rules"]),
    }
    validate_plan(plan)
    return plan


def validate_plan(plan: dict) -> None:
    """Check counts, actual pairing and order; never evaluate experimental gates."""
    main = plan["main_trials"]
    controls = plan["native_hf_controls"]["trials"]
    inputs = {item["input_id"]: item for item in plan["inputs"]}
    if len(plan["cells"]) != 8 or len(main) != 160 or len(controls) != 32 or len(inputs) != 8:
        raise ValueError("Unexpected cell/trial/input count")
    if sum(t["phase"] == "warmup" for t in main) != 40 or sum(t["phase"] == "measured" for t in main) != 120:
        raise ValueError("Main warmup/measured counts differ from 40/120")
    if sum(t["phase"] == "warmup" for t in controls) != 8 or sum(t["phase"] == "measured" for t in controls) != 24:
        raise ValueError("Separate control counts differ from 8/24")
    all_ids = [trial["trial_id"] for trial in main + controls]
    if len(set(all_ids)) != 192:
        raise ValueError("Trial IDs are not unique")
    main_map = {t["trial_id"]: t for t in main}
    control_map = {t["trial_id"]: t for t in controls}
    flattened = []
    for cell in plan["cells"]:
        if len(cell["groups"]["warmup"]) != 1 or len(cell["groups"]["measured"]) != 3:
            raise ValueError("Each cell needs one warmup and three measured paired groups")
        for phase in ("warmup", "measured"):
            for group in cell["groups"][phase]:
                trials = [main_map[trial_id] for trial_id in group["trial_ids_in_order"]]
                if len(trials) != 5 or set(t["action"] for t in trials) != set(ACTIONS):
                    raise ValueError("A paired group does not contain each action exactly once")
                if [t["action"] for t in trials] != group["action_order"]:
                    raise ValueError("Paired action order is inconsistent")
                expected_order, expected_seed = _randomized_order(plan["randomization"]["global_seed"], group["pair_id"])
                if group["action_order"] != expected_order or group["action_order_seed_sha256"] != expected_seed:
                    raise ValueError("Paired action randomization is not reproducible")
                record = inputs[group["input_id"]]
                for index, trial in enumerate(trials):
                    if (trial["pair_id"] != group["pair_id"] or trial["phase"] != phase
                            or trial["action_order_index"] != index or trial["cell_id"] != cell["cell_id"]
                            or trial["repetition"] != group["repetition"]
                            or trial["input_id"] != record["input_id"]
                            or trial["input_seed"] != record["input_seed"]
                            or trial["tokens_identity"] != record["tokens_identity"]):
                        raise ValueError("Trial pairing/input identity mismatch")
                    for key in ("prefix_tokens", "suffix_tokens", "generated_tokens", "rate_bits_per_second_per_direction"):
                        if trial[key] != cell[key]:
                            raise ValueError("Trial shape/rate differs from its cell")
                control = control_map[group["separate_native_hf_control_trial_id"]]
                if (control["input_id"] != record["input_id"] or control["tokens_identity"] != record["tokens_identity"]
                        or control["counted_among_five_actions"] is not False
                        or control["paired_recompute_trial_id"] != f"{group['pair_id']}/recompute"):
                    raise ValueError("Native-HF control is not separately paired")
                flattened.extend(group["trial_ids_in_order"])
    if flattened != plan["main_execution_order"] or flattened != [t["trial_id"] for t in main]:
        raise ValueError("Main execution order is inconsistent")
    if any(t["main_sequence_index"] != index for index, t in enumerate(main)):
        raise ValueError("Main sequence indices are not contiguous")
    for record in inputs.values():
        if identity_hash(record["input_recipe"]) != record["input_recipe_sha256"]:
            raise ValueError("Input recipe identity mismatch")
        tokens = record["token_ids"]
        if tokens is None:
            if record["token_ids_sha256"] is not None or record["tokens_identity"]["kind"] != "input_recipe_sha256":
                raise ValueError("Unmaterialized recipe was misrepresented as actual token IDs")
        elif len(tokens) != record["input_recipe"]["window_tokens"] or identity_hash(tokens) != record["token_ids_sha256"]:
            raise ValueError("Materialized token count/hash is wrong")
    if plan["gating"]["overall"] != "not_evaluated" or plan["execution_authorized_by_this_file"] is not False:
        raise ValueError("A trial plan cannot certify readiness or authorize execution")
    if any(trial["e0_gate_status"] != "not_evaluated" or trial["execution_status"] != "planning_only" for trial in main + controls):
        raise ValueError("A planned trial was incorrectly marked as executed/ready")


def render_plan(plan: dict) -> bytes:
    return (json.dumps(plan, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _load_inputs(protocol_path: Path, token_bank_path: Path | None):
    raw = protocol_path.read_bytes()
    protocol = json.loads(raw.decode("utf-8"))
    validate_protocol(protocol)
    root = Path(__file__).resolve().parent
    sources = {name: sha256_bytes((root / name).read_bytes()) for name in PLANNING_SOURCE_NAMES}
    gates = {name: sha256_bytes((root / name).read_bytes()) for name in E0_PROTOCOL_NAMES}
    design_path = root / "e1_runner_design.json"
    provenance = {
        "protocol_filename": protocol_path.name, "protocol_file_sha256": sha256_bytes(raw),
        "planning_and_available_component_source_sha256": sources,
        "published_e0_protocol_sha256": gates,
        "runner_design_sha256": sha256_bytes(design_path.read_bytes()),
        "execution_runner_source_sha256": None,
        "source_scope": "Planner and currently available components only; not a complete frozen execution dependency manifest.",
    }
    bank = None
    if token_bank_path is not None:
        bank_raw = token_bank_path.read_bytes()
        bank = json.loads(bank_raw.decode("utf-8"))
        validate_token_bank(bank, protocol)
        provenance["supplied_token_bank"] = {"filename": token_bank_path.name,
                                             "file_sha256": sha256_bytes(bank_raw),
                                             "canonical_sha256": identity_hash(bank)}
    return protocol, provenance, bank


def self_check(protocol: dict, provenance: dict) -> dict:
    first = build_plan(protocol, provenance)
    second = build_plan(copy.deepcopy(protocol), copy.deepcopy(provenance))
    if render_plan(first) != render_plan(second):
        raise AssertionError("Deterministic regeneration failed")
    changed_seed = build_plan(protocol, provenance, seed=DEFAULT_SEED + 1)
    if first["main_execution_order"] == changed_seed["main_execution_order"]:
        raise AssertionError("Different seed did not affect randomized action ordering")
    for mutation in ("grid", "version", "precision", "gates"):
        altered = copy.deepcopy(protocol)
        if mutation == "grid":
            altered["grid"]["measured_repetitions_per_action_per_cell"] = 4
        elif mutation == "version":
            altered["schema_version"] = 3
        elif mutation == "precision":
            altered["platform"]["precision"] = "unsupported change"
        else:
            altered["start_gates"] = []
        try:
            build_plan(altered, provenance)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsupported protocol mutation accepted: {mutation}")
    corrupted = copy.deepcopy(first)
    corrupted["main_trials"][0]["input_seed"] += 1
    try:
        validate_plan(corrupted)
    except ValueError:
        pass
    else:
        raise AssertionError("Broken pairing was not detected")
    bank = {"model": protocol["platform"]["model"], "revision": protocol["platform"]["revision"],
            "add_special_tokens": False, "tokenizer_identity": "CPU-only synthetic self-check fixture, not model evidence",
            "token_ids": list(range(2048))}
    materialized = build_plan(protocol, provenance, token_bank=bank)
    if any(record["token_ids_sha256"] is None for record in materialized["inputs"]):
        raise AssertionError("Supplied token IDs were not hashed")
    if any(record["token_ids_sha256"] is not None for record in first["inputs"]):
        raise AssertionError("Recipe-only plan claims actual token hashes")
    if any(cell["rate_bits_per_second_per_direction"] not in (100_000_000, 1_000_000_000) for cell in first["cells"]):
        raise AssertionError("Decimal Mbps conversion is wrong")
    return {"passed": True, "checks": ["40 warmup / 120 measured main trials", "32 separate native-HF controls",
            "five-action pairing and exact randomized order", "identical regeneration bytes", "seed sensitivity",
            "unsupported protocol mutations rejected", "corrupted pairing rejected", "optional exact token-bank hashes",
            "unmaterialized recipe never presented as token hash", "decimal bits/second rates"],
            "files_written": 0, "experiments_started": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("e1_pilot_protocol.json"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--token-bank", type=Path)
    parser.add_argument("--output", type=Path, help="explicit new JSON path; its parent must exist")
    parser.add_argument("--self-check", action="store_true", help="CPU/in-memory checks; no output file required")
    args = parser.parse_args()
    if not args.self_check and args.output is None:
        parser.error("Specify --output explicitly, or use --self-check")
    protocol, provenance, bank = _load_inputs(args.protocol, args.token_bank)
    if args.self_check:
        print(json.dumps(self_check(protocol, provenance), sort_keys=True))
    if args.output is not None:
        plan = build_plan(protocol, provenance, seed=args.seed, token_bank=bank)
        encoded = render_plan(plan)
        with args.output.open("xb") as output:
            output.write(encoded)
        print(json.dumps({"output": str(args.output), "sha256": sha256_bytes(encoded),
                          "status": plan["status"], "counts": plan["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
