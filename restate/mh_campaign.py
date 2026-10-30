"""E6/E7 target-side campaign driver: real execution, sampled correctness audit, records on /cephfs.

Every trial is executed once for measurement. A deterministic sample (all smoke
trials; about one in VALIDATE_EVERY full trials) is re-executed as a diagnostic
(same offered set, recorded layer routes for remote/repair) and every request in
it is compared with a same-shape HF reference under the frozen BF16 tolerance.
Unsampled trials carry no numerical claim beyond executing the validated code
path. Records are persisted after every trial; a session can resume after a
failure by skipping trials that already have a passed record.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

from e4_destination import digest, require

VALIDATE_EVERY = 6


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def json_safe(value):
    if type(value) is float and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(json_safe(value), f, indent=1, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sampled(trial):
    if trial["sample_role"] in ("functional_smoke", "calibration_smoke"):
        return True
    return int(digest(trial["trial_id"])[:8], 16) % VALIDATE_EVERY == 0


def request_metrics(item, arrival):
    tokens = item["token_delivery_s"]
    itls = [b - a for a, b in zip(tokens, tokens[1:])]
    publications = item.get("e4_layer_publications") or []
    segments = item.get("transfer_segments") or []
    migrated = sum(p["bytes"] for p in publications) if publications else sum(s.get("payload_bytes", 0) for s in segments)
    return dict(ttft_s=tokens[0] - arrival, completion_s=tokens[-1] - arrival,
                settled_s=item["request_settled_s"] - arrival, mean_itl_s=statistics.fmean(itls), max_itl_s=max(itls),
                p95_itl_s=sorted(itls)[max(0, math.ceil(.95 * len(itls)) - 1)], delivered_tokens=len(tokens),
                migrated_bytes=migrated, remote_calls=len(item.get("remote_calls") or []),
                prefill_s=item.get("prefill_duration_s"), outer_setup_s=item.get("outer_setup_elapsed_s"))


def normalize(trial, measured, validation):
    accepted = measured["accepted_s"]
    rows = []
    for spec in trial["requests"]:
        item = measured["requests"][spec["role"]]
        require(item.get("passed") is True, "Failed request cannot be normalized")
        metrics = request_metrics(item, item["arrival_s"])
        rows.append(dict(role=spec["role"], action=spec["action"], holder=spec["holder"], prefix_tokens=spec["prefix_tokens"],
                         arrival_offset_s=spec["arrival_offset_s"], token_ids_sha256=digest(item["token_ids"]),
                         **{k: spec.get(k) for k in ("prefix_location", "slo_ttft_s", "slo_mean_itl_s", "input_id")},
                         **metrics))
    group = measured["group"]
    holders = {h: dict(socket_totals=v.get("socket_totals"), rate_bps=trial["holders"][h]["scheduler"]["rate_bps"],
                       policy=trial["holders"][h]["scheduler"]["policy"], load=trial["holders"][h]["load_location"])
               for h, v in measured["holders"].items()}
    return dict(trial_id=trial["trial_id"], status="passed", stage=trial["stage"], cell_id=trial["cell_id"],
                method=trial["method"], block_id=trial["block_id"], sample_role=trial["sample_role"],
                requests=rows, holders=holders, destination_load=bool(trial.get("destination_load")),
                group_service_settled_s=group["group_service_settled_latency_s"],
                makespan_s=max(r["settled_s"] + r["arrival_offset_s"] for r in rows),
                validation=validation, measured_artifact_sha256=measured.get("artifact_sha256"))


class MHCampaign:
    def __init__(self, engine, tokenizer, holders, plan, directory, report, *, bindings=None):
        from mh_destination import MHDestination
        from campaign_runner import materialize
        from campaign_client import verify_outputs
        from e0_holder_decode import compare
        self.engine, self.tokenizer, self.plan = engine, tokenizer, plan
        self.directory, self.report = Path(directory), report
        self.materialize, self.verify_outputs, self.compare = materialize, verify_outputs, compare
        self.trials = copy.deepcopy(plan["trials"])
        self.destination = MHDestination(engine, holders, trials={t["trial_id"]: t for t in self.trials})
        self.queue = engine.compute_queue
        self.paths = dict(records=self.directory / "normalized-records.json", inputs=self.directory / "frozen-input-arrays.json",
                          calibration=self.directory / "background-calibration.json", progress=self.directory / "progress.json")
        self.records, self.inputs = [], {}
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def input_key(spec):
        return digest({k: spec[k] for k in ("input_id", "input_family_id", "seed", "prefix_tokens", "suffix_tokens")})

    def initialize(self):
        if self.paths["records"].exists():
            self.records = json.loads(self.paths["records"].read_text())["records"]
        for trial in self.trials:
            for spec in trial["requests"]:
                key = self.input_key(spec)
                if key in self.inputs:
                    continue
                ids = self.materialize(dict(input_id=spec["input_id"], seed=spec["seed"], prefix=spec["prefix_tokens"],
                                            suffix=spec["suffix_tokens"]), self.tokenizer)
                require(len(ids) == spec["prefix_tokens"] + spec["suffix_tokens"], "Frozen input geometry differs")
                self.inputs[key] = dict(token_ids=ids, token_ids_sha256=digest(ids))
        background = self.materialize(dict(input_id=self.plan["background_family"], prefix=1024, suffix=16), self.tokenizer)
        if not self.paths["inputs"].exists():
            save(self.paths["inputs"], dict(plan_sha256=self.plan["plan_sha256"], arrays=self.inputs,
                                            background_token_ids=background))
        calibration = self.destination.initialize(background[:1024], background[1024:])
        save(self.directory / ("background-calibration-%s.json" % self.report["session_id"]), calibration)

    def role_inputs(self, trial):
        out = {}
        for spec in trial["requests"]:
            ids = self.inputs[self.input_key(spec)]["token_ids"]
            out[spec["role"]] = dict(prefix_ids=ids[:spec["prefix_tokens"]], suffix_ids=ids[spec["prefix_tokens"]:])
        return out

    def run_once(self, trial, *, diagnostic=False, measured=None):
        name = "diagnostic" if diagnostic else "candidate"
        path = self.directory / "trials" / trial["trial_id"] / (name + "-" + self.report["session_id"] + ".json")
        raw = None
        try:
            replay = None
            if diagnostic:
                replay = {r: copy.deepcopy(row["cutovers"]) for r, row in measured["requests"].items()
                          if row.get("mh_action") in ("remote", "repair")}
            raw, logits = self.destination.run_trial(trial["trial_id"], self.role_inputs(trial), diagnostic=diagnostic,
                                                     replay_by_role=replay, timeout_s=self.plan["trial_timeout_s"])
            require(raw.get("passed") is True, "Actual MH execution failed")
            return raw, logits
        except BaseException as error:
            raw = copy.deepcopy(getattr(error, "report", raw or {}))
            raw.update(driver_error=repr(error), driver_traceback=traceback.format_exc(), passed=False)
            raise
        finally:
            if raw is not None:
                raw.update(trial_id=trial["trial_id"], diagnostic=diagnostic, plan_sha256=self.plan["plan_sha256"],
                           session_id=self.report["session_id"], artifact=str(path))
                save(path, raw)
                raw["artifact_sha256"] = sha_bytes(path.read_bytes())

    def validate(self, trial, measured):
        diagnostic, logits = self.run_once(trial, diagnostic=True, measured=measured)
        torch = self.engine.torch
        inputs = self.role_inputs(trial)
        checks = {}
        try:
            for spec in trial["requests"]:
                role = spec["role"]
                ids = inputs[role]
                row, main = diagnostic["requests"][role], measured["requests"][role]
                prefix = torch.tensor([ids["prefix_ids"]], dtype=torch.int64, device=self.engine.device)
                suffix = torch.tensor([ids["suffix_ids"]], dtype=torch.int64, device=self.engine.device)
                with self.queue.bind(trial["trial_id"] + ".HF." + role, deadline=time.monotonic() + 1200):
                    actual = self.engine.HF_reference(prefix, suffix, 64)
                check = self.verify_outputs(torch, actual, row, logits[role], main)
                require(actual["state_report"]["owned_cache_references_released"] is True, "HF cache ownership leaked")
                checks[role] = dict(passed=True, action=spec["action"], numerical_check=check,
                                    reference_token_ids_sha256=digest(actual["token_ids"]))
                del actual, prefix, suffix
            return dict(performed=True, passed=True, requests=checks,
                        diagnostic_artifact_sha256=diagnostic.get("artifact_sha256"))
        finally:
            for rows in logits.values():
                rows.clear()
            self.queue.clear_completed_events()

    def persist(self):
        save(self.paths["records"], dict(plan_sha256=self.plan["plan_sha256"], records=self.records))
        done = {r["trial_id"] for r in self.records if r["status"] == "passed"}
        self.report["progress"] = dict(expected_trials=len(self.trials), passed_trials=len(done),
                                       failed_records=sum(r["status"] != "passed" for r in self.records),
                                       updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        save(self.paths["progress"], dict(run_id=self.plan["run_id"], stage=self.plan["stage"], mode=self.plan["mode"],
                                          **self.report["progress"]))

    def run(self):
        try:
            self.initialize()
            done = {r["trial_id"] for r in self.records if r["status"] == "passed"}
            self.persist()
            for trial in self.trials:
                if trial["trial_id"] in done:
                    continue
                started = time.monotonic()
                try:
                    measured, logits = self.run_once(trial)
                    require(all(not v for v in logits.values()), "Main captured full-vocabulary logits")
                    validation = self.validate(trial, measured) if sampled(trial) else dict(performed=False)
                    record = normalize(trial, measured, validation)
                    record["ownership_elapsed_s"] = time.monotonic() - started
                    self.records.append(record)
                    self.persist()
                except BaseException as error:
                    self.records.append(dict(trial_id=trial["trial_id"], status="failed", error=repr(error),
                                             session_id=self.report["session_id"], cell_id=trial["cell_id"],
                                             method=trial["method"], block_id=trial["block_id"]))
                    self.persist()
                    raise
            self.report["campaign_passed"] = True
        finally:
            try:
                self.report["destination_close"] = self.destination.close()
            finally:
                if self.records:
                    self.persist()
