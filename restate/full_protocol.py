"""Execution envelope for E1; an unfrozen planning artifact never launches work."""
import math

from plan import validate_plan, POLICY, FIXED_ACTIONS, CONTROL


SMOKE_CONDITIONS = (
    (128, 16, 1000, 0, "none"),
    (4096, 64, 100, 0, "none"),
    (1024, 16, 1000, 10, "none"),
    (1024, 16, 400, 0, "holder"),
    (1024, 16, 400, 0, "destination"),
)


def smoke_trials(protocol):
    """Fixed new-mechanism/max-geometry smoke; no measured outcomes select rows."""
    cells = {tuple(c[k] for k in ("prefix_tokens", "realized_stop_tokens",
                                  "application_cap_mbps_decimal",
                                  "added_experiment_path_round_trip_delay_ms",
                                  "background_load_location")): c["cell_id"] for c in protocol["cells"]}
    result = []
    for condition in SMOKE_CONDITIONS:
        methods = tuple(FIXED_ACTIONS) + (CONTROL,) + ((POLICY,) if condition in SMOKE_CONDITIONS[:2] else ())
        for method in methods:
            found = [t for t in protocol["trials"] if t["phase"] == "test"
                     and t["sample_role"] == "measured" and t["block_id"] == "b00"
                     and t["cell_id"] == cells[condition] and t["method"] == method]
            if len(found) != 1:
                raise ValueError("Smoke row is missing or duplicated")
            result.extend(found)
    return result


def validate_wrapper(wrapper):
    if wrapper.get("stage") != "E1" or wrapper.get("mode") not in ("e1_smoke", "e1_full"):
        raise ValueError("This runtime executes only E1 smoke or full E1")
    validate_plan(wrapper["full_protocol"])
    limits = wrapper["limits"]
    for key in ("trial_timeout_s", "stage_timeout_s"):
        value = limits.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Finite positive deadline required: " + key)
    if limits["trial_timeout_s"] > 1200 or limits["stage_timeout_s"] > 172800:
        raise ValueError("Deadline exceeds frozen maximum")
    if wrapper["mode"] == "e1_smoke" and wrapper.get("smoke_trials") != smoke_trials(wrapper["full_protocol"]):
        raise ValueError("Smoke coverage differs from prescribed new-mechanism cases")
    if wrapper.get("E0_acceptance", {}).get("E0_complete") is not True:
        raise ValueError("Explicit full E0 prerequisite required")
    if wrapper["mode"] == "e1_full" and wrapper.get("E1_smoke_acceptance", {}).get("passed") is not True:
        raise ValueError("Full E1 requires new-runtime smoke acceptance")
    return True
