"""Compute-throttled C3 run: MPS SM cap on all three GPUs, original two-holder X34 cells (2026-10-29).

MPS gate -> BASE calibration (116 trials, B* under the cap) -> dual-holder probe grid -> preregistered
rate selection (about 0.5/1.5/2.5 B*) -> SWEEP (x34_pipeline cal,x34 with its analysis) -> post-probe
-> MPS gate -> MPS stopped. BASE and SWEEP resume through their own pipelines; probes, the selection
and the bindings are immutable once written.
"""
from __future__ import annotations

import restate_config
import argparse
import math
from pathlib import Path
import subprocess
import sys

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
import mh_pipeline as pipeline  # noqa: E402
import mh_plan  # noqa: E402

PROBE_SRC = Path(restate_config.PROBE_SRC)
TARGET_IP = restate_config.TARGET_IP
HOLDER_IPS = restate_config.HOLDER_IPS
HOLDERS = [ip + ":0" for ip in HOLDER_IPS]
CANDIDATES = [250_000_000, 400_000_000, 500_000_000, 600_000_000, 750_000_000, 1_000_000_000, 1_250_000_000,
              1_500_000_000, 2_000_000_000, 2_500_000_000, 3_124_000_000, 3_600_000_000]
RAW_RATE, RAW_CHUNK, HEADROOM = 100_000_000_000, 4_194_304, 0.9


def on_host(ip, command, timeout=60):
    argv = ["bash", "-c", command] if ip == TARGET_IP else ["ssh", "-o", "BatchMode=yes", "root@" + ip, command]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def mps_state():
    out = {}
    for ip in [TARGET_IP] + HOLDER_IPS:
        r = on_host(ip, "pgrep -f '^nvidia-cuda-mps-control -d' >/dev/null || exit 3; "
                        "echo get_default_active_thread_percentage | timeout 10 nvidia-cuda-mps-control")
        lines = r.stdout.strip().splitlines()
        try:
            out[ip] = float(lines[-1]) if r.returncode == 0 and lines else None
        except ValueError:
            out[ip] = None
    return out


def mps_gate(root, stage, percentage):
    state = mps_state()
    pipeline.log(root, stage="mps", event=stage, state=state, expected=percentage)
    pipeline.require(all(v is not None and abs(v - percentage) < 1e-6 for v in state.values()),
                     "MPS cap not in force at %s: %s" % (stage, state))


def freeze(source, destination):
    raw = Path(source).read_bytes()
    if destination.exists():
        pipeline.require(destination.read_bytes() == raw, "Existing preregistration differs; refusing silent reuse")
    else:
        with destination.open("xb") as stream:
            stream.write(raw)
    return pipeline.sha(destination)


def probe_grid(root, name, rates):
    """Both holders sending at once (scaled chunks, 3 reps each) plus the raw ceiling; min over holders."""
    path = root / (name + ".json")
    if path.exists():
        return pipeline.read(path)
    if str(PROBE_SRC) not in sys.path:
        sys.path.insert(1, str(PROBE_SRC))
    import queued_network_probe as network
    rows = []
    for rate, chunk, reps in [(RAW_RATE, RAW_CHUNK, 2)] + [(r, mh_plan.scaled_chunk(r), 3) for r in rates]:
        out = network.probe(PKG, root / "network-probes" / ("%s_%d" % (name, rate)), rate, chunk, HOLDER_IPS, TARGET_IP,
                            reps=reps, seconds=6)
        pipeline.require(out.get("passed") is True, "Probe failed: %s %d: %s" % (name, rate, out.get("errors")))
        rows.append(dict(rate_bps=rate, chunk_bytes=chunk, reps=reps, effective_min_bps=out["global_min_bps"],
                         holder_min_bps=out["holder_min_bps"], summary_sha256=pipeline.sha(Path(out["summary_path"]))))
    result = dict(name=name, raw_min_bps=rows[0]["effective_min_bps"], rows=rows[1:])
    pipeline.immutable_json(path, result)
    pipeline.log(root, stage=name, event="measured", raw_min_bps=result["raw_min_bps"],
                 effective_bps={str(r["rate_bps"]): r["effective_min_bps"] for r in rows[1:]})
    return result


def select_rates(b_star, grid):
    """Preregistered rule: low point nearest 0.5 B* below B*, high point nearest 2.5 B* above B*, middle point
    nearest 1.5 B* strictly between B* and the high point; nominal caps within 0.9 x the raw dual minimum."""
    cands = [r for r in grid["rows"] if r["rate_bps"] <= HEADROOM * grid["raw_min_bps"]]

    def dist(row, ratio):
        return abs(math.log(row["effective_min_bps"] / b_star) - math.log(ratio))
    low = [r for r in cands if r["effective_min_bps"] < b_star]
    high = [r for r in cands if r["effective_min_bps"] > b_star]
    pipeline.require(low and high, "Probe grid does not straddle B*")
    lo, hi = min(low, key=lambda r: dist(r, 0.5)), min(high, key=lambda r: dist(r, 2.5))
    middle = [r for r in high if r["effective_min_bps"] < hi["effective_min_bps"]]
    pipeline.require(middle, "No candidate strictly between B* and the high point")
    mid = min(middle, key=lambda r: dist(r, 1.5))
    points = {"0.5": lo, "1.5": mid, "2.5": hi}
    return dict(b_star_bps=b_star, raw_min_bps=grid["raw_min_bps"], rates=[lo["rate_bps"], mid["rate_bps"], hi["rate_bps"]],
                effective_over_bstar={k: v["effective_min_bps"] / b_star for k, v in points.items()},
                nominal_bps={k: v["rate_bps"] for k, v in points.items()}, scientific_acceptance=False)


def run(root, name, argv):
    with (root / (name + "-command.log")).open("ab") as stream:
        code = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT).returncode
    pipeline.log(root, stage=name, event="command_exit", exit=code)
    pipeline.require(code == 0, name + " failed with exit %d" % code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--mps-percentage", type=float, required=True)
    ap.add_argument("--prereg", type=Path, required=True)
    ap.add_argument("--throttle-selection", type=Path, required=True)
    args = ap.parse_args()
    root, p = args.root.resolve(), args.mps_percentage
    root.mkdir(parents=True, exist_ok=True)
    with pipeline.pipeline_lock(root):
        prereg = root / "preregistration.md"
        prereg_sha = freeze(args.prereg, prereg)
        throttle = pipeline.read(args.throttle_selection)
        pipeline.require(throttle["mps_percentage"] == p, "Requested cap differs from the frozen selection")
        pipeline.immutable_json(root / "frozen-input-bindings.json", dict(
            preregistration_sha256=prereg_sha, mps_percentage=p, throttle_selection_sha256=pipeline.sha(args.throttle_selection),
            source_sha256=pipeline.source_closure(), holders=HOLDERS, target_ip=TARGET_IP, scientific_acceptance=False))
        mps_gate(root, "before_base", p)
        run(root, "base", [sys.executable, "-B", str(PKG / "mh_pipeline.py"), "--root", str(root / "BASE"),
                           "--holders", *HOLDERS, "--target-gpu", "0", "--stages", "cal"])
        base = pipeline.read(root / "BASE/cost-model.json")
        pipeline.validate_hash(base, "model_sha256")
        grid = probe_grid(root, "probe-grid", CANDIDATES)
        path = root / "selected-rates.json"
        if not path.exists():
            pipeline.immutable_json(path, select_rates(base["b_star_bps"], grid))
        selection = pipeline.read(path)
        pipeline.log(root, stage="rate_selection", event="frozen", **{k: selection[k] for k in
                     ("b_star_bps", "rates", "effective_over_bstar")})
        rates = [str(r) for r in selection["rates"]]
        mps_gate(root, "before_sweep", p)
        run(root, "sweep", [sys.executable, "-B", str(PKG / "x34_pipeline.py"), "--root", str(root / "SWEEP"),
                            "--holders", *HOLDERS, "--target-gpu", "0", "--base-model", str(root / "BASE/cost-model.json"),
                            "--rates", *rates, "--x5-rate", rates[0], "--stages", "cal,x34", "--prereg", str(prereg)])
        probe_grid(root, "post-probe", [int(r) for r in rates])
        mps_gate(root, "after_sweep", p)
        stopped = {ip: on_host(ip, "echo quit | timeout 20 nvidia-cuda-mps-control").returncode
                   for ip in [TARGET_IP] + HOLDER_IPS}
        pipeline.log(root, stage="mps", event="stopped", exit_codes=stopped, state=mps_state())
        pipeline.log(root, stage="pipeline", event="complete", scientific_acceptance=False)


if __name__ == "__main__":
    main()
