#!/usr/bin/env python3
"""Plot follow-up TTFT against the transfer-to-compute ratio for the project page.

Reads the released aggregate (per policy and ratio: the mean over mobility seeds and
the mean minus/plus one sample standard deviation) and writes ``rho-sweep.svg`` and ``rho-sweep.png``. Requires matplotlib
(``pip install -e ".[plot]"``).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# (key in the CSV, label, colour); MobiKV is drawn last so it stays on top.
POLICIES = [
    ("follow_me", "Follow-Me", "#4DBBD5"),
    ("mdp", "MDP", "#E64B35"),
    ("lyapunov", "Lyapunov", "#00A087"),
    ("preble", "Preble", "#CC79A7"),
    ("mooncake", "Mooncake", "#E6A01E"),
    ("esp", "ESP", "#8491B4"),
    ("mobikv", "MobiKV", "#1F2A44"),
]
METRICS = [("p50", "P50"), ("p90", "P90"), ("p99", "P99")]
INK = "#293641"


def load(path: Path) -> dict:
    """Return {(policy, metric): [(rho, mean, lower, upper), ...]} sorted by rho."""
    rows = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            for metric, _ in METRICS:
                rows[r["method"], metric].append(
                    tuple(float(r[c]) for c in ("rho", metric, f"{metric}_lo", f"{metric}_hi"))
                )
    return {k: sorted(v) for k, v in rows.items()}


def plain(value: float, _pos: int) -> str:
    return f"{value:g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data", type=Path, default=ROOT / "docs/data/aggregate/rho_sweep_qwen3_a800.csv"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "docs/assets/rho-sweep")
    args = parser.parse_args()
    data = load(args.data)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.edgecolor": "#AEB6C2",
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "svg.fonttype": "path",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.1), constrained_layout=True)
    handles = []
    for ax, (metric, title) in zip(axes, METRICS, strict=True):
        lo, hi = float("inf"), 0.0
        for key, label, colour in POLICIES:
            pts = data[key, metric]
            x = [p[0] for p in pts]
            mean = [p[1] for p in pts]
            lower = [p[2] for p in pts]
            upper = [p[3] for p in pts]
            own = key == "mobikv"
            ax.fill_between(x, lower, upper, color=colour, alpha=0.22, linewidth=0, zorder=2)
            (line,) = ax.plot(
                x,
                mean,
                color=colour,
                linewidth=1.7 if own else 1.1,
                zorder=4 if own else 3,
                label=label,
                solid_capstyle="round",
            )
            if ax is axes[0]:
                handles.append(line)
            lo, hi = min(lo, *lower), max(hi, *upper)
        ax.set_yscale("log")
        ax.set_ylim(lo / 1.2, hi * 1.2)
        ax.set_xlim(0, 10)
        ax.set_xticks(range(0, 11, 2))
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(plain))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", color="#E3E7EE", linewidth=0.7, zorder=0)
        ax.axvline(1, color="#8B95A3", linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(
            r"$\rho_{8k}=1$",
            xy=(1, 1),
            xycoords=("data", "axes fraction"),
            xytext=(5, -13),
            textcoords="offset points",
            fontsize=10.5,
            color="#6B7685",
        )
        ax.set_title(title, fontsize=13.5, fontweight="bold", pad=8)
        ax.set_xlabel(r"Transfer-to-compute ratio $\rho_{8k}$")
        ax.tick_params(length=3, width=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Follow-up TTFT (s)")
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncols=len(POLICIES),
        frameon=False,
        fontsize=11.5,
        handlelength=1.8,
        columnspacing=1.8,
    )
    fig.savefig(args.out.with_suffix(".svg"), facecolor="white")
    fig.savefig(args.out.with_suffix(".png"), facecolor="white", dpi=200)
    print(f"wrote {args.out}.svg and {args.out}.png")


if __name__ == "__main__":
    main()
