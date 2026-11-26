#!/usr/bin/env python3
"""Plot the real-GPU figures of the project page from the released aggregates.

Writes ``bounds.{svg,png}`` (two-GPU RPP speedup against its bound) and
``tradeoff.{svg,png}`` (three-GPU replay, follow-up TTFT and TPOT of seven policies).
Requires matplotlib (``pip install -e ".[plot]"``).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK = "#293641"
GRID = "#E3E7EE"
STYLE = {
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
MODELS = [
    ("Qwen2.5-7B", "#0072B2", "o"),
    ("Qwen2.5-14B", "#D55E00", "s"),
    ("Qwen3-8B", "#009E73", "^"),
    ("Llama-3.1-8B", "#CC79A7", "D"),
    ("Mistral-7B-v0.3", "#E69F00", "p"),
]
POLICIES = [
    ("follow_me_cloud", "Follow-Me", "#4DBBD5"),
    ("wang_ton19_mdp", "MDP", "#E64B35"),
    ("ouyang_jsac18", "Lyapunov", "#00A087"),
    ("preble_iclr25", "Preble", "#CC79A7"),
    ("mooncake_fast25", "Mooncake", "#E6A01E"),
    ("loongserve_sosp24", "ESP", "#8491B4"),
    ("mobikv", "MobiKV", "#1F2A44"),
]
PERCENTILES = [("P50", "o"), ("P90", "s"), ("P99", "^")]


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plain(value: float, _pos: int) -> str:
    return f"{value:g}"


def tidy(ax) -> None:
    ax.grid(True, which="major", color=GRID, linewidth=0.7, zorder=0)
    ax.tick_params(length=3, width=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, out: Path) -> None:
    fig.savefig(out.with_suffix(".svg"), facecolor="white")
    fig.savefig(out.with_suffix(".png"), facecolor="white", dpi=200)
    plt.close(fig)
    print(f"wrote {out}.svg and {out}.png")


def plot_bound(cells: list[dict], out: Path, paper: bool = False) -> None:
    """Speedup of one-helper RPP on two GPUs against min(2, 1 + 1/rho).

    With ``paper``, the figure is written as a PDF at the width of one IEEE column with 8 pt text.
    """
    size, base = ((3.45, 2.35), 8.0) if paper else ((7.4, 4.5), 12.0)
    k = base / 12.0
    with plt.rc_context({"font.size": base, "axes.linewidth": 0.8 * k**0.5, "pdf.fonttype": 42}):
        _plot_bound(cells, out, paper, size, k)


def _plot_bound(cells: list[dict], out: Path, paper: bool, size: tuple, k: float) -> None:
    fig, ax = plt.subplots(figsize=size, constrained_layout=True)
    area = 30 * k**1.5
    xs = [0.12 * (13 / 0.12) ** (i / 400) for i in range(401)]
    ax.plot(xs, [min(2.0, 1 + 1 / x) for x in xs], color=INK, linewidth=1.3 * k**0.5, zorder=2)
    handles = [
        Line2D([], [], color=INK, linewidth=1.3 * k**0.5, label=r"bound $\min\{2,\,1+1/\rho\}$")
    ]
    for model, colour, marker in MODELS:
        rows = [r for r in cells if r["model"] == model]
        measured = [r for r in rows if r["verified"] == "1"]
        bounded = [r for r in rows if r["verified"] == "0"]
        ax.scatter(
            [float(r["rho"]) for r in measured],
            [float(r["speedup"]) for r in measured],
            s=area,
            marker=marker,
            color=colour,
            linewidths=0,
            zorder=4,
        )
        # The delivered rate is only bounded from below: rho lies between rho_min and rho_max.
        ax.hlines(
            [float(r["speedup"]) for r in bounded],
            [float(r["rho_min"]) for r in bounded],
            [float(r["rho_max"]) for r in bounded],
            color=colour,
            linewidth=0.9 * k**0.5,
            zorder=3,
        )
        ax.scatter(
            [float(r["rho_min"]) for r in bounded],
            [float(r["speedup"]) for r in bounded],
            s=area,
            marker=marker,
            facecolors="white",
            edgecolors=colour,
            linewidths=1.0 * k**0.5,
            zorder=4,
        )
        handles.append(
            Line2D(
                [],
                [],
                color=colour,
                marker=marker,
                markersize=6 * k**0.75,
                linestyle="",
                label=model,
            )
        )
    ax.set_xscale("log")
    ax.set_xlim(0.12, 13)
    ax.set_ylim(0.92, 2.08)
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 10])
    ax.xaxis.set_major_formatter(FuncFormatter(plain))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Transfer-to-compute ratio $\rho$ of the turn")
    ax.set_ylabel("Speedup over one GPU")
    ax.grid(True, which="major", color=GRID, linewidth=0.7 * k**0.5, zorder=0)
    ax.tick_params(length=3 * k, width=0.8 * k**0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # The region above the bound for rho > 1 holds no data, so the legend sits there.
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        borderaxespad=0.2,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="none",
        fontsize=10.5 * k,
        handlelength=1.6,
        labelspacing=0.35,
    )
    if paper:
        fig.savefig(out.with_suffix(".pdf"), facecolor="white")
        plt.close(fig)
        print(f"wrote {out}.pdf")
        return
    save(fig, out)


def plot_replay(pooled: list[dict], out: Path) -> None:
    """Pooled follow-up TTFT and TPOT of the seven replayed policies, with 95% intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.1), constrained_layout=True)
    panels = [("ttft", "Follow-up TTFT (s)"), ("tpot", "TPOT (ms)")]
    for ax, (metric, ylabel) in zip(axes, panels, strict=True):
        for i, (key, _label, colour) in enumerate(POLICIES):
            for k, (percentile, marker) in enumerate(PERCENTILES):
                (row,) = (
                    r
                    for r in pooled
                    if (r["policy"], r["metric"], r["percentile"]) == (key, metric, percentile)
                )
                value, lo, hi = (float(row[c]) for c in ("value", "ci95_lo", "ci95_hi"))
                ax.errorbar(
                    i + 0.24 * (k - 1),
                    value,
                    yerr=[[value - lo], [hi - value]],
                    fmt=marker,
                    color=colour,
                    markersize=6.5 if key == "mobikv" else 5.5,
                    elinewidth=1.0,
                    capsize=2.5,
                    zorder=3,
                )
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(plain))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(range(len(POLICIES)))
        ax.set_xticklabels([label for _, label, _ in POLICIES])
        ax.get_xticklabels()[-1].set_fontweight("bold")
        ax.set_xlim(-0.6, len(POLICIES) - 0.4)
        ax.set_ylabel(ylabel)
        tidy(ax)
        ax.grid(False, axis="x")
    handles = [
        Line2D([], [], color=INK, marker=marker, linestyle="", label=percentile)
        for percentile, marker in PERCENTILES
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=3, frameon=False, fontsize=11.5)
    save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "docs/data/aggregate")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/assets")
    parser.add_argument(
        "--paper-bound", type=Path, help="write only the RPP bound figure as a print-size PDF here"
    )
    args = parser.parse_args()
    plt.rcParams.update(STYLE)
    if args.paper_bound:
        plot_bound(read(args.data / "rpp_two_gpu_cells.csv"), args.paper_bound, paper=True)
        return
    plot_bound(read(args.data / "rpp_two_gpu_cells.csv"), args.out / "bounds")
    plot_replay(read(args.data / "replay_three_gpu_pooled.csv"), args.out / "tradeoff")


if __name__ == "__main__":
    main()
