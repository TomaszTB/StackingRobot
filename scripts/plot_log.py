"""Plot a run_csv.py --log file: each joint's expected angle (line) against
its measured reads (dots), one panel per joint, sharing the time axis.

Saves <log>.png next to the log (or <log>_rows<A>-<B>.png with --rows).

Usage:
    uv run --with matplotlib python scripts/plot_log.py logs/run1.csv
    uv run --with matplotlib python scripts/plot_log.py logs/run1.csv --rows 2-9
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXPECTED_COLOR = "#2a78d6"
MEASURED_COLOR = "#eb6834"
INK = "#2b2b29"
MUTED = "#6f6f6b"
GRID = "#e4e4e0"
SURFACE = "#fcfcfb"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    parser.add_argument("--rows", help="Only these CSV rows, e.g. 2-9.")
    args = parser.parse_args()

    with open(args.log) as f:
        reads = list(csv.DictReader(f))
    out = args.log.with_suffix(".png")
    if args.rows:
        first, _, last = args.rows.partition("-")
        first, last = int(first), int(last or first)
        reads = [r for r in reads if first <= int(r["row"]) <= last]
        out = args.log.with_name(f"{args.log.stem}_rows{first}-{last}.png")
    if not reads:
        parser.error("no reads to plot")

    # Base to wrist, whatever order the selected rows happen to start in.
    order = ["base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll"]
    logged = {r["joint"] for r in reads}
    joints = [j for j in order if j in logged] + sorted(logged - set(order))
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(len(joints), 1, figsize=(12, 2 * len(joints)), sharex=True, facecolor=SURFACE)
    for ax, joint in zip(axes, joints):
        rows = [r for r in reads if r["joint"] == joint]
        t = [float(r["time_s"]) for r in rows]
        ax.plot(t, [float(r["expected_deg"]) for r in rows], color=EXPECTED_COLOR, linewidth=2, label="expected")
        ok = [r for r in rows if r["measured_deg"]]
        ax.scatter(
            [float(r["time_s"]) for r in ok], [float(r["measured_deg"]) for r in ok],
            s=9, color=MEASURED_COLOR, edgecolors=SURFACE, linewidths=0.5, zorder=3, label="measured",
        )
        failed = [float(r["time_s"]) for r in rows if not r["measured_deg"]]
        for x in failed:
            ax.axvline(x, color=MUTED, linewidth=0.5, linestyle=":")
        ax.set_ylabel(f"{joint}\n(deg)")
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[0].legend(loc="upper right", frameon=False, ncol=2)
    axes[-1].set_xlabel("time since start (s)")
    title = f"{args.log.name}: expected vs measured joint angles"
    fig.suptitle(title + (f" (rows {args.rows})" if args.rows else ""), x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
