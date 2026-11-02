"""Analyze the block-pyramid trials: the results sheet (pass/fail, failure point)
joined with each run's logs/<material>_block_<mode>_loop_..._run<N>.csv log.

Progress comes from the sheet (runs were sometimes left going past a failure,
so where a log stops isn't the failure point); everything else from the logs.
Saves PDF figures and CSV tables to analysis/figures/.

Usage:
    uv run --with matplotlib python analysis/analyze_results.py
"""

import bisect
import csv
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "analysis" / "project_2_results.csv"
LOG_DIR = ROOT / "logs"
OUT_DIR = ROOT / "analysis" / "figures"
SEQUENCE_PATHS = {"wood": ROOT / "wood_block_pyramid.csv", "silicone": ROOT / "silicone_block_pyramid.csv"}
MATERIALS = ("wood", "silicone")
MODES = ("open", "closed")
JOINTS = ("base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll")
ACTIONS = ("pickup", "transport", "placement")

LOG_NAME = re.compile(r"(wood|silicone)_block_(open|closed)_loop_\d+hz_base_(\d+)_run(\d+)\.csv")
SECTION_HEADER = re.compile(r"# (Stack|Top|Middle|Bottom)\b")
FAIL_ACTION = re.compile(r"(\d)\s+(pickup|transport|placement)", re.IGNORECASE)

MODE_COLOR = {"open": "#2a78d6", "closed": "#eb6834"}
ACTION_COLOR = {"pickup": "#1baf7a", "transport": "#eda100", "placement": "#e87ba4"}
SUCCESS_COLOR = "#d6d5d0"
MOVING_DEG = 5.0  # a joint whose commands span this much in a row counts as moving in it
TIME_COLORS = ("#52514e", "#9c9b95", "#d6d5d0")  # planned, settling/overrun, gaps
INK = "#2b2b29"
MUTED = "#6f6f6b"
GRID = "#e4e4e0"
SURFACE = "#ffffff"


@dataclass
class Waypoint:
    line: int
    time_scale: float
    gripper_close: float
    pose: tuple = ()  # x, y, z, tilt, roll as written

    @property
    def holding(self) -> bool:
        """Gripping a block (0.5 is home's half-open gripper, with nothing in it)."""
        return 0 < self.gripper_close < 0.5


@dataclass
class Sequence:
    rows: list
    sections: list  # waypoint indices per block section, up-stack then down-stack

    def action_start(self, downstack: bool, block: int, action: str) -> int:
        """Index of the first row of a block's pickup/transport/placement (= rows done before it).
        Blocks are numbered by pyramid spot, so the down-stack moves them 6..1.
        """
        section = self.sections[12 - block if downstack else block - 1]
        grasp = next(i for i in section if self.rows[i].gripper_close > 0)
        release = next(i for i in section if i > grasp and self.rows[i].gripper_close == 0)
        return {"pickup": section[0], "transport": grasp + 1, "placement": release - 1}[action]


@dataclass
class LogStats:
    lines: list  # logged sequence lines, in order
    moves: int
    corrections: int = 0
    correction_s: float = 0.0
    correction_deg: list = field(default_factory=list)  # per nudged joint
    corrections_by_line: Counter = field(default_factory=Counter)
    joint_corrections: Counter = field(default_factory=Counter)  # (joint, holding) -> count
    holding_moves: int = 0
    settle_timeouts: int = 0
    duration_s: float = 0.0  # first row's start to last row's end
    planned_s: float = 0.0
    overrun_s: float = 0.0  # moves' time past planned (settling, in closed loop)
    gap_s: float = 0.0  # between moves (IK planning)
    tracking_err: list = field(default_factory=list)  # expected - measured, during planned motion
    end_err: list = field(default_factory=list)  # target - last measured, per row and joint
    resting_err: list = field(default_factory=list)  # target - measured, late in gripper-only rows
    envelope: list = field(default_factory=list)  # (time since move start / planned, |plan - measured|), moving joints


@dataclass
class Trial:
    material: str
    run: int
    base_ms: int
    mode: str
    up_pass: bool
    full_pass: bool
    fail_downstack: bool
    fail_action: str | None
    fail_reason: str
    progress: int  # sequence rows done before the failed action (all of them on a pass)
    log: LogStats
    log_path: Path


def read_sequence(path: Path) -> Sequence:
    rows, sections = [], []
    for line_no, text in enumerate(path.read_text().splitlines(), 1):
        text = text.strip()
        if SECTION_HEADER.match(text):
            sections.append([])
        elif text and not text.startswith("#"):
            fields = [f.strip() for f in text.split(",")]
            rows.append(Waypoint(line_no, float(fields[6]), float(fields[5]), tuple(fields[:5])))
            if sections:
                sections[-1].append(len(rows) - 1)
    if len(sections) != 12:
        raise ValueError(f"{path.name}: expected 12 block sections, found {len(sections)}")
    return Sequence(rows, sections)


def read_log(path: Path, sequence: Sequence) -> LogStats:
    """Row 0 (homing) is left out of the stats, as it isn't a sequence row."""
    with open(path.with_name(f"{path.stem}_events.csv")) as f:
        events = list(csv.DictReader(f))
    reads_by_joint = defaultdict(lambda: ([], []))  # times, expected
    reads_by_row = defaultdict(list)  # (time, joint, expected, measured or None)
    with open(path) as f:
        for r in csv.DictReader(f):
            t, expected = float(r["time_s"]), float(r["expected_deg"])
            reads_by_joint[r["joint"]][0].append(t)
            reads_by_joint[r["joint"]][1].append(expected)
            reads_by_row[r["row"]].append((t, r["joint"], expected, float(r["measured_deg"]) if r["measured_deg"] else None))

    holding = {w.line for w in sequence.rows if w.holding}
    # Gripper-only rows hold the arm still, so reads late in them show where it rests.
    still = {w.line for previous, w in zip(sequence.rows, sequence.rows[1:]) if w.pose == previous.pose}
    resting_window = {}
    timing = {}  # row -> (start, planned duration)
    moves = [e for e in events if e["event"] == "move" and e["row"] != "0"]
    stats = LogStats([int(m["line"]) for m in moves], len(moves))
    stats.duration_s = float(moves[-1]["start_s"]) + float(moves[-1]["duration_s"]) - float(moves[0]["start_s"])
    for move, after in zip(moves, moves[1:] + [None]):
        start_s, duration_s = float(move["start_s"]), float(move["duration_s"])
        planned_s = int(re.search(r"\d+", move["detail"])[0]) / 1000
        stats.planned_s += planned_s
        stats.overrun_s += duration_s - planned_s
        if after:
            stats.gap_s += float(after["start_s"]) - start_s - duration_s
        stats.holding_moves += int(move["line"]) in holding
        timing[move["row"]] = (start_s, planned_s)
        if int(move["line"]) in still:
            resting_window[move["row"]] = (start_s + planned_s / 2, start_s + planned_s)
        stats.tracking_err += [
            e - m for t, _, e, m in reads_by_row[move["row"]] if m is not None and t <= start_s + planned_s
        ]

    # A nudge shifts the joint's commanded (expected) angle by the correction;
    # nudges carry into later rows unless that row's settle timed out.
    offset = defaultdict(float)
    by_row = defaultdict(list)
    for e in events:
        by_row[e["row"]].append(e)
    for row, row_events in by_row.items():
        if not row.isdigit():
            continue
        nudges = []
        for c in (e for e in row_events if e["event"] == "correction"):
            start_s, duration_s = float(c["start_s"]), float(c["duration_s"])
            for joint in c["detail"].split():
                times, expected = reads_by_joint[joint]
                before = bisect.bisect_right(times, start_s) - 1
                after = bisect.bisect_right(times, start_s + duration_s) - 1
                if 0 <= before < after:
                    nudges.append((start_s, joint, expected[after] - expected[before]))
            if row != "0":
                stats.corrections += 1
                stats.correction_s += duration_s
                stats.corrections_by_line[int(c["line"])] += 1
                for joint in c["detail"].split():
                    stats.joint_corrections[joint, int(c["line"]) in holding] += 1
        if row != "0":
            stats.correction_deg += [abs(d) for _, _, d in nudges]
            stats.settle_timeouts += any(e["event"] == "settle" and e["detail"] == "timed out" for e in row_events)
            last = {}
            for read in reads_by_row[row]:
                if read[3] is not None:
                    last[read[1]] = read
            def offset_at(joint, t):
                return offset[joint] + sum(d for s, j, d in nudges if j == joint and s < t)

            for t, joint, expected, measured in last.values():
                stats.end_err.append(expected - offset_at(joint, t) - measured)
            if row in timing:
                start_s, planned_s = timing[row]
                span = defaultdict(list)
                for _, joint, expected, _ in reads_by_row[row]:
                    span[joint].append(expected)
                stats.envelope += [
                    ((t - start_s) / planned_s, abs(e - offset_at(j, t) - m))
                    for t, j, e, m in reads_by_row[row]
                    if m is not None and max(span[j]) - min(span[j]) >= MOVING_DEG
                ]
            if row in resting_window:
                # Before this row's own nudges, which come after the planned motion.
                window_start_s, window_end_s = resting_window[row]
                window = [(j, e - offset[j], m) for t, j, e, m in reads_by_row[row] if m is not None and window_start_s <= t <= window_end_s]
                stats.resting_err += [target - m for _, target, m in window]
        if not any(e["event"] == "settle" and e["detail"] == "timed out" for e in row_events):
            for _, joint, d in nudges:
                offset[joint] += d
    return stats


def find_logs() -> dict:
    logs = {}
    for path in LOG_DIR.glob("*_run*.csv"):
        if m := LOG_NAME.fullmatch(path.name):
            material, mode, base_ms, run = m.groups()
            logs[int(run)] = (path, material, mode, int(base_ms))
    return logs


def read_trials(sequences: dict) -> list:
    logs = find_logs()
    trials = []
    with open(RESULTS_PATH) as f:
        for r in csv.DictReader(f):
            m = re.fullmatch(r"(wood|silicone) (\d+)", r["Run"].strip())
            up = r["Upstack Pass/Fail"].strip().lower()
            if not m or not up:
                continue
            material, run = m.group(1), int(m.group(2))
            base_ms = int(r["base time ms"])
            if run not in logs:
                print(f"warning: {r['Run']} has no log; skipped")
                continue
            path, log_material, mode, log_base_ms = logs[run]
            if (log_material, log_base_ms) != (material, base_ms):
                print(f"warning: {r['Run']} ({base_ms} ms) doesn't match its log {path.name}")

            sequence = sequences[material]
            up_pass, full_pass = up == "pass", up == "pass" and r["Downstack Pass/Fail"].strip().lower() == "pass"
            fail_action = None
            if full_pass:
                progress = len(sequence.rows)
            else:
                action = FAIL_ACTION.search(r["Fail Action"])
                if not action:
                    raise ValueError(f"{r['Run']}: can't parse fail action {r['Fail Action']!r}")
                fail_action = action.group(2).lower()
                progress = sequence.action_start(up_pass, int(action.group(1)), fail_action)

            log = read_log(path, sequence)
            reached = max(i for i, w in enumerate(sequence.rows) if w.line <= log.lines[-1])
            if reached < min(progress, len(sequence.rows) - 1):
                print(f"warning: {r['Run']}'s log stops at line {log.lines[-1]}, before the sheet's failure point")
            trials.append(Trial(
                material, run, base_ms, mode, up_pass, full_pass, up_pass and not full_pass, fail_action,
                r["Failure Reason"].strip(), progress, log, path,
            ))
    return trials


# --- Statistics ---


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple:
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided p for the 2x2 table [[a, b], [c, d]]: tables no likelier than the observed one."""
    row1, row2, col1 = a + b, c + d, a + c

    def p(x):
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(row1 + row2, col1)

    observed = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, col1 - row2), min(col1, row1) + 1) if p(x) <= observed * (1 + 1e-9)))


def fmt(value: float | None, spec: str) -> str:
    return "" if value is None else format(value, spec)


def rms(values: list) -> float | None:
    return math.sqrt(statistics.mean(v * v for v in values)) if values else None


# --- Plotting ---


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def material_panels(title: str, ylabel: str, bases: list | None, rows: int = 1, figsize=(9, 3.6)):
    """One column per material (and `rows` rows); speeds on x if `bases` is given."""
    fig, axes = plt.subplots(rows, 2, figsize=figsize, sharey=True, squeeze=False, facecolor=SURFACE)
    for col, material in enumerate(MATERIALS):
        for ax in axes[:, col]:
            style_axes(ax)
            if bases:
                ax.set_xticks(bases)
        axes[0, col].set_title(f"{material.capitalize()} blocks", loc="left", fontsize=10)
        if bases:
            axes[-1, col].set_xlabel("base move time (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    return fig, axes if rows > 1 else axes[0]


def save(fig, name: str, rect=(0, 0, 1, 1)):
    fig.tight_layout(rect=rect)
    fig.savefig(OUT_DIR / f"{name}.pdf", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {OUT_DIR / name}.pdf")


def strip_with_mean(ax, values_by_base: dict, color: str, label: str):
    """Each trial as a faint dot, so coinciding trials build up darker; the per-speed mean as a line."""
    bases = sorted(b for b, v in values_by_base.items() if v)
    for base in bases:
        values = values_by_base[base]
        ax.scatter([base] * len(values), values, s=20, color=color, alpha=0.2, linewidths=0, zorder=2)
    means = [statistics.mean(values_by_base[b]) for b in bases]
    ax.plot(bases, means, color=color, linewidth=2, label=label, zorder=3)


def group(trials, material, mode, value) -> dict:
    by_base = defaultdict(list)
    for t in trials:
        if t.material == material and t.mode == mode and (v := value(t)) is not None:
            by_base[t.base_ms].append(v)
    return by_base


def mode_strips(trials, bases, title, ylabel, value, modes=MODES):
    fig, axes = material_panels(title, ylabel, bases)
    for ax, material in zip(axes, MATERIALS):
        for mode in modes:
            strip_with_mean(ax, group(trials, material, mode, value), MODE_COLOR[mode], f"{mode} loop")
    if len(modes) > 1:
        figure_legend(fig, axes[0])
    return fig, axes


def figure_legend(fig, ax=None, handles=None, ncol=3):
    """Top right, beside the title, so it never covers data."""
    handles, labels = (handles, [h.get_label() for h in handles]) if handles else ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=ncol, frameon=False)


def closed_only(value):
    return lambda t: value(t) if t.mode == "closed" else None


def configurations(trials):
    """(material, mode, base_ms, trials) for every tested configuration."""
    for material in MATERIALS:
        for mode in MODES:
            for base, ts in sorted(group(trials, material, mode, lambda t: t).items()):
                yield material, mode, base, ts


def plot_success_rate(trials, bases):
    """Full sequence solid, up-stack alone dashed."""
    fig, axes = material_panels("Success rate vs speed", "success rate (%)", bases)
    for ax, material in zip(axes, MATERIALS):
        for mode in MODES:
            for attr, dash, what in (("full_pass", "-", "full sequence"), ("up_pass", "--", "up-stack only")):
                rates = {b: 100 * statistics.mean(getattr(t, attr) for t in v) for b, v in group(trials, material, mode, lambda t: t).items()}
                ax.plot(
                    bases, [rates[b] for b in bases], dash, color=MODE_COLOR[mode], linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{mode} loop, {what}",
                )
        ax.set_ylim(-5, 105)
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    save(fig, "success_rate")


def plot_success_rate_ci(trials, bases):
    """Success (full sequence, then up-stack alone) with 95% Wilson intervals; * where
    open and closed loop differ by Fisher's exact test (p < 0.05). Also writes stats.csv.
    """
    rows = defaultdict(list)  # (material, base) -> columns
    for attr, title, name in (
        ("full_pass", "Full-sequence success rate", "success_rate_ci"),
        ("up_pass", "Up-stack (pyramid built) success rate", "success_rate_upstack_ci"),
    ):
        fig, axes = material_panels(f"{title}, 95% confidence intervals", "success rate (%)", bases)
        for ax, material in zip(axes, MATERIALS):
            counts = {}
            for mode in MODES:
                by_base = group(trials, material, mode, lambda t: t)
                counts[mode] = {b: (sum(getattr(t, attr) for t in v), len(v)) for b, v in by_base.items()}
                intervals = [wilson_interval(*counts[mode][b]) for b in bases]
                ax.fill_between(bases, [100 * lo for lo, _ in intervals], [100 * hi for _, hi in intervals], color=MODE_COLOR[mode], alpha=0.12, linewidth=0)
                ax.plot(bases, [100 * counts[mode][b][0] / counts[mode][b][1] for b in bases], color=MODE_COLOR[mode], linewidth=2, label=f"{mode} loop")
            for b in bases:
                (ko, no), (kc, nc) = counts["open"][b], counts["closed"][b]
                p = fisher_exact(ko, no - ko, kc, nc - kc)
                if p < 0.05:
                    ax.annotate("*", (b, 102), ha="center", fontsize=14, color=INK)
                rows[material, b] += [f"{ko}/{no}", f"{kc}/{nc}"]
                for k, n in ((ko, no), (kc, nc)):
                    lo, hi = wilson_interval(k, n)
                    rows[material, b].append(f"{max(0.0, 100 * lo):.0f}-{100 * hi:.0f}")
                rows[material, b].append(f"{p:.3f}")
            ax.set_ylim(-5, 110)
        figure_legend(fig, axes[0])
        save(fig, name)
    with open(OUT_DIR / "stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["material", "base_ms"] + [
            f"{stage}_{column}" for stage in ("full", "up")
            for column in ("open", "closed", "open_ci95_pct", "closed_ci95_pct", "fisher_p")
        ])
        writer.writerows([material, b, *columns] for (material, b), columns in rows.items())
    print(f"Saved {OUT_DIR / 'stats.csv'}")


def plot_failure_modes(trials, bases):
    """Outcome counts per speed: successes, then failures by action (hatched: down-stack)."""
    fig, axes = material_panels("Outcomes by speed", "trials", None, rows=2, figsize=(9, 6))
    x = range(len(bases))
    for row, mode in enumerate(MODES):
        for col, material in enumerate(MATERIALS):
            ax = axes[row, col]
            by_base = group(trials, material, mode, lambda t: t)
            bottom = [0] * len(bases)
            layers = [("success", SUCCESS_COLOR, "", lambda t: t.full_pass)]
            for downstack, hatch in ((False, ""), (True, "///")):
                for action in ACTIONS:
                    layers.append((action, ACTION_COLOR[action], hatch, lambda t, d=downstack, a=action: t.fail_action == a and t.fail_downstack == d))
            for _, color, hatch, match in layers:
                heights = [sum(map(match, by_base[b])) for b in bases]
                ax.bar(x, heights, bottom=bottom, width=0.7, color=color, hatch=hatch, edgecolor=SURFACE, linewidth=1)
                bottom = [b + h for b, h in zip(bottom, heights)]
            ax.set_xticks(list(x), [str(b) for b in bases])
            if row == 1:
                ax.set_xlabel("base move time (ms)")
        axes[row, 0].set_ylabel(f"{mode} loop: trials")
    handles = [Patch(facecolor=SUCCESS_COLOR, label="success")]
    handles += [Patch(facecolor=ACTION_COLOR[a], label=f"failed at {a}") for a in ACTIONS]
    handles.append(Patch(facecolor="white", edgecolor=MUTED, hatch="///", label="hatched: down-stack"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    save(fig, "failure_modes", rect=(0, 0.05, 1, 1))
    with open(OUT_DIR / "failure_reasons.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["material", "mode", "base_ms", "stage", "failure_reason", "count"])
        for material, mode, base, ts in configurations(trials):
            reasons = Counter(("down" if t.fail_downstack else "up", t.fail_reason.lower()) for t in ts if not t.full_pass)
            for (stage, reason), count in sorted(reasons.items()):
                writer.writerow([material, mode, base, stage, reason, count])
    print(f"Saved {OUT_DIR / 'failure_reasons.csv'}")


def plot_corrections_by_joint(trials):
    """Closed loop, all speeds: corrections per 100 moves, holding a block or not."""
    fig, axes = material_panels("Closed-loop corrections per joint", "corrections per 100 moves", None)
    x = range(len(JOINTS))
    for ax, material in zip(axes, MATERIALS):
        logs = [t.log for t in trials if t.material == material and t.mode == "closed"]
        holding_moves = sum(log.holding_moves for log in logs)
        free_moves = sum(log.moves for log in logs) - holding_moves
        for i, (holding, moves, alpha, label) in enumerate(((False, free_moves, 0.4, "empty gripper"), (True, holding_moves, 1.0, "holding a block"))):
            rates = [100 * sum(log.joint_corrections[j, holding] for log in logs) / moves for j in JOINTS]
            ax.bar([v + (i - 0.5) * 0.38 for v in x], rates, width=0.36, color=MODE_COLOR["closed"], alpha=alpha, label=label)
        ax.set_xticks(list(x), [j.replace("_", "\n") for j in JOINTS])
    figure_legend(fig, axes[0])
    save(fig, "corrections_by_joint")


def plot_corrections_by_line(trials, sequences):
    """Closed loop, all speeds: corrections per run at each sequence row, over the runs that got there."""
    fig, axes = material_panels("Closed-loop corrections along the sequence", "corrections per run", None, figsize=(11, 3.6))
    for ax, material in zip(axes, MATERIALS):
        logs = [t.log for t in trials if t.material == material and t.mode == "closed"]
        rows = sequences[material].rows
        rates = []
        for w in rows:
            reached = sum(w.line in log.lines for log in logs)
            rates.append(sum(log.corrections_by_line[w.line] for log in logs) / reached if reached else 0)
        ax.bar(range(len(rows)), rates, width=0.8, color=MODE_COLOR["closed"])
        downstack = sequences[material].sections[6][0]
        ax.axvline(downstack - 0.5, color=MUTED, linewidth=1, linestyle=":")
        ax.annotate("down-stack →", (downstack, 1), xytext=(3, 0), textcoords="offset points", xycoords=("data", "axes fraction"), color=MUTED, fontsize=8, va="top")
        ax.set_xlabel("sequence row")
        top = sorted(range(len(rows)), key=lambda i: -rates[i])[:5]
        print(f"{material}: most-corrected rows (CSV line: corrections/run): " + ", ".join(f"{rows[i].line}: {rates[i]:.2f}" for i in top))
    save(fig, "corrections_by_line")


def plot_time_breakdown(trials, bases):
    """Mean time per move: planned motion, time past it (settling, in closed loop), and gaps between moves."""
    fig, axes = material_panels("Where the time goes", "time per move (ms)", None, rows=2, figsize=(9, 6))
    x = range(len(bases))
    labels = ("planned motion", "settling / overrun", "between moves (planning)")
    for row, mode in enumerate(MODES):
        for col, material in enumerate(MATERIALS):
            ax = axes[row, col]
            by_base = group(trials, material, mode, lambda t: t.log)
            bottom = [0.0] * len(bases)
            for attr, color, label in zip(("planned_s", "overrun_s", "gap_s"), TIME_COLORS, labels):
                heights = [1000 * sum(getattr(log, attr) for log in by_base[b]) / sum(log.moves for log in by_base[b]) for b in bases]
                ax.bar(x, heights, bottom=bottom, width=0.7, color=color, edgecolor=SURFACE, linewidth=1, label=label)
                bottom = [b + h for b, h in zip(bottom, heights)]
            ax.set_xticks(list(x), [str(b) for b in bases])
            ax.text(0.02, 0.97, f"{mode} loop", transform=ax.transAxes, va="top", color=MUTED)
            if row == 1:
                ax.set_xlabel("base move time (ms)")
    figure_legend(fig, axes[0, 0])
    save(fig, "time_breakdown")


def plot_joint_traces(trials, bases, sequences, bins_per_row: int = 5, min_reads: int = 3):
    """Median expected vs measured angle of each joint through block 1's pick and place, per
    speed. Time within each row is a fraction of that row's actual duration (settling
    included), so runs line up; reads are pooled across runs into bins_per_row bins (median: a stray read can't drag a sparse bin;
    bins with fewer than min_reads reads are left out).
    """
    joints = JOINTS[:4]  # wrist roll stays at 0 throughout
    for material in MATERIALS:
        lines = [sequences[material].rows[i].line for i in sequences[material].sections[0]]
        fig, axes = plt.subplots(len(joints), len(bases), figsize=(14, 8), sharex=True, sharey="row", facecolor=SURFACE)
        for col, base in enumerate(bases):
            for mode in MODES:
                pooled = defaultdict(lambda: ([], []))  # (joint, x) -> expected, measured
                for t in trials:
                    if (t.material, t.mode, t.base_ms) != (material, mode, base):
                        continue
                    with open(t.log_path.with_name(f"{t.log_path.stem}_events.csv")) as f:
                        moves = {e["row"]: e for e in csv.DictReader(f) if e["event"] == "move" and e["line"]}
                    with open(t.log_path) as f:
                        for r in csv.DictReader(f):
                            move = moves.get(r["row"])
                            if not move or int(move["line"]) not in lines or not r["measured_deg"] or r["joint"] not in joints:
                                continue
                            phase = (float(r["time_s"]) - float(move["start_s"])) / float(move["duration_s"])
                            b = min(bins_per_row - 1, max(0, int(phase * bins_per_row)))
                            x = lines.index(int(move["line"])) + (b + 0.5) / bins_per_row
                            pooled[r["joint"], x][0].append(float(r["expected_deg"]))
                            pooled[r["joint"], x][1].append(float(r["measured_deg"]))
                for ax, joint in zip(axes[:, col], joints):
                    xs = sorted(x for j, x in pooled if j == joint and len(pooled[j, x][1]) >= min_reads)
                    expected = [statistics.median(pooled[joint, x][0]) for x in xs]
                    measured = [statistics.median(pooled[joint, x][1]) for x in xs]
                    ax.plot(xs, expected, "--", color=MODE_COLOR[mode], linewidth=1.2, alpha=0.7, label=f"{mode} loop, expected")
                    ax.plot(xs, measured, color=MODE_COLOR[mode], linewidth=2, label=f"{mode} loop, measured")
            axes[0, col].set_title(f"{base} ms", loc="left", fontsize=10)
            axes[-1, col].set_xticks(range(len(lines)), [str(line) for line in lines], fontsize=7)
            axes[-1, col].set_xlabel("CSV line (row start)")
        for ax in axes.flat:
            style_axes(ax)
        for ax, joint in zip(axes[:, 0], joints):
            ax.set_ylabel(f"{joint}\n(deg)")
        fig.suptitle(f"{material.capitalize()} blocks: median joint angles through block 1's pick and place", x=0.01, ha="left", fontsize=11)
        figure_legend(fig, axes[0, 0], ncol=4)
        save(fig, f"joint_traces_{material}", rect=(0, 0, 1, 0.96))


def plot_error_envelope(trials, bases, bin_width: float = 0.1, max_phase: float = 3.0, min_samples: int = 20):
    """Moving joints' error to the plan (the logged command, closed loop's correction
    offsets removed) through a move, pooled over every move of every run: median (line)
    and quartiles (band) per bin of time since the move started / its planned duration.
    Open loop never reads past t/T = 1, as the next move starts then.
    """
    fig, axes = plt.subplots(2, len(bases), figsize=(14, 6), sharex=True, sharey=True, facecolor=SURFACE)
    for row_index, material in enumerate(MATERIALS):
        for col, base in enumerate(bases):
            ax = axes[row_index, col]
            for mode in MODES:
                bins = defaultdict(list)
                for t in trials:
                    if (t.material, t.mode, t.base_ms) == (material, mode, base):
                        for phase, error in t.log.envelope:
                            if 0 <= phase < max_phase:
                                bins[int(phase / bin_width)].append(error)
                kept = sorted(b for b, v in bins.items() if len(v) >= min_samples)
                xs = [(b + 0.5) * bin_width for b in kept]
                quartiles = [statistics.quantiles(bins[b], n=4) for b in kept]
                ax.fill_between(xs, [q[0] for q in quartiles], [q[2] for q in quartiles], color=MODE_COLOR[mode], alpha=0.15, linewidth=0)
                ax.plot(xs, [q[1] for q in quartiles], color=MODE_COLOR[mode], linewidth=2, label=f"{mode} loop (median, quartiles)")
            ax.axvline(1, color=MUTED, linewidth=1, linestyle=":")
            ax.axhline(2, color=MUTED, linewidth=1, linestyle="--")
            ax.set_yscale("log")
            style_axes(ax)
            if row_index == 0:
                ax.set_title(f"{base} ms", loc="left", fontsize=10)
            else:
                ax.set_xlabel("time since move start / planned duration")
        axes[row_index, 0].set_ylabel(f"{material.capitalize()} blocks\n|plan - measured| (deg)")
    axes[0, 0].annotate("planned end", (1, 1), xycoords=("data", "axes fraction"), xytext=(3, -3), textcoords="offset points", va="top", fontsize=7, color=MUTED)
    axes[0, 0].annotate("2° tolerance", (max_phase, 2), xytext=(-3, 3), textcoords="offset points", ha="right", fontsize=7, color=MUTED)
    fig.suptitle("Joint error to the plan through a move (moving joints only)", x=0.01, ha="left", fontsize=11)
    figure_legend(fig, axes[0, 0])
    save(fig, "error_envelope")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "legend.handlelength": 3.5})
    sequences = {m: read_sequence(p) for m, p in SEQUENCE_PATHS.items()}
    trials = read_trials(sequences)
    bases = sorted({t.base_ms for t in trials})

    plot_success_rate(trials, bases)
    plot_success_rate_ci(trials, bases)
    plot_failure_modes(trials, bases)

    total_rows = len(sequences["wood"].rows)
    fig, axes = mode_strips(trials, bases, "Progress before failure vs speed", "sequence rows completed", lambda t: t.progress)
    for ax in axes:
        ax.axhline(total_rows, color=MUTED, linewidth=1, linestyle=":")
        ax.annotate("complete", (bases[0], total_rows), xytext=(0, 3), textcoords="offset points", color=MUTED, fontsize=8)
        ax.axhline(sequences["wood"].sections[6][0], color=MUTED, linewidth=1, linestyle=":")
        ax.annotate("down-stack starts", (bases[0], sequences["wood"].sections[6][0]), xytext=(0, 3), textcoords="offset points", color=MUTED, fontsize=8)
    save(fig, "progress")

    closed = ("closed",)
    fig, _ = mode_strips(trials, bases, "Closed-loop corrections per move", "corrections per move", closed_only(lambda t: t.log.corrections / t.log.moves), closed)
    save(fig, "corrections_per_move")
    fig, _ = mode_strips(
        trials, bases, "Closed-loop correction time per move", "correction time per move (ms)",
        closed_only(lambda t: 1000 * t.log.correction_s / t.log.moves), closed,
    )
    save(fig, "correction_time_per_move")
    fig, _ = mode_strips(
        trials, bases, "Closed-loop mean correction magnitude", "mean correction (deg)",
        closed_only(lambda t: statistics.mean(t.log.correction_deg) if t.log.correction_deg else None), closed,
    )
    save(fig, "correction_magnitude")
    fig, _ = mode_strips(
        trials, bases, "Closed-loop settle timeouts", "moves timed out (%)",
        closed_only(lambda t: 100 * t.log.settle_timeouts / t.log.moves), closed,
    )
    save(fig, "settle_timeouts")
    plot_corrections_by_joint(trials)
    plot_corrections_by_line(trials, sequences)

    fig, _ = mode_strips(trials, bases, "Tracking error during motion", "RMS joint error (deg)", lambda t: rms(t.log.tracking_err))
    save(fig, "tracking_error")
    fig, _ = mode_strips(trials, bases, "End-of-move error (vs target)", "RMS joint error (deg)", lambda t: rms(t.log.end_err))
    save(fig, "end_of_move_error")
    fig, _ = mode_strips(
        trials, bases, "Resting error during gripper-only rows (vs target)", "RMS joint error (deg)", lambda t: rms(t.log.resting_err)
    )
    save(fig, "resting_error")

    # Execution time, from logs covering the whole sequence (whatever the sheet says).
    def complete(t):
        rows = sequences[t.material].rows
        return t.log.lines[0] == rows[0].line and t.log.lines[-1] == rows[-1].line

    fig, axes = mode_strips(
        trials, bases, "Expected vs actual execution time", "execution time (s), complete logs only",
        lambda t: t.log.duration_s if complete(t) else None,
    )
    for ax, material in zip(axes, MATERIALS):
        scale = sum(w.time_scale for w in sequences[material].rows)
        ax.plot(bases, [scale * b / 1000 for b in bases], "--", color=MUTED, linewidth=1.5, label="expected (planned)")
    fig.legends.clear()
    figure_legend(fig, axes[0])
    save(fig, "execution_time")
    plot_time_breakdown(trials, bases)
    plot_joint_traces(trials, bases, sequences)
    plot_error_envelope(trials, bases)

    summary_path = OUT_DIR / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "material", "mode", "base_ms", "trials", "up_success_pct", "full_success_pct", "mean_progress_rows",
            "corrections_per_move", "correction_ms_per_move", "mean_correction_deg", "settle_timeouts_per_move",
            "tracking_rms_deg", "end_error_rms_deg", "resting_error_rms_deg", "complete_logs", "expected_s", "mean_actual_s",
        ])
        for material, mode, base, ts in configurations(trials):
            scale = sum(w.time_scale for w in sequences[material].rows)
            moves = sum(t.log.moves for t in ts)
            magnitudes = [d for t in ts for d in t.log.correction_deg]
            durations = [t.log.duration_s for t in ts if complete(t)]
            writer.writerow([
                material, mode, base, len(ts),
                f"{100 * statistics.mean(t.up_pass for t in ts):.0f}",
                f"{100 * statistics.mean(t.full_pass for t in ts):.0f}",
                f"{statistics.mean(t.progress for t in ts):.1f}",
                f"{sum(t.log.corrections for t in ts) / moves:.3f}",
                f"{1000 * sum(t.log.correction_s for t in ts) / moves:.1f}",
                f"{statistics.mean(magnitudes):.2f}" if magnitudes else "",
                f"{sum(t.log.settle_timeouts for t in ts) / moves:.3f}",
                fmt(rms([e for t in ts for e in t.log.tracking_err]), ".2f"),
                fmt(rms([e for t in ts for e in t.log.end_err]), ".2f"),
                fmt(rms([e for t in ts for e in t.log.resting_err]), ".2f"),
                len(durations), f"{scale * base / 1000:.1f}",
                f"{statistics.mean(durations):.1f}" if durations else "",
            ])
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
