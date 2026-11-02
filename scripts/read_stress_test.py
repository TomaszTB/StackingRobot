"""MOVES THE ARM. Stress-tests the xArm 1S's position reads, using the stock
xarm library with none of the controller's workarounds, then adds them back
one at a time. Run it unchanged on each robot and compare the logs.

Phases (each prints what to watch for, then asks what you saw/heard):
  S*  arm still: reads alone, then reads while a hold command streams at 50 Hz
  M*  stock pattern while sweeping: a 6-servo command, then the stock
      all-at-once read right after it, every tick, at 50/20/10/5 Hz
  V*  at 5 Hz, one variable at a time: read timing, the read type, and
      the controller's own read path (drain stale replies, check IDs, gaps)
  T*, W*  candidate controller patterns at 10 Hz (T) and 20 Hz (W): safe
      all-at-once reads (every tick, mid-tick, every Nth tick), and single
      reads (all five, as many as fit in a tick, or one per tick)
Motion phases sweep between two poses 4 times (T/W: 8), 2 s each (base swings ~65 deg,
the rest 10-25 deg), with the tip at least 12 cm up -- keep the area clear.

After each move, every joint is read the safe way at the moment it should have
arrived: how far behind it is ("lag") measures hitches objectively. Compare
against the no-reads phase at the same rate (M0, T0, W0, V0).

Before running: power from the usual supply, check the servo cable connectors
(especially elbow -> wrist), and have the robot's label handy for --robot.

Writes logs/stress_test<N>.log (setup, per-phase summaries, your observations,
and a final table) and logs/stress_test<N>_requests.csv (every command/read).

Usage:
    uv run python scripts/read_stress_test.py                   # dry run (fake robot)
    uv run python scripts/read_stress_test.py --execute --robot "lab copy"
    uv run python scripts/read_stress_test.py --execute --robot "lab copy" --phases T,W   # whole groups
"""

import argparse
import contextlib
import csv
import io
import math
import platform
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import xarm

from xarm_ik import GripperPoseController, config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_csv import approach_vector  # noqa: E402

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

# x, y, z (m), tilt, roll (deg) -- see run_csv.py. Swept between in joint space.
POSE_A = (0.15, -0.2, 0.22, 90.0, 20.0)
POSE_B = (-0.1, -0.2, 0.12, 60.0, -20.0)
GRIPPER_OPEN_FRACTION = 0.5
MOVE_S = 2.0
MOVES_PER_PHASE = 4
STILL_READS = 100
HOLD_S = 5.0

CMD_GET_SERVO_POSITION = 0x15
CMD_GET_BATTERY_VOLTAGE = 0x0F
STOCK_TIMEOUT_MS = 50  # what xarm's getPosition waits
# Fixed (not config's), so logs stay comparable if the controller is retuned.
SAFE_TIMEOUT_MS = 30
GAP_MS = 20
# A reply sooner than this per servo asked was already waiting: a late reply to an
# earlier request (real ones took >= ~6 ms for one servo, >= ~12 ms for five).
STALE_MS_PER_SERVO = 2.0
LATE_TICK_MS = 10.0

ARM_IDS = [joint.servo_id for joint in config.JOINTS]
JOINT_BY_ID = {joint.servo_id: joint for joint in config.JOINTS}


def raw_to_deg(servo_id: int, position: float) -> float:
    return JOINT_BY_ID[servo_id].to_angle_deg(position)


@dataclass
class Phase:
    key: str
    title: str
    question: str  # what this phase isolates, for whoever reads the log
    kind: str  # "still", "hold" or "move"
    rate_hz: float = 0.0
    reads: str = "none"  # see Runner._tick_reads
    gap_before_command: bool = False
    moves: int = MOVES_PER_PHASE
    read_every: int = 1  # read on every Nth tick only


WATCH_STILL = "The arm shouldn't move. LISTEN for beeps (count them)."
WATCH_HOLD = "The arm should stay still. LISTEN for beeps; WATCH for twitches."
WATCH_MOVE = (
    "WATCH {moves} sweeps: smooth / occasional hitch / regular jerks / stop-and-go? "
    "Which joints? LISTEN for beeps (count them)."
)

PHASES = [
    Phase("S1", "still: stock all-at-once reads, back to back", "Do reads fail with no motion or commands at all?", "still", reads="all"),
    Phase("S2", "still: stock single reads, back to back, rotating", "Same, for one servo per request.", "still", reads="rotate"),
    Phase("S3", "hold: 6-servo hold command + stock all-at-once read, 50 Hz", "Do commands + reads interfere without motion/load?", "hold", 50, "all"),
    Phase("M0", "sweep @ 50 Hz, no reads", "Baseline: smoothness and lag with no reads.", "move", 50),
    Phase("M1", "sweep @ 50 Hz, stock all-at-once read after each command", "The no-workarounds target. Does it work?", "move", 50, "all"),
    Phase("M2", "sweep @ 20 Hz, stock all-at-once read after each command", "Does the stock pattern work at a lower rate?", "move", 20, "all"),
    Phase("M3", "sweep @ 10 Hz, stock all-at-once read after each command", "(rate ladder)", "move", 10, "all"),
    Phase("M4", "sweep @ 5 Hz, stock all-at-once read after each command", "(rate ladder; compare with V0)", "move", 5, "all"),
    Phase("V0", "sweep @ 5 Hz, no reads", "Baseline for the V phases.", "move", 5),
    Phase("V1", "sweep @ 5 Hz, stock all-at-once read mid-tick", "Is it the read landing right after a command?", "move", 5, "all_mid"),
    Phase("V2", "sweep @ 5 Hz, stock all-at-once read 20 ms after the command", "Does a 20 ms gap before the read help?", "move", 5, "all_gap"),
    Phase("V3", "sweep @ 5 Hz, safe all-at-once read (drain stale, check IDs, 30 ms timeout)", "Are failures stale/mismatched replies (the library) or the board?", "move", 5, "all_safe"),
    Phase("V4", "sweep @ 5 Hz, five stock single reads back to back", "Is it the all-at-once request itself?", "move", 5, "singles"),
    Phase("V5", "sweep @ 5 Hz, five safe single reads with 20 ms gaps (= our controller)", "Reference: today's workaround.", "move", 5, "singles_safe", True),
    Phase("V6", "sweep @ 5 Hz, one stock single read per tick, rotating", "Spreading reads out: one servo hitches at a time?", "move", 5, "rotate"),
    Phase("T0", "sweep @ 10 Hz, no reads", "Baseline for the T phases.", "move", 10, moves=8),
    Phase("T1", "sweep @ 10 Hz, safe all-at-once read after each command", "Full feedback every tick at 10 Hz?", "move", 10, "all_safe", moves=8),
    Phase("T2", "sweep @ 10 Hz, safe all-at-once read mid-tick", "Does reading away from the command help at 10 Hz?", "move", 10, "all_safe_mid", moves=8),
    Phase("T3", "sweep @ 10 Hz, safe all-at-once read every 2nd tick", "Full feedback at 5 Hz under 10 Hz commands?", "move", 10, "all_safe", moves=8, read_every=2),
    Phase("T4", "sweep @ 10 Hz, five stock single reads back to back", "Do 5 singles (~110 ms) fit a 100 ms tick, and stay smooth?", "move", 10, "singles", moves=8),
    Phase("T5", "sweep @ 10 Hz, safe single reads, as many as fit per tick, rotating", "Most single-read feedback 10 Hz allows?", "move", 10, "fit", moves=8),
    Phase("T6", "sweep @ 10 Hz, one stock single read per tick, rotating", "Light single-read feedback at 10 Hz.", "move", 10, "rotate", moves=8),
    Phase("W0", "sweep @ 20 Hz, no reads", "Baseline for the W phases.", "move", 20, moves=8),
    Phase("W1", "sweep @ 20 Hz, safe all-at-once read after each command", "Full feedback every tick at 20 Hz?", "move", 20, "all_safe", moves=8),
    Phase("W2", "sweep @ 20 Hz, safe all-at-once read mid-tick", "Does reading away from the command help at 20 Hz?", "move", 20, "all_safe_mid", moves=8),
    Phase("W3", "sweep @ 20 Hz, safe all-at-once read every 2nd tick", "Full feedback at 10 Hz under 20 Hz commands?", "move", 20, "all_safe", moves=8, read_every=2),
    Phase("W4", "sweep @ 20 Hz, safe all-at-once read every 4th tick", "Full feedback at 5 Hz under 20 Hz commands?", "move", 20, "all_safe", moves=8, read_every=4),
    Phase("W5", "sweep @ 20 Hz, safe single reads, as many as fit per tick, rotating", "Most single-read feedback 20 Hz allows?", "move", 20, "fit", moves=8),
    Phase("W6", "sweep @ 20 Hz, one stock single read per tick, rotating", "Light single-read feedback at 20 Hz.", "move", 20, "rotate", moves=8),
]


@dataclass
class ReadResult:
    requested: list
    sent_s: float
    done_s: float
    outcome: str  # ok, timeout, partial, wrong_ids, reordered, bad_value, wrong_cmd
    positions: dict
    returned_ids: list
    drained: int = 0

    @property
    def latency_ms(self) -> float:
        return (self.done_s - self.sent_s) * 1000


def classify(report: list, ids: list) -> tuple:
    """(outcome, valid positions of requested IDs, IDs in the reply's order)."""
    if len(report) < 5 or report[0] != 0x55 or report[1] != 0x55 or report[3] != CMD_GET_SERVO_POSITION:
        return "wrong_cmd", {}, []
    returned, positions, bad_value = [], {}, False
    for i in range(report[4]):
        entry = report[5 + 3 * i : 8 + 3 * i]
        if len(entry) < 3:
            break
        returned.append(entry[0])
        position = entry[2] * 256 + entry[1]
        if not 0 <= position <= 1000:
            bad_value = True
        elif entry[0] in ids:
            positions[entry[0]] = position
    if set(returned) - set(ids):
        return "wrong_ids", positions, returned
    if bad_value:
        return "bad_value", positions, returned
    if len(returned) < len(ids):
        return "partial", positions, returned
    if returned != list(ids):
        return "reordered", positions, returned
    return "ok", positions, returned


class Bus:
    """The board, through xarm.Controller: commands via stock setPosition; reads
    either stock (send, take the first report within 50 ms -- what getPosition
    does, minus its crash on a timeout) or safe (the controller's read path).
    """

    def __init__(self, execute: bool):
        self.device = xarm.Controller("USB") if execute else FakeController()
        self.hid = self.device._device
        self.last_request_s = 0.0

    def serial_number(self) -> str:
        try:
            return self.hid.get_serial_number_string()
        except Exception as e:
            return f"unknown ({e})"

    def wait_gap(self) -> None:
        remaining_s = self.last_request_s + GAP_MS / 1000 - time.perf_counter()
        if remaining_s > 0:
            time.sleep(remaining_s)

    def command(self, positions: dict, duration_ms: int) -> float:
        sent_s = time.perf_counter()
        self.device.setPosition(list(positions.items()), duration=max(1, duration_ms), wait=False)
        self.last_request_s = time.perf_counter()
        return sent_s

    def read(self, ids: list, safe: bool = False) -> ReadResult:
        drained = 0
        if safe:
            while self.hid.read(64, 0):
                drained += 1
        sent_s = time.perf_counter()
        self.device._send(CMD_GET_SERVO_POSITION, bytearray([len(ids), *ids]))
        deadline_s = sent_s + (SAFE_TIMEOUT_MS if safe else STOCK_TIMEOUT_MS) / 1000
        outcome, positions, returned = "timeout", {}, []
        while (remaining_ms := int((deadline_s - time.perf_counter()) * 1000)) > 0:
            report = self.hid.read(64, remaining_ms)
            if not report:
                break
            outcome, positions, returned = classify(report, ids)
            if not safe or outcome not in ("wrong_cmd", "wrong_ids"):  # stock takes whatever comes first
                break
            outcome = "timeout"
        done_s = time.perf_counter()
        self.last_request_s = done_s
        return ReadResult(list(ids), sent_s, done_s, outcome, positions, returned, drained)

    def voltage(self) -> float | None:
        while self.hid.read(64, 0):
            pass
        self.device._send(CMD_GET_BATTERY_VOLTAGE)
        deadline_s = time.perf_counter() + 0.1
        try:
            while (remaining_ms := int((deadline_s - time.perf_counter()) * 1000)) > 0:
                report = self.hid.read(64, remaining_ms)
                if not report:
                    return None
                if len(report) >= 6 and report[3] == CMD_GET_BATTERY_VOLTAGE:
                    return (report[5] * 256 + report[4]) / 1000
            return None
        finally:
            self.last_request_s = time.perf_counter()


class FakeController:
    """Dry-run stand-in for xarm.Controller: replies ~5 ms late, sometimes
    very late, partial, or not at all, so every code path gets exercised.
    """

    def __init__(self):
        self._device = FakeHid()

    def setPosition(self, servos, duration=1000, wait=False):
        self._device.positions.update(dict(servos))

    def _send(self, cmd, data=b""):
        self._device.request(cmd, bytes(data))


class FakeHid:
    def __init__(self):
        self.positions = {servo_id: 500 for servo_id in range(1, 7)}
        self.queue = []

    def request(self, cmd, data):
        roll = random.random()
        if cmd == CMD_GET_BATTERY_VOLTAGE:
            report = [0x55, 0x55, 4, cmd, 0x58, 0x1D]
        elif roll < 0.03:
            return
        else:
            ids = list(data[1:])[: -1 if roll < 0.06 and len(data) > 2 else None]
            report = [0x55, 0x55, 3 + 3 * len(ids), cmd, len(ids)]
            for servo_id in ids:
                report += [servo_id, self.positions[servo_id] & 0xFF, self.positions[servo_id] >> 8]
        delay_s = 0.06 if roll < 0.08 else 0.004 + 0.0025 * max(1, len(data) - 1)
        self.queue.append((time.perf_counter() + delay_s, report))

    def read(self, max_length, timeout_ms=0):
        deadline_s = time.perf_counter() + timeout_ms / 1000
        while True:
            ready = [item for item in self.queue if item[0] <= time.perf_counter()]
            if ready:
                self.queue.remove(ready[0])
                return ready[0][1]
            if time.perf_counter() >= deadline_s:
                return []
            time.sleep(0.0005)

    def get_serial_number_string(self):
        return "dry-run"


class Expected:
    """Where the servos should be: each command moves them in a straight line
    from wherever they were to its target, over its duration.
    """

    def __init__(self, positions: dict):
        self.start = self.end = dict(positions)
        self.start_s = self.end_s = 0.0

    def at(self, time_s: float) -> dict:
        span_s = self.end_s - self.start_s
        t = 1.0 if span_s <= 0 else min(1.0, max(0.0, (time_s - self.start_s) / span_s))
        return {i: self.start[i] + t * (self.end[i] - self.start[i]) for i in self.end}

    def commanded(self, sent_s: float, positions: dict, arrive_s: float) -> None:
        self.start, self.end = self.at(sent_s), dict(positions)
        self.start_s, self.end_s = sent_s, max(sent_s, arrive_s)


@dataclass
class PhaseStats:
    reads: list = field(default_factory=list)  # ReadResult
    command_times: list = field(default_factory=list)  # one list per sweep/hold
    late_ticks: int = 0
    lags: list = field(default_factory=list)  # per move: {servo_id: deg}
    tracking: dict = field(default_factory=lambda: {i: [] for i in ARM_IDS})  # measured - expected, deg
    volts: tuple = (None, None)
    observation: str = ""
    aborted: bool = False


class Runner:
    def __init__(self, bus: Bus, poses: dict, requests_csv, prompt: bool):
        self.bus, self.poses, self.csv, self.prompt = bus, poses, requests_csv, prompt
        self.t0 = time.perf_counter()
        self.expected = Expected(poses["home"])
        self.rotation = 0

    # --- Recording ---

    def _row(self, phase, kind, ids="", duration_ms="", result=None, note="", commanded=None):
        if result is None:
            commanded_raw = " ".join(f"{i}:{p}" for i, p in commanded.items()) if commanded else ""
            self.csv.writerow([phase, f"{time.perf_counter() - self.t0:.4f}", kind, ids, duration_ms, "", "", "", "", "", note, commanded_raw])
            return
        expected = self.expected.at(result.done_s)
        self.csv.writerow([
            phase,
            f"{result.sent_s - self.t0:.4f}",
            kind,
            " ".join(map(str, ids)),
            "",
            f"{result.latency_ms:.2f}",
            result.outcome,
            " ".join(map(str, result.returned_ids)),
            " ".join(f"{i}:{p}" for i, p in result.positions.items()),
            " ".join(f"{i}:{expected[i]:.1f}" for i in ids),
            f"drained {result.drained}" if result.drained else note,
        ])

    def _read(self, phase: Phase, stats: PhaseStats, ids: list, safe=False, kind="read") -> ReadResult:
        result = self.bus.read(ids, safe=safe)
        self._row(phase.key, kind + ("_safe" if safe else "_stock"), ids, result=result)
        if kind == "read":
            stats.reads.append(result)
            expected = self.expected.at(result.done_s)
            for i, position in result.positions.items():
                stats.tracking[i].append(raw_to_deg(i, position) - raw_to_deg(i, expected[i]))
        return result

    def _command(self, phase_key: str, positions: dict, arrive_s: float) -> float:
        duration_ms = max(1, round((arrive_s - time.perf_counter()) * 1000))
        sent_s = self.bus.command(positions, duration_ms)
        self.expected.commanded(sent_s, positions, arrive_s)
        self._row(phase_key, "command", " ".join(map(str, positions)), duration_ms, commanded=positions)
        return sent_s

    # --- Moves without measurements (getting into position) ---

    def reposition(self, positions: dict, duration_s: float, label: str) -> None:
        self.bus.wait_gap()
        self._command("-", positions, time.perf_counter() + duration_s)
        self._row("-", "note", note=label)
        time.sleep(duration_s + 0.5)

    # --- Phases ---

    def run(self, phase: Phase) -> PhaseStats:
        stats = PhaseStats()
        self.reposition(self.poses["A"], 2.0, f"to pose A before {phase.key}")
        duration = {
            "still": f"~{STILL_READS} reads",
            "hold": f"{HOLD_S:g} s",
            "move": f"{phase.moves} sweeps, {phase.moves * MOVE_S:g} s",
        }[phase.kind]
        watch = {"still": WATCH_STILL, "hold": WATCH_HOLD, "move": WATCH_MOVE.format(moves=phase.moves)}[phase.kind]
        print(f"\n=== {phase.key}: {phase.title} ({duration}) ===")
        print(f"    {watch}")
        stats.volts = (self.bus.voltage(), None)
        time.sleep(1.0)
        print("    ...running", flush=True)
        try:
            if phase.kind == "still":
                self._run_still(phase, stats)
            elif phase.kind == "hold":
                self._run_ticks(phase, stats, [self.poses["A"]] * 2, HOLD_S)
            else:
                ends = [self.poses["B"], self.poses["A"]]
                for move in range(phase.moves):
                    path = [self.poses["A"], self.poses["B"]] if move % 2 == 0 else [self.poses["B"], self.poses["A"]]
                    self._run_ticks(phase, stats, path, MOVE_S)
                    stats.lags.append(self._arrival_lag(phase, stats, ends[move % 2]))
        except KeyboardInterrupt:
            stats.aborted = True
            print("    aborted (Ctrl-C)")
        time.sleep(0.3)
        self.bus.wait_gap()
        stats.volts = (stats.volts[0], self.bus.voltage())
        if self.prompt and not stats.aborted:
            try:
                stats.observation = input("    What did you see/hear? (Enter = smooth, no beeps): ").strip() or "smooth, no beeps"
            except EOFError:
                pass
        return stats

    def _run_still(self, phase: Phase, stats: PhaseStats) -> None:
        for _ in range(STILL_READS):
            self._tick_reads(phase, stats, tick_s=time.perf_counter(), period_s=0.0)

    def _run_ticks(self, phase: Phase, stats: PhaseStats, path: list, duration_s: float) -> None:
        """Stream `path` (start, end raw positions) over duration_s at the phase's
        rate: each tick commands where the arm should be one tick later, so it
        arrives on schedule, then reads. Late ticks don't catch up -- the next
        command just aims further along.
        """
        start, end = path
        period_s = 1 / phase.rate_hz
        self.bus.wait_gap()
        start_s = next_tick_s = time.perf_counter()
        stats.command_times.append([])
        tick = 0
        while True:
            now_s = time.perf_counter()
            if now_s < next_tick_s:
                time.sleep(next_tick_s - now_s)
            elif (now_s - next_tick_s) * 1000 > LATE_TICK_MS:
                stats.late_ticks += 1
                next_tick_s = now_s
            tick_s = next_tick_s
            next_tick_s += period_s
            target_s = min(tick_s - start_s + period_s, duration_s)
            fraction = target_s / duration_s
            positions = {i: round(start[i] + fraction * (end[i] - start[i])) for i in start}
            if phase.gap_before_command:
                self.bus.wait_gap()
            stats.command_times[-1].append(self._command(phase.key, positions, start_s + target_s))
            if tick % phase.read_every == 0:
                self._tick_reads(phase, stats, tick_s, period_s)
            tick += 1
            if target_s >= duration_s:
                break
        remaining_s = start_s + duration_s - time.perf_counter()
        if remaining_s > 0:
            time.sleep(remaining_s)

    def _tick_reads(self, phase: Phase, stats: PhaseStats, tick_s: float, period_s: float) -> None:
        mode = phase.reads
        if mode == "all":
            self._read(phase, stats, ARM_IDS)
        elif mode == "all_mid":
            remaining_s = tick_s + period_s / 2 - time.perf_counter()
            if remaining_s > 0:
                time.sleep(remaining_s)
            self._read(phase, stats, ARM_IDS)
        elif mode == "all_gap":
            self.bus.wait_gap()
            self._read(phase, stats, ARM_IDS)
        elif mode == "all_safe":
            self._read(phase, stats, ARM_IDS, safe=True)
        elif mode == "all_safe_mid":
            remaining_s = tick_s + period_s / 2 - time.perf_counter()
            if remaining_s > 0:
                time.sleep(remaining_s)
            self._read(phase, stats, ARM_IDS, safe=True)
        elif mode == "singles":
            for servo_id in ARM_IDS:
                self._read(phase, stats, [servo_id])
        elif mode == "singles_safe":
            for servo_id in ARM_IDS:
                self.bus.wait_gap()
                self._read(phase, stats, [servo_id], safe=True)
        elif mode == "rotate":
            self._read(phase, stats, [ARM_IDS[self.rotation]])
            self.rotation = (self.rotation + 1) % len(ARM_IDS)
        elif mode == "fit":
            # Only start a read if even a timed-out one ends before the next tick.
            while time.perf_counter() + (SAFE_TIMEOUT_MS + 5) / 1000 <= tick_s + period_s:
                self._read(phase, stats, [ARM_IDS[self.rotation]], safe=True)
                self.rotation = (self.rotation + 1) % len(ARM_IDS)

    def _arrival_lag(self, phase: Phase, stats: PhaseStats, target: dict) -> dict:
        """|measured - target| (deg) per joint, read the safe way right after
        the move should have arrived; None where the read failed.
        """
        lag = {}
        for servo_id in ARM_IDS:
            self.bus.wait_gap()
            result = self._read(phase, stats, [servo_id], safe=True, kind="lag")
            position = result.positions.get(servo_id)
            lag[servo_id] = None if position is None else abs(raw_to_deg(servo_id, position) - raw_to_deg(servo_id, target[servo_id]))
        return lag


# --- Reporting ---


def percentile(values: list, q: float) -> float:
    return float(np.percentile(values, q)) if values else math.nan


def worst_lag(lag: dict) -> float:
    values = [v for v in lag.values() if v is not None]
    return max(values) if len(values) == len(lag) else math.nan


def summarize(phase: Phase, stats: PhaseStats, baseline_lag: float | None) -> tuple:
    """Detail lines for the log, and the phase's row in the final table."""
    lines = [f"{phase.key}: {phase.title}", f"  question: {phase.question}"]
    reads = stats.reads
    outcomes = {}
    for result in reads:
        outcomes[result.outcome] = outcomes.get(result.outcome, 0) + 1
    failed = sum(n for outcome, n in outcomes.items() if outcome != "ok")
    stale = sum(1 for r in reads if r.outcome != "timeout" and r.latency_ms < STALE_MS_PER_SERVO * len(r.requested))
    drained = sum(r.drained for r in reads)
    ok_latency = [r.latency_ms for r in reads if r.outcome == "ok"]
    if reads:
        lines.append(
            f"  reads: {len(reads)}, failed {failed} ({100 * failed / len(reads):.0f}%) "
            + ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items()))
        )
        lines.append(
            f"  reply latency (ok reads): median {percentile(ok_latency, 50):.1f} ms, "
            f"p95 {percentile(ok_latency, 95):.1f} ms, max {max(ok_latency, default=math.nan):.1f} ms"
        )
        lines.append(f"  likely stale replies (< {STALE_MS_PER_SERVO:g} ms per servo asked): {stale}; stale replies drained (safe reads): {drained}")
        partial_missing = {}
        for result in reads:
            if result.outcome == "partial":
                ids = [i for i in ARM_IDS if i not in result.returned_ids]
                key = " ".join(map(str, ids))
                partial_missing[key] = partial_missing.get(key, 0) + 1
        if partial_missing:
            lines.append("  partial replies, servos missing: " + "; ".join(f"[{k}] x{v}" for k, v in partial_missing.items()))
        tracking = {
            JOINT_BY_ID[i].name: percentile(np.abs(v).tolist(), 50) for i, v in stats.tracking.items() if v
        }
        if phase.kind == "move" and tracking:
            lines.append("  median |measured - expected| during sweeps: " + ", ".join(f"{k} {v:.1f}" for k, v in tracking.items()) + " deg")
    else:
        lines.append("  reads: none")

    achieved_hz = math.nan
    max_gap_ms = math.nan
    # Within each sweep only: the pauses between sweeps (lag reads) aren't stalls.
    gaps = np.concatenate([np.diff(times) * 1000 for times in stats.command_times] or [[]])
    if len(gaps):
        max_gap_ms = float(gaps.max())
        achieved_hz = 1000 / float(np.median(gaps))
        lines.append(
            f"  commands: {sum(map(len, stats.command_times))} @ {phase.rate_hz:g} Hz planned, median achieved {achieved_hz:.1f} Hz, "
            f"longest gap {max_gap_ms:.0f} ms, late ticks {stats.late_ticks}"
        )

    mean_lag = math.nan
    if stats.lags:
        per_move = [worst_lag(lag) for lag in stats.lags]
        mean_lag = float(np.nanmean(per_move)) if not all(math.isnan(v) for v in per_move) else math.nan
        lines.append("  worst-joint lag at planned arrival, per move: " + ", ".join(f"{v:.1f}" for v in per_move) + f" deg (mean {mean_lag:.1f})")
        by_joint = {
            JOINT_BY_ID[i].name: np.nanmean([lag[i] if lag[i] is not None else math.nan for lag in stats.lags]) for i in ARM_IDS
        }
        lines.append("  mean lag by joint: " + ", ".join(f"{k} {v:.1f}" for k, v in by_joint.items()) + " deg")
        if baseline_lag is not None and not math.isnan(baseline_lag):
            lines.append(f"  vs no-reads baseline at this rate: {mean_lag - baseline_lag:+.1f} deg")

    volts = ", ".join("?" if v is None else f"{v:.3f}" for v in stats.volts)
    lines.append(f"  battery voltage before, after: {volts} V")
    if stats.aborted:
        lines.append("  ABORTED partway (Ctrl-C)")
    lines.append(f"  observed: {stats.observation or '(not recorded)'}")

    fail_pct = f"{100 * failed / len(reads):.0f}%" if reads else "-"
    row = (
        phase.key,
        f"{phase.rate_hz:g}" if phase.rate_hz else "-",
        phase.reads,
        str(len(reads)) if reads else "-",
        fail_pct,
        str(stale) if reads else "-",
        f"{percentile(ok_latency, 50):.1f}" if ok_latency else "-",
        f"{achieved_hz:.1f}" if not math.isnan(achieved_hz) else "-",
        f"{max_gap_ms:.0f}" if not math.isnan(max_gap_ms) else "-",
        f"{mean_lag:.1f}" if not math.isnan(mean_lag) else "-",
        stats.observation,
    )
    return lines, row


def table(rows: list) -> list:
    header = ("phase", "Hz", "reads", "n", "fail", "stale", "lat ms", "cmd Hz", "max gap", "lag deg", "observed")
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header) - 1)]
    fmt = lambda r: "  ".join(str(v).ljust(w) for v, w in zip(r, widths)) + "  " + r[-1]
    return [fmt(header), fmt(tuple("-" * w for w in widths) + ("-" * 8,)), *map(fmt, rows)]


def plan_poses() -> dict:
    """Raw servo positions for home (all joints 0), pose A and pose B."""
    ik = GripperPoseController(dry_run=True, rate_hz=1000)
    home = dict(ik._commands_for({joint.name: 0.0 for joint in config.JOINTS}, GRIPPER_OPEN_FRACTION))
    poses = {"home": home}
    for name, (x, y, z, tilt, roll) in (("A", POSE_A), ("B", POSE_B)):
        with contextlib.redirect_stdout(io.StringIO()):
            angles = ik.move_to_pose(
                x, y, z, approach=approach_vector(x, y, tilt), roll_deg=roll,
                gripper_open_fraction=GRIPPER_OPEN_FRACTION, duration_ms=1, linear=False,
            )
        poses[name] = dict(ik._commands_for(ik._angles_deg_by_name(angles), GRIPPER_OPEN_FRACTION))
    return poses


def next_log_number() -> int:
    numbers = [int(m.group(1)) for f in LOG_DIR.glob("stress_test*.log") if (m := re.fullmatch(r"stress_test(\d+)\.log", f.name))]
    return max(numbers, default=0) + 1


class Tee:
    """print() to the terminal and the log file at once."""

    def __init__(self, file):
        self.file, self.terminal = file, sys.stdout

    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="Use the real arm (default: dry run against a fake one).")
    parser.add_argument("--robot", help="Which robot this is, e.g. 'my copy' or 'lab copy' (asked for if omitted).")
    parser.add_argument("--phases", help="Comma-separated phases or groups to run (default: all), e.g. M1,V3 or T,W.")
    parser.add_argument("--no-prompt", action="store_true", help="Don't ask for observations after each phase.")
    args = parser.parse_args()

    phases = PHASES
    if args.phases:
        keys = [k.strip().upper() for k in args.phases.split(",")]
        # A bare letter (e.g. T) selects that whole group.
        selected = lambda p, k: p.key == k or p.key[0] == k
        unknown = [k for k in keys if not any(selected(p, k) for p in PHASES)]
        if unknown:
            parser.error(f"unknown phase(s) {unknown}; known: {[p.key for p in PHASES]}")
        phases = [p for p in PHASES if any(selected(p, k) for k in keys)]
    prompt = not args.no_prompt and sys.stdin.isatty()
    robot = args.robot or (input("Which robot is this (e.g. 'my copy', 'lab copy')? ").strip() if prompt else "") or "unlabeled"
    setup = input("Anything notable about the setup (power supply, cable, USB port)? ").strip() if prompt else ""

    poses = plan_poses()
    bus = Bus(execute=args.execute)
    LOG_DIR.mkdir(exist_ok=True)
    number = next_log_number()
    log_path = LOG_DIR / f"stress_test{number}.log"
    csv_path = LOG_DIR / f"stress_test{number}_requests.csv"
    with open(log_path, "x") as log_file, open(csv_path, "x", newline="") as csv_file:
        sys.stdout = Tee(log_file)
        requests_csv = csv.writer(csv_file)
        requests_csv.writerow(
            ["phase", "time_s", "kind", "servo_ids", "duration_ms", "latency_ms", "outcome", "reply_ids", "measured_raw", "expected_raw", "note", "commanded_raw"]
        )
        try:
            print(f"xArm 1S read stress test -- {'REAL ARM' if args.execute else 'DRY RUN (fake robot)'}")
            print(f"date: {datetime.now().isoformat(timespec='seconds')}")
            print(f"robot: {robot}")
            print(f"setup notes: {setup or '(none)'}")
            print(f"board USB serial: {bus.serial_number()}")
            print(f"host: {platform.node()}, {platform.platform()}, python {platform.python_version()}")
            print(f"xarm {version('xarm')}, hidapi {version('hidapi')}")
            print(f"pose A {POSE_A} -> raw {poses['A']}")
            print(f"pose B {POSE_B} -> raw {poses['B']}")
            print(
                f"stock read: first report within {STOCK_TIMEOUT_MS} ms, no drain, no ID check; "
                f"safe read: drain, {SAFE_TIMEOUT_MS} ms timeout, skip non-position replies; gap {GAP_MS} ms"
            )
            print(f"requests: {csv_path.name}")
            print(f"battery voltage: {bus.voltage()} V")

            runner = Runner(bus, poses, requests_csv, prompt)
            print("\nMoving to home (upright), then pose A...")
            runner.reposition(poses["home"], 3.0, "home")

            results = []
            baselines = {}
            for phase in phases:
                try:
                    stats = runner.run(phase)
                except KeyboardInterrupt:  # e.g. at the observation prompt
                    print("\n    aborted (Ctrl-C)")
                    break
                if phase.kind == "move" and phase.reads == "none" and stats.lags:
                    baselines[phase.rate_hz] = float(np.nanmean([worst_lag(lag) for lag in stats.lags]))
                results.append((phase, stats))
                if stats.aborted:
                    break

            print("\nReturning home...")
            runner.reposition(poses["home"], 3.0, "home")
            if prompt:
                try:
                    closing = input("Any overall notes (e.g. which phases beeped most, anything odd)? ").strip()
                except (KeyboardInterrupt, EOFError):  # still write the summary below
                    closing = ""
                if closing:
                    print(f"overall notes: {closing}")

            print("\n" + "=" * 70 + "\nPHASE DETAILS\n")
            rows = []
            for phase, stats in results:
                lines, row = summarize(phase, stats, baselines.get(phase.rate_hz) if phase.reads != "none" else None)
                print("\n".join(lines) + "\n")
                rows.append(row)
            print("=" * 70 + "\nSUMMARY (fail = read failure rate; stale = replies that arrived "
                  f"< {STALE_MS_PER_SERVO:g} ms per servo asked; lat = median ok-read latency; max gap = longest gap "
                  "between commands within a sweep; lag = mean worst-joint lag at planned arrival)\n")
            print("\n".join(table(rows)))
        finally:
            sys.stdout = sys.stdout.terminal if isinstance(sys.stdout, Tee) else sys.stdout
    print(f"\nLogged to {log_path} and {csv_path}")


if __name__ == "__main__":
    main()
