"""Run a sequence of gripper waypoints from a CSV file.

Each row: x, y, z, tilt_deg, roll_deg, gripper_close, time_scale, tolerance_m, linear
  x, y, z        target tip position (m), base frame (see controller.py)
  tilt_deg       lean from straight down (0) toward (x, y); 90 = horizontal.
                 Blank/"any": unconstrained (frees 2 joints, reaches more)
  roll_deg       wrist roll (jaw orientation). Blank/"any": unchanged
  gripper_close  0.0 open .. 1.0 closed
  time_scale     duration as a multiple of --base-time-ms (default 1000)
  tolerance_m    max IK miss (m). Blank/"any": config default; "inf": accept any
  linear         1: straight-line tip path; 0: joint-space arc. Blank/"any": 1
Blank lines and '#' lines are skipped. --start-line N skips rows above file
line N; after homing, the arm moves straight to the first row kept.

Dry runs by default (prints servo commands); --execute moves the real arm.
Each row runs at --rate-hz (a command, then single-joint reads, per tick);
--closed-loop then corrects until within --settle-tolerance-deg, warning past
--settle-timeout-scale x the row's duration.

--log (or --log-prefix NAME) writes logs/[NAME_]run<N>.csv: every joint read's
time_s, row (0 = homing), joint, expected_deg and measured_deg (blank = failed).
run<N>_events.csv has a `move` per row (duration, settling included); closed
loop adds a `settle` per row (count = corrections) and a `correction` per nudge
(until the next one or settled); last, `sequence` totals rows 1+.

Usage:
    uv run python scripts/run_csv.py waypoints.csv             # dry run (safe)
    uv run python scripts/run_csv.py waypoints.csv --execute    # moves the real arm
    uv run python scripts/run_csv.py waypoints.csv --sim        # also show it in PyBullet
    uv run python scripts/run_csv.py waypoints.csv --base-time-ms 500   # twice as fast
    uv run python scripts/run_csv.py waypoints.csv --execute --closed-loop --log-prefix fast_cups
    uv run python scripts/run_csv.py waypoints.csv --execute --start-line 80
"""

import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path

from xarm_ik import GripperPoseController
from xarm_ik.sim import PybulletArmView


def approach_vector(x: float, y: float, tilt_deg: float) -> tuple:
    """move_to_pose's approach for a lean of tilt_deg from straight down toward
    (x, y) (toward -Y at the origin).
    """
    tilt = math.radians(tilt_deg)
    horizontal = math.sin(tilt)
    radius = math.hypot(x, y)
    lean_x, lean_y = (x / radius, y / radius) if radius > 0 else (0.0, -1.0)
    return (horizontal * lean_x, horizontal * lean_y, -math.cos(tilt))


def tilt_deg_of(approach: tuple) -> float:
    """Inverse of approach_vector: the angle from straight down."""
    _, _, az = approach
    return math.degrees(math.acos(max(-1.0, min(1.0, -az))))


class FeedbackLog:
    """The controller's on_feedback: writes each joint read, tagged with `row`
    (set it per row), and timings to <stem>_events.csv (see log_move).
    """

    def __init__(self, path):
        self.events_path = path.with_name(f"{path.stem}_events.csv")
        self._file = open(path, "x", newline="")  # never overwrite an earlier run
        self._events_file = open(self.events_path, "x", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["time_s", "row", "joint", "expected_deg", "measured_deg"])
        self._events = csv.writer(self._events_file)
        self._events.writerow(["event", "row", "line", "start_s", "duration_s", "count", "detail"])
        self._start_s = time.perf_counter()
        self.row = 0
        self._sequence_start_s = self._sequence_end_s = None
        self._closed_loop = False
        self._corrections = 0
        self._correction_s = 0.0  # first nudge to settled, summed over rows
        self._settle_s = 0.0

    def _event(self, event, row, line, start_s, duration_s, count="", detail=""):
        self._events.writerow([event, row, line, f"{start_s - self._start_s:.4f}", f"{duration_s:.4f}", count, detail])

    def log_move(self, row, line, planned_ms, timing):
        """Log a row's `move`, and closed loop its `settle` and `correction`s.
        Row 0 (homing) is left out of the `sequence` totals.
        """
        self._event("move", row, line, timing.start_s, timing.end_s - timing.start_s, detail=f"planned {planned_ms} ms")
        if timing.settled is not None:
            self._closed_loop = True
            settle_s = timing.end_s - timing.motion_end_s
            result = "settled" if timing.settled else "timed out"
            self._event("settle", row, line, timing.motion_end_s, settle_s, len(timing.corrections), result)
            ends = [t for t, _ in timing.corrections[1:]] + [timing.end_s]
            for (t, joints), end_s in zip(timing.corrections, ends):
                self._event("correction", row, line, t, end_s - t, detail=" ".join(joints))
            if row:
                self._corrections += len(timing.corrections)
                self._settle_s += settle_s
                if timing.corrections:
                    self._correction_s += timing.end_s - timing.corrections[0][0]
        if row:
            if self._sequence_start_s is None:
                self._sequence_start_s = timing.start_s
            self._sequence_end_s = timing.end_s

    def __call__(self, joint_name, sample):
        measured = "" if sample.measured_deg is None else f"{sample.measured_deg:.2f}"
        self._writer.writerow(
            [f"{sample.time_s - self._start_s:.4f}", self.row, joint_name, f"{sample.expected_deg:.2f}", measured]
        )

    def close(self):
        """Write the `sequence` event (rows 1+) and close; returns it as text."""
        summary = "no rows run"
        if self._sequence_start_s is not None:
            total_s = self._sequence_end_s - self._sequence_start_s
            summary = f"sequence took {total_s:.2f} s"
            if self._closed_loop:
                detail = f"correcting {self._correction_s:.3f} s, settling {self._settle_s:.3f} s"
                self._event("sequence", "", "", self._sequence_start_s, total_s, self._corrections, detail)
                summary += f", {self._corrections} correction(s) over {self._correction_s:.2f} s, {self._settle_s:.2f} s settling"
            else:
                self._event("sequence", "", "", self._sequence_start_s, total_s)
        self._file.close()
        self._events_file.close()
        return summary


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def next_log_path(prefix: str = "") -> Path:
    """logs/[<prefix>_]run<N>.csv; N counts up across all prefixes."""
    numbers = [
        int(m.group(1)) for f in LOG_DIR.glob("*run*.csv") if (m := re.fullmatch(r"(?:.+_)?run(\d+)\.csv", f.name))
    ]
    return LOG_DIR / f"{prefix + '_' if prefix else ''}run{max(numbers, default=0) + 1}.csv"


_UNSPECIFIED = {"", "any", "none"}


def _parse_optional_float(value: str):
    """None for blank/"any"/"none", else the float ("inf" included)."""
    return None if value.strip().lower() in _UNSPECIFIED else float(value)


def _parse_linear(value: str) -> bool:
    value = value.strip().lower()
    if value in _UNSPECIFIED or value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    raise ValueError(f"linear must be 1/0/true/false, got {value!r}")


def read_waypoints(path):
    waypoints = []
    with open(path, newline="") as f:
        reader = csv.reader(f, skipinitialspace=True)
        for line_num, row in enumerate(reader, start=1):
            if not row or not row[0].strip() or row[0].strip().startswith("#"):
                continue
            if len(row) != 9:
                raise ValueError(
                    f"{path}:{line_num}: expected 9 values "
                    f"(x, y, z, tilt_deg, roll_deg, gripper_close, time_scale, "
                    f"tolerance_m, linear), got {len(row)}"
                )
            x_str, y_str, z_str, tilt_str, roll_str, gripper_str, time_scale_str, tolerance_str, linear_str = row
            try:
                x, y, z = float(x_str), float(y_str), float(z_str)
                tilt_deg = _parse_optional_float(tilt_str)
                roll_deg = _parse_optional_float(roll_str)
                gripper_close, time_scale = float(gripper_str), float(time_scale_str)
                tolerance_m = _parse_optional_float(tolerance_str)
                linear = _parse_linear(linear_str)
            except ValueError as e:
                raise ValueError(f"{path}:{line_num}: {e}") from e
            if not 0.0 <= gripper_close <= 1.0:
                raise ValueError(
                    f"{path}:{line_num}: gripper_close must be between 0.0 and 1.0, "
                    f"got {gripper_close}"
                )
            if time_scale <= 0:
                raise ValueError(f"{path}:{line_num}: time_scale must be positive, got {time_scale}")
            if tolerance_m is not None and tolerance_m < 0:
                raise ValueError(f"{path}:{line_num}: tolerance_m must not be negative, got {tolerance_m}")
            waypoints.append(
                (line_num, (x, y, z, tilt_deg, roll_deg, gripper_close, time_scale, tolerance_m, linear))
            )
    return waypoints


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "csv_path",
        help=(
            "CSV file of x, y, z, tilt_deg, roll_deg, gripper_close, "
            "time_scale, tolerance_m, linear waypoints"
        ),
    )
    parser.add_argument(
        "--base-time-ms",
        type=float,
        default=1000.0,
        help=(
            "Milliseconds that a time_scale of 1.0 in the CSV corresponds to; "
            "each row's duration is time_scale * this (default: 1000)."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Send commands to the real arm over USB instead of dry-running.",
    )
    parser.add_argument("--com-port", default="USB", help="Passed to xarm.Controller (default: USB)")
    parser.add_argument("--sim", action="store_true", help="Also open a PyBullet window and mirror the motion there.")
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=10.0,
        help="Control loop rate: one command and single-joint reads (as many as fit) per tick (default: 10)",
    )
    parser.add_argument(
        "--closed-loop",
        action="store_true",
        help="After each row, read joint positions back and correct until settled on the target.",
    )
    parser.add_argument("--kp", type=float, default=0.5, help="Closed-loop settle correction gain (default: 0.5)")
    parser.add_argument(
        "--settle-tolerance-deg",
        type=float,
        default=2.0,
        help="Closed-loop: max per-joint error (deg) to count a row as done (default: 2.0)",
    )
    parser.add_argument(
        "--settle-timeout-scale",
        type=float,
        default=1.5,
        help="Closed-loop: give up settling (with a warning) once a row has taken this x its duration (default: 1.5)",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Write every joint read (expected vs measured angle) to logs/run<N>.csv, the next free N.",
    )
    parser.add_argument(
        "--log-prefix",
        default="",
        help="Label the log: logs/<prefix>_run<N>.csv instead (implies --log).",
    )
    parser.add_argument(
        "--start-line",
        type=int,
        default=1,
        help="Skip waypoints above this CSV file line number (as shown in an editor); default: 1",
    )
    args = parser.parse_args()
    if args.base_time_ms <= 0:
        parser.error(f"--base-time-ms must be positive, got {args.base_time_ms}")
    if args.rate_hz <= 0:
        parser.error(f"--rate-hz must be positive, got {args.rate_hz}")
    if args.settle_timeout_scale < 1:
        parser.error(f"--settle-timeout-scale must be >= 1, got {args.settle_timeout_scale}")

    try:
        waypoints = read_waypoints(args.csv_path)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    waypoints = [(line_num, w) for line_num, w in waypoints if line_num >= args.start_line]
    if not waypoints:
        parser.error(f"no waypoints at or after line {args.start_line} of {args.csv_path}")

    log = log_path = None
    if args.log_prefix and not re.fullmatch(r"[\w.-]+", args.log_prefix):
        parser.error(f"--log-prefix may only use letters, digits, '_', '-' and '.', got {args.log_prefix!r}")
    if args.log or args.log_prefix:
        log_path = next_log_path(args.log_prefix)
        try:
            LOG_DIR.mkdir(exist_ok=True)
            log = FeedbackLog(log_path)
        except OSError as e:
            parser.error(f"can't open log file: {e}")
        print(f"Logging joint feedback to {log_path}, timings to {log.events_path}")

    try:
        arm = GripperPoseController(
            com_port=args.com_port,
            dry_run=not args.execute,
            rate_hz=args.rate_hz,
            closed_loop=args.closed_loop,
            kp=args.kp,
            settle_tolerance_deg=args.settle_tolerance_deg,
            settle_timeout_scale=args.settle_timeout_scale,
            on_feedback=log,
        )
    except ValueError as e:
        parser.error(str(e))
    view = PybulletArmView() if args.sim else None
    on_tick = view.mirror_tick if view else None

    print("Homing...")
    arm.home(on_tick=on_tick)
    if log:
        log.log_move(0, "", 1000, arm.last_move)

    for row_num, (line_num, (x, y, z, tilt_deg, roll_deg, gripper_close, time_scale, tolerance_m, linear)) in enumerate(
        waypoints, start=1
    ):
        if log:
            log.row = row_num
        duration_ms = max(1, int(round(time_scale * args.base_time_ms)))
        open_fraction = 1.0 - gripper_close
        tilt_desc = "unconstrained tilt" if tilt_deg is None else f"tilt={tilt_deg:.0f} deg"
        roll_desc = "roll unchanged" if roll_deg is None else f"roll={roll_deg:.0f} deg"
        tolerance_desc = "default tolerance" if tolerance_m is None else f"tolerance={tolerance_m * 1000:g} mm"
        print(
            f"[{row_num}/{len(waypoints)}, line {line_num}] Moving to (x={x:.3f}, y={y:.3f}, z={z:.3f}) m, "
            f"{tilt_desc}, {roll_desc}, {tolerance_desc}, {'linear' if linear else 'joint-space'}, "
            f"gripper={open_fraction * 100:.0f}% open, over {duration_ms} ms"
        )
        approach = None if tilt_deg is None else approach_vector(x, y, tilt_deg)
        try:
            arm.move_to_pose(
                x, y, z,
                approach=approach,
                roll_deg=roll_deg,
                gripper_open_fraction=open_fraction,
                duration_ms=duration_ms,
                linear=linear,
                position_tolerance_m=tolerance_m,
                on_tick=on_tick,
            )
        except ValueError as e:
            print(f"Error at row {row_num} (line {line_num}): {e}", file=sys.stderr)
            sys.exit(1)
        if log:
            log.log_move(row_num, line_num, duration_ms, arm.last_move)

        achieved_x, achieved_y, achieved_z = arm.get_pose()
        achieved_tilt_deg = tilt_deg_of(arm.get_approach_vector())
        achieved_roll_deg = arm.get_joint_angles_deg()["wrist_roll"]
        error_m = math.dist((achieved_x, achieved_y, achieved_z), (x, y, z))
        print(
            f"    Achieved (x={achieved_x:.3f}, y={achieved_y:.3f}, z={achieved_z:.3f}) m, "
            f"tilt={achieved_tilt_deg:.1f} deg, roll={achieved_roll_deg:.1f} deg, "
            f"error={error_m * 1000:.1f} mm"
        )
        if args.closed_loop:
            commanded, measured = arm.get_joint_angles_deg(), arm.get_measured_joint_angles_deg()
            print(
                "    Measured joint error "
                + ", ".join(f"{name}={measured[name] - commanded[name]:+.1f}" if name in measured else f"{name}=?"
                    for name in commanded)
                + " deg"
            )

    print("Done.")
    if log:
        summary = log.close()
        print(f"Logged joint feedback to {log_path}, timings to {log.events_path} ({summary})")
    if view:
        print("Close the PyBullet window to exit.")
        view.wait_until_closed()


if __name__ == "__main__":
    main()
