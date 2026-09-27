"""Run a sequence of gripper waypoints from a CSV file.

Each row: x, y, z, tilt_deg, roll_deg, gripper_close, time_scale, tolerance_m

x, y, z: target gripper position (m), base frame (see controller.py for
the axis convention).

tilt_deg: lean from straight down (0) toward (x, y), degrees -- 90 points
straight at (x, y). Blank/"any" leaves it unconstrained: the solver picks
whatever's easiest, also freeing 2 joints and reaching more targets.

roll_deg: wrist_roll angle (jaw orientation) -- independent of tilt_deg,
since tilt is 2 of the position IK's DOF and roll is the 1 it never
touches, so it's commanded directly. Blank/"any" leaves it as last set.

gripper_close: 0.0 open, 1.0 closed, partial values in between.

time_scale: move duration as a multiple of --base-time-ms (default 1000),
e.g. 1.0 = 1000 ms, 1.5 = 1500 ms.

tolerance_m: max IK position miss (m) before the row is rejected as
unreachable. Blank/"any" uses the config default; "inf" accepts any
solved position -- useful for best-effort/exploratory rows.

Blank lines and lines starting with '#' are skipped. Defaults to
--dry-run, printing servo commands instead of moving the real arm (link
lengths in xarm_ik/config.py are only as accurate as their source URDF).

Usage:
    uv run python scripts/run_csv.py waypoints.csv             # dry run (safe)
    uv run python scripts/run_csv.py waypoints.csv --execute    # moves the real arm
    uv run python scripts/run_csv.py waypoints.csv --sim        # also show it in PyBullet
    uv run python scripts/run_csv.py waypoints.csv --base-time-ms 500   # run twice as fast
"""

import argparse
import csv
import math
import sys

from xarm_ik import GripperPoseController
from xarm_ik.sim import PybulletArmView


def approach_vector(x: float, y: float, tilt_deg: float) -> tuple:
    """Convert tilt_deg into move_to_pose's approach vector: leans tilt_deg
    away from straight down, horizontally toward (x, y) (only its direction
    matters, not distance); (x,y)=(0,0) has none, so leans +Y arbitrarily.
    """
    tilt = math.radians(tilt_deg)
    horizontal = math.sin(tilt)
    radius = math.hypot(x, y)
    lean_x, lean_y = (x / radius, y / radius) if radius > 0 else (0.0, -1.0)
    return (horizontal * lean_x, horizontal * lean_y, -math.cos(tilt))


def tilt_deg_of(approach: tuple) -> float:
    """Inverse of approach_vector: angle (deg) between `approach` and
    straight down (0, 0, -1), regardless of horizontal lean.
    """
    _, _, az = approach
    return math.degrees(math.acos(max(-1.0, min(1.0, -az))))


_UNSPECIFIED = {"", "any", "none"}


def _parse_optional_float(value: str):
    """Return None for a blank/"any"/"none" field, else the parsed float.

    float() already accepts "inf"/"infinity" (any case, optional sign),
    which is all tolerance_m needs for an unlimited tolerance.
    """
    return None if value.strip().lower() in _UNSPECIFIED else float(value)


def read_waypoints(path):
    waypoints = []
    with open(path, newline="") as f:
        reader = csv.reader(f, skipinitialspace=True)
        for line_num, row in enumerate(reader, start=1):
            if not row or not row[0].strip() or row[0].strip().startswith("#"):
                continue
            if len(row) != 8:
                raise ValueError(
                    f"{path}:{line_num}: expected 8 values "
                    f"(x, y, z, tilt_deg, roll_deg, gripper_close, time_scale, "
                    f"tolerance_m), got {len(row)}"
                )
            x_str, y_str, z_str, tilt_str, roll_str, gripper_str, time_scale_str, tolerance_str = row
            try:
                x, y, z = float(x_str), float(y_str), float(z_str)
                tilt_deg = _parse_optional_float(tilt_str)
                roll_deg = _parse_optional_float(roll_str)
                gripper_close, time_scale = float(gripper_str), float(time_scale_str)
                tolerance_m = _parse_optional_float(tolerance_str)
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
                (x, y, z, tilt_deg, roll_deg, gripper_close, time_scale, tolerance_m)
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
            "time_scale, tolerance_m waypoints"
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
    args = parser.parse_args()
    if args.base_time_ms <= 0:
        parser.error(f"--base-time-ms must be positive, got {args.base_time_ms}")

    try:
        waypoints = read_waypoints(args.csv_path)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    arm = GripperPoseController(com_port=args.com_port, dry_run=not args.execute)
    view = PybulletArmView() if args.sim else None

    print("Homing...")
    arm.home(wait=True)
    if view:
        view.animate_to(arm.get_joint_angles_deg(), arm.get_gripper_open_fraction(), duration_s=1.0)

    for row_num, (x, y, z, tilt_deg, roll_deg, gripper_close, time_scale, tolerance_m) in enumerate(
        waypoints, start=1
    ):
        duration_ms = max(1, int(round(time_scale * args.base_time_ms)))
        open_fraction = 1.0 - gripper_close
        tilt_desc = "unconstrained tilt" if tilt_deg is None else f"tilt={tilt_deg:.0f} deg"
        roll_desc = "roll unchanged" if roll_deg is None else f"roll={roll_deg:.0f} deg"
        tolerance_desc = "default tolerance" if tolerance_m is None else f"tolerance={tolerance_m * 1000:g} mm"
        print(
            f"[{row_num}/{len(waypoints)}] Moving to (x={x:.3f}, y={y:.3f}, z={z:.3f}) m, "
            f"{tilt_desc}, {roll_desc}, {tolerance_desc}, "
            f"gripper={open_fraction * 100:.0f}% open, over {duration_ms} ms"
        )
        approach = None if tilt_deg is None else approach_vector(x, y, tilt_deg)
        try:
            # Roll first (seeds IK; doesn't affect position so it just
            # sticks). Gripper sent non-blocking; move_to_pose blocks for
            # the arm move, which already spans ~duration_ms on its own.
            if roll_deg is not None:
                arm.move_joint("wrist_roll", roll_deg, duration_ms=duration_ms, wait=False)
            arm.set_gripper(open_fraction, duration_ms=duration_ms, wait=False)
            if view:
                # Mirror the gripper immediately -- it's commanded
                # concurrently with the arm move on the real hardware too.
                view.set_gripper(open_fraction)
            arm.move_to_pose(
                x, y, z,
                approach=approach,
                duration_ms=duration_ms,
                wait=True,
                position_tolerance_m=tolerance_m,
                # Replay move_to_pose's own waypoints instead of
                # animate_to's 2-point interpolation, which doesn't know
                # the real Cartesian path and would draw a curved one.
                on_step=(lambda angles, dur_ms: view.step_to(angles, dur_ms / 1000)) if view else None,
            )
        except ValueError as e:
            print(f"Error at row {row_num}: {e}", file=sys.stderr)
            sys.exit(1)

        achieved_x, achieved_y, achieved_z = arm.get_pose()
        achieved_tilt_deg = tilt_deg_of(arm.get_approach_vector())
        achieved_roll_deg = arm.get_joint_angles_deg()["wrist_roll"]
        error_m = math.dist((achieved_x, achieved_y, achieved_z), (x, y, z))
        print(
            f"    Achieved (x={achieved_x:.3f}, y={achieved_y:.3f}, z={achieved_z:.3f}) m, "
            f"tilt={achieved_tilt_deg:.1f} deg, roll={achieved_roll_deg:.1f} deg, "
            f"error={error_m * 1000:.1f} mm"
        )

    print("Done.")
    if view:
        print("Close the PyBullet window to exit.")
        view.wait_until_closed()


if __name__ == "__main__":
    main()
