"""Exercise the arm's range of motion.

Homes, cycles the gripper, homes again, then sweeps each joint (servos
2-6) from its lower limit to upper limit and back to 0 deg, bypassing IK
-- this exercises each servo's own range, not a Cartesian target.

Usage:
    uv run python scripts/range_of_motion.py            # dry run (safe)
    uv run python scripts/range_of_motion.py --execute   # moves the real arm
    uv run python scripts/range_of_motion.py --sim       # also show it in PyBullet
"""

import argparse

from xarm_ik import GripperPoseController, config
from xarm_ik.sim import PybulletArmView

MOTION_DURATION_MS = 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Send commands to the real arm over USB instead of dry-running.",
    )
    parser.add_argument("--com-port", default="USB", help="Passed to xarm.Controller (default: USB)")
    parser.add_argument("--sim", action="store_true", help="Also open a PyBullet window and mirror the motion there.")
    args = parser.parse_args()

    arm = GripperPoseController(com_port=args.com_port, dry_run=not args.execute)
    view = PybulletArmView() if args.sim else None
    on_tick = view.mirror_tick if view else None

    for label, step in (("Homing", arm.home), ("Opening gripper", arm.open_gripper), ("Closing gripper", arm.close_gripper), ("Homing", arm.home)):
        print(f"{label}...")
        step(duration_ms=MOTION_DURATION_MS, on_tick=on_tick)

    for joint in sorted(config.JOINTS, key=lambda joint: joint.servo_id):
        lower_deg, upper_deg = joint.limit_deg
        for label, angle_deg in (("lower limit", lower_deg), ("upper limit", upper_deg), ("zero", 0.0)):
            print(f"{joint.name} (servo {joint.servo_id}): moving to {label} ({angle_deg:.0f} deg)")
            arm.move_joint(joint.name, angle_deg, duration_ms=MOTION_DURATION_MS, on_tick=on_tick)

    print("Done.")
    if view:
        print("Close the PyBullet window to exit.")
        view.wait_until_closed()


if __name__ == "__main__":
    main()
