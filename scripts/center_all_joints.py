"""Move all six xArm 1S servos to the center position (500)."""

import xarm

SERVO_IDS = [1, 2, 3, 4, 5, 6]
CENTER_POSITION = 500
MOVE_DURATION_MS = 1000


def main():
    arm = xarm.Controller("USB")

    servos = [(servo_id, CENTER_POSITION) for servo_id in SERVO_IDS]
    arm.setPosition(servos, duration=MOVE_DURATION_MS, wait=True)

    print(f"All servos moved to position {CENTER_POSITION}.")


if __name__ == "__main__":
    main()
