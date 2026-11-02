"""Hardware constants for the Hiwonder xArm 1S.

Link lengths and joint limits come from xarm_ik/urdf/xarm_v2.urdf (from
https://github.com/jonathanrandall/autonomous_robot_gardener); this adds each
joint's servo ID, its mounting direction, and the raw position <-> angle mapping.
"""

import math
from dataclasses import dataclass

from .kinematics import ACTIVE_LINKS_MASK, build_chain

# The servo bus's hard ceiling; caps the URDF's limits in case it ever exceeds them.
HARDWARE_LIMIT_DEG = 125.0

# Raw positions: 500 is a joint's zero, 0/1000 its own lower/upper limit_deg.
SERVO_POSITION_CENTER = 500
SERVO_POSITION_MIN = 0
SERVO_POSITION_MAX = 1000

# IK results can overshoot a limit by float noise; clamp within this, reject beyond.
JOINT_LIMIT_EPSILON_DEG = 1e-4


@dataclass(frozen=True)
class JointConfig:
    name: str
    servo_id: int
    limit_deg: tuple  # (lower, upper) of the true (URDF) angle
    sign: int = 1  # -1: servo mounted opposite the URDF axis

    @property
    def _servo_limit_deg(self) -> tuple:
        """limit_deg in the servo's own direction (mirrored if sign flips it)."""
        lower_deg, upper_deg = self.limit_deg
        return self.limit_deg if self.sign > 0 else (-upper_deg, -lower_deg)

    def to_position(self, angle_deg: float) -> int:
        """Raw servo position for a joint angle; ValueError outside limit_deg."""
        lower_deg, upper_deg = self.limit_deg
        if not lower_deg - JOINT_LIMIT_EPSILON_DEG <= angle_deg <= upper_deg + JOINT_LIMIT_EPSILON_DEG:
            raise ValueError(f"{self.name} angle {angle_deg:.1f} deg is outside its limits {self.limit_deg}.")
        servo_deg = self.sign * max(lower_deg, min(upper_deg, angle_deg))
        servo_lower, servo_upper = self._servo_limit_deg
        span_deg, span_position = (
            (servo_upper, SERVO_POSITION_MAX - SERVO_POSITION_CENTER)
            if servo_deg >= 0
            else (servo_lower, SERVO_POSITION_MIN - SERVO_POSITION_CENTER)
        )
        position = SERVO_POSITION_CENTER + (servo_deg / span_deg if span_deg else 0.0) * span_position
        return int(round(max(SERVO_POSITION_MIN, min(SERVO_POSITION_MAX, position))))

    def to_angle_deg(self, position: float) -> float:
        """Inverse of to_position (past a limit maps beyond it, unclamped)."""
        servo_lower, servo_upper = self._servo_limit_deg
        offset = position - SERVO_POSITION_CENTER
        if offset >= 0:
            servo_deg = offset / (SERVO_POSITION_MAX - SERVO_POSITION_CENTER) * servo_upper
        else:
            servo_deg = offset / (SERVO_POSITION_MIN - SERVO_POSITION_CENTER) * servo_lower
        return self.sign * servo_deg


# Chain index of each active joint, base to wrist.
ACTIVE_LINK_INDICES = tuple(i for i, active in enumerate(ACTIVE_LINKS_MASK) if active)


def _limit_deg(chain, link_index: int) -> tuple:
    lower_rad, upper_rad = chain.links[link_index].bounds
    return (max(math.degrees(lower_rad), -HARDWARE_LIMIT_DEG), min(math.degrees(upper_rad), HARDWARE_LIMIT_DEG))


_chain = build_chain()
# Base to wrist, in chain order. Base yaw's sign was confirmed on hardware.
JOINTS = tuple(
    JointConfig(name, servo_id, _limit_deg(_chain, link_index), sign)
    for (name, servo_id, sign), link_index in zip(
        [("base_yaw", 6, 1), ("shoulder_pitch", 5, 1), ("elbow_pitch", 4, -1), ("wrist_pitch", 3, -1), ("wrist_roll", 2, 1)],
        ACTIVE_LINK_INDICES,
    )
)

GRIPPER_SERVO_ID = 1
GRIPPER_OPEN_POS = 0
GRIPPER_CLOSED_POS = 1000

# Max IK miss (m) before a target counts as unreachable.
IK_POSITION_TOLERANCE_M = 0.01

# Max spacing (m) of linear moves' IK waypoints, so the tip traces a line, not a joint-space arc.
LINEAR_MOVE_STEP_M = 0.01

# Intermediate waypoints only: cap scipy evals per solve (they start near the
# answer), and bias each toward the last one's branch (cut a ~218 deg elbow
# flip to ~9 deg; not monotonic, so re-sweep if retuning).
LINEAR_STEP_OPTIMIZER_BUDGET = 100
LINEAR_STEP_REGULARIZATION = 0.005

# A tick starting later than this counts as late.
TICK_LATE_SLACK_S = 0.010

# Closed loop: nudge a joint only once its tracking error changes less than
# this per tick (it has stopped catching up); each nudge lasts SETTLE_NUDGE_S
# (at most a tick); the total nudge per joint is capped, so a blocked joint
# isn't pushed hard.
SETTLE_STEADY_DEG = 0.5
SETTLE_NUDGE_S = 0.1
SETTLE_MAX_CORRECTION_DEG = 5.0

# Single-servo read: healthy replies take 6-24 ms, so 30 ms only cuts off real
# failures; a tick budgets 25 ms per read when fitting another one in.
FEEDBACK_READ_TIMEOUT_MS = 30
FEEDBACK_READ_BUDGET_MS = 25

# Min wait after a read reply or command before the next command: the board
# drops commands sent within ~5 ms of a reply (stress tests).
COMMAND_GAP_MS = 20
