"""Hardware constants for the Hiwonder xArm 1S.

Link lengths and joint limits come from xarm_ik/urdf/xarm_v2.urdf (from
https://github.com/jonathanrandall/autonomous_robot_gardener); this adds each
joint's servo ID, its mounting direction, and the raw position <-> angle mapping.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .kinematics import ACTIVE_LINKS_MASK, build_chain

# The servo bus's hard ceiling; caps the URDF's limits in case it ever exceeds them.
HARDWARE_LIMIT_DEG = 125.0

# Raw positions: 500 is a joint's zero, 0/1000 its own lower/upper limit_deg.
SERVO_POSITION_CENTER = 500
SERVO_POSITION_MIN = 0
SERVO_POSITION_MAX = 1000

# IK results can overshoot a limit by float noise; clamp within this, reject beyond.
JOINT_LIMIT_EPSILON_DEG = 1e-4


# Calibrated servo reading model (scripts/hand_eye_calibration.py): a joint's
# true angle = its servo's nominal angle n (the plain linear position mapping)
# + zero offset + sum_k c_k * (n / CORRECTION_SCALE_DEG)^k, k = 1, 2, ...
# The servos' sensors read a little off in a way that varies over their range
# (a constant offset alone left ~6-7 mm of locating error at the stacker's poses).
CORRECTION_SCALE_DEG = 90.0


def true_from_nominal_deg(nominal_deg: float, offset_deg: float = 0.0, correction_deg: tuple = (),
                          correction_range_deg: tuple | None = None) -> float:
    """The correction is held flat outside `correction_range_deg` (the nominal
    angles the calibration saw): extrapolated, the curve was centimeters off."""
    clamped = nominal_deg if correction_range_deg is None else min(max(nominal_deg, correction_range_deg[0]),
                                                                   correction_range_deg[1])
    x = clamped / CORRECTION_SCALE_DEG
    return nominal_deg + offset_deg + sum(c * x ** (k + 1) for k, c in enumerate(correction_deg))


def nominal_from_true_deg(true_deg: float, offset_deg: float = 0.0, correction_deg: tuple = (),
                          correction_range_deg: tuple | None = None) -> float:
    """Inverse of true_from_nominal_deg (Newton; the correction is small and smooth)."""
    nominal = true_deg - offset_deg
    for _ in range(30):
        error = true_from_nominal_deg(nominal, offset_deg, correction_deg, correction_range_deg) - true_deg
        if abs(error) < 1e-9:
            break
        inside = correction_range_deg is None or correction_range_deg[0] <= nominal <= correction_range_deg[1]
        x = nominal / CORRECTION_SCALE_DEG
        slope = 1.0 + (sum(c * (k + 1) * x ** k for k, c in enumerate(correction_deg)) / CORRECTION_SCALE_DEG
                       if inside else 0.0)
        nominal -= error / slope
    return nominal


@dataclass(frozen=True)
class JointConfig:
    name: str
    servo_id: int
    limit_deg: tuple  # (lower, upper) of the true angle (the servo's range, through the reading model)
    sign: int = 1  # -1: servo mounted opposite the URDF axis
    # Calibrated reading model (see true_from_nominal_deg and JOINT_OFFSETS_PATH).
    # The servo maps its range as before; only the angle each position means changes.
    zero_offset_deg: float = 0.0
    correction_deg: tuple = ()
    correction_range_deg: tuple | None = None  # nominal angles the correction was fitted over

    def true_deg(self, nominal_deg: float) -> float:
        return true_from_nominal_deg(nominal_deg, self.zero_offset_deg, self.correction_deg, self.correction_range_deg)

    def nominal_deg(self, true_deg: float) -> float:
        return nominal_from_true_deg(true_deg, self.zero_offset_deg, self.correction_deg, self.correction_range_deg)

    @property
    def _servo_limit_deg(self) -> tuple:
        """The servo's nominal range (limit_deg through the inverse model), in its own
        direction (mirrored if sign flips it)."""
        lower_deg, upper_deg = (self.nominal_deg(limit) for limit in self.limit_deg)
        return (lower_deg, upper_deg) if self.sign > 0 else (-upper_deg, -lower_deg)

    def to_position(self, angle_deg: float) -> int:
        """Raw servo position for a joint angle; ValueError outside limit_deg."""
        lower_deg, upper_deg = self.limit_deg
        if not lower_deg - JOINT_LIMIT_EPSILON_DEG <= angle_deg <= upper_deg + JOINT_LIMIT_EPSILON_DEG:
            raise ValueError(f"{self.name} angle {angle_deg:.1f} deg is outside its limits {self.limit_deg}.")
        servo_deg = self.sign * self.nominal_deg(max(lower_deg, min(upper_deg, angle_deg)))
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
        return self.true_deg(self.sign * servo_deg)


# Chain index of each active joint, base to wrist.
ACTIVE_LINK_INDICES = tuple(i for i, active in enumerate(ACTIVE_LINKS_MASK) if active)


def _limit_deg(chain, link_index: int) -> tuple:
    lower_rad, upper_rad = chain.links[link_index].bounds
    return (max(math.degrees(lower_rad), -HARDWARE_LIMIT_DEG), min(math.degrees(upper_rad), HARDWARE_LIMIT_DEG))


# The joints' reading model, written by scripts/hand_eye_calibration.py
# (fitted with the hand-eye calibration from the board views):
# {"zero_offsets_deg": {joint: deg}, "corrections_deg": {joint: [c1, c2, ...]},
#  "correction_range_deg": {joint: [lo, hi]}}.
# Without the file, the URDF as-is (offsets 0, no correction).
JOINT_OFFSETS_PATH = Path(__file__).resolve().parent.parent / "joint_offsets.json"


def load_joint_model(path: Path = JOINT_OFFSETS_PATH) -> tuple:
    """({joint: zero offset deg}, {joint: correction coefficients deg}, {joint: (lo, hi) nominal deg it holds over})."""
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}, {}, {}
    return ({name: float(deg) for name, deg in data.get("zero_offsets_deg", {}).items()},
            {name: tuple(float(c) for c in cs) for name, cs in data.get("corrections_deg", {}).items()},
            {name: (float(lo), float(hi)) for name, (lo, hi) in data.get("correction_range_deg", {}).items()})


JOINT_ZERO_OFFSETS_DEG, JOINT_CORRECTIONS_DEG, JOINT_CORRECTION_RANGES_DEG = load_joint_model()

_chain = build_chain()


def _joint(name: str, servo_id: int, sign: int, link_index: int) -> JointConfig:
    offset, correction = JOINT_ZERO_OFFSETS_DEG.get(name, 0.0), JOINT_CORRECTIONS_DEG.get(name, ())
    correction_range = JOINT_CORRECTION_RANGES_DEG.get(name)
    nominal_limits = _limit_deg(_chain, link_index)
    return JointConfig(name, servo_id,
                       tuple(true_from_nominal_deg(limit, offset, correction, correction_range) for limit in nominal_limits),
                       sign, offset, correction, correction_range)


# Base to wrist, in chain order. Base yaw's sign was confirmed on hardware.
JOINTS = tuple(
    _joint(name, servo_id, sign, link_index)
    for (name, servo_id, sign), link_index in zip(
        [("base_yaw", 6, 1), ("shoulder_pitch", 5, 1), ("elbow_pitch", 4, -1), ("wrist_pitch", 3, -1), ("wrist_roll", 2, 1)],
        ACTIVE_LINK_INDICES,
    )
)

# USB dropouts (the board falls off the bus, e.g. "disabled by hub (EMI?)"):
# keep trying to reopen it this long, then resume only if every joint still
# reads within RECONNECT_MAX_DRIFT_DEG of where it was commanded. A dropout mid
# move leaves a fast joint short of the plan by its tracking lag (the base was
# 13.7 deg behind mid-turn), so that's generous; it catches a reset or a knock.
RECONNECT_TIMEOUT_S = 20.0
RECONNECT_MAX_DRIFT_DEG = 30.0
RECONNECT_RESEND_MS = 500  # re-sending the last command on resuming: at least this long ...
RECONNECT_CATCHUP_DEG_PER_S = 30.0  # ... and no faster than this for the joint farthest off

GRIPPER_SERVO_ID = 1
GRIPPER_OPEN_POS = 0
GRIPPER_CLOSED_POS = 1000

# Max IK miss (m) before a target counts as unreachable.
IK_POSITION_TOLERANCE_M = 0.01

# Auto tilt (choose_tilt): the tilts tried (deg from straight down; 90 =
# horizontal, past it = upward), swept at the coarse step, then the fine one
# within a coarse step of the best. Misses within AUTO_TILT_TIE_M of the best
# count as tied, and the tie goes to the least joint travel.
AUTO_TILT_RANGE_DEG = (0.0, 100.0)
AUTO_TILT_COARSE_STEP_DEG = 1.0
AUTO_TILT_FINE_STEP_DEG = 0.05
AUTO_TILT_TIE_M = 1e-4

# IK may reach a target turned around: base yaw 180 deg from facing it, the arm
# leaning back over the base, wrist upside down. That swings the gripper (and
# the wrist camera) at the arm's own body, so it's off unless the facing yaw is
# past the base's limits.
ALLOW_TURNED_AROUND = False
# Targets nearer the base axis than this (m) face every way; never turned around.
TURNED_AROUND_MIN_RADIUS_M = 0.02

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
