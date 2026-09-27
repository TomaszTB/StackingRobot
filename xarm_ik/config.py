"""Hardware-mapping constants for the Hiwonder xArm 1S.

Link lengths and joint axes/limits come from the reference URDF at
xarm_ik/urdf/xarm_v2.urdf (see kinematics.py), sourced from
https://github.com/jonathanrandall/autonomous_robot_gardener.

This module only adds what the URDF can't: each joint's Hiwonder servo ID
and that servo's raw position <-> angle conversion.

SERVO MAPPING: 6=base yaw, 5=shoulder pitch, 4=elbow pitch, 3=wrist pitch,
2=wrist roll, 1=gripper, per the URDF's own joint naming -- verify on
your unit by moving one servo ID at a time.

`signs` flips servos wired opposite the URDF's <axis> convention;
confirmed needed for base_yaw (servo 6), which rotates opposite the
simulated model.

JOINT LIMITS: pulled from the loaded URDF chain (kinematics.build_chain),
re-capped here at the servo bus's +/-125 deg hard ceiling as a defensive
backstop in case the URDF is ever swapped for one that exceeds it.
"""

import math
from dataclasses import dataclass

from .kinematics import ACTIVE_LINKS_MASK, build_chain

HARDWARE_LIMIT_DEG = 125.0


@dataclass(frozen=True)
class JointConfig:
    name: str
    servo_ids: tuple
    limit_deg: tuple
    signs: tuple = (1,)


def _capped_limit_deg(chain, active_link_index: int) -> tuple:
    """The chain link's own bounds (radians), in degrees, capped at the
    servo bus's +/-125 degree hardware ceiling.
    """
    lower_rad, upper_rad = chain.links[active_link_index].bounds
    lower_deg, upper_deg = math.degrees(lower_rad), math.degrees(upper_rad)
    return (max(lower_deg, -HARDWARE_LIMIT_DEG), min(upper_deg, HARDWARE_LIMIT_DEG))


_chain = build_chain()
# Index of each active (non-fixed) chain link, base-to-wrist (where
# ACTIVE_LINKS_MASK is True). Public so callers map config.JOINTS entries
# to chain indices without hardcoding the chain's layout.
ACTIVE_LINK_INDICES = tuple(i for i, active in enumerate(ACTIVE_LINKS_MASK) if active)

# --- Kinematic joints, base to wrist, in chain order. ---
JOINTS = (
    JointConfig(name="base_yaw", servo_ids=(6,), limit_deg=_capped_limit_deg(_chain, ACTIVE_LINK_INDICES[0]), signs=(1,)),
    JointConfig(name="shoulder_pitch", servo_ids=(5,), limit_deg=_capped_limit_deg(_chain, ACTIVE_LINK_INDICES[1]), signs=(1,)),
    JointConfig(name="elbow_pitch", servo_ids=(4,), limit_deg=_capped_limit_deg(_chain, ACTIVE_LINK_INDICES[2]), signs=(-1,)),
    JointConfig(name="wrist_pitch", servo_ids=(3,), limit_deg=_capped_limit_deg(_chain, ACTIVE_LINK_INDICES[3]), signs=(-1,)),
    JointConfig(name="wrist_roll", servo_ids=(2,), limit_deg=_capped_limit_deg(_chain, ACTIVE_LINK_INDICES[4]), signs=(1,)),
)

GRIPPER_SERVO_ID = 1
GRIPPER_OPEN_POS = 0
GRIPPER_CLOSED_POS = 1000

# Position 500 = the joint's zero angle; 0/1000 map to that joint's own
# limit_deg (not a fixed +/-125 deg scale), since each servo is limited to
# its own usable range -- so the same raw position differs per joint.
SERVO_POSITION_CENTER = 500
SERVO_POSITION_MIN = 0
SERVO_POSITION_MAX = 1000

# Reject an IK solution whose forward kinematics misses the requested point
# by more than this much -- almost always means the target is out of reach.
IK_POSITION_TOLERANCE_M = 0.01

# Max spacing (m) between move_to_pose's intermediate waypoints, so the
# tip traces a straight line -- servos interpolate joint *angles*
# linearly, which traces an arc in Cartesian space, not a line.
LINEAR_MOVE_STEP_M = 0.01

# Floor on each waypoint's duration, however small LINEAR_MOVE_STEP_M
# would otherwise call for -- below this a step is too brief for the
# servo bus to act on before the next command arrives.
MIN_STEP_DURATION_MS = 100

# Cap on scipy's function evals per intermediate waypoint (move_to_pose)
# -- these land roughly on the line from a nearby seed, so ikpy's default
# budget is overkill; the final (real target) waypoint stays uncapped.
LINEAR_STEP_OPTIMIZER_BUDGET = 100

# Regularization for intermediate waypoints only, biasing each solve to
# match the last one's branch (e.g. avoids an elbow-up/down flip); cut one
# reproduced ~218 deg flip to ~9 deg. Not monotonic -- re-sweep if retuning.
LINEAR_STEP_REGULARIZATION = 0.005

# Solutions near a joint's hard limit can overshoot it by floating-point
# noise alone (e.g. a best-effort solve with a loose/infinite tolerance).
# Clamp overshoots within this margin; anything past it is a real error.
JOINT_LIMIT_EPSILON_DEG = 1e-4


def angle_deg_to_position(angle_deg: float, limit_deg: tuple) -> int:
    lower_deg, upper_deg = limit_deg
    if angle_deg < lower_deg - JOINT_LIMIT_EPSILON_DEG or angle_deg > upper_deg + JOINT_LIMIT_EPSILON_DEG:
        raise ValueError(
            f"Joint angle {angle_deg:.1f} deg is outside its configured "
            f"limits {limit_deg}."
        )
    angle_deg = max(lower_deg, min(upper_deg, angle_deg))

    if angle_deg >= 0:
        span_deg, span_position = upper_deg, SERVO_POSITION_MAX - SERVO_POSITION_CENTER
    else:
        span_deg, span_position = lower_deg, SERVO_POSITION_MIN - SERVO_POSITION_CENTER

    fraction = (angle_deg / span_deg) if span_deg != 0 else 0.0
    position = SERVO_POSITION_CENTER + fraction * span_position
    return int(round(max(SERVO_POSITION_MIN, min(SERVO_POSITION_MAX, position))))
