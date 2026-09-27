"""Kinematic chain for the xArm 1S, loaded from the reference URDF
(xarm_ik/urdf/xarm_v2.urdf -- see config.py for its source) instead of
hand-transcribed link lengths and joint limits.

Built with ikpy (https://github.com/Phylliade/ikpy), solving joint
angles via scipy least-squares.

Frame: uses the URDF's own axes unremapped, so "forward" (base_yaw=0) is
-Y, not +X. See controller.py's docstring for the full convention.
"""

from pathlib import Path

from ikpy.chain import Chain

URDF_PATH = Path(__file__).parent / "urdf" / "xarm_v2.urdf"

# Explicit base-to-tip walk through the URDF tree: it branches at
# eef_base_link (2 finger joints + this fixed one), and auto-follow would
# arbitrarily pick a finger instead of the fixed finger_pinch_joint we want.
URDF_CHAIN_PATH = [
    "xarm_base_link",
    "xarm_6_joint", "xarm_6_link",
    "xarm_5_joint", "xarm_5_link",
    "xarm_4_joint", "xarm_4_link",
    "xarm_3_joint", "xarm_3_link",
    "xarm_2_joint", "eef_base_link",
    "finger_pinch_joint", "finger_pinch_link",
]

# base_link (fixed) + 5 revolute joints (xarm_6..xarm_2) + the fixed
# finger_pinch_joint, matching URDF_CHAIN_PATH's order.
ACTIVE_LINKS_MASK = [False, True, True, True, True, True, False]


def build_chain() -> Chain:
    return Chain.from_urdf_file(
        str(URDF_PATH),
        base_elements=URDF_CHAIN_PATH,
        active_links_mask=ACTIVE_LINKS_MASK,
        name="xarm_1s",
    )
