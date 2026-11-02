"""The xArm 1S kinematic chain, loaded with ikpy from xarm_ik/urdf/xarm_v2.urdf.

Uses the URDF's axes as-is: forward (base_yaw=0) is -Y (see controller.py).
"""

from pathlib import Path

from ikpy.chain import Chain

URDF_PATH = Path(__file__).parent / "urdf" / "xarm_v2.urdf"

# Explicit walk, since the tree branches at eef_base_link (two fingers plus
# the fixed finger_pinch_joint we want) and ikpy would pick a finger.
URDF_CHAIN_PATH = [
    "xarm_base_link",
    "xarm_6_joint", "xarm_6_link",
    "xarm_5_joint", "xarm_5_link",
    "xarm_4_joint", "xarm_4_link",
    "xarm_3_joint", "xarm_3_link",
    "xarm_2_joint", "eef_base_link",
    "finger_pinch_joint", "finger_pinch_link",
]

# Fixed base, revolute xarm_6..xarm_2, fixed finger_pinch_joint.
ACTIVE_LINKS_MASK = [False, True, True, True, True, True, False]


def build_chain() -> Chain:
    return Chain.from_urdf_file(
        str(URDF_PATH), base_elements=URDF_CHAIN_PATH, active_links_mask=ACTIVE_LINKS_MASK, name="xarm_1s"
    )
