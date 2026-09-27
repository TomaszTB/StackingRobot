"""A PyBullet viewer for the reference URDF (see xarm_ik/urdf/), driven by
the same joint angles xarm_ik.controller.GripperPoseController computes.

Purely visual -- never touches real hardware. Lets you preview an IK
solution's shape before trusting it with --execute.

Uses a different reference URDF than config.py's numbers (this one
models a 2-finger linear gripper), so treat it as "does this motion look
reasonable", not a pixel-perfect preview of the real unit.
"""

import time
from pathlib import Path

import numpy as np
import pybullet as p

URDF_PATH = Path(__file__).parent / "urdf" / "xarm_v2.urdf"

# Maps our own joint names (config.JOINTS) to this URDF's joint names.
_URDF_JOINT_NAME = {
    "base_yaw": "xarm_6_joint",
    "shoulder_pitch": "xarm_5_joint",
    "elbow_pitch": "xarm_4_joint",
    "wrist_pitch": "xarm_3_joint",
    "wrist_roll": "xarm_2_joint",
}
_GRIPPER_JOINT_NAMES = ("xarm_1_joint", "xarm_1_joint_mirror")
_GRIPPER_CLOSED_M = 0.0
_GRIPPER_OPEN_M = 0.028  # this URDF's prismatic finger travel, meters


class PybulletArmView:
    def __init__(self, gui: bool = True):
        self._client = p.connect(p.GUI if gui else p.DIRECT)
        p.setGravity(0, 0, 0, physicsClientId=self._client)
        p.resetDebugVisualizerCamera(
            cameraDistance=0.5,
            cameraYaw=50,
            cameraPitch=-35,
            cameraTargetPosition=[0.05, 0, 0.1],
            physicsClientId=self._client,
        )
        self._body = p.loadURDF(str(URDF_PATH), useFixedBase=True, physicsClientId=self._client)

        self._joint_index = {}
        for i in range(p.getNumJoints(self._body, physicsClientId=self._client)):
            name = p.getJointInfo(self._body, i, physicsClientId=self._client)[1].decode()
            self._joint_index[name] = i

    def set_joint_angles_deg(self, angles_by_name: dict) -> None:
        """Snap directly to the given {our_joint_name: degrees} angles."""
        for our_name, degrees in angles_by_name.items():
            index = self._joint_index[_URDF_JOINT_NAME[our_name]]
            p.resetJointState(self._body, index, np.radians(degrees), physicsClientId=self._client)

    def set_gripper(self, open_fraction: float) -> None:
        """Snap directly to the given gripper open fraction (0=closed, 1=open)."""
        open_fraction = max(0.0, min(1.0, open_fraction))
        travel_m = _GRIPPER_CLOSED_M + open_fraction * (_GRIPPER_OPEN_M - _GRIPPER_CLOSED_M)
        for name in _GRIPPER_JOINT_NAMES:
            p.resetJointState(self._body, self._joint_index[name], travel_m, physicsClientId=self._client)

    def step_to(self, angles_by_name: dict, duration_s: float) -> None:
        """Snap to the given angles, then pace to duration_s -- for replaying an
        already fine-grained sequence (e.g. move_to_pose's `on_step`). Unlike
        animate_to, never interpolates on its own, avoiding a re-introduced arc.
        """
        self.set_joint_angles_deg(angles_by_name)
        if duration_s > 0:
            time.sleep(duration_s)

    def animate_to(self, angles_by_name: dict, gripper_open_fraction: float, duration_s: float, fps: int = 60) -> None:
        """Smoothly interpolate from the current pose to the given one."""
        start_angles = {
            our_name: np.degrees(
                p.getJointState(self._body, self._joint_index[urdf_name], physicsClientId=self._client)[0]
            )
            for our_name, urdf_name in _URDF_JOINT_NAME.items()
        }
        steps = max(1, int(duration_s * fps))
        for step in range(1, steps + 1):
            t = step / steps
            frame = {
                name: start_angles[name] + t * (angles_by_name[name] - start_angles[name])
                for name in angles_by_name
            }
            self.set_joint_angles_deg(frame)
            time.sleep(duration_s / steps)
        self.set_gripper(gripper_open_fraction)

    def is_open(self) -> bool:
        return p.isConnected(physicsClientId=self._client)

    def wait_until_closed(self) -> None:
        """Block (rendering) until the user closes the PyBullet window."""
        while self.is_open():
            p.stepSimulation(physicsClientId=self._client)
            time.sleep(1 / 60)

    def close(self) -> None:
        if self.is_open():
            p.disconnect(physicsClientId=self._client)
