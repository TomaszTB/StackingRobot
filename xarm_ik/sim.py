"""A PyBullet view of the arm's URDF, driven by the controller's joint angles,
to preview motion before --execute. Visual only; never touches hardware.
"""

import time

import numpy as np
import pybullet as p

from .kinematics import URDF_PATH

# Our joint names (config.JOINTS) -> the URDF's.
_URDF_JOINT_NAME = {
    "base_yaw": "xarm_6_joint",
    "shoulder_pitch": "xarm_5_joint",
    "elbow_pitch": "xarm_4_joint",
    "wrist_pitch": "xarm_3_joint",
    "wrist_roll": "xarm_2_joint",
}
_GRIPPER_JOINT_NAMES = ("xarm_1_joint", "xarm_1_joint_mirror")
_GRIPPER_OPEN_M = 0.028  # prismatic finger travel


class PybulletArmView:
    def __init__(self):
        self._client = p.connect(p.GUI)
        p.setGravity(0, 0, 0, physicsClientId=self._client)
        p.resetDebugVisualizerCamera(
            cameraDistance=0.5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0.05, 0, 0.1], physicsClientId=self._client
        )
        self._gripper_open_fraction = 0.0
        self._body = p.loadURDF(str(URDF_PATH), useFixedBase=True, physicsClientId=self._client)
        self._joint_index = {
            p.getJointInfo(self._body, i, physicsClientId=self._client)[1].decode(): i
            for i in range(p.getNumJoints(self._body, physicsClientId=self._client))
        }

    def set_joint_angles_deg(self, angles_by_name: dict) -> None:
        for name, degrees in angles_by_name.items():
            p.resetJointState(self._body, self._joint_index[_URDF_JOINT_NAME[name]], np.radians(degrees), physicsClientId=self._client)

    def set_gripper(self, open_fraction: float) -> None:
        """0 = closed, 1 = open."""
        self._gripper_open_fraction = max(0.0, min(1.0, open_fraction))
        for name in _GRIPPER_JOINT_NAMES:
            p.resetJointState(
                self._body, self._joint_index[name], self._gripper_open_fraction * _GRIPPER_OPEN_M, physicsClientId=self._client
            )

    def mirror_tick(self, angles_by_name: dict, gripper_open_fraction: float, remaining_ms: int) -> None:
        """The controller's on_tick: animate to the tick's target in the time left
        (so the view runs up to a tick behind the arm).
        """
        self.animate_to(angles_by_name, gripper_open_fraction, remaining_ms / 1000)

    def animate_to(self, angles_by_name: dict, gripper_open_fraction: float, duration_s: float, fps: int = 60) -> None:
        start_angles = {
            name: np.degrees(p.getJointState(self._body, self._joint_index[urdf_name], physicsClientId=self._client)[0])
            for name, urdf_name in _URDF_JOINT_NAME.items()
        }
        start_gripper = self._gripper_open_fraction
        steps = max(1, int(duration_s * fps))
        start_s = time.perf_counter()
        for step in range(1, steps + 1):
            t = step / steps
            self.set_joint_angles_deg({name: start_angles[name] + t * (angles_by_name[name] - start_angles[name]) for name in angles_by_name})
            self.set_gripper(start_gripper + t * (gripper_open_fraction - start_gripper))
            # Paced off the start, so rendering time doesn't overrun duration_s.
            remaining_s = start_s + t * duration_s - time.perf_counter()
            if remaining_s > 0:
                time.sleep(remaining_s)

    def wait_until_closed(self) -> None:
        """Keep rendering until the window is closed."""
        while p.isConnected(physicsClientId=self._client):
            p.stepSimulation(physicsClientId=self._client)
            time.sleep(1 / 60)
