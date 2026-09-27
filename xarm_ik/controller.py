"""Command the xArm 1S gripper by Cartesian pose relative to the base frame.

Frame: origin at the base center, URDF axes unremapped (see kinematics.py)
-- -Y is forward (base_yaw=0), Z is up, X is left (right-handed FLU).
Units are meters.
"""

import math
from typing import Callable

import numpy as np
import xarm
from ikpy.chain import Chain

from . import config
from .kinematics import build_chain



# Fallback seeds (shoulder, elbow, wrist deg) tried when IK from the
# current pose lands in a bad local minimum (the all-zeros pose is
# singular and prone to this). Spans elbow-up/down so one is usually close.
_FALLBACK_SEEDS_DEG = [
    (30, 80, -60),
    (-30, -80, 60),
    (60, 85, -75),
    (-60, -85, 75),
]


class GripperPoseController:
    def __init__(self, com_port: str = "USB", dry_run: bool = False, debug: bool = False):
        self.dry_run = dry_run
        self.chain = build_chain()
        # Second chain, wrist_roll excluded, for move_to_pose's position/
        # orientation solve only (roll can't affect that objective -- its
        # axis runs through the tool point). self.chain still reports/sends.
        position_only_mask = list(self.chain.active_links_mask)
        position_only_mask[config.ACTIVE_LINK_INDICES[4]] = False
        self._position_chain = Chain(self.chain.links, active_links_mask=position_only_mask, name="xarm_1s_position_only")
        self._joint_angles_rad = np.zeros(len(self.chain.links))
        self._gripper_open_fraction = 0.5
        self._device = None if dry_run else xarm.Controller(com_port, debug=debug)

    def home(self, duration_ms: int = 1000, wait: bool = True) -> None:
        """Send every joint (and the gripper) to its mechanical center."""
        zero_angles_rad = np.zeros(len(self.chain.links))
        self._send_joint_angles(zero_angles_rad, duration_ms=duration_ms, wait=wait)
        self._joint_angles_rad[:] = 0.0
        self.set_gripper(0.5, duration_ms=duration_ms, wait=False)

    def move_to_pose(
        self,
        x: float,
        y: float,
        z: float,
        approach: tuple | None = None,
        duration_ms: int = 1000,
        wait: bool = True,
        position_tolerance_m: float | None = None,
        linear_step_m: float | None = None,
        on_step: Callable[[dict, int], None] | None = None,
    ) -> np.ndarray:
        """Move the gripper tip to (x, y, z) meters in the base frame.

        `approach` is the direction the gripper should point, as a vector
        in the base frame (it is normalized for you). The default points
        straight down, i.e. picking up from above.

        `position_tolerance_m` overrides config.IK_POSITION_TOLERANCE_M -- how
        far (m) a solution may miss before raising (inf accepts any). Seed
        retries still chase the stricter default, so this only relaxes acceptance.

        Splits the move into waypoints <= `linear_step_m` apart (default
        config.LINEAR_MOVE_STEP_M) so the tip travels a straight line, not
        the arc that comes from interpolating joint angles linearly.

        Each waypoint gets an even share of duration_ms. Only the final one
        is held to `position_tolerance_m` -- intermediates are best-effort,
        since a straight line between reachable points can cross bad ground.

        `on_step`, if given, runs after each waypoint as
        `on_step(joint_angles_deg, duration_ms)` -- for mirroring the real
        path (e.g. a sim viewer) instead of a cruder straight-line start/end guess.

        Returns the joint angles (radians) used for the final waypoint,
        including the fixed links. Raises ValueError if the final target
        is outside the arm's reach (i.e. outside `position_tolerance_m`).
        """
        if position_tolerance_m is None:
            position_tolerance_m = config.IK_POSITION_TOLERANCE_M
        if linear_step_m is None:
            linear_step_m = config.LINEAR_MOVE_STEP_M

        target_position = np.array([x, y, z], dtype=float)
        target_orientation = None
        if approach is not None:
            target_orientation = np.array(approach, dtype=float)
            target_orientation /= np.linalg.norm(target_orientation)

        start_position = self.get_pose()
        start_orientation = self.get_approach_vector() if target_orientation is not None else None
        num_steps = self._num_linear_steps(start_position, target_position, duration_ms, linear_step_m)
        step_duration_ms = duration_ms // num_steps

        solution = None
        for step in range(1, num_steps + 1):
            fraction = step / num_steps
            step_position = start_position + fraction * (target_position - start_position)
            step_orientation = None
            if target_orientation is not None:
                step_orientation = start_orientation + fraction * (target_orientation - start_orientation)
                step_orientation /= np.linalg.norm(step_orientation)

            is_final_step = step == num_steps
            # Only the final waypoint must satisfy position_tolerance_m --
            # intermediates just steer the path straighter and get
            # best-effort placement (the line can still cross bad ground).
            step_tolerance = position_tolerance_m if is_final_step else math.inf
            # Final solve gets full optimizer effort; interim ones are
            # capped (LINEAR_STEP_OPTIMIZER_BUDGET) so many small steps
            # don't multiply per-call IK cost into seconds of latency.
            optimizer_budget = None if is_final_step else config.LINEAR_STEP_OPTIMIZER_BUDGET
            # Bias intermediate solves toward the previous step's branch;
            # the final solve gets none, since it must reach the actual
            # target even if that needs a different configuration.
            regularization = None if is_final_step else config.LINEAR_STEP_REGULARIZATION
            solution, error_m = self._solve_ik(
                step_position, step_orientation, step_tolerance, optimizer_budget, regularization
            )
            if error_m > step_tolerance:
                raise ValueError(
                    f"Target pose ({x:.3f}, {y:.3f}, {z:.3f}) m is not reachable "
                    f"with the current link lengths/joint limits "
                    f"(residual error {error_m * 1000:.1f} mm, "
                    f"tolerance {position_tolerance_m * 1000:.1f} mm)."
                )

            # Every waypoint but the last has to actually finish before
            # the next command is sent, or it'd just get redirected
            # mid-motion. The last one alone honors the caller's `wait`.
            self._send_joint_angles(solution, duration_ms=step_duration_ms, wait=(wait if is_final_step else True))
            self._joint_angles_rad = solution
            if on_step is not None:
                on_step(self.get_joint_angles_deg(), step_duration_ms)

        return solution

    def _num_linear_steps(
        self,
        start_position: np.ndarray,
        target_position: np.ndarray,
        duration_ms: int,
        linear_step_m: float,
    ) -> int:
        """How many waypoints to split a move into: enough that consecutive
        waypoints are within linear_step_m of each other, but never so many
        that a waypoint would get less than config.MIN_STEP_DURATION_MS.
        """
        distance_m = float(np.linalg.norm(target_position - start_position))
        steps_for_smoothness = max(1, math.ceil(distance_m / linear_step_m))
        steps_for_min_duration = max(1, duration_ms // config.MIN_STEP_DURATION_MS)
        return max(1, min(steps_for_smoothness, steps_for_min_duration))

    def _solve_ik(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray | None,
        position_tolerance_m: float = config.IK_POSITION_TOLERANCE_M,
        optimizer_budget: int | None = None,
        regularization_parameter: float | None = None,
    ):
        """Run IK from the current pose, retrying from alternate seeds if the
        optimizer lands in a local minimum. Returns (best_solution, error_m).

        Stops early once a seed beats whichever tolerance is stricter
        (config's default or the caller's) -- a loose tolerance only
        relaxes acceptance; a tighter one keeps trying more seeds.

        `optimizer_budget` caps scipy's function evals per seed -- ikpy's
        default is generous, and times dozens of calls/move is the
        dominant latency cost. Retries still run; this just speeds up each attempt.

        `regularization_parameter` (ikpy's) biases the solve toward its
        seed, avoiding an equally-valid but different branch (e.g. an
        elbow flip). Only safe with wrist_roll excluded (see __init__).
        """
        best_solution = None
        best_error = np.inf
        early_exit_error_m = min(config.IK_POSITION_TOLERANCE_M, position_tolerance_m)

        orientation_mode = "Z" if target_orientation is not None else None
        for seed in self._ik_seeds():
            solution = self._position_chain.inverse_kinematics(
                target_position=target_position,
                target_orientation=target_orientation,
                orientation_mode=orientation_mode,
                initial_position=seed,
                optimizer_budget=optimizer_budget,
                regularization_parameter=regularization_parameter,
            )
            reached_position = self.chain.forward_kinematics(solution)[:3, 3]
            error_m = float(np.linalg.norm(reached_position - target_position))

            if error_m < best_error:
                best_solution, best_error = solution, error_m
            if best_error <= early_exit_error_m:
                break

        return best_solution, best_error

    def _ik_seeds(self):
        yield self._joint_angles_rad
        # config.JOINTS[1:4] (shoulder/elbow/wrist pitch) are most prone to
        # local minima. Mapped via config.ACTIVE_LINK_INDICES rather than
        # hardcoded so this stays correct if the chain layout ever changes.
        shoulder_idx, elbow_idx, wrist_idx = config.ACTIVE_LINK_INDICES[1:4]
        shoulder_bounds, elbow_bounds, wrist_bounds = (
            config.JOINTS[1].limit_deg,
            config.JOINTS[2].limit_deg,
            config.JOINTS[3].limit_deg,
        )
        for shoulder_deg, elbow_deg, wrist_deg in _FALLBACK_SEEDS_DEG:
            # Start from the current state, not all-zeros: wrist_roll is
            # inactive in self._position_chain, so a winning seed's roll
            # value is kept as-is instead of silently resetting to 0.
            seed = self._joint_angles_rad.copy()
            seed[shoulder_idx] = np.radians(np.clip(shoulder_deg, *shoulder_bounds))
            seed[elbow_idx] = np.radians(np.clip(elbow_deg, *elbow_bounds))
            seed[wrist_idx] = np.radians(np.clip(wrist_deg, *wrist_bounds))
            yield seed

    def move_joint(self, joint_name: str, angle_deg: float, duration_ms: int = 1000, wait: bool = True) -> None:
        """Command a single arm joint directly, bypassing inverse kinematics.

        `joint_name` is one of the names in config.JOINTS (e.g. "elbow_pitch").
        """
        for chain_index, joint in zip(config.ACTIVE_LINK_INDICES, config.JOINTS):
            if joint.name == joint_name:
                break
        else:
            raise ValueError(
                f"Unknown joint {joint_name!r}. Known joints: "
                f"{[j.name for j in config.JOINTS]}"
            )

        self._send(self._joint_commands(joint, angle_deg), duration_ms, wait)
        self._joint_angles_rad[chain_index] = np.radians(angle_deg)

    def get_pose(self) -> np.ndarray:
        """Return the gripper tip's current (x, y, z) in the base frame."""
        return self.chain.forward_kinematics(self._joint_angles_rad)[:3, 3]

    def get_approach_vector(self) -> np.ndarray:
        """Return the gripper's current pointing direction (a unit vector
        in the base frame) -- the same "Z" axis move_to_pose's `approach`
        targets, read back from the current joint angles.
        """
        return self.chain.forward_kinematics(self._joint_angles_rad)[:3, 2]

    def get_joint_angles_deg(self) -> dict:
        """Return the current commanded angle (degrees) of each arm joint, by name."""
        active_angles_rad = [
            angle
            for angle, is_active in zip(self._joint_angles_rad, self.chain.active_links_mask)
            if is_active
        ]
        return {
            joint.name: np.degrees(angle_rad)
            for joint, angle_rad in zip(config.JOINTS, active_angles_rad)
        }

    def get_gripper_open_fraction(self) -> float:
        """Return the gripper's last-commanded open fraction (0=closed, 1=open)."""
        return self._gripper_open_fraction

    def set_gripper(self, open_fraction: float, duration_ms: int = 300, wait: bool = True) -> None:
        open_fraction = max(0.0, min(1.0, open_fraction))
        position = int(round(
            config.GRIPPER_CLOSED_POS
            + open_fraction * (config.GRIPPER_OPEN_POS - config.GRIPPER_CLOSED_POS)
        ))
        self._send_servo(config.GRIPPER_SERVO_ID, position, duration_ms, wait)
        self._gripper_open_fraction = open_fraction

    def open_gripper(self, **kwargs) -> None:
        self.set_gripper(1.0, **kwargs)

    def close_gripper(self, **kwargs) -> None:
        self.set_gripper(0.0, **kwargs)

    def _send_joint_angles(self, joint_angles_rad: np.ndarray, duration_ms: int, wait: bool) -> None:
        active_angles_rad = [
            angle
            for angle, is_active in zip(joint_angles_rad, self.chain.active_links_mask)
            if is_active
        ]

        commands = []
        for joint, angle_rad in zip(config.JOINTS, active_angles_rad):
            commands.extend(self._joint_commands(joint, np.degrees(angle_rad)))

        self._send(commands, duration_ms, wait)

    def _joint_commands(self, joint: config.JointConfig, angle_deg: float):
        commands = []
        for servo_id, sign in zip(joint.servo_ids, joint.signs):
            # limit_deg bounds the true angle; when sign flips it for a
            # reverse-mounted servo, the flipped value's valid range is
            # limit_deg's mirror image (equal only if symmetric).
            lower_deg, upper_deg = joint.limit_deg
            signed_limit_deg = joint.limit_deg if sign > 0 else (-upper_deg, -lower_deg)
            commands.append((servo_id, config.angle_deg_to_position(sign * angle_deg, signed_limit_deg)))
        return commands

    def _send_servo(self, servo_id: int, position: int, duration_ms: int, wait: bool) -> None:
        self._send([(servo_id, position)], duration_ms, wait)

    def _send(self, commands, duration_ms: int, wait: bool) -> None:
        if self.dry_run:
            print(f"[dry run] setPosition({commands}, duration={duration_ms}, wait={wait})")
            return
        self._device.setPosition(commands, duration=duration_ms, wait=wait)
