"""Closed-form IK for the xArm 1S at a given tilt.

Past base yaw, shoulder/elbow/wrist pitch all turn about parallel X axes and
every offset between them has zero X, so the arm is a planar 3R chain in its
yaw frame's Y-Z plane; wrist roll turns about the tool axis, so it moves
neither the tip nor the approach. Fixing the tilt then leaves a 2-link problem:
back the wrist off the tip along the approach, then law of cosines.

Every target has up to 4 solutions: 2 base yaws (reach toward it, or turn the
other way and lean back over) x 2 elbow branches; turned_around=False rules
out the second yaw (error inf) unless the facing yaw is past its limits. Joints are clamped to their
limits, so a solution may miss; `error_m` says by how much (the closest the
clamped pose gets, exact when unclamped).

Tilt is run_csv's: the lean from straight down (0) toward the target's (x, y)
(for a target over the base: toward where base yaw `centered_yaw_rad` faces,
default -Y); 90 = horizontal, 100 = 10 deg upward.
"""

import math
from dataclasses import dataclass

import numpy as np
from ikpy.chain import Chain

from . import config

# A target this close (m) to the base axis counts as over it: any yaw reaches it.
CENTERED_M = 1e-6


def _wrap(angle_rad):
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class PlanarSolutions:
    """Solutions for T tilts x 4 (yaw, elbow) choices: `joints_rad` (T, 4, 4)
    as base_yaw, shoulder, elbow, wrist pitch; `error_m` (T, 4) tip miss.
    """

    joints_rad: np.ndarray
    error_m: np.ndarray


class PlanarArm:
    """The chain's geometry as planar link offsets (checked on load)."""

    def __init__(self, chain: Chain):
        yaw, shoulder, elbow, wrist, roll, tip = (chain.links[i] for i in range(1, 7))
        for link in (shoulder, elbow, wrist):
            axis = np.asarray(link.rotation)
            if not (np.allclose(np.abs(axis), [1, 0, 0]) and np.allclose(link.origin_translation[0], 0)
                    and np.allclose(link.origin_orientation[1:], 0)):
                raise ValueError(f"{link.name} breaks the planar-arm assumption")
        if not (np.allclose(yaw.rotation, [0, 0, 1]) and np.allclose(yaw.origin_translation[:2], 0)
                and np.allclose(yaw.origin_orientation[:2], 0) and np.allclose(roll.rotation, [0, 0, 1])
                and np.allclose(roll.origin_translation[0], 0) and np.allclose(roll.origin_orientation, 0)
                and np.allclose(tip.origin_translation[:2], 0) and np.allclose(tip.origin_orientation, 0)):
            raise ValueError("base yaw / wrist roll / tip break the planar-arm assumption")

        self._base_z = yaw.origin_translation[2]
        self._yaw_offset = yaw.origin_orientation[2]
        # Plane coords (v, w) = the yaw frame's (y, z); each pitch joint adds
        # its fixed X rotation plus axis_sign * q to the in-plane angle.
        self._shoulder_origin = np.array(shoulder.origin_translation[1:])
        self._fixed_rad = np.array([shoulder.origin_orientation[0], elbow.origin_orientation[0], wrist.origin_orientation[0]])
        self._signs = np.array([np.sign(link.rotation[0]) for link in (shoulder, elbow, wrist)])
        self._upper_arm = np.array(elbow.origin_translation[1:])  # in the shoulder link's frame
        self._forearm = np.array(wrist.origin_translation[1:])  # in the elbow link's frame
        # Wrist pitch axis to tip, in the wrist link's frame (roll can't move it).
        self._tool = np.array(roll.origin_translation[1:]) + np.array([0.0, tip.origin_translation[2]])

        limits = [joint.limit_deg for joint in config.JOINTS[:4]]
        self._lower_rad = np.radians([lo for lo, _ in limits])
        self._upper_rad = np.radians([hi for _, hi in limits])

    def facing_yaw(self, target) -> float:
        """The base yaw facing `target`'s (x, y) (not wrapped into the limits)."""
        return float(_wrap(math.atan2(-float(target[0]), float(target[1])) - self._yaw_offset))

    def turned_around(self, yaw_rad: float, target) -> bool:
        """Whether base yaw `yaw_rad` faces away from `target`'s (x, y) when it
        needn't (the facing yaw is within limits, the target off the base axis)."""
        radius = math.hypot(float(target[0]), float(target[1]))
        if radius < config.TURNED_AROUND_MIN_RADIUS_M or not self.can_face(target):
            return False
        return float(self.facing(yaw_rad) @ np.asarray(target[:2], dtype=float)) < 0.0

    def can_face(self, target) -> bool:
        """Whether the yaw facing `target` is within base yaw's limits."""
        return bool(self._lower_rad[0] <= self.facing_yaw(target) <= self._upper_rad[0])

    def facing(self, yaw_rad: float) -> np.ndarray:
        """The horizontal unit (x, y) base yaw `yaw_rad` reaches toward (its +v)."""
        heading = self._yaw_offset + yaw_rad
        return np.array([-math.sin(heading), math.cos(heading)])

    def solve(self, target, tilts_rad, centered_yaw_rad: float | None = None,
              turned_around: bool = True) -> PlanarSolutions:
        """All solutions putting the tip at `target` (x, y, z) at each tilt.
        `turned_around`=False: the turned-around yaw's solutions get error inf
        (unless the facing yaw is past its limits)."""
        forbid_turned = (not turned_around and math.hypot(float(target[0]), float(target[1]))
                         >= config.TURNED_AROUND_MIN_RADIUS_M and self.can_face(target))
        x, y, z = map(float, target)
        tilts = np.atleast_1d(np.asarray(tilts_rad, dtype=float))
        radius = math.hypot(x, y)
        # A yaw frame's +v points along (-sin, cos) of (yaw_offset + yaw).
        if radius > CENTERED_M:
            heading = math.atan2(-x, y)
        elif centered_yaw_rad is not None:
            heading = self._yaw_offset + centered_yaw_rad
        else:
            heading = math.atan2(0.0, -1.0)
        joints, errors = [], []
        for v_sign in (1.0, -1.0):  # target on the +v side, or turned around onto -v
            yaw = _wrap(heading - self._yaw_offset + (0.0 if v_sign > 0 else math.pi))
            tip = np.array([v_sign * radius, z - self._base_z])
            # Approach leans outward (away from the base) by the tilt.
            approach = np.stack([v_sign * np.sin(tilts), -np.cos(tilts)], axis=-1)
            for elbow_sign in (1.0, -1.0):
                q = self._solve_branch(tip, approach, elbow_sign)
                q = np.column_stack([np.full(len(tilts), yaw), q])
                q = np.clip(q, self._lower_rad, self._upper_rad)
                joints.append(q)
                error = np.linalg.norm(self.tip_position(q) - [x, y, z], axis=-1)
                if forbid_turned and v_sign < 0:
                    error = np.full_like(error, np.inf)
                errors.append(error)
        return PlanarSolutions(np.stack(joints, axis=1), np.stack(errors, axis=1))

    def solve_radial(self, outward_xy, reaches, zs, tilt_rad: float) -> PlanarSolutions:
        """solve() for many targets at once, all on one line out from the base: target i at
        reaches[i] along the horizontal unit `outward_xy`, height zs[i], each at this tilt,
        facing it (no turned-around solutions). `joints_rad` (N, 2, 4) for the two elbow
        branches, `error_m` (N, 2)."""
        reaches = np.asarray(reaches, dtype=float)
        zs = np.asarray(zs, dtype=float)
        x, y = float(outward_xy[0]), float(outward_xy[1])
        yaw = _wrap(math.atan2(-x, y) - self._yaw_offset)
        tips = np.column_stack([reaches, zs - self._base_z])
        approach = np.tile([math.sin(tilt_rad), -math.cos(tilt_rad)], (len(reaches), 1))
        targets = np.column_stack([reaches * x, reaches * y, zs])
        joints, errors = [], []
        for elbow_sign in (1.0, -1.0):
            q = self._solve_branch(tips, approach, elbow_sign)
            q = np.clip(np.column_stack([np.full(len(reaches), yaw), q]), self._lower_rad, self._upper_rad)
            joints.append(q)
            errors.append(np.linalg.norm(self.tip_position(q) - targets, axis=-1))
        return PlanarSolutions(np.stack(joints, axis=1), np.stack(errors, axis=1))

    def _solve_branch(self, tip, approach, elbow_sign):
        """(T, 3) shoulder/elbow/wrist pitch for one elbow branch."""
        # Tool z is the wrist frame's +w, so its in-plane angle is the approach's - 90 deg.
        wrist_angle = np.arctan2(approach[:, 1], approach[:, 0]) - np.pi / 2
        wrist_point = tip - _rotate(self._tool, wrist_angle)
        delta = wrist_point - self._shoulder_origin
        l1, l2 = np.linalg.norm(self._upper_arm), np.linalg.norm(self._forearm)
        # Link directions as angles in their own frames, so offsets needn't be pure +w.
        offset1 = math.atan2(self._upper_arm[1], self._upper_arm[0])
        offset2 = math.atan2(self._forearm[1], self._forearm[0])
        distance_sq = np.sum(delta**2, axis=-1)
        # Out of reach: clamp to fully stretched/folded, the nearest the annulus gets.
        cos_bend = np.clip((distance_sq - l1**2 - l2**2) / (2 * l1 * l2), -1.0, 1.0)
        bend = elbow_sign * np.arccos(cos_bend)
        direction1 = np.arctan2(delta[:, 1], delta[:, 0]) - np.arctan2(l2 * np.sin(bend), l1 + l2 * np.cos(bend))
        upper_angle = direction1 - offset1
        fore_angle = direction1 + bend - offset2
        absolute = np.column_stack([upper_angle, fore_angle, wrist_angle])
        relative = np.diff(absolute, axis=1, prepend=0.0) - self._fixed_rad
        return _wrap(relative * self._signs)

    def tip_position(self, joints_rad):
        """FK for (..., 4) joints: (..., 3) tip (x, y, z)."""
        q = np.asarray(joints_rad, dtype=float)
        absolute = np.cumsum(self._fixed_rad + self._signs * q[..., 1:], axis=-1)
        plane = (self._shoulder_origin + _rotate(self._upper_arm, absolute[..., 0])
                 + _rotate(self._forearm, absolute[..., 1]) + _rotate(self._tool, absolute[..., 2]))
        heading = self._yaw_offset + q[..., 0]
        v, w = plane[..., 0], plane[..., 1]
        return np.stack([-v * np.sin(heading), v * np.cos(heading), w + self._base_z], axis=-1)


def _rotate(vector, angle_rad):
    """`vector` (v, w) turned by each angle about the pitch axis: (..., 2)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.stack([vector[0] * c - vector[1] * s, vector[0] * s + vector[1] * c], axis=-1)
