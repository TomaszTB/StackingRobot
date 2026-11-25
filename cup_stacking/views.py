"""Where to put the camera to look at a point: reachable tip poses, scored by the view."""

import math

import numpy as np

from xarm_ik import GripperPoseController, config
from xarm_ik.camera import IMAGE_SIZE

LOOK_DISTANCE_M = 0.20  # preferred camera-to-target distance
LOOK_MIN_DISTANCE_M = 0.16  # at LOOK_DISTANCE_M; closer blurs (focus is locked)
LOOK_CLEARANCE_M = 0.04  # the tip stays this far above the target and cups under it
LOOK_OBSTACLE_M = 0.08  # cups within this (horizontally) of the tip are under it
LOOK_MAX_OFF_CENTER = 0.4  # max target offset from the aim point, as a fraction of half the image height
LOOK_ALSO_SEE_MARGIN = 0.8  # `also_see` points stay within this fraction of the image half-size


def view_poses(arm: GripperPoseController, camera_in_gripper: np.ndarray, matrix: np.ndarray, target,
               obstacles=(), also_see=(), distance_m=LOOK_DISTANCE_M, aim_row=0.0) -> list:
    """Reachable tip poses (x, y, z, tilt) seeing `target`, best first. The tip stays on the
    target's radial line (roll 0), LOOK_CLEARANCE_M above the target and any `obstacles` (cup
    tops) under it. Scored by how steeply down the camera looks, how near `target` is to the
    aim point, and how near the camera is to `distance_m`. `also_see` points must be in frame.
    `aim_row`: the aim point's row, as a fraction of half the image height (-1 top, 0 center, 1 bottom).
    """
    min_distance_m = distance_m - (LOOK_DISTANCE_M - LOOK_MIN_DISTANCE_M)
    aim_pixel = np.array(IMAGE_SIZE) / 2 + np.array([0.0, aim_row * IMAGE_SIZE[1] / 2])
    target = np.asarray(target, dtype=float)
    tops = np.array([target, *obstacles], dtype=float)
    radius = math.hypot(target[0], target[1])
    outward = target[:2] / radius
    # (reach, z) tip positions on the radial line, clear of what's under them.
    grid = []
    for reach in np.arange(0.04, radius + 0.05, 0.01):
        tip_xy = reach * outward
        under = tops[np.linalg.norm(tops[:, :2] - tip_xy, axis=1) < LOOK_OBSTACLE_M]
        lowest = max([target[2], *under[:, 2]]) + LOOK_CLEARANCE_M
        grid += [(reach, z) for z in np.arange(max(0.10, lowest), 0.46, 0.01)]
    grid = np.array(grid).reshape(-1, 2)
    scored = []
    for tilt in np.arange(30.0, 111.0, 5.0):
        if not len(grid):
            break
        solutions = arm._planar.solve_radial(outward, grid[:, 0], grid[:, 1], math.radians(tilt))
        best = np.argmin(solutions.error_m, axis=1)
        for i in np.flatnonzero(solutions.error_m[np.arange(len(grid)), best] <= 0.001):
            reach, z = grid[i]
            tip = (*(reach * outward), z)
            angles = arm._joint_angles_rad.copy()
            angles[list(config.ACTIVE_LINK_INDICES[:4])] = solutions.joints_rad[i, best[i]]
            angles[config.ACTIVE_LINK_INDICES[4]] = 0.0
            camera = arm.chain.forward_kinematics(angles) @ camera_in_gripper
            in_camera = np.linalg.inv(camera) @ np.append(target, 1.0)
            if in_camera[2] < min_distance_m:
                continue
            pixel = matrix[:2, :2] @ (in_camera[:2] / in_camera[2]) + matrix[:2, 2]
            off_center = np.linalg.norm(pixel - aim_pixel) / (IMAGE_SIZE[1] / 2)
            if off_center > LOOK_MAX_OFF_CENTER:
                continue
            if not all(in_frame(camera, matrix, point) for point in also_see):
                continue
            ray = camera[:3, 3] - target
            obliquity = math.degrees(math.acos(ray[2] / np.linalg.norm(ray)))  # 0: camera straight above
            score = (obliquity / 30) ** 2 + (off_center / 0.3) ** 2 + ((in_camera[2] - distance_m) / 0.02) ** 2
            scored.append((score, (tip[0], tip[1], z, tilt)))
    scored.sort(key=lambda s: s[0])
    return [pose for _, pose in scored]


def in_frame(camera: np.ndarray, matrix: np.ndarray, point) -> bool:
    """Whether `point` is in front of `camera` and within LOOK_ALSO_SEE_MARGIN of the image center."""
    in_camera = np.linalg.inv(camera) @ np.append(np.asarray(point, dtype=float), 1.0)
    if in_camera[2] < 0.05:
        return False
    pixel = matrix[:2, :2] @ (in_camera[:2] / in_camera[2]) + matrix[:2, 2]
    half = np.array(IMAGE_SIZE) / 2
    return bool(np.all(np.abs(pixel - half) < LOOK_ALSO_SEE_MARGIN * half))
