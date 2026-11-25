"""Where things are: cup geometry, the pyramid's slots, the stack and the board (base frame, m).

Positions are in the calibrated arm model (joint_offsets.json).
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xarm_ik import charuco_board

ROOT = Path(__file__).resolve().parent.parent
HAND_EYE_PATH = ROOT / "camera_hand_eye.json"  # camera placement, board poses, fingertip offset

# Closed finger ends in the tip frame (touch calibration; 0 = the URDF tip point).
FINGER_ENDS_IN_TIP = np.array(json.loads(HAND_EYE_PATH.read_text()).get("fingertip_offset_m", [0.0, 0.0, 0.0]))

CUP_HEIGHT_M = 0.063  # rim to base
CUP_RIM_RADIUS_M = 0.035  # wide end, on the desk
CUP_TOP_RADIUS_M = 0.025  # narrow end, marker up

# A grasp point is on the cup's center line, GRIP_BELOW_TOP_M under its top (the marker).
# The finger ends go FINGER_ENDS_OUT_M past it, away from the base. Tune with --phase aim.
GRIP_BELOW_TOP_M = 0.005
FINGER_ENDS_OUT_M = 0.022
# Grasping `depth` lower moves the finger ends back toward the base by the cup's taper.
FINGER_ENDS_BACK_PER_DEPTH = (CUP_RIM_RADIUS_M - CUP_TOP_RADIUS_M) / CUP_HEIGHT_M

# Cup tops at the pyramid's slots, in fill order: bottom row, middle row, top.
PYRAMID_SLOT_TOPS = [
    (-0.2411, 0.0768, 0.0464), (-0.2411, 0.0025, 0.0464), (-0.2411, -0.0718, 0.0464),
    (-0.2420, 0.0397, 0.1420), (-0.2420, -0.0347, 0.1420),
    (-0.2420, 0.0025, 0.2238),
]
PYRAMID_SLOTS = [(x, y, top - GRIP_BELOW_TOP_M) for x, y, top in PYRAMID_SLOT_TOPS]  # as grasp points
BOTTOM_ROW = 3  # the first this-many slots stand on the desk

# Nominal stack: a lone cup's top, and the rise per nested cup.
STACK_BOTTOM_TOP_Z = 0.0515
STACK_NEST_STEP_M = 0.017

CUP_NEAR_M = 0.06  # cups closer than this (horizontally) are at the same spot
FAR_SIDE_OFFSET_M = 0.07  # stack_spots()["far"]: past the board center, away from the base


@dataclass
class Cup:
    """A seen (or simulated) cup: marker id and its top center in the base frame."""

    marker_id: int
    top: np.ndarray


def board_in_base() -> np.ndarray:
    """Board 0's pose (about its center), from the hand-eye calibration."""
    return np.array(json.loads(HAND_EYE_PATH.read_text())["board_in_base"])


def stack_spots() -> dict:
    """Sample stack spots (x, y): the board center and its far side."""
    center = board_in_base()[:2, 3]
    far = center + FAR_SIDE_OFFSET_M * center / np.linalg.norm(center)
    return {"center": center, "far": far}


def board_points(z: float, inset: float, step: float | None = None) -> list:
    """Points (x, y, z) over board 0 at height `z`, its edges brought in by `inset`: its four
    corners, or (with `step`) a grid that far apart."""
    pose = board_in_base()
    board = charuco_board.make_board()
    width, height = board.getChessboardSize()
    half = np.array([width, height]) * board.getSquareLength() / 2 - inset
    if step is None:
        local = [(sx * half[0], sy * half[1]) for sx in (-1, 1) for sy in (-1, 1)]
    else:
        local = [(u, v) for u in np.arange(-half[0], half[0] + 1e-9, step) for v in np.arange(-half[1], half[1] + 1e-9, step)]
    return [(*(pose @ [u, v, 0.0, 1.0])[:2], z) for u, v in local]


def stack_top_z(cups: int) -> float:
    """Nominal top of a stack of `cups` nested cups."""
    return STACK_BOTTOM_TOP_Z + (cups - 1) * STACK_NEST_STEP_M


def lean_lift_m(lean_deg: float, depth_m: float = 0.0) -> float:
    """How much higher to let go of a cup leaning `lean_deg` than an upright one, so its lowest
    rim point is where the upright rim would be. `depth_m`: grasped that much below GRIP_BELOW_TOP_M."""
    lean, below = math.radians(abs(lean_deg)), CUP_HEIGHT_M - GRIP_BELOW_TOP_M - depth_m  # grasp point to rim
    return below * math.cos(lean) + CUP_RIM_RADIUS_M * math.sin(lean) - below
