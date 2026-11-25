"""Stack the cups into a 3-2-1 pyramid and back, finding them with the wrist camera.

The cups stand upside down, each with an ArUco marker on its base, facing up
(xarm_ik/cup_markers.py). Only uncovered cups show a marker, so the highest cup
seen is free to pick.

Up-stack:   look at the stack, pick its top cup (recording that level's height),
            place it in the next pyramid slot; until the stack is empty.
Down-stack: look at the pyramid, pick its highest cup, nest it on the stack at
            the height recorded for that level; until the pyramid is empty.

The stack can be anywhere on the ChArUco board: the first look keeps the whole
board in frame; later looks follow where the stack was seen. Picks and releases
use one gripper angle (--tilt: degrees from straight down, 90 = horizontal, or
"auto"); travel between them always uses auto. A transfer a fixed angle can't
do uses the nearest angle that can (logged as chosen=). Every transfer is
checked for reach before the arm moves; an unreachable one ends the phase.

Logs (--execute, or --log on a dry run) go to logs/analysis/cups_<time>_tilt<T>/
with --analysis, else logs/dev/ (dry runs: cupsdry_...):
feedback.csv and feedback_events.csv in the Project 2 format (xarm_ik/
feedback_log.py; `line` is the step, e.g. "up3 pick descend"; tilt events add
measured=, the gripper angle from the last joint reads), transfers.json
(each transfer's plan and where the tip got to), run.json (the outcome),
looks.json and the camera frames. With --analysis, the end of an --execute run (also
on Ctrl-C) asks where it failed and why, for run.json and analysis/project_3_results.csv.

--locate: "arm" places the camera by the measured joint angles and the hand-eye
calibration; "board" by the ChArUco board in view (falling back to the arm).
Both are compared and recorded whenever the board is seen.

Dry run by default: a simulated world stands in for the camera; --sim also
shows it in PyBullet. --execute moves the arm and uses the camera, saving each
look's annotated frame to its log folder; --show adds a live camera window.

Usage:
    uv run python scripts/stack_cups.py                     # dry run, simulated cups
    uv run python scripts/stack_cups.py --sim
    uv run python scripts/stack_cups.py --execute
    uv run python scripts/stack_cups.py --execute --show          # with the live camera view
    uv run python scripts/stack_cups.py --execute --phase look   # only look and report; no picks
    uv run python scripts/stack_cups.py --execute --phase down   # only take the pyramid down
    uv run python scripts/stack_cups.py --execute --tilt auto
    uv run python scripts/stack_cups.py --sim-stack 0.06 -0.24     # dry run with the simulated stack there (m)
    uv run python scripts/stack_cups.py --phase reach            # which angles reach what (no moves)
    uv run python scripts/stack_cups.py --execute --phase aim --tilt 60   # where the cup sits in the hand
"""

import argparse
import json
import math
import sys
import time

import numpy as np

from cup_stacking.layout import (BOTTOM_ROW, CUP_NEAR_M, FINGER_ENDS_OUT_M, GRIP_BELOW_TOP_M, HAND_EYE_PATH,
                                 PYRAMID_SLOTS, ROOT, STACK_NEST_STEP_M, Cup, board_points, stack_spots, stack_top_z)
from cup_stacking import results
from cup_stacking.planner import (APPROACH_STRAIGHT_M, APPROACH_STRAIGHT_PICK_M, HOVER_Z, REACH_TOLERANCE_M,
                                  GraspPlanner, Unreachable, tilt_name)
from cup_stacking.views import LOOK_DISTANCE_M, view_poses
from cup_stacking.vision import CameraVision, SimulatedVision
from run_csv import approach_vector, tilt_deg_of
from xarm_ik import ArmDisconnected, GripperPoseController, config
from xarm_ik.camera import load_intrinsics
from xarm_ik.feedback_log import FeedbackLog

LOG_DIR = ROOT / "logs"

GRIP_CLOSE = 0.35  # gripper close fraction holding a cup

# Upright releases: let go this far above the resting height, to drop the rest of the way.
BOTTOM_ROW_RELEASE_LIFT_M = 0.02  # onto the desk
STACK_RELEASE_LIFT_M = 0.015  # nesting onto the stack
# Desk releases no one tilt reaches: before leaning, drop from up to this high, and (the
# down-stack's first cup) move up to this far out from the base.
MAX_DESK_DROP_LIFT_M = 0.04
MAX_DESK_SHIFT_OUT_M = 0.06
# Down-stack picks may grasp up to this much lower for a better tilt (no cup nested inside).
# Lower, the closing fingers push the widening cup away.
DOWN_GRIP_EXTRA_MAX_M = 0.015

# Straight part of release approaches (see GraspPlanner.approach_path).
APPROACH_STRAIGHT_ON_CUPS_M = 0.03  # onto the pyramid's lower row
APPROACH_STRAIGHT_DESK_M = 0.01  # onto the bare desk
APPROACH_MAX_DIP_M = 0.01  # moving into an approach may dip this far below its ends

# First stack look: the whole board in frame (edges inset by a cup radius), a cup anywhere
# within FIRST_LOOK_RADIUS_M of its center taken.
FIRST_LOOK_RADIUS_M = 0.15
FIRST_LOOK_BOARD_INSET_M = 0.03
# Pyramid looks: the lower layers from farther back with the target high in the image;
# the top cup close and centered (seen from further off, its marker is too oblique).
PYRAMID_LOOK_DISTANCE_M = 0.25
PYRAMID_AIM_ROW = -0.4
PYRAMID_TOP_LOOK_DISTANCE_M = 0.18
PYRAMID_TOP_AIM_ROW = 0.0

# Move durations are base durations / MOTION_SPEEDUP; settle timeout scales grow with it to
# keep the same settle time.
MOTION_SPEEDUP = 1.0
MOVE_MS = round(1000 / MOTION_SPEEDUP)  # short moves
TRANSFER_MS = round(2000 / MOTION_SPEEDUP)  # between areas
GRIP_MS = round(600 / MOTION_SPEEDUP)
SETTLE_TIMEOUT_SCALE = 1 + MOTION_SPEEDUP * (1.5 - 1)
GRASP_SETTLE_TIMEOUT_SCALE = 1 + MOTION_SPEEDUP * (3.0 - 1)

# The last move down to a grasp or release settles to GRASP_SETTLE_TOLERANCE_DEG (~2 servo
# steps; 2 deg is ~1 cm at the tip), held for GRASP_SETTLE_HOLD_S so backlash sag shows,
# with GRASP_SETTLE_RETRIES more tries of GRASP_SETTLE_RETRY_S before the gripper acts.
GRASP_SETTLE_TOLERANCE_DEG = 0.5
GRASP_SETTLE_HOLD_S = 0.5
GRASP_SETTLE_RETRIES = 2
GRASP_SETTLE_RETRY_S = 1.5
GRASP_SETTLE_S = 0.4  # wait before measuring where the tip got to


def release_lift_m(slot_index: int) -> float:
    """How far above PYRAMID_SLOTS[slot_index] to let go of an upright cup."""
    return BOTTOM_ROW_RELEASE_LIFT_M if slot_index < BOTTOM_ROW else 0.0


def parse_tilt(text: str):
    """--tilt: degrees from straight down, or "auto"."""
    if text.lower() == "auto":
        return "auto"
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f'expected degrees or "auto", got {text!r}') from None


def chain_angles(angles_deg: dict, start_rad: np.ndarray) -> np.ndarray:
    """`start_rad` with the named joints set to `angles_deg`."""
    angles = start_rad.copy()
    for j, index in zip(config.JOINTS, config.ACTIVE_LINK_INDICES):
        angles[index] = math.radians(angles_deg[j.name])
    return angles


class CupStacker:
    def __init__(self, arm, vision, camera_in_gripper, on_tick, simulated: bool, tilt=90.0):
        self.arm, self.vision, self.on_tick, self.simulated = arm, vision, on_tick, simulated
        self.camera_in_gripper = camera_in_gripper
        self.planner = GraspPlanner(arm, tilt)
        self.matrix, _ = load_intrinsics()
        self.board_center = stack_spots()["center"]
        self.stack_xy = None  # last seen
        self.stack_home_xy = None  # first seen; the down-stack rebuilds the stack there
        self.stack_top_z = None  # believed, at stack_xy (None: empty)
        self.stack_seen = []  # grasp points of the cups taken off the stack, top first
        self.pyramid = []  # tops of the cups believed in the pyramid
        self.transfers = []  # records, for transfers.json
        self.held = None  # the held cup's grasp point in the tip frame
        self._view_cache = {}
        # Run log (Project 2 format): each move's row is labelled "<context> <step>".
        self.log: FeedbackLog | None = None
        self.context = ""  # the transfer, e.g. "up3"
        self.step = "home"
        self.requested_tilt = "any"  # "<deg|auto|any> [chosen=<deg>]", for the tilt events

    def on_move(self, timing, planned_ms):
        """The controller's on_move: log the move and its requested and achieved tilt."""
        if self.log is None:
            return
        line = f"{self.context} {self.step}".strip()
        self.log.log_move(self.log.row, line, planned_ms, timing)
        detail = f"requested={self.requested_tilt} achieved={tilt_deg_of(self.arm.get_approach_vector()):.2f}"
        last_read = self.arm.get_last_read_joint_angles_deg()
        if all(joint.name in last_read for joint in config.JOINTS):
            detail += f" measured={self.tilt_deg(last_read):.2f}"
        self.log.log_tilt(self.log.row, line, detail)
        self.log.row += 1

    def tilt_deg(self, angles_deg: dict) -> float:
        """The gripper's angle from straight down at these joint angles."""
        pose = self.arm.chain.forward_kinematics(chain_angles(angles_deg, np.zeros(len(self.arm.chain.links))))
        return tilt_deg_of(pose[:3, 2])

    @property
    def tilt(self):
        return self.planner.tilt

    def _move(self, x, y, z, tilt=None, close=None, ms=MOVE_MS, linear=True):
        """Move the tip at `tilt` (default self.tilt); `close`: gripper close fraction."""
        tilt = self.tilt if tilt is None else tilt
        self.requested_tilt = tilt if tilt == "auto" else f"{tilt:g}"
        approach = "auto" if tilt == "auto" else approach_vector(x, y, tilt)
        self.arm.move_to_pose(x, y, z, approach=approach, roll_deg=0.0,
                              gripper_open_fraction=None if close is None else 1.0 - close,
                              duration_ms=ms, linear=linear, position_tolerance_m=REACH_TOLERANCE_M,
                              on_tick=self.on_tick)

    def _rise(self):
        """Straight up to HOVER_Z."""
        x, y, z = self.arm.get_pose()
        if z < HOVER_Z - 0.005:
            self.step = "rise"
            self._move(x, y, HOVER_Z, tilt="auto")

    def transfer(self, phase: str, cup, grip, place, place_straight_m=APPROACH_STRAIGHT_M, **plan_options):
        """Pick the cup at grasp point `grip` and put it to rest at `place`, recording it.
        `plan_options`: see GraspPlanner.plan_transfer. Raises Unreachable before moving.
        Returns where the cup rests (its grasp point)."""
        record = {"phase": phase, "cup": int(cup.marker_id), "tilt": tilt_name(self.tilt),
                  "stack_xy": None if self.stack_xy is None else [round(float(v), 4) for v in self.stack_xy],
                  "grip": [round(float(v), 4) for v in grip],
                  "place": [round(float(v), 4) for v in place]}
        try:
            plan = self.planner.plan_transfer(grip, place, place_straight_m, **plan_options)
        except Unreachable as e:
            self.transfers.append({**record, "reachable": False, "error": str(e)})
            raise
        record.update(grip_depth_mm=round(plan.depth * 1000), release_lean_deg=plan.lean,
                      grip=[round(float(v), 4) for v in plan.grip], place=[round(float(v), 4) for v in plan.place],
                      reachable=True)
        self.transfers.append(record)  # before moving, so a stopped run still has it
        record["pick_tilt_deg"], record["pick_reached"] = self.pick(plan.grip, plan.pick_tilt, plan.depth)
        record["place_tilt_deg"], record["place_reached"] = self.place(plan.place, plan.place_tilt, place_straight_m)
        return plan.rest

    def believed_tops(self) -> list:
        """Tops of every cup believed out there."""
        stack = [] if self.stack_top_z is None or self.stack_xy is None else [np.array([*self.stack_xy, self.stack_top_z])]
        return stack + list(self.pyramid)

    def look_at(self, target, label: str, radius_m: float, obstacles=None, near_xy=None, also_see=(),
                distance_m=LOOK_DISTANCE_M, aim_row=0.0) -> list:
        """Point the camera at `target`; return the cups seen within `radius_m` (horizontally)
        of `near_xy` (default: the target). The tip stays above `obstacles` (default: the
        believed cups). `also_see`, `distance_m`, `aim_row`: see view_poses."""
        obstacles = self.believed_tops() if obstacles is None else obstacles
        near_xy = np.asarray(target[:2] if near_xy is None else near_xy, dtype=float)
        nearby = sorted(tuple(np.round(top, 2)) for top in obstacles
                        if np.linalg.norm(np.asarray(top[:2]) - np.asarray(target[:2])) < 0.15)
        also_see = [tuple(np.round(point, 3)) for point in also_see]
        key = (tuple(np.round(target, 2)), tuple(nearby), tuple(also_see), distance_m, aim_row)
        if key not in self._view_cache:
            self._view_cache[key] = view_poses(self.arm, self.camera_in_gripper, self.matrix, target, nearby,
                                               also_see, distance_m, aim_row)
            if not self._view_cache[key]:
                sys.exit(f"Error: no reachable pose sees {key[0]}.")
            x, y, z, tilt = self._view_cache[key][0]
            print(f"  ({label} view: tip ({x:.3f}, {y:.3f}, {z:.3f}) m, tilt {tilt:.0f} deg)")
        x, y, z, tilt = self._view_cache[key][0]
        self.vision.set_status(f"moving to look at {label}")
        self._rise()
        self.step = f"look {label}"
        self._move(x, y, z, tilt=tilt, close=0.0, ms=TRANSFER_MS, linear=False)  # fingers open, out of view
        cups, ignored = [], []
        for cup in self.vision.look(label):
            (cups if np.linalg.norm(cup.top[:2] - near_xy) < radius_m else ignored).append(cup)
        seen = ", ".join(f"{c.marker_id} at ({c.top[0]:.3f}, {c.top[1]:.3f}, {c.top[2]:.3f})" for c in cups)
        print(f"  looked at {label}: " + (seen if cups else "no cups"))
        for c in ignored:
            print(f"    (ignored cup {c.marker_id} at ({c.top[0]:.3f}, {c.top[1]:.3f}, {c.top[2]:.3f}): "
                  f"{np.linalg.norm(c.top[:2] - near_xy) * 100:.1f} cm from ({near_xy[0]:.3f}, {near_xy[1]:.3f}), "
                  f"over the {radius_m * 100:g} cm limit)")
        return cups

    def look_at_stack(self, top_z: float, obstacles=None, seen_xy=None) -> list:
        """Look for the stack's top at height `top_z` at `seen_xy`, or (None) anywhere on the
        board: whole board in frame, the tip clear of a full stack anywhere on it."""
        if seen_xy is not None:
            return self.look_at((*seen_xy, top_z), "stack", CUP_NEAR_M, obstacles=obstacles)
        anywhere = board_points(stack_top_z(len(PYRAMID_SLOTS)), FIRST_LOOK_BOARD_INSET_M, step=0.04)
        return self.look_at((*self.board_center, top_z), "stack", FIRST_LOOK_RADIUS_M,
                            obstacles=[*(self.believed_tops() if obstacles is None else obstacles), *anywhere],
                            near_xy=self.board_center, also_see=board_points(top_z, FIRST_LOOK_BOARD_INSET_M))

    def _to_grasp(self, grip, close=None, held=None, tilt=None, straight_m=APPROACH_STRAIGHT_PICK_M, depth=0.0) -> float:
        """Travel to above the grasp's tip target, switch to the approach path's arm
        configuration at HOVER_Z, then go down the path, settling tightly; returns the tilt.
        `held`, `depth`: see GraspPlanner.grasp_pose."""
        kind = "pick" if held is None else "place"
        tip, tilt = self.planner.grasp_pose(grip, held, tilt, depth)
        path = self.planner.approach_path(tip, tilt, straight_m)
        if path is None:
            raise Unreachable(f"no straight way down to ({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f}) at {tilt:g} deg")
        self._rise()
        self.step = f"{kind} travel"
        self._move(tip[0], tip[1], HOVER_Z, tilt="auto", close=close, ms=TRANSFER_MS)
        top = path[0]
        start_z = self.arm.get_pose()[2]
        end_z = self.arm.chain.forward_kinematics(chain_angles(top, np.zeros(len(self.arm.chain.links))))[2, 3]
        lowest = self._lowest_tip(self.arm._joint_angles_rad, top)
        if lowest < min(start_z, end_z) - APPROACH_MAX_DIP_M:
            raise Unreachable(f"getting into the approach would swing the hand down to {lowest * 100:.0f} cm "
                              f"(from {start_z * 100:.0f} cm, to {end_z * 100:.0f} cm)")
        self.step = f"{kind} approach"
        self.requested_tilt = f"{'auto' if self.tilt == 'auto' else f'{self.tilt:g}'} chosen={tilt:g}"
        self.arm.move_through([top], duration_ms=MOVE_MS, on_tick=self.on_tick)
        loose = self.arm.settle_tolerance_deg, self.arm.settle_timeout_scale, self.arm.settle_hold_s
        self.arm.settle_tolerance_deg = min(loose[0], GRASP_SETTLE_TOLERANCE_DEG)
        self.arm.settle_timeout_scale = max(loose[1], GRASP_SETTLE_TIMEOUT_SCALE)
        self.arm.settle_hold_s = GRASP_SETTLE_HOLD_S
        try:
            self.step = f"{kind} descend"
            self.arm.move_through(path[1:], duration_ms=MOVE_MS, on_tick=self.on_tick)
            self.step = f"{kind} settle"
            self.grasp_settle_traces = [self.arm.last_settle_trace]
            settled = self.arm.last_move is None or self.arm.last_move.settled is not False
            for attempt in range(GRASP_SETTLE_RETRIES):
                if settled:
                    break
                worst = max(self.arm.last_settle_residual_deg.items(), key=lambda item: abs(item[1]), default=("?", 0.0))
                print(f"    not settled ({worst[0]} {worst[1]:+.1f} deg off); settling again ({attempt + 1}/{GRASP_SETTLE_RETRIES})")
                settled = self.arm.settle(GRASP_SETTLE_RETRY_S, on_tick=self.on_tick)
                self.grasp_settle_traces.append(self.arm.last_settle_trace)
            self.grasp_settled = settled
        finally:
            self.arm.settle_tolerance_deg, self.arm.settle_timeout_scale, self.arm.settle_hold_s = loose
        return tilt

    def _lowest_tip(self, start_rad, end_deg: dict) -> float:
        """The lowest the tip gets on the joint-space move from `start_rad` to `end_deg`."""
        end = chain_angles(end_deg, start_rad)
        return min(self.arm.chain.forward_kinematics(start_rad + f * (end - start_rad))[2, 3]
                   for f in np.linspace(0, 1, 21))

    def reached(self, label: str) -> dict:
        """Where the tip is by the measured joints vs where it was commanded; printed,
        recorded and photographed."""
        commanded = self.arm.get_pose()
        if self.simulated:
            return {"commanded_tip": [round(float(v), 4) for v in commanded]}
        time.sleep(GRASP_SETTLE_S)
        measured_deg = {**self.arm.get_joint_angles_deg(), **self.arm.get_measured_joint_angles_deg()}
        measured = self.arm.chain.forward_kinematics(chain_angles(measured_deg, np.zeros(len(self.arm.chain.links))))[:3, 3]
        off = measured - commanded
        print(f"    {label}: tip measured {np.linalg.norm(off) * 1000:.0f} mm from commanded "
              f"(dx {off[0] * 1000:+.0f}, dy {off[1] * 1000:+.0f}, dz {off[2] * 1000:+.0f} mm)")
        self.vision.snapshot(label)
        return {"commanded_tip": [round(float(v), 4) for v in commanded],
                "measured_tip": [round(float(v), 4) for v in measured],
                "commanded_tilt_deg": round(tilt_deg_of(self.arm.get_approach_vector()), 2),
                "measured_tilt_deg": round(self.tilt_deg(measured_deg), 2),
                "measured_joints_deg": {name: round(float(v), 1) for name, v in measured_deg.items()},
                "off_mm": round(float(np.linalg.norm(off)) * 1000, 1),
                "dz_mm": round(float(off[2]) * 1000, 1),
                "settled": getattr(self, "grasp_settled", None),
                "settle_residual_deg": {name: round(float(v), 2) for name, v in self.arm.last_settle_residual_deg.items()},
                # Per try, per tick: [s, {joint: target - measured deg}, [joints nudged]].
                "settle_traces": [[[round(t, 2), {n: round(float(v), 2) for n, v in r.items()}, nudged]
                                   for t, r, nudged in trace] for trace in getattr(self, "grasp_settle_traces", [])]}

    def _from_grasp(self, kind: str):
        self.step = f"{kind} rise"
        x, y, _ = self.arm.get_pose()
        self._move(x, y, HOVER_Z, tilt="auto")

    def pick(self, grip, tilt=None, depth=0.0):
        """Grasp the cup at grasp point `grip` and rise; returns (tilt, reached())."""
        x, y, z = grip
        self.vision.set_status(f"picking at ({x:.3f}, {y:.3f}, {z:.3f}), {tilt_name(self.tilt)}")
        tilt = self._to_grasp(grip, close=0.0, tilt=tilt, depth=depth)
        # The held point as planned, not as reached: the release was planned with it.
        self.held = self.planner.hold(grip, self.planner.grasp_pose(grip, tilt=tilt, depth=depth)[0], tilt)
        reached = self.reached("grasp")
        self.step = "grip"
        self.arm.set_gripper(1.0 - GRIP_CLOSE, duration_ms=GRIP_MS, on_tick=self.on_tick)
        self._from_grasp("pick")
        return tilt, reached

    def place(self, grip, tilt=None, straight_m=APPROACH_STRAIGHT_M):
        """Put the held cup's grasp point at `grip`, let go and rise; returns (tilt, reached())."""
        x, y, z = grip
        self.vision.set_status(f"placing at ({x:.3f}, {y:.3f}, {z:.3f}), {tilt_name(self.tilt)}")
        tilt = self._to_grasp(grip, held=self.held, tilt=tilt, straight_m=straight_m)
        reached = self.reached("release")
        self.step = "release"
        self.arm.set_gripper(1.0, duration_ms=GRIP_MS, on_tick=self.on_tick)
        self._from_grasp("place")
        return tilt, reached

    def up_stack(self):
        print("Up-stack: stack -> pyramid")
        expected_top_z = stack_top_z(len(PYRAMID_SLOTS))
        self.stack_top_z = expected_top_z
        seen_xy = None
        for slot_index, slot in enumerate(PYRAMID_SLOTS):
            self.context = f"up{slot_index + 1}"
            cups = self.look_at_stack(expected_top_z, seen_xy=seen_xy)
            if not cups:
                print("  stack is empty")
                self.stack_top_z = None
                return
            cup = max(cups, key=lambda c: c.top[2])
            self.stack_top_z = cup.top[2]
            seen_xy = self.stack_xy = cup.top[:2].copy()
            if self.stack_home_xy is None:
                self.stack_home_xy = cup.top[:2].copy()
            grip = (cup.top[0], cup.top[1], cup.top[2] - GRIP_BELOW_TOP_M)
            print(f"  picking cup {cup.marker_id} at grip ({grip[0]:.3f}, {grip[1]:.3f}, {grip[2]:.3f}) "
                  f"-> pyramid slot ({slot[0]:.3f}, {slot[1]:.3f}, {slot[2]:.3f})")
            on_desk = slot_index < BOTTOM_ROW
            self.transfer("up", cup, grip, slot, APPROACH_STRAIGHT_DESK_M if on_desk else APPROACH_STRAIGHT_ON_CUPS_M,
                          drop_lift_m=release_lift_m(slot_index),
                          max_drop_lift_m=MAX_DESK_DROP_LIFT_M if on_desk else 0.0)
            self.stack_seen.append(grip)
            expected_top_z = cup.top[2] - STACK_NEST_STEP_M
            self.stack_top_z = expected_top_z
            self._sim(lambda: self.vision.picked(cup))
            self.pyramid.append(np.array([slot[0], slot[1], slot[2] + GRIP_BELOW_TOP_M]))
            self._sim(lambda: self.vision.placed(cup, slot))

    def down_stack(self):
        print("Down-stack: pyramid -> stack")
        # Bottom first, where the stack first stood, at the heights seen unstacking
        # (no up-stack this run: nominal heights at the board's far side).
        stack_xy = stack_spots()["far"] if self.stack_home_xy is None else self.stack_home_xy
        levels = [(*stack_xy, z) for _, _, z in reversed(self.stack_seen)] or [
            (*stack_xy, stack_top_z(n + 1) - GRIP_BELOW_TOP_M) for n in range(len(PYRAMID_SLOTS))]
        self.stack_xy = stack_xy
        if not self.pyramid:  # not built this run: assume it's full
            self.pyramid = [np.array([x, y, z + GRIP_BELOW_TOP_M]) for x, y, z in PYRAMID_SLOTS]
        below = None  # top of the last cup put on the stack, as seen after letting go
        for level_index, level in enumerate(levels):
            if below is not None:
                # Onto the last cup as it landed, nesting as deep as unstacking saw.
                level = (below[0], below[1], below[2] - GRIP_BELOW_TOP_M + level[2] - levels[level_index - 1][2])
            self.context = f"down{level_index + 1}"
            cups = self.look_at_pyramid()
            if not cups:
                print("  pyramid is empty")
                return
            top_z = max(c.top[2] for c in cups)
            # Highest layer; in it, the cup nearest the base.
            cup = min((c for c in cups if c.top[2] > top_z - 0.03), key=lambda c: np.linalg.norm(c.top[:2]))
            cup = self.look_closer(cup)
            top = cup.top.copy()
            if self.stack_seen and abs(top[2] - self.desk_cup_top_z) < 0.03:
                top[2] = self.desk_cup_top_z  # on the desk: as high as the stack's bottom cup was seen
            grip = (top[0], top[1], top[2] - GRIP_BELOW_TOP_M)
            print(f"  picking cup {cup.marker_id} at grip ({grip[0]:.3f}, {grip[1]:.3f}, {grip[2]:.3f}) "
                  f"-> stack level ({level[0]:.3f}, {level[1]:.3f}, {level[2]:.3f})")
            first = level_index == 0  # onto the desk
            level = self.transfer("down", cup, grip, level, APPROACH_STRAIGHT_DESK_M if first else APPROACH_STRAIGHT_M,
                                  max_extra_depth_m=DOWN_GRIP_EXTRA_MAX_M,
                                  drop_lift_m=BOTTOM_ROW_RELEASE_LIFT_M if first else STACK_RELEASE_LIFT_M,
                                  max_drop_lift_m=MAX_DESK_DROP_LIFT_M if first else 0.0,
                                  max_shift_out_m=MAX_DESK_SHIFT_OUT_M if first else 0.0)
            self.stack_xy = (level[0], level[1])
            self.pyramid = [top for top in self.pyramid if np.linalg.norm(top[:2] - cup.top[:2]) > CUP_NEAR_M / 2]
            self._sim(lambda: self.vision.picked(cup))
            self.stack_top_z = level[2] + GRIP_BELOW_TOP_M
            self._sim(lambda: self.vision.placed(cup, level))
            if level_index + 1 < len(levels):
                below = self.look_at_placed(cup, level)
        cups = self.look_at_pyramid()
        if cups:
            print(f"  warning: {len(cups)} cup(s) still in the pyramid")

    def look_at_placed(self, cup, level):
        """Where `cup`'s top landed after letting go at grasp point `level`, or None if unseen."""
        planned_top = np.array([level[0], level[1], level[2] + GRIP_BELOW_TOP_M])
        seen = self.look_at_stack(planned_top[2], seen_xy=planned_top[:2])
        same = [c for c in seen if c.marker_id == cup.marker_id]
        if not same:
            print(f"    warning: cup {cup.marker_id} not seen on the stack after letting go; "
                  "the next goes where it was let go")
            return None
        top = same[0].top
        off = (top - planned_top) * 1000
        print(f"    cup {cup.marker_id} landed {np.linalg.norm(off[:2]):.0f} mm from where it was let go "
              f"(dx {off[0]:+.0f}, dy {off[1]:+.0f}, dz {off[2]:+.0f} mm)")
        self.stack_xy, self.stack_top_z = top[:2].copy(), float(top[2])
        return top.copy()

    def look_around(self):
        """Report the cups seen at the stack and at each pyramid layer; no picks."""
        print("Looking (no picks):")
        full_pyramid = [np.array([x, y, z + GRIP_BELOW_TOP_M]) for x, y, z in PYRAMID_SLOTS]
        for count in (len(PYRAMID_SLOTS), 1):
            self.look_at_stack(stack_top_z(count), obstacles=[])
        for first, last in ((5, 6), (3, 5), (0, 3)):
            target = np.mean(full_pyramid[first:last], axis=0)
            self.look_at(target, "pyramid", radius_m=0.16, obstacles=full_pyramid, **self._pyramid_view(target))

    def aim(self, wait):
        """Go to grasp the stack's top cup with the fingers open, hold until `wait()` returns,
        and back off without grasping."""
        print("Aiming (no grasp):")
        cups = self.look_at_stack(stack_top_z(len(PYRAMID_SLOTS)))
        if not cups:
            print("  no cup seen at the stack")
            return
        cup = max(cups, key=lambda c: c.top[2])
        grip = (cup.top[0], cup.top[1], cup.top[2] - GRIP_BELOW_TOP_M)
        self.planner.check_reach(pick=grip)
        tip, tilt = self.planner.grasp_pose(grip)
        print(f"  cup {cup.marker_id}: top ({cup.top[0]:.3f}, {cup.top[1]:.3f}, {cup.top[2]:.3f}); grasp point "
              f"({grip[0]:.3f}, {grip[1]:.3f}, {grip[2]:.3f}) -> fingertip point ({tip[0]:.3f}, {tip[1]:.3f}, "
              f"{tip[2]:.3f}) at {tilt:g} deg")
        self.vision.set_status(f"aiming at cup {cup.marker_id} (no grasp)")
        self._to_grasp(grip, close=0.0)
        print(f"  holding: the cup's center should sit where the closing fingers would center it, "
              f"{GRIP_BELOW_TOP_M * 100:g} cm under its top.")
        wait()
        self._from_grasp("pick")

    def reach_table(self, tilts):
        """Print which picks and places each gripper angle reaches (no moves)."""
        rows = []
        for spot, xy in stack_spots().items():
            for cups in (len(PYRAMID_SLOTS), 1):
                rows.append((f"stack@{spot} {cups} cup{'s' if cups > 1 else ''}",
                             (*xy, stack_top_z(cups) - GRIP_BELOW_TOP_M)))
        for i, slot in enumerate(PYRAMID_SLOTS):
            rows.append((f"pyramid slot {i}", (slot[0], slot[1], slot[2] + release_lift_m(i))))
        names = [tilt_name(t) for t in tilts]
        print(f"Miss (mm) per grasp point (as picks: finger ends {GRIP_BELOW_TOP_M * 1000:g} mm under the top, "
              f"{FINGER_ENDS_OUT_M * 1000:g} mm out) and gripper angle; '.' = reached (within "
              f"{REACH_TOLERANCE_M * 1000:g} mm). Auto's chosen angle in brackets.")
        print(f"{'':24s}" + "".join(f"{n:>11s}" for n in names))
        for label, point in rows:
            cells = []
            for tilt in tilts:
                tip, chosen = self.planner.grasp_pose(point, tilt=tilt)
                miss = self.planner.miss_m(tip, chosen)
                cell = "." if miss <= REACH_TOLERANCE_M else f"{miss * 1000:.0f}"
                if tilt == "auto":
                    cell = f"{cell} [{chosen:.0f}]"
                cells.append(cell)
            print(f"{label:24s}" + "".join(f"{c:>11s}" for c in cells))

    @staticmethod
    def _pyramid_view(target) -> dict:
        """look_at's distance_m and aim_row for a pyramid look at `target`."""
        middle_top_z = PYRAMID_SLOTS[3][2] + GRIP_BELOW_TOP_M
        if target[2] > middle_top_z + 0.03:
            return {"distance_m": PYRAMID_TOP_LOOK_DISTANCE_M, "aim_row": PYRAMID_TOP_AIM_ROW}
        return {"distance_m": PYRAMID_LOOK_DISTANCE_M, "aim_row": PYRAMID_AIM_ROW}

    @property
    def desk_cup_top_z(self) -> float:
        """The top of a cup on the desk: the stack's bottom cup, as seen unstacking."""
        return self.stack_seen[-1][2] + GRIP_BELOW_TOP_M

    def look_closer(self, cup):
        """`cup` as seen by a close, centered look (a layer look sees it off to the side), or
        `cup` unchanged if not seen."""
        seen = self.look_at(cup.top, "pyramid_cup", CUP_NEAR_M / 2)
        same = [c for c in seen if c.marker_id == cup.marker_id]
        if not same:
            print(f"    cup {cup.marker_id} not seen close up: using the pyramid look")
            return cup
        return same[0]

    def look_at_pyramid(self) -> list:
        """Look at the highest pyramid layer believed to hold cups (or the pyramid's middle);
        cups seen there join the belief."""
        if self.pyramid:
            top_z = max(top[2] for top in self.pyramid)
            target = np.mean([top for top in self.pyramid if top[2] > top_z - 0.03], axis=0)
        else:
            target = np.mean([np.array([x, y, z + GRIP_BELOW_TOP_M]) for x, y, z in PYRAMID_SLOTS], axis=0)
        cups = self.look_at(target, "pyramid", radius_m=0.16, **self._pyramid_view(target))
        for cup in cups:
            if all(np.linalg.norm(top[:2] - cup.top[:2]) > CUP_NEAR_M / 2 or abs(top[2] - cup.top[2]) > 0.03
                   for top in self.pyramid):
                self.pyramid.append(cup.top)
        return cups

    def _sim(self, update):
        if self.simulated:
            update()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="move the real arm and use the camera")
    parser.add_argument("--sim", action="store_true", help="show the motion in PyBullet")
    parser.add_argument("--phase", choices=("both", "up", "down", "look", "aim", "reach"), default="both",
                        help="both, up (stack -> pyramid), down (pyramid -> stack), look (report what's seen; "
                             "no picks), aim (go to grasp the stack's top cup, fingers open, and hold; no grasp), "
                             "or reach (print what each gripper angle reaches; no moves); default: both")
    parser.add_argument("--sim-stack", type=float, nargs=2, metavar=("X", "Y"), default=None,
                        help="dry run: where the simulated stack stands, m in the base frame (default: the board center)")
    parser.add_argument("--tilt", type=parse_tilt, default=90.0,
                        help='gripper angle for picks and places: degrees from straight down (90 = horizontal), '
                             'or "auto" (default: 90)')
    parser.add_argument("--cups", type=int, default=6, help="dry run: cups in the simulated stack (default: 6)")
    parser.add_argument("--show", action="store_true",
                        help="with --execute: a live window of what the camera sees, markers outlined")
    parser.add_argument("--locate", choices=("arm", "board"), default="arm",
                        help="place the camera by the arm's kinematics (default) or by the ChArUco board in view "
                             "(falls back to the arm where the board isn't seen); both are compared when possible")
    parser.add_argument("--open-loop", action="store_true", help="skip closed-loop settling after moves")
    parser.add_argument("--analysis", action="store_true",
                        help="with --execute: at the end, ask where the run failed and why, and record it in "
                             "analysis/project_3_results.csv")
    parser.add_argument("--log", action="store_true",
                        help="dry run: also write the run's logs (to logs/dev/cupsdry_<time>_tilt<T>/); --execute always does")
    parser.add_argument("--device", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--com-port", default="USB", help="passed to xarm.Controller (default: USB)")
    args = parser.parse_args()
    if args.analysis and not args.execute:
        parser.error("--analysis needs --execute (it records real runs)")
    if args.show and not args.execute:
        parser.error("--show needs --execute (a dry run has no camera)")

    camera_in_gripper = np.array(json.loads(HAND_EYE_PATH.read_text())["camera_in_gripper"])
    if args.phase == "reach":
        arm = GripperPoseController(dry_run=True)
        stacker = CupStacker(arm, SimulatedVision(0, (0, 0)), camera_in_gripper, None, simulated=True)
        stacker.reach_table([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 100.0, "auto"])
        return
    print(f"Picks and places at {tilt_name(args.tilt)}; the stack can be anywhere on the board")
    session = log = None
    if args.execute or args.log:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        tilt_tag = "auto" if args.tilt == "auto" else f"{args.tilt:g}"
        folder = LOG_DIR / ("analysis" if args.analysis else "dev")
        session = folder / f"{'cups' if args.execute else 'cupsdry'}_{stamp}_tilt{tilt_tag}"
        session.mkdir(parents=True)
        log = FeedbackLog(session / "feedback.csv")
        print(f"Logging to {session}")
    arm = GripperPoseController(com_port=args.com_port, dry_run=not args.execute, closed_loop=not args.open_loop,
                                settle_timeout_scale=SETTLE_TIMEOUT_SCALE, on_feedback=log)
    if args.execute:
        try:
            vision = CameraVision(arm, camera_in_gripper, session, args.device, args.show, args.locate)
        except RuntimeError as e:
            sys.exit(f"Error: {e}")
    else:
        sim_xy = stack_spots()["center"] if args.sim_stack is None else np.array(args.sim_stack)
        print(f"Simulated stack at ({sim_xy[0]:.3f}, {sim_xy[1]:.3f})")
        vision = SimulatedVision(args.cups if args.phase != "down" else 0, sim_xy)
        if args.phase == "down":  # a full pyramid to take down
            for i, slot in enumerate(PYRAMID_SLOTS):
                vision.placed(Cup(i, np.zeros(3)), slot)

    ticks = []
    if getattr(vision, "view", None):
        ticks.append(vision.view.tick)
    if args.sim:
        from xarm_ik.sim import PybulletArmView
        ticks.append(PybulletArmView().mirror_tick)
    on_tick = (lambda *tick: [callback(*tick) for callback in ticks]) if ticks else None

    stacker = CupStacker(arm, vision, camera_in_gripper, on_tick, simulated=not args.execute, tilt=args.tilt)
    stacker.log = log
    arm.on_move = stacker.on_move
    summary = {"tilt": tilt_name(args.tilt), "phase": args.phase, "executed": args.execute, "locate": args.locate,
               "closed_loop": not args.open_loop, "start": time.strftime("%Y-%m-%d %H:%M:%S"), "stopped": None}
    start_s = time.perf_counter()
    interrupted, ended_at = False, ("", "")
    try:
        print("Homing...")
        arm.home(on_tick=on_tick)
        try:
            if args.phase in ("both", "up"):
                stacker.up_stack()
            if args.phase in ("both", "down"):
                stacker.down_stack()
            if args.phase == "look":
                stacker.look_around()
            if args.phase == "aim":
                stacker.aim((lambda: input("  Press Enter to back off... ")) if args.execute else (lambda: None))
        except Unreachable as e:
            summary["stopped"] = f"unreachable: {e}"
            print(f"\n*** STOPPED, nothing picked: {e}.\n*** Try another --tilt (see --phase reach).\n")
        except KeyboardInterrupt:  # stop where it is; no homing
            interrupted = True
            summary["stopped"] = "Ctrl-C"
            print("\n*** STOPPED (Ctrl-C); the arm stays where it is.")
        except BaseException as e:  # an error: record it, then let it through
            summary["stopped"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            ended_at = (stacker.context, stacker.step)
            if stacker.transfers:
                done = sum("place_reached" in t for t in stacker.transfers)
                print(f"{done} transfer(s) done at {tilt_name(args.tilt)}"
                      + ("" if done == len(stacker.transfers) else ", then one not finished"))
            if session:
                write_run_files(session, stacker, summary, time.perf_counter() - start_s)
        if not interrupted:
            stacker._rise()
            print("Homing...")
            vision.set_status("homing")
            stacker.context, stacker.step, stacker.requested_tilt = "", "home", "any"
            arm.home(duration_ms=2000, on_tick=on_tick)
    except KeyboardInterrupt:
        print("\n*** STOPPED (Ctrl-C) while homing.")
    except ArmDisconnected as e:
        summary["stopped"] = f"arm disconnected: {e}"
        sys.exit(f"\n*** STOPPED, arm disconnected: {e}\n*** Logs (if any) are in the run folder above.")
    finally:
        vision.close()
        if log:
            print(f"Joint feedback: {log.close()}")
        phases = {"both": ("up", "down"), "up": ("up",), "down": ("down",)}.get(args.phase)
        if args.analysis and args.execute and session and phases:
            results.ask(session, summary, phases, *ended_at)
    if not args.execute:
        print("Dry run done (simulated cups). Add --execute to run it on the arm.")


def write_run_files(session, stacker: CupStacker, summary: dict, duration_s: float):
    """transfers.json (every transfer) and run.json (the run's outcome)."""
    transfers = stacker.transfers
    (session / "transfers.json").write_text(json.dumps(transfers, indent=2) + "\n")
    requested = None if stacker.tilt == "auto" else float(stacker.tilt)
    summary.update(
        duration_s=round(duration_s, 1),
        stack_home_xy=None if stacker.stack_home_xy is None else [round(float(v), 4) for v in stacker.stack_home_xy],
        transfers_planned=len(transfers),
        transfers_done=sum("place_reached" in t for t in transfers),
        up_done=sum("place_reached" in t for t in transfers if t["phase"] == "up"),
        down_done=sum("place_reached" in t for t in transfers if t["phase"] == "down"),
        off_angle=None if requested is None else sum(
            "pick_tilt_deg" in t and abs(t["pick_tilt_deg"] - requested) > 1e-6 for t in transfers),
        leaning_releases=sum(t.get("release_lean_deg", 0) > 0 for t in transfers),
        deeper_grasps=sum(t.get("grip_depth_mm", 0) > 0 for t in transfers),
        settle_timeouts=sum(t.get(key, {}).get("settled") is False for t in transfers
                            for key in ("pick_reached", "place_reached")),
    )
    (session / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {session / 'transfers.json'} and {session / 'run.json'}")


if __name__ == "__main__":
    main()
