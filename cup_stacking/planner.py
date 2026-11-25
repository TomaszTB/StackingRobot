"""Grasp planning: tip poses for picks and releases, the tilt(s) a transfer uses, and the
vertical approach paths down to them. Plans only; nothing here moves the arm."""

import math
from dataclasses import dataclass

import numpy as np

from xarm_ik import GripperPoseController, config

from .layout import FINGER_ENDS_BACK_PER_DEPTH, FINGER_ENDS_IN_TIP, FINGER_ENDS_OUT_M, lean_lift_m

REACH_TOLERANCE_M = 0.005  # a pose missing by more than this is unreachable
HOVER_Z = 0.235  # travel height, ~2 cm over the pyramid top
SHOULDER_MIN_DEG = -88.0  # grasps and releases; the shoulder doesn't physically get past ~-90

# Approach paths: vertical, one arm configuration, 1 cm steps of at most DESCENT_MAX_STEP_DEG
# per joint, up to HOVER_Z (at least APPROACH_MIN_RISE_M). The tilt may change by up to
# APPROACH_TILT_STEP_DEG per step, except over the lowest `straight_m`.
DESCENT_MAX_STEP_DEG = 15.0
APPROACH_MIN_RISE_M = 0.03
APPROACH_TILT_STEP_DEG = 10.0
APPROACH_STRAIGHT_M = 0.04  # default straight part: a held cup nests ~4.5 cm over the one below
APPROACH_STRAIGHT_PICK_M = 0.03  # picks: the fingers clear the rim

# A release at another tilt than its pick lets the cup go leaning by the difference. It tips
# back onto its rim up to ~45 deg.
MAX_RELEASE_LEAN_DEG = 40.0

GRIP_DEPTH_STEP_M = 0.005
DROP_LIFT_STEP_M = 0.005  # also the step for moving a release out


class Unreachable(Exception):
    """A pick or place the gripper can't reach."""


def tilt_name(tilt) -> str:
    return "auto" if tilt == "auto" else f"{tilt:g} deg"


@dataclass
class TransferPlan:
    grip: tuple  # grasp point at pick (depth applied)
    place: tuple  # grasp point at release
    rest: tuple  # where the cup ends up (its grasp point, on what's under it)
    pick_tip: np.ndarray
    pick_tilt: float
    place_tilt: float
    held: np.ndarray  # grasp point in the tip frame
    depth: float  # extra grasp depth below GRIP_BELOW_TOP_M
    lean: float  # deg the cup leans at release


class GraspPlanner:
    """`tilt`: picks and releases at this many degrees from straight down, or "auto"."""

    def __init__(self, arm: GripperPoseController, tilt=90.0):
        self.arm, self.tilt = arm, tilt

    def _tip_rotation(self, tip, tilt: float) -> np.ndarray:
        """The tip frame's rotation with the tip at `tip` at this tilt (leaning radially, roll 0)."""
        solutions = self.arm._planar.solve(np.asarray(tip, float), [math.radians(tilt)],
                                           turned_around=config.ALLOW_TURNED_AROUND)
        best = int(np.argmin(solutions.error_m[0]))
        angles = np.zeros(len(self.arm.chain.links))
        angles[list(config.ACTIVE_LINK_INDICES[:4])] = solutions.joints_rad[0, best]
        return self.arm.chain.forward_kinematics(angles)[:3, :3]

    def _tip_for(self, target, in_tip, tilt: float) -> np.ndarray:
        """The tip position putting the tip-frame point `in_tip` at `target` at this tilt."""
        target = np.asarray(target, float)
        tip = target.copy()
        for _ in range(4):  # the rotation depends slightly on the tip position
            tip = target - self._tip_rotation(tip, tilt) @ in_tip
        return tip

    def grasp_pose(self, point, held=None, tilt=None, depth=0.0) -> tuple:
        """(tip target, tilt) for a grasp point. Picks (`held` None): the finger ends go
        FINGER_ENDS_OUT_M out from `point`, less FINGER_ENDS_BACK_PER_DEPTH * `depth`.
        Releases: the held grasp point (`held`, in the tip frame; see hold()) goes to `point`.
        At `tilt`, else self.tilt; for auto, the tilt auto picks for the resulting tip."""
        point = np.asarray(point, float)
        if held is None:
            outward = np.append(point[:2] / np.linalg.norm(point[:2]), 0.0)
            out = FINGER_ENDS_OUT_M - FINGER_ENDS_BACK_PER_DEPTH * depth
            target, in_tip = point + out * outward, FINGER_ENDS_IN_TIP
        else:
            target, in_tip = point, np.asarray(held, float)
        tilt = self.tilt if tilt is None else tilt
        if tilt != "auto":
            return self._tip_for(target, in_tip, tilt), float(tilt)
        tilt = self.arm.choose_tilt(target).tilt_deg
        for _ in range(4):
            tilt = self.arm.choose_tilt(self._tip_for(target, in_tip, tilt)).tilt_deg
        return self._tip_for(target, in_tip, tilt), round(float(tilt), 1)

    def hold(self, grip, tip, tilt: float) -> np.ndarray:
        """Where grasp point `grip` sits in the tip frame with the tip at `tip` at this tilt."""
        return self._tip_rotation(tip, tilt).T @ (np.asarray(grip, float) - np.asarray(tip, float))

    def miss_m(self, point, tilt=None) -> float:
        """How far the tip misses `point` at this tilt (default self.tilt)."""
        tilt = self.tilt if tilt is None else tilt
        if tilt == "auto":
            return float(self.arm.choose_tilt(point).error_m)
        return float(self.arm._planar.solve(point, [math.radians(tilt)],
                                            turned_around=config.ALLOW_TURNED_AROUND).error_m.min())

    def _reaches(self, tip, tilt: float, straight_m: float) -> bool:
        """`tip` and its hover are reachable, with an approach path down to it."""
        return (self.miss_m(tip, tilt) <= REACH_TOLERANCE_M
                and self.miss_m((*tip[:2], HOVER_Z), "auto") <= REACH_TOLERANCE_M
                and self.approach_path(tip, tilt, straight_m) is not None)

    def check_reach(self, held=None, tilt=None, depth=0.0, **grasps):
        """Raise Unreachable unless each grasp (name=grasp point; "place" uses `held`) and its
        hover are within reach."""
        for name, point in grasps.items():
            tip, tilt = self.grasp_pose(point, held if name == "place" else None, tilt, depth)
            for label, target, at in ((name, tip, tilt), (f"{name} hover", (*tip[:2], HOVER_Z), "auto")):
                miss = self.miss_m(target, at)
                if miss > REACH_TOLERANCE_M:
                    raise Unreachable(f"{label} at ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) is out of reach "
                                      f"at {tilt_name(self.tilt)} (misses by {miss * 1000:.0f} mm)")

    def _preferred_tilt(self, grip, depth: float) -> float:
        """The tilt transfers stay nearest to: self.tilt, or for auto, auto's pick tilt."""
        return float(self.tilt) if self.tilt != "auto" else self.grasp_pose(grip, depth=depth)[1]

    def _tilts_near(self, preferred: float, max_deviation=None) -> list:
        """Candidate tilts (whole degrees, plus a fixed self.tilt), nearest `preferred` first."""
        tilts = {*np.arange(*config.AUTO_TILT_RANGE_DEG, 1.0), *([] if self.tilt == "auto" else [float(self.tilt)])}
        return sorted((float(t) for t in tilts if max_deviation is None or abs(t - preferred) <= max_deviation + 1e-9),
                      key=lambda t: abs(t - preferred))

    def transfer_tilt(self, grip, place, straight_m=APPROACH_STRAIGHT_M, depth=0.0, max_deviation=None):
        """One tilt for both the pick at `grip` and the release at `place` (the cup turns with
        the hand): the reaching tilt nearest _preferred_tilt, within `max_deviation`, or None."""
        for tilt in self._tilts_near(self._preferred_tilt(grip, depth), max_deviation):
            pick_tip, _ = self.grasp_pose(grip, tilt=tilt, depth=depth)
            place_tip, _ = self.grasp_pose(place, self.hold(grip, pick_tip, tilt), tilt=tilt)
            if self._reaches(pick_tip, tilt, APPROACH_STRAIGHT_PICK_M) and self._reaches(place_tip, tilt, straight_m):
                return float(tilt)
        return None

    def leaning_tilts(self, grip, place, straight_m=APPROACH_STRAIGHT_M, depth=0.0, max_lean=MAX_RELEASE_LEAN_DEG):
        """Pick and release tilts differing least (at most `max_lean`), the release raised by
        lean_lift_m; ties go to the pick tilt nearest _preferred_tilt. (pick tilt, release
        tilt, release point, lean), or None."""
        tilts = np.arange(*config.AUTO_TILT_RANGE_DEG, 1.0)
        picks = []  # (tilt, held), nearest the preferred tilt first
        for tilt in self._tilts_near(self._preferred_tilt(grip, depth)):
            tip, _ = self.grasp_pose(grip, tilt=tilt, depth=depth)
            if self._reaches(tip, tilt, APPROACH_STRAIGHT_PICK_M):
                picks.append((float(tilt), self.hold(grip, tip, tilt)))
        for lean in np.arange(1.0, max_lean + 0.5, 1.0):
            at = (place[0], place[1], place[2] + lean_lift_m(lean, depth))
            for pick_tilt, held in picks:
                for tilt in (pick_tilt - lean, pick_tilt + lean):
                    if not tilts[0] <= tilt <= tilts[-1]:
                        continue
                    tip, _ = self.grasp_pose(at, held, tilt=tilt)
                    if self._reaches(tip, tilt, straight_m):
                        return pick_tilt, float(tilt), at, float(lean)
        return None

    def plan_transfer(self, grip, place, straight_m=APPROACH_STRAIGHT_M, max_extra_depth_m=0.0, drop_lift_m=0.0,
                      max_drop_lift_m=0.0, max_shift_out_m=0.0) -> TransferPlan:
        """Plan moving the cup with grasp point `grip` to rest with it at `place`. Upright if
        one tilt serves both ends, trying in order: the release `drop_lift_m` up (up to
        `max_drop_lift_m`), moved out from the base (up to `max_shift_out_m`), and for each the
        shallowest grasp (up to `max_extra_depth_m` deeper). A fixed tilt is tried exactly
        first, then the nearest that works. Else leaning least, let go lean_lift_m up.
        Raises Unreachable."""
        depths = np.arange(0.0, max_extra_depth_m + 1e-9, GRIP_DEPTH_STEP_M)
        lifts = np.arange(drop_lift_m, max(drop_lift_m, max_drop_lift_m) + 1e-9, DROP_LIFT_STEP_M)
        shifts = np.arange(0.0, max_shift_out_m + 1e-9, DROP_LIFT_STEP_M)
        out_xy = np.asarray(place[:2], float) / max(np.linalg.norm(place[:2]), 1e-9)
        lowered = lambda point, d: (point[0], point[1], point[2] - d)  # noqa: E731
        found = None  # (depth, tilt, shift, lift)
        for max_deviation in ([0.0, None] if self.tilt != "auto" else [None]):
            for shift in shifts:
                for lift in lifts:
                    dropped = (place[0] + shift * out_xy[0], place[1] + shift * out_xy[1], place[2] + lift)
                    for d in depths:
                        tilt = self.transfer_tilt(lowered(grip, d), lowered(dropped, d), straight_m, d, max_deviation)
                        if tilt is not None:
                            found = (float(d), tilt, shift, lift)
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found:
            depth, tilt, shift, lift = found
            place_tilt, lean = tilt, 0.0
            if lift > drop_lift_m + 1e-9 or shift > 0:
                how = [f"from {lift * 1000:.0f} mm (not {drop_lift_m * 1000:.0f})"] * bool(lift > drop_lift_m + 1e-9) \
                    + [f"{shift * 1000:.0f} mm further from the base"] * bool(shift > 0)
                print(f"    no one tilt reaches the usual release: letting the cup go upright {', '.join(how)}")
            rest = (place[0] + shift * out_xy[0], place[1] + shift * out_xy[1], place[2])
            release = lowered((rest[0], rest[1], place[2] + lift), depth)
        else:
            print("    warning: no one tilt reaches both the pick and the release; the cup will be let go leaning")
            best = None
            for d in depths:
                leaning = self.leaning_tilts(lowered(grip, d), lowered(place, d), straight_m, d,
                                             MAX_RELEASE_LEAN_DEG if best is None else best[0][3] - 1)
                if leaning is not None:
                    best = (leaning, float(d))
            if best is None:
                raise Unreachable(f"the release at ({place[0]:.3f}, {place[1]:.3f}, {place[2]:.3f}) would let the cup "
                                  f"go leaning over {MAX_RELEASE_LEAN_DEG:g} deg")
            (tilt, place_tilt, release, lean), depth = best
            rest = tuple(place)
            print(f"    picking at {tilt:.0f} deg, releasing at {place_tilt:.0f}: the cup leans {lean:.0f} deg "
                  f"(let go {lean_lift_m(lean, depth) * 1000:.0f} mm higher)")
        if self.tilt != "auto" and abs(tilt - float(self.tilt)) > 1e-9:
            print(f"    {tilt_name(self.tilt)} doesn't reach this transfer: picking at {tilt:g} deg instead")
        grip = lowered(grip, depth)
        if depth > 0:
            print(f"    grasping {depth * 1000:.0f} mm lower on the cup than usual")
        pick_tip, pick_tilt = self.grasp_pose(grip, tilt=tilt, depth=depth)
        self.check_reach(pick=grip, tilt=pick_tilt, depth=depth)
        held = self.hold(grip, pick_tip, pick_tilt)
        self.check_reach(held=held, tilt=place_tilt, place=release)
        for name, (target, at), straight in (("pick", (pick_tip, pick_tilt), APPROACH_STRAIGHT_PICK_M),
                                             ("place", self.grasp_pose(release, held, place_tilt), straight_m)):
            if self.approach_path(target, at, straight) is None:
                raise Unreachable(f"{name}: no straight way down to ({target[0]:.3f}, {target[1]:.3f}, "
                                  f"{target[2]:.3f}) at {at:g} deg")
        return TransferPlan(grip, release, rest, pick_tip, pick_tilt, place_tilt, held, depth, lean)

    def solutions(self, point, tilt: float) -> list:
        """Closed-form arm solutions (yaw, shoulder, elbow, wrist; rad) reaching `point` at this
        tilt within REACH_TOLERANCE_M, inside the joint limits and SHOULDER_MIN_DEG."""
        solutions = self.arm._planar.solve(np.asarray(point, float), [math.radians(tilt)],
                                           turned_around=config.ALLOW_TURNED_AROUND)
        return [q for q, error in zip(solutions.joints_rad[0], solutions.error_m[0])
                if error <= REACH_TOLERANCE_M and all(j.limit_deg[0] <= math.degrees(a) <= j.limit_deg[1]
                                                      for j, a in zip(config.JOINTS, q))
                and math.degrees(q[1]) >= SHOULDER_MIN_DEG]

    def approach_path(self, tip, tilt: float, straight_m: float = APPROACH_STRAIGHT_M):
        """Joint-angle keyframes (deg by name), top first, of a vertical approach arriving at
        `tip` at this tilt, in one arm configuration (elbow-up first). Each step up keeps the
        tilt, else takes the nearest reaching one, except over the lowest `straight_m`. None if
        no configuration rises far enough."""
        tip = np.asarray(tip, float)
        tilt_steps = sorted(np.arange(-APPROACH_TILT_STEP_DEG, APPROACH_TILT_STEP_DEG + 0.5, 1.0), key=abs)
        for final in sorted(self.solutions(tip, tilt), key=lambda q: -q[2]):
            path, here, at = [final], final, tilt
            top_z = max(HOVER_Z, tip[2] + APPROACH_MIN_RISE_M)
            for z in np.arange(tip[2] + config.LINEAR_MOVE_STEP_M, top_z + 1e-9, config.LINEAR_MOVE_STEP_M):
                step = None
                straight = z <= min(tip[2] + straight_m, HOVER_Z) + 1e-9
                for change in ([0.0] if straight else tilt_steps):
                    near = [q for q in self.solutions((tip[0], tip[1], z), at + change)
                            if math.degrees(np.max(np.abs(q - here))) <= DESCENT_MAX_STEP_DEG]
                    if near:
                        step = (min(near, key=lambda q: float(np.max(np.abs(q - here)))), at + change)
                        break
                if step is None:
                    break
                (here, at) = step
                path.append(here)
            need = max(APPROACH_MIN_RISE_M, min(straight_m, HOVER_Z - tip[2]))
            if (len(path) - 1) * config.LINEAR_MOVE_STEP_M >= need - config.LINEAR_MOVE_STEP_M / 2:
                return [{**{j.name: math.degrees(a) for j, a in zip(config.JOINTS, q)}, "wrist_roll": 0.0}
                        for q in reversed(path)]
        return None
