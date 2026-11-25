"""Hand-eye calibration: where the gripper-mounted camera sits on the gripper.

With the ChArUco board taped to the desk, the arm visits varied poses looking
down at it (varying position, base yaw, tilt and wrist roll; hand-eye needs
rotations about several axes), then the joint angles the stacking script
itself uses near the board (its looks and grasps, recorded from a dry run of
it, with small changes): the fit has to hold where the stacker works, not
only where random poses land. At each it settles, reads the servos' measured
angles (for forward kinematics) and grabs a frame. Then OpenCV's hand-eye
methods solve for the camera-to-gripper transform.

Check: the board doesn't move, so every view's board pose in the base frame
(gripper pose x camera-in-gripper x board-in-camera) should agree; their
spread is the accuracy you'd get locating things. It also reports where the
board is (compare with where you taped it).

Joint reading model: the servos' zero isn't exactly the URDF's (the elbow was
~6 deg off), and their sensors read a little off in a way that varies over
their range; both bend every pose a little differently -- no camera placement
can absorb that. So the solve then refines, from every detected board corner
at once (pixel error), zero offsets for shoulder, elbow and wrist pitch and a
2-term correction curve for those (xarm_ik/config.py,
true_from_nominal_deg), together with the camera placement and the board
pose, the board held level at BOARD_SURFACE_Z_M (the desk, relative to the
base's underside). Base yaw and wrist roll offsets can't be told apart from
turning the board or the camera, so they stay 0. The model goes to
joint_offsets.json, which xarm_ik/config.py loads: every script then commands
and reads the arm with it. Each session records the model it ran with, so
any session can be re-solved (--solve) whatever model is active now; several
sessions solve together (shared joint model, each its own camera placement
and its own pose for each board it saw; the last session's camera and each
board's latest pose are written).

Boards: a session captures every located board (0 the stack side's, then 1 the
pyramid side's, homing between them), so it alone fits the joint model on both
sides with one camera placement; --board N captures just that one. --refit-intrinsics recalibrates the
camera's intrinsics from the sessions' board images first.

Assumes the camera turns with wrist roll (mounted past it, on the gripper).
The gripper frame is the tip frame (finger_pinch_link) of xarm_ik's chain.

Dry run by default (plans, checks reach; --sim shows it). --execute moves the
arm and captures, saving frames and joint angles to logs/calibration/handeye_<time>/;
--solve DIR re-solves a saved session without the robot.

Usage:
    uv run python scripts/hand_eye_calibration.py                 # dry run: plan + reach check
    uv run python scripts/hand_eye_calibration.py --sim
    uv run python scripts/hand_eye_calibration.py --execute       # moves the arm, captures, solves
    uv run python scripts/hand_eye_calibration.py --solve logs/calibration/handeye_20261118_141500
    uv run python scripts/hand_eye_calibration.py --solve logs/calibration/handeye_A logs/calibration/handeye_B --refit-intrinsics
"""

import argparse
import contextlib
import io
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from run_csv import approach_vector
from xarm_ik import GripperPoseController, charuco_board, config
from xarm_ik.camera import IMAGE_SIZE, load_intrinsics, open_camera
from xarm_ik.kinematics import build_chain

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
CALIBRATION_DIR = LOG_DIR / "calibration"  # sessions, touches and backups
OUTPUT = Path(__file__).resolve().parent.parent / "camera_hand_eye.json"
INTRINSICS_PATH = Path(__file__).resolve().parent.parent / "camera_intrinsics.json"

# A view needs this many board corners (not in a line) to count.
MIN_CORNERS = 12

# Joint-offset refinement: the joints fitted, and the board's surface height in
# the base frame (the base origin is the underside of its bottom plate, which
# stands ~3 mm off the desk on its feet).
OFFSET_JOINTS = ("shoulder_pitch", "elbow_pitch", "wrist_pitch")
# Reading corrections (config.true_from_nominal_deg): this many terms for these
# joints. Two terms cut held-out (other-session) stacker-pose error ~6.4-7.4 ->
# 4.1-5.0 mm and three did no better on camera views alone; with fingertip
# touches reaching past the views (shoulder to -89 deg), three fit those
# (0-4 mm vs 4-14 mm). Not base yaw: the views only span its ~66-124
# deg (the board side), so a curve there would be guesswork at the pyramid
# (~0 deg); fitting one gained nothing on held-out views (4.0/4.6 vs 3.9/4.7 mm).
CORRECTION_JOINTS = ("shoulder_pitch", "elbow_pitch", "wrist_pitch")
CORRECTION_TERMS = 3
BOARD_SURFACE_Z_M = -0.003
REFINE_LOSS_SCALE_PX = 2.0  # soft-L1 scale: corners off by more than this count less
# Fingertip touches (scripts/tip_touch.py): a touch's miss in mm counts as this
# many pixels of view error. The views can't see the fingertip at all (a
# constant wrist error looks like the camera mount turning), so even a few
# touches pin that; this weight decides how touches and views trade elsewhere
# (touching a corner by eye is good to ~1 mm). At 10 the few touches past the
# views' reach (shoulder < -60 deg, the pyramid's releases) stayed 4-14 mm
# off; 40 (with 3 correction terms) brings them to 0-4 mm at no cost to the views.
TOUCH_WEIGHT_PX_PER_MM = 40.0
# Which corner a touch was: the nearest to the fingertip as the model has it
# once the fingertip's offset from the URDF tip point is fitted (the URDF's is
# ~2.5 cm past the real finger ends, which put some touches nearer a neighbor).
# Touches still missing their corner by more than this are dropped (recorded
# before the fingers were down, or the arm moved).
TOUCH_OUTLIER_MM = 15.0

# Pose planning, by what the camera would see: candidate tip poses (base yaw
# around the board, reach, height, tilt, wrist roll) are kept if the board shows
# well in the image, as predicted with the previous calibration (camera
# placement and board pose). Kept poses are spread evenly over bands of camera
# distance to the board center, from close-up to as high as the arm reaches
# (like the stacking script's first look; the camera tops out ~32 cm up, about
# 35 cm from the board), so the solve sees the board at every scale it's used
# at. Focus is locked, so closer than ~12 cm blurs. Bands the arm can't fill
# leave their share to the others.
DISTANCE_BANDS_M = ((0.12, 0.17), (0.17, 0.21), (0.21, 0.25), (0.25, 0.29), (0.29, 0.33), (0.33, 0.40))
CANDIDATE_TRIES = 60000  # random poses tried to fill the bands
YAW_SPREAD_DEG = 35.0  # base yaw either side of the line to the board center
REACH_M = (0.06, 0.30)  # tip distance from the base axis
TIP_HEIGHT_M = (0.05, 0.40)  # above the base origin (the board is ~2 cm below it)
TILT_RANGE_DEG = (0.0, 110.0)
ROLL_DEG = (-45.0, 45.0)
MAX_VIEW_ANGLE_DEG = 60.0  # board normal vs the line of sight; steeper reads badly
MIN_PLANNED_CORNERS = 20  # predicted inner corners in frame (with margin); solving needs MIN_CORNERS
FRAME_MARGIN = 0.9  # predicted corners must sit within this fraction of the half-image
REACH_TOLERANCE_M = 0.002
CAMERA_CLEARANCE_M = 0.03  # the camera stays this far above the board

# Self-collision guard, checked at each pose and along every move to it (moves
# are straight lines in joint space, as the arm runs them): the bulky camera
# must stay SELF_CLEARANCE_M from the base column and upper arm (center lines),
# the elbow may not flip through straight (an IK branch flip, bent more than
# ELBOW_FLIP_DEG both before and after: it sweeps the arm through itself; the
# last crash went +97 -> -124 deg), no joint may swing more than
# MAX_JOINT_SWING_DEG in one move, the shoulder stays within
# MAX_SHOULDER_DEG of upright, and the camera and fingertips stay
# PATH_BOARD_CLEARANCE_M above the board. The moves are simulated in order with
# the controller (as the real run will solve them) and re-checked before each
# real move.
SELF_CLEARANCE_M = 0.10
ELBOW_FLIP_DEG = 30.0
MAX_JOINT_SWING_DEG = 130.0
MAX_SHOULDER_DEG = 85.0  # the last run jammed at 93
PATH_BOARD_CLEARANCE_M = 0.02
PATH_SAMPLES = 25
TOUR_LOOKAHEAD = 20  # nearest candidates tried for each next pose

# Stacker poses: the joint angles the stacking script itself moves to near the
# board (its looks over the stack, grasps and releases there), recorded from a
# dry run of it at both stack spots. Random poses alone never lean the shoulder
# back or straighten the elbow the way its grasps and looks do, so the fitted
# model was extrapolating there (cup heights off by up to ~16 mm, varying with
# the look). Each is visited as-is (if it sees the board) and with small
# changes to every joint (JITTER_DEG; more wrist pitch, which brings the board
# into view, and wrist roll, for hand-eye rotation variety). Its moves at the
# pyramid are turned to face the board: their shoulder/elbow angles (down to
# ~-77 deg shoulder) are past what the board-side moves cover, and the reading
# corrections are only trusted over the range calibrated. Recording takes
# minutes, so it's cached in STACKER_POSES_PATH (--record-stacker-poses redoes it).
STACKER_POSES_PATH = CALIBRATION_DIR / "stacker_poses.json"
STACKER_VARIANTS = 4  # jittered variants tried per recorded pose (plus the pose itself)
JITTER_DEG = {"base_yaw": 4.0, "shoulder_pitch": 4.0, "elbow_pitch": 4.0, "wrist_pitch": 12.0, "wrist_roll": 25.0}

HANDEYE_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


# --- Planning ---


def board_prior(board_index: int = 0):
    """Board `board_index`'s pose (about its center) in the base frame from the last
    calibration (or --locate-board), or None if it has none."""
    try:
        saved = json.loads(OUTPUT.read_text())
    except (OSError, ValueError):
        return None
    boards = saved.get("boards_in_base", {})
    if str(board_index) in boards:
        return np.array(boards[str(board_index)])
    return np.array(saved["board_in_base"]) if board_index == 0 and "board_in_base" in saved else None


def load_prior(board_xy, board_index: int = 0) -> tuple:
    """(camera_in_gripper, board_in_base about the board center) to plan with: the
    previous calibration's, or (no calibration yet) a rough camera placement and
    a level board at `board_xy`."""
    try:
        prior = json.loads(OUTPUT.read_text())
        board_in_base = board_prior(board_index)
        if board_in_base is None:
            if board_xy is None:
                sys.exit(f"Error: board {board_index} hasn't been located; run with --locate-board --board "
                         f"{board_index} --execute first (or pass --board-xy)")
            board_in_base = _level_board(0.0, *board_xy)
        board_in_base = board_in_base.copy()
        if board_xy is not None:
            board_in_base[:2, 3] = board_xy
        return np.array(prior["camera_in_gripper"]), board_in_base
    except (OSError, KeyError, ValueError):
        if board_xy is None:
            sys.exit(f"Error: no previous calibration in {OUTPUT}; pass --board-xy")
        # Camera 6 cm off the tip, looking ~45 deg off the approach; board face up.
        camera = np.eye(4)
        camera[:3, :3] = [[1, 0, 0], [0, 0.71, 0.71], [0, -0.71, 0.71]]
        camera[:3, 3] = [0, -0.06, -0.07]
        board = np.diag([1.0, -1.0, -1.0, 1.0])
        board[:3, 3] = [board_xy[0], board_xy[1], -0.017]
        return camera, board


def _board_corners(board_in_base: np.ndarray) -> np.ndarray:
    """The board's inner chessboard corners in the base frame (N x 3)."""
    board = charuco_board.make_board()
    width, height = board.getChessboardSize()
    square = board.getSquareLength()
    xs, ys = np.meshgrid(np.arange(1, width) * square, np.arange(1, height) * square)
    local = np.stack([xs.ravel() - width * square / 2, ys.ravel() - height * square / 2,
                      np.zeros(xs.size), np.ones(xs.size)], axis=1)
    return (board_in_base @ local.T).T[:, :3]


def predict_view(arm, camera_in_gripper, matrix, board_in_base, corners, pose):
    """(camera distance to the board center m, corners in frame, view angle deg) at
    `pose` (x, y, z, tilt, roll), or None if unreachable or the board's behind the camera."""
    x, y, z, tilt, roll = pose
    solutions = arm._planar.solve((x, y, z), [math.radians(tilt)], turned_around=config.ALLOW_TURNED_AROUND)
    best = int(np.argmin(solutions.error_m[0]))
    if solutions.error_m[0, best] > REACH_TOLERANCE_M:
        return None
    angles = np.zeros(len(arm.chain.links))
    angles[list(config.ACTIVE_LINK_INDICES[:4])] = solutions.joints_rad[0, best]
    angles[config.ACTIVE_LINK_INDICES[4]] = math.radians(roll)
    return view_at_angles(arm.chain, camera_in_gripper, matrix, board_in_base, corners, angles)


def view_at_angles(chain, camera_in_gripper, matrix, board_in_base, corners, angles):
    """predict_view for chain angles (rad)."""
    camera = chain.forward_kinematics(angles) @ camera_in_gripper
    if camera[2, 3] < board_in_base[2, 3] + CAMERA_CLEARANCE_M:
        return None
    to_camera = np.linalg.inv(camera)
    in_camera = (to_camera[:3, :3] @ corners.T).T + to_camera[:3, 3]
    if np.any(in_camera[:, 2] <= 0.05):
        return None
    pixels = (matrix[:2, :2] @ (in_camera[:, :2] / in_camera[:, 2:]).T).T + matrix[:2, 2]
    half = np.array(IMAGE_SIZE) / 2
    in_frame = int(np.sum(np.all(np.abs(pixels - half) < FRAME_MARGIN * half, axis=1)))
    sight = board_in_base[:3, 3] - camera[:3, 3]
    distance = float(np.linalg.norm(sight))
    view_angle = math.degrees(math.acos(min(1.0, abs(float(board_in_base[:3, 2] @ sight)) / distance)))
    return distance, in_frame, view_angle


def plan_poses(arm: GripperPoseController, board_xy, count: int, seed: int, board_index: int = 0) -> list:
    """`count` reachable poses (x, y, z, tilt_deg, roll_deg) seeing the board,
    spread evenly over DISTANCE_BANDS_M, ordered to keep moves short."""
    rng = np.random.default_rng(seed)
    camera_in_gripper, board_in_base = load_prior(board_xy, board_index)
    matrix, _ = load_intrinsics()
    corners = _board_corners(board_in_base)
    center = board_in_base[:3, 3]
    board_heading = math.atan2(center[1], center[0])
    bands = [[] for _ in DISTANCE_BANDS_M]  # candidates
    for _ in range(CANDIDATE_TRIES):
        if all(len(band) >= 3 * count for band in bands):  # spares for the safe ordering
            break
        heading = board_heading + math.radians(rng.uniform(-YAW_SPREAD_DEG, YAW_SPREAD_DEG))
        reach = rng.uniform(*REACH_M)
        pose = (reach * math.cos(heading), reach * math.sin(heading), rng.uniform(*TIP_HEIGHT_M),
                rng.uniform(*TILT_RANGE_DEG), rng.uniform(*ROLL_DEG))
        view = predict_view(arm, camera_in_gripper, matrix, board_in_base, corners, pose)
        if view is None:
            continue
        distance, in_frame, view_angle = view
        if in_frame < MIN_PLANNED_CORNERS or view_angle > MAX_VIEW_ANGLE_DEG:
            continue
        for band, (low, high) in zip(bands, DISTANCE_BANDS_M):
            if low <= distance < high and len(band) < 3 * count:
                band.append(pose)
    # Round-robin quotas over the bands, so each gets an even share of what it can give.
    quotas = [0] * len(bands)
    while sum(quotas) < count and any(q < len(b) for q, b in zip(quotas, bands)):
        for i, band in enumerate(bands):
            if quotas[i] < len(band) and sum(quotas) < count:
                quotas[i] += 1
    poses = safe_tour(arm, bands, quotas, camera_in_gripper, float(board_in_base[2, 3]))
    for band, (low, high) in zip(bands, DISTANCE_BANDS_M):
        n = sum(pose in band for pose in poses)
        print(f"  camera {low * 100:.0f}-{high * 100:.0f} cm from the board center: {n} pose(s)")
    if len(poses) < count // 2:
        sys.exit(f"Error: only {len(poses)} safe poses found; check the board position.")
    return poses


def record_stacker_poses() -> list:
    """Dry-run the stacking script (up then down, auto tilt) at each stack spot and
    return the joint angles (deg by name) after every move, saved to STACKER_POSES_PATH."""
    import stack_cups  # only needed here
    from cup_stacking.layout import stack_spots
    from cup_stacking.vision import SimulatedVision

    print("Recording the stacking script's moves (a dry run per stack spot; takes a few minutes)...")
    camera_in_gripper = np.array(json.loads(OUTPUT.read_text())["camera_in_gripper"])
    recorded = []
    for spot, xy in stack_spots().items():
        arm = GripperPoseController(dry_run=True)
        arm._wait_for_tick = lambda count_late=True: time.perf_counter()  # no need to run in real time
        move_to_pose = arm.move_to_pose

        def recording_move(*args, _move=move_to_pose, _arm=arm, _spot=spot, **kwargs):
            result = _move(*args, **kwargs)
            recorded.append({"spot": _spot, "joints_deg": _arm.get_joint_angles_deg()})
            return result

        arm.move_to_pose = recording_move
        stacker = stack_cups.CupStacker(arm, SimulatedVision(6, xy), camera_in_gripper, None,
                                        simulated=True, tilt="auto")
        with contextlib.redirect_stdout(io.StringIO()):
            arm.home()
            stacker.up_stack()
            stacker.down_stack()
    STACKER_POSES_PATH.parent.mkdir(exist_ok=True)
    STACKER_POSES_PATH.write_text(json.dumps(recorded, indent=1))
    print(f"  {len(recorded)} moves, saved to {STACKER_POSES_PATH}")
    return recorded


def real_run_poses() -> list:
    """The joint angles (deg by name) the arm really reached in stack_cups runs (logs/analysis/cups_*, logs/dev/cups_*):
    each up-stack grasp and down-stack release, both at the stack. Dry-run moves missed the
    tall-stack grasps (shoulder back ~11 deg, wrist ~-53 deg) by ~20 deg, and the model put the
    fingertip there 4-10 mm high (the board in the grasp photos, Nov 25)."""
    poses = []
    for path in sorted(LOG_DIR.glob("*/cups_*/transfers.json")):
        try:
            transfers = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for transfer in transfers:
            key = "pick_reached" if transfer.get("phase") == "up" else "place_reached"
            joints = (transfer.get(key) or {}).get("measured_joints_deg")
            if joints:
                poses.append({"spot": f"{path.parent.name} {transfer['phase']}", "joints_deg": joints})
    return poses


def plan_stacker_poses(chain, count: int, seed: int, record: bool, board_index: int = 0, board_xy=None) -> list:
    """Up to `count` joint-angle poses (deg by name) from the stacking script's own
    moves (and small changes to them) that see the board well and are safe: its dry-run
    moves and what real runs reached (real_run_poses)."""
    if count <= 0:
        return []
    recorded = record_stacker_poses() if record or not STACKER_POSES_PATH.exists() \
        else json.loads(STACKER_POSES_PATH.read_text())
    real = real_run_poses()
    print(f"  stacker moves: {len(recorded)} from a dry run, {len(real)} from real runs")
    recorded = recorded + real
    rng = np.random.default_rng(seed)
    camera_in_gripper, board_in_base = load_prior(board_xy, board_index)
    matrix, _ = load_intrinsics()
    corners = _board_corners(board_in_base)
    board_z = float(board_in_base[2, 3])
    names = [joint.name for joint in config.JOINTS]
    # Moves at the pyramid face away from the board: turned to face it (base yaw
    # isn't corrected; the shoulder/elbow/wrist angles are what's calibrated there).
    center = board_in_base[:3, 3]
    yaw_offset = float(chain.links[config.ACTIVE_LINK_INDICES[0]].origin_orientation[2])
    board_yaw = math.degrees((math.atan2(-center[0], center[1]) - yaw_offset + math.pi) % (2 * math.pi) - math.pi)
    unique = {}
    for r in recorded:
        joints = dict(r["joints_deg"])
        if abs(joints["base_yaw"] - board_yaw) > YAW_SPREAD_DEG:
            joints["base_yaw"] = board_yaw
        unique[tuple(round(joints[n], 1) for n in names)] = None
    unique = list(unique)
    per_source = []
    for source in unique:
        options = []
        for variant in range(STACKER_VARIANTS + 1):
            joints = {n: a + (rng.uniform(-JITTER_DEG[n], JITTER_DEG[n]) if variant else 0.0) for n, a in zip(names, source)}
            if any(not lo <= joints[n] <= hi for n, (lo, hi) in zip(names, (j.limit_deg for j in config.JOINTS))):
                continue
            angles = _chain_angles(joints, chain)
            view = view_at_angles(chain, camera_in_gripper, matrix, board_in_base, corners, angles)
            if view is None or view[1] < MIN_PLANNED_CORNERS or view[2] > MAX_VIEW_ANGLE_DEG:
                continue
            if unsafe_config(chain, camera_in_gripper, angles, board_z):
                continue
            options.append(joints)
        if options:
            per_source.append(options)
    # Round-robin over the recorded poses, so every one gets a turn before any gets a second.
    chosen = []
    order = rng.permutation(len(per_source))
    while len(chosen) < count and any(per_source):
        for i in order:
            if per_source[i] and len(chosen) < count:
                chosen.append(per_source[i].pop(0))
    print(f"  stacker poses: {len(chosen)} (from {len(unique)} recorded moves, {len(per_source)} of them near "
          "a pose that sees the board)")
    return chosen


def joint_tour(chain, poses: list, start_rad: np.ndarray, camera_in_gripper, board_z: float) -> list:
    """Order joint-angle poses into a tour from `start_rad`, each move safe, nearest first."""
    left = list(poses)
    here = start_rad.copy()
    tour = []
    while left:
        left.sort(key=lambda joints: float(np.linalg.norm(_chain_angles(joints, chain) - here)))
        for k, joints in enumerate(left[:TOUR_LOOKAHEAD]):
            end = _chain_angles(joints, chain)
            if unsafe_move(chain, camera_in_gripper, here, end, board_z) is None:
                tour.append(joints)
                here = end
                del left[k]
                break
        else:
            break
    while tour and unsafe_move(chain, camera_in_gripper, here, np.zeros(len(chain.links)), board_z):
        tour.pop()  # the run homes from the last pose
        here = _chain_angles(tour[-1], chain) if tour else start_rad
    return tour


def _segment_distance(point, a, b) -> float:
    ab = b - a
    t = float(np.clip((point - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * ab)))


def unsafe_config(chain, camera_in_gripper, angles_rad, board_z: float) -> str | None:
    """Why the arm at chain angles `angles_rad` is unsafe, or None."""
    shoulder = math.degrees(angles_rad[config.ACTIVE_LINK_INDICES[1]])
    if abs(shoulder) > MAX_SHOULDER_DEG:
        return f"shoulder at {shoulder:.0f} deg"
    frames = chain.forward_kinematics(angles_rad, full_kinematics=True)
    # base, base_yaw, shoulder, elbow, wrist, wrist_roll, tip frames
    shoulder_point, elbow_point = frames[2][:3, 3], frames[3][:3, 3]
    camera = (frames[-1] @ camera_in_gripper)[:3, 3]
    for name, a, b in (("base", np.zeros(3), shoulder_point), ("upper arm", shoulder_point, elbow_point)):
        distance = _segment_distance(camera, a, b)
        if distance < SELF_CLEARANCE_M:
            return f"camera {distance * 100:.1f} cm from the {name}"
    for name, z in (("camera", camera[2]), ("fingertips", frames[-1][2, 3])):
        if z < board_z + PATH_BOARD_CLEARANCE_M:
            return f"{name} {(z - board_z) * 100:.1f} cm above the board"
    return None


def unsafe_move(chain, camera_in_gripper, start_rad, end_rad, board_z: float) -> str | None:
    """Why the joint-space move from `start_rad` to `end_rad` is unsafe, or None."""
    elbow = config.ACTIVE_LINK_INDICES[2]
    before, after = math.degrees(start_rad[elbow]), math.degrees(end_rad[elbow])
    if before * after < 0 and min(abs(before), abs(after)) > ELBOW_FLIP_DEG:
        return f"elbow flips {before:+.0f} -> {after:+.0f} deg"
    swing = np.degrees(np.abs(end_rad - start_rad))
    if swing.max() > MAX_JOINT_SWING_DEG:
        joint = config.JOINTS[list(config.ACTIVE_LINK_INDICES).index(int(np.argmax(swing)))].name \
            if int(np.argmax(swing)) in config.ACTIVE_LINK_INDICES else "a joint"
        return f"{joint} swings {swing.max():.0f} deg"
    for fraction in np.linspace(0, 1, PATH_SAMPLES):
        reason = unsafe_config(chain, camera_in_gripper, start_rad + fraction * (end_rad - start_rad), board_z)
        if reason:
            return f"{reason} {'at the pose' if fraction == 1 else 'on the way'}"
    return None


def try_move(arm: GripperPoseController, pose, camera_in_gripper, board_z: float):
    """Dry-run `arm` (its commanded state = where the real arm is) to `pose` the way
    the capture moves; returns (end chain angles, None) if safe, else (None, reason).
    Leaves `arm` where it started."""
    x, y, z, tilt, roll = pose
    start = arm._joint_angles_rad.copy()
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # the dry run's per-move print
            arm.move_to_pose(x, y, z, approach=approach_vector(x, y, tilt), roll_deg=roll, gripper_open_fraction=1.0,
                             duration_ms=1, linear=False, position_tolerance_m=0.005)
    except ValueError as e:
        return None, f"unreachable ({e})"
    end = arm._joint_angles_rad.copy()
    arm._joint_angles_rad = start
    reason = unsafe_move(arm.chain, camera_in_gripper, start, end, board_z)
    return (None, reason) if reason else (end, None)


def safe_tour(arm: GripperPoseController, candidates: list, quotas: list, camera_in_gripper, board_z: float) -> list:
    """Order poses from `candidates` (lists per band) into a tour from home whose
    every move (as the controller solves it) passes the self-collision guard,
    taking up to quotas[i] from band i; each step goes to the nearest safe one."""
    def features(pose):
        x, y, z, tilt, roll = pose
        return np.array([x * 100, y * 100, z * 100, tilt / 3, roll / 3])
    left = [(band, pose) for band, poses in enumerate(candidates) for pose in poses]
    taken = [0] * len(quotas)
    tour, rejected = [], {}
    arm._joint_angles_rad = np.zeros(len(arm.chain.links))  # the run starts from home
    here = features((0.0, 0.0, 0.44, 0.0, 0.0))
    while left and len(tour) < sum(quotas):
        left = [(band, pose) for band, pose in left if taken[band] < quotas[band]]
        left.sort(key=lambda item: np.linalg.norm(features(item[1]) - here))
        for k, (band, pose) in enumerate(left[:TOUR_LOOKAHEAD]):
            end, reason = try_move(arm, pose, camera_in_gripper, board_z)
            if end is None:
                rejected[reason.split(" (")[0]] = rejected.get(reason.split(" (")[0], 0) + 1
                continue
            arm._joint_angles_rad = end
            tour.append(pose)
            taken[band] += 1
            here = features(pose)
            del left[k]
            break
        else:
            break  # nothing nearby is safe from here
    # The run ends homing from the last pose: drop poses until that's safe too.
    while tour:
        reason = unsafe_move(arm.chain, camera_in_gripper, arm._joint_angles_rad, np.zeros(len(arm.chain.links)), board_z)
        if reason is None:
            break
        tour.pop()
        arm._joint_angles_rad = np.zeros(len(arm.chain.links))
        for pose in tour:
            arm._joint_angles_rad, _ = try_move(arm, pose, camera_in_gripper, board_z)
    if rejected:
        print("  moves rejected while ordering: " + ", ".join(f"{n} x {r}" for r, n in sorted(rejected.items(), key=lambda kv: -kv[1])[:6]))
    return tour


# --- Capture ---


def capture_session(args, plan: list) -> Path | None:
    """Visit each (board index, pose) in `plan` (None: go home, between boards); with
    --execute, save frames + joint angles. Returns the session dir."""
    arm = GripperPoseController(com_port=args.com_port, dry_run=not args.execute, closed_loop=True)
    view = None
    if args.sim:
        from xarm_ik.sim import PybulletArmView
        view = PybulletArmView()
    on_tick = view.mirror_tick if view else None

    camera = session = None
    records = []
    if args.execute:
        try:
            camera = open_camera(args.device, IMAGE_SIZE)
        except RuntimeError as e:
            sys.exit(f"Error: {e}")
        session = CALIBRATION_DIR / time.strftime("handeye_%Y%m%d_%H%M%S")
        session.mkdir(parents=True)
        print(f"Saving to {session}")
    detectors = _detectors()
    priors = {board: load_prior(args.board_xy, board) for board in {item[0] for item in plan if item is not None}}
    twin = GripperPoseController(dry_run=True)  # solves each move first, to check it's safe
    poses = [item for item in plan if item is not None]

    print("Homing...")
    arm.home(on_tick=on_tick)
    i = -1
    try:
        for item in plan:
            if item is None:
                print("Homing (next board)...")
                arm.home(on_tick=on_tick)
                continue
            i += 1
            board, pose = item
            _, detector = detectors[board]
            camera_in_gripper, board_in_base = priors[board]
            board_z = float(board_in_base[2, 3])
            if isinstance(pose, dict):  # a stacker pose: joint angles, moved to directly
                kind, target = "stacker", pose
                print(f"[{i + 1}/{len(poses)}] board {board} stacker pose: "
                      + ", ".join(f"{name} {angle:.1f}" for name, angle in pose.items()) + " deg", flush=True)
                reason = unsafe_move(arm.chain, camera_in_gripper, arm._joint_angles_rad,
                                     _chain_angles(pose, arm.chain), board_z)
            else:
                kind, target = "random", list(pose)
                x, y, z, tilt, roll = pose
                print(f"[{i + 1}/{len(poses)}] board {board} tip ({x:.3f}, {y:.3f}, {z:.3f}) m, tilt {tilt:.1f}, "
                      f"roll {roll:.1f} deg", flush=True)
                twin._joint_angles_rad = arm._joint_angles_rad.copy()
                _, reason = try_move(twin, pose, camera_in_gripper, board_z)
            if reason:
                print(f"    skipped (unsafe: {reason})")
                continue
            try:
                # Gripper fully open: closed fingers hang into the camera's view of the board.
                if kind == "stacker":
                    arm.move_to_joints(pose, gripper_open_fraction=1.0, duration_ms=args.move_ms, on_tick=on_tick)
                else:
                    arm.move_to_pose(x, y, z, approach=approach_vector(x, y, tilt), roll_deg=roll,
                                     gripper_open_fraction=1.0, duration_ms=args.move_ms, linear=False,
                                     position_tolerance_m=0.005, on_tick=on_tick)
            except ValueError as e:
                print(f"    skipped: {e}")
                continue
            if not args.execute:
                continue
            time.sleep(args.settle_s)  # let it stop swaying
            commanded = arm.get_joint_angles_deg()
            measured = arm.get_measured_joint_angles_deg()
            for _ in range(5):  # drop frames buffered while moving
                camera.read()
            ok, frame = camera.read()
            if not ok:
                print("    camera read failed")
                continue
            corners, _, _, _ = detector.detectBoard(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            found = 0 if corners is None else len(corners)
            image_name = f"view_{i:03d}.png"
            cv2.imwrite(str(session / image_name), frame)
            records.append({"image": image_name, "kind": kind, "target": target, "board": board,
                            "commanded_deg": commanded, "measured_deg": measured,
                            # The angles above include these; solving takes them back off.
                            "joint_offsets_deg": {j.name: j.zero_offset_deg for j in config.JOINTS},
                            "joint_corrections_deg": {j.name: list(j.correction_deg) for j in config.JOINTS
                                                      if j.correction_deg},
                            "joint_correction_range_deg": {j.name: list(j.correction_range_deg) for j in config.JOINTS
                                                           if j.correction_range_deg}})
            missing = sorted(set(commanded) - set(measured))
            print(f"    {found} corners" + (f" (no servo read: {', '.join(missing)})" if missing else ""))
            if not args.no_show:
                small = cv2.resize(frame, (960, 540))
                cv2.putText(small, f"view {i} (board {board}): {found} corners", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("hand-eye", small)
                cv2.waitKey(1)
            (session / "views.json").write_text(json.dumps(records, indent=1))  # after each, in case of a crash
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
    print("Homing...")
    arm.home(on_tick=on_tick)
    return session


# Locating a board with no calibration yet (--locate-board): the arm looks out over
# its side of the robot from up high (the whole side in view at a time), turning
# across it, and each view that finds the board puts it in the base frame through
# the current calibration (camera placement and joint model); their median, made
# level at the desk, is saved to plan the board's session with.
LOCATE_SIDES_DEG = {0: 90.0, 1: 180.0}  # the board's side: heading (atan2(y, x)) to sweep around
LOCATE_SWEEP_DEG = np.arange(-40.0, 40.1, 20.0)  # headings either side of it
# (reach m, height m, tilt deg) per heading: reachable, and seeing most of the desk on that
# side from ~30 cm (each sees ~25 x 50 cm of it, the first nearer in, the others farther out).
LOCATE_POSES = ((0.08, 0.20, 90.0), (0.16, 0.20, 70.0), (0.16, 0.16, 70.0))


def locate_board(args) -> None:
    """Find board args.board with the camera (see LOCATE_*), save its pose to OUTPUT's boards_in_base."""
    board = charuco_board.make_board(board_index=args.board)
    detector = cv2.aruco.CharucoDetector(board)
    camera_in_gripper = np.array(json.loads(OUTPUT.read_text())["camera_in_gripper"])
    matrix, distortion = load_intrinsics()
    arm = GripperPoseController(com_port=args.com_port, dry_run=False, closed_loop=True)
    twin = GripperPoseController(dry_run=True)
    try:
        camera = open_camera(args.device, IMAGE_SIZE)
    except RuntimeError as e:
        sys.exit(f"Error: {e}")
    side = LOCATE_SIDES_DEG.get(args.board, 180.0)
    if args.board_xy is not None:
        side = math.degrees(math.atan2(args.board_xy[1], args.board_xy[0]))
    found = []  # board center poses (4x4, level) in the base frame
    print(f"Looking for board {args.board} ({charuco_board.BOARD_NAMES[args.board]}) around heading {side:.0f} deg...")
    arm.home()
    try:
        for offset in LOCATE_SWEEP_DEG:
            heading = math.radians(side + offset)
            for reach, height, tilt in LOCATE_POSES:
                x, y = reach * math.cos(heading), reach * math.sin(heading)
                pose = (x, y, height, tilt, 0.0)
                twin._joint_angles_rad = arm._joint_angles_rad.copy()
                _, reason = try_move(twin, pose, camera_in_gripper, BOARD_SURFACE_Z_M)
                if reason:
                    continue
                try:
                    arm.move_to_pose(x, y, height, approach=approach_vector(x, y, tilt), roll_deg=0.0,
                                     gripper_open_fraction=1.0, duration_ms=args.move_ms, linear=False,
                                     position_tolerance_m=0.005)
                except ValueError:
                    continue
                time.sleep(args.settle_s)
                angles = {**arm.get_joint_angles_deg(), **arm.get_measured_joint_angles_deg()}
                for _ in range(5):  # drop frames buffered while moving
                    camera.read()
                ok, frame = camera.read()
                if not ok:
                    continue
                corners, ids, _, _ = detector.detectBoard(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                if corners is None or len(corners) < MIN_CORNERS or board.checkCharucoCornersCollinear(ids.reshape(-1, 1)):
                    print(f"  heading {side + offset:4.0f}, tilt {tilt:2.0f}: "
                          f"{0 if corners is None else len(corners)} corners")
                    continue
                object_points, image_points = board.matchImagePoints(corners.reshape(-1, 1, 2), ids.reshape(-1, 1))
                ok, rvec, tvec = cv2.solvePnP(object_points, image_points, matrix, distortion, flags=cv2.SOLVEPNP_IPPE)
                if not ok:
                    continue
                corner_frame = (arm.chain.forward_kinematics(_chain_angles(angles, arm.chain)) @ camera_in_gripper
                                @ _transform(cv2.Rodrigues(rvec)[0], tvec))
                center = corner_frame @ np.linalg.inv(_center_to_corner())
                found.append(center)
                print(f"  heading {side + offset:4.0f}, tilt {tilt:2.0f}: {len(corners)} corners, board center at "
                      f"({center[0, 3]:.3f}, {center[1, 3]:.3f}, {center[2, 3]:.3f}) m")
    finally:
        camera.release()
        print("Homing...")
        arm.home()
    if not found:
        sys.exit(f"Error: board {args.board} not seen; pass --board-xy X Y (its center, roughly) to look around there.")
    xs = np.median([f[0, 3] for f in found])
    ys = np.median([f[1, 3] for f in found])
    yaws = [math.atan2(f[1, 0], f[0, 0]) for f in found]
    yaw = math.atan2(np.median(np.sin(yaws)), np.median(np.cos(yaws)))
    level = _level_board(yaw, xs, ys)
    spread = max(np.linalg.norm(f[:2, 3] - [xs, ys]) for f in found) * 1000
    saved = json.loads(OUTPUT.read_text())
    boards = dict(saved.get("boards_in_base", {}))
    boards.setdefault("0", saved.get("board_in_base"))
    boards[str(args.board)] = level.tolist()
    saved["boards_in_base"] = {k: v for k, v in boards.items() if v is not None}
    OUTPUT.write_text(json.dumps(saved, indent=2) + "\n")
    print(f"Board {args.board}: center ({xs:.3f}, {ys:.3f}) m, turned {math.degrees(yaw):.0f} deg, from {len(found)} "
          f"view(s) (within {spread:.0f} mm of each other). Saved to {OUTPUT.name}.")
    print(f"Next: uv run python scripts/hand_eye_calibration.py --board {args.board} --execute")


# --- Solving ---


def _transform(rotation, translation) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = np.ravel(translation)
    return t


def _chain_angles(angles_deg: dict, chain) -> np.ndarray:
    angles = np.zeros(len(chain.links))
    for i, joint in zip(config.ACTIVE_LINK_INDICES, config.JOINTS):
        angles[i] = math.radians(angles_deg[joint.name])
    return angles


def _detectors() -> dict:
    """board index -> (board, CharucoDetector)."""
    out = {}
    for index in range(len(charuco_board.BOARD_NAMES)):
        board = charuco_board.make_board(board_index=index)
        out[index] = (board, cv2.aruco.CharucoDetector(board))
    return out


def load_views(session: Path, use_commanded: bool, session_index: int = 0) -> list:
    """Per usable view: dict with the image name, pose kind, session index, board
    index, the servos' nominal joint angles (rad, config.JOINTS order; the reading
    model the session ran with taken back off), the board corners (board
    coordinates, and undistorted normalized image coordinates), the focal length
    (px) and board-in-camera (4x4)."""
    detectors = _detectors()
    matrix, distortion = load_intrinsics()
    views = []
    for record in json.loads((session / "views.json").read_text()):
        board_index = int(record.get("board", 0))
        board, detector = detectors[board_index]
        image = cv2.imread(str(session / record["image"]), cv2.IMREAD_GRAYSCALE)
        corners, ids, _, _ = detector.detectBoard(image)
        if corners is None or len(corners) < MIN_CORNERS or board.checkCharucoCornersCollinear(ids.reshape(-1, 1)):
            print(f"  {record['image']}: too few board corners, skipped")
            continue
        object_points, image_points = board.matchImagePoints(corners.reshape(-1, 1, 2), ids.reshape(-1, 1))
        ok, rvec, tvec = cv2.solvePnP(object_points, image_points, matrix, distortion, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        # Measured angles where the servo answered, commanded otherwise.
        angles = record["commanded_deg"] if use_commanded else {**record["commanded_deg"], **record["measured_deg"]}
        offsets = record.get("joint_offsets_deg", {})  # older sessions ran without a reading model
        corrections = record.get("joint_corrections_deg", {})
        ranges = record.get("joint_correction_range_deg", {})
        nominal = np.radians([config.nominal_from_true_deg(angles[j.name], offsets.get(j.name, 0.0),
                                                           tuple(corrections.get(j.name, ())),
                                                           tuple(ranges[j.name]) if j.name in ranges else None)
                              for j in config.JOINTS])
        views.append({
            "name": f"{session.name}/{record['image']}", "kind": record.get("kind", "random"), "session": session_index,
            "board": board_index, "nominal_rad": nominal, "object": object_points.reshape(-1, 3),
            "normalized": cv2.undistortPoints(image_points.reshape(-1, 1, 2), matrix, distortion).reshape(-1, 2),
            "focal_px": float(matrix[0, 0]), "board_in_camera": _transform(cv2.Rodrigues(rvec)[0], tvec)})
    return views


def current_model() -> dict:
    """The joint reading model in use: {"offsets": {joint: deg}, "corrections": {joint: [deg, ...]},
    "ranges": {joint: [lo, hi] nominal deg the correction holds over}}."""
    return {"offsets": {j.name: j.zero_offset_deg for j in config.JOINTS},
            "corrections": {j.name: list(j.correction_deg) for j in config.JOINTS if j.correction_deg},
            "ranges": {j.name: list(j.correction_range_deg) for j in config.JOINTS if j.correction_range_deg}}


def true_angles_rad(nominal_rad: np.ndarray, model: dict) -> np.ndarray:
    """config.JOINTS-order true angles (rad) for nominal ones, under `model`."""
    ranges = model.get("ranges", {})
    return np.radians([config.true_from_nominal_deg(math.degrees(n), model["offsets"].get(j.name, 0.0),
                                                    tuple(model["corrections"].get(j.name, ())),
                                                    tuple(ranges[j.name]) if j.name in ranges else None)
                       for j, n in zip(config.JOINTS, nominal_rad)])


def gripper_poses(views: list, chain, model: dict) -> list:
    """(name, gripper-in-base, board-in-camera) per view, under this joint reading model."""
    out = []
    for view in views:
        angles = np.zeros(len(chain.links))
        angles[list(config.ACTIVE_LINK_INDICES)] = true_angles_rad(view["nominal_rad"], model)
        out.append((view["name"], chain.forward_kinematics(angles), view["board_in_camera"]))
    return out


def _center_to_corner() -> np.ndarray:
    """Board-center frame -> the board's own (corner-origin) frame."""
    board = charuco_board.make_board()
    width, height = board.getChessboardSize()
    shift = np.eye(4)
    shift[:2, 3] = [-width * board.getSquareLength() / 2, -height * board.getSquareLength() / 2]
    return shift


def _level_board(yaw: float, x: float, y: float) -> np.ndarray:
    """Board pose (about its center) lying flat at BOARD_SURFACE_Z_M, face up (its z axis down), turned by yaw."""
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", yaw).as_matrix() @ np.diag([1.0, -1.0, -1.0])
    pose[:3, 3] = [x, y, BOARD_SURFACE_Z_M]
    return pose


def refine(views: list, chain, model: dict, cameras: list, boards: list, fit_corrections: bool = True,
           touches: list = ()) -> tuple:
    """Least-squares refinement, over every board corner's pixel error, of the
    OFFSET_JOINTS zero offsets, (fit_corrections) the CORRECTION_JOINTS reading
    corrections, per session the camera placement, and per board slot (a board as
    one session saw it; each view's "slot") a level board, from the given start.
    With `touches` (load_touches; each on its "slot"'s board),
    also each touch's fingertip-to-corner miss (mm x TOUCH_WEIGHT_PX_PER_MM),
    and where the physical fingertip point sits in the tip frame (the URDF's
    may not be at the finger ends). Returns (model, cameras, boards, rms_px
    before, rms_px after, fingertip offset m (3; zeros without touches),
    touch misses mm before (N x 3), after)."""
    sessions, slots = len(cameras), len(boards)
    correction_joints = CORRECTION_JOINTS if fit_corrections else ()
    # Each correction holds over the nominal angles these views (and touches) cover (flat beyond).
    nominal_deg = np.degrees(np.array([item["nominal_rad"] for item in [*views, *touches]]))
    ranges = dict(model.get("ranges", {}))
    for name in correction_joints:
        column = nominal_deg[:, [j.name for j in config.JOINTS].index(name)]
        ranges[name] = [round(float(column.min()), 1), round(float(column.max()), 1)]
    shift = _center_to_corner()

    def unpack(params):
        k = len(OFFSET_JOINTS)
        offsets = {**model["offsets"], **{n: math.degrees(p) for n, p in zip(OFFSET_JOINTS, params[:k])}}
        corrections = dict(model["corrections"])
        for name in correction_joints:
            corrections[name] = list(np.degrees(params[k:k + CORRECTION_TERMS]))
            k += CORRECTION_TERMS
        out_cameras, out_boards = [], []
        for _ in range(sessions):
            out_cameras.append(_transform(Rotation.from_rotvec(params[k:k + 3]).as_matrix(), params[k + 3:k + 6]))
            k += 6
        for _ in range(slots):
            out_boards.append(_level_board(*params[k:k + 3]))
            k += 3
        fingertip = params[k:k + 3] if touches else np.zeros(3)
        return {"offsets": offsets, "corrections": corrections, "ranges": ranges}, out_cameras, out_boards, fingertip

    def touch_misses_mm(params):
        fitted, _, fitted_boards, fingertip = unpack(params)
        misses = []
        for touch in touches:
            corner_frame = fitted_boards[touch["slot"]] @ shift
            angles = np.zeros(len(chain.links))
            angles[list(config.ACTIVE_LINK_INDICES)] = true_angles_rad(touch["nominal_rad"], fitted)
            tip = (chain.forward_kinematics(angles) @ np.append(fingertip, 1.0))[:3]
            misses.append((tip - (corner_frame @ np.append(touch["corner_in_board_m"], 1.0))[:3]) * 1000)
        return np.array(misses).reshape(-1, 3)

    def residuals(params):
        return np.concatenate([view_residuals(params), touch_misses_mm(params).ravel() * TOUCH_WEIGHT_PX_PER_MM])

    def view_residuals(params):
        fitted, fitted_cameras, fitted_boards, _ = unpack(params)
        corner_frames = [board @ shift for board in fitted_boards]
        out = []
        for view in views:
            angles = np.zeros(len(chain.links))
            angles[list(config.ACTIVE_LINK_INDICES)] = true_angles_rad(view["nominal_rad"], fitted)
            board_in_camera = (np.linalg.inv(chain.forward_kinematics(angles) @ fitted_cameras[view["session"]])
                               @ corner_frames[view["slot"]])
            points = view["object"] @ board_in_camera[:3, :3].T + board_in_camera[:3, 3]
            out.append(((points[:, :2] / points[:, 2:] - view["normalized"]) * view["focal_px"]).ravel())
        return np.concatenate(out)

    def rms(params):
        r = view_residuals(params).reshape(-1, 2)
        return math.sqrt(np.mean(np.sum(r ** 2, axis=1)))

    start = [np.radians([model["offsets"].get(n, 0.0) for n in OFFSET_JOINTS])]
    for name in correction_joints:
        start.append(np.radians((list(model["corrections"].get(name, [])) + [0.0] * CORRECTION_TERMS)[:CORRECTION_TERMS]))
    for camera in cameras:
        start.append(np.concatenate([Rotation.from_matrix(camera[:3, :3]).as_rotvec(), camera[:3, 3]]))
    for board in boards:
        start.append(np.array([math.atan2(board[1, 0], board[0, 0]), board[0, 3], board[1, 3]]))
    if touches:
        start.append(np.zeros(3))
    start = np.concatenate(start)
    start_rms = rms(start)  # with each start board made level at the desk
    misses_before = touch_misses_mm(start)
    solution = least_squares(residuals, start, loss="soft_l1", f_scale=REFINE_LOSS_SCALE_PX, x_scale="jac")
    fitted, fitted_cameras, fitted_boards, fingertip = unpack(solution.x)
    fitted = {"offsets": {n: round(d, 3) for n, d in fitted["offsets"].items()},
              "corrections": {n: [round(c, 3) for c in cs] for n, cs in fitted["corrections"].items()},
              "ranges": fitted["ranges"]}
    return (fitted, fitted_cameras, fitted_boards, start_rms, rms(solution.x), fingertip,
            misses_before, touch_misses_mm(solution.x))


def label_touches(touches: list, chain, model: dict, board_in_base: np.ndarray) -> list:
    """Fit the fingertip's offset from the URDF tip point alone (joint model as is),
    relabelling each touch to its nearest corner as it goes; returns the touches
    within TOUCH_OUTLIER_MM of their corner (relabelled), printing what changed."""
    board = charuco_board.make_board()
    width, height = board.getChessboardSize()
    square = board.getSquareLength()
    xs, ys = np.meshgrid(np.arange(1, width) * square, np.arange(1, height) * square)
    local = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    corner_frame = board_in_base @ _center_to_corner()
    corners = (corner_frame[:3, :3] @ local.T).T + corner_frame[:3, 3]
    poses = []
    for touch in touches:
        angles = np.zeros(len(chain.links))
        angles[list(config.ACTIVE_LINK_INDICES)] = true_angles_rad(touch["nominal_rad"], model)
        poses.append(chain.forward_kinematics(angles))

    def tips(offset):
        return np.array([(pose @ np.append(offset, 1.0))[:3] for pose in poses])

    offset = np.zeros(3)
    for _ in range(4):
        nearest = [int(np.argmin(np.linalg.norm(corners - tip, axis=1))) for tip in tips(offset)]
        offset = least_squares(lambda o: ((tips(o) - corners[nearest]) * 1000).ravel(), offset,
                               loss="soft_l1", f_scale=3.0).x
    misses = np.linalg.norm(tips(offset) - corners[nearest], axis=1) * 1000
    print(f"  touches: fingertip {np.round(offset * 1000, 1).tolist()} mm from the URDF tip point (tip frame), "
          f"median miss {np.median(misses):.1f} mm with the current joint model")
    kept = []
    for touch, corner, miss in zip(touches, nearest, misses):
        if miss > TOUCH_OUTLIER_MM:
            print(f"    dropped a touch recorded as corner {touch['corner']}: {miss:.0f} mm from any corner")
            continue
        if corner != touch["corner"]:
            print(f"    corner {touch['corner']} -> {corner} (relabelled)")
        kept.append({**touch, "corner": corner, "corner_in_board_m": local[corner]})
    return kept


def load_touches(directory: Path) -> list:
    """Fingertip touches (scripts/tip_touch.py): per touch, the servos' nominal joint
    angles (rad, config.JOINTS order; the model it ran with taken back off) and the
    touched corner in the board's corner frame (m)."""
    touches = []
    for record in json.loads((directory / "touches.json").read_text()):
        angles = {**record["commanded_deg"], **record["measured_deg"]}
        offsets = record.get("joint_offsets_deg", {})
        corrections = record.get("joint_corrections_deg", {})
        ranges = record.get("joint_correction_range_deg", {})
        nominal = np.radians([config.nominal_from_true_deg(angles[j.name], offsets.get(j.name, 0.0),
                                                           tuple(corrections.get(j.name, ())),
                                                           tuple(ranges[j.name]) if j.name in ranges else None)
                              for j in config.JOINTS])
        touches.append({"nominal_rad": nominal, "corner_in_board_m": np.array(record["corner_in_board_m"]),
                        "corner": record["corner"], "board": int(record.get("board", 0))})
    return touches


def board_spread(views: list, camera_in_gripper: np.ndarray) -> tuple:
    """Board center in base per view -> (mean center, rms deviation m, rms rotation
    deviation deg, mean board pose 4x4, per-view deviation m).
    """
    board = charuco_board.make_board()
    width, height = board.getChessboardSize()
    center_on_board = np.array([width * board.getSquareLength() / 2, height * board.getSquareLength() / 2, 0, 1])
    poses = [gripper_in_base @ camera_in_gripper @ board_in_camera for _, gripper_in_base, board_in_camera in views]
    centers = np.array([pose @ center_on_board for pose in poses])[:, :3]
    mean_center = centers.mean(axis=0)
    deviations = np.linalg.norm(centers - mean_center, axis=1)
    # Mean rotation: the nearest rotation to the summed matrices.
    u, _, vt = np.linalg.svd(sum(pose[:3, :3] for pose in poses))
    mean_rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    angles = [math.degrees(math.acos(np.clip((np.trace(mean_rotation.T @ pose[:3, :3]) - 1) / 2, -1, 1))) for pose in poses]
    rms_rotation = math.sqrt(np.mean(np.square(angles)))
    return mean_center, math.sqrt(np.mean(deviations**2)), rms_rotation, _transform(mean_rotation, mean_center), deviations


def initial_hand_eye(views: list) -> tuple:
    """OpenCV's hand-eye methods on these views (one session); returns the best's
    (name, camera_in_gripper, mean board_in_base, spread m, spread deg)."""
    gripper_r = [g[:3, :3] for _, g, _ in views]
    gripper_t = [g[:3, 3] for _, g, _ in views]
    board_r = [b[:3, :3] for _, _, b in views]
    board_t = [b[:3, 3] for _, _, b in views]
    best = None
    for name, method in HANDEYE_METHODS.items():
        rotation, translation = cv2.calibrateHandEye(gripper_r, gripper_t, board_r, board_t, method=method)
        camera_in_gripper = _transform(rotation, translation)
        _, rms_m, rms_deg, board_in_base, _ = board_spread(views, camera_in_gripper)
        if best is None or rms_m < best[3]:
            best = (name, camera_in_gripper, board_in_base, rms_m, rms_deg)
    return best


def refit_intrinsics(sessions: list) -> None:
    """Calibrate the intrinsics from the sessions' board images (the stream's own
    resolution and locked focus), backing up the old file; writes camera_intrinsics.json."""
    objects, images, size = [], [], IMAGE_SIZE
    detectors = _detectors()
    for session in sessions:
        for record in json.loads((session / "views.json").read_text()):
            board, detector = detectors[int(record.get("board", 0))]
            image = cv2.imread(str(session / record["image"]), cv2.IMREAD_GRAYSCALE)
            corners, ids, _, _ = detector.detectBoard(image)
            if corners is None or len(corners) < MIN_CORNERS or board.checkCharucoCornersCollinear(ids.reshape(-1, 1)):
                continue
            object_points, image_points = board.matchImagePoints(corners.reshape(-1, 1, 2), ids.reshape(-1, 1))
            objects.append(object_points.astype(np.float32))
            images.append(image_points.astype(np.float32))
    old_matrix, old_distortion = load_intrinsics()
    rms_px, matrix, distortion, _, _ = cv2.calibrateCamera(objects, images, size, old_matrix.copy(), old_distortion.copy(),
                                                          flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    print(f"Intrinsics from {len(objects)} board views at {size[0]}x{size[1]}: {rms_px:.2f} px rms; focal "
          f"{old_matrix[0, 0]:.1f} -> {matrix[0, 0]:.1f} px ({(matrix[0, 0] / old_matrix[0, 0] - 1) * 100:+.2f}%), center "
          f"({old_matrix[0, 2]:.0f}, {old_matrix[1, 2]:.0f}) -> ({matrix[0, 2]:.0f}, {matrix[1, 2]:.0f})")
    backup = CALIBRATION_DIR / time.strftime("camera_intrinsics_before_%Y%m%d_%H%M%S.json")
    backup.write_text(INTRINSICS_PATH.read_text())
    INTRINSICS_PATH.write_text(json.dumps({
        "image_size": list(size), "camera_matrix": matrix.tolist(), "distortion": distortion.ravel().tolist(),
        "rms_px": rms_px, "views": len(objects), "source": [str(s) for s in sessions],
        "note": "refit from hand-eye calibration board views (scripts/hand_eye_calibration.py --refit-intrinsics)",
    }, indent=2) + "\n")
    print(f"Wrote {INTRINSICS_PATH} (old one in {backup})")


def solve(sessions: list, use_commanded: bool, fit_model: bool = True, touch_dirs: list = ()) -> None:
    """Solve from one or more sessions: the joint reading model is shared; each
    session gets its own camera placement and board pose (the camera can shift
    on its mount between sessions), and the LAST session's are written."""
    print(f"Solving {', '.join(map(str, sessions))}" + (" (commanded angles)" if use_commanded else " (measured angles)") + "...")
    data = []
    for i, session in enumerate(sessions):
        data += load_views(session, use_commanded, i)
    if len(data) < 5:
        sys.exit(f"Error: only {len(data)} usable views; need at least 5 (more like 15+).")
    # Board slots: each board as one session saw it (its own pose there: a board can be moved
    # between sessions), in session then capture order, so a board's last slot is its latest.
    slots = list(dict.fromkeys((view["session"], view["board"]) for view in data))
    for view in data:
        view["slot"] = slots.index((view["session"], view["board"]))

    def last_slot(board: int) -> int:
        return max(i for i, (_, b) in enumerate(slots) if b == board)

    boards_seen = sorted({b for _, b in slots})
    # Each touch is on its board as the last session that saw that board has it.
    touches = [touch for directory in touch_dirs for touch in load_touches(directory)]
    for touch in touches:
        if touch["board"] not in boards_seen:
            sys.exit(f"Error: touches on board {touch['board']}, but no session given saw it")
        touch["slot"] = last_slot(touch["board"])
    if touches:
        print(f"  {len(touches)} fingertip touches (" + ", ".join(
            f"{sum(t['board'] == b for t in touches)} on board {b}" for b in sorted({t["board"] for t in touches})) + ")")
    chain = build_chain()
    model = current_model()
    cameras, boards = [], []
    fingertip = np.zeros(3)
    for i, session in enumerate(sessions):
        session_views = [v for v in data if v["session"] == i]
        if len(session_views) < 5:
            sys.exit(f"Error: {session} has only {len(session_views)} usable views")
        # The camera from the board this session saw most (hand-eye needs one fixed target).
        counts = {}
        for v in session_views:
            counts[v["board"]] = counts.get(v["board"], 0) + 1
        main = max(counts, key=counts.get)
        name, camera_in_gripper, _, _, _ = initial_hand_eye(
            gripper_poses([v for v in session_views if v["board"] == main], chain, model))
        cameras.append(camera_in_gripper)
        print(f"  {session.name}: {len(session_views)} views (" + ", ".join(
            f"{n} of board {b}" for b, n in sorted(counts.items())) + f"); hand-eye ({name}, current joint model) "
            f"from board {main}")
    for index, (session_index, board) in enumerate(slots):
        views = gripper_poses([v for v in data if v["slot"] == index], chain, model)
        center, rms_m, rms_deg, board_in_base, _ = board_spread(views, cameras[session_index])
        boards.append(board_in_base)
        print(f"    board {board} as {sessions[session_index].name} saw it: at ({center[0]:.3f}, {center[1]:.3f}, "
              f"{center[2]:.3f}) m, its views agree to {rms_m * 1000:.1f} mm / {rms_deg:.2f} deg")

    if fit_model:
        print(f"\nRefining the joint reading model (offsets: {', '.join(OFFSET_JOINTS)}; {CORRECTION_TERMS}-term "
              f"corrections: {', '.join(CORRECTION_JOINTS)}) with each session's camera and a level board at "
              f"z = {BOARD_SURFACE_Z_M * 1000:.0f} mm (every corner's pixel error)...")
        # With touches: label them, fit, then label again with the fitted model (a
        # touch labelled by the unfitted one can land on a neighbor) and refit.
        all_touches = touches
        for _ in range(2 if touches else 1):
            if touches:
                touches = []
                for index in sorted({t["slot"] for t in all_touches}):
                    on = boards[index]
                    touches += label_touches([t for t in all_touches if t["slot"] == index], chain, model,
                                             _level_board(math.atan2(on[1, 0], on[0, 0]), on[0, 3], on[1, 3]))
            model, cameras, boards, before_px, after_px, fingertip, misses_before, misses_after = refine(
                data, chain, model, cameras, boards, touches=touches)
        print(f"  corner error {before_px:.1f} -> {after_px:.1f} px rms")
        if touches:
            def miss_summary(misses):
                return (f"{math.sqrt(np.mean(np.sum(misses ** 2, axis=1))):.1f} mm rms, mean (x, y, z) "
                        f"({misses[:, 0].mean():+.1f}, {misses[:, 1].mean():+.1f}, {misses[:, 2].mean():+.1f})")
            print(f"  fingertip touches: model's fingertip off the touched corner {miss_summary(misses_before)} "
                  f"-> {miss_summary(misses_after)} mm")
            for touch, before, after in zip(touches, misses_before, misses_after):
                print(f"    corner {touch['corner']:2d}: ({before[0]:+5.1f}, {before[1]:+5.1f}, {before[2]:+5.1f}) -> "
                      f"({after[0]:+5.1f}, {after[1]:+5.1f}, {after[2]:+5.1f}) mm")
            print(f"  the physical fingertip sits ({fingertip[0] * 1000:+.1f}, {fingertip[1] * 1000:+.1f}, "
                  f"{fingertip[2] * 1000:+.1f}) mm from the URDF's tip point (tip frame; z = along the fingers)")
        for j in config.JOINTS:
            if j.name in OFFSET_JOINTS or j.name in CORRECTION_JOINTS:
                corrections = model["corrections"].get(j.name, [])
                low, high = model["ranges"].get(j.name, (-180, 180))
                curve = [config.true_from_nominal_deg(a, 0.0, tuple(corrections), (low, high)) - a for a in (-90, -45, 0, 45, 90)]
                print(f"  {j.name:15s} offset {model['offsets'].get(j.name, 0.0):+6.2f} deg; correction at -90/-45/0/45/90: "
                      + "/".join(f"{c:+.2f}" for c in curve) + f" (fitted over {low:.0f}..{high:.0f}, flat beyond)")
    camera_in_gripper, board_in_base = cameras[-1], boards[-1]
    if len(sessions) > 1:
        moved = np.linalg.inv(cameras[0]) @ cameras[-1]
        print(f"  camera placement, first vs last session: {np.linalg.norm(moved[:3, 3]) * 1000:.1f} mm, "
              f"{math.degrees(np.linalg.norm(Rotation.from_matrix(moved[:3, :3]).as_rotvec())):.2f} deg apart")

    # Per kind of pose: where each view puts the board center vs its session's
    # solved board (a stacker-pose bias here is what cup locating would be off by there).
    center_on_board = np.linalg.inv(_center_to_corner())[:, 3]
    errors = {}
    views = gripper_poses(data, chain, model)
    for view, (_, gripper_in_base, board_in_camera) in zip(data, views):
        seen = (gripper_in_base @ cameras[view["session"]] @ board_in_camera @ center_on_board)[:3]
        errors.setdefault(view["kind"], []).append(seen - boards[view["slot"]][:3, 3])
    for kind, kind_errors in sorted(errors.items()):
        e = np.array(kind_errors) * 1000
        print(f"  {kind} poses ({len(e)}): board center off by {math.sqrt(np.mean(np.sum(e ** 2, axis=1))):.1f} mm rms, "
              f"mean (x, y, z) ({e[:, 0].mean():+.1f}, {e[:, 1].mean():+.1f}, {e[:, 2].mean():+.1f}) mm")
    # The last session's views, per board it saw: how well they agree on where it is.
    last_session = len(sessions) - 1
    names, deviations, rotations = [], [], []
    for index, (session_index, board) in enumerate(slots):
        if session_index != last_session:
            continue
        last = [v for v, d in zip(views, data) if d["slot"] == index]
        _, slot_m, slot_deg, _, slot_deviations = board_spread(last, camera_in_gripper)
        print(f"  last session, board {board}: {len(last)} views agree to {slot_m * 1000:.1f} mm / {slot_deg:.2f} deg rms")
        names += [v[0] for v in last]
        deviations += list(slot_deviations)
        rotations += [slot_deg] * len(last)
    deviations = np.array(deviations)
    rms_m, rms_deg = math.sqrt(np.mean(deviations ** 2)), math.sqrt(np.mean(np.square(rotations)))
    worst = int(np.argmax(deviations))
    axis_angle = math.degrees(math.acos(np.clip(camera_in_gripper[2, 2], -1, 1)))
    print(f"Last session's views agree to {rms_m * 1000:.1f} mm / {rms_deg:.2f} deg rms; worst {names[worst]}: "
          f"{deviations[worst] * 1000:.1f} mm. Camera {np.linalg.norm(camera_in_gripper[:3, 3]) * 100:.1f} cm from the tip, "
          f"looking {axis_angle:.0f} deg off the approach axis.")

    # Each board as the last session that saw it has it; boards not in this solve keep their saved pose.
    boards_in_base = {}
    with contextlib.suppress(OSError, ValueError):
        saved = json.loads(OUTPUT.read_text())
        boards_in_base = dict(saved.get("boards_in_base", {}))
        boards_in_base.setdefault("0", saved.get("board_in_base"))
    for board in boards_seen:
        boards_in_base[str(board)] = boards[last_slot(board)].tolist()
    boards_in_base = {k: v for k, v in boards_in_base.items() if v is not None}
    OUTPUT.write_text(json.dumps({
        "camera_in_gripper": camera_in_gripper.tolist(),
        "gripper_frame": "finger_pinch_link (xarm_ik chain tip)",
        "method": "hand-eye" + (" + joint reading model refinement" if fit_model else ""),
        "views": len(data),
        "board_spread_rms_mm": rms_m * 1000,
        "board_spread_rms_deg": rms_deg,
        "board_in_base": boards_in_base.get("0", board_in_base.tolist()),  # the stack side's (board 0)
        "boards_in_base": boards_in_base,
        "joint_offsets_deg": model["offsets"],
        "joint_corrections_deg": model["corrections"],
        "joint_correction_range_deg": model.get("ranges", {}),
        "joint_angles": "commanded" if use_commanded else "measured",
        "sessions": [str(s) for s in sessions],
        "touches": [str(d) for d in touch_dirs],
        "fingertip_offset_m": fingertip.tolist(),
    }, indent=2) + "\n")
    print(f"Wrote {OUTPUT} (camera from {sessions[-1].name}; board(s) " + ", ".join(
        f"{b} from {sessions[slots[last_slot(b)][0]].name}" for b in boards_seen) + ")")
    if fit_model:
        config.JOINT_OFFSETS_PATH.write_text(json.dumps({
            "zero_offsets_deg": model["offsets"],
            "corrections_deg": model["corrections"],
            "correction_range_deg": model.get("ranges", {}),
            "note": "true joint angle = nominal + offset + sum_k c_k (nominal / 90 deg)^k, k = 1.. "
                    "(xarm_ik/config.py); fitted by scripts/hand_eye_calibration.py",
            "board_surface_z_m": BOARD_SURFACE_Z_M,
            "sessions": [str(s) for s in sessions],
            "touches": [str(d) for d in touch_dirs],
        }, indent=2) + "\n")
        print(f"Wrote {config.JOINT_OFFSETS_PATH} (every script uses this from now on)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="move the real arm and capture")
    parser.add_argument("--sim", action="store_true", help="show the motion in PyBullet")
    parser.add_argument("--solve", metavar="DIR", nargs="+",
                        help="only solve saved session(s), no robot; the camera placement written is the last one's")
    parser.add_argument("--touches", metavar="DIR", nargs="+", default=[],
                        help="with --solve: fingertip touch sessions (scripts/tip_touch.py) on the last session's board")
    parser.add_argument("--refit-intrinsics", action="store_true",
                        help="with --solve: first recalibrate camera_intrinsics.json from the sessions' board images")
    parser.add_argument("--board", type=int, choices=range(len(charuco_board.BOARD_NAMES)), default=None,
                        help="capture only this board: 0 the stack side's, 1 the pyramid side's (default: both, "
                             "one after the other, in one session)")
    parser.add_argument("--locate-board", action="store_true",
                        help="with --execute: find --board with the camera (looking around its side of the "
                             "robot) and save its pose to plan with; no calibration")
    parser.add_argument("--board-xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
                        help="board center in the base frame, m, if it moved (default: from the previous calibration)")
    parser.add_argument("--poses", type=int, default=32, help="random poses to visit per board (default: 32)")
    parser.add_argument("--stacker-poses", type=int, default=64,
                        help="poses per board from the stacking script's own moves (dry-run and real-run), after "
                             "the random ones (default: 64)")
    parser.add_argument("--record-stacker-poses", action="store_true",
                        help=f"re-record the stacking script's moves (else cached in {STACKER_POSES_PATH.name})")
    parser.add_argument("--seed", type=int, default=0, help="pose sampling seed (default: 0)")
    parser.add_argument("--move-ms", type=int, default=1500, help="duration of each move (default: 1500)")
    parser.add_argument("--settle-s", type=float, default=0.7, help="wait after each move before capturing (default: 0.7)")
    parser.add_argument("--device", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--com-port", default="USB", help="passed to xarm.Controller (default: USB)")
    parser.add_argument("--no-show", action="store_true", help="don't show each capture in a window")
    parser.add_argument("--commanded", action="store_true", help="solve with commanded, not measured, joint angles")
    parser.add_argument("--no-joint-offsets", action="store_true",
                        help="keep the current joint reading model (don't refine it or write joint_offsets.json)")
    args = parser.parse_args()

    if args.solve:
        sessions = [Path(s) for s in args.solve]
        if args.refit_intrinsics:
            refit_intrinsics(sessions)
        solve(sessions, args.commanded, not args.no_joint_offsets, [Path(d) for d in args.touches])
        return
    if args.locate_board:
        if not args.execute:
            parser.error("--locate-board needs --execute (it looks with the camera)")
        if args.board is None:
            parser.error("--locate-board needs --board (which one to look for)")
        locate_board(args)
        return
    if args.board_xy is not None and args.board is None:
        parser.error("--board-xy needs --board (which board it is)")
    if args.board_xy is not None and np.linalg.norm(args.board_xy) < 0.1:
        parser.error("--board-xy must be at least 0.1 m from the base")
    if args.board is not None:
        boards = [args.board]
    else:  # every board located so far, the stack side's first
        boards = [b for b in range(len(charuco_board.BOARD_NAMES)) if board_prior(b) is not None]
        missing = [b for b in range(len(charuco_board.BOARD_NAMES)) if b not in boards]
        if missing:
            print(f"Note: board(s) {', '.join(map(str, missing))} not located yet (--locate-board --board N "
                  "--execute), so not captured")
        if not boards:
            parser.error("no board located yet; pass --board and --board-xy")
    # Per board: home, its random poses (planned from home), then its stacker poses
    # (from where those end). Homing between boards keeps the big base swing out of
    # the planned moves.
    planner = GripperPoseController(dry_run=True)
    plan = []
    for board in boards:
        print(f"Board {board} ({charuco_board.BOARD_NAMES[board]}):")
        poses = plan_poses(planner, args.board_xy, args.poses, args.seed, board)
        tilts, rolls = [p[3] for p in poses], [p[4] for p in poses]
        yaws = [math.degrees(math.atan2(p[0], -p[1])) for p in poses]
        print(f"  planned {len(poses)} random poses: tilt {min(tilts):.0f}..{max(tilts):.0f}, roll {min(rolls):.0f}.."
              f"{max(rolls):.0f}, base yaw {min(yaws):.0f}..{max(yaws):.0f} deg off the board line")
        if args.stacker_poses > 0:
            camera_in_gripper, board_in_base = load_prior(args.board_xy, board)
            stacker = plan_stacker_poses(planner.chain, args.stacker_poses, args.seed, args.record_stacker_poses,
                                         board, args.board_xy)
            stacker = joint_tour(planner.chain, stacker, planner._joint_angles_rad, camera_in_gripper,
                                 float(board_in_base[2, 3]))
            print(f"  then {len(stacker)} stacker poses (in a safe order)")
            poses = poses + stacker
        if plan:
            plan.append(None)  # home first
        plan += [(board, pose) for pose in poses]
    print(f"{sum(item is not None for item in plan)} poses in all.")
    session = capture_session(args, plan)
    if session is not None and 0 not in boards:
        print(f"Saved {session}. Solve it together with the stack side's session(s) (one session alone fits the "
              f"joint model to this side only):\n  uv run python scripts/hand_eye_calibration.py --solve "
              f"<stack-side session(s)> {session} --touches <touch session(s)>")
    elif session is not None:
        solve([session], args.commanded, not args.no_joint_offsets)
    else:
        print("Dry run done (no camera, nothing saved). Add --execute to run it on the arm.")


if __name__ == "__main__":
    main()
