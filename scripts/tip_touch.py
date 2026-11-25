"""Fingertip touch calibration, hand-guided: pin where the fingertip really is.

The hand-eye calibration only sees where the camera is. Anything that moves
the fingertip relative to the camera without moving the camera (a constant
wrist-pitch error looks just like the camera sitting at another angle on its
mount) it can't tell apart, so its fits put the fingertip 1-3 cm off at
negative-shoulder poses. Here the closed fingertips touch ChArUco inner corners
(known points: the board lies level at z = -3 mm, located by the last hand-eye
calibration) in varied arm configurations, down to the pyramid's bottom-row
releases (shoulder ~-77 deg). The hand-eye solve then fits the joint model to
the camera views and these touches together
(hand_eye_calibration.py --solve SESSION... --touches THIS_DIR).

The arm homes, closes the gripper, then (once you're holding it) its five arm
joints go limp; the gripper stays powered and closed. Move the arm by hand
until the very ends of the closed fingers touch the paper ON a corner, hold it
still and press Enter in the window: it records the servos' readings once
they're steady (the nearest corner, by the model, is the one recorded; any
corner counts). The window suggests targets spread over the arm's
configurations (the corner, and roughly the shoulder / elbow / wrist angles to
aim for, with the live angles beside them; map: touch_corners.png); matching
them closely doesn't matter, varying them does. Keys (in the window): Enter
record, n next suggestion, u undo the last touch, q finish. At the end it asks
(in the terminal) before powering the arm again and homing.

Don't move the board after the hand-eye session you'll solve this with. --board 1
touches the pyramid side's board (locate it first: hand_eye_calibration.py
--locate-board --board 1 --execute).

Usage:
    uv run python scripts/tip_touch.py             # dry run: list the suggested touches
    uv run python scripts/tip_touch.py --execute   # on the arm
    uv run python scripts/tip_touch.py --board 1 --execute   # on the pyramid side's board
"""

import argparse
import contextlib
import io
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

import hand_eye_calibration as he
from run_csv import approach_vector
from xarm_ik import GripperPoseController, charuco_board, config

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs" / "calibration"
HAND_EYE_PATH = ROOT / "camera_hand_eye.json"

WINDOW = "tip touch"
STEADY_READS = 4  # a touch is recorded from this many reads in a row ...
STEADY_DEG = 0.6  # ... each joint within this of the others (the arm held still)
STEADY_TIMEOUT_S = 4.0

# Suggested touches are spread over every arm configuration that can put the
# closed fingertips on a corner, the farthest-back shoulder first (the pyramid's
# bottom-row releases).
SHOULDER_GRID_DEG = np.arange(-80.0, 41.0, 3.0)
ELBOW_GRID_DEG = np.arange(-40.0, 125.0, 4.0)
TILT_LIMITS_DEG = (5.0, 80.0)  # fingers pointing down (not past horizontal), so the fingertips touch first
CORNER_NEAR_M = 0.01  # a grid configuration's fingertip reach this near a corner's gets solved onto it


def board_corners(board_index: int = 0) -> tuple:
    """(board-in-base about its center, inner corners in the board's corner frame (N x 3), same in base (N x 3))."""
    board_in_base = he.board_prior(board_index)
    if board_in_base is None:
        raise SystemExit(f"Error: board {board_index} hasn't been located; run hand_eye_calibration.py "
                         f"--locate-board --board {board_index} --execute first")
    board = charuco_board.make_board(board_index=board_index)
    width, height = board.getChessboardSize()
    square = board.getSquareLength()
    xs, ys = np.meshgrid(np.arange(1, width) * square, np.arange(1, height) * square)
    local = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)  # corner frame (origin at a board corner)
    center_frame = local - [width * square / 2, height * square / 2, 0]
    in_base = (board_in_base[:3, :3] @ center_frame.T).T + board_in_base[:3, 3]
    return board_in_base, local, in_base


def _deg(angles_rad) -> dict:
    return {j.name: math.degrees(a) for j, a in zip(config.JOINTS, angles_rad[list(config.ACTIVE_LINK_INDICES)])}


def plan_touches(arm: GripperPoseController, board_in_base, in_base: np.ndarray, count: int) -> list:
    """`count` suggested touches spread over the arm configurations that can touch a
    corner: dicts with the corner index, tilt and touch joint angles (deg)."""
    names = [j.name for j in config.JOINTS]
    radii = np.linalg.norm(in_base[:, :2], axis=1)
    board_z = float(board_in_base[2, 3])

    def fk(shoulder, elbow, wrist):
        angles = np.zeros(len(arm.chain.links))
        angles[list(config.ACTIVE_LINK_INDICES)] = np.radians([0.0, shoulder, elbow, wrist, 0.0])
        pose = arm.chain.forward_kinematics(angles)
        return pose[:3, 3], pose[:3, 2]

    candidates = []
    wrists = np.arange(-125.0, 125.1, 2.5)
    for shoulder in SHOULDER_GRID_DEG:
        for elbow in ELBOW_GRID_DEG:
            # Wrist angles putting the fingertip at the board's surface: sign changes of the height, bisected.
            heights = [fk(shoulder, elbow, w)[0][2] - board_z for w in wrists]
            for w0, w1, h0, h1 in zip(wrists, wrists[1:], heights, heights[1:]):
                if h0 * h1 > 0:
                    continue
                for _ in range(20):
                    mid = (w0 + w1) / 2
                    hm = fk(shoulder, elbow, mid)[0][2] - board_z
                    w0, h0, w1 = (mid, hm, w1) if h0 * hm > 0 else (w0, h0, mid)
                wrist = (w0 + w1) / 2
                tip, approach = fk(shoulder, elbow, wrist)
                tilt = math.degrees(math.acos(np.clip(-approach[2], -1, 1)))
                if not TILT_LIMITS_DEG[0] <= tilt <= TILT_LIMITS_DEG[1] or approach[:2] @ tip[:2] < 0:
                    continue  # fingers must point down and lean outward, as grasps do
                if tip[0] > 0:
                    continue  # reaching back over the base (base yaw 0 faces -x), not out in front
                k = int(np.argmin(np.abs(radii - np.linalg.norm(tip[:2]))))
                if abs(radii[k] - np.linalg.norm(tip[:2])) < CORNER_NEAR_M:
                    candidates.append((float(shoulder), float(elbow), float(wrist), k, tip, tilt))

    def solve(candidate):
        """Exact touch joints for a candidate (base yaw onto the corner, IK onto it), or None."""
        shoulder, elbow, wrist, k, tip, tilt = candidate
        turn = math.atan2(in_base[k][1], in_base[k][0]) - math.atan2(tip[1], tip[0])
        yaw = math.degrees((turn + math.pi) % (2 * math.pi) - math.pi)
        seed = {"base_yaw": yaw, "shoulder_pitch": shoulder, "elbow_pitch": elbow, "wrist_pitch": wrist, "wrist_roll": 0.0}
        if any(not j.limit_deg[0] <= seed[j.name] <= j.limit_deg[1] for j in config.JOINTS):
            return None
        corner = in_base[k]
        approach = approach_vector(corner[0], corner[1], tilt)
        arm._joint_angles_rad = he._chain_angles(seed, arm.chain)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                arm.move_to_pose(*corner, approach=approach, duration_ms=1, position_tolerance_m=0.001)
        except ValueError:
            return None
        touch = _deg(arm._joint_angles_rad)
        if max(abs(touch[n] - seed[n]) for n in names) > 8:
            return None  # the IK went to another branch
        return {"corner": k, "tilt_deg": round(tilt, 1), "touch_deg": touch}

    def features(candidate):
        shoulder, elbow, wrist, k, tip, tilt = candidate
        return np.array([shoulder, elbow, wrist, tilt]) / 10

    # Greedy spread (farthest from those picked), the farthest-back shoulder first.
    left = sorted(candidates, key=lambda c: c[0])
    picked, out = [], []
    while left and len(out) < count:
        if not picked:
            candidate = left.pop(0)
        else:
            i = max(range(len(left)), key=lambda i: min(np.linalg.norm(features(left[i]) - features(p)) for p in picked))
            candidate = left.pop(i)
        touch = solve(candidate)
        if touch:
            picked.append(candidate)
            out.append(touch)
    return out


def tip_of(arm: GripperPoseController, angles_deg: dict) -> np.ndarray:
    """The model's fingertip position (m) for these joint angles."""
    return arm.chain.forward_kinematics(he._chain_angles(angles_deg, arm.chain))[:3, 3]


def show(lines: list) -> int:
    """Draw `lines` (str, or (str, BGR color)) on a dark image, show it; returns the key pressed (or -1)."""
    image = np.full((420, 900, 3), 30, np.uint8)
    for i, line in enumerate(lines):
        text, color = line if isinstance(line, tuple) else (line, (235, 235, 235))
        cv2.putText(image, text, (14, 34 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
    cv2.imshow(WINDOW, image)
    return cv2.waitKey(50)


def steady_reading(arm: GripperPoseController) -> dict | None:
    """The joints' readings averaged over STEADY_READS reads in a row that agree to
    STEADY_DEG (the arm held still), or None if that doesn't happen in time."""
    reads = []
    deadline = time.monotonic() + STEADY_TIMEOUT_S
    while time.monotonic() < deadline:
        read = arm.get_measured_joint_angles_deg()
        if len(read) < len(config.JOINTS):
            continue  # a failed read: try again
        reads = (reads + [read])[-STEADY_READS:]
        if len(reads) == STEADY_READS:
            spread = max(max(r[n] for r in reads) - min(r[n] for r in reads) for n in read)
            if spread <= STEADY_DEG:
                return {n: float(np.mean([r[n] for r in reads])) for n in read}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="use the real arm")
    parser.add_argument("--count", type=int, default=12, help="suggested touches (default: 12)")
    parser.add_argument("--board", type=int, choices=range(len(charuco_board.BOARD_NAMES)), default=0,
                        help="the board touched: 0 the stack side's (default), 1 the pyramid side's")
    parser.add_argument("--com-port", default="USB", help="passed to xarm.Controller (default: USB)")
    args = parser.parse_args()

    planner = GripperPoseController(dry_run=True)
    board_in_base, local, in_base = board_corners(args.board)
    suggestions = plan_touches(planner, board_in_base, in_base, args.count)
    print(f"Board {args.board} ({charuco_board.BOARD_NAMES[args.board]}) center ({board_in_base[0, 3]:.3f}, "
          f"{board_in_base[1, 3]:.3f}, {board_in_base[2, 3]:.3f}) m (from {HAND_EYE_PATH.name}). {len(suggestions)} "
          "suggested touches (map: touch_corners.png; corner numbers are the same on both boards):")
    for k, t in enumerate(suggestions):
        j = t["touch_deg"]
        print(f"  [{k + 1}] corner {t['corner']:2d}, tilt {t['tilt_deg']:.0f}: base {j['base_yaw']:+.0f}, "
              f"shoulder {j['shoulder_pitch']:+.0f}, elbow {j['elbow_pitch']:+.0f}, wrist {j['wrist_pitch']:+.0f} deg")
    if not args.execute:
        print("Dry run (suggestions only). Add --execute to use the arm.")
        return

    arm = GripperPoseController(com_port=args.com_port, dry_run=False, closed_loop=True)
    session = LOG_DIR / time.strftime("tiptouch_%Y%m%d_%H%M%S")
    session.mkdir(parents=True)
    records = []
    print("Homing...")
    arm.home()
    arm.close_gripper()
    input("\nHold the arm: its joints go limp when you press Enter here (the gripper stays closed). ")
    arm.servos_off()
    print("Limp. Use the window: Enter record, n next suggestion, u undo, q finish.")
    names = [j.name for j in config.JOINTS]
    k, message = 0, ""
    try:
        while True:
            live = arm.get_measured_joint_angles_deg()
            target = suggestions[k % len(suggestions)] if suggestions else None
            lines = [(f"{len(records)} touches recorded  (any corner counts; vary the arm's pose)", (120, 220, 120))]
            if target:
                lines.append(f"Suggestion {k % len(suggestions) + 1}/{len(suggestions)}: corner {target['corner']} "
                             f"(touch_corners.png), tilt ~{target['tilt_deg']:.0f} deg")
                for name in names[:4]:
                    have = live.get(name)
                    lines.append(f"   {name:15s} aim {target['touch_deg'][name]:+7.1f}   now "
                                 + ("   ?   " if have is None else f"{have:+7.1f}"))
            if len(live) == len(names):
                tip = tip_of(arm, live)
                nearest = int(np.argmin(np.linalg.norm(in_base - tip, axis=1)))
                off = (tip - in_base[nearest]) * 1000
                lines.append(f"Model's fingertip: corner {nearest} nearest, ({off[0]:+.0f}, {off[1]:+.0f}, {off[2]:+.0f}) mm from it")
            lines += ["", "Touch the corner with the closed fingertips, hold still, press Enter.",
                      "Enter: record   n: next suggestion   u: undo last   q: finish", (message, (100, 200, 255))]
            key = show(lines)
            if key in (13, 10):
                reading = steady_reading(arm)
                if reading is None:
                    message = "not steady: hold the arm still and try again"
                    continue
                tip = tip_of(arm, reading)
                # The corner touched: the model's nearest (the model is off by mm, not a 25 mm square).
                touched = int(np.argmin(np.linalg.norm(in_base - tip, axis=1)))
                records.append({
                    "board": args.board, "corner": touched, "corner_in_board_m": local[touched].tolist(),
                    "suggested_corner": target["corner"] if target else None, "hand_guided": True,
                    "commanded_deg": reading, "measured_deg": reading,
                    "joint_offsets_deg": {j.name: j.zero_offset_deg for j in config.JOINTS},
                    "joint_corrections_deg": {j.name: list(j.correction_deg) for j in config.JOINTS if j.correction_deg},
                    "joint_correction_range_deg": {j.name: list(j.correction_range_deg) for j in config.JOINTS
                                                   if j.correction_range_deg},
                })
                (session / "touches.json").write_text(json.dumps(records, indent=1))
                off = (tip - in_base[touched]) * 1000
                message = (f"recorded corner {touched}; the model put the fingertip ({off[0]:+.1f}, {off[1]:+.1f}, "
                           f"{off[2]:+.1f}) mm from it")
                print(f"  {message}")
                k += 1
            elif key == ord("n"):
                k += 1
                message = ""
            elif key == ord("u") and records:
                records.pop()
                (session / "touches.json").write_text(json.dumps(records, indent=1))
                message = "removed the last touch"
                print(f"  {message}")
            elif key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()
        input("\nHold the arm somewhere safe, clear of the board: it powers up and homes when you press Enter here. ")
        print("Homing...")
        arm.home(duration_ms=3000)
    print(f"{len(records)} touches saved to {session / 'touches.json'}")
    if records:
        print("Solve with: uv run python scripts/hand_eye_calibration.py --solve <hand-eye session(s), latest last> "
              f"--touches {session}")


if __name__ == "__main__":
    main()
