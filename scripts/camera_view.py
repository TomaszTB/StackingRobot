"""Show the webcam live, to check it works and aim it.

The overlay shows resolution, frame rate and focus. With --board (or `b`), it
also finds the ChArUco board: its corners, and (using camera_intrinsics.json)
its distance and tilt.

Keys: q / Esc quit, b toggle the board overlay, s save a snapshot (full
resolution, no overlay) to snapshots/.

Usage:
    uv run python scripts/camera_view.py
    uv run python scripts/camera_view.py --board --focus 68   # another focus (C615 steps: 0, 17, 34, 51, 68, ...)
"""

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from xarm_ik import charuco_board
from xarm_ik.camera import CAMERA_FOCUS, IMAGE_SIZE, load_intrinsics, open_camera

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "snapshots"
WINDOW = "camera"

# Fewer corners than this (or all in a line) give no trustworthy board pose.
MIN_POSE_CORNERS = 6


def annotate_board(image, gray, detector, board, intrinsics) -> str:
    """Draw the board's corners (and axes, with intrinsics) on `image`; returns a status line."""
    corners, ids, _, _ = detector.detectBoard(gray)
    if corners is None or len(corners) < 4:
        return "board: not found"
    cv2.aruco.drawDetectedCornersCharuco(image, corners.reshape(-1, 1, 2), ids.reshape(-1, 1), (0, 255, 0))
    status = f"board: {len(corners)} corners"
    if intrinsics is None:
        return status
    if len(corners) < MIN_POSE_CORNERS or board.checkCharucoCornersCollinear(ids.reshape(-1, 1)):
        return f"{status} (too few / in a line for a pose)"
    matrix, distortion = intrinsics
    object_points, image_points = board.matchImagePoints(corners.reshape(-1, 1, 2), ids.reshape(-1, 1))
    try:
        # IPPE: made for planar targets like the board.
        ok, rvec, tvec = cv2.solvePnP(object_points, image_points, matrix, distortion, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error as e:
        return f"{status} (pose failed: {str(e).strip().splitlines()[-1]})"
    if not ok:
        return status
    cv2.drawFrameAxes(image, matrix, distortion, rvec, tvec, charuco_board.SQUARE_LENGTH_M * 2, 3)
    tilt = math.degrees(math.acos(min(1.0, abs(cv2.Rodrigues(rvec)[0][2, 2]))))
    return f"{status}, {float(np.linalg.norm(tvec)) * 100:.1f} cm away, tilted {tilt:.0f} deg"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0, help="camera index, /dev/video<N> (default: 0)")
    parser.add_argument("--focus", type=int, default=CAMERA_FOCUS, help=f"locked focus (default: {CAMERA_FOCUS})")
    parser.add_argument("--board", action="store_true", help="start with the board overlay on")
    parser.add_argument("--display-scale", type=float, default=1.0, help="window size vs the image (default: 1.0)")
    args = parser.parse_args()

    try:
        capture = open_camera(args.device, IMAGE_SIZE, focus=args.focus)
    except RuntimeError as e:
        sys.exit(f"Error: {e}")
    board = charuco_board.make_board()
    detector = cv2.aruco.CharucoDetector(board)
    try:
        intrinsics = load_intrinsics()
    except (OSError, ValueError) as e:
        print(f"No usable intrinsics ({e}); the board overlay will show corners only.")
        intrinsics = None

    show_board = args.board
    frame_times = []
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, int(IMAGE_SIZE[0] * args.display_scale), int(IMAGE_SIZE[1] * args.display_scale))
    while True:
        ok, frame = capture.read()
        if not ok:
            print("Error: camera stopped returning frames.", file=sys.stderr)
            break
        frame_times = [t for t in frame_times if t > time.monotonic() - 1.0] + [time.monotonic()]

        shown = frame.copy()
        focus = "auto" if capture.get(cv2.CAP_PROP_AUTOFOCUS) else "locked"
        lines = [f"{frame.shape[1]}x{frame.shape[0]}  {len(frame_times)} fps  focus {capture.get(cv2.CAP_PROP_FOCUS):.0f} ({focus})"]
        if show_board:
            lines.append(annotate_board(shown, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), detector, board, intrinsics))
        lines.append("q quit   b board   s snapshot")
        for i, line in enumerate(lines):
            y = 30 + i * 30
            cv2.putText(shown, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(shown, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, shown)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("b"):
            show_board = not show_board
        if key == ord("s"):
            SNAPSHOT_DIR.mkdir(exist_ok=True)
            path = SNAPSHOT_DIR / time.strftime("snapshot_%Y%m%d_%H%M%S.png")
            cv2.imwrite(str(path), frame)
            print(f"Saved {path}")

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
