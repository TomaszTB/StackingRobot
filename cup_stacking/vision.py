"""Finding cups: with the wrist camera (CameraVision) or in a simulated world (SimulatedVision).

Both have look(label) -> [Cup], set_status, snapshot and close; the simulation also has
picked and placed, to update its world.
"""

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from xarm_ik import config, cup_markers
from xarm_ik.camera import IMAGE_SIZE, FrameGrabber, load_exposure, load_intrinsics, open_camera
from xarm_ik.charuco_board import BoardLocator
from xarm_ik.live_view import LiveView

from .layout import CUP_NEAR_M, GRIP_BELOW_TOP_M, STACK_BOTTOM_TOP_Z, STACK_NEST_STEP_M, Cup, board_in_base

LOOK_SETTLE_S = 0.5  # wait before a look's frames
LOOK_FRAMES = 5  # per look; each cup's position is the median over them
FRAME_DELAY_S = 0.04  # a frame read this soon after a request may have been exposed before it


class CameraVision:
    """Locates cup markers in camera frames, placing the camera by the arm's measured joints
    and the hand-eye calibration ("arm"), or by board 0 in view ("board", falling back to the
    arm). Every look compares both where it can and records them in <session>/looks.json."""

    def __init__(self, arm, camera_in_gripper, session: Path, device: int, show: bool, locate: str = "arm"):
        self.arm, self.camera_in_gripper, self.session, self.locate = arm, camera_in_gripper, session, locate
        self.board = BoardLocator(board_in_base())
        self.records = []
        self.matrix, self.distortion = load_intrinsics()
        self.camera = open_camera(device, IMAGE_SIZE)
        exposure = load_exposure()
        print(f"Camera exposure: {f'locked at {exposure}' if exposure else 'auto (run scripts/exposure_tuner.py to lock it)'}")
        self.grabber = FrameGrabber(self.camera)
        self.view = LiveView(self.grabber) if show else None
        self.detector = cup_markers.detector()
        half = cup_markers.MARKER_LENGTH_M / 2
        self.marker_corners = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], np.float32)
        self.images = 0  # saved so far, numbering the next

    def set_status(self, text: str):
        if self.view:
            self.view.status = text
            self.view.update(force=True)

    def _wait(self, seconds: float):
        if self.view:
            self.view.wait(seconds)
        else:
            time.sleep(seconds)

    def look(self, label: str) -> list:
        self.set_status(f"looking at {label}...")
        self._wait(LOOK_SETTLE_S)
        measured = {**self.arm.get_joint_angles_deg(), **self.arm.get_measured_joint_angles_deg()}
        angles = np.zeros(len(self.arm.chain.links))
        for i, joint in zip(config.ACTIVE_LINK_INDICES, config.JOINTS):
            angles[i] = math.radians(measured[joint.name])
        camera_by_arm = self.arm.chain.forward_kinematics(angles) @ self.camera_in_gripper
        frames = self.grabber.frames_after(time.monotonic() + FRAME_DELAY_S, LOOK_FRAMES, wait=self._wait)
        by_arm, by_board, frame, board_found = {}, {}, None, []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            camera_by_board = self.board.camera_in_base(gray, self.matrix, self.distortion)
            board_found.append(camera_by_board is not None)
            corners, ids, _ = self.detector.detectMarkers(gray)
            for marker_corners, marker_id in zip(corners, [] if ids is None else ids.ravel()):
                if marker_id not in cup_markers.CUP_IDS:
                    continue
                ok, rvec, tvec = cv2.solvePnP(self.marker_corners, marker_corners.reshape(4, 2), self.matrix,
                                              self.distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    in_camera = np.append(tvec.ravel(), 1.0)
                    by_arm.setdefault(int(marker_id), []).append((camera_by_arm @ in_camera)[:3])
                    if camera_by_board is not None:
                        by_board.setdefault(int(marker_id), []).append((camera_by_board @ in_camera)[:3])
        cups, record = [], {"label": label, "locate": self.locate, "board_frames": f"{sum(board_found)}/{len(board_found)}",
                            "cups": {}}
        for marker_id, points in by_arm.items():
            arm_top = np.median(points, axis=0)
            entry = {"by_arm": [round(float(v), 4) for v in arm_top]}
            top, how = arm_top, "arm"
            if marker_id in by_board:
                board_top = np.median(by_board[marker_id], axis=0)
                entry["by_board"] = [round(float(v), 4) for v in board_top]
                difference = arm_top - board_top
                entry["arm_minus_board_mm"] = [round(float(v) * 1000, 1) for v in difference]
                print(f"    cup {marker_id}: arm-based is ({difference[0] * 1000:+.0f}, {difference[1] * 1000:+.0f}, "
                      f"{difference[2] * 1000:+.0f}) mm from board-based")
                if self.locate == "board":
                    top, how = board_top, "board"
            elif self.locate == "board":
                print(f"    cup {marker_id}: board not found, located by the arm")
            entry["used"] = how
            record["cups"][str(marker_id)] = entry
            cups.append(Cup(marker_id, top))
        self.records.append(record)
        (self.session / "looks.json").write_text(json.dumps(self.records, indent=1) + "\n")
        if frame is not None:
            self._save_annotated(frame.copy(), label, cups)
        if self.view:
            self.view.lines = [f"last look ({label}): " + (f"{len(cups)} cup(s)" if cups else "no cups")] + [
                f"  cup {c.marker_id} top ({c.top[0]:.3f}, {c.top[1]:.3f}, {c.top[2]:.3f})" for c in cups]
        return cups

    def snapshot(self, label: str):
        """Save the next frame as-is."""
        frames = self.grabber.frames_after(time.monotonic() + FRAME_DELAY_S, 1, wait=self._wait)
        if frames:
            self._write(frames[-1], label)

    def _save_annotated(self, frame, label, cups):
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i, cup in enumerate(cups):
            text = f"cup {cup.marker_id}: ({cup.top[0]:.3f}, {cup.top[1]:.3f}, {cup.top[2]:.3f})"
            cv2.putText(frame, text, (14, 34 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        self._write(frame, label)

    def _write(self, frame, label):
        cv2.imwrite(str(self.session / f"look_{self.images:02d}_{label}.jpg"), frame)
        self.images += 1

    def close(self):
        self.grabber.stop()
        if self.view:
            self.view.close()
        self.camera.release()


class SimulatedVision:
    """A world of cups for dry runs; a cup is hidden while another sits on it."""

    def __init__(self, cup_count: int, stack_xy):
        # Cup 0 at the bottom, so ids count down from the top.
        self.cups = [Cup(i, np.array([*stack_xy, STACK_BOTTOM_TOP_Z + i * STACK_NEST_STEP_M]))
                     for i in reversed(range(cup_count))]

    def look(self, label: str) -> list:
        return [cup for cup in self.cups if not any(
            other.top[2] > cup.top[2] + 0.005 and np.linalg.norm(other.top[:2] - cup.top[:2]) < CUP_NEAR_M
            for other in self.cups)]

    def picked(self, cup: Cup):
        self.cups = [c for c in self.cups if c.marker_id != cup.marker_id]

    def placed(self, cup: Cup, grip):
        self.cups.append(Cup(cup.marker_id, np.array([grip[0], grip[1], grip[2] + GRIP_BELOW_TOP_M])))

    def set_status(self, text: str):
        pass

    def snapshot(self, label: str):
        pass

    def close(self):
        pass
