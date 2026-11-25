"""Pick a locked exposure (and white balance) for the cups' ArUco markers, by hand: no robot.

Shows the webcam live with the cup-marker detection running, as the stacking
script does it (same detector, resolution and focus). Move the camera around
the cups by hand and adjust exposure and gain until every marker reads, from
the angles and distances the robot looks from (up to ~60 deg off straight-on,
~25 cm away). Then press w: the settings are saved to camera_exposure.json,
and every script that opens the camera (stack_cups, hand-eye calibration,
camera_view) locks them from then on. Delete the file to go back to auto.

Overlay: each marker's ID, distance and angle off straight-on, how often each
cup was read over the last 2 s, and how much of the image is blown out
(pure white). Glare shows as blown-out marker squares that stop reading.

Sliders show the values themselves: exposure (100 us units, so 50 = 5.0 ms),
gain, and white balance (kelvin). Moving the exposure or gain slider locks
exposure; moving the white balance slider locks white balance. Over exposure
~330 (33 ms) the frame rate drops below 30 fps, and motion blurs. White balance
hardly matters to detection (it runs on gray), but locking it keeps the colors
in saved images steady.

Keys:
    a      toggle auto-exposure (switching to manual starts from auto's values)
    b      toggle auto white balance (likewise)
    [ ]    exposure down / up ~10%
    x      sweep: hold the camera still on the cups; tries exposures from
           short to long and prints how many markers each one read
    w      save the current (locked) exposure, gain and white balance to
           camera_exposure.json (white balance saved as auto unless locked)
    s      snapshot (full resolution, no overlay) to snapshots/
    q/Esc  quit

Usage:
    uv run python scripts/exposure_tuner.py
    uv run python scripts/exposure_tuner.py --auto      # start on auto-exposure, ignoring the saved setting
"""

import argparse
import collections
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from xarm_ik import cup_markers
from xarm_ik.camera import (IMAGE_SIZE, EXPOSURE_PATH, EXPOSURE_RANGE, GAIN_RANGE, WHITE_BALANCE_RANGE, load_exposure,
                            load_intrinsics, open_camera, save_exposure, set_exposure, set_white_balance)

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "snapshots"
WINDOW = "exposure tuner"
EXPOSURE_SLIDER_MAX = 1000  # 100 ms; longer is far past useful (and the slider would be too coarse)
RATE_WINDOW_S = 2.0
BLOWN_OUT = 250  # gray level counted as blown out
SWEEP = (10, 15, 20, 30, 40, 60, 80, 110, 150, 200, 260, 330, 450)
SWEEP_SETTLE_FRAMES = 6  # the camera takes a few frames to apply a new exposure
SWEEP_FRAMES = 10


class Markers:
    """The cup-marker detection, with each marker's distance and viewing angle."""

    def __init__(self):
        self.detector = cup_markers.detector()
        try:
            self.matrix, self.distortion = load_intrinsics()
        except (OSError, ValueError) as e:
            print(f"No usable intrinsics ({e}): no distances or angles.")
            self.matrix = None
        half = cup_markers.MARKER_LENGTH_M / 2
        self.object_points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
                                      np.float32)

    def detect(self, gray) -> list:
        """[(id, corners (4x2), distance m or None, degrees off straight-on or None)] for the cups' IDs."""
        corners, ids, _ = self.detector.detectMarkers(gray)
        found = []
        for marker_corners, marker_id in zip(corners, [] if ids is None else ids.ravel()):
            if marker_id not in cup_markers.CUP_IDS:
                continue
            points = marker_corners.reshape(4, 2)
            distance = angle = None
            if self.matrix is not None:
                ok, rvec, tvec = cv2.solvePnP(self.object_points, points, self.matrix, self.distortion,
                                              flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    tvec = tvec.ravel()
                    normal = cv2.Rodrigues(rvec)[0][:, 2]
                    distance = float(np.linalg.norm(tvec))
                    angle = math.degrees(math.acos(min(1.0, abs(float(normal @ tvec)) / distance)))
            found.append((int(marker_id), points, distance, angle))
        return found


def draw(image, found):
    for marker_id, points, distance, angle in found:
        cv2.polylines(image, [points.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
        label = f"cup {marker_id}"
        if distance is not None:
            label += f"  {distance * 100:.0f} cm  {angle:.0f} deg"
        x, y = points.min(axis=0).astype(int)
        for color, thick in (((0, 0, 0), 4), ((0, 255, 0), 2)):
            cv2.putText(image, label, (int(x), int(y) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thick, cv2.LINE_AA)


def put_lines(image, lines):
    for i, line in enumerate(lines):
        y = 30 + i * 30
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


def blown_out(gray) -> float:
    return float(np.count_nonzero(gray >= BLOWN_OUT)) / gray.size


class Tuner:
    def __init__(self, capture, markers, saved: dict | None):
        self.capture, self.markers = capture, markers
        # Locked {"exposure", "gain"}, or None on auto; locked kelvin, or None on auto.
        self.manual = None if saved is None else {"exposure": saved["exposure"], "gain": saved["gain"]}
        self.white_balance = None if saved is None else saved["white_balance"]
        self.sightings = collections.deque()  # (time, set of IDs seen in that frame)
        self.frame_times = collections.deque()
        self.sweep_result = []
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, IMAGE_SIZE[0], IMAGE_SIZE[1])
        start = {**self.camera_settings(), **(self.manual or {})}
        if self.white_balance is not None:
            start["white_balance"] = self.white_balance
        for name, (low, high) in (("exposure", (EXPOSURE_RANGE[0], EXPOSURE_SLIDER_MAX)), ("gain", GAIN_RANGE),
                                  ("white balance", WHITE_BALANCE_RANGE)):
            key = name.replace(" ", "_")
            cv2.createTrackbar(name, WINDOW, min(max(int(start[key]), low), high), high, lambda _: None)
            cv2.setTrackbarMin(name, WINDOW, low)
            cv2.setTrackbarPos(name, WINDOW, min(max(int(start[key]), low), high))  # setting the min can move it
        self.last_sliders = self.sliders()

    def camera_settings(self) -> dict:
        """What the camera reports it's using (on auto, auto's current choice)."""
        return {"exposure": int(self.capture.get(cv2.CAP_PROP_EXPOSURE)), "gain": int(self.capture.get(cv2.CAP_PROP_GAIN)),
                "white_balance": int(self.capture.get(cv2.CAP_PROP_WB_TEMPERATURE))}

    def sliders(self) -> dict:
        return {"exposure": cv2.getTrackbarPos("exposure", WINDOW), "gain": cv2.getTrackbarPos("gain", WINDOW),
                "white_balance": cv2.getTrackbarPos("white balance", WINDOW)}

    def apply(self, settings: dict | None):
        """Exposure and gain: locks {"exposure", "gain"}, or None for auto."""
        self.manual = settings
        set_exposure(self.capture, settings)
        if settings is not None:
            cv2.setTrackbarPos("exposure", WINDOW, min(settings["exposure"], EXPOSURE_SLIDER_MAX))
            cv2.setTrackbarPos("gain", WINDOW, settings["gain"])
        self.last_sliders = self.sliders()

    def apply_white_balance(self, kelvin: int | None):
        self.white_balance = kelvin
        set_white_balance(self.capture, kelvin)
        if kelvin is not None:
            cv2.setTrackbarPos("white balance", WINDOW, kelvin)
        self.last_sliders = self.sliders()

    def read_sliders(self):
        """Applies slider moves (which also switch that setting off auto)."""
        wanted = self.sliders()
        if wanted == self.last_sliders:
            return
        if (wanted["exposure"], wanted["gain"]) != (self.last_sliders["exposure"], self.last_sliders["gain"]):
            self.manual = {"exposure": wanted["exposure"], "gain": wanted["gain"]}
            set_exposure(self.capture, self.manual)
        if wanted["white_balance"] != self.last_sliders["white_balance"]:
            self.white_balance = wanted["white_balance"]
            set_white_balance(self.capture, self.white_balance)
        self.last_sliders = wanted

    def grab(self):
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("camera stopped returning frames")
        return frame

    def show(self, frame, found, gray):
        now = time.monotonic()
        self.frame_times.append(now)
        self.sightings.append((now, {marker_id for marker_id, *_ in found}))
        while self.frame_times[0] < now - 1.0:
            self.frame_times.popleft()
        while self.sightings[0][0] < now - RATE_WINDOW_S:
            self.sightings.popleft()

        shown = frame.copy()
        draw(shown, found)
        actual = self.camera_settings()
        mode = "LOCKED" if self.manual is not None else "AUTO"
        white_balance = f"white balance {actual['white_balance']} K ({'locked' if self.white_balance else 'auto'})"
        rates = "  ".join(
            f"{marker_id}:{100 * sum(marker_id in seen for _, seen in self.sightings) / len(self.sightings):3.0f}%"
            for marker_id in cup_markers.CUP_IDS)
        lines = [
            f"{mode}  exposure {actual['exposure']} ({actual['exposure'] / 10:.1f} ms)  gain {actual['gain']}  "
            f"{len(self.frame_times)} fps   {white_balance}",
            f"{len(found)} marker(s) now   blown out {100 * blown_out(gray):.1f}% of image",
            f"read rate, last {RATE_WINDOW_S:.0f} s:  {rates}",
        ]
        if self.sweep_result:
            lines.append("last sweep (exposure: markers/frame): " + "  ".join(self.sweep_result))
        lines.append("a auto exposure   b auto white balance   [ ] exposure   x sweep   w save   s snapshot   q quit")
        put_lines(shown, lines)
        cv2.imshow(WINDOW, shown)

    def sweep(self):
        """Tries each SWEEP exposure at the current gain; prints the markers read per frame."""
        before = self.manual
        gain = (self.manual or self.camera_settings())["gain"]
        print(f"\nSweep at gain {gain} (hold the camera still):")
        print("  exposure    ms   markers/frame   blown out   cups read")
        results = []
        for exposure in SWEEP:
            set_exposure(self.capture, {"exposure": exposure, "gain": gain})
            for _ in range(SWEEP_SETTLE_FRAMES):
                self.grab()
            counts, blown, ids = [], [], collections.Counter()
            for _ in range(SWEEP_FRAMES):
                frame = self.grab()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found = self.markers.detect(gray)
                counts.append(len(found))
                blown.append(blown_out(gray))
                ids.update(marker_id for marker_id, *_ in found)
                shown = frame.copy()
                draw(shown, found)
                put_lines(shown, [f"SWEEP  exposure {exposure}  gain {gain}  (hold still)"])
                cv2.imshow(WINDOW, shown)
                cv2.waitKey(1)
            mean = sum(counts) / len(counts)
            read = " ".join(f"{i}" for i in sorted(ids)) or "-"
            print(f"  {exposure:8d} {exposure / 10:5.1f}   {mean:13.1f}   {100 * np.mean(blown):8.1f}%   {read}")
            results.append((exposure, mean))
        best = max(mean for _, mean in results)
        good = [exposure for exposure, mean in results if mean >= best - 0.05] if best > 0 else []
        if good:
            middle = good[len(good) // 2]
            print(f"  Best ({best:.1f} markers/frame): exposure {good[0]}-{good[-1]}; the middle, {middle}, "
                  f"leaves room for lighting changes.")
        else:
            print("  No markers read at any exposure: point the camera at the cups.")
        self.sweep_result = [f"{exposure}:{mean:.1f}" for exposure, mean in results]
        self.apply(before)

    def run(self):
        while True:
            self.read_sliders()
            frame = self.grab()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.show(frame, self.markers.detect(gray), gray)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return
            if key == ord("a"):
                if self.manual is None:
                    self.apply(self.camera_settings())  # start manual from auto's choice
                    print(f"Manual: exposure {self.manual['exposure']}, gain {self.manual['gain']}")
                else:
                    self.apply(None)
                    print("Auto-exposure")
            if key == ord("b"):
                if self.white_balance is None:
                    self.apply_white_balance(self.camera_settings()["white_balance"])
                    print(f"White balance locked at {self.white_balance} K")
                else:
                    self.apply_white_balance(None)
                    print("Auto white balance")
            if key in (ord("["), ord("]")):
                current = self.manual or self.camera_settings()
                factor = 1.1 if key == ord("]") else 1 / 1.1
                exposure = min(max(round(current["exposure"] * factor), EXPOSURE_RANGE[0]), EXPOSURE_RANGE[1])
                if exposure == current["exposure"]:
                    exposure += 1 if key == ord("]") else -1
                self.apply({"exposure": exposure, "gain": current["gain"]})
            if key == ord("x"):
                self.sweep()
            if key == ord("w"):
                if self.manual is None:
                    print("Exposure is on auto: nothing to save (press a, or move a slider, first).")
                else:
                    actual = self.camera_settings()
                    white_balance = None if self.white_balance is None else actual["white_balance"]
                    save_exposure(actual["exposure"], actual["gain"], white_balance)
                    print(f"Saved exposure {actual['exposure']} ({actual['exposure'] / 10:.1f} ms), gain "
                          f"{actual['gain']}, white balance {white_balance or 'auto'} to {EXPOSURE_PATH}")
            if key == ord("s"):
                SNAPSHOT_DIR.mkdir(exist_ok=True)
                path = SNAPSHOT_DIR / time.strftime("exposure_%Y%m%d_%H%M%S.png")
                cv2.imwrite(str(path), frame)
                print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0, help="camera index, /dev/video<N> (default: 0)")
    parser.add_argument("--auto", action="store_true", help="start on auto-exposure, ignoring the saved setting")
    args = parser.parse_args()

    saved = None if args.auto else load_exposure()
    print(f"Starting {'locked at ' + str(saved) if saved else 'on auto-exposure'}.")
    try:
        capture = open_camera(args.device, IMAGE_SIZE, exposure=saved)
    except RuntimeError as e:
        sys.exit(f"Error: {e}")
    try:
        Tuner(capture, Markers(), saved).run()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
