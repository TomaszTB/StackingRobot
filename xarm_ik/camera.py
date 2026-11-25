"""The webcam: opening it with locked focus, exposure and white balance, and its
intrinsics (camera_intrinsics.json, fitted at IMAGE_SIZE from the hand-eye
calibration's board images: scripts/hand_eye_calibration.py --refit-intrinsics).
"""

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "camera_intrinsics.json"

# Locked focus (autofocus off): intrinsics hold only at a fixed focus, and the
# driver resets it when the webcam is unplugged, so every open sets it. The
# C615 focuses in steps of 17 (it rounds to the nearest), so 51, not 50.
CAMERA_FOCUS = 51

# Locked exposure (auto-exposure off) and white balance, as saved by
# scripts/exposure_tuner.py: auto-exposure blows out the white cups and their
# markers under bright light. Like focus, the driver forgets them when the
# webcam is unplugged, so every open sets them. Without the file, both are left
# on auto; a null white_balance in it leaves just white balance on auto.
EXPOSURE_PATH = Path(__file__).resolve().parent.parent / "camera_exposure.json"
EXPOSURE_RANGE = (3, 2047)  # V4L2 exposure_time_absolute, in 100 us units (C615)
GAIN_RANGE = (0, 255)
WHITE_BALANCE_RANGE = (2800, 6500)  # kelvin (C615)

# The resolution everything streams at (and the intrinsics are fitted at). 720p
# is the full sensor view scaled down from 1080p (same field of view, no crop);
# it halves the USB traffic and the detection time. Cup markers are ~75 px
# across at the stack looks' 25 cm.
IMAGE_SIZE = (1280, 720)


def load_intrinsics(path: str | Path = DEFAULT_PATH) -> tuple:
    """(camera matrix, distortion) for IMAGE_SIZE images. ValueError if the
    calibration is at another resolution."""
    calibration = json.loads(Path(path).read_text())
    if tuple(calibration["image_size"]) != IMAGE_SIZE:
        raise ValueError(f"{path} is for {calibration['image_size'][0]}x{calibration['image_size'][1]} images, "
                         f"not {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}; refit it (hand_eye_calibration.py --solve ... "
                         "--refit-intrinsics)")
    return np.array(calibration["camera_matrix"], dtype=float), np.array(calibration["distortion"], dtype=float)


def load_exposure(path: str | Path = EXPOSURE_PATH) -> dict | None:
    """The saved {"exposure", "gain", "white_balance" (kelvin, or None: auto)}, or
    None (no file: everything on auto)."""
    try:
        saved = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None
    white_balance = saved.get("white_balance")
    return {"exposure": int(saved["exposure"]), "gain": int(saved["gain"]),
            "white_balance": None if white_balance is None else int(white_balance)}


def save_exposure(exposure: int, gain: int, white_balance: int | None, path: str | Path = EXPOSURE_PATH):
    Path(path).write_text(json.dumps({"exposure": int(exposure), "gain": int(gain),
                                      "white_balance": None if white_balance is None else int(white_balance)},
                                     indent=2) + "\n")


def set_exposure(capture, settings: dict | None):
    """Locks exposure and gain ({"exposure", "gain"}), or None: back to auto-exposure."""
    if settings is None:
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # V4L2 aperture priority, i.e. auto
        return
    capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # V4L2 manual
    capture.set(cv2.CAP_PROP_EXPOSURE, settings["exposure"])
    capture.set(cv2.CAP_PROP_GAIN, settings["gain"])


def set_white_balance(capture, kelvin: int | None):
    """Locks white balance at `kelvin`, or None: back to auto."""
    if kelvin is None:
        capture.set(cv2.CAP_PROP_AUTO_WB, 1)
        return
    capture.set(cv2.CAP_PROP_AUTO_WB, 0)
    capture.set(cv2.CAP_PROP_WB_TEMPERATURE, kelvin)


def open_camera(device: int = 0, image_size: tuple = IMAGE_SIZE, fps: float = 30, focus: int | None = CAMERA_FOCUS,
                exposure: dict | None | str = "saved"):
    """The webcam (V4L2), in MJPEG (raw YUV can't do 1080p at 30 fps over
    USB 2). `focus`: autofocus off, locked here; None leaves focus as it is.
    `exposure`: "saved" (camera_exposure.json, else auto), {"exposure", "gain",
    "white_balance"} to lock, or None for auto. RuntimeError if it can't be opened or read.
    """
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise RuntimeError(f"can't open camera {device} (/dev/video{device})")
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, image_size[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size[1])
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # newest frame, not a stale queued one
    if focus is not None:
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        capture.set(cv2.CAP_PROP_FOCUS, focus)
    settings = load_exposure() if exposure == "saved" else exposure
    set_exposure(capture, settings)
    set_white_balance(capture, settings.get("white_balance") if settings else None)
    ok, frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError(f"camera {device} opened but returned no frame")
    actual = (frame.shape[1], frame.shape[0])
    if actual != tuple(image_size):
        print(f"Warning: asked for {image_size[0]}x{image_size[1]}, camera gives {actual[0]}x{actual[1]}")
    return capture


class FrameGrabber:
    """Reads the camera on a background thread, keeping only the newest frame."""

    def __init__(self, capture, timeout_s: float = 3.0):
        self.capture = capture
        self.timeout_s = timeout_s  # frames_after gives up after this
        self._lock = threading.Lock()
        self._frame, self._time_s, self._count = None, 0.0, 0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame, self._time_s, self._count = frame, time.monotonic(), self._count + 1

    def latest(self):
        """(count, monotonic time read, frame); frame is None before the first."""
        with self._lock:
            return self._count, self._time_s, self._frame

    def frames_after(self, start_s: float, n: int, wait=time.sleep) -> list:
        """The next `n` distinct frames read after monotonic `start_s` (fewer on timeout)."""
        frames, last = [], -1
        deadline = time.monotonic() + self.timeout_s
        while len(frames) < n and time.monotonic() < deadline:
            count, time_s, frame = self.latest()
            if frame is not None and count != last and time_s > start_s:
                frames.append(frame)
                last = count
            else:
                wait(0.005)
        return frames

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)
