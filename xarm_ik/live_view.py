"""A live camera window: the newest frame with cup markers outlined, a status line and
extra text lines. Redrawn from the arm's ticks, so OpenCV's window stays on the main thread."""

import time

import cv2

from . import cup_markers
from .camera import IMAGE_SIZE, FrameGrabber

SCALE = 0.75  # window and overlay, as a fraction of the image size
PERIOD_S = 0.1  # minimum time between redraws


class LiveView:
    WINDOW = "camera"

    def __init__(self, grabber: FrameGrabber):
        self.grabber = grabber
        self.detector = cup_markers.detector()
        self.status = "starting"
        self.lines = []  # shown under the status
        self._shown_s = 0.0
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, int(IMAGE_SIZE[0] * SCALE), int(IMAGE_SIZE[1] * SCALE))

    def tick(self, *_):
        """For the controller's on_tick."""
        self.update()

    def update(self, force=False):
        if not force and time.monotonic() - self._shown_s < PERIOD_S:
            return
        self._shown_s = time.monotonic()
        _, _, frame = self.grabber.latest()
        if frame is None:
            cv2.waitKey(1)
            return
        shown = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(shown, cv2.COLOR_BGR2GRAY))
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(shown, corners, ids)
        for i, line in enumerate([self.status, *self.lines]):
            y = 30 + 28 * i
            cv2.putText(shown, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(shown, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(self.WINDOW, shown)
        cv2.waitKey(1)

    def wait(self, seconds: float):
        """Sleep, keeping the window responsive."""
        end = time.monotonic() + seconds
        while (left := end - time.monotonic()) > 0:
            self.update()
            time.sleep(min(left, 0.02))

    def close(self):
        cv2.destroyWindow(self.WINDOW)
