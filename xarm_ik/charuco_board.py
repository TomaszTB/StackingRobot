"""The ChArUco calibration board, shared by the generator and calibration.

Sized for US letter: 7 x 9 squares of 25 mm (17.5 x 22.5 cm). Its markers
come from DICT_5X5_100, so they can't be mistaken for the cups' DICT_4X4_50
markers (and vice versa).

Print at 100% ("Actual size", no "fit to page"), then measure a square with
calipers: calibration should use the measured size, as the board's scale sets
every pose's.
"""

import cv2
import numpy as np

SQUARES_X = 7
SQUARES_Y = 9
SQUARE_LENGTH_M = 0.025
MARKER_LENGTH_M = 0.018
DICTIONARY = cv2.aruco.DICT_5X5_100


# Each board's markers: board 0 (the stack side's) takes ids 0-30, board 1 (the pyramid
# side's) the next 31, so a detection says which board it is, and a frame showing
# both can't mix their corners up.
MARKERS_PER_BOARD = (SQUARES_X * SQUARES_Y) // 2
BOARD_NAMES = ("stack side", "pyramid side")


def make_board(square_length_m: float = SQUARE_LENGTH_M, board_index: int = 0) -> cv2.aruco.CharucoBoard:
    """Board `board_index` (0: the stack side's, 1: the pyramid side's); pass the measured
    square size when calibrating (the marker scales with it).
    """
    marker_length_m = MARKER_LENGTH_M * square_length_m / SQUARE_LENGTH_M
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    first = board_index * MARKERS_PER_BOARD
    ids = np.arange(first, first + MARKERS_PER_BOARD, dtype=np.int32)
    return cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), square_length_m, marker_length_m, dictionary, ids)


def center_to_corner(board: cv2.aruco.CharucoBoard) -> np.ndarray:
    """4x4: board-corner coordinates -> board-center coordinates."""
    width, height = board.getChessboardSize()
    shift = np.eye(4)
    shift[:2, 3] = [-width * board.getSquareLength() / 2, -height * board.getSquareLength() / 2]
    return shift


class BoardLocator:
    """The camera's pose in the base frame from a board in view, given the board's pose
    (about its center) in the base frame. Independent of the arm's kinematics."""

    def __init__(self, board_in_base: np.ndarray, board_index: int = 0, min_corners: int = 6, min_markers: int = 3):
        self.board = make_board(board_index=board_index)
        self.min_corners, self.min_markers = min_corners, min_markers
        self.charuco_detector = cv2.aruco.CharucoDetector(self.board)
        self.marker_detector = cv2.aruco.ArucoDetector(self.board.getDictionary(), cv2.aruco.DetectorParameters())
        self.marker_ids = [int(i) for i in np.asarray(self.board.getIds()).ravel()]
        self.marker_points = [np.asarray(points, np.float32).reshape(4, 3) for points in self.board.getObjPoints()]
        self.corner_frame_in_base = np.asarray(board_in_base, float) @ center_to_corner(self.board)

    def camera_in_base(self, gray, matrix, distortion):
        """4x4, or None if too little of the board is visible. Uses ChArUco corners (at least
        min_corners, not collinear), else the board's own markers (at least min_markers)."""
        corners, ids, _, _ = self.charuco_detector.detectBoard(gray)
        if (corners is not None and len(corners) >= self.min_corners
                and not self.board.checkCharucoCornersCollinear(ids.reshape(-1, 1))):
            object_points, image_points = self.board.matchImagePoints(corners.reshape(-1, 1, 2), ids.reshape(-1, 1))
        else:
            marker_corners, marker_ids, _ = self.marker_detector.detectMarkers(gray)
            known = [] if marker_ids is None else [
                (self.marker_points[self.marker_ids.index(int(i))], c.reshape(4, 2))
                for c, i in zip(marker_corners, marker_ids.ravel()) if int(i) in self.marker_ids]
            if len(known) < self.min_markers:
                return None
            object_points = np.concatenate([points for points, _ in known])
            image_points = np.concatenate([c for _, c in known]).astype(np.float32)
        try:
            ok, rvec, tvec = cv2.solvePnP(object_points, image_points, matrix, distortion, flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            return None
        if not ok:
            return None
        board_in_camera = np.eye(4)
        board_in_camera[:3, :3] = cv2.Rodrigues(rvec)[0]
        board_in_camera[:3, 3] = tvec.ravel()
        return self.corner_frame_in_base @ np.linalg.inv(board_in_camera)
