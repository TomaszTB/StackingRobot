"""The ArUco markers on the cups, shared by the generator and detection.

DICT_4X4_50: at 2 cm its cells are 3.3 mm (bigger than AprilTag 36h11's 2.5
mm), so it reads from farther away; the ChArUco board uses DICT_5X5_100, so
neither is mistaken for the other. One ID per cup.
"""

import cv2

DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_LENGTH_M = 0.020  # black outer edge to edge
CUP_IDS = tuple(range(6))


def dictionary() -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(DICTIONARY)


def detector() -> cv2.aruco.ArucoDetector:
    """A detector for the cups' markers, with subpixel corners (small markers need them for a good pose)."""
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary(), parameters)
