"""Write the ChArUco calibration board (xarm_ik/charuco_board.py) as a
US-letter PDF, at exact physical size.

Print it at 100% / "Actual size" (not "Fit to page"), check the 100 mm bar
with a ruler, and tape it flat to something rigid.

Usage:
    uv run python scripts/make_charuco_board.py              # charuco_board.pdf
    uv run python scripts/make_charuco_board.py out.pdf
    uv run python scripts/make_charuco_board.py --board 1     # the pyramid side's (charuco_board_2.pdf)
"""

import argparse

import cv2
import numpy as np
from PIL import Image

from xarm_ik import charuco_board

LETTER_IN = (8.5, 11.0)
SQUARE_PX = 600  # whole pixels per square, so every square comes out the same size


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", nargs="?", default=None,
                        help="output PDF (default: charuco_board.pdf, or charuco_board_2.pdf for --board 1)")
    parser.add_argument("--board", type=int, choices=range(len(charuco_board.BOARD_NAMES)), default=0,
                        help="0: the stack side's board (marker ids 0-30); 1: the pyramid side's (ids 31-61)")
    args = parser.parse_args()
    args.out = args.out or ("charuco_board.pdf" if args.board == 0 else f"charuco_board_{args.board + 1}.pdf")

    square_mm = charuco_board.SQUARE_LENGTH_M * 1000
    dpi = SQUARE_PX * 25.4 / square_mm
    px_per_mm = dpi / 25.4
    page_w, page_h = (round(side * dpi) for side in LETTER_IN)
    board_w, board_h = charuco_board.SQUARES_X * SQUARE_PX, charuco_board.SQUARES_Y * SQUARE_PX

    board = charuco_board.make_board(board_index=args.board)
    image = board.generateImage((board_w, board_h), marginSize=0, borderBits=1)
    page = np.full((page_h, page_w), 255, np.uint8)
    left, top = (page_w - board_w) // 2, (page_h - board_h) // 2 - round(6 * px_per_mm)  # room for the notes below
    page[top : top + board_h, left : left + board_w] = image

    # A 100 mm scale bar with end ticks, then the board's specs.
    bar_y = top + board_h + round(8 * px_per_mm)
    bar_len, thick = round(100 * px_per_mm), max(1, round(0.4 * px_per_mm))
    cv2.line(page, (left, bar_y), (left + bar_len, bar_y), 0, thick)
    for x in (left, left + bar_len):
        cv2.line(page, (x, bar_y - round(2 * px_per_mm)), (x, bar_y + round(2 * px_per_mm)), 0, thick)
    font_scale, font_thick = 1.2 * px_per_mm / 10, max(1, round(0.25 * px_per_mm))
    text_y = bar_y + round(7 * px_per_mm)
    notes = [
        "100 mm  (print at 100% / Actual size; check this bar with a ruler)",
        f"ChArUco {charuco_board.SQUARES_X}x{charuco_board.SQUARES_Y}, square {square_mm:g} mm, "
        f"marker {charuco_board.MARKER_LENGTH_M * 1000:g} mm, DICT_5X5_100, "
        f"ids {args.board * charuco_board.MARKERS_PER_BOARD}-{(args.board + 1) * charuco_board.MARKERS_PER_BOARD - 1} "
        f"({charuco_board.BOARD_NAMES[args.board]})",
    ]
    for i, note in enumerate(notes):
        cv2.putText(page, note, (left, text_y + i * round(5 * px_per_mm)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 0, font_thick, cv2.LINE_AA)

    Image.fromarray(page).save(args.out, "PDF", resolution=dpi)
    print(f"Wrote {args.out}: {page_w / dpi:.2f} x {page_h / dpi:.2f} in page, "
          f"{board_w / px_per_mm:.1f} x {board_h / px_per_mm:.1f} mm board")


if __name__ == "__main__":
    main()
