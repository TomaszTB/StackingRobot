"""Write the cups' ArUco markers (xarm_ik/cup_markers.py) as a US-letter PDF,
at exact physical size: one per cup, plus a spare row.

Each marker sits in a white margin (the detector needs at least one cell of
it, ~3.3 mm) inside a light cut line; cut on the line, not at the black edge.
Print at 100% / "Actual size", check the 100 mm bar, and measure a marker:
it should be 20.0 mm across its black edge.

Usage:
    uv run python scripts/make_cup_markers.py              # cup_markers.pdf
    uv run python scripts/make_cup_markers.py out.pdf
"""

import argparse

import cv2
import numpy as np
from PIL import Image

from xarm_ik import cup_markers

LETTER_IN = (8.5, 11.0)
CELLS = 6  # 4x4 data + a 1-cell black border each side
CELL_PX = 100  # whole pixels per cell, so every cell comes out the same size
MARGIN_MM = 5.0  # white around each marker, inside the cut line
LABEL_MM = 6.0  # room under the margin for the ID label
GAP_MM = 8.0  # between tiles
SETS = 2  # one per cup, plus spares
PER_ROW = 3  # 3 tiles (+ gaps) fit letter width with printer margins to spare


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", nargs="?", default="cup_markers.pdf", help="output PDF (default: cup_markers.pdf)")
    args = parser.parse_args()

    marker_mm = cup_markers.MARKER_LENGTH_M * 1000
    marker_px = CELLS * CELL_PX
    dpi = marker_px * 25.4 / marker_mm
    px_per_mm = dpi / 25.4
    page_w, page_h = (round(side * dpi) for side in LETTER_IN)
    page = np.full((page_h, page_w), 255, np.uint8)

    margin, label, gap = (round(mm * px_per_mm) for mm in (MARGIN_MM, LABEL_MM, GAP_MM))
    tile_w, tile_h = marker_px + 2 * margin, marker_px + 2 * margin + label
    left = (page_w - (PER_ROW * tile_w + (PER_ROW - 1) * gap)) // 2
    if left < round(10 * px_per_mm):
        raise SystemExit("markers don't fit the page width; lower PER_ROW")
    top = round(20 * px_per_mm)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.9 * px_per_mm / 10, max(1, round(0.25 * px_per_mm))
    dictionary = cup_markers.dictionary()

    tiles = [(spare, marker_id) for spare in range(SETS) for marker_id in cup_markers.CUP_IDS]
    rows = -(-len(tiles) // PER_ROW)
    for n, (spare, marker_id) in enumerate(tiles):
        x = left + (n % PER_ROW) * (tile_w + gap)
        y = top + (n // PER_ROW) * (tile_h + gap)
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px, borderBits=1)
        page[y + margin : y + margin + marker_px, x + margin : x + margin + marker_px] = marker
        cv2.rectangle(page, (x, y), (x + tile_w, y + tile_h), 180, max(1, round(0.15 * px_per_mm)))  # cut line
        text = f"cup {marker_id}" + (" (spare)" if spare else "")
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.putText(page, text, (x + (tile_w - text_w) // 2, y + 2 * margin + marker_px + (label + text_h) // 2 - margin // 2),
                    font, scale, 0, thick, cv2.LINE_AA)

    # A 100 mm scale bar and the specs.
    bar_y = top + rows * (tile_h + gap) + round(10 * px_per_mm)
    bar_len, bar_thick = round(100 * px_per_mm), max(1, round(0.4 * px_per_mm))
    cv2.line(page, (left, bar_y), (left + bar_len, bar_y), 0, bar_thick)
    for x in (left, left + bar_len):
        cv2.line(page, (x, bar_y - round(2 * px_per_mm)), (x, bar_y + round(2 * px_per_mm)), 0, bar_thick)
    notes = [
        "100 mm  (print at 100% / Actual size;",
        "check this bar with a ruler)",
        f"ArUco DICT_4X4_50, IDs {cup_markers.CUP_IDS[0]}-{cup_markers.CUP_IDS[-1]}, {marker_mm:g} mm across",
        "the black edge. Cut on the gray lines.",
    ]
    for i, note in enumerate(notes):
        cv2.putText(page, note, (left, bar_y + round((7 + 5 * i) * px_per_mm)), font, 1.2 * px_per_mm / 10, 0,
                    max(1, round(0.25 * px_per_mm)), cv2.LINE_AA)

    Image.fromarray(page).save(args.out, "PDF", resolution=dpi)
    print(f"Wrote {args.out}: {len(cup_markers.CUP_IDS)} markers x {SETS}, {marker_mm:g} mm, "
          f"{MARGIN_MM:g} mm white margin")


if __name__ == "__main__":
    main()
