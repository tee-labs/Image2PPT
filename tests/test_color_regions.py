"""Color-region detection: flat-fill cards carved out of fused blobs.

Real renders tint card fills 10-20 channel steps off the canvas, fusing
the whole content area into one foreground blob where contour/CC card
detectors die. detect_color_regions reconstructs the cards from
dominant-fill-color connected components:
  * same-color spatially-separate cards → separate regions
  * a pale card nested on a similarly-tinted band → BOTH regions
    (mask-pixel dedupe, not bbox dedupe)
  * a gradient band split across colour buckets → strips merged back
  * text-ink colour buckets and the canvas itself never become regions
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAGE_DIR = ROOT / "scripts" / "page"
SCRIPTS_DIR = ROOT / "scripts"
for _p in (str(PAGE_DIR), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from icon.regions import detect_color_regions  # noqa: E402


def _rr(img, x1, y1, x2, y2, r, colour):
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), colour, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), colour, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r),
                   (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, colour, -1)


TINT_BGR = (235, 241, 252)     # pale blue band fill   (bucket 14,15,15)
OFF_BGR = (254, 248, 243)      # warm off-white card   (bucket 15,15,15)
GREEN_BGR = np.array((232, 244, 236))  # pale green band base
BLUE_BGR = (200, 100, 40)


def _boxes_close(box, expected, tol=2) -> bool:
    return all(abs(a - b) <= tol for a, b in zip(box, expected))


class ColorRegionTests(unittest.TestCase):
    def test_offwhite_cards_found_on_white_canvas(self) -> None:
        # The real-slide killer: near-white fills 10-20 steps off the
        # canvas fuse the content blob; color regions must still carve
        # each card out.
        img = np.full((720, 1280, 3), 254, np.uint8)
        _rr(img, 60, 60, 660, 260, 14, OFF_BGR)
        _rr(img, 700, 60, 1220, 260, 14, OFF_BGR)
        regions = detect_color_regions(img, scale=1.0)
        boxes = [(r[0], r[1], r[2], r[3]) for r in regions]
        matches = sum(1 for box in boxes
                      if _boxes_close(box, (60, 60, 660, 260))
                      or _boxes_close(box, (700, 60, 1220, 260)))
        self.assertEqual(len(boxes), 2)
        self.assertEqual(matches, 2)

    def test_card_nested_on_similar_tint_band_both_found(self) -> None:
        # Band tint and card fill differ by ~19 channel steps — bbox
        # nesting must NOT dedupe them (mask-pixel dedupe keeps both).
        img = np.full((720, 1280, 3), 254, np.uint8)
        _rr(img, 100, 100, 1100, 300, 12, TINT_BGR)
        _rr(img, 200, 130, 620, 270, 10, OFF_BGR)
        regions = detect_color_regions(img, scale=1.0)
        fills = [tuple(int(v) for v in r[4]) for r in regions]
        self.assertEqual(len(regions), 2, f"fills={fills}")
        self.assertIn(tuple(TINT_BGR), fills)
        self.assertIn(tuple(OFF_BGR), fills)

    def test_gradient_band_strips_merge(self) -> None:
        # A vertical gradient splits across colour buckets; strips must
        # re-merge into one full-height region.
        img = np.full((720, 1280, 3), 254, np.uint8)
        top = GREEN_BGR + 8
        bot = GREEN_BGR - 8
        for y in range(100, 300):
            t = (y - 100) / 200.0
            colour = tuple(int(v) for v in (top * (1 - t) + bot * t))
            cv2.line(img, (150, y), (1050, y), colour, 1)
        regions = detect_color_regions(img, scale=1.0)
        self.assertEqual(len(regions), 1)
        x1, y1, x2, y2, _c, _m = regions[0]
        self.assertLessEqual(y1, 110)
        self.assertGreaterEqual(y2, 290)

    def test_thin_ink_strip_never_region(self) -> None:
        # Sub-min_dim ink (glyph lines, rules) must not become regions;
        # solid dark banners legitimately do.
        img = np.full((720, 1280, 3), 254, np.uint8)
        cv2.rectangle(img, (100, 100), (500, 112), (60, 60, 60), -1)
        regions = detect_color_regions(img, scale=1.0)
        self.assertEqual(regions, [])


if __name__ == "__main__":
    unittest.main()
