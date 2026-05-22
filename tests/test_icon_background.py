from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE_DIR = REPO_ROOT / "scripts" / "page"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (SCRIPTS_DIR, PAGE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inventory_to_layout import (  # noqa: E402
    _line_art_alpha,
)
from icon.detect import (  # noqa: E402
    _fill_has_reasonable_parent_support,
    detect_line_art_subicons,
    detect_white_subicons,
)


class IconBackgroundTests(unittest.TestCase):
    def test_child_fill_must_match_immediate_parent_background(self) -> None:
        crop = np.full((80, 180, 3), 255, dtype=np.uint8)
        blue = np.array([152, 71, 5], dtype=np.uint8)
        crop[10:70, 10:170] = blue
        mask = np.zeros((80, 180), dtype=bool)
        mask[30:50, 55:80] = True

        self.assertFalse(_fill_has_reasonable_parent_support(
            crop, mask, np.array([255, 255, 255], dtype=np.uint8), 1.0))
        self.assertTrue(_fill_has_reasonable_parent_support(
            crop, mask, blue, 1.0))

    def test_white_subicon_extracts_compact_dark_badge_as_whole_icon(self) -> None:
        img = np.zeros((120, 150, 3), dtype=np.uint8)
        panel_bg = np.array([250, 250, 250], dtype=np.uint8)
        blue = np.array([152, 71, 5], dtype=np.uint8)
        img[:] = panel_bg
        cv2.circle(img, (60, 60), 31, tuple(int(v) for v in blue),
                   -1, cv2.LINE_AA)
        cv2.circle(img, (55, 55), 9, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(img, (62, 62), (76, 76), (255, 255, 255), 4, cv2.LINE_AA)

        boxes, fill_jobs = detect_white_subicons(
            img, 0, 0, img.shape[1], img.shape[0],
            ocr_text_items=[],
            min_dim=12,
            max_dim=120,
            min_area=35,
            scale=1.0,
        )

        self.assertTrue(boxes)
        x1, y1, x2, y2 = boxes[0]
        self.assertLessEqual(x1, 31)
        self.assertLessEqual(y1, 31)
        self.assertGreaterEqual(x2, 89)
        self.assertGreaterEqual(y2, 89)
        mask, bg = fill_jobs[0]
        self.assertGreater(int(mask.sum()), 2500)
        self.assertLessEqual(
            int(np.max(np.abs(bg.astype(np.int16)
                              - panel_bg.astype(np.int16)))),
            5,
        )

    def test_line_subicon_alpha_keeps_badge_without_rect_bg(self) -> None:
        crop = np.zeros((70, 70, 3), dtype=np.uint8)
        crop[:] = np.array([130, 72, 8], dtype=np.uint8)
        blue = (152, 71, 5)
        cv2.circle(crop, (35, 35), 25, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(crop, (35, 35), 20, blue, 4, cv2.LINE_AA)
        cv2.line(crop, (22, 36), (32, 46), blue, 4, cv2.LINE_AA)
        cv2.line(crop, (32, 46), (50, 24), blue, 4, cv2.LINE_AA)

        alpha = _line_art_alpha(crop)

        self.assertGreater(int(alpha[35, 35]), 220)
        self.assertGreater(int(alpha[18, 35]), 220)
        self.assertLess(int(alpha[2, 2]), 20)

    def test_line_detector_recovers_clipped_badge_as_foreground_island(self) -> None:
        img = np.zeros((120, 140, 3), dtype=np.uint8)
        img[:] = np.array([120, 58, 5], dtype=np.uint8)
        cv2.circle(img, (69, 61), 27, (255, 255, 255), -1, cv2.LINE_AA)
        blue = (152, 71, 5)
        cv2.circle(img, (69, 61), 18, blue, 4, cv2.LINE_AA)
        cv2.line(img, (56, 62), (66, 72), blue, 4, cv2.LINE_AA)
        cv2.line(img, (66, 72), (83, 50), blue, 4, cv2.LINE_AA)

        boxes, fill_jobs = detect_line_art_subicons(
            img, 0, 0, img.shape[1], img.shape[0],
            ocr_text_items=[],
            min_dim=12,
            max_dim=120,
            min_area=35,
            scale=1.0,
        )

        self.assertTrue(boxes)
        x1, y1, x2, y2 = max(
            boxes,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
        )
        self.assertLessEqual(x1, 42)
        self.assertLessEqual(y1, 34)
        self.assertGreaterEqual(x2, 96)
        self.assertGreaterEqual(y2, 88)
        self.assertGreaterEqual(x1, 36)
        self.assertGreaterEqual(y1, 28)
        self.assertLessEqual(x2, 102)
        self.assertLessEqual(y2, 94)
        self.assertEqual(len(fill_jobs), len(boxes))
        mask, bg = fill_jobs[boxes.index((x1, y1, x2, y2))]
        self.assertGreater(int(mask.sum()), 600)
        self.assertLess(int(np.max(np.abs(bg.astype(int) - np.array([120, 58, 5])))), 20)

    def test_line_subicon_alpha_preserves_warning_hole(self) -> None:
        crop = np.zeros((60, 70, 3), dtype=np.uint8)
        crop[:] = np.array([246, 240, 255], dtype=np.uint8)
        red = (44, 44, 194)
        pts = np.array([[35, 8], [12, 50], [58, 50]], dtype=np.int32)
        cv2.fillPoly(crop, [pts], red, cv2.LINE_AA)
        cv2.line(crop, (35, 20), (35, 34), (255, 255, 255), 4, cv2.LINE_AA)
        cv2.circle(crop, (35, 42), 2, (255, 255, 255), -1, cv2.LINE_AA)

        alpha = _line_art_alpha(crop)

        self.assertGreater(int(alpha[27, 35]), 220)
        self.assertGreater(int(alpha[42, 35]), 220)
        self.assertLess(int(alpha[2, 2]), 20)


if __name__ == "__main__":
    unittest.main()
