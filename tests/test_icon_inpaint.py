from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from icon.inpaint import inpaint_region_inplace  # noqa: E402


class IconInpaintTests(unittest.TestCase):
    def test_flat_background_uses_fill_colour_without_fringe(self) -> None:
        bg = np.array([246, 251, 255], dtype=np.uint8)
        img = np.zeros((90, 120, 3), dtype=np.uint8)
        img[:] = bg

        mask_u8 = np.zeros((90, 120), dtype=np.uint8)
        cv2.circle(img, (60, 45), 16, (180, 80, 10), -1)
        cv2.circle(mask_u8, (60, 45), 13, 255, -1)
        mask = mask_u8 > 0

        shadow = np.zeros(mask.shape, dtype=np.uint8)
        cv2.circle(shadow, (60, 45), 19, 255, -1)
        shadow = cv2.GaussianBlur(shadow, (9, 9), 0)
        fringe = (shadow > 10) & ~mask
        img[fringe] = (
            (img[fringe].astype(np.uint16) * 3
             + np.array([200, 130, 70], dtype=np.uint16))
            // 4
        ).astype(np.uint8)

        inpaint_region_inplace(img, mask, scale=1.0, fill_color=bg)

        footprint = cv2.dilate(
            mask.astype(np.uint8) * 255,
            np.ones((3, 3), np.uint8),
            iterations=5,
        ) > 0
        delta = np.abs(img[footprint].astype(np.int16) - bg.astype(np.int16))
        self.assertLessEqual(int(delta.max()), 2)


if __name__ == "__main__":
    unittest.main()
