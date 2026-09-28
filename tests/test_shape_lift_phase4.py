"""Native shape lift phase 4: pale borderless cards + outline-role fills.

Closes the two gates that kept the classic "white page + pale tinted
card" motif on the flattened-PNG path:

* ``_vet_fill_colors`` no longer rejects ghost-pale fills without a
  crisp border — a uniform borderless tint card lifts with a
  self-coloured (invisible) border. Gradients and baked-in glyphs
  still return None.
* ``LayoutBuilder`` lets outline-role elements fall through to
  ``classify_filled_shape`` when the ring lift did not happen (ring
  classification skips on a top badge, or fails on an odd mask) —
  a filled card is not a hollow ring, so it used to drop straight
  onto the keep-full-crop PNG path.
"""
from __future__ import annotations

import json
import sys
import tempfile
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

from layout.outline import classify_filled_shape  # noqa: E402
from layout.builder import LayoutBuilder  # noqa: E402

PALE_BGR = (250, 247, 245)     # → #F5F7FA, the classic tint card
TINT_BGR = (235, 227, 216)     # → #D8E3EB
GRAY_BGR = (60, 60, 60)        # → #3C3C3C
BLUE_BGR = (200, 100, 40)      # → #2864C8


def _img(w: int, h: int) -> np.ndarray:
    return np.full((h, w, 3), 255, np.uint8)


def _round_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                r: int, colour) -> None:
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), colour, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), colour, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r),
                   (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, colour, -1)


class PaleFillTests(unittest.TestCase):
    def test_pale_borderless_card_lifts(self) -> None:
        img = _img(300, 140)
        _round_rect(img, 12, 12, 288, 128, 30, PALE_BGR)
        hit = classify_filled_shape(img)
        self.assertIsNotNone(hit)
        kind, fill, line, radius, _line_px = hit
        self.assertIn(kind, ("rect", "round_rect"))
        self.assertEqual(fill, "#F5F7FA")
        # Borderless card: self-coloured border reads as no border.
        self.assertEqual(line, "#F5F7FA")

    def test_tint_bordered_card_still_lifts(self) -> None:
        # Near-white fills segmented via the dark border are a
        # pre-existing classifier case (the ring path owns those in the
        # real pipeline); a mid-tint fill + border must keep lifting.
        img = _img(300, 140)
        _round_rect(img, 12, 12, 288, 128, 30, GRAY_BGR)
        _round_rect(img, 15, 15, 285, 125, 27, TINT_BGR)
        hit = classify_filled_shape(img)
        self.assertIsNotNone(hit)
        _kind, fill, line, _radius, _line_px = hit
        self.assertEqual(fill, "#D8E3EB")
        self.assertEqual(line, "#3C3C3C")

    def test_gradient_card_stays_png(self) -> None:
        img = _img(300, 140)
        top = np.array([255, 255, 255], np.int16)
        bottom = np.array([60, 120, 180], np.int16)
        for y in range(12, 128):
            t = (y - 12) / 116.0
            row = (top + (bottom - top) * t).astype(np.uint8)
            img[y, 12:288] = row
        self.assertIsNone(classify_filled_shape(img))

    def test_glyph_baked_inside_stays_png(self) -> None:
        img = _img(300, 140)
        cv2.rectangle(img, (12, 12), (288, 128), BLUE_BGR, -1)
        cv2.rectangle(img, (120, 56), (160, 84), (255, 255, 255), -1)
        self.assertIsNone(classify_filled_shape(img))

    def test_dark_solid_rect_still_lifts(self) -> None:
        img = _img(300, 140)
        cv2.rectangle(img, (12, 12), (288, 128), GRAY_BGR, -1)
        kind, fill, _line, radius, _line_px = classify_filled_shape(img)
        self.assertEqual(kind, "rect")
        self.assertEqual(fill, "#3C3C3C")
        self.assertEqual(radius, 0.0)


class OutlineRoleFilledCardTests(unittest.TestCase):
    def test_outline_role_filled_card_lifts(self) -> None:
        # Filled tint card with a small tab straddling its top edge:
        # the top badge skips classify_outline_ring, so the card must
        # reach classify_filled_shape instead of the PNG full-crop path.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "page_01.png"
            img = _img(960, 540)
            cv2.rectangle(img, (100, 120), (700, 360), TINT_BGR, -1)
            cv2.rectangle(img, (150, 80), (260, 118), GRAY_BGR, -1)
            cv2.imwrite(str(src), img)
            inventory = [
                {"id": "v000", "type": "image",
                 "bbox": [100, 120, 700, 360],
                 "source": "cleaned", "role": "outline"},
                {"id": "v001", "type": "image",
                 "bbox": [150, 80, 260, 118],
                 "source": "cleaned", "role": "subicon"},
            ]
            inv_path = td / "inventory.json"
            inv_path.write_text(json.dumps(inventory), encoding="utf-8")
            args = type("Args", (), {
                "inventory": str(inv_path),
                "source": str(src),
                "cleaned": str(src),
                "out_assets_dir": str(td / "assets"),
                "asset_prefix": "assets/page_01",
                "out_manifest": str(td / "m.json"),
                "out_layout": str(td / "l.json"),
                "slide_width_in": None,
                "slide_height_in": 7.5,
            })()
            builder = LayoutBuilder(args)
            builder.build()
            builder.write()
            card_shapes = [s for s in builder.front_shape_elements
                           if s.get("box") == [100, 120, 600, 240]]
            self.assertEqual(len(card_shapes), 1)
            self.assertEqual(card_shapes[0]["shape"], "rect")
            self.assertEqual(card_shapes[0]["fill"], "#D8E3EB")
            card_images = [i for i in builder.image_elements
                           if i["name"] == "v000"]
            self.assertEqual(card_images, [])


if __name__ == "__main__":
    unittest.main()
