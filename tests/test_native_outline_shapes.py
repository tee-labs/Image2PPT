"""Native outline shapes: card frames become PPT round_rect shapes.

Regression check for ENABLE_NATIVE_OUTLINE_SHAPES — with the flag on,
a wide card-like outline record must emit a native `shape` element
instead of an image crop.
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
if str(PAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PAGE_DIR))

from layout.builder import LayoutBuilder  # noqa: E402
from inventory_to_layout import ENABLE_NATIVE_OUTLINE_SHAPES  # noqa: E402


def _make_slide(path: Path, size=(960, 540)) -> None:
    img = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (100, 120), (700, 360), (205, 160, 90), 3)
    cv2.imwrite(str(path), img)


class NativeOutlineShapeTests(unittest.TestCase):
    def test_flag_is_on(self) -> None:
        self.assertTrue(ENABLE_NATIVE_OUTLINE_SHAPES)

    def test_wide_card_outline_becomes_native_shape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "page_01.png"
            _make_slide(src)
            inventory = [
                {"id": "v000", "type": "image",
                 "bbox": [100, 120, 700, 360],
                 "source": "cleaned", "role": "outline"},
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
            self.assertEqual(len(builder.shape_elements), 1)
            shape = builder.shape_elements[0]
            # Sharp-cornered frame lifts as a plain rect (phase 3: no
            # artificial 0.08 corner rounding on sharp source frames).
            self.assertEqual(shape["shape"], "rect")
            self.assertEqual(shape["box"], [100, 120, 600, 240])
            self.assertTrue(shape["line"])


if __name__ == "__main__":
    unittest.main()
